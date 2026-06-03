#!/usr/bin/env python3
"""Control::PatternMutate operator: stage or apply CRUD operations to topology.source.json programs.

CKA programs define seq/par/choice/star execution structures used by the source
topology compiler. Mutating them reshapes generated topology and generated
patterns without writing the deleted legacy pattern asset.

Operations (pattern_ops array):
  {"op": "create",  "pattern": {"name": "...", "expr": {...}, "confidence": 0.95, "cost": 1.0, "safety": 0.9}}
  {"op": "update",  "name": "...", "patch": {"confidence": 0.8}}
  {"op": "replace", "name": "...", "pattern": {"name": "...", "expr": {...}, ...}}
  {"op": "delete",  "name": "..."}

Expr grammar:
  "one"                                    — identity (always matches)
  "zero"                                   — blocked (never matches)
  "NodeName"                               — single node
  {"seq":    ["A", "B", "C"]}              — sequential execution
  {"choice": [{...}, {...}]}               — exclusive choice
  {"par":    [{...}, {...}]}               — parallel (requires effect independence)
  {"star":   {"body": {...}, "max_unroll": N}}  — bounded Kleene star

Rate limiting: skips if last mutation was within MIN_INTERVAL_S seconds.

Output (success):
  {"pattern_mutate": {"applied": [...], "pattern_count": N}}
Output (staged):
  {"pattern_mutate": {"staged": true, "mutation_id": "...", "queue_path": "...", "summary": {...}}}
Output (skipped):
  {"pattern_mutate": {"skipped": "rate_limited", "next_allowed_in_s": N}}
Output (failure):
  {"pattern_mutate": {"error": "..."}}
"""

import datetime
import json
import pathlib
import sys

import mutation_policy

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SOURCE_TOPOLOGY_PATH = _PROJECT_ROOT / "assets" / "topology.source.json"
SOURCE_TOPOLOGY_BAK  = _PROJECT_ROOT / "assets" / "topology.source.json.bak"
MUTATIONS_LOG = _PROJECT_ROOT / "state" / "pattern_mutations.jsonl"
MIN_INTERVAL_S = 30
_EFFECTS = ["pattern_write"]


def _load() -> dict:
    return json.loads(SOURCE_TOPOLOGY_PATH.read_text())


def _write(data: dict) -> None:
    SOURCE_TOPOLOGY_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _unwrap(payload: dict) -> dict:
    for _ in range(4):
        if "pattern_ops" in payload:
            return payload
        ctx = payload.get("context")
        if not isinstance(ctx, str):
            break
        try:
            inner = json.loads(ctx.strip())
            if isinstance(inner, dict):
                payload = inner
                continue
        except json.JSONDecodeError:
            break
        break
    return payload


def _rate_limited() -> tuple[bool, float]:
    if not MUTATIONS_LOG.exists():
        return False, 0.0
    try:
        for line in reversed(MUTATIONS_LOG.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts_str = rec.get("ts", "")
            if not ts_str:
                continue
            ts = datetime.datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            elapsed = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds()
            remaining = MIN_INTERVAL_S - elapsed
            return remaining > 0, max(0.0, remaining)
    except Exception:
        pass
    return False, 0.0


def _apply_ops(data: dict, ops: list) -> tuple[list, str | None]:
    programs: list = data.get("programs", [])
    applied = []

    for op in ops:
        kind = op.get("op")

        if kind == "create":
            pat = op.get("pattern", {})
            name = pat.get("name", "")
            if not name:
                return applied, f"create: missing name in {op}"
            if any(p.get("name") == name for p in programs):
                return applied, f"create: pattern already exists: {name}"
            if "expr" not in pat:
                return applied, f"create: missing expr in {op}"
            programs.append(pat)
            applied.append({"op": "create", "name": name})

        elif kind == "update":
            name = op.get("name", "")
            patch = op.get("patch", {})
            idx = next((i for i, p in enumerate(programs) if p.get("name") == name), None)
            if idx is None:
                return applied, f"update: not found: {name}"
            programs[idx].update({k: v for k, v in patch.items() if k != "name"})
            applied.append({"op": "update", "name": name})

        elif kind == "replace":
            name = op.get("name", "")
            replacement = dict(op.get("pattern", {}))
            idx = next((i for i, p in enumerate(programs) if p.get("name") == name), None)
            if idx is None:
                return applied, f"replace: not found: {name}"
            replacement["name"] = name
            programs[idx] = replacement
            applied.append({"op": "replace", "name": name})

        elif kind == "delete":
            name = op.get("name", "")
            before = len(programs)
            programs = [p for p in programs if p.get("name") != name]
            if len(programs) == before:
                return applied, f"delete: not found: {name}"
            applied.append({"op": "delete", "name": name})

        else:
            return applied, f"unknown op: {kind!r}"

    data["programs"] = programs
    return applied, None


def _log(applied: list, error: str | None) -> None:
    MUTATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "result": {"applied": applied, "error": error},
    }
    try:
        with MUTATIONS_LOG.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("[pattern_mutate] empty stdin\n")
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[pattern_mutate] stdin parse error: {exc}\n")
        sys.exit(1)

    payload = _unwrap(payload)
    ops: list = payload.get("pattern_ops", [])

    if not ops:
        print(json.dumps({"pattern_mutate": {"applied": [], "note": "no ops provided"}}))
        return

    try:
        data = _load()
    except Exception as exc:
        sys.stderr.write(f"[pattern_mutate] cannot load {SOURCE_TOPOLOGY_PATH}: {exc}\n")
        sys.exit(1)

    applied, error = _apply_ops(data, ops)
    if error:
        _log(applied, error)
        print(json.dumps({"pattern_mutate": {"error": error}}))
        sys.exit(1)

    decision = mutation_policy.decision_for_effects(_EFFECTS)
    if decision == "deny":
        error = "side-effect policy denied pattern mutation"
        _log(applied, error)
        print(json.dumps({"pattern_mutate": {"error": error}}))
        sys.exit(1)
    if decision == "stage":
        staged = mutation_policy.stage_mutation(
            source_node="Control::PatternMutate",
            kind="pattern_patch",
            effects=_EFFECTS,
            payload={
                "pattern_ops": ops,
                "reason": payload.get("reason", ""),
            },
            summary={
                "applied_if_approved": applied,
                "program_count_after": len(data.get("programs", [])),
            },
            target_paths=[str(SOURCE_TOPOLOGY_PATH.relative_to(_PROJECT_ROOT))],
        )
        print(json.dumps({"pattern_mutate": staged}))
        return

    limited, wait = _rate_limited()
    if limited:
        print(json.dumps({"pattern_mutate": {"skipped": "rate_limited", "next_allowed_in_s": round(wait, 1)}}))
        return

    try:
        SOURCE_TOPOLOGY_BAK.write_text(json.dumps(data, indent=2) + "\n")
    except OSError:
        pass

    try:
        _write(data)
    except OSError as exc:
        _log(applied, str(exc))
        print(json.dumps({"pattern_mutate": {"error": f"write failed: {exc}"}}))
        sys.exit(1)

    _log(applied, None)
    print(json.dumps({
        "pattern_mutate": {
            "applied": applied,
            "program_count": len(data.get("programs", [])),
        }
    }))


if __name__ == "__main__":
    main()
