#!/usr/bin/env python3
"""Control::PatternMutate operator: apply CRUD operations to assets/patterns.json.

CKA patterns define seq/par/choice/star execution structures used by the batch
scheduler. Mutating them reshapes which execution paths are explored in parallel.

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
Output (skipped):
  {"pattern_mutate": {"skipped": "rate_limited", "next_allowed_in_s": N}}
Output (failure):
  {"pattern_mutate": {"error": "..."}}
"""

import datetime
import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PATTERNS_PATH = _PROJECT_ROOT / "assets" / "patterns.json"
PATTERNS_BAK  = _PROJECT_ROOT / "assets" / "patterns.json.bak"
MUTATIONS_LOG = _PROJECT_ROOT / "state" / "pattern_mutations.jsonl"
MIN_INTERVAL_S = 30


def _load() -> dict:
    return json.loads(PATTERNS_PATH.read_text())


def _write(data: dict) -> None:
    PATTERNS_PATH.write_text(json.dumps(data, indent=2) + "\n")


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
    patterns: list = data.get("patterns", [])
    applied = []

    for op in ops:
        kind = op.get("op")

        if kind == "create":
            pat = op.get("pattern", {})
            name = pat.get("name", "")
            if not name:
                return applied, f"create: missing name in {op}"
            if any(p.get("name") == name for p in patterns):
                return applied, f"create: pattern already exists: {name}"
            if "expr" not in pat:
                return applied, f"create: missing expr in {op}"
            patterns.append(pat)
            applied.append({"op": "create", "name": name})

        elif kind == "update":
            name = op.get("name", "")
            patch = op.get("patch", {})
            idx = next((i for i, p in enumerate(patterns) if p.get("name") == name), None)
            if idx is None:
                return applied, f"update: not found: {name}"
            patterns[idx].update({k: v for k, v in patch.items() if k != "name"})
            applied.append({"op": "update", "name": name})

        elif kind == "replace":
            name = op.get("name", "")
            replacement = dict(op.get("pattern", {}))
            idx = next((i for i, p in enumerate(patterns) if p.get("name") == name), None)
            if idx is None:
                return applied, f"replace: not found: {name}"
            replacement["name"] = name
            patterns[idx] = replacement
            applied.append({"op": "replace", "name": name})

        elif kind == "delete":
            name = op.get("name", "")
            before = len(patterns)
            patterns = [p for p in patterns if p.get("name") != name]
            if len(patterns) == before:
                return applied, f"delete: not found: {name}"
            applied.append({"op": "delete", "name": name})

        else:
            return applied, f"unknown op: {kind!r}"

    data["patterns"] = patterns
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

    limited, wait = _rate_limited()
    if limited:
        print(json.dumps({"pattern_mutate": {"skipped": "rate_limited", "next_allowed_in_s": round(wait, 1)}}))
        return

    try:
        data = _load()
    except Exception as exc:
        sys.stderr.write(f"[pattern_mutate] cannot load {PATTERNS_PATH}: {exc}\n")
        sys.exit(1)

    try:
        PATTERNS_BAK.write_text(json.dumps(data, indent=2) + "\n")
    except OSError:
        pass

    applied, error = _apply_ops(data, ops)
    if error:
        _log(applied, error)
        print(json.dumps({"pattern_mutate": {"error": error}}))
        sys.exit(1)

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
            "pattern_count": len(data.get("patterns", [])),
        }
    }))


if __name__ == "__main__":
    main()
