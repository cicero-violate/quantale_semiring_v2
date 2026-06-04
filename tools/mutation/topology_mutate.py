#!/usr/bin/env python3
"""Control::TopologyMutate operator: stage or apply CRUD operations to topology nodes and edges.

Reads a JSON payload from stdin with a list of operations, applies them to
assets/topology.source.json (and assets/operators.json for new operator nodes) only
when side-effect policy allows direct apply. By default, mutating effects are
queued in state/mutation_queue.jsonl for explicit review/apply.

Operations (topology_ops array):
  {"op": "create_node",  "node": {"name": "...", "type": "State|Control|Event|Execution|Analysis", "action": "...", "description": "..."}}
  {"op": "create_edge",  "from": "...", "to": "...", "confidence": 0.9, "cost": 0.1, "safety": 0.9}
  {"op": "update_node",  "name": "...", "patch": {"action": "..."}}
  {"op": "update_edge",  "from": "...", "to": "...", "patch": {"confidence": 0.8}}
  {"op": "replace_node", "name": "...", "node": {...}}
  {"op": "replace_edge", "from": "...", "to": "...", "edge": {...}}
  {"op": "delete_node",  "name": "..."}
  {"op": "delete_edge",  "from": "...", "to": "..."}

Operator contracts — add new (operator_contracts array):
  {"node_name": "...", "executable": "python3|true|jit_cuda|cargo", "static_args": [...],
   "description": "...", "effects": {"reads": [...], "writes": [...], "locks": []},
   "jit_body": "out[i] = ...;"  (jit_cuda only)}

Operator contract ops — update/replace existing (operator_contract_ops array):
  {"op": "update", "node_name": "...", "patch": {"executable": "python3", "static_args": [...]}}
  {"op": "replace", "contract": {"node_name": "...", "executable": "python3", ...}}

Rate limiting: skips if last mutation was within MIN_MUTATION_INTERVAL_S seconds.
Backup: writes assets/topology.source.json.bak and assets/operators.json.bak before each mutation.

Output (success):
  {"topology_mutate": {"applied": [...], "contracts_added": [...], "contracts_updated": [...], "node_count": N, "edge_count": N}}
Output (staged):
  {"topology_mutate": {"staged": true, "mutation_id": "...", "queue_path": "...", "summary": {...}}}
Output (skipped — rate limited):
  {"topology_mutate": {"skipped": "rate_limited", "next_allowed_in_s": N}}
Output (failure):
  {"topology_mutate": {"error": "..."}}
"""

import datetime
import json
import pathlib
import sys

import mutation_policy

_PROJECT_ROOT   = pathlib.Path(__file__).resolve().parents[2]
TOPOLOGY_PATH   = _PROJECT_ROOT / "assets" / "topology.source.json"
OPERATORS_PATH  = _PROJECT_ROOT / "assets" / "operators.json"
EXPLORATION_PATH = _PROJECT_ROOT / "assets" / "exploration.json"
MUTATIONS_LOG   = _PROJECT_ROOT / "state" / "topology_mutations.jsonl"
_EFFECTS = ["topology_write", "operator_registry_write"]

_DEFAULT_NODE_FEATURES = {"novelty": 0.3, "entropy": 0.3}
TOPOLOGY_BAK   = _PROJECT_ROOT / "assets" / "topology.source.json.bak"
OPERATORS_BAK  = _PROJECT_ROOT / "assets" / "operators.json.bak"
MIN_MUTATION_INTERVAL_S = 30


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _write(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _append_mutation_log(applied: list, contracts_added: list, contracts_updated: list, error: str | None) -> None:
    MUTATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "result": {
            "applied": applied,
            "contracts_added": contracts_added,
            "contracts_updated": contracts_updated,
            "error": error,
        },
    }
    try:
        with MUTATIONS_LOG.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def _apply_contract_ops(operators: dict, ops: list) -> tuple[list, str | None]:
    """Update or replace existing operator contracts."""
    updated = []
    for op in ops:
        kind = op.get("op")
        if kind == "update":
            name = op.get("node_name", "")
            patch = op.get("patch", {})
            idx = next((i for i, o in enumerate(operators["operators"]) if o.get("node_name") == name), None)
            if idx is None:
                return updated, f"update operator_contract: not found: {name}"
            operators["operators"][idx].update({k: v for k, v in patch.items() if k != "node_name"})
            updated.append({"op": "update", "node_name": name})
        elif kind == "replace":
            contract = op.get("contract", {})
            name = contract.get("node_name", "")
            if not name:
                return updated, "replace operator_contract: missing node_name"
            idx = next((i for i, o in enumerate(operators["operators"]) if o.get("node_name") == name), None)
            if idx is None:
                return updated, f"replace operator_contract: not found: {name}"
            operators["operators"][idx] = contract
            updated.append({"op": "replace", "node_name": name})
        else:
            return updated, f"unknown operator_contract_op: {kind!r}"
    return updated, None


def _rate_limited() -> tuple[bool, float]:
    """Return (is_limited, seconds_until_allowed)."""
    if not MUTATIONS_LOG.exists():
        return False, 0.0
    try:
        lines = MUTATIONS_LOG.read_text().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts_str = rec.get("ts", "")
            if not ts_str:
                continue
            import datetime as _dt
            ts = _dt.datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            now = _dt.datetime.now(_dt.timezone.utc)
            elapsed = (now - ts).total_seconds()
            remaining = MIN_MUTATION_INTERVAL_S - elapsed
            return remaining > 0, max(0.0, remaining)
    except Exception:
        pass
    return False, 0.0


def _update_exploration(new_node_names: list[str]) -> None:
    """Add default node_features entries for newly created nodes."""
    if not new_node_names or not EXPLORATION_PATH.exists():
        return
    try:
        exp = _load(EXPLORATION_PATH)
        features = exp.setdefault("node_features", {})
        changed = False
        for name in new_node_names:
            if name not in features:
                features[name] = _DEFAULT_NODE_FEATURES.copy()
                changed = True
        if changed:
            _write(EXPLORATION_PATH, exp)
    except Exception:
        pass


def _node_name(node: dict) -> str:
    return node.get("name", "")


def _edge_key(edge: dict) -> tuple:
    return (edge.get("from", ""), edge.get("to", ""))


def _apply_ops(topology: dict, ops: list) -> tuple[list, str | None]:
    nodes: list = topology.get("nodes", [])
    transitions: list = topology.get("transitions", [])
    applied = []

    for op in ops:
        kind = op.get("op")

        if kind == "create_node":
            node = dict(op.get("node", {}))
            name = node.get("name", "")
            if not name:
                return applied, f"create_node: missing name in {op}"
            if any(_node_name(n) == name for n in nodes):
                return applied, f"create_node: node already exists: {name}"
            node.pop("id", None)
            nodes.append(node)
            applied.append({"op": "create_node", "name": name})

        elif kind == "create_edge":
            src, dst = op.get("from", ""), op.get("to", "")
            if not src or not dst:
                return applied, f"create_edge: missing from/to in {op}"
            known = {_node_name(n) for n in nodes}
            if src not in known:
                return applied, f"create_edge: unknown source: {src}"
            if dst not in known:
                return applied, f"create_edge: unknown destination: {dst}"
            if any(_edge_key(e) == (src, dst) for e in transitions):
                return applied, f"create_edge: edge already exists: {src} -> {dst}"
            edge: dict = {
                "from": src,
                "to": dst,
                "confidence": op.get("confidence", 0.5),
                "cost": op.get("cost", 0.5),
                "safety": op.get("safety", 0.5),
            }
            if op.get("policy_effect"):
                edge["policy_effect"] = op["policy_effect"]
            transitions.append(edge)
            applied.append({"op": "create_edge", "from": src, "to": dst})

        elif kind == "update_node":
            name = op.get("name", "")
            patch = op.get("patch", {})
            idx = next((i for i, n in enumerate(nodes) if _node_name(n) == name), None)
            if idx is None:
                return applied, f"update_node: not found: {name}"
            nodes[idx].update({k: v for k, v in patch.items() if k != "name"})
            applied.append({"op": "update_node", "name": name})

        elif kind == "update_edge":
            src, dst = op.get("from", ""), op.get("to", "")
            patch = op.get("patch", {})
            idx = next((i for i, e in enumerate(transitions) if _edge_key(e) == (src, dst)), None)
            if idx is None:
                return applied, f"update_edge: not found: {src} -> {dst}"
            transitions[idx].update({k: v for k, v in patch.items() if k not in ("from", "to")})
            applied.append({"op": "update_edge", "from": src, "to": dst})

        elif kind == "replace_node":
            name = op.get("name", "")
            replacement = dict(op.get("node", {}))
            idx = next((i for i, n in enumerate(nodes) if _node_name(n) == name), None)
            if idx is None:
                return applied, f"replace_node: not found: {name}"
            replacement["name"] = name
            replacement.pop("id", None)
            nodes[idx] = replacement
            applied.append({"op": "replace_node", "name": name})

        elif kind == "replace_edge":
            src, dst = op.get("from", ""), op.get("to", "")
            replacement = dict(op.get("edge", {}))
            idx = next((i for i, e in enumerate(transitions) if _edge_key(e) == (src, dst)), None)
            if idx is None:
                return applied, f"replace_edge: not found: {src} -> {dst}"
            replacement["from"] = src
            replacement["to"] = dst
            transitions[idx] = replacement
            applied.append({"op": "replace_edge", "from": src, "to": dst})

        elif kind == "delete_node":
            name = op.get("name", "")
            before = len(nodes)
            nodes = [n for n in nodes if _node_name(n) != name]
            if len(nodes) == before:
                return applied, f"delete_node: not found: {name}"
            transitions = [e for e in transitions
                           if e.get("from") != name and e.get("to") != name]
            applied.append({"op": "delete_node", "name": name})

        elif kind == "delete_edge":
            src, dst = op.get("from", ""), op.get("to", "")
            before = len(transitions)
            transitions = [e for e in transitions if _edge_key(e) != (src, dst)]
            if len(transitions) == before:
                return applied, f"delete_edge: not found: {src} -> {dst}"
            applied.append({"op": "delete_edge", "from": src, "to": dst})

        else:
            return applied, f"unknown op: {kind!r}"

    topology["nodes"] = nodes
    topology["transitions"] = transitions
    return applied, None


def _apply_contracts(operators: dict, contracts: list) -> tuple[list, str | None]:
    existing = {op.get("node_name") for op in operators.get("operators", [])}
    added = []
    for contract in contracts:
        name = contract.get("node_name", "")
        if not name:
            return added, f"operator_contract missing node_name: {contract}"
        if name in existing:
            return added, f"operator_contract already exists: {name}"
        operators["operators"].append(contract)
        existing.add(name)
        added.append(name)
    return added, None


def _unwrap(payload: dict) -> dict:
    """Unwrap {"context": "<json string>"} envelopes the main loop injects."""
    for _ in range(4):
        if "topology_ops" in payload:
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


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("[topology_mutate] empty stdin\n")
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[topology_mutate] stdin parse error: {exc}\n")
        sys.exit(1)

    payload = _unwrap(payload)
    ops: list              = payload.get("topology_ops", [])
    contracts: list        = payload.get("operator_contracts", [])
    contract_ops: list     = payload.get("operator_contract_ops", [])

    if not ops and not contracts and not contract_ops:
        print(json.dumps({"topology_mutate": {"applied": [], "note": "no ops provided"}}))
        return

    try:
        topology = _load(TOPOLOGY_PATH)
    except Exception as exc:
        sys.stderr.write(f"[topology_mutate] cannot load {TOPOLOGY_PATH}: {exc}\n")
        sys.exit(1)

    applied, error = _apply_ops(topology, ops)
    if error:
        _append_mutation_log(applied, [], [], error)
        print(json.dumps({"topology_mutate": {"error": error}}))
        sys.exit(1)

    contracts_added: list = []
    contracts_updated: list = []
    needs_operators = contracts or contract_ops
    operators_orig: str | None = None

    if needs_operators:
        try:
            operators = _load(OPERATORS_PATH)
        except Exception as exc:
            sys.stderr.write(f"[topology_mutate] cannot load {OPERATORS_PATH}: {exc}\n")
            sys.exit(1)
        operators_orig = json.dumps(operators, indent=2) + "\n"

        if contracts:
            contracts_added, error = _apply_contracts(operators, contracts)
            if error:
                _append_mutation_log(applied, contracts_added, [], error)
                print(json.dumps({"topology_mutate": {"error": error}}))
                sys.exit(1)

        if contract_ops:
            contracts_updated, error = _apply_contract_ops(operators, contract_ops)
            if error:
                _append_mutation_log(applied, contracts_added, contracts_updated, error)
                print(json.dumps({"topology_mutate": {"error": error}}))
                sys.exit(1)

        decision = mutation_policy.decision_for_effects(_EFFECTS)
        if decision == "deny":
            error = "side-effect policy denied topology mutation"
            _append_mutation_log(applied, contracts_added, contracts_updated, error)
            print(json.dumps({"topology_mutate": {"error": error}}))
            sys.exit(1)
        if decision == "stage":
            staged = mutation_policy.stage_mutation(
                source_node="Control::TopologyMutate",
                kind="topology_patch",
                effects=_EFFECTS,
                payload={
                    "topology_ops": ops,
                    "operator_contracts": contracts,
                    "operator_contract_ops": contract_ops,
                    "reason": payload.get("reason", ""),
                },
                summary={
                    "applied_if_approved": applied,
                    "contracts_added_if_approved": contracts_added,
                    "contracts_updated_if_approved": contracts_updated,
                    "node_count_after": len(topology.get("nodes", [])),
                    "edge_count_after": len(topology.get("transitions", [])),
                },
                target_paths=[
                    str(TOPOLOGY_PATH.relative_to(_PROJECT_ROOT)),
                    str(OPERATORS_PATH.relative_to(_PROJECT_ROOT)),
                    str(EXPLORATION_PATH.relative_to(_PROJECT_ROOT)),
                ],
            )
            print(json.dumps({"topology_mutate": staged}))
            return

        limited, wait = _rate_limited()
        if limited:
            print(json.dumps({"topology_mutate": {"skipped": "rate_limited", "next_allowed_in_s": round(wait, 1)}}))
            return

        try:
            TOPOLOGY_BAK.write_text(json.dumps(topology, indent=2) + "\n")
        except OSError:
            pass
        try:
            OPERATORS_BAK.write_text(operators_orig)
        except OSError:
            pass

        try:
            _write(OPERATORS_PATH, operators)
        except OSError as exc:
            if operators_orig:
                try: OPERATORS_PATH.write_text(operators_orig)
                except OSError: pass
            _append_mutation_log(applied, contracts_added, contracts_updated, str(exc))
            print(json.dumps({"topology_mutate": {"error": f"operators write failed: {exc}"}}))
            sys.exit(1)
    else:
        decision = mutation_policy.decision_for_effects(_EFFECTS)
        if decision == "deny":
            error = "side-effect policy denied topology mutation"
            _append_mutation_log(applied, contracts_added, contracts_updated, error)
            print(json.dumps({"topology_mutate": {"error": error}}))
            sys.exit(1)
        if decision == "stage":
            staged = mutation_policy.stage_mutation(
                source_node="Control::TopologyMutate",
                kind="topology_patch",
                effects=_EFFECTS,
                payload={
                    "topology_ops": ops,
                    "operator_contracts": contracts,
                    "operator_contract_ops": contract_ops,
                    "reason": payload.get("reason", ""),
                },
                summary={
                    "applied_if_approved": applied,
                    "contracts_added_if_approved": contracts_added,
                    "contracts_updated_if_approved": contracts_updated,
                    "node_count_after": len(topology.get("nodes", [])),
                    "edge_count_after": len(topology.get("transitions", [])),
                },
                target_paths=[
                    str(TOPOLOGY_PATH.relative_to(_PROJECT_ROOT)),
                    str(OPERATORS_PATH.relative_to(_PROJECT_ROOT)),
                    str(EXPLORATION_PATH.relative_to(_PROJECT_ROOT)),
                ],
            )
            print(json.dumps({"topology_mutate": staged}))
            return

        limited, wait = _rate_limited()
        if limited:
            print(json.dumps({"topology_mutate": {"skipped": "rate_limited", "next_allowed_in_s": round(wait, 1)}}))
            return

        try:
            TOPOLOGY_BAK.write_text(json.dumps(topology, indent=2) + "\n")
        except OSError:
            pass

    try:
        _write(TOPOLOGY_PATH, topology)
    except OSError as exc:
        # restore operators backup
        if operators_orig:
            try: OPERATORS_PATH.write_text(operators_orig)
            except OSError: pass
        _append_mutation_log(applied, contracts_added, contracts_updated, str(exc))
        print(json.dumps({"topology_mutate": {"error": f"topology write failed: {exc}"}}))
        sys.exit(1)

    # update exploration.json for new nodes
    new_nodes = [op["node"]["name"] for op in ops if op.get("op") == "create_node" and "node" in op]
    _update_exploration(new_nodes)

    _append_mutation_log(applied, contracts_added, contracts_updated, None)

    print(json.dumps({
        "topology_mutate": {
            "applied": applied,
            "contracts_added": contracts_added,
            "contracts_updated": contracts_updated,
            "node_count": len(topology.get("nodes", [])),
            "edge_count": len(topology.get("transitions", [])),
        }
    }))


if __name__ == "__main__":
    main()
