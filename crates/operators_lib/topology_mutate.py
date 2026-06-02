#!/usr/bin/env python3
"""Control::TopologyMutate operator: apply CRUD operations to topology nodes and edges.

Reads a JSON payload from stdin with a list of operations, applies them to
assets/topology.json (and assets/operators.json for new operator nodes), and
writes the results back. The runtime fingerprint watcher detects the change on
the next tick; follow with Control::BuildTopologyOverlay to recompile.

Operations (topology_ops array):
  {"op": "create_node",  "node": {"name": "...", "type": "State|Control|Event|Execution|Analysis", "action": "..."}}
  {"op": "create_edge",  "from": "...", "to": "...", "default_weight": 0.5, "confidence": 0.9, "cost": 0.1, "safety": 0.9}
  {"op": "update_node",  "name": "...", "patch": {"action": "..."}}
  {"op": "update_edge",  "from": "...", "to": "...", "patch": {"confidence": 0.8}}
  {"op": "replace_node", "name": "...", "node": {...}}
  {"op": "replace_edge", "from": "...", "to": "...", "edge": {...}}
  {"op": "delete_node",  "name": "..."}
  {"op": "delete_edge",  "from": "...", "to": "..."}

Operator contracts (operator_contracts array):
  {"node_name": "...", "executable": "python3", "static_args": [...], "effects": {...}}

Output (success):
  {"topology_mutate": {"applied": [...], "operator_contracts_added": [...], "node_count": N, "edge_count": N}}

Output (failure):
  {"topology_mutate": {"error": "..."}}
"""

import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TOPOLOGY_PATH = _PROJECT_ROOT / "assets" / "topology.json"
OPERATORS_PATH = _PROJECT_ROOT / "assets" / "operators.json"


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _write(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


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
            dw = op.get("default_weight", 0.5)
            edge: dict = {
                "from": src,
                "to": dst,
                "default_weight": dw,
                "confidence": op.get("confidence", dw),
                "cost": op.get("cost", round(1.0 - dw, 4)),
                "safety": op.get("safety", dw),
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

    ops: list = payload.get("topology_ops", [])
    contracts: list = payload.get("operator_contracts", [])

    if not ops and not contracts:
        print(json.dumps({"topology_mutate": {"applied": [], "note": "no ops provided"}}))
        return

    try:
        topology = _load(TOPOLOGY_PATH)
    except Exception as exc:
        sys.stderr.write(f"[topology_mutate] cannot load {TOPOLOGY_PATH}: {exc}\n")
        sys.exit(1)

    applied, error = _apply_ops(topology, ops)
    if error:
        print(json.dumps({"topology_mutate": {"error": error}}))
        sys.exit(1)

    contracts_added: list = []
    if contracts:
        try:
            operators = _load(OPERATORS_PATH)
        except Exception as exc:
            sys.stderr.write(f"[topology_mutate] cannot load {OPERATORS_PATH}: {exc}\n")
            sys.exit(1)
        contracts_added, error = _apply_contracts(operators, contracts)
        if error:
            print(json.dumps({"topology_mutate": {"error": error}}))
            sys.exit(1)
        _write(OPERATORS_PATH, operators)

    _write(TOPOLOGY_PATH, topology)

    print(json.dumps({
        "topology_mutate": {
            "applied": applied,
            "operator_contracts_added": contracts_added,
            "node_count": len(topology.get("nodes", [])),
            "edge_count": len(topology.get("transitions", [])),
        }
    }))


if __name__ == "__main__":
    main()
