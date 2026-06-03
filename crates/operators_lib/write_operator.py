#!/usr/bin/env python3
"""Control::WriteOperator operator: stage or write a Python operator file.

Receives a payload (possibly wrapped in context envelopes) containing:
  filename              — bare filename, must end in .py, no path separators
  source                — complete Python source code
  node_name             — topology node this operator implements (optional, for logging)
  operator_contract_ops — optional list of contract update ops to apply to operators.json
                          e.g. [{"op": "update", "node_name": "...", "patch": {"executable": "python3", ...}}]

Output:
  {"write_operator": {"filename": "...", "node_name": "...", "bytes": N,
                       "path": "...", "contracts_updated": [...]}}
  {"write_operator": {"staged": true, "mutation_id": "...", "queue_path": "..."}}
"""

import json
import pathlib
import sys

import mutation_policy

_PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent
_OPERATORS_LIB = _PROJECT_ROOT / "crates" / "operators_lib"
_OPERATORS_JSON = _PROJECT_ROOT / "assets" / "operators.json"
_EFFECTS = ["repo_write", "operator_registry_write"]


def _apply_contract_ops(ops: list) -> tuple[list, str | None]:
    """Apply operator_contract_ops (update/replace) to operators.json."""
    if not ops:
        return [], None
    try:
        data = json.loads(_OPERATORS_JSON.read_text())
    except Exception as exc:
        return [], f"cannot load operators.json: {exc}"

    updated = []
    for op in ops:
        kind = op.get("op")
        if kind == "update":
            name = op.get("node_name", "")
            patch = op.get("patch", {})
            idx = next((i for i, o in enumerate(data["operators"]) if o.get("node_name") == name), None)
            if idx is None:
                return updated, f"update: operator not found: {name}"
            data["operators"][idx].update({k: v for k, v in patch.items() if k != "node_name"})
            updated.append(name)
        elif kind == "replace":
            contract = op.get("contract", {})
            name = contract.get("node_name", "")
            idx = next((i for i, o in enumerate(data["operators"]) if o.get("node_name") == name), None)
            if idx is None:
                return updated, f"replace: operator not found: {name}"
            data["operators"][idx] = contract
            updated.append(name)
        else:
            return updated, f"unknown op: {kind!r}"

    try:
        _OPERATORS_JSON.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        return updated, f"write failed: {exc}"

    return updated, None


def _unwrap(payload: dict) -> dict:
    """Peel {"context": "<json>"} envelopes injected by the main loop."""
    for _ in range(6):
        if "filename" in payload or "source" in payload:
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


def _validate_filename(name: str) -> str | None:
    """Return error string or None if valid."""
    if not name:
        return "filename is empty"
    if not name.endswith(".py"):
        return f"filename must end in .py: {name!r}"
    if "/" in name or "\\" in name or ".." in name:
        return f"filename must be a bare name with no path components: {name!r}"
    if name.startswith("."):
        return f"filename must not start with '.': {name!r}"
    return None


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("[write_operator] empty stdin\n")
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[write_operator] stdin parse error: {exc}\n")
        sys.exit(1)

    payload = _unwrap(payload)

    filename      = payload.get("filename", "")
    source        = payload.get("source", "")
    node_name     = payload.get("node_name", "")
    contract_ops  = payload.get("operator_contract_ops", [])

    err = _validate_filename(filename)
    if err:
        print(json.dumps({"write_operator": {"error": err}}))
        sys.exit(1)

    if not source.strip():
        print(json.dumps({"write_operator": {"error": "source is empty"}}))
        sys.exit(1)

    # syntax-check before touching disk
    import ast as _ast
    try:
        _ast.parse(source)
    except SyntaxError as exc:
        print(json.dumps({"write_operator": {"error": f"syntax error: {exc}"}}))
        sys.exit(1)

    dest = _OPERATORS_LIB / filename
    if dest.exists():
        print(json.dumps({"write_operator": {
            "error": f"file already exists: {filename} — use a different name or delete first"
        }}))
        sys.exit(1)

    decision = mutation_policy.decision_for_effects(_EFFECTS)
    if decision == "deny":
        print(json.dumps({"write_operator": {"error": "side-effect policy denied operator write"}}))
        sys.exit(1)
    if decision == "stage":
        staged = mutation_policy.stage_mutation(
            source_node="Control::WriteOperator",
            kind="operator_write",
            effects=_EFFECTS,
            payload={
                "filename": filename,
                "source": source,
                "node_name": node_name,
                "operator_contract_ops": contract_ops,
            },
            summary={
                "filename": filename,
                "node_name": node_name,
                "bytes": len(source.encode()),
                "contracts_to_update": len(contract_ops),
            },
            target_paths=[
                str(dest.relative_to(_PROJECT_ROOT)),
                str(_OPERATORS_JSON.relative_to(_PROJECT_ROOT)),
            ],
        )
        print(json.dumps({"write_operator": staged}))
        return

    try:
        dest.write_text(source)
    except OSError as exc:
        print(json.dumps({"write_operator": {"error": f"write failed: {exc}"}}))
        sys.exit(1)

    # apply any operator_contract_ops to upgrade executable: true -> python3
    contracts_updated, contract_err = _apply_contract_ops(contract_ops)
    if contract_err:
        sys.stderr.write(f"[write_operator] contract_ops warning: {contract_err}\n")

    print(json.dumps({
        "write_operator": {
            "filename": filename,
            "node_name": node_name,
            "bytes": len(source.encode()),
            "path": str(dest),
            "contracts_updated": contracts_updated,
        }
    }))


if __name__ == "__main__":
    main()
