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
import re
import sys

import mutation_policy

_PROJECT_ROOT  = pathlib.Path(__file__).resolve().parents[2]
_OPERATORS_LIB = _PROJECT_ROOT / "crates" / "operators_lib"
_OPERATORS_JSON = _PROJECT_ROOT / "assets" / "operators.json"
_EFFECTS = ["repo_write", "operator_registry_write"]


def _error(message: str, **details) -> None:
    body = {"error": message}
    body.update(details)
    print(json.dumps({"write_operator": body}))
    sys.exit(1)


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


def _contract_entry(node_name: str) -> dict | None:
    try:
        data = json.loads(_OPERATORS_JSON.read_text())
    except Exception:
        return None
    for op in data.get("operators", []):
        if op.get("node_name") == node_name:
            return op
    return None


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


def _validate_contract_ops(contract_ops, *, node_name: str, filename: str) -> list[str]:
    errors = []
    if not isinstance(contract_ops, list):
        return ["operator_contract_ops must be a list"]
    if not node_name:
        errors.append("node_name is required")
    if not contract_ops:
        errors.append("operator_contract_ops must include an update op")
        return errors

    expected_arg = f"crates/operators_lib/{filename}"
    found_update = False
    for idx, op in enumerate(contract_ops):
        if not isinstance(op, dict):
            errors.append(f"operator_contract_ops[{idx}] must be an object")
            continue
        if op.get("op") != "update":
            errors.append(f"operator_contract_ops[{idx}].op must be 'update'")
        if op.get("node_name") != node_name:
            errors.append(f"operator_contract_ops[{idx}].node_name must be {node_name!r}")
        patch = op.get("patch")
        if not isinstance(patch, dict):
            errors.append(f"operator_contract_ops[{idx}].patch must be an object")
            continue
        if patch.get("executable") != "python3":
            errors.append(f"operator_contract_ops[{idx}].patch.executable must be 'python3'")
        static_args = patch.get("static_args")
        if static_args != [expected_arg]:
            errors.append(
                f"operator_contract_ops[{idx}].patch.static_args must be [{expected_arg!r}]"
            )
        input_mapping = patch.get("input_mapping", {})
        if input_mapping.get("stdin_mode") != "json":
            errors.append(f"operator_contract_ops[{idx}].patch.input_mapping.stdin_mode must be 'json'")
        found_update = True

    entry = _contract_entry(node_name)
    if entry is None:
        errors.append(f"operator contract not found in assets/operators.json: {node_name}")
    elif entry.get("executable") != "true":
        errors.append(f"operator contract is not an executable=true stub: {node_name}")
    if not found_update:
        errors.append("operator_contract_ops must contain an update op")
    return errors


def _validate_source(source, *, node_name: str) -> tuple[str | None, int]:
    if not isinstance(source, str):
        return "source must be a string", 0
    if not source.strip():
        return "source is empty", 0
    if len(source.splitlines()) > 150:
        return "source must be under 150 lines", 0
    if not source.startswith("#!/usr/bin/env python3\n"):
        return "source must start with #!/usr/bin/env python3", 0
    if "pathlib.Path(file)" in source or re.search(r"\bPath\s*\(\s*file\s*\)", source):
        return "source uses file instead of __file__", 0
    if re.search(r"if\s+name\s*==\s*[\"']main[\"']", source):
        return "source uses name/main instead of __name__/__main__", 0
    if 'if __name__ == "__main__":' not in source and "if __name__ == '__main__':" not in source:
        return "source must include an exact __name__ == '__main__' guard", 0

    import ast as _ast
    try:
        tree = _ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc}", 0

    doc = _ast.get_docstring(tree) or ""
    if node_name and node_name not in doc:
        return "module docstring must include node name", 0
    doc_lower = doc.lower()
    for required in ("stdin", "stdout"):
        if required not in doc_lower:
            return f"module docstring must describe {required} shape", 0

    stdlib = getattr(sys, "stdlib_module_names", set())
    for node in _ast.walk(tree):
        module = ""
        if isinstance(node, _ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in stdlib:
                    return f"source imports non-stdlib module: {alias.name}", 0
        elif isinstance(node, _ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module and module not in stdlib:
                return f"source imports non-stdlib module: {node.module}", 0

    return None, len(source.encode())


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("[write_operator] empty stdin\n")
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _error(f"stdin parse error: {exc}")

    payload = _unwrap(payload)

    filename      = payload.get("filename", "")
    source        = payload.get("source", "")
    node_name     = payload.get("node_name", "")
    contract_ops  = payload.get("operator_contract_ops", [])

    err = _validate_filename(filename)
    if err:
        _error(err)

    source_err, source_bytes = _validate_source(source, node_name=node_name)
    if source_err:
        _error(source_err)

    contract_errors = _validate_contract_ops(contract_ops, node_name=node_name, filename=filename)
    if contract_errors:
        _error("invalid operator_contract_ops", validation_errors=contract_errors)

    dest = _OPERATORS_LIB / filename
    if dest.exists():
        _error(f"file already exists: {filename}; operator creation does not overwrite existing files")

    decision = mutation_policy.decision_for_effects(_EFFECTS)
    if decision == "deny":
        _error("side-effect policy denied operator write")
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
                "bytes": source_bytes,
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
        _error(f"write failed: {exc}")

    # apply any operator_contract_ops to upgrade executable: true -> python3
    contracts_updated, contract_err = _apply_contract_ops(contract_ops)
    if contract_err:
        sys.stderr.write(f"[write_operator] contract_ops warning: {contract_err}\n")

    print(json.dumps({
        "write_operator": {
            "filename": filename,
            "node_name": node_name,
            "bytes": source_bytes,
            "path": str(dest),
            "contracts_updated": contracts_updated,
        }
    }))


if __name__ == "__main__":
    main()
