#!/usr/bin/env python3
"""Control::WriteOperator operator: write a new Python operator file to crates/operators_lib/.

Receives a payload (possibly wrapped in context envelopes) containing:
  filename  — bare filename, must end in .py, no path separators
  source    — complete Python source code
  node_name — the topology node this operator implements (optional, for logging)

Output:
  {"write_operator": {"filename": "...", "node_name": "...", "bytes": N, "path": "..."}}
"""

import json
import pathlib
import sys

_PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent
_OPERATORS_LIB = _PROJECT_ROOT / "crates" / "operators_lib"


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

    filename  = payload.get("filename", "")
    source    = payload.get("source", "")
    node_name = payload.get("node_name", "")

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

    try:
        dest.write_text(source)
    except OSError as exc:
        print(json.dumps({"write_operator": {"error": f"write failed: {exc}"}}))
        sys.exit(1)

    print(json.dumps({
        "write_operator": {
            "filename": filename,
            "node_name": node_name,
            "bytes": len(source.encode()),
            "path": str(dest),
        }
    }))


if __name__ == "__main__":
    main()
