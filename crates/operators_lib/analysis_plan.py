#!/usr/bin/env python3
"""Standalone analysis-plan validator: calls call_llm.py --template analysis,
validates the returned chain against declared topology and operators, then
prints a structured analysis-plan decision.

This script performs extra chain validation beyond what call_llm.py emits.
The operator registry entry for State::AnalysisPlan calls call_llm.py directly;
this script is useful for testing and as an alternative operator with validation.

Validation rules:
  - analysis_chain must be a non-empty array of strings
  - each operator name must exist in assets/topology.generated.json nodes
  - each operator name must exist in assets/operators.generated.json with executable=jit_cuda
  - slot dependencies must be satisfiable: each operator's reads must be
    covered by a previous operator's writes or the market feed features
"""

import json
import os
import pathlib
import subprocess
import sys

_OPERATORS_LIB = pathlib.Path(__file__).resolve().parent
_ASSET_DIR = _OPERATORS_LIB.parent.parent.parent / "assets"
TOPOLOGY_PATH = pathlib.Path(os.environ.get("QUANTALE_TOPOLOGY", _ASSET_DIR / "topology.generated.json"))
OPERATORS_PATH = pathlib.Path(os.environ.get("QUANTALE_OPERATORS", _ASSET_DIR / "operators.generated.json"))
MARKET_ANALYSIS_PATH = _ASSET_DIR / "market_analysis.json"
ANALYSIS_SCHEMA_PATH = _ASSET_DIR / "analysis_decision_schema.json"


def load_topology_node_names() -> set:
    data = json.loads(TOPOLOGY_PATH.read_text())
    return {n["name"] for n in data.get("nodes", [])}


def load_jit_operators() -> dict:
    data = json.loads(OPERATORS_PATH.read_text())
    return {
        op["node_name"]: op
        for op in data.get("operators", [])
        if op.get("executable") == "jit_cuda"
    }


def load_market_features() -> set:
    try:
        data = json.loads(MARKET_ANALYSIS_PATH.read_text())
        return set(data.get("features", {}).values())
    except Exception:
        return {"market.price", "market.volume"}


def validate_chain(chain: list, topology_nodes: set, jit_ops: dict, initial_slots: set) -> str | None:
    """Return an error string if invalid, None if valid."""
    if not chain:
        return "analysis_chain is empty"
    available_slots = set(initial_slots)
    for name in chain:
        if name not in topology_nodes:
            return f"operator not in topology: {name}"
        if name not in jit_ops:
            return f"operator not a jit_cuda operator: {name}"
        op = jit_ops[name]
        effects = op.get("effects", {})
        reads = effects.get("reads", [])
        for slot in reads:
            if slot not in available_slots:
                return f"slot '{slot}' required by '{name}' is not available"
        available_slots.update(effects.get("writes", []))
    return None


def read_stdin_payload() -> str:
    return sys.stdin.read()


def main() -> None:
    stdin_data = read_stdin_payload()

    result = subprocess.run(
        [sys.executable, str(_OPERATORS_LIB / "call_llm.py"), "--template", "analysis"],
        input=stdin_data,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.stderr.write(f"[analysis_plan] call_llm.py failed (exit {result.returncode})\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
        sys.exit(result.returncode)

    raw = result.stdout.strip()
    if not raw:
        sys.stderr.write("[analysis_plan] call_llm.py returned empty output\n")
        sys.exit(1)

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[analysis_plan] LLM output is not valid JSON: {exc}\n")
        sys.stderr.write(f"raw output: {raw[:512]}\n")
        sys.exit(1)

    chain = decision.get("analysis_chain")
    reason = decision.get("reason", "")

    if not isinstance(chain, list):
        sys.stderr.write("[analysis_plan] analysis_chain must be an array\n")
        sys.exit(1)

    try:
        topology_nodes = load_topology_node_names()
        jit_ops = load_jit_operators()
        initial_slots = load_market_features()
    except Exception as exc:
        sys.stderr.write(f"[analysis_plan] asset load failed: {exc}\n")
        sys.exit(1)

    error = validate_chain(chain, topology_nodes, jit_ops, initial_slots)
    if error:
        sys.stderr.write(f"[analysis_plan] chain validation failed: {error}\n")
        sys.exit(1)

    output = {
        "analysis_plan": {
            "analysis_chain": chain,
            "reason": reason,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
