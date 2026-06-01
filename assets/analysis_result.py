#!/usr/bin/env python3
"""State analysis-result normalizer: reads JIT analysis receipts from context,
normalizes outputs into analysis.result, persists to state/analysis_results.jsonl,
and prints structured analysis output for State::TradePlan.

This script does no math; math belongs in JIT kernels declared in operators.json.
"""

import datetime
import json
import pathlib
import sys

STATE_DIR = pathlib.Path("state")
RESULTS_LOG = STATE_DIR / "analysis_results.jsonl"


def read_stdin_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[analysis_result] stdin parse error: {exc}\n")
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_jit_receipts(payload) -> list:
    """Collect all JIT kernel result objects from the nested payload."""
    receipts = []
    if isinstance(payload, dict):
        if "kernel" in payload and "results" in payload:
            receipts.append(payload)
        else:
            for v in payload.values():
                receipts.extend(extract_jit_receipts(v))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                stdout = item.get("stdout", "")
                if stdout:
                    try:
                        parsed = json.loads(stdout)
                        receipts.extend(extract_jit_receipts(parsed))
                    except json.JSONDecodeError:
                        pass
                receipts.extend(extract_jit_receipts(item))
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                receipts.extend(extract_jit_receipts(parsed))
            except json.JSONDecodeError:
                pass
    return receipts


def normalize_receipts(receipts: list) -> dict:
    """Build analysis.result from JIT kernel receipts."""
    result = {}
    for receipt in receipts:
        node = receipt.get("node", "unknown")
        outputs = receipt.get("outputs", [])
        values = receipt.get("results", [])
        for i, slot in enumerate(outputs):
            result[slot] = {
                "node": node,
                "values": values[:8] if isinstance(values, list) else [],
            }
    return result


def main() -> None:
    payload = read_stdin_payload()
    receipts = extract_jit_receipts(payload)
    analysis_result = normalize_receipts(receipts)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    record = {
        "ts": ts,
        "analysis_result": analysis_result,
        "receipt_count": len(receipts),
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with RESULTS_LOG.open("a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:
        sys.stderr.write(f"[analysis_result] log write failed: {exc}\n")

    output = {
        "analysis_result": analysis_result,
        "ts": ts,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
