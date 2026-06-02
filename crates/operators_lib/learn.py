#!/usr/bin/env python3
"""State::Learn operator: derive tensor edge deltas from recent receipts."""

import json
import os
import pathlib
import sys
import datetime


MEMORY_FILE = pathlib.Path(os.environ.get("QUANTALE_MEMORY_FILE", "state/memory.jsonl"))
MEMORY_WINDOW = int(os.environ.get("QUANTALE_MEMORY_WINDOW", "10"))
TLOG_FILE = pathlib.Path(os.environ.get("QUANTALE_LEARN_LOG", "state/quantale.tlog"))
LEARNED_EDGES_FILE = pathlib.Path(os.environ.get("QUANTALE_LEARNED_EDGES", "state/learned_edges.jsonl"))
TLOG_WINDOW = int(os.environ.get("QUANTALE_LEARN_WINDOW", "50"))
DELTA_UP = float(os.environ.get("QUANTALE_LEARN_DELTA_UP", "0.05"))
PENALTY = float(os.environ.get("QUANTALE_LEARN_PENALTY", "2.0"))
SHRINK = float(os.environ.get("QUANTALE_LEARN_SHRINK", "0.95"))
TOPOLOGY_FILE = pathlib.Path(os.environ.get("QUANTALE_TOPOLOGY", str(pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "topology.json")))


def read_json_stdin() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {}


def tail_jsonl(path: pathlib.Path, window: int) -> list[dict]:
    if window <= 0 or not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()[-window:]
    except OSError as exc:
        sys.stderr.write(f"read failed for {path}: {exc}\n")
        sys.exit(1)

    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def load_topology() -> tuple[dict[int, str], set[tuple[str, str]]]:
    try:
        data = json.loads(TOPOLOGY_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"topology load failed ({TOPOLOGY_FILE}): {exc}\n")
        sys.exit(1)
    id_to_name = {
        int(node["id"]): node["name"]
        for node in data.get("nodes", [])
        if "id" in node and "name" in node
    }
    static_edges = {
        (transition["from"], transition["to"])
        for transition in data.get("transitions", [])
        if "from" in transition and "to" in transition
    }
    return id_to_name, static_edges


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def edge_from_receipt(receipt: dict, last_decision: dict | None, id_to_name: dict[int, str]) -> dict | None:
    payload = receipt.get("payload", receipt)
    if not isinstance(payload, dict) or "exit_code" not in payload:
        return None

    src_id = payload.get("selected_src")
    dst_id = payload.get("first_hop", payload.get("selected_dst"))
    if (src_id is None or dst_id is None) and last_decision:
        decision_payload = last_decision.get("payload", last_decision)
        if isinstance(decision_payload, dict):
            src_id = decision_payload.get("selected_src", src_id)
            dst_id = decision_payload.get("first_hop", decision_payload.get("selected_dst", dst_id))

    try:
        src_name = id_to_name[int(src_id)]
        dst_name = id_to_name[int(dst_id)]
    except (TypeError, ValueError, KeyError):
        return None

    base = payload.get("selected_value")
    if base is None and last_decision:
        decision_payload = last_decision.get("payload", last_decision)
        if isinstance(decision_payload, dict):
            base = decision_payload.get("selected_value")
    try:
        confidence = clamp(float(base), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.75

    try:
        exit_code = int(payload.get("exit_code"))
    except (TypeError, ValueError):
        return None

    if exit_code == 0:
        return {
            "from": src_name,
            "to": dst_name,
            "confidence": clamp(confidence + DELTA_UP, 0.0, 1.0),
            "cost": max(0.0, (1.0 - confidence) * SHRINK),
            "safety": clamp(confidence + DELTA_UP, 0.0, 1.0),
        }

    return {
        "from": src_name,
        "to": dst_name,
        "confidence": clamp(confidence * 0.9, 0.0, 1.0),
        "cost": max(0.0, (1.0 - confidence) + PENALTY),
        "safety": clamp(confidence * 0.9, 0.0, 1.0),
    }


def append_learned_edges(edges: list[dict]) -> None:
    if not edges:
        return
    LEARNED_EDGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    try:
        with LEARNED_EDGES_FILE.open("a") as fh:
            for edge in edges:
                fh.write(json.dumps({
                    "ts": ts,
                    "source": "State::Learn",
                    "edge": edge,
                }, sort_keys=True) + "\n")
    except OSError as exc:
        sys.stderr.write(f"learned edge write failed: {exc}\n")
        sys.exit(1)


def main() -> None:
    payload = read_json_stdin()
    memory_tail = tail_jsonl(MEMORY_FILE, MEMORY_WINDOW)
    _ = (payload, memory_tail)

    tlog_records = tail_jsonl(TLOG_FILE, TLOG_WINDOW)
    id_to_name, static_edges = load_topology()
    edges = []
    last_decision = None

    for record in tlog_records:
        if record.get("kind") == "Decision":
            last_decision = record
            continue
        edge = edge_from_receipt(record, last_decision, id_to_name)
        if edge and (edge["from"], edge["to"]) in static_edges:
            edges.append(edge)

    append_learned_edges(edges)
    print(json.dumps(edges))


if __name__ == "__main__":
    main()
