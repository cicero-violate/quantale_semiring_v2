#!/usr/bin/env python3
"""Review or apply pending records from state/mutation_queue.jsonl.

This is the explicit escape hatch for queued repo mutations. It replays each
pending record through the original operator with QUANTALE_MUTATION_MODE=apply
only when --apply is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import mutation_policy

SCRIPT_BY_KIND = {
    "operator_write": "write_operator.py",
    "topology_patch": "topology_mutate.py",
    "pattern_patch": "pattern_mutate.py",
}


def _load_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_records(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _apply_record(record: dict) -> dict:
    script_name = SCRIPT_BY_KIND.get(record.get("kind", ""))
    if not script_name:
        return {"exit_code": 1, "stderr": f"unknown mutation kind: {record.get('kind')!r}"}

    script_path = pathlib.Path(__file__).resolve().parent / script_name
    env = os.environ.copy()
    env["QUANTALE_MUTATION_MODE"] = "apply"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(record.get("payload", {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _list_records(records: list[dict], mutation_id: str = "") -> dict:
    pending = []
    for record in records:
        if mutation_id and record.get("id") != mutation_id:
            continue
        if record.get("status") != "pending":
            continue
        pending.append({
            "id": record.get("id", ""),
            "ts": record.get("ts", ""),
            "kind": record.get("kind", ""),
            "source_node": record.get("source_node", ""),
            "effects": record.get("effects", []),
            "target_paths": record.get("target_paths", []),
            "summary": record.get("summary", {}),
        })
    return {"pending": pending, "pending_count": len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="mutation_id", default="")
    parser.add_argument("--queue", default=str(mutation_policy.DEFAULT_QUEUE_PATH))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    queue_path = pathlib.Path(args.queue)
    records = _load_records(queue_path)

    if args.list or not args.apply:
        listed = _list_records(records, args.mutation_id)
        listed["queue_path"] = str(queue_path)
        print(json.dumps({"mutation_queue": listed}))
        return

    applied = []
    for record in records:
        if record.get("status") != "pending":
            continue
        if args.mutation_id and record.get("id") != args.mutation_id:
            continue
        result = _apply_record(record)
        record["apply_result"] = result
        record["status"] = "applied" if result["exit_code"] == 0 else "failed"
        applied.append({"id": record.get("id"), "status": record["status"]})

    _write_records(queue_path, records)
    print(json.dumps({"apply_mutations": {"records": applied, "queue_path": str(queue_path)}}))


if __name__ == "__main__":
    main()
