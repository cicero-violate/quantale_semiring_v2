#!/usr/bin/env python3
"""Shared Event::* acknowledgement operator."""

import argparse
import datetime
import json
import os
import pathlib
import sys


EVENT_LOG = pathlib.Path(os.environ.get("QUANTALE_EVENT_LOG", "state/events.jsonl"))
FACT_LOG = pathlib.Path(os.environ.get("QUANTALE_FACT_LOG", "state/facts.jsonl"))


def read_stdin_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantale event acknowledgement")
    parser.add_argument("--event", required=True)
    args = parser.parse_args()

    payload = read_stdin_payload()
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": args.event,
        "payload": payload,
    }

    try:
        append_jsonl(EVENT_LOG, record)
        if args.event == "FactArrived":
            append_jsonl(FACT_LOG, record)
    except OSError as exc:
        sys.stderr.write(f"event log write failed: {exc}\n")
        sys.exit(1)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
