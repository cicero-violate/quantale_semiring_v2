#!/usr/bin/env python3
"""Control::Commit operator: persist commit evidence only."""

import datetime
import json
import os
import pathlib
import sys


COMMIT_LOG = pathlib.Path(os.environ.get("QUANTALE_COMMIT_LOG", "state/commits.jsonl"))


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "node": "Control::Commit",
        "payload": payload,
    }

    try:
        COMMIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with COMMIT_LOG.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        sys.stderr.write(f"commit log write failed: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
