#!/usr/bin/env python3
"""Control::Rollback operator: emit rollback intent evidence without mutating git."""

import datetime
import json
import os
import pathlib
import sys


ROLLBACK_LOG = pathlib.Path(os.environ.get("QUANTALE_ROLLBACK_LOG", "state/rollbacks.jsonl"))


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[rollback] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "node": "Control::Rollback",
        "payload": payload,
        "applied": False,
    }
    try:
        ROLLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ROLLBACK_LOG.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        sys.stderr.write(f"[rollback] log write failed: {exc}\n")
        sys.exit(1)
    print(json.dumps({"rollback": record}))


if __name__ == "__main__":
    main()
