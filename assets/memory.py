#!/usr/bin/env python3
"""State::Memory operator — appends the current context to a rolling JSONL
memory file and emits the recent window to stdout as context for the next step.

Exit codes:
  0   success
  1   read/write failure
"""

import sys
import json
import pathlib
import datetime

MEMORY_FILE = pathlib.Path("state/memory.jsonl")
WINDOW = 10  # entries to carry forward as context


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)

    context = payload.get("context", "").strip()
    if not context:
        sys.exit(0)

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "context": context}

    try:
        with MEMORY_FILE.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        sys.stderr.write(f"memory write failed: {exc}\n")
        sys.exit(1)

    # Read the tail for context chaining.
    try:
        lines = MEMORY_FILE.read_text().splitlines()
        recent = [json.loads(l) for l in lines[-WINDOW:] if l.strip()]
    except OSError as exc:
        sys.stderr.write(f"memory read failed: {exc}\n")
        sys.exit(1)

    print(json.dumps({"memory": recent}))


if __name__ == "__main__":
    main()
