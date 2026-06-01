#!/usr/bin/env python3
"""State::Memory operator — appends the current context to a rolling JSONL
memory file and emits the recent window as context for the next step.

Exit codes:
  0   success
  1   read/write failure

Configuration (all via environment variables):
  QUANTALE_MEMORY_FILE    path to memory store, default state/memory.jsonl
  QUANTALE_MEMORY_WINDOW  number of recent entries to forward, default 10
  QUANTALE_MEMORY_KEY     JSON field to extract from stdin as context, default "context"
"""

import sys
import json
import os
import pathlib
import datetime

MEMORY_FILE = pathlib.Path(os.environ.get("QUANTALE_MEMORY_FILE", "state/memory.jsonl"))
WINDOW      = int(os.environ.get("QUANTALE_MEMORY_WINDOW", "10"))
CONTEXT_KEY = os.environ.get("QUANTALE_MEMORY_KEY", "context")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)

    context = payload.get(CONTEXT_KEY, "")
    if isinstance(context, (dict, list)):
        context = json.dumps(context)
    context = str(context).strip()

    if not context:
        sys.exit(0)

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        CONTEXT_KEY: context,
    }

    try:
        with MEMORY_FILE.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        sys.stderr.write(f"memory write failed: {exc}\n")
        sys.exit(1)

    try:
        lines = MEMORY_FILE.read_text().splitlines()
        recent = [json.loads(ln) for ln in lines[-WINDOW:] if ln.strip()]
    except OSError as exc:
        sys.stderr.write(f"memory read failed: {exc}\n")
        sys.exit(1)

    print(json.dumps({"memory": recent}))


if __name__ == "__main__":
    main()
