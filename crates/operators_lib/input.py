#!/usr/bin/env python3
"""State::Input operator: resolve the current task into a context payload."""

import json
import os
import pathlib
import sys


TASK_FILE = pathlib.Path(os.environ.get("QUANTALE_TASK_FILE", "state/task.txt"))


def read_stdin_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {}


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def main() -> None:
    payload = read_stdin_payload()

    if "task" in payload:
        task = stringify(payload.get("task"))
    elif "QUANTALE_TASK" in os.environ:
        task = os.environ.get("QUANTALE_TASK", "")
    elif TASK_FILE.exists():
        try:
            task = TASK_FILE.read_text()
        except OSError as exc:
            sys.stderr.write(f"task file read failed: {exc}\n")
            sys.exit(1)
    else:
        task = ""

    print(json.dumps({"context": task.strip()}))


if __name__ == "__main__":
    main()
