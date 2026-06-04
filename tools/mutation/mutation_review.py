#!/usr/bin/env python3
"""State::MutationReview operator: summarize staged self-modification records."""

import json
import os
import pathlib
import sys


QUEUE_PATH = pathlib.Path(os.environ.get("QUANTALE_MUTATION_QUEUE", "state/mutation_queue.jsonl"))


def load_pending() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    records = []
    for line in QUEUE_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status", "pending") == "pending":
            records.append(record)
    return records


def main() -> None:
    try:
        pending = load_pending()
    except OSError as exc:
        sys.stderr.write(f"[mutation_review] queue read failed: {exc}\n")
        sys.exit(1)
    print(json.dumps({
        "mutation_review": {
            "pending_count": len(pending),
            "pending": pending[:16],
        }
    }))


if __name__ == "__main__":
    main()
