#!/usr/bin/env python3
"""State::Parse operator: produce a minimal parse tree from task context."""

import json
import sys


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[parse] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    context = payload.get("context", payload.get("task", payload))
    text = json.dumps(context, sort_keys=True) if isinstance(context, (dict, list)) else str(context)
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    tree = {
        "kind": "task",
        "tokens": [{"text": word, "index": idx} for idx, word in enumerate(words[:64])],
    }
    print(json.dumps({"parse": {"tree": tree, "token_count": len(tree["tokens"])}}))


if __name__ == "__main__":
    main()
