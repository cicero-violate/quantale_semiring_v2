#!/usr/bin/env python3
"""State::Map operator: derive candidate strings from task context."""

import json
import sys


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[map] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    context = payload.get("context", payload.get("task", payload))
    if isinstance(context, (dict, list)):
        text = json.dumps(context, sort_keys=True)
    else:
        text = str(context)
    tokens = [token for token in text.replace("\n", " ").split(" ") if token]
    candidates = [{"term": token, "rank": idx} for idx, token in enumerate(tokens[:32])]
    print(json.dumps({"map": {"candidates": candidates, "count": len(candidates)}}))


if __name__ == "__main__":
    main()
