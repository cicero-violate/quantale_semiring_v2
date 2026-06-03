#!/usr/bin/env python3
"""State::Score operator: score parse-tree tokens deterministically."""

import json
import sys


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[score] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    tree = payload.get("parse", {}).get("tree") or payload.get("tree") or {}
    tokens = tree.get("tokens", []) if isinstance(tree, dict) else []
    scored = []
    for token in tokens[:64]:
        text = token.get("text", "") if isinstance(token, dict) else str(token)
        scored.append({"text": text, "score": min(1.0, max(0.0, len(text) / 24.0))})
    print(json.dumps({"score": {"items": scored, "count": len(scored)}}))


if __name__ == "__main__":
    main()
