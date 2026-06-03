#!/usr/bin/env python3
"""State::Select operator: select top scored items."""

import json
import sys


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[select] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    items = payload.get("score", {}).get("items") or payload.get("items") or []
    if not isinstance(items, list):
        items = []
    selected = sorted(
        [item for item in items if isinstance(item, dict)],
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )[:8]
    print(json.dumps({"select": {"selected": selected, "count": len(selected)}}))


if __name__ == "__main__":
    main()
