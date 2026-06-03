#!/usr/bin/env python3
"""State::Search operator: rank mapped candidates without external I/O."""

import json
import sys


def read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[search] stdin parse error: {exc}\n")
        sys.exit(1)
    return payload if isinstance(payload, dict) else {"value": payload}


def main() -> None:
    payload = read_payload()
    candidates = (
        payload.get("map", {}).get("candidates")
        or payload.get("candidates")
        or []
    )
    if not isinstance(candidates, list):
        candidates = []
    results = []
    for item in candidates[:32]:
        term = item.get("term", "") if isinstance(item, dict) else str(item)
        score = 1.0 / (1.0 + len(results))
        results.append({"term": term, "score": score})
    print(json.dumps({"search": {"results": results, "count": len(results)}}))


if __name__ == "__main__":
    main()
