#!/usr/bin/env python3
"""Control::GoalMutate: receipt reviewed goal deltas and optionally apply them.

stdin: {"goal_review": {...}}
stdout: {"goal_mutate": {...}}
"""

import json
import pathlib
import sys
import uuid
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOALS = ROOT / "tools" / "mutation" / "assets" / "goals.source.json"
LOG = ROOT / "state" / "goal_mutations.jsonl"
BAK = ROOT / "tools" / "mutation" / "assets" / "goals.source.json.bak"

APPLY_ALLOWED = False


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def unwrap_review(payload):
    for _ in range(4):
        if isinstance(payload, dict) and "goal_review" in payload:
            return payload["goal_review"]
        ctx = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(ctx, str):
            break
        try:
            payload = json.loads(ctx.strip())
        except json.JSONDecodeError:
            break
    return payload


def find_goal(goals, collection, goal_id):
    arr = goals.get(collection, [])
    for index, item in enumerate(arr):
        if item.get("id") == goal_id:
            return arr, index
    return arr, None


def apply_delta(goals, delta):
    op = delta.get("op")
    target = delta.get("target", "")
    if target.startswith("root_goal"):
        raise ValueError("root_goal is immutable")

    parts = target.split(".")
    collection = parts[0]
    goal_id = ".".join(parts[1:]) if len(parts) > 1 else delta.get("goal", {}).get("id", "")
    if collection not in ("strategic_goals", "tactical_goals"):
        raise ValueError(f"unsupported collection: {collection}")

    arr, index = find_goal(goals, collection, goal_id)

    if op == "create":
        goal = dict(delta.get("goal", {}))
        if not goal.get("id"):
            raise ValueError("create requires goal.id")
        if index is not None:
            raise ValueError(f"goal already exists: {goal.get('id')}")
        goal.setdefault("mutable", True)
        arr.append(goal)
    elif op == "update":
        if index is None:
            raise ValueError(f"goal not found: {goal_id}")
        if arr[index].get("mutable") is False:
            raise ValueError(f"goal immutable: {goal_id}")
        arr[index].update({k: v for k, v in delta.get("patch", {}).items() if k != "id"})
    elif op == "replace":
        if index is None:
            raise ValueError(f"goal not found: {goal_id}")
        replacement = dict(delta.get("goal", {}))
        replacement["id"] = goal_id
        replacement.setdefault("mutable", True)
        arr[index] = replacement
    elif op == "delete":
        if index is None:
            raise ValueError(f"goal not found: {goal_id}")
        if arr[index].get("mutable") is False:
            raise ValueError(f"goal immutable: {goal_id}")
        del arr[index]
    elif op == "reprioritize":
        if index is None:
            raise ValueError(f"goal not found: {goal_id}")
        arr[index]["priority"] = float(delta.get("priority"))
    else:
        raise ValueError(f"unsupported op: {op}")

    goals[collection] = arr
    return goals


def log_record(record):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    review = unwrap_review(payload)
    decision = review.get("decision")
    delta = review.get("delta", {})
    mutation_id = "goal_" + uuid.uuid4().hex[:16]

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mutation_id": mutation_id,
        "decision": decision,
        "target": delta.get("target"),
        "op": delta.get("op"),
        "reason": delta.get("reason", ""),
    }

    if decision not in ("approved", "staged"):
        record["applied"] = False
        record["error"] = "review was not approved or staged"
        log_record(record)
        print(json.dumps({"goal_mutate": record}, sort_keys=True))
        sys.exit(1)

    if not APPLY_ALLOWED:
        record["applied"] = False
        record["staged_only"] = True
        record["note"] = "APPLY_ALLOWED is false; mutation receipted but not applied"
        log_record(record)
        print(json.dumps({"goal_mutate": record}, sort_keys=True))
        return

    goals = load_json(GOALS)
    BAK.write_text(GOALS.read_text())
    goals = apply_delta(goals, delta)
    write_json(GOALS, goals)
    record["applied"] = True
    record["backup"] = str(BAK.relative_to(ROOT))
    log_record(record)
    print(json.dumps({"goal_mutate": record}, sort_keys=True))


if __name__ == "__main__":
    main()
