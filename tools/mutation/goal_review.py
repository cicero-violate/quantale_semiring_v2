#!/usr/bin/env python3
"""Control::GoalReview: validate a proposed goal delta against goal policy.

stdin: {"goal_delta": {...}} or an unwrapped delta
stdout: {"goal_review": {"decision": "staged"|"denied", ...}}
"""

import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOALS = ROOT / "tools" / "mutation" / "assets" / "goals.source.json"
POLICY = ROOT / "tools" / "mutation" / "assets" / "goal_policy.json"


def load_json(path):
    return json.loads(path.read_text())


def unwrap(payload):
    for _ in range(4):
        if isinstance(payload, dict) and "goal_delta" in payload:
            return payload
        ctx = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(ctx, str):
            break
        try:
            payload = json.loads(ctx.strip())
        except json.JSONDecodeError:
            break
    return payload


def starts_any(value, prefixes):
    return any(str(value).startswith(prefix) for prefix in prefixes)


def benefit_ok(delta, policy):
    benefit = delta.get("expected_benefit", {})
    mins = policy.get("benefit_requirements", {}).get("min_any_of", {})
    return any(float(benefit.get(k, 0)) >= float(v) for k, v in mins.items())


def risk_ok(delta, policy):
    risk = delta.get("risk", {})
    limits = policy.get("risk_limits", {})
    mapping = {
        "alignment_drift": "max_alignment_drift",
        "execution_instability": "max_execution_instability",
        "governance_weakening": "max_governance_weakening",
        "hidden_io_risk": "max_hidden_io_risk",
    }
    for risk_key, limit_key in mapping.items():
        if float(risk.get(risk_key, 0)) > float(limits.get(limit_key, 0)):
            return False, f"risk too high: {risk_key}"
    return True, None


def review(delta, goals, policy):
    _ = goals
    errors = []
    op = delta.get("op")
    target = delta.get("target", "")

    if op not in policy.get("allowed_ops", []):
        errors.append(f"op not allowed: {op}")

    for field in policy.get("required_fields", {}).get("all_ops", []):
        if field not in delta:
            errors.append(f"missing required field: {field}")
    for field in policy.get("required_fields", {}).get(str(op), []):
        if field not in delta:
            errors.append(f"missing required field for {op}: {field}")

    if starts_any(target, policy.get("denied_target_prefixes", [])):
        errors.append(f"denied target: {target}")
    if not starts_any(target, policy.get("allowed_target_prefixes", [])):
        errors.append(f"target prefix not allowed: {target}")

    if not benefit_ok(delta, policy):
        errors.append("expected_benefit does not meet minimum")

    ok, err = risk_ok(delta, policy)
    if not ok:
        errors.append(err)

    patch_text = json.dumps(delta.get("patch", {})).lower()
    denied_terms = [
        "human_authority",
        "receipts_required",
        "governance",
        "rollback",
    ]
    for term in denied_terms:
        if term in patch_text and "false" in patch_text:
            errors.append(f"possible forbidden weakening: {term}")

    decision = "denied"
    if not errors:
        decision = "approved"
        if policy.get("default_decision") == "stage":
            decision = "staged"
        if policy.get("approval", {}).get("auto_apply_allowed", False):
            decision = "approved"

    return {
        "goal_review": {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decision": decision,
            "errors": errors,
            "delta": delta,
            "root_goal_preserved": target != "root_goal"
            and not target.startswith("root_goal"),
        }
    }


def main():
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    payload = unwrap(payload)
    delta = payload.get("goal_delta", payload)
    goals = load_json(GOALS)
    policy = load_json(POLICY)
    result = review(delta, goals, policy)
    print(json.dumps(result, sort_keys=True))
    if result["goal_review"]["decision"] == "denied":
        sys.exit(1)


if __name__ == "__main__":
    main()
