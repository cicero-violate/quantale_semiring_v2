#!/usr/bin/env python3
"""Shared mutation staging policy for side-effecting operators."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import uuid

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
POLICY_PATH = PROJECT_ROOT / "assets" / "side_effect_policy.json"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "state" / "mutation_queue.jsonl"


def _queue_path() -> pathlib.Path:
    override = os.environ.get("QUANTALE_MUTATION_QUEUE", "").strip()
    return pathlib.Path(override) if override else DEFAULT_QUEUE_PATH


def _load_policy() -> dict:
    try:
        return json.loads(POLICY_PATH.read_text())
    except Exception:
        return {"default": "allow", "effects": {}}


def decision_for_effects(effects: list[str]) -> str:
    """Return allow, stage, or deny for the declared side effects."""
    override = os.environ.get("QUANTALE_MUTATION_MODE", "").strip().lower()
    if override in {"apply", "allow"}:
        return "allow"
    if override in {"stage", "deny"}:
        return override

    policy = _load_policy()
    effect_policy = policy.get("effects", {})
    default = policy.get("default", "allow")
    modes = [effect_policy.get(effect, default) for effect in effects]
    if "deny" in modes:
        return "deny"
    if "stage" in modes:
        return "stage"
    return "allow"


def stage_mutation(
    *,
    source_node: str,
    kind: str,
    effects: list[str],
    payload: dict,
    summary: dict,
    target_paths: list[str],
) -> dict:
    """Append a pending mutation proposal and return the staged result."""
    queue_path = _queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "status": "pending",
        "source_node": source_node,
        "kind": kind,
        "effects": effects,
        "target_paths": target_paths,
        "summary": summary,
        "payload": payload,
    }
    with queue_path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "staged": True,
        "mutation_id": record["id"],
        "queue_path": str(queue_path),
        "summary": summary,
    }
