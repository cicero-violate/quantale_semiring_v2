#!/usr/bin/env python3
"""State::GoalIntrospect: summarize current goals and recent runtime context.

stdin: ignored JSON object
stdout: {"goal_report": {...}}
"""

import json
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOALS = ROOT / "tools" / "mutation" / "assets" / "goals.source.json"
TLOG = ROOT / "state" / "quantale.tlog"


def load_json(path):
    return json.loads(path.read_text())


def tail(path, n=40):
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-n:]


def main():
    goals = load_json(GOALS)
    tactical = goals.get("tactical_goals", [])
    strategic = goals.get("strategic_goals", [])
    report = {
        "goal_report": {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "root_goal_id": goals.get("root_goal", {}).get("id"),
            "root_mutable": goals.get("root_goal", {}).get("mutable"),
            "strategic_count": len(strategic),
            "tactical_count": len(tactical),
            "top_tactical": sorted(
                tactical,
                key=lambda goal: goal.get("priority", 0),
                reverse=True,
            )[:5],
            "recent_tlog_tail": tail(TLOG, 20),
        }
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
