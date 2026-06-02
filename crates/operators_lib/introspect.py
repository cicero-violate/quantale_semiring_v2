#!/usr/bin/env python3
"""State::Introspect operator: produce a structured diagnostic report for State::TopologyPlan.

Reads state/quantale.tlog, state/learned_edges.jsonl, state/topology_mutations.jsonl,
and assets/operators.json to report on node health, stubs, weight trends, recent
mutations, and goal metrics. Output becomes the context fed into State::TopologyPlan.

Output:
  {"introspect": {
    "tlog_window": N,
    "node_stats": [{"node": "...", "visits": N, "success_rate": F}],
    "stub_nodes": [...],
    "never_fired": [...],
    "high_failure_nodes": [...],
    "declining_edges": [...],
    "recent_mutations": [...],
    "goal_metrics": {...}
  }}
"""

import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_ASSET_DIR    = _PROJECT_ROOT / "assets"
_STATE_DIR    = _PROJECT_ROOT / "state"

TLOG_PATH       = _STATE_DIR / "quantale.tlog"
EDGES_PATH      = _STATE_DIR / "learned_edges.jsonl"
MUTATIONS_PATH  = _STATE_DIR / "topology_mutations.jsonl"
FILLS_PATH      = _STATE_DIR / "paper_fills.jsonl"
OPERATORS_PATH  = _ASSET_DIR / "operators.json"
TOPOLOGY_PATH   = _ASSET_DIR / "topology.json"

TLOG_WINDOW      = 300
EDGES_WINDOW     = 60
MUTATIONS_WINDOW = 15
FILLS_WINDOW     = 20
FAILURE_THRESHOLD = 0.4
DECLINE_THRESHOLD = 0.4


def _tail(path: pathlib.Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        for line in path.read_text().splitlines()[-n * 3:]:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    records.append(r)
            except json.JSONDecodeError:
                pass
    except OSError:
        return []
    return records[-n:]


def _node_stats(tlog: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for rec in tlog:
        if rec.get("kind") != "AgentStep":
            continue
        p = rec.get("payload", {})
        node = p.get("node")
        if not node:
            continue
        if node not in stats:
            stats[node] = {"visits": 0, "successes": 0, "failures": 0}
        stats[node]["visits"] += 1
        if p.get("exit_code", 1) == 0:
            stats[node]["successes"] += 1
        else:
            stats[node]["failures"] += 1
    return stats


def _stub_nodes(operators: list[dict]) -> list[str]:
    return [op["node_name"] for op in operators if op.get("executable") == "true"]


def _never_fired(all_nodes: list[str], stats: dict[str, dict]) -> list[str]:
    return [n for n in all_nodes if n not in stats]


def _high_failure(stats: dict[str, dict]) -> list[dict]:
    result = []
    for node, s in stats.items():
        v = s["visits"]
        if v < 2:
            continue
        rate = s["failures"] / v
        if rate >= FAILURE_THRESHOLD:
            result.append({"node": node, "failure_rate": round(rate, 3), "visits": v})
    return sorted(result, key=lambda x: -x["failure_rate"])


def _declining_edges(edges: list[dict]) -> list[dict]:
    # latest-wins per (from, to) pair
    latest: dict[tuple, dict] = {}
    for rec in edges:
        e = rec.get("edge", {})
        key = (e.get("from", ""), e.get("to", ""))
        if key[0] and key[1]:
            latest[key] = e
    return [
        {"from": e["from"], "to": e["to"], "confidence": round(e.get("confidence", 1.0), 4)}
        for e in latest.values()
        if e.get("confidence", 1.0) < DECLINE_THRESHOLD
    ]


def _goal_metrics(tlog: list[dict]) -> dict:
    steps = [r["payload"] for r in tlog if r.get("kind") == "AgentStep"]
    total = len(steps)
    failures = sum(1 for s in steps if s.get("exit_code", 0) != 0)
    failure_rate = round(failures / total, 3) if total else None

    fills = _tail(FILLS_PATH, FILLS_WINDOW)
    net = 0.0
    for f in fills:
        n = f.get("notional", 0.0)
        net += n if f.get("side") == "sell" else -n

    metrics: dict = {"steps_sampled": total}
    if failure_rate is not None:
        metrics["step_failure_rate"] = failure_rate
    if fills:
        metrics["recent_fills"] = len(fills)
        metrics["net_notional_last_fills"] = round(net, 4)
    return metrics


def _mutation_outcomes(mutations: list[dict], tlog: list[dict]) -> list[dict]:
    """Annotate each recent mutation with what happened in the tlog after it fired."""
    import datetime as _dt
    results = []
    # index tlog by sequence for fast range queries
    agent_steps = [
        r for r in tlog if r.get("kind") == "AgentStep"
    ]
    for mut in mutations:
        ts_str = mut.get("ts", "")
        applied = mut.get("result", {}).get("applied", [])
        entry: dict = {"ts": ts_str, "applied_count": len(applied), "applied": applied}
        if ts_str:
            try:
                mut_ts = _dt.datetime.fromisoformat(ts_str)
                if mut_ts.tzinfo is None:
                    mut_ts = mut_ts.replace(tzinfo=_dt.timezone.utc)
                # steps after this mutation
                post_steps = []
                for r in agent_steps:
                    # tlog records don't carry timestamps, so use sequence ordering:
                    # approximate by taking all steps with sequence > mutation sequence
                    # We can't correlate exactly, so just show post-mutation node visits
                    post_steps.append(r["payload"])
                # only keep last 20 steps as "post mutation sample"
                post_steps = post_steps[-20:]
                if post_steps:
                    total = len(post_steps)
                    fails = sum(1 for s in post_steps if s.get("exit_code", 0) != 0)
                    entry["post_failure_rate"] = round(fails / total, 3)
                    entry["post_steps_sampled"] = total
                    new_node_names = [op.get("name") or (op.get("node", {}).get("name")) for op in applied if op.get("op") == "create_node"]
                    new_node_names = [n for n in new_node_names if n]
                    visited_new = [s["node"] for s in post_steps if s.get("node") in new_node_names]
                    entry["new_nodes_visited"] = list(set(visited_new))
            except Exception:
                pass
        results.append(entry)
    return results


def _rollback_recommended(mutations: list[dict], tlog: list[dict]) -> bool:
    """True if the most recent mutation was followed by a significant failure spike."""
    if not mutations:
        return False
    steps = [r["payload"] for r in tlog if r.get("kind") == "AgentStep"]
    if len(steps) < 10:
        return False
    # compare last 10 steps vs prior steps
    recent = steps[-10:]
    prior  = steps[-30:-10] if len(steps) >= 30 else steps[:-10]
    if not prior:
        return False
    recent_fail = sum(1 for s in recent if s.get("exit_code", 0) != 0) / len(recent)
    prior_fail  = sum(1 for s in prior  if s.get("exit_code", 0) != 0) / len(prior)
    return recent_fail > prior_fail + 0.3  # >30pp regression


def main() -> None:
    _ = sys.stdin.read()  # consume stdin (context payload, unused)

    tlog      = _tail(TLOG_PATH, TLOG_WINDOW)
    edges     = _tail(EDGES_PATH, EDGES_WINDOW)
    mutations = _tail(MUTATIONS_PATH, MUTATIONS_WINDOW)

    try:
        operators = json.loads(OPERATORS_PATH.read_text()).get("operators", [])
    except Exception:
        operators = []

    try:
        topo = json.loads(TOPOLOGY_PATH.read_text())
        all_nodes = [n["name"] for n in topo.get("nodes", [])]
    except Exception:
        all_nodes = []

    stats        = _node_stats(tlog)
    stubs        = _stub_nodes(operators)
    never_fired  = _never_fired(all_nodes, stats)
    high_failure = _high_failure(stats)
    declining    = _declining_edges(edges)
    goal         = _goal_metrics(tlog)
    mut_outcomes = _mutation_outcomes(mutations, tlog)
    rollback_rec = _rollback_recommended(mutations, tlog)

    node_stats_out = [
        {
            "node": node,
            "visits": s["visits"],
            "success_rate": round(s["successes"] / s["visits"], 3) if s["visits"] else 0,
        }
        for node, s in sorted(stats.items(), key=lambda x: -x[1]["visits"])
    ]

    # include stub descriptions so LLM knows what to implement
    stub_detail = []
    op_map = {op["node_name"]: op for op in operators}
    for name in stubs:
        entry: dict = {"node": name}
        op = op_map.get(name, {})
        if op.get("description"):
            entry["description"] = op["description"]
        if op.get("effects"):
            entry["effects"] = op["effects"]
        stub_detail.append(entry)

    print(json.dumps({
        "introspect": {
            "tlog_window": len(tlog),
            "node_stats": node_stats_out,
            "stub_nodes": stub_detail,
            "never_fired": never_fired,
            "high_failure_nodes": high_failure,
            "declining_edges": declining,
            "recent_mutations": mut_outcomes,
            "rollback_recommended": rollback_rec,
            "goal_metrics": goal,
        }
    }))


if __name__ == "__main__":
    main()
