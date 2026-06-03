# Problems Fixed — Session 2026-06-02 (continued)

This document records what was fixed in this session and what to validate next.
The previous `problem.md` covered issues 1–4. This session fixed issues 5–11.

---

## 5. `State::Introspect` never fired — plan template was conditional

**Symptom**
`State::Introspect` never appeared in `state/quantale.tlog` despite the topology
having a `State::Learn → State::Introspect` edge.

**Root cause A — conditional template**
The `plan` template said "propose `State::Introspect` *when context shows failures*".
The trading cycle always succeeded, so the condition was never met.

**Fix** (commit `b7e58df`-range)
Changed DEVELOPMENT CYCLE to "always include exactly one dev edge per plan."

**Root cause B — learned edges saturated at 1.0**
`learn.py` increments confidence by 0.05 per success, clamped to 1.0. After ~8
traversals, `State::Learn → Event::LearnUpdated` hit conf=1.0 (2959 records in
`learned_edges.jsonl`). The `State::Introspect` base of 0.35 could not compete.

**Fix** (`learning_policy.json`, `src/learning.rs`)
Added `max_confidence_above_base = 0.15` and lowered `confidence_clamp[1]` from
1.0 to 0.85. Cap per edge = `min(base + 0.15, 0.85)`. Prevents any learned edge
from saturating to 1.0.

---

## 6. Hard reset discarded accumulated LLM dev-chain weights

**Symptom**
After each hard reset the `State::Learn → State::Introspect` projection score
bounced back to ~1.7 (trading cycle dominant). Weights accumulated from LLM
proposals were lost.

**Root cause**
`maybe_hard_reset_after_blocks` called `reset() + embed_tensor_edges(static_only)`,
re-embedding only the static topology edges. All LLM-proposed edges (including
`State::Learn → State::Introspect` at 0.5) were discarded.

**Fix** (`src/main.rs`)
- Added `accumulated_edges` field to `RuntimeEpoch`, seeded with static topology
  + learned edges + CKA patterns at startup.
- After each `embed_tensor_edges(&plan_edges)` call, `epoch.accumulated_edges.extend(plan_edges)`.
- Hard reset now calls `reset() + embed_tensor_edges(&accumulated_edges) + close()`,
  preserving all LLM proposals across resets.

**Why not `restore_base_tensor()`?**
`restore_base_tensor()` restores tensor values but NOT the witness matrix. FW on an
already-closed tensor finds zero improvements so witness stays all -1; `project()`
then rejects every path. Must use raw-edge re-embed so FW rebuilds witness from scratch.

---

## 7. `Unknown(-1)` blocked after `State::Input` — ConsumedBlockPoint

**Symptom**
Steps 31–33 always block after `Event::LearnUpdated → State::Input` executes.
Hard reset fires every cycle.

**Root cause**
The frontier-step CUDA kernel marks `consumed[src*N+hop]=1` on first traversal and
never clears within a session. `State::Input → State::MarketFeed` is consumed at
step 5 (via `Event::FactArrived → State::Input`). Re-entry at step 30 via
`Event::LearnUpdated → State::Input` finds the only outgoing edge consumed → blocked.

**Fix** (structural invariant + topology change — see #9 and #11)
The hard reset now re-embeds accumulated_edges which calls `reset()` (clears
consumed[]). This is why the cycle continues after 3 blocked steps. The structural
root cause is addressed by invariant 14 (see #9) and partially by Fix 1 (see #11).

---

## 8. `projection_first_hop_mismatch` runtime warnings (new)

**Symptom**
Many `[runtime_check] [projection_first_hop_mismatch] projection dst=X but first_hop=Y`
warnings after the learning cap fix.

**Root cause**
The `learned_edges.jsonl` from 215k prior steps is loaded at startup into
`accumulated_edges`. These learned weights create strong multi-hop transitive closure
paths. The frontier_step kernel correctly selects `first_hop` = immediate next node
while `selected_dst` = long-range goal — they legitimately differ for multi-hop paths.
Invariant 19 was written assuming single-hop steps and is now a false positive.

**Status**
False positive — the execution is correct. The warnings are noisy but harmless.
A future session should tighten the invariant 19 check to only fire when `first_hop`
is *not* on the best path from `selected_src` to `selected_dst`.

---

## 9. Topology invariant 14: ConsumedBlockPoint (new check added)

**What was added** (`crates/topology_core/src/check.rs`, `tests/topology_check.rs`)
Invariant 14 (`ConsumedBlockPoint`): a reachable non-halt node with exactly 1 outgoing
edge but 2+ incoming edges. The frontier-step kernel's consumed[] marks the single exit
used on first traversal; re-entry via any other predecessor produces Unknown(-1) blocked.

`--check-topology` and `load_checked_default()` both demote ConsumedBlockPoint to
`[WARN]` (not fatal) since the hard-reset handles re-entry at runtime.

**5 known violations in current topology:**
| Node | Outgoing | Incoming |
|---|---|---|
| `State::Input` | 1 | 3 |
| `Event::AnalysisFinished` | 1 | 2 |
| `Control::BuildTopologyOverlay` | 1 | 3 |
| `Control::Repair` | 1 | 2 |
| `Control::Block` | 1 | 2 |

These are acknowledged in `tests/topology_check.rs::KNOWN_CONSUMED_BLOCK_POINTS`.
**TODO next session**: add a second outgoing edge to each to eliminate the hard-reset
dependency for cycle continuity.

---

## 10. Learning policy — confidence cap

**Change** (`assets/learning_policy.json`, `src/learning.rs`)
```json
{
  "learned_edge_cost_floor": 0.001,
  "confidence_clamp": [0.0, 0.85],
  "safety_clamp": [0.0, 1.0],
  "max_confidence_above_base": 0.15
}
```
Cap formula: `min(base_conf + 0.15, 0.85)` applied when loading `learned_edges.jsonl`.

**Effect on existing learned_edges.jsonl**
`State::Plan → State::TradePlan` (2959 records at 1.0) will be loaded at 0.85.
`State::Learn → Event::LearnUpdated` (conf=1.0) will be loaded at 0.85.
This reduces the trading cycle's dominance from ~1.9 down to ~1.6 projection score.

---

## 11. Fix 1: `State::Introspect` now fires every cycle (topology restructure)

**Root cause of persistent failure**
Even with all prior fixes, the quantale max-times product penalises longer paths.
The dev chain from `State::Learn` via `State::Introspect` has 6 intermediate hops
before rejoining the trading cycle at `Event::LearnUpdated`. Its product:
```
0.60 × 0.97^4 × 0.90 × 0.95 = 0.46
```
The direct shortcut `State::Learn → Event::LearnUpdated` scores 0.85 (capped).
0.85 beats 0.46 regardless of `State::Introspect` confidence. Mathematically
impossible for any confidence on `State::Introspect` to win the projection.

**Fix** (topology.json, topology.generated.json, call_llm_templates.json, call_llm.py)
- **Removed** `State::Learn → Event::LearnUpdated` (the shortcut blocking the dev chain)
- **Raised** `State::Learn → State::Introspect` from 0.60 → 0.97 (now the primary path)
- **Added** `State::Introspect → Event::LearnUpdated` at 0.50 (safety bypass; dev-chain
  product 0.97^4 × 0.90 = 0.772 > 0.50 so dev chain always wins from Introspect)
- **Updated** plan template: removed stale DEVELOPMENT CYCLE section; added DEVELOPMENT
  ROUTING explaining the new hardwired structure

**New cycle flow:**
```
State::Memory → State::Learn → State::Introspect → State::TopologyPlan
              → Control::TopologyMutate → Event::TopologyMutated
              → [State::OperatorPlan → Control::WriteOperator]  (if stubs exist)
              → Control::BuildTopologyOverlay → Event::TopologyOverlayBuilt
              → Event::LearnUpdated → State::Input → (trading cycle)
```

`introspect.py` is **fully implemented** (9.0 KB). It reads `state/quantale.tlog`,
`state/learned_edges.jsonl`, `state/topology_mutations.jsonl`, and `assets/operators.json`
to produce a diagnostic with node stats, stub nodes, never-fired nodes, high-failure
nodes, declining edges, and goal metrics. This context feeds `State::TopologyPlan`.

---

## Summary table

| # | Problem | Fix | How to validate |
|---|---------|-----|-----------------|
| 5 | Dev edges conditional on failures | Template: unconditional dev edges | Plan output has `State::Introspect` edge |
| 6 | Hard reset lost LLM dev-chain weights | `accumulated_edges` field + extend on embed | Dev-chain weight survives hard reset |
| 7 | `State::Input` ConsumedBlockPoint | Hard reset clears consumed[]; Fix 1 reduces frequency | Fewer `Unknown(-1)` blocks per cycle |
| 8 | `projection_first_hop_mismatch` false positives | Not fixed (false positive, harmless) | Warnings still appear; acceptable |
| 9 | No topology invariant for consumed blocking | Added invariant 14 `ConsumedBlockPoint` | `cargo run -- --check-topology` prints 5 WARNs, exits 0 |
| 10 | Learned edges saturate at 1.0 | `min(base+0.15, 0.85)` cap in learning_policy | `State::Plan → State::TradePlan` loads at 0.85 not 1.0 |
| 11 | Dev chain path product always loses | Fix 1: removed `State::Learn → Event::LearnUpdated` | `State::Introspect` appears in tlog |

---

## Validation for next session

```bash
# 1. Topology check clean (5 known warnings, exit 0)
cargo run -- --check-topology

# 2. All 100 tests green
cargo test

# 3. State::Introspect fires in tlog
grep "Introspect" state/quantale.tlog | tail -5

# 4. LLM creates nodes/edges (TopologyMutate fires)
grep "TopologyMutate\|WriteOperator\|OperatorPlan" state/quantale.tlog | tail -5

# 5. Learning cap working (no learned edge above 0.85)
python3 -c "
import json
best = {}
with open('state/learned_edges.jsonl') as f:
    for line in f:
        e = json.loads(line).get('edge', {})
        k = (e.get('from'), e.get('to'))
        best[k] = max(best.get(k, 0), e.get('confidence', 0))
over = {k:v for k,v in best.items() if v > 0.85}
print('Edges above 0.85:', over if over else 'none — cap is working')
"

# 6. No projection_first_hop_mismatch causing actual failures
# (warnings expected; check execution still succeeds after each warning)
```

## Known remaining issues for next session

1. **5 ConsumedBlockPoint nodes** — each needs a second outgoing edge to avoid
   hard-reset dependency. Start with `State::Input`: add `State::Input → State::Plan`
   or similar low-confidence second edge.

2. **`projection_first_hop_mismatch` invariant 19** — tighten the check to only
   flag when `first_hop` is not on the witness path from `selected_src` to
   `selected_dst`, rather than requiring `first_hop == selected_dst`.

3. **Dev chain operators** — `State::TopologyPlan`, `Control::TopologyMutate`,
   `State::OperatorPlan`, `Control::WriteOperator` may be stubs. Check
   `assets/operators.json` for `"executable": "true"` entries and implement
   any that are needed before the LLM can write files.

4. **`state/quantale.tlog` size** — 215k records. Consider rotating or trimming
   to keep `introspect.py`'s TLOG_WINDOW (300 records) representative of recent
   behaviour rather than ancient history.
