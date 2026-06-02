# Problems Fixed — Session 2026-06-02

This document records what we believe we fixed. Validate each item by running
the system and observing the described expected behaviour.

---

## 1. `State::AnalysisPlan` crashes with `NameError: name '_EDGE_SCHEMA' is not defined`

**Symptom**
```
[STEP] operator=State::AnalysisPlan exit=1 outcome=Failure
stderr: NameError: name '_EDGE_SCHEMA' is not defined
```

**Root cause**
During a template-reconstruction step, `call_llm.py` was written starting at
`_BUILTIN_TEMPLATES = {`, losing the 73-line header above it — shebang, all
imports, module-level constants (`ASSET_DIR`, `_ROUTER_BASE`), and `_EDGE_SCHEMA`
itself. `_BUILTIN_TEMPLATES` references `_EDGE_SCHEMA` at import time, so any
`--template` invocation crashed immediately.

**Fix** (`b4d4263`)
Rebuilt the file by prepending the original header from `git show HEAD` before
the new templates block.

**Validation**
- `python3 crates/operators_lib/call_llm.py --template plan` with JSON stdin
  should return a valid JSON edge array without error.
- `cargo test` should pass (96 tests).

---

## 2. `cargo test` fails — `unresolved import quantale_semiring_v2::topology_check`

**Symptom**
```
error[E0432]: unresolved import `quantale_semiring_v2::topology_check`
 --> tests/topology_check.rs:1:63
```

**Root cause**
`topology_check.rs` was collapsed into `topology.rs` (module consolidation
earlier in the session). `tests/topology_check.rs` still imported
`topology_check::{self, ViolationKind}` as a module path and called
`topology_check::check(...)` at 5 call sites.

**Fix** (`0b73853`)
Updated `tests/topology_check.rs`: changed the import to
`use quantale_semiring_v2::{..., ViolationKind, check}` and replaced all
`topology_check::check(` call sites with `check(`.

**Validation**
- `cargo test` passes — all 96 tests green.

---

## 3. The autonomous development cycle never fires

The system has a development chain wired into the topology:
```
State::Learn → State::Introspect → State::TopologyPlan → Control::TopologyMutate
                                                        → State::OperatorPlan → Control::WriteOperator
```
Three independent blockers prevented it from ever running.

### 3a. Edge weights too low to win projection

**Symptom**
The agent loops through the trading cycle indefinitely, never routing through
`State::Introspect`, `State::TopologyPlan`, or `State::OperatorPlan`.

**Root cause**
`State::Learn → State::Introspect` had `confidence=0.15` while the competing
`State::Learn → Event::LearnUpdated` had `confidence=0.91`. The quantale world
projects the highest-scoring path — 0.15 vs 0.91 means the development branch
almost never wins.

**Fix** (`c43cf28`)
- `State::Learn → State::Introspect`: 0.15 → 0.35
- `Event::TopologyMutated → State::OperatorPlan`: 0.25 → 0.45
- `State::Learn → State::PatternPlan`: 0.12 → 0.25

**Validation**
After several thousand ticks, at least one of these should appear in
`state/quantale.tlog`:
- `AgentStep` with `"node": "State::Introspect"`
- `AgentStep` with `"node": "State::TopologyPlan"`
Check with: `grep Introspect state/quantale.tlog | tail -5`

### 3b. `plan` template never proposed development edges

**Symptom**
Even when `State::Introspect` is reachable, the LLM's plan proposals never
include `State::Learn → State::Introspect` edges, so the tensor world never
accumulates weight on that path.

**Root cause**
The `plan` template said "PRIMARY CYCLE — always prefer this market trading
chain" and gave only the trading cycle. The LLM followed instructions and
never proposed development edges.

**Fix** (`c43cf28`, `b7e58df`)
Added a `DEVELOPMENT CYCLE` section to the `plan` template:
> Propose `State::Learn -> State::Introspect` (confidence 0.3-0.5) when the
> context shows failures, stub nodes firing, or stagnant learning. At most
> one development edge per plan.

**Validation**
Inspect recent plan outputs in `state/quantale.tlog` for `TensorEdges` records
containing `State::Introspect` or `State::PatternPlan` as a destination.

### 3c. `write_operator.py` discarded `operator_contract_ops`

**Symptom**
After `State::OperatorPlan` writes a new `.py` file, the stub operator in
`operators.json` still has `executable: true`. The new file is never invoked
because the operator contract was not updated.

**Root cause**
The `operator_write` LLM template tells the LLM to include
`operator_contract_ops` in its output (to upgrade `executable: true → python3`).
`write_operator.py` unwrapped the payload, wrote the file, and exited — the
`operator_contract_ops` key was read but ignored.

**Fix** (`c43cf28`)
`write_operator.py` now calls `_apply_contract_ops()` after writing the file.
It reads `assets/operators.json`, applies each `update`/`replace` op, and
writes the file back. The receipt includes `contracts_updated: [...]`.

**Validation**
Manually trigger the chain:
```bash
echo '{
  "filename": "test_impl.py",
  "source": "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({\"test\":1}))\n",
  "node_name": "State::Goal",
  "operator_contract_ops": [{"op": "update", "node_name": "State::Goal",
    "patch": {"executable": "python3",
              "static_args": ["crates/operators_lib/test_impl.py"],
              "input_mapping": {"stdin_mode": "json"}}}]
}' | python3 crates/operators_lib/write_operator.py
```
Expected: `contracts_updated: ["State::Goal"]` in the response AND
`operators.json` entry for `State::Goal` should now have `"executable": "python3"`.
Clean up: `rm crates/operators_lib/test_impl.py` and restore `operators.json`.

---

## 4. `plan` template was hardcoded to the trading domain

**Symptom**
The `plan` template contained:
```
PRIMARY CYCLE — always prefer this market trading chain:
  State::Input -> State::MarketFeed -> Event::MarketFeedUpdated -> ...
CRITICAL RULES:
- Emit State::Input -> State::MarketFeed (confidence>=0.97). NEVER emit ...
- Emit State::Plan -> State::TradePlan (confidence>=0.94). Do NOT emit ...
```
These instructions named specific node names, preventing the agent from
working in any non-trading domain and making the plan brittle if the topology
was restructured.

**Fix** (`b7e58df`)
Removed all hardcoded node names and domain-specific rules. Replaced with:
> "Derive the primary cycle by reading the high-confidence transitions below.
> The backbone is the sequence of highest-confidence edges — follow it."

`load_transition_summary()` now sorts edges by confidence descending, so the
LLM sees the primary cycle at the top of the `{transitions}` list without
being told what it is.

**Validation**
- The plan template in `assets/call_llm_templates.json` should contain no
  references to `MarketFeed`, `TradePlan`, `BTC`, `ETH`, or `SOL`.
- `grep -i "marketfeed\|tradetplan\|BTC\|ETH\|SOL" assets/call_llm_templates.json`
  should return nothing from the `plan` or `repair` entries (only `analysis`
  and `trade` templates, which are intentionally trading-specific).
- `python3 -c "import sys; sys.path.insert(0,'crates/operators_lib'); import call_llm; lines = call_llm.load_transition_summary().split('\n'); print(lines[0])"` 
  should print a high-confidence edge (confidence ≥ 0.97).

---

## Summary table

| # | Problem | Fix commit | How to validate |
|---|---------|-----------|-----------------|
| 1 | `call_llm.py` header lost, `_EDGE_SCHEMA` undefined | `b4d4263` | `--template plan` works, 96 tests pass |
| 2 | `topology_check` module import in tests | `0b73853` | `cargo test` passes |
| 3a | Dev chain weights too low to win | `c43cf28` | `State::Introspect` appears in tlog |
| 3b | `plan` template never proposed dev edges | `c43cf28`, `b7e58df` | Dev edges appear in TensorEdges records |
| 3c | `operator_contract_ops` discarded | `c43cf28` | `contracts_updated` non-empty in receipt |
| 4 | `plan` template domain-locked to trading | `b7e58df` | No node names in plan template |
