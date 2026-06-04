# Plan: Real Operators and Critical Kernels

## What this plan does not repeat

The data-driven node registry, topology validation, egress routing, and the
three-layer tensor kernel split are already done. This plan only covers what
is still stubbed out or missing.

---

## Current state

```text
Real today                   | Still stub / missing
-----------------------------|--------------------------------------------
State::Plan  → call_llm.py   | State::Learn   → true (no-op)
Control::Repair → call_llm.py| State::Input   → true (no-op)
State::Memory → memory.py    | State::Map     → true (no-op)
State::Validate → cargo test | State::Search  → true (no-op)
Control::GateExecution →patch| State::Score   → true (no-op)
All Control/Event → true     | State::Select  → true (no-op)
                             | State::Execute → true (no-op)
                             | All Event nodes → true (no-op)
                             | action="" on all topology nodes
                             | Execution:: cuda_ptx kernels → stubs
```

Both Python scripts (`call_llm.py`, `memory.py`) are now data-driven:
- `call_llm.py` loads valid node names from `assets/topology.json` at runtime.
  Templates load from `assets/call_llm_templates.json` if it exists; fall back
  to built-ins. All endpoints are env vars.
- `memory.py` reads file path, window size, and context key from env vars.
- `operators.json` stale `addons/kernel_fusion/` paths fixed to
  `cuda/trading_execution_kernels.ptx`.

---

## 1. Add `action` fields to `assets/topology.json`

`action_label()` returns `"unknown"` for every node. Fix by adding an `"action"`
field to nodes that have semantic meaning in the main loop:

```json
{ "id": 9,  "name": "State::Execute",   "type": "State",   "action": "execute" },
{ "id": 17, "name": "Control::Commit",  "type": "Control", "action": "commit"  },
{ "id": 15, "name": "Control::Retry",   "type": "Control", "action": "retry"   },
{ "id": 16, "name": "Control::Repair",  "type": "Control", "action": "repair"  },
{ "id": 18, "name": "Control::Rollback","type": "Control", "action": "rollback"},
{ "id": 19, "name": "Control::Halt",    "type": "Control", "action": "halt"    }
```

Leaf state nodes (`State::Plan`, `State::Memory`, etc.) can omit `action` —
`"unknown"` is fine for display-only.

---

## 2. `assets/learn.py` — `State::Learn` operator

`State::Learn` is the only state node that is genuinely a stub affecting the
tensor. It should read the recent execution log and emit tensor edge deltas
that reinforce successful paths and weaken failed ones.

Inputs (stdin JSON):
```json
{
  "context": "...",
  "memory": [{"ts": "...", "context": "..."}]
}
```

Behaviour:
1. Read `state/memory.jsonl` tail (same window as `memory.py`).
2. Read the most recent `state/quantale.tlog` entries via a configurable tail
   (`QUANTALE_LEARN_LOG`, default `state/quantale.tlog`; `QUANTALE_LEARN_WINDOW`,
   default `50`).
3. For each receipt in the tail:
   - `exit_code == 0` → emit a strengthening edge (`confidence += δ_up`,
     `cost *= shrink`, `safety += δ_up`).
   - `exit_code != 0` → emit a weakening edge (`confidence *= 0.9`,
     `cost += δ_penalty`, `safety *= 0.9`).
4. Output a JSON tensor-edge array — same format as `call_llm.py` so
   `output_mode: tensor_plan` ingests it.

Register in `operators.json`:
```json
{
  "node_name": "State::Learn",
  "executable": "python3",
  "static_args": ["assets/learn.py"],
  "input_mapping": { "stdin_mode": "json" },
  "output_mode": "tensor_plan",
  "effects": {
    "reads":  ["memory.store", "state/quantale.tlog"],
    "writes": ["learning.tensor"],
    "locks":  ["learning"]
  }
}
```

Environment variables:
```text
QUANTALE_LEARN_LOG      path to tlog (default state/quantale.tlog)
QUANTALE_LEARN_WINDOW   receipt entries to look back (default 50)
QUANTALE_LEARN_DELTA_UP delta applied on success (default 0.05)
QUANTALE_LEARN_PENALTY  cost delta on failure (default 2.0)
```

---

## 3. `assets/input.py` — `State::Input` operator

Currently `State::Input` is `true`. It should accept a task from the operator
or environment and emit it as context.

Sources (checked in order):
1. `stdin` field `"task"` if present.
2. Env var `QUANTALE_TASK`.
3. File at `QUANTALE_TASK_FILE` (default `state/task.txt`) if it exists.
4. Empty string if nothing found (silent pass-through).

Output (stdout JSON, no `output_mode` — just context forwarding):
```json
{ "context": "<task string>" }
```

Register in `operators.json`:
```json
{
  "node_name": "State::Input",
  "executable": "python3",
  "static_args": ["assets/input.py"],
  "input_mapping": { "stdin_mode": "json" },
  "effects": {
    "reads":  ["env.task"],
    "writes": ["task.context"],
    "locks":  []
  }
}
```

---

## 4. `assets/event.py` — shared event acknowledgement script

All 18 Event nodes run `true` today. Most are purely symbolic and `true` is
correct. But a few need to emit an observation or log entry:

- `Event::FactArrived` — log the fact to `state/facts.jsonl`
- `Event::InputAccepted` — echo back the accepted input as context
- `Event::ParseOk` / `Event::ParseErr` — log parse outcome
- `Event::ReceiptAccepted` / `Event::ReceiptRejected` — log receipt result
- `Event::MemoryWritten` — confirm memory was committed
- `Event::LearnUpdated` — confirm learning step completed

A single `assets/event.py` can handle all of them with a `--event` argument:

```bash
python3 assets/event.py --event FactArrived
python3 assets/event.py --event MemoryWritten
```

Behaviour:
1. Read stdin JSON payload.
2. Append a timestamped record to `state/events.jsonl`
   (path from `QUANTALE_EVENT_LOG`, default `state/events.jsonl`).
3. Echo the payload back as stdout (allows context forwarding).
4. Exit 0.

Register each event node in `operators.json`:
```json
{
  "node_name": "Event::FactArrived",
  "executable": "python3",
  "static_args": ["assets/event.py", "--event", "FactArrived"],
  "input_mapping": { "stdin_mode": "json" },
  "effects": {
    "reads":  ["event.symbolic"],
    "writes": ["event.factarrived.observed"],
    "locks":  []
  }
}
```

Replace only the semantically meaningful events listed above. The rest stay
`true`.

---

## 5. CUDA execution kernels — real implementations

`cuda/trading_execution_kernels.cu` has three stub kernels. Replace the
stub arithmetic with real logic:

### `fused_alpha_and_risk_kernel`

Fuses DynamicAlphaSignalEvaluator + PortfolioRiskConstraintFilter.

```c
// alpha  = tanh(feed) * signal
// risk   = 1 - |position|
// result = alpha * risk, clamped to [-1, 1]
results[idx] = fmaxf(-1.0f, fminf(1.0f,
    tanhf(market_feed[idx]) * trading_signals[idx]
    * (1.0f - fabsf(portfolio_state[idx]))));
```

### `fused_orderbook_and_alpha_kernel`

Fuses OrderbookImbalanceWeaver + AlphaSignalEvaluator.

```c
// imbalance-weighted alpha with volume normalization
float imb = tanhf(orderbook[idx]);
results[idx] = alpha_signals[idx] * (1.0f + 0.5f * imb)
               / fmaxf(1.0f, fabsf(alpha_signals[idx]));
```

### `fused_feed_alpha_and_risk_kernel`

Three-pass fused kernel.

```c
float norm  = tanhf(feed[idx]);
float alpha = alpha_signals[idx] * norm;
float risk  = 1.0f - fabsf(portfolio_state[idx]);
results[idx] = fmaxf(-1.0f, fminf(1.0f, alpha * risk));
```

Update the function signatures to match — these take 3 input arrays and 1
output. The existing signatures in `trading_execution_kernels.cu` already
match (a, b, c, results, n); map:

```text
fused_alpha_and_risk:       a=market_feed  b=portfolio_state  c=trading_signals
fused_orderbook_and_alpha:  a=orderbook    b=alpha_signals    c=(unused, zeros)
fused_feed_alpha_and_risk:  a=feed         b=alpha_signals    c=portfolio_state
```

---

## 6. `assets/call_llm_templates.json` — external template file

`call_llm.py` now loads templates from this file if it exists. Create it to
let templates be edited without touching Python:

```json
{
  "plan":   "You are a neuro-symbolic planning engine...\n\nContext:\n{context}\n\nPropose transitions:",
  "repair": "You are a repair subsystem...\n\nFailure context:\n{context}\n\nPropose recovery:"
}
```

The `_EDGE_SCHEMA` block (with node validation and format rules) is always
appended by `call_llm.py` after loading — templates only need the preamble.

---

## 7. Completion criteria

```text
[ ] topology.json has action fields for Halt, Commit, Retry, Repair, Rollback, Execute
[ ] action="unknown" no longer appears in main loop output for those nodes
[ ] assets/learn.py exists, reads tlog, emits tensor edges, exits 0 on cold log
[ ] State::Learn operator in operators.json points to learn.py with output_mode=tensor_plan
[ ] assets/input.py exists, reads QUANTALE_TASK or state/task.txt, exits 0 on empty
[ ] State::Input operator in operators.json points to input.py
[ ] assets/event.py exists, logs to state/events.jsonl, echoes payload
[ ] FactArrived, InputAccepted, ParseOk, ParseErr, ReceiptAccepted, ReceiptRejected,
    MemoryWritten, LearnUpdated in operators.json point to event.py
[ ] call_llm.py loads node names from topology.json (no hardcoded VALID_NODES)
[ ] memory.py reads MEMORY_FILE and WINDOW from env vars
[ ] operators.json has no addons/ paths
[ ] trading_execution_kernels.cu has real arithmetic in all three fused kernels
[ ] cargo test --no-default-features passes (53 tests)
```
