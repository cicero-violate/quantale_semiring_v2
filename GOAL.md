# Goal
Data defines execution; CPU only launches kernels, never decides orchestration. GPU runs the show.

Build a CUDA-resident tensor quantale orchestrator where workflow routing is algebraic, data-driven, and receipt-grounded.

## Canonical model

```text
T ∈ R^(3 × N × N)
```

Layers:

```text
confidence/correctness: max-times
compute/time cost:      min-plus
security/safety:        max-min
```

## Structural layer

```text
CKA = { 0, 1, +, ;, *, || }
```

CKA patterns describe executable structure. They compile into `TensorEdge` deltas and safe parallel-group hints. They do not replace the tensor runtime.

## Runtime invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Non-negotiables

- Tensor state remains GPU-resident.
- Tensor edges carry confidence, cost, and safety directly.
- CKA remains data-driven through JSON assets.
- Exploration remains data-driven through `assets/exploration.json`.
- `par` requires effect independence.
- CUDA batch projection is read-only.
- CUDA batch commit occurs only after whole-group validation.
- Runtime feedback updates tensor cells directly.
- Receipts remain the canonical truth gate.
- No scalar sidecar metadata model.
- No CPU routing planner.
- No hidden imperative graph traversal.

## Current milestone

Implemented:

```text
Tensor engine (60-node topology)
Tensor topology compilation
Tensor rule deltas
Tensor frontier step
Tensor tick
Tensor runtime loop
Full CKA pattern compiler
Effect-gated par validation
CUDA batch projection
CUDA batch commit
Host parallel scheduler dispatch
Append-only batch trace logging
CUDA exploration seed/expand/score/top-k/commit kernels
Exploration-first scheduler integration
Receipt-prior feedback into exploration
Runtime smoke-tested batch execution
Static topology invariant checker — phases 1–3 (topology_check.rs)
  identity, weight domain, operator binding, gate dominance,
  receipt cutset, SCC progress, zero-cost cycle detection
Semiring law unit tests — phase 4 (tests/semiring_laws.rs)
Runtime frontier assertions — phase 5 (tensor.rs, main.rs)
Runtime decision invariant checker — phase 6 (runtime_check.rs)
  decision_is_safe() guard (inv 20), check_decision() diagnostics (inv 18/19/24)
  base_tensor CPU snapshot for hard-reset groundwork (inv 23)
  action/output_mode operator field check (inv 25)
```

Validated test count: **99 passed** (8 suites, `--no-default-features`).

Validated runtime smoke includes concurrent dispatch for:

```text
Event::InputAccepted → State::Map
Event::InputAccepted → State::Parse
```
