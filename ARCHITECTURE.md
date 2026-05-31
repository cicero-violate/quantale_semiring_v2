# Architecture

`quantale_semiring_v2` is now a CUDA-first tensor quantale orchestrator.

## Core tensor

```text
T ∈ R^(3 × 44 × 44)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

## Node universe

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 1936
TENSOR_LEN  = 5808
```

## GPU ownership

CUDA owns:

```text
tensor[3 × 44 × 44]
scratch[3 × 44 × 44]
witness[3 × 44 × 44]
scratch_witness[3 × 44 × 44]
consumed[44 × 44]
active[44]
next_active[44]
decision[1]
```

Rust owns:

```text
operator execution
edge-delta upload
compact report decoding
transaction logging
```

Rust does not own a CPU planner or a CPU mirror of the tensor.

## Tensor runtime flow

```text
tensor topology edges
  ↓
TensorQuantaleWorld
  ↓
tensor_quantale_tick
  ↓
tensor_quantale_closure
  ↓
tensor_quantale_frontier_step
  ↓
operator execution
  ↓
ExecutionOutcome + ExecutionReceipt
  ↓
tensor feedback / tensor receipt edges
  ↓
T := T ∨ ΔT
```

## Projection

Projection blends the closed tensor:

```text
score = α·confidence - β·cost + γ·safety
```

The active frontier advances by selected first hop. The consumed mask prevents repeated first-hop execution from the same source.

## Data assets

```text
assets/topology.json      graph structure and tensor edge defaults
assets/rule_delta.json    policy/receipt tensor deltas
assets/operators.json     OS process operator contracts
```

## Legacy removals

Removed:

```text
src/policy.rs
src/receipt.rs
assets/policy.json
assets/receipt.json
```

Policy and receipt routing now live in `src/rule_delta.rs` and `assets/rule_delta.json`.
