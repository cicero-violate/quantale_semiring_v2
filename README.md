# quantale_semiring_v2

CUDA-first neuro-symbolic orchestrator built on a max-times quantale semiring.

The system represents workflow state, control flow, execution receipts, memory, learning, and search evidence as weighted graph edges inside a single GPU-resident matrix.

## Core model

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
Q = ([0,1], max, ×, 0, 1)
M ∈ Q^(N × N)
```

Current node universe:

```text
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 1936
```

## Runtime architecture

```text
CUDA closure engine
CUDA decision projection engine
Witness-matrix path reconstruction
Data-driven topology loader
Rule-delta compiler
Receipt routing system
Search-evidence compiler
Operator execution runtime
JSONL transaction logging
Lean algebra specification
```

## Major public surfaces

```text
embed_elements()
join_empirical_element()
join_policy_edges()
join_receipt_edges()
join_search_candidates()
star_assign()
frontier_step()
tick()
project_decision_path()
```

## Data assets

```text
assets/topology.json
assets/rule_delta.json
assets/operators.json
```

Topology, receipt routing, policy routing, and operator contracts are data-driven.

## Decision model

```text
A*      = quantale closure
π(A*)   = executable projection
W       = witness matrix
```

Projection returns:

```text
selected_src
selected_dst
first_hop
selected_value
halted
blocked
```

Paths are reconstructed from the witness matrix.

## Execution feedback

```text
Projection
→ operator execution
→ ProcessReceipt
→ ExecutionReceipt
→ receipt edges
→ M := M ∨ ΔM
```


## Tensor quantale engine

The crate now includes a first-class three-layer tensor engine:

```text
T ∈ R^(3 × 44 × 44)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

Primary tensor API:

```text
TensorEdge
ProjectionBias
ExecutionOutcome
TensorQuantaleWorld
TensorQuantaleWorld::from_tensor_edges()
TensorQuantaleWorld::close()
TensorQuantaleWorld::project()
TensorQuantaleWorld::update_lattice_edge()
TensorQuantaleWorld::decay()
```

Tensor CUDA kernels:

```text
tensor_quantale_reset
tensor_quantale_embed_edges
tensor_quantale_closure
tensor_quantale_project
tensor_quantale_update_edge
tensor_quantale_decay
```

Projection blends the closed tensor using:

```text
score = α·confidence - β·cost + γ·safety
```

The tensor implementation is not a sidecar. Tensor state, tensor closure, tensor witnesses, blended projection, feedback updates, and decay are implemented as CUDA-resident buffers and kernels.

## Validation

Current source state:

```text
cargo fmt --check ✓
cargo check       ✓
cargo test        ✓
bench_quantale        ✓
bench_tensor_quantale ✓
39 tests passing
```

## CUDA kernels

```text
quantale_reset
quantale_embed_elements
quantale_supremum_assign
quantale_tensor_assign
quantale_least_fixed_point
quantale_step
quantale_tick
quantale_morphism
quantale_frontier_step
```
