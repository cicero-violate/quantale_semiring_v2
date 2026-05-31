# Usage

## Requirements

- Rust
- CUDA with NVRTC support
- Compatible NVIDIA GPU

## Build

```bash
cargo check
cargo test
```

## Run benchmark

```bash
cargo run --bin bench_quantale -- 100
cargo run --release --bin bench_quantale -- 100
```

The benchmark measures:

```text
closure
projection
frontier_step
end_to_end_tick
```


## Run tensor benchmark

```bash
cargo run --bin bench_tensor_quantale -- 100
cargo run --release --bin bench_tensor_quantale -- 100
```

The tensor benchmark measures:

```text
tensor_closure
tensor_projection
tensor_decay
```

## Tensor engine

### Topology tensor schema

`assets/topology.json` uses explicit tensor values on every transition:

```json
{
  "from": "State::Plan",
  "to": "State::Optimize",
  "default_weight": 0.95,
  "confidence": 0.95,
  "cost": 0.05,
  "safety": 0.95
}
```

Create tensor edges directly:

```rust
use quantale_semiring_v2::{TensorEdge, TensorQuantaleWorld, ProjectionBias};

let edges = [
    TensorEdge::new(src, dst, 0.90, 2.0, 0.95),
];
let mut world = TensorQuantaleWorld::from_tensor_edges(&edges)?;
world.close()?;
let decision = world.project(ProjectionBias::default())?;
```

Layer semantics:

```text
confidence: max-times
cost:       min-plus
safety:     max-min
```

Projection score:

```text
score = α·confidence - β·cost + γ·safety
```

Feedback updates:

```rust
world.update_lattice_edge(src, dst, ExecutionOutcome::Success)?;
world.update_lattice_edge(src, dst, ExecutionOutcome::Failure)?;
world.update_lattice_edge(src, dst, ExecutionOutcome::Timeout)?;
world.update_lattice_edge(src, dst, ExecutionOutcome::SafetyViolation)?;
world.decay(0.99)?;
```

## Run orchestrator

```bash
cargo run
```

The default executable uses `TensorQuantaleWorld`. The runtime:

1. Creates a CUDA world.
2. Loads topology edges.
3. Seeds ingress candidates.
4. Computes closure and projection.
5. Executes the selected operator.
6. Converts process results into tensor receipt edges.
7. Reinjects feedback into the tensor graph.
8. Logs activity into quantale.tlog.

## Operators

Operators are configured in:

```text
assets/operators.json
```

Supported input modes:

```text
stdin_mode = json
stdin_source = field name
```

## Search evidence

External candidates are transformed into graph updates through:

```text
DomainCandidate
→ score_candidates
→ select_top_k
→ build_search_edges
→ M := M ∨ ΔM
```

The system does not implement retrieval or database search internally.

## Transaction log

Runtime records are written as JSONL to:

```text
quantale.tlog
```

Record types:

```text
Decision
CudaReport
Receipt
LatticeEdges
AgentStep
```

## Formal model

Lean specification:

```text
lean/QuantaleSemiringV2/Spec.lean
```
