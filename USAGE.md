# Usage

## Requirements

- Rust nightly compatible with the workspace
- CUDA with NVRTC support
- Compatible NVIDIA GPU

## Build and test

```bash
cargo fmt --check
cargo check
cargo test --lib
cargo test --test exploration
cargo test --test tensor_quantale
```

## Run orchestrator

```bash
cargo run --bin quantale_semiring_v2
```

The default executable uses `TensorQuantaleWorld`, loads topology, exploration, and CKA pattern assets, tries GPU exploration first, falls back to effect-safe CKA batch scheduling, then falls back to single frontier projection when no higher-level route is ready. Operators execute on the host; receipts update tensor deltas and exploration priors; all records are logged to `quantale.tlog`.

Expected runtime batch smoke lines:

```text
batch_step=5 projection=(Event::InputAccepted->State::Map) first_hop=State::Map
batch_step=5 projection=(Event::InputAccepted->State::Parse) first_hop=State::Parse
[BATCH] operator=State::Map exit=0 outcome=Success
[BATCH] operator=State::Parse exit=0 outcome=Success
```

All compact node IDs now have operator contracts in `assets/operators.json`. Symbolic Control/Event nodes use safe `true` no-op contracts unless a real operator is required.

## Run tensor benchmark

```bash
cargo run --bin bench_tensor_quantale -- 100
cargo run --release --bin bench_tensor_quantale -- 1000
```

The tensor benchmark measures:

```text
tensor_closure
tensor_projection
tensor_decay
```

## Release benchmark baseline

Captured with the already-built release binary:

```bash
../../target/release/bench_tensor_quantale 10
```

```text
profile=release
iterations=10
edge_count=45
tensor_closure     avg_us=217.630
tensor_projection  avg_us=33.412
tensor_decay       avg_us=12.782
```


## Tensor engine API

Create tensor edges directly:

```rust
use quantale_semiring_v2::{ProjectionBias, TensorEdge, TensorQuantaleWorld};

let edges = [TensorEdge::new(src, dst, 0.90, 2.0, 0.95)];
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

## Exploration policy

Exploration policy lives in:

```text
assets/exploration.json
```

It controls beam width, max depth, scoring weights, anti-repeat limits, and named strategies. The runtime maps each strategy start node into CUDA exploration tokens, expands token paths over closed tensor geometry, selects top-K candidates, validates effect safety, skips candidates whose terminal or first hop reached the configured visit limit, and commits the selected route.

Anti-repeat fields:

```json
{
  "repeat_penalty": 1.25,
  "max_terminal_visits": 1,
  "max_first_hop_visits": 1
}
```

Trace labels:

```text
ExplorationSeed
ExplorationTopK
ExplorationCommit
ExplorationReceipt
```

## CKA patterns

Patterns live in:

```text
assets/patterns.json
```

Example:

```json
{
  "name": "parallel_prepare",
  "expr": {
    "seq": [
      "Event::InputAccepted",
      {
        "par": [
          { "seq": ["State::Map", "State::Search"] },
          { "seq": ["State::Parse", "State::Score"] }
        ]
      }
    ]
  },
  "confidence": 0.99,
  "cost": 0.01,
  "safety": 0.99
}
```

Supported CKA expression forms:

```text
"zero"
"one"
"State::Plan"
{ "node": "State::Plan" }
{ "seq": [...] }
{ "choice": [...] }
{ "star": { "body": ..., "max_unroll": 3 } }
{ "par": [...] }
```

## Operator effects

`par` requires effect metadata in:

```text
assets/operators.json
```

Example:

```json
{
  "node_name": "State::Map",
  "executable": "true",
  "static_args": [],
  "input_mapping": { "stdin_source": null },
  "effects": {
    "reads": ["task.context"],
    "writes": ["map.candidates"],
    "locks": []
  }
}
```

Safety rule:

```text
writes(a) ∩ writes(b) = ∅
writes(a) ∩ reads(b) = ∅
reads(a) ∩ writes(b) = ∅
locks(a) ∩ locks(b) = ∅
```

## Tensor plan operators

Operators are configured in:

```text
assets/operators.json
```

A tensor-plan-producing operator must declare:

```json
{
  "node_name": "State::Plan",
  "output_mode": "tensor_plan"
}
```

When `output_mode` is `tensor_plan`, stdout must be a JSON array of tensor edges with explicit `confidence`, `cost`, and `safety` fields.

## Transaction log

Runtime records are written as JSONL to:

```text
quantale.tlog
```

Important labels:

```text
topology:tensor
pattern:cka
scheduler:cka_parallel
exploration:receipt
exploration:plan_tensor
egress:receipt
plan:tensor_llm
plan:tensor_batch
```

## Formal model

Lean specification boundary:

```text
lean/QuantaleSemiringV2/Spec.lean
```

The proof boundary should cover tensor closure/projection/frontier/tick and the batch projection/commit scheduler contract.
