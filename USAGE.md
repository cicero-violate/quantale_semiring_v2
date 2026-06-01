# Usage

## Requirements

- Rust nightly compatible with the workspace
- CUDA with NVRTC support (for the quantale world kernels — always required)
- nvcc (for `--features cuda` operator kernel compilation — optional)
- Compatible NVIDIA GPU

## Build and test

```bash
cargo fmt --check
cargo check --no-default-features
cargo test --no-default-features
```

With a CUDA build host (compiles `cuda/trading_execution_kernels.cu` via nvcc):

```bash
cargo test --features cuda
```

Current no-default-features test count: **47 passed** across 6 suites.

## Run orchestrator

```bash
cargo run --bin quantale_semiring_v2
```

The default executable uses `TensorQuantaleWorld`, loads topology, learned edge checkpoints, exploration, and CKA pattern assets, tries GPU exploration first, falls back to effect-safe CKA batch scheduling, then falls back to single frontier projection when no higher-level route is ready. Operators execute on the host; receipts update tensor deltas and exploration priors; all records are logged to `state/quantale.tlog`.

Expected runtime batch smoke lines:

```text
batch_step=5 projection=(Event::InputAccepted->State::Map) first_hop=State::Map
batch_step=5 projection=(Event::InputAccepted->State::Parse) first_hop=State::Parse
[BATCH] operator=State::Map exit=0 outcome=Success
[BATCH] operator=State::Parse exit=0 outcome=Success
```

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

```text
profile=release
iterations=10
edge_count=45
tensor_closure     avg_us=217.630
tensor_projection  avg_us=33.412
tensor_decay       avg_us=12.782
```

## Adding nodes

The node universe lives entirely in `assets/topology.json`. No Rust code changes are required to add a new node.

1. Add an entry to `assets/topology.json`:

```json
{ "id": 44, "name": "Execution::FusedAlphaAndRisk", "type": "Execution" }
```

2. Add an operator contract to `assets/operators.json`:

```json
{
  "node_name": "Execution::FusedAlphaAndRisk",
  "executable": "cuda_ptx",
  "static_args": [],
  "input_mapping": {
    "module_name": "quantale_trading_execution_kernels",
    "kernel": "fused_alpha_and_risk_kernel",
    "scheduler_contract": "atomic_operator_fixed_budget"
  },
  "effects": {
    "reads": ["market.feed", "portfolio.state"],
    "writes": ["execution.gpu.results"],
    "locks": []
  }
}
```

3. Optionally add graph edges in `assets/topology.json` and a CKA pattern in `assets/patterns.json`.

When adding nodes beyond id 43 (current N=44), the CUDA kernel `quantale_world.cu` must also be updated:

```c
#define STATE_NODE_COUNT   13
#define CONTROL_NODE_COUNT 13
#define EVENT_NODE_COUNT   18
// add new category counts here and update N
```

And `TENSOR_NODE_COUNT` in `src/tensor.rs` must be updated to match.

## Operator CUDA kernels (`--features cuda`)

Operator kernels in `cuda/trading_execution_kernels.cu` are compiled by `build.rs` when `--features cuda` is active:

```bash
cargo build --features cuda
```

`build.rs` invokes nvcc:

```bash
nvcc cuda/trading_execution_kernels.cu -ptx -o $OUT_DIR/trading_execution_kernels.ptx \
     -std=c++17 --use_fast_math -Xcompiler -fPIC
```

The PTX is embedded at compile time via `include_bytes!(concat!(env!("OUT_DIR"), "/trading_execution_kernels.ptx"))` and loaded at runtime via cudarc. nvcc is found from `CUDA_HOME/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, or PATH in that order.

Without `--features cuda`, any operator declaring `"executable": "cuda_ptx"` returns an explicit error receipt — no process-spawn attempt is made.

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

## Node registry API

```rust
use quantale_semiring_v2::GraphTopology;

let registry = GraphTopology::bundled_registry()?;
let id   = registry.id_of("State::Execute").unwrap();   // usize
let name = registry.name_of(9).unwrap();                 // &str
let act  = registry.action_of(17);                       // Option<&str>
let n    = registry.len();                               // total node count
```

Node IDs are stable integers. Human names and action labels come from the registry, not from Rust constants.

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

Node names in patterns are validated against the bundled `NodeRegistry` at compile time.

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
state/quantale.tlog
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
