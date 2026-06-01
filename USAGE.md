# Usage

## Requirements

- Rust nightly compatible with the workspace
- CUDA with NVRTC support (for the quantale world kernels — always required)
- CUDA feature support is enabled by default for `jit_cuda` operator execution
- Compatible NVIDIA GPU

## Build and test

```bash
cargo fmt --check
cargo check
cargo test
cargo check --no-default-features
cargo test --no-default-features
```

Use `--no-default-features` to exercise the explicit non-CUDA fallback path.

Current default and no-default-features test count: **101 passed** across 8 suites.

## Validate topology

```bash
cargo run --bin quantale_semiring_v2 -- --check-topology
```

Runs all static invariant checks (phases 1–3: identity, weight domain, gate dominance, receipt cutset, SCC progress, zero-cost cycles) on `assets/topology.json` and exits. Prints `topology OK (N nodes, M transitions)` on success; prints each violation and exits 1 on failure. Use this before committing topology changes.

## Run orchestrator

```bash
cargo run --bin quantale_semiring_v2
```

The default executable uses `TensorQuantaleWorld`, loads topology, learned edge checkpoints, exploration, and CKA pattern assets, tries GPU exploration first, falls back to effect-safe CKA batch scheduling, then falls back to single frontier projection when no higher-level route is ready. Operators execute on the host; receipts update tensor deltas and exploration priors; all records are logged to `state/quantale.tlog`.

Before each executor call, `runtime_check::decision_is_safe()` guards against score=⊥ with blocked=0 (invariant 20). After 3 consecutive blocked or unsafe steps, a hard reset restores the world via `reset() + embed_tensor_edges() + close()`.

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
{ "id": 60, "name": "Execution::FusedAlphaAndRisk", "type": "Execution" }
```

2. Add an operator contract to `assets/operators.json`:

```json
{
  "node_name": "Analysis::Return1",
  "executable": "jit_cuda",
  "effects": {
    "reads": ["market_feed"],
    "writes": ["return_1"],
    "locks": []
  }
}
```

3. Optionally add graph edges in `assets/topology.json` and a CKA pattern in `assets/patterns.json`.

When adding nodes beyond id 59 (current N=60), the CUDA kernel `quantale_world.cu` must also be updated:

```c
// update N and any per-category count constants
#define N 61
```

And `TENSOR_NODE_COUNT` in `src/tensor.rs` must be updated to match.

## Operator CUDA kernels

`jit_cuda` operator execution is compiled by default through the `cuda` feature:

```bash
cargo build
```

`jit_cuda` operators synthesize CUDA source for the selected operator chain,
compile it with NVRTC through cudarc, cache the loaded module, and keep
intermediate slots on device where possible.

With `--no-default-features`, operators declaring `"executable": "jit_cuda"`
return an explicit capability error receipt — no process-spawn attempt is made.

## Tensor engine API

Create tensor edges directly:

```rust
use quantale_semiring_v2::{ProjectionBias, TensorEdge, TensorQuantaleWorld};

let edges = [TensorEdge::new(src, dst, 0.90, 2.0, 0.95)];
// from_tensor_edges embeds edges and snapshots base_tensor for hard reset
let mut world = TensorQuantaleWorld::from_tensor_edges(&edges)?;
world.close()?;
let decision = world.project(ProjectionBias::default())?;
```

Runtime decision guard (call before every executor invocation):

```rust
use quantale_semiring_v2::runtime_check;

// Invariant 20: skip execution when score=⊥ with blocked=0
if !runtime_check::decision_is_safe(&decision) {
    // increment consecutive_blocks and continue
}

// Invariants 18/19/24: log any structural decision violations
for v in runtime_check::check_decision(&decision, node_name) {
    eprintln!("[runtime_check] {v}");
}
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
