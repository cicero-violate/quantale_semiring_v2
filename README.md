# quantale_semiring_v2

CUDA-first tensor quantale orchestrator with a data-driven Concurrent Kleene Algebra pattern layer.

The runtime compiles declarative JSON assets into tensor-edge deltas, runs tensor closure and projection on CUDA, dispatches selected operators, and feeds execution receipts back into the tensor as ordinary edge deltas.

## Runtime invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Canonical runtime path

```text
assets/topology.json
  → NodeRegistry (single source of truth for node names and IDs)
  → full_tensor_transition_edges()
assets/patterns.json
  → CKA pattern compiler
  → TensorEdge deltas
  → TlogWriter::append_tensor_edges("pattern:cka", ...)
  → TensorQuantaleWorld::from_tensor_edges(...)
  → tensor_quantale_closure
  → ExplorationEngine / assets/exploration.json
      → tensor_quantale_seed_exploration
      → tensor_quantale_expand_tokens
      → tensor_quantale_score_tokens
      → tensor_quantale_select_topk_tokens
      → tensor_quantale_commit_exploration
  → CKA fallback via project_ready_batch_plan(...)
      → tensor_quantale_project_batch
      → effect-safe DecisionBatch
      → tensor_quantale_commit_batch
      → host parallel operator dispatch
  → fallback tensor_quantale_frontier_step when no batch is ready
  → operator execution from assets/operators.json
  → ProcessReceipt
  → tensor edge feedback kernel
  → TensorQuantaleWorld::embed_tensor_edges
  → TensorQuantaleWorld::decay
```

`main.rs` uses the tensor path only. The scalar CUDA matrix runtime, scalar LLM plan format, CPU routing planner, DSL/search/ingress demo layer, paging registry, and side-channel policy files must not be reintroduced.

## Tensor model

Canonical tensor:

```text
T ∈ R^(3 × 44 × 44)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

Projection score:

```text
score = α·confidence - β·cost + γ·safety
```

The tensor is the execution substrate. It is not a sidecar.

## Data assets

```text
assets/topology.json      node registry and base tensor topology
assets/patterns.json      CKA structural patterns compiled to TensorEdge deltas
assets/operators.json     external operator contracts, output modes, and effect metadata
assets/exploration.json   token-value exploration policy, beam width, depth, scoring and anti-repeat weights
assets/call_llm.py        example tensor-plan operator
assets/memory.py          example memory operator
state/learned_edges.jsonl learned sparse tensor-edge checkpoints
```

Tensor edges use explicit tensor-native fields:

```json
{
  "from": "State::Plan",
  "to": "State::Optimize",
  "confidence": 0.95,
  "cost": 2.0,
  "safety": 0.90
}
```

The legacy scalar `weight` plan format is intentionally rejected by `compile_tensor_plan`.

## Node registry

The node universe is defined entirely in `assets/topology.json`. There are no hard-coded node constants in Rust.

`topology.rs::NodeRegistry` is the single source of truth:

```rust
registry.id_of("State::Execute")   // Option<usize>
registry.name_of(9)                 // Option<&str>
registry.action_of(17)              // Option<&str>  — from JSON "action" field
registry.len()                      // node count
registry.matrix_len()               // len * len
```

To add a node: edit `assets/topology.json`. No Rust code changes required.

Operator CUDA kernels (`cuda_ptx` backend) declare fused paths as normal operator entries in `assets/operators.json` and normal graph nodes in `assets/topology.json`. Fusion is expressed as a `choice` in `assets/patterns.json` with lower-cost topology weights — the quantale scheduler selects the fused path when cost favors it. No Rust `if fused` logic.

## CUDA kernel split

Two separate compilation paths — both use cudarc at runtime:

```text
cuda/quantale_world.cu
  → compiled at runtime via NVRTC (cudarc::nvrtc::compile_ptx)
  → kernels: closure, projection, exploration, frontier, tick, decay

cuda/trading_execution_kernels.cu
  → compiled at build time via nvcc (build.rs, --features cuda)
  → output: $OUT_DIR/trading_execution_kernels.ptx
  → loaded at runtime via cudarc::driver (CudaDevice::load_ptx)
  → kernels: fused_alpha_and_risk_kernel, fused_orderbook_and_alpha_kernel,
             fused_feed_alpha_and_risk_kernel
```

Operator kernels use nvcc because they require full optimization flags, specific GPU arch targeting, and cooperative-group / cub primitives that NVRTC does not support. The quantale world kernel uses NVRTC for fast iteration without a build step.

## Operator CUDA dispatch

When an operator declares `"executable": "cuda_ptx"` in `operators.json`, `egress.rs` routes to the CUDA PTX executor instead of spawning a process:

```json
{
  "node_name": "Execution::FusedAlphaAndRisk",
  "executable": "cuda_ptx",
  "input_mapping": {
    "module_name": "quantale_trading_execution_kernels",
    "kernel": "fused_alpha_and_risk_kernel",
    "scheduler_contract": "atomic_operator_fixed_budget"
  }
}
```

Without `--features cuda` the dispatcher returns an explicit capability error — never a process-spawn failure.

## Exploration anti-repeat policy

`assets/exploration.json` includes:

```json
{
  "repeat_penalty": 1.25,
  "max_terminal_visits": 1,
  "max_first_hop_visits": 1
}
```

CUDA top-k selection receives terminal and first-hop visit vectors from `ExplorationEngine`. Candidates at the configured visit limit are skipped; remaining repeated candidates are score-penalized before selection. The host reference selector applies the same policy.

## Concurrent Kleene Algebra

Implemented CKA operators:

```text
zero   blocked / impossible
one    identity / skip
node   atomic endpoint
seq    a ; b
choice a + b
star   bounded Kleene repetition
par    a || b
```

Main surfaces:

```text
src/pattern.rs       CKA JSON model, validation, and TensorEdge compiler
src/batch.rs         effect-safe DecisionBatch preparation and parallel dispatch
assets/patterns.json bundled CKA patterns
```

Compilation path:

```text
CkaExpr
  → validate_cka_expr(...)
  → CompiledCkaPattern { edges, parallel_groups }
  → TensorEdge deltas
  → TensorQuantaleWorld
```

Bounded `star` is finite unroll only. `par` requires effect independence before CUDA commit or host dispatch.

## Operator coverage

`assets/operators.json` contains an explicit contract for every topology node. Symbolic Control/Event nodes that do not need real side effects are covered by safe `true` no-op contracts with declared read/write metadata, so the executor no longer emits missing-contract receipts for normal symbolic traversal. Real operators keep their concrete contracts.

## Effect safety

Operator effects live in `assets/operators.json`:

```json
{
  "effects": {
    "reads": [],
    "writes": [],
    "locks": []
  }
}
```

Parallel branches must satisfy:

```text
safe_parallel(a,b) =
  writes(a) ∩ writes(b) = ∅
  ∧ writes(a) ∩ reads(b) = ∅
  ∧ reads(a) ∩ writes(b) = ∅
  ∧ locks(a) ∩ locks(b) = ∅
```

## CUDA kernels (quantale world)

The tensor runtime loads kernels from `cuda/quantale_world.cu` via NVRTC:

```text
tensor_quantale_reset
tensor_quantale_embed_edges
tensor_quantale_closure
tensor_quantale_project
tensor_quantale_project_batch
tensor_quantale_commit_batch
tensor_quantale_seed_exploration
tensor_quantale_expand_tokens
tensor_quantale_score_tokens
tensor_quantale_select_topk_tokens
tensor_quantale_commit_exploration
tensor_quantale_frontier_step
tensor_quantale_tick
tensor_quantale_update_edge
tensor_quantale_decay
```

`tensor_quantale_project_batch` projects candidate decisions for a CKA `par` group without mutating frontier state. After host-side effect validation and batch construction, `tensor_quantale_commit_batch` marks the selected first hops consumed and advances the active frontier to all committed batch nodes.

## Main Rust surfaces

```text
src/main.rs          runtime loop and scheduler integration
src/tensor.rs        CUDA tensor world, TensorEdge API, batch projection/commit API
src/topology.rs      topology.json parser, NodeRegistry (primary node/action lookup)
src/pattern.rs       CKA pattern compiler
src/batch.rs         DecisionBatch, BatchPlan, scheduler dispatch (backend-agnostic)
src/egress.rs        data-driven executor: process operators and cuda_ptx operators
src/config.rs        operator registry and runtime config (dimensions from registry)
src/projection.rs    DecisionReport and action_label (data-driven via registry)
src/transitions.rs   bundled tensor topology entrypoint
src/learning.rs      learned_edges.jsonl checkpoint loader
src/plan.rs          tensor LLM plan compiler
src/tlog.rs          append-only JSONL trace log
src/path.rs          tensor witness path reconstruction
src/node.rs          thin Node(i32) ID wrapper (names/actions owned by NodeRegistry)
src/exploration.rs   ExplorationEngine, token management, anti-repeat policy
```

## Execution loop

At each loop iteration:

1. Learned edge checkpoints from `state/learned_edges.jsonl` are embedded with topology and CKA edges at startup.
2. CUDA closes the tensor.
3. Exploration seeds strategies from `assets/exploration.json`.
4. CUDA expands, scores, top-k selects, and commits the best effect-safe exploration candidate that passes terminal and first-hop anti-repeat limits.
5. If exploration cannot commit, the scheduler attempts a CKA batch projection for compiled `par` groups.
6. If a full batch is runnable and effect-safe, CUDA commits the batch and host workers execute operators concurrently. All backends (`cuda_ptx`, process) go through the same dispatch path; routing by backend is `egress.rs`'s responsibility.
7. If no batch is ready, CUDA runs the normal single frontier step.
8. Process results become `ProcessReceipt` evidence.
9. Receipts update the selected tensor edge through the GPU feedback kernel and update exploration receipt priors; committed terminals and first hops are tracked to prevent repeated exploration dominance.
10. Tensor-plan stdout is parsed and embedded when an operator declares `output_mode = "tensor_plan"`.
11. Decisions, exploration records, batch plans, receipts, and edge deltas are logged to `state/quantale.tlog`.

## Validation

```bash
cargo fmt --check
cargo check
cargo test --no-default-features
cargo run --bin bench_tensor_quantale -- 3
cargo run --bin quantale_semiring_v2
```

With a CUDA build host:

```bash
cargo test --features cuda
```

Current validated test counts:

```text
cargo test --no-default-features   47 passed (6 suites)
```

Current debug benchmark sample:

```text
iterations=3
edge_count=45
tensor_closure avg_us≈220
tensor_projection avg_us≈26
tensor_decay avg_us≈12
```

Use release mode for baseline collection:

```bash
cargo run --release --bin bench_tensor_quantale -- 1000
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

Record hardware, CUDA version, Rust profile, iteration count, and per-kernel avg_us when capturing a new baseline.

## Non-goals

Do not add or reintroduce:

```text
scalar CUDA world
scalar LLM plan format
src/edge.rs
src/search.rs
src/ingress.rs
src/dsl.rs
src/paging.rs
scalar benchmark
CPU routing planner
side-channel policy files
PyTorch/JAX/Triton alternate runtime
hard-coded node ID constants in Rust (StateNode, ControlNode, EventNode, NODE_COUNT)
separate kernel_fusion crate or addons/ directory
runtime PTX stitching or FusionPlan
fake CUDA planned-success receipts
```

## Proof boundary

Lean/cLean artifacts live under `lean/`. They name the proof boundary for tensor closure, projection, frontier, tick, and batch projection/commit behavior. They are specification artifacts unless a local Lean/cLean toolchain is installed and run.
