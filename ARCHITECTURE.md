# Architecture

`quantale_semiring_v2` is a CUDA-first tensor quantale orchestrator with a data-driven CKA structural compiler and effect-safe parallel scheduler.

## Core invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Tensor substrate

```text
T ∈ R^(3 × 44 × 44)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

Node universe:

```text
Current topology: 44 nodes (State × 13, Control × 13, Event × 18)
Source of truth:  assets/topology.json
Rust constant:    TENSOR_NODE_COUNT = 44 (must match N in quantale_world.cu)
MATRIX_LEN        = 1936
TENSOR_LEN        = 5808
```

No Rust code encodes the node list. `topology.rs::NodeRegistry` loads it from JSON at startup. Adding a node requires only a JSON edit (and updating `N` in the CUDA kernel and `TENSOR_NODE_COUNT` in `tensor.rs` if the count changes).

## Data-driven node registry

`topology.rs::NodeRegistry` is the primary registry. Every component that needs a node ID or name goes through it:

```rust
registry.id_of("State::Execute")   // → Option<usize>
registry.name_of(9)                 // → Option<&str>
registry.action_of(17)              // → Option<&str> from JSON "action" field
registry.len()                      // → node count
registry.matrix_len()               // → len * len
```

`SystemConfig`, `ExplorationEngine`, `batch.rs`, and `egress.rs` all derive dimensions from the registry. No component holds a hard-coded `NODE_COUNT` constant. `Node(i32)` is a thin ID wrapper; the registry owns the names.

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
exploration_tokens[44 × 44]
exploration_scores[44 × 44]
exploration_parents[44 × 44]
exploration_selected[44]
```

Rust owns:

```text
JSON asset loading and NodeRegistry
CKA pattern compilation
operator effect validation
batch scheduling policy
host-side operator execution (process and cuda_ptx backends)
edge-delta upload
compact report decoding
append-only transaction logging
```

Rust does not own a CPU planner or a CPU mirror of the tensor.

## CUDA kernel split

Two distinct compilation and loading paths — both dispatch at runtime through cudarc:

```text
cuda/quantale_world.cu
  Compiled: NVRTC at runtime (cudarc::nvrtc::compile_ptx)
  Kernels:  tensor_quantale_reset, embed_edges, closure, project,
            project_batch, commit_batch, frontier_step, tick,
            update_edge, decay, seed_exploration, expand_tokens,
            score_tokens, select_topk_tokens, commit_exploration

cuda/trading_execution_kernels.cu
  Compiled: nvcc at build time (build.rs, --features cuda)
  Output:   $OUT_DIR/trading_execution_kernels.ptx  (embedded via include_bytes!)
  Loaded:   CudaDevice::load_ptx at runtime
  Kernels:  fused_alpha_and_risk_kernel
            fused_orderbook_and_alpha_kernel
            fused_feed_alpha_and_risk_kernel
```

`quantale_world.cu` uses NVRTC for rapid iteration without a build step. Operator kernels use nvcc because they need full `-O3`, specific GPU arch flags, and CUDA library access (cooperative groups, cub) that NVRTC does not expose.

## Data-flow architecture

```text
assets/topology.json
  → NodeRegistry
  → TensorEdge[]

assets/patterns.json
  → CkaExpr
  → CompiledCkaPattern { edges, parallel_groups }
  → TensorEdge[]

TensorEdge[]
  → TensorQuantaleWorld
  → tensor_quantale_closure
  → exploration scheduler attempt
      → tensor_quantale_seed_exploration
      → tensor_quantale_expand_tokens
      → tensor_quantale_score_tokens
      → tensor_quantale_select_topk_tokens
      → tensor_quantale_commit_exploration
  → CKA scheduler fallback
      → tensor_quantale_project_batch
      → validate_parallel_group_effects
      → prepare_parallel_batch_plan
      → tensor_quantale_commit_batch
      → dispatch_decision_batch_blocking (all backends: process + cuda_ptx)
  → single-step fallback
      → tensor_quantale_frontier_step
  → ProcessReceipt
  → tensor_quantale_update_edge
  → learned TensorEdge checkpoints
  → T := T ∨ ΔT
```

## CKA layer

CKA is an edge-delta compiler above the tensor substrate, not a replacement runtime.

```text
CKA = { 0, 1, +, ;, *, || }
```

Rust model:

```text
CkaExpr::Zero
CkaExpr::One
CkaExpr::Node(String)
CkaExpr::Seq(Vec<CkaExpr>)
CkaExpr::Choice(Vec<CkaExpr>)
CkaExpr::Star { body, max_unroll }
CkaExpr::Par(Vec<CkaExpr>)
```

Node names in `CkaExpr::Node` are validated against `NodeRegistry` at compile time. The compiler never uses hard-coded IDs.

Semantics:

```text
Zero   produces no executable edges
One    identity/skip
Node   atomic endpoint
Seq    compiles adjacent endpoints
Choice compiles alternatives without false sequencing
Star   bounded finite unroll only
Par    compiles branches and records effect-safe parallel groups
```

## Exploration layer

Exploration is the dynamic strategy-selection layer above closed tensor geometry and below host execution.

```text
assets/exploration.json
  → ExplorationConfig
  → ExplorationEngine (node_count from NodeRegistry)
  → TensorQuantaleWorld::expand_exploration
  → ExplorationCandidate top-K
  → effect validation
  → TensorQuantaleWorld::commit_exploration_candidate
```

Scoring:

```text
V(H) = confidence - cost + safety + η·novelty + ρ·receipt_prior - λ·entropy
```

Receipt feedback:

```text
success → raises node receipt_prior
failure/timeout/safety violation → lowers node receipt_prior
```

Backtracking is preserved through `ExplorationToken.parent` and surfaced by `commit_record(...).path`.

Anti-repeat state is host-owned and uploaded into CUDA top-k selection each tick:

```text
terminal_visits[N]
first_hop_visits[N]
repeat_penalty
max_terminal_visits
max_first_hop_visits
```

Vector lengths are `registry.len()` — not a compile-time constant.

## Parallel scheduler

The scheduler is deliberately split into projection, validation, commit, and dispatch:

```text
project_ready_batch_plan(...)
  → TensorQuantaleWorld::project_parallel_group(...)
  → tensor_quantale_project_batch
  → validate_parallel_group_effects(...)
  → prepare_parallel_batch_plan(...)
  → TensorQuantaleWorld::commit_decision_batch(...)
  → tensor_quantale_commit_batch
  → dispatch_decision_batch_blocking(...)
```

This prevents partial mutation. CUDA projection is read-only. CUDA commit only occurs after the whole group is runnable and effect-safe. All backends go through a single uniform dispatch path; the `egress.rs` executor routes by `executable` field, not by batch-layer branching.

## Operator dispatch

`egress.rs::execute_abstract_node_blocking` routes by the `executable` field in `operators.json`:

```text
executable = "cuda_ptx"   → execute_cuda_ptx_blocking (cudarc, --features cuda)
executable = anything else → Command::new(binary)
```

`cuda_ptx` dispatch reads `input_mapping.module_name` and `input_mapping.kernel` from JSON to select the function from the preloaded PTX module. Without `--features cuda` it returns an explicit error receipt — no process-spawn attempt.

## Projection

`projection.rs` exposes `DecisionReport` and a data-driven action label:

```rust
pub fn action_label(node_id: i32, registry: &NodeRegistry) -> &str
```

`action_label` reads the `"action"` field from `topology.json` (via the registry) — no hard-coded node comparisons. There is no `QuantaleAction` enum. The score blend is still:

```text
score = α·confidence - β·cost + γ·safety
```

The active frontier advances by selected first hop. The consumed mask prevents repeated first-hop execution from the same source.

## Operator coverage

Every topology node has an operator contract. Symbolic Control/Event nodes default to explicit safe no-op contracts:

```text
executable = true
locks = []
reads/writes = symbolic metadata only
```

This keeps normal symbolic traversal from generating missing-contract process receipts while preserving real contracts for executable stateful nodes.

## Effect safety contract

```text
safe_parallel(a,b) =
  writes(a) ∩ writes(b) = ∅
  ∧ writes(a) ∩ reads(b) = ∅
  ∧ reads(a) ∩ writes(b) = ∅
  ∧ locks(a) ∩ locks(b) = ∅
```

Effects are data-driven through `assets/operators.json`:

```text
reads[]
writes[]
locks[]
```

If independence cannot be proved from metadata, `par` validation fails.

## CUDA kernels (quantale world)

Tensor kernels (NVRTC, `cuda/quantale_world.cu`):

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

Batch kernels:

```text
tensor_quantale_project_batch:
  read tensor/witness/active/consumed
  emit one DecisionReport per requested par-group node
  do not mutate active or consumed

tensor_quantale_commit_batch:
  validate non-blocked decisions on host first
  mark consumed[src, first_hop]
  set active frontier to all committed first_hop nodes
```

## Trace logging

`tlog.rs` records append-only JSONL events:

```text
Decision
Receipt
TensorEdges
AgentStep
ExplorationSeed
ExplorationExpand
ExplorationTopK
ExplorationCommit
ExplorationReceipt
BatchPlan via append_batch_plan(...)
```

Compiled CKA edges are logged with:

```text
label = "pattern:cka"
```

Runtime batch plans are logged with:

```text
label = "scheduler:cka_parallel"
```

## Legacy removals

Removed or forbidden from runtime reintroduction:

```text
scalar CUDA world
scalar LLM plan format
CPU routing planner
policy side-channel files
search/ingress demo planner
DSL compiler
paging registry
PyTorch/JAX/Triton runtime
hard-coded StateNode/ControlNode/EventNode constants
NODE_COUNT / MATRIX_LEN / THREAD_COUNT in src/node.rs
separate kernel_fusion crate or addons/ directory
runtime PTX stitching or FusionPlan
fake CUDA planned-success receipts
QuantaleAction enum / selected_action()
batch_contains_cuda_ptx / CUDA-specific batch branching
```

## Benchmark baseline

```text
profile=release
iterations=10
edge_count=45
tensor_closure     avg_us=217.630
tensor_projection  avg_us=33.412
tensor_decay       avg_us=12.782
```
