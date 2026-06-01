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
  → ExecutionReceipt
  → build_tensor_receipt_edges from assets/rule_delta.json
  → TensorQuantaleWorld::embed_tensor_edges
  → TensorQuantaleWorld::decay
```

`main.rs` uses the tensor path only. The scalar CUDA matrix runtime, scalar LLM plan format, CPU routing planner, DSL/search/ingress demo layer, paging registry, and side-channel policy/receipt files must not be reintroduced.

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
assets/rule_delta.json    receipt/policy rule deltas compiled to TensorEdge values
assets/call_llm.py        example tensor-plan operator
assets/memory.py          example memory operator
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
src/pattern.rs      CKA JSON model, validation, and TensorEdge compiler
src/batch.rs        effect-safe DecisionBatch preparation and parallel dispatch
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

`assets/operators.json` now contains an explicit contract for every compact node ID. Symbolic Control/Event nodes that do not need real side effects are covered by safe `true` no-op contracts with declared read/write metadata, so the executor no longer emits missing-contract receipts for normal symbolic traversal. Real operators keep their concrete contracts.

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

## CUDA kernels

The tensor runtime loads kernels from `cuda/quantale_world.cu`:

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
src/pattern.rs       CKA pattern compiler
src/batch.rs         DecisionBatch, BatchPlan, scheduler dispatch
src/topology.rs      topology.json parser/compiler
src/transitions.rs   bundled tensor topology entrypoint
src/rule_delta.rs    rule_delta.json receipt/policy compiler
src/plan.rs          tensor LLM plan compiler
src/egress.rs        data-driven OS process executor
src/config.rs        operator registry and runtime config
src/projection.rs    DecisionReport and action mapping
src/tlog.rs          append-only JSONL trace log
src/path.rs          tensor witness path reconstruction
src/node.rs          compact node universe
```

## Execution loop

At each loop iteration:

1. CUDA closes the tensor.
2. Exploration seeds strategies from `assets/exploration.json`.
3. CUDA expands, scores, top-k selects, and commits the best effect-safe exploration candidate that passes terminal and first-hop anti-repeat limits.
4. If exploration cannot commit, the scheduler attempts a CKA batch projection for compiled `par` groups.
5. If a full batch is runnable and effect-safe, CUDA commits the batch and host workers execute operators concurrently.
6. If no batch is ready, CUDA runs the normal single frontier step.
7. Process results become `ProcessReceipt` and then `ExecutionReceipt`.
8. Receipts update both tensor deltas and exploration receipt priors; committed terminals and first hops are tracked to prevent repeated exploration dominance.
9. Tensor-plan stdout is parsed and embedded when an operator declares `output_mode = "tensor_plan"`.
10. Decisions, exploration records, batch plans, receipts, and edge deltas are logged to `quantale.tlog`.

Runtime smoke output for the default CKA fork includes:

```text
batch_step=5 projection=(Event::InputAccepted->State::Map) first_hop=State::Map
batch_step=5 projection=(Event::InputAccepted->State::Parse) first_hop=State::Parse
[BATCH] operator=State::Map exit=0 outcome=Success
[BATCH] operator=State::Parse exit=0 outcome=Success
```

## Validation

```bash
cargo fmt --check
cargo check
cargo test --lib
cargo test --test exploration
cargo test --test tensor_quantale
cargo run --bin bench_tensor_quantale -- 3
cargo run --bin quantale_semiring_v2
```

Current validated test slices:

```text
cargo test --lib                  24 passed
cargo test --test exploration
cargo test --test exploration       17 passed
cargo test --test tensor_quantale  8 passed
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

Captured with:

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

The 1000-iteration release benchmark command was attempted first, but exceeded the tool timeout in this environment. The 10-iteration release baseline above completed successfully.


Record:

```text
hardware
CUDA version
Rust profile
iteration count
tensor_closure avg_us
tensor_projection avg_us
tensor_decay avg_us
```

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
side-channel policy/receipt files
PyTorch/JAX/Triton alternate runtime
```

## Proof boundary

Lean/cLean artifacts live under `lean/`. They name the proof boundary for tensor closure, projection, frontier, tick, and batch projection/commit behavior. They are specification artifacts unless a local Lean/cLean toolchain is installed and run.
