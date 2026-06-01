# Pending: Tensor Quantale Orchestrator

## Current canonical state

The project has crossed the tensor-runtime, CKA-parallel-runtime, and CUDA exploration-runtime boundaries.

Canonical execution path:

```text
TensorEdge
  → TensorQuantaleWorld
  → tensor_quantale_closure
  → exploration scheduler attempt
      → CUDA seed / expand / score / top-K / commit
  → CKA batch scheduler fallback
      → tensor_quantale_project_batch
      → effect-safe DecisionBatch
      → tensor_quantale_commit_batch
      → host parallel dispatch
  → tensor_quantale_frontier_step fallback
  → operator execution
  → tensor feedback / tensor receipt deltas
  → T := T ∨ ΔT
```

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

## Completed and collapsed into docs/tests

- Tensor topology fields in `assets/topology.json`.
- Tensor-native `TensorEdge` compilation.
- Tensor LLM plan compiler.
- Tensor feedback updates through GPU edge kernels.
- Tensor CUDA closure/projection/frontier/tick kernels.
- Tensor feedback update and decay kernels.
- Tensor witness path reconstruction API.
- Full CKA expression model: zero, one, node, seq, choice, star, par.
- Data-driven `assets/patterns.json` compiler.
- Effect metadata in `assets/operators.json`.
- Effect independence validation for `par`.
- CUDA batch projection kernel.
- CUDA batch commit kernel.
- CUDA exploration seed/expand/score/top-k/commit kernels.
- Exploration-first scheduler dispatch.
- Receipt-prior feedback into exploration.
- Host parallel scheduler dispatch.
- Append-only batch-plan trace logging.
- Runtime smoke test showing `[BATCH]` execution.

## Implemented surface

Runtime:

```text
main.rs uses TensorQuantaleWorld
main.rs loads and compiles CKA patterns
main.rs attempts CKA batch scheduling before single-step fallback
assets/operators.json uses output_mode=tensor_plan where needed
assets/operators.json includes effects metadata
assets/call_llm.py emits confidence/cost/safety tensor plans
```

Tensor APIs:

```text
TensorEdge
ProjectionBias
ExecutionOutcome
TensorQuantaleWorld::from_tensor_edges
TensorQuantaleWorld::embed_tensor_edges
TensorQuantaleWorld::close
TensorQuantaleWorld::project
TensorQuantaleWorld::project_parallel_group
TensorQuantaleWorld::commit_decision_batch
TensorQuantaleWorld::frontier_step
TensorQuantaleWorld::tick
TensorQuantaleWorld::update_lattice_edge
TensorQuantaleWorld::decay
TensorQuantaleWorld::reconstruct_tensor_path
TensorQuantaleWorld::seed_exploration
TensorQuantaleWorld::expand_exploration
TensorQuantaleWorld::commit_exploration_candidate
```

CKA / batch APIs:

```text
load_default_patterns
compile_pattern
compile_patterns_to_tensor_edges
validate_cka_expr
safe_parallel
project_ready_batch_plan
prepare_parallel_batch_plan
dispatch_decision_batch_blocking
TlogWriter::append_batch_plan
```

CUDA kernels:

```text
tensor_quantale_reset
tensor_quantale_embed_edges
tensor_quantale_closure
tensor_quantale_project
tensor_quantale_project_batch
tensor_quantale_commit_batch
tensor_quantale_frontier_step
tensor_quantale_tick
tensor_quantale_update_edge
tensor_quantale_decay
tensor_quantale_seed_exploration
tensor_quantale_expand_tokens
tensor_quantale_score_tokens
tensor_quantale_select_topk_tokens
tensor_quantale_commit_exploration
```

Validation command set:

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
cargo test --lib                  24 passing
cargo test --test exploration
cargo test --test exploration       15 passing
cargo test --test tensor_quantale  8 passing
```

## Active pending queue

### P1. Typecheck and prove Lean/cLean tensor + batch boundary

Status: spec-boundary names exist; not typechecked in this workspace.

Reason:

```text
lean / lake / cLean binaries are not installed locally
```

Remaining work when toolchain exists:

- Typecheck `lean/QuantaleSemiringV2/Spec.lean`.
- Replace abstract boundary predicates with real refinement statements.
- Prove layer laws for:
  - max-times confidence
  - min-plus cost
  - max-min safety
- Attach CUDA kernels to cLean refinement boundary:
  - `tensor_quantale_closure`
  - `tensor_quantale_project`
  - `tensor_quantale_project_batch`
  - `tensor_quantale_commit_batch`
  - `tensor_quantale_frontier_step`
  - `tensor_quantale_tick`

### P2. Add release-mode tensor + exploration benchmark baseline

Status: benchmark exists; release baseline not recorded.

Command:

```bash
cargo run --release --bin bench_tensor_quantale -- 1000
```

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

### P3. Expand operator coverage

Status: scheduler and tensor runtime are working. Some nodes intentionally return `Node operator contract missing from registry` during full orchestrator smoke runs because not every topology node has an operator contract.

Options:

```text
A. Add no-op contracts for all non-executable Event/Control nodes.
B. Keep missing contracts as explicit failure receipts.
C. Split executable State nodes from symbolic Event/Control nodes in scheduler policy.
```

Recommended current choice:

```text
A for smoke-clean demos, C for stricter runtime semantics.
```

## Explicit non-goals

Do not add:

- Python/PyTorch/JAX/Triton alternate engine.
- CPU routing planner.
- Hidden imperative graph traversal.
- Scalar sidecar metadata model.
- New policy side-channel files.
- Scalar CUDA world.
- Scalar LLM plan format.

## Next recommended task

Add release benchmark measurements to `README.md` or a dedicated `BENCHMARKS.md`, then expand operator coverage for symbolic Event/Control nodes if clean smoke-test logs are desired.
