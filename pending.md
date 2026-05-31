# Pending: Tensor Quantale Orchestrator

## Current canonical state

The project has crossed the tensor-runtime boundary. The canonical execution path is now:

```text
TensorEdge
  → TensorQuantaleWorld
  → tensor_quantale_tick
  → tensor_quantale_closure
  → tensor_quantale_frontier_step
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

## Collapse / delete list

These items are now done and should not remain as active backlog.

### Collapsed into tensor runtime

- Scalar-only runtime loop.
- Scalar-only topology compilation.
- Scalar-only LLM plan contract.
- Scalar-only policy/receipt routing.
- Sidecar explanation model.
- Separate policy/receipt modules.
- Duplicate planning backlog docs.

### Deleted from repo

- `src/policy.rs`
- `src/receipt.rs`
- `assets/policy.json`
- `assets/receipt.json`
- `pending.v2.md`
- `plan.md`
- `plan_1.md`

### Completed and collapsed into docs/tests

- Tensor topology fields in `assets/topology.json`.
- Tensor-native `TensorEdge` compilation.
- Tensor LLM plan compiler.
- Tensor policy/receipt rule deltas.
- Tensor CUDA closure/projection/frontier/tick kernels.
- Tensor feedback update and decay kernels.
- Tensor witness path reconstruction API.
- Per-layer tensor witness tests.
- Lean tensor proof-boundary names.

## Implemented surface

Runtime:

```text
main.rs uses TensorQuantaleWorld
assets/operators.json uses output_mode=tensor_plan
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
TensorQuantaleWorld::frontier_step
TensorQuantaleWorld::tick
TensorQuantaleWorld::update_lattice_edge
TensorQuantaleWorld::decay
TensorQuantaleWorld::reconstruct_tensor_path
```

CUDA kernels:

```text
tensor_quantale_reset
tensor_quantale_embed_edges
tensor_quantale_closure
tensor_quantale_project
tensor_quantale_frontier_step
tensor_quantale_tick
tensor_quantale_update_edge
tensor_quantale_decay
```

Validation command set:

```bash
cargo fmt --check
cargo check
cargo test
cargo run --quiet --bin bench_tensor_quantale -- 3
```

Current expected test count:

```text
54 passing
0 failing
```

## Active pending queue

### P1. Typecheck and prove Lean/cLean tensor boundary

Status: spec-boundary names exist; not typechecked in this workspace.

Reason:

```text
lean / lake / cLean binaries are not installed locally
```

Remaining work when toolchain exists:

- Typecheck `lean/QuantaleSemiringV2/Spec.lean`.
- Replace abstract `True` boundary predicates with real refinement statements.
- Prove layer laws for:
  - max-times confidence
  - min-plus cost
  - max-min safety
- Attach CUDA kernels to cLean refinement boundary:
  - `tensor_quantale_closure`
  - `tensor_quantale_project`
  - `tensor_quantale_frontier_step`
  - `tensor_quantale_tick`

### P2. Add release-mode tensor benchmark baseline

Status: benchmark exists; baseline not recorded.

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

### P3. Decide scalar compatibility lifecycle

Status: scalar engine is no longer runtime-canonical, but still useful for compatibility and law tests.

Options:

```text
A. Keep scalar engine as reference/compatibility surface
B. Move scalar engine behind a Cargo feature
C. Delete scalar engine after tensor proof/benchmark confidence is sufficient
```

Recommended current choice:

```text
A. Keep scalar engine as reference/compatibility surface
```

Reason:

```text
It supports old tests, scalar laws, and migration comparisons without affecting tensor runtime.
```

## Explicit non-goals

Do not add:

- Python/PyTorch/JAX/Triton alternate engine.
- CPU routing planner.
- Hidden imperative graph traversal.
- Scalar sidecar metadata model.
- New policy/receipt side-channel files.

## Next recommended task

Run a release benchmark and write the measured baseline into `README.md` or a dedicated `BENCHMARKS.md`.
