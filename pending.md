# Pending: Tensor Quantale Orchestrator

## Implemented

- Scalar max-times engine remains available for compatibility and tests.
- Tensor quantale engine is the canonical runtime path.
- Tensor state is CUDA-resident: `T ∈ R^(3 × 44 × 44)`.
- Tensor layers:
  - confidence/correctness: max-times
  - compute/time cost: min-plus
  - security/safety: max-min
- Tensor topology compilation emits `TensorEdge`.
- Tensor LLM plans require explicit `confidence`, `cost`, and `safety` fields.
- Tensor policy and receipt rule deltas emit `TensorEdge`.
- Tensor closure, projection, frontier step, tick, feedback update, and decay are CUDA kernels.
- `main.rs` uses `TensorQuantaleWorld`.
- Legacy `policy.rs`, `receipt.rs`, `assets/policy.json`, `assets/receipt.json`, `pending.v2.md`, `plan.md`, and `plan_1.md` are removed.

## Validation

Current expected validation set:

```bash
cargo fmt --check
cargo check
cargo test
cargo run --quiet --bin bench_tensor_quantale -- 3
```

## Remaining work

### 1. Make `assets/topology.json` explicitly tensor-valued

The compiler supports explicit tensor fields, but many asset transitions still depend on scalar fallback:

```json
{
  "default_weight": 0.99
}
```

Replace with canonical tensor edge values:

```json
{
  "confidence": 0.99,
  "cost": 0.01,
  "safety": 0.99
}
```

Keep `default_weight` only while scalar compatibility is still required.

### 2. Extend Lean spec to tensor quantale

Formalize:

- max-times confidence layer
- min-plus cost layer
- max-min safety layer
- tensor closure relation
- blended projection boundary

### 3. Add tensor path reconstruction tests per layer

Tensor witness storage is implemented. Add tests for:

- confidence path reconstruction
- cheapest path reconstruction
- safest path reconstruction
- proof that different layers may choose different paths

### 4. Add release-mode tensor benchmark baseline

Current benchmark is available. Add recorded release baseline once hardware target is fixed.

### 5. Decide scalar engine lifecycle

The scalar engine is still useful for compatibility and law tests. Later decision:

- keep as compatibility/reference surface, or
- move behind a feature flag, or
- remove after tensor runtime is fully mature.

## Do not add

- Python/PyTorch/JAX/Triton alternate engine
- CPU routing planner
- hidden imperative graph traversal
- scalar sidecar metadata model
