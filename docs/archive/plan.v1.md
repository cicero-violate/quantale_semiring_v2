# Cleanup Plan: Data-Driven Static Kernel Fusion

## Goal

Stop carrying duplicate architecture.

The final system is simple:

```text
JSON assets define the graph and operators
Rust parses the JSON
CUDA kernels are static and precompiled
Quantale weights select fused vs unfused paths
executor runs the selected operator
```

No generated node ABI.
No separate fusion crate.
No runtime fusion planner.
No PTX parsing or stitching.
No fake CUDA success receipts.

---

## 1. Delete the debt ✓

Remove all separate fusion/addon scaffolding:

```text
addons/kernel_fusion/
addons/fused_kernel/
src/fusion.rs
```

Remove from `Cargo.toml`:

```toml
kernel_fusion = { path = "addons/kernel_fusion", optional = true }
```

Remove from `[workspace]`:

```toml
"addons/kernel_fusion"
```

Regenerate `Cargo.lock` after deletion.

---

## 2. Consolidate around `topology.rs::NodeRegistry` ✓

`topology.rs` already had a `NodeRegistry`. No second one was created in `node.rs`.

`src/node.rs` is now a thin `Node(i32)` wrapper only. All constants deleted:

```text
STATE_NODE_COUNT, CONTROL_NODE_COUNT, EVENT_NODE_COUNT
STATE_OFFSET, CONTROL_OFFSET, EVENT_OFFSET
NODE_COUNT, STATE_COUNT, MATRIX_LEN, THREAD_COUNT
NODE_NAMES static array
StateNode, ControlNode, EventNode (all deleted)
START_NODE, EXECUTE_PROBE_NODE, LEARN_PROBE_NODE
Node::state() / Node::control() / Node::event() constructors
Node::name() returning &'static str  →  now returns Option<&str> from registry
```

`topology.rs::NodeRegistry` extended with:

```rust
pub fn matrix_len(&self) -> usize { self.len() * self.len() }
pub fn contains_name(&self, name: &str) -> bool { ... }
pub fn action_of(&self, id: usize) -> Option<&str> { ... }
```

`GraphTopology::bundled_registry()` added as a convenience constructor.

`TopologyNode` now accepts an optional `"action"` field in JSON:

```json
{ "id": 17, "name": "Control::Commit", "type": "Control", "action": "commit" }
```

Topology validation now self-contained: dense-id check, transition name check — no external `NODE_COUNT`.

Tests no longer use `Node::state(StateNode::Map)` etc. — all use registry lookups.

---

## 3. Make config use the compiled registry ✓

`SystemConfig` derives `matrix_dim` and `matrix_len` from `GraphTopology::bundled_registry()`.

Validation block compares against `registry.len()` / `registry.matrix_len()`.

---

## 4. Make topology validation trust JSON ✓

`build_registry()` removed the `> NODE_COUNT` guard and the `Node::decode_index` bound check.

Added:
- Dense-id validation: ids must be `0..N`
- Transition name validation: source and destination must be registered node names

---

## 5. Fix `exploration.rs` vector allocation ✓

`ExplorationEngine` now holds `node_count: usize` set from `registry.len()` at construction.

`receipt_prior_vector()` and `visit_vector()` use `self.node_count`.

---

## 6. Remove CPU orchestration debt ✓

### `src/batch.rs`

Deleted:
```text
batch_contains_cuda_ptx()
dispatch_cuda_ptx_batch_blocking()
build_cuda_ptx_fusion_receipt() (both cfg variants)
cuda_ptx_error_receipts()
failed_cuda_ptx_receipt()
payload_shape()
```

`dispatch_decision_batch_blocking` is now uniform — all operators including `cuda_ptx` go through the same thread-scope fan-out. Backend routing lives entirely in `egress.rs`.

`node_name()` / `decision_node_name()` now take `&NodeRegistry` and return `&str` (not `&'static str`).

Tests updated to use registry lookups instead of `StateNode`/`Node::state`.

### `src/projection.rs`

Deleted:
```text
QuantaleAction enum
DecisionReport::selected_action()
all hard-coded node comparisons
```

Replaced with:

```rust
pub fn action_label(node_id: i32, registry: &NodeRegistry) -> &str {
    if node_id < 0 { return "blocked"; }
    registry.action_of(node_id as usize).unwrap_or("unknown")
}
```

### `src/egress.rs`

`execute_abstract_node_blocking` routes by backend:

```rust
if binary == "cuda_ptx" {
    return execute_cuda_ptx_blocking(node_name, op_config, dynamic_payload);
}
// otherwise: Command::new path
```

`UniversalExecutor` holds a `NodeRegistry` loaded from the bundled topology.

---

## 7. CUDA kernel compilation split

### Compilation: `build.rs` + nvcc

`build.rs` in the main crate compiles `cuda/trading_execution_kernels.cu` to PTX
when the `cuda` feature is active:

```rust
// Skeleton — see actual build.rs
if env::var("CARGO_FEATURE_CUDA").is_err() { return; }

nvcc(
    "cuda/trading_execution_kernels.cu",
    &["-ptx", "-o", &out_path, "-std=c++17", "--use_fast_math"],
);
// PTX lands at $OUT_DIR/trading_execution_kernels.ptx
```

`cargo:rerun-if-changed=cuda/trading_execution_kernels.cu` ensures incremental rebuilds.

The main quantale world kernel (`cuda/quantale_world.cu`) keeps NVRTC — it is small,
iterated on frequently, and needs no nvcc flags. Operator kernels use nvcc because:

- Full `--use_fast_math`, arch flags, and `-O3` optimization
- No NVRTC limitations (cooperative groups, cub, complex atomics)
- PTX artifact is deterministic across builds
- No JIT startup cost in the dispatch hot path

### Runtime dispatch: `egress.rs` + cudarc

`execute_cuda_ptx_blocking` (cuda feature only):

```rust
// PTX embedded at compile time from build.rs output
const PTX_BYTES: &[u8] = include_bytes!(
    concat!(env!("OUT_DIR"), "/trading_execution_kernels.ptx")
);

// Device and module loaded once via OnceLock
// kernel name read from operators.json input_mapping.kernel
// inputs marshalled from JSON payload → device slices
// result copied back and returned as JSON in stdout_payload
```

`cudarc` is the runtime API for both the quantale world and operator kernels.
`build.rs`/nvcc is only the ahead-of-time compilation step for operator kernels.

### Kernel stubs in `cuda/trading_execution_kernels.cu`

```text
fused_alpha_and_risk_kernel(market_feed, portfolio_state, trading_signals, results, n)
fused_orderbook_and_alpha_kernel(orderbook, alpha_signals, results, n)
fused_feed_alpha_and_risk_kernel(feed, alpha_signals, portfolio_state, results, n)
```

Each stub is a real `extern "C" __global__` function with the correct symbol name.
Placeholder arithmetic; replace with real quant logic once the dispatch path is verified.

---

## 8. Add fused nodes to assets

### `assets/operators.json`

```json
{
  "node_name": "Execution::FusedAlphaAndRisk",
  "executable": "cuda_ptx",
  "static_args": [],
  "input_mapping": {
    "module": "cuda/trading_execution_kernels.ptx",
    "module_name": "quantale_trading_execution_kernels",
    "kernel": "fused_alpha_and_risk_kernel",
    "plane": "execution",
    "scheduler_contract": "atomic_operator_fixed_budget"
  },
  "effects": {
    "reads": ["market.feed", "portfolio.state", "trading.signals"],
    "writes": ["execution.gpu.results"],
    "locks": []
  }
}
```

### `assets/topology.json`

Add fused operators as normal graph nodes:

```text
Execution::FusedAlphaAndRisk
Execution::FusedOrderbookAndAlpha
Execution::FusedFeedAlphaAndRisk
```

No Rust edits needed — registry parses JSON.

### `assets/patterns.json`

```json
{
  "name": "trading_alpha_risk_choice",
  "expr": {
    "choice": [
      { "seq": ["Execution::DynamicAlphaSignalEvaluator", "Execution::PortfolioRiskConstraintFilter"] },
      "Execution::FusedAlphaAndRisk"
    ]
  },
  "confidence": 0.96,
  "cost": 0.35,
  "safety": 0.95
}
```

### Graph weights

```text
State::Execute -> Execution::DynamicAlphaSignalEvaluator  cost 1.0
State::Execute -> Execution::FusedAlphaAndRisk            cost 0.35
```

The scheduler picks the fused node because the data says it is cheaper. No Rust `if fused`.

---

## 9. Tests

Required:

```text
[ ] topology.rs::NodeRegistry round-trips name -> id -> name for every bundled node
[ ] NodeRegistry::action_of returns the action declared in JSON
[ ] NodeRegistry::matrix_len returns len * len
[ ] adding Execution::FusedAlphaAndRisk to topology.json requires no Rust node-code edit
[ ] operators.json references only known topology nodes
[ ] patterns.json references only known topology nodes
[ ] no NODE_NAMES array exists anywhere in src/
[ ] no StateNode / ControlNode / EventNode constants exist anywhere in src/
[ ] addons/kernel_fusion does not exist
[ ] kernel_fusion does not appear in Cargo.toml
[ ] FusionPlan / FusionKey / ptx_body_stitching do not appear in source
[ ] batch_contains_cuda_ptx does not exist
[ ] Command::new("cuda_ptx") is never called
[ ] cuda_ptx disabled returns explicit error (not process spawn failure)
[ ] fused path wins when topology cost is lower and safety/confidence allow it
[ ] unfused path can win if fused safety/confidence is degraded
[ ] exploration receipt_prior_vector length matches registry.len()
[ ] exploration visit_vector length matches registry.len()
```

Run:

```text
cargo fmt --all
cargo test --workspace --no-default-features
cargo test --workspace --features cuda   # on CUDA hosts
```

---

## Completion Criteria

```text
[✓] src/node.rs constants deleted; topology.rs::NodeRegistry is the single source of truth.
[✓] New nodes added by editing JSON only.
[ ] Fused kernels are static CUDA kernels in cuda/trading_execution_kernels.cu.
[ ] Fused operators declared in operators.json.
[ ] Fused choices declared in patterns.json.
[ ] Fused preference encoded in topology/tensor weights.
[✓] No separate fusion crate.
[✓] No runtime fusion planner.
[✓] No PTX parsing/stitching.
[✓] No CPU thread fanout special-casing fusion or CUDA.
[✓] No fake CUDA planned-success receipt.
[✓] egress.rs routes cuda_ptx by backend, not node identity.
[✓] projection.rs has no hard-coded node comparisons.
[✓] exploration.rs allocates vectors from registry.len().
[ ] build.rs compiles cuda/trading_execution_kernels.cu → PTX.
[ ] egress.rs execute_cuda_ptx_blocking loads PTX via cudarc and launches.
[ ] batch.rs tests use registry lookups, not StateNode constants.
```
