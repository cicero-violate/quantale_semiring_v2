# fix.md — Quantale Semiring v2 Upgrade Review and Fix Plan

Last reviewed against current working tree: 2026-06-04.

## Review scope

Reviewed the current implementation against the seven stated upgrade priorities and the core architecture invariant:

```text
topology.source.json -> generated control/hot/fusion/runtime views -> GPU execution -> truthful receipts -> tensor update
```

Reviewed surfaces:

```text
assets/topology.source.json
assets/topology.generated.json
assets/topology.control.json
assets/topology.hot.json
assets/regions.hot.json
assets/topology.fusion.json
assets/operators.generated.json
src/topology.rs
src/config.rs
src/device_slots.rs
src/hot_region.rs
src/ir.rs
src/fusion_dispatch.rs
src/jit_kernel_fusion/cache.rs
src/egress.rs
src/main.rs
src/tensor.rs
cuda/quantale_world.cu
tests/cuda_smoke.rs
```

---

## Current verification result

### Confirmed completed or mostly completed

```text
Priority 1 — Region slot names fixed.
Priority 2 — Split topology runtime types added and loaded through SystemConfig.
Priority 3 — Hot topology transitions added.
Priority 4 — DeviceRingBuffer push/pop surfaces added.
Priority 5 — CPU ingress staging/upload surfaces added.
Priority 6 — GPU-side hot-region dispatch functions added.
Priority 7 — TypedIR unsafe placeholder lowerings removed.
cargo check --no-default-features passes.
cargo test --lib --no-default-features passes: 58 library tests.
```

### Still needs attention

```text
cargo test --no-default-features still needs verification/fix for integration tests.
Region::CommitReceipt is in hot topology/regions but was not found as a source node in topology.source.json during validation.
PinnedHostBuffer is a staging abstraction, not confirmed true OS/CUDA pinned host memory.
AsyncUploadQueue uses staged host buffers and flush; verify whether it is truly async or currently htod_copy-style synchronous upload.
DeviceRingBuffer push/pop exists; add functional tests that prove order, wraparound, and overflow behavior.
GPU dispatch switch exists; verify each region's device function computes declared outputs, not only emits receipt metadata.
```

---

## Core invariant

Keep one orchestration source of truth:

```text
assets/topology.source.json
```

Generated/runtime views:

```text
assets/topology.generated.json
assets/topology.control.json
assets/topology.hot.json
assets/regions.hot.json
assets/topology.fusion.json
assets/operators.generated.json
assets/patterns.source.json
```

Required equation:

```text
topology.source.json -> build-overlay -> generated runtime views
```

Do not manually maintain split-brain topology files.

---

## Priority 1 — Region slot names

### Verification

`assets/regions.hot.json` now uses source-declared slots for the hot regions.

Expected math slots are present:

```text
math.a
math.b
math.add_out
math.scale
math.out
market.open
market.price
analysis.return
analysis.volatility
analysis.signal_score
```

No `execution.*` slot references were found in `regions.hot.json` during the latest validation.

The latest slot validation found:

```text
missing region slots from topology.source.json slots = 0
```

### Status

```text
COMPLETED
```

### Remaining guardrail

Add a permanent validator:

```text
validate_hot_region_slots:
  for every region in regions.hot.json:
    reads ∪ writes ⊆ topology.source.json.slots
```

---

## Priority 2 — Activate split topology

### Verification

The code now contains split topology runtime surfaces:

```text
ControlTopologyRuntime
HotTopologyRuntime
SplitTopologyRuntime
```

`SystemConfig` stores split topology runtime state and reloads it through the hot-region reload path.

Expected invariant set:

```text
1. control ∩ hot = empty
2. hot contains no IO/boundary/host-only nodes
3. hot regions map to region metadata
4. generated split artifacts match source topology partition rules
```

### Status

```text
MOSTLY COMPLETED
```

### Remaining issue

The latest validation found:

```text
Region::CommitReceipt appears in hot topology / regions.hot.json
Region::CommitReceipt was not found as a declared source node in topology.source.json
```

This may be acceptable only if `Region::CommitReceipt` is explicitly a compiler-generated synthetic runtime node. If it is intended to participate in orchestration, declare it in `topology.source.json`.

### Required fix

Choose one:

#### Option A — source-declared region node

Add `Region::CommitReceipt` to `topology.source.json` with hot/kernel metadata.

#### Option B — synthetic runtime node

Document and validate it as a generated synthetic node:

```text
synthetic_hot_nodes = { Region::CommitReceipt }
```

Then update validators so this exception is explicit, not accidental.

---

## Priority 3 — Hot topology transitions

### Verification

`assets/topology.hot.json` now has:

```text
nodes = 7
transitions = 6
```

Expected transition shape was added:

```text
Analysis::Return1 -> Analysis::Volatility
Analysis::Volatility -> Analysis::SignalScore
Execution::VectorAdd -> Execution::VectorScale
leaf regions -> Region::CommitReceipt
```

### Status

```text
COMPLETED
```

### Remaining guardrail

Add validator:

```text
validate_hot_topology_is_executable:
  hot.nodes.len > 0
  hot.transitions.len > 0
  all transition endpoints exist in hot.nodes
```

---

## Priority 4 — DeviceRingBuffer push/pop

### Verification

The code now contains ring-buffer push/pop surfaces:

```text
cuda/quantale_world.cu:
  device_ring_push
  device_ring_pop

src/device_slots.rs:
  DeviceRingBuffer::push(...)
  DeviceRingBuffer::pop(...)

src/tensor.rs:
  ring kernels registered during PTX load
```

### Status

```text
IMPLEMENTED, NEEDS BEHAVIORAL TESTS
```

### Required tests

Add CUDA-feature tests for:

```text
push then pop returns same value
multiple push/pop preserves FIFO order
wraparound works
full buffer behavior is defined
empty pop behavior is defined
head/tail consistency after mixed operations
```

### Important note

This confirms ring-buffer methods exist. It does not yet prove live multi-stream ingestion is complete.

---

## Priority 5 — CPU ingress pipeline

### Verification

The code now contains:

```text
PinnedHostBuffer
AsyncUploadQueue
stage(...)
flush(...)
```

These are re-exported through the public surfaces.

### Status

```text
PARTIALLY COMPLETED
```

### Remaining semantic check

Confirm whether `PinnedHostBuffer` is true pinned host memory or only a named staging buffer.

True pinned CUDA memory usually requires APIs such as:

```text
cudaHostAlloc
cudaHostRegister
mapped host allocation
```

If the current implementation uses normal `Vec<f32>` plus `htod_copy`, then rename or document it accurately:

```text
HostStagingBuffer
```

instead of:

```text
PinnedHostBuffer
```

Likewise, if `AsyncUploadQueue::flush()` performs synchronous `htod_copy`, then call it staged upload, not async upload.

### Required tests

```text
stage accumulates streams into the expected slots
flush uploads staged values to DeviceSlotRegistry
flush clears staged queue only after successful upload
failed upload preserves or reports staged state
```

---

## Priority 6 — GPU-side device function dispatch

### Verification

`cuda/quantale_world.cu` now contains:

```text
seven __device__ hot-region functions
tensor_quantale_gpu_dispatch switch table
quantale_parallel_reduce
quantale_topk_bitonic
```

Device receipt outcome now comes from:

```cpp
mailbox->outcome
```

not hardcoded success.

### Status

```text
IMPLEMENTED, NEEDS SEMANTIC VALIDATION
```

### Remaining checks

For each region ID in `regions.hot.json`, verify:

```text
region_id maps to the intended __device__ function
function reads the declared input slots
function writes the declared output slots
function emits receipt only after valid compute path
failure outcome does not strengthen tensor edge
```

### Important risk

A switch table in CUDA is only useful if device pointers for declared slots are actually available to the device dispatch function.

If `tensor_quantale_gpu_dispatch` still lacks real slot pointers and only records receipts, then it is still a receipt bridge plus dispatch stub, not full data-region execution.

---

## Priority 7 — TypedIR lowering

### Verification

`src/ir.rs` now returns explicit `Err` for operations that cannot be safely lowered to scalar element bodies:

```text
Reduce
TopK
MatMul
Join
Sort
GraphTraverse
```

This fixes the previous silent-wrong-result risk.

### Status

```text
COMPLETED
```

### Remaining guardrail

Add tests asserting these return `Err`:

```text
ir_reduce_rejects_scalar_lowering
ir_topk_rejects_scalar_lowering
ir_matmul_rejects_scalar_lowering
ir_join_rejects_scalar_lowering
ir_sort_rejects_scalar_lowering
ir_graph_traverse_rejects_scalar_lowering
```

---

## Build and test status

### Verified passing

```text
cargo check --no-default-features --quiet
cargo test --lib --no-default-features --quiet
```

Library tests reported:

```text
58 passed
```

### Needs fix / verification

```text
cargo test --no-default-features --quiet
```

Previously this failed in `tests/cuda_smoke.rs` due to non-CUDA unused imports and unreachable code. Re-check after the latest edits. If still failing, split the CUDA smoke test into cfg-gated modules:

```rust
#[cfg(feature = "cuda")]
mod cuda_smoke {
    use quantale_semiring_v2::{JitCache, JitChain, OperatorRegistry, load_operator_registry};

    #[test]
    fn cuda_smoke_executes_jit_chain() -> Result<(), String> {
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_smoke_skipped_without_cuda_feature() {}
```

---

## Remaining P0/P1 fixes

### P0.1 — Resolve `Region::CommitReceipt` source status

Current validation concern:

```text
Region::CommitReceipt not found in topology.source.json nodes
```

Fix by source-declaring it or marking it as an explicit generated synthetic node.

### P0.2 — Ensure all no-default tests pass

Required command:

```text
cargo test --no-default-features --quiet
```

must pass, not only `cargo test --lib`.

### P1.1 — Add regression tests for prior bugs

Add tests for:

```text
fusion dispatch wins over single-node hot dispatch
failed JIT receipt maps to failure outcome on device
hot region slots all exist in source topology
hot topology has no IO nodes
split topology loads through SystemConfig
```

### P1.2 — Validate generated artifact determinism

Add tempdir build-overlay test:

```text
topology.source.json -> generated topology.control/hot/regions/fusion
```

and assert the generated artifacts are stable.

---

## Do not proceed to zero-copy yet

The next valid milestone is still correctness, not transport optimization.

Do not prioritize:

```text
GPUDirect
zero-copy mapped host memory
persistent runtime kernel
multi-stream production ingestion
```

until these are true:

```text
cargo test --no-default-features passes
Region::CommitReceipt source/synthetic status is resolved
one fused hot path executes and emits truthful device receipt
ring-buffer push/pop has behavioral CUDA tests
staged ingress has behavioral tests
```

---

## Final target loop

The immediate target remains:

```text
topology.source.json
  -> build-overlay
  -> topology.fusion.json / topology.hot.json / regions.hot.json
  -> JIT fused GPU region
  -> actual execution outcome
  -> truthful device receipt
  -> tensor update
  -> next decision
```

That loop should be made boring, tested, and deterministic before adding more architecture.
