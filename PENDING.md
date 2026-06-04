# Pending

Progressive gaps in the current system. Items here are forward improvements only — nothing that would require touching the active dispatch path without a clear replacement ready.

---

## 1. `TensorQuantaleWorld::project_parallel_group` / `commit_decision_batch` are dormant public API

**What:** `src/tensor.rs` exposes `project_parallel_group` and `commit_decision_batch`, backed by the `tensor_quantale_project_batch` and `tensor_quantale_commit_batch` CUDA kernels in `quantale_world.cu`. The CPU batch scheduler that called them is gone, but both methods have full integration-test coverage in `tests/tensor_quantale.rs` and the CUDA kernel implementations are correct.

**Decision:** Keep as dormant GPU-parallel substrate. These are the natural execution primitives for a future GPU-native parallel commit tier that bypasses a CPU scheduler entirely. The test coverage means they are not dead code — they are a tested capability waiting for a policy layer.

**Action when ready:** Wire `project_parallel_group` into `execute_active_node_blocking` under a new dispatch branch, replacing the old CPU-side `batch.rs` policy with a GPU-side decision. Keep the kernel implementations unchanged.

---

## ~~2. CKA pattern compilation belongs at build time~~ ✓ Implemented

Build-overlay now emits `assets/patterns.compiled.json`. The runtime loads from it on epoch start; falls back to runtime CKA compilation only when the file is absent. `cli.rs` calls `compile_and_emit_pattern_edges(".")` after `build_overlay_assets`.

---

## ~~3. `TlogRecordKind::ExplorationExpand` and `append_exploration_expand` are never called~~ ✓ Removed

Both the enum variant and the writer method were deleted from `tlog.rs`.

---

## ~~4. Tensor feedback does not persist across epoch reloads~~ ✓ Implemented

`LearningBuffer` added to `learning.rs`. Wired into `RuntimeEpoch` and both execution paths in `main.rs`. Successful topology-edge executions are buffered and flushed to `state/learned_edges.jsonl` every 10 records, on epoch reload, and at shutdown. Three tests added: flush-writes-readable-jsonl, auto-flush-at-threshold, empty-flush-is-noop.

---

## ~~5. `CompiledCkaPattern.parallel_groups` is dead weight~~ ✓ Removed

`parallel_groups: Vec<Vec<i32>>` removed from `CompiledCkaPattern`. All `compile_*` internal functions updated to drop the parameter. The par effect-safety validator (`validate_parallel_independence`) is retained — it guards the constraint, not the output artifact.
