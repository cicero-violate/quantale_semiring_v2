# Pending

Progressive gaps in the current system. Items here are forward improvements only — nothing that would require touching the active dispatch path without a clear replacement ready.

---

## ~~1. GPU-native parallel tier~~ ✓ Implemented

`src/runtime_parallel.rs` added. `try_dispatch_parallel_group` projects a CKA `par` group on the GPU (read-only), validates pairwise effect independence, calls `commit_decision_batch` for an atomic GPU commit, then dispatches operators concurrently with `std::thread::scope`.

The main loop now has three tiers between `close` and `frontier_step`:
1. Exploration-first (existing)
2. **GPU-native parallel**: iterates `epoch.topology.parallel_groups` (from `topology.generated.json`), commits and dispatches the first ready group, continues the tick
3. Single frontier step fallback (existing)

`GraphTopology.parallel_groups` field added (serde default = `[]`). `TopologyRuntime.parallel_groups` resolves names to IDs at load time. Receipt processing, tlog, learning, and consecutive-block accounting mirror the other dispatch paths.

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
