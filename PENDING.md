# Pending

Progressive gaps in the current system. Items here are forward improvements only — nothing that would require touching the active dispatch path without a clear replacement ready.

---

## ~~1. GPU-native parallel tier~~ ✓ Implemented

New kernel `tensor_quantale_par_group_step` in `cuda/quantale_world.cu`:
- Iterates the GPU-resident par group table (`ParGroupGpuData`)
- For each eligible group: projects all members toward their target nodes on-device
- First group where all members are unblocked and unhalted: committed atomically
- Result (`group_idx`, `decisions`) written to device output buffer
- CPU reads result, dispatches operators via `dispatch_gpu_parallel_group`
- CPU does not iterate groups, validate effects, or call `project_parallel_group` + `commit_decision_batch` separately

The main loop is now three-tiered:
1. Exploration-first (existing)
2. **GPU-native parallel** — `par_group_step` one kernel, CPU dispatches only if group was selected
3. Single frontier step fallback (existing)

Supporting changes:
- `GraphTopology.parallel_groups: Vec<Vec<String>>` (serde default = `[]`)
- `TopologyRuntime.parallel_groups: Vec<Vec<i32>>` resolved at load time
- `ParGroupGpuData`: packed group table + eligibility mask, uploaded at epoch start
- Eligibility = every operator in the group is `jit_cuda` / fusion-entry / hot-region
- `runtime_parallel.rs` reduced to `dispatch_gpu_parallel_group` only

---

## 2. Par-tier: remaining gaps to fully GPU-native execution

Current honest classification:

```text
G_s = 1   GPU selects the par group
G_c = 1   GPU commits consumed/active state
E_g = ½   eligibility checked on-device, but mask is CPU-precomputed at epoch start
D_h = 1   CPU still dispatches operators (thread::scope → execute_*_blocking)
R_d = ½   hot-region par members route receipts through device ring; fusion/abstract use CPU path
```

What changed: the CPU group-selection loop is gone. The GPU now selects and commits. Hot-region par members now issue `gpu_dispatch_region` + `drain_device_receipts` instead of `queue_lattice_update`, so their tensor updates are applied on-device without a CPU drain kernel. What remains CPU-side:

**Gap A — operator dispatch is still host-bound.**
`dispatch_gpu_parallel_group` uses `thread::scope → execute_fusion_entry_blocking / execute_abstract_node_blocking`. Operators run on the GPU (jit_cuda/fusion), but the launch is CPU-initiated per member.

**Gap B — effect validation is precomputed, not on-device.**
The kernel trusts `eligible[g]` from `ParGroupGpuData`. The mask is built at epoch start from `executor.is_hot_node || fusion_dispatch.is_fusion_entry`. Build-overlay already validated `par` independence structurally, so runtime effect conflicts cannot arise. The correct wording is "GPU consumes prevalidated eligibility" not "GPU validates conflicts on-device".

**Gap C — commit is single-threaded control-flow, not CUDA atomic.**
The kernel runs `threadIdx.x == 0` only. The commit is all-or-nothing by control flow. Correct term: "device-side sequential commit", not "atomic GPU commit" (unless atomic is used in the logical/transactional sense).

**Gap D — par-tier receipts partially use the device ring (hot-region members only).**
Hot-region par members now call `gpu_dispatch_region` + `drain_device_receipts` after CPU execution, routing their tensor updates through `tensor_quantale_gpu_dispatch` → `tensor_quantale_drain_device_receipts` without a CPU drain-queue round-trip. Fusion-entry and abstract-node par members still use `queue_lattice_update` + `drain_lattice_queue`.

Full closure requires eliminating the CPU dispatch hop entirely: GPU kernel emits dispatch descriptors → GPU executes operators (CUDA dynamic parallelism or pre-compiled hot functions) → device writes receipt to ring. That removes the `thread::scope` from `dispatch_gpu_parallel_group` and closes Gap A simultaneously.

**Correct label for the current state:**
```text
GPU-selected parallel dispatch tier with CPU-dispatched GPU operators;
hot-region par receipts applied via device ring
```

Not: "fully GPU-native orchestration" — CPU still handles dispatch (Gap A) and eligibility (Gap B).

---

## ~~3. CKA pattern compilation belongs at build time~~ ✓ Implemented

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
