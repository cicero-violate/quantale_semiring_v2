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
E_g = 1   eligibility computed on-device from per-member is_gpu_dispatchable flags in the table
D_h = ½   hot-region par members dispatch in-kernel through the H_f path;
          fusion/abstract members still use thread::scope → execute_*_blocking
R_d = ¾   H_f hot-region receipts are written by the par kernel to the device ring;
          CPU-dispatched hot-region fallback still routes via gpu_dispatch_region
R_k = 1   per-member (region_id, is_gpu_dispatchable) encoded in table; kernel emits region_ids
          and dispatched_on_device flags (no per-tick hot_region_registry lookup)
```

Next implementation plan:
1. **~~Tighten H_f dispatch accounting.~~ ✓ Implemented**
   The par kernel now marks `dispatched_on_device = 1` only after a known H_f region handler runs and writes its device receipt. Rust uses a shared `GPU_HOT_REGION_COUNT` constant instead of passing an over-broad region-count placeholder.
2. **~~Add a GPU-side dispatch descriptor ABI for non-H_f members.~~ ✓ Implemented**
   The par kernel now emits a compact per-member descriptor containing member index, node id, region id, selected src/dst, and dispatch kind. H_f members are upgraded to `PAR_DISPATCH_HF_DEVICE`; the remaining fusion/abstract members stay marked `PAR_DISPATCH_HOST_FALLBACK` for the next execution tier.
3. Replace descriptor CPU execution incrementally.
   - **~~Consume descriptors in the par dispatcher.~~ ✓ Implemented** `dispatch_gpu_parallel_group` now uses the GPU-emitted `dispatch_kind` as the source of truth for H_f skip behavior, and the receipt-routing loop uses the same descriptor rather than re-deriving device dispatch from a parallel flag vector.
   - **~~Preserve fusion-entry routing in descriptors.~~ ✓ Implemented** Epoch-start par tables now include an initial dispatch kind; the GPU emits `PAR_DISPATCH_FUSION_ENTRY` descriptors for fusion entries, and the host dispatcher routes those descriptors through the fusion path instead of generic abstract fallback.
   - **~~Route fusion descriptors through a batch dispatch boundary.~~ ✓ Implemented** The par dispatcher now collects all `PAR_DISPATCH_FUSION_ENTRY` members and sends them through `execute_fusion_entries_batch_blocking` in one fusion worker, leaving only abstract/process members on the generic host fallback path.
   - Next: replace the batch boundary internals with a true multi-chain GPU/JIT batch launch.
4. Rework par commit from thread-0 control flow into a parallel readiness/commit kernel using per-member lanes and atomics or a two-kernel select+commit protocol. Keep the current sequential commit until the replacement is proven.
5. Collapse receipt routing so every GPU-dispatched par member writes exactly one device-ring receipt; CPU queue routing remains only for process/abstract fallbacks.

What changed: the CPU group-selection loop is gone. The GPU now selects and commits. The par table encodes `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuples; the kernel computes eligibility on-device from `is_gpu_dispatchable` flags without a separate CPU-precomputed mask and emits per-member dispatch descriptors. Hot-region par members with registered slot tables now run their precompiled `__device__` region function inside `tensor_quantale_par_group_step`; the kernel writes a `DeviceReceipt` to the device ring and emits `dispatched_on_device = 1`, so `dispatch_gpu_parallel_group` skips `execute_*_blocking` for that member and returns a synthetic success receipt. What remains CPU-side:

**Gap A — operator dispatch is still host-bound for non-H_f members.**
Hot-region par members with slot tables close the CPU dispatch hop: `par_group_step` calls the precompiled H_f device function and writes the receipt in-kernel. Fusion-entry, abstract-node, and hot-region fallback members still use `thread::scope → execute_fusion_entry_blocking / execute_abstract_node_blocking`.

**~~Gap B — effect validation is precomputed, not on-device.~~ ✓ Closed**
Per-member `is_gpu_dispatchable` flags are encoded in the table as the third element of each `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuple. The kernel computes `all_eligible` on-device by scanning flags — no separate CPU-precomputed `eligible[]` array. Build-overlay still validates `par` independence structurally, so runtime effect conflicts cannot arise.

**Gap C — commit is single-threaded control-flow, not CUDA atomic.**
The kernel runs `threadIdx.x == 0` only. The commit is all-or-nothing by control flow. Correct term: "device-side sequential commit", not "atomic GPU commit" (unless atomic is used in the logical/transactional sense).

**Gap D — par-tier receipts partially use the device ring.**
H_f hot-region par members write receipts directly from `tensor_quantale_par_group_step` to the device ring. Hot-region fallback members still call `gpu_dispatch_region` + `drain_device_receipts` after CPU execution. Fusion-entry and abstract-node par members still use `queue_lattice_update` + `drain_lattice_queue`.

Full closure requires eliminating the CPU dispatch hop for fusion-entry and abstract members too: GPU kernel emits dispatch descriptors or calls precompiled device functions for every eligible member → device writes receipt to ring. That removes the remaining `thread::scope` work from `dispatch_gpu_parallel_group`.

**Correct label for the current state:**
```text
GPU-selected parallel dispatch tier with in-kernel H_f hot-region dispatch;
fusion/abstract members still CPU-dispatched
```

Not: "fully GPU-native orchestration" — CPU still handles non-H_f operator dispatch (Gap A).

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
