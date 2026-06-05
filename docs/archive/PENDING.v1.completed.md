# Pending

Progressive gaps in the current system. Items here are forward improvements only — nothing that would require touching the active dispatch path without a clear replacement ready.

---

## ~~1. GPU-native parallel tier~~ ✓ Implemented

New kernel `tensor_quantale_par_group_step` in `cuda/quantale_world.cu`:
- Iterates the GPU-resident par group table (`ParGroupGpuData`)
- For each eligible group: projects all members toward their target nodes on-device
- First group where all members are unblocked and unhalted: committed on-device
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
- `ParGroupGpuData`: packed `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` table, uploaded at epoch start
- Eligibility = every operator in the group is `jit_cuda` / fusion-entry / hot-region
- `runtime_parallel.rs` reduced to par dispatch and receipt-routing helpers

---

## 2. Par-tier: remaining gaps to fully GPU-native execution

Current honest classification:

```text
G_s = 1   GPU selects the par group
G_c = 1   GPU commits consumed/active state; selection remains deterministic on
          thread 0, while commit writeback uses block lanes and atomic edge marks
E_g = 1   eligibility computed on-device from per-member is_gpu_dispatchable flags in the table;
          host-bound fusion descriptors are no longer marked GPU-dispatchable
D_h = ⅔   hot-region par members and lowerable fusion-entry par members dispatch
          in-kernel through the H_f path; non-lowerable fusion/abstract members
          remain host fallback and are excluded from GPU par selection
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
   - **~~Add multi-chain fusion batch kernel synthesis/cache.~~ ✓ Implemented** `synthesize_batch_kernel` emits one CUDA kernel for multiple JIT chains and `JitCache::get_or_compile_batch` caches that batch module by chain set.
   - **~~Launch compiled fusion batch kernels from the batch boundary.~~ ✓ Implemented** `execute_fusion_entries_batch_blocking` now compiles the batch kernel, prepares dynamic slot buffers, launches bounded one/two-chain CUDA batches, stores output slots, and returns per-member receipts.
   - **~~Route successful fusion-batch receipts through the device ring.~~ ✓ Implemented** Successful `jit_fused_batch` par receipts are host-detected, pushed into the GPU `DeviceReceipt` ring, and drained on-device instead of using the CPU lattice queue.
   - **~~Generalize the bounded batch launcher to three chains.~~ ✓ Implemented** The runtime launch matcher now supports one, two, or three chains with one to three inputs each, up to the current cudarc tuple arity ceiling (three-chain batches are capped at eight total inputs).
   - **~~Lower matching fusion entries into the H_f dispatch path.~~ ✓ Implemented** Fusion entries whose generated region name maps to a static in-kernel H_f handler, or whose read/write signature matches a pure `jit_fused` hot region with an in-kernel handler, are classified as `PAR_DISPATCH_HF_DEVICE`; unsupported fusion entries remain `PAR_DISPATCH_FUSION_ENTRY` and keep the host batch fallback.
   - **~~Make fusion lowering independent of hot-registry aliases.~~ ✓ Implemented** `fusion_hf_region_id` provides an explicit generated-region-name → H_f region-id table, so known fused batches lower even when `regions.hot.json` has no matching fused alias. Signature matching remains as a compatibility fallback.
   - **~~Tighten par eligibility to GPU-native dispatch only.~~ ✓ Implemented** Epoch-start `is_gpu_dispatchable` is now derived from dispatch kind. `PAR_DISPATCH_HF_DEVICE` remains eligible; host-bound `PAR_DISPATCH_FUSION_ENTRY` and `PAR_DISPATCH_HOST_FALLBACK` are excluded from GPU par selection instead of being selected and dispatched through `thread::scope`.
   - **~~Emit generated H_f coverage and stub targets for future fusion batches.~~ ✓ Implemented** `topology build-overlay` now writes `assets/fusion_hf.generated.json` plus `assets/fusion_hf.stubs.cu`. Runtime lowering loads the coverage manifest and validates covered mappings against the compiled static H_f table; future uncovered fusion regions receive deterministic generated handler symbols instead of silent host fallback.
   - **~~Generate concrete H_f handler fragments for uncovered fusion batches.~~ ✓ Implemented** `fusion_hf.stubs.cu` now emits CUDA device-function fragments from each uncovered fusion chain's `jit_body` and slot effects. Current assets have no uncovered fusion regions, so the file records that all generated fusion regions are covered by static H_f handlers.
   - **~~Wire generated H_f fragments into runtime CUDA source assembly.~~ ✓ Implemented** `TensorQuantaleWorld` now assembles the CUDA source by splicing `assets/fusion_hf.stubs.cu` into `quantale_world.cu` before PTX compilation. The assembler also supports generated `// hf_case: <region_id> <symbol>` switch metadata for both standalone GPU dispatch and par-group dispatch contexts.
   - **~~Assign generated region IDs + slot metadata for uncovered fusion handlers.~~ ✓ Implemented** `fusion_hf.generated.json` now carries `hf_region_id`, `symbol`, and ordered `slots`. Generated handlers start at region id 8, emit `hf_case` metadata, and runtime par setup uses the manifest for handler validation, H_f `region_count`, and per-member slot table construction.
   - **~~Add uncovered-fusion fixture for generated handler promotion.~~ ✓ Implemented** `topology_core` now builds a synthetic `Fixture::Add → Fixture::Scale` CUDA chain that is absent from the static H_f table and verifies generated id 8, generated symbol, ordered slots, `hf_case`, CUDA fragment emission, and local-register lowering for internal slots. `tensor.rs` also verifies generated coverage parsing, region count, switch-case rendering, and CUDA source marker replacement.
   - **~~Implement the reserved `PAR_DISPATCH_ABSTRACT_DEVICE` path.~~ ✓ Implemented** `topology build-overlay` now emits `assets/abstract_device.generated.json` for safe no-op/marker nodes; runtime loads the manifest, classifies covered nodes as `PAR_DISPATCH_ABSTRACT_DEVICE`, and `tensor_quantale_par_group_step` writes their device-ring receipts directly without host `execute_abstract_node_blocking`.
   - **~~Add CUDA-backed integration tests for generated H_f + abstract-device receipts.~~ ✓ Implemented** `tests/cuda_smoke.rs` now executes a generated id 8 H_f handler through `par_group_step`, verifies slot output and device dispatch descriptors, and also executes a `PAR_DISPATCH_ABSTRACT_DEVICE` par group that writes device-ring receipts on GPU.
   - Next: no known software-side Gap A/D_h blocker remains. Remaining work is broader orchestration hardening, performance tuning, and any production topology additions that intentionally create abstract-device manifest entries.
4. **~~Move par commit writeback off the thread-0 memory loop.~~ ✓ Implemented**
   Selection remains deterministic on thread 0, but committed effects are now
   published through shared memory and written back by block lanes: lanes clear
   `next_active`, atomically mark `consumed[src, hop]`, set `next_active[hop]`,
   and copy the new frontier back to `active`.
5. Next Gap C work: rework readiness/selection itself into per-member lanes, or
   split par execution into a two-kernel select+commit protocol if deterministic
   first-ready selection and fully parallel readiness checks cannot coexist in one
   maintainable kernel.
6. Collapse receipt routing so every GPU-dispatched par member writes exactly one device-ring receipt; CPU queue routing remains only for process/abstract fallbacks.

What changed: the CPU group-selection loop is gone. The GPU now selects and commits. The par table encodes `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuples; the kernel computes eligibility on-device from `is_gpu_dispatchable` flags without a separate CPU-precomputed mask and emits per-member dispatch descriptors. Hot-region par members with a valid H_f region id now run their precompiled `__device__` region function inside `tensor_quantale_par_group_step`; when slot tables are present, the handler uses real slot data, and when absent it can still produce a receipt-only device dispatch. The kernel writes a `DeviceReceipt` to the device ring and emits `dispatched_on_device = 1`, so `dispatch_gpu_parallel_group` skips `execute_*_blocking` for that member and returns a synthetic success receipt. What remains CPU-side:

**Gap A — operator dispatch is still host-bound for non-lowerable non-H_f members.**
Hot-region par members and lowerable fusion-entry par members with a valid H_f region id close the CPU dispatch hop: `par_group_step` calls the precompiled H_f device function and writes the receipt in-kernel. Slot-backed members use real device slot data; slotless members are receipt-only. Covered abstract/no-op marker members now use `PAR_DISPATCH_ABSTRACT_DEVICE`, which writes a device-ring receipt directly without host execution. Non-lowerable fusion entries and host fallback members are no longer GPU-par eligible; they remain on the non-par host path until a generated H_f handler or abstract-device manifest entry exists.

**~~Gap B — effect validation is precomputed, not on-device.~~ ✓ Closed**
Per-member `is_gpu_dispatchable` flags are encoded in the table as the third element of each `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuple. The flag is derived from GPU-native dispatch kind, so host-bound fusion and fallback descriptors are not eligible for GPU par selection. The kernel computes `all_eligible` on-device by scanning flags — no separate CPU-precomputed `eligible[]` array. Build-overlay still validates `par` independence structurally, so runtime effect conflicts cannot arise.

**Gap C — readiness/selection is still thread-0 control flow.**
The kernel still uses `threadIdx.x == 0` to scan groups and choose the first all-ready group. Once selected, commit writeback is block-parallel: lanes clear/copy frontier state and use atomics to mark consumed edges. Correct term: "deterministic thread-0 selection with parallel device commit", not fully parallel par selection.

**Gap D — par-tier receipts partially use the device ring.**
H_f hot-region and lowerable fusion-entry par members write receipts directly from `tensor_quantale_par_group_step` to the device ring. Successful non-lowerable batched fusion par members push generic `DeviceReceipt`s into the same ring after the host-launched batch kernel completes. Hot-region fallback members still call `gpu_dispatch_region` + `drain_device_receipts` after CPU execution. Abstract-node and failed fallback members still use `queue_lattice_update` + `drain_lattice_queue`.

Gap A/D_h is closed for the implemented GPU-native categories: static H_f, generated id ≥ 8 H_f, and manifest-covered abstract-device receipts all have software and CUDA smoke coverage. Host-bound process/IO nodes remain outside GPU par selection rather than re-entering `thread::scope` from `dispatch_gpu_parallel_group`.

**Correct label for the current state:**
```text
GPU-selected parallel dispatch tier with in-kernel static/generated H_f dispatch
and manifest-covered abstract-device receipt dispatch; host-bound process/IO
members are excluded from GPU par selection
```

Not: "fully GPU-native orchestration" — CPU still handles process/IO operator dispatch and broader orchestration, but those host-bound members are no longer admitted into GPU par selection (Gap A).

---

## ~~3. CKA pattern compilation belongs at build time~~ ✓ Implemented

Build-overlay emits `assets/patterns.compiled.json` as a generated artifact. The runtime loads from it on epoch start; falls back to runtime CKA compilation when the file is absent. `cli.rs` calls `compile_and_emit_pattern_edges(".")` after `build_overlay_assets`.

---

## ~~3. `TlogRecordKind::ExplorationExpand` and `append_exploration_expand` are never called~~ ✓ Removed

Both the enum variant and the writer method were deleted from `tlog.rs`.

---

## ~~4. Tensor feedback does not persist across epoch reloads~~ ✓ Implemented

`LearningBuffer` added to `learning.rs`. Wired into `RuntimeEpoch` and both execution paths in `main.rs`. Successful topology-edge executions are buffered and flushed to `state/learned_edges.jsonl` every 10 records, on epoch reload, and at shutdown. Three tests added: flush-writes-readable-jsonl, auto-flush-at-threshold, empty-flush-is-noop.

---

## ~~5. `CompiledCkaPattern.parallel_groups` is dead weight~~ ✓ Removed

`parallel_groups: Vec<Vec<i32>>` removed from `CompiledCkaPattern`. All `compile_*` internal functions updated to drop the parameter. The par effect-safety validator (`validate_parallel_independence`) is retained — it guards the constraint, not the output artifact.
