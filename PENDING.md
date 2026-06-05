# Pending

Only remaining gaps are listed here. Completed items were removed after verification.

---

## GPU-native orchestration (in progress — PENDING.gpu.native.orchestration.md)

**Phase 0 complete** (2026-06-05):
- Design note added to ARCHITECTURE.md (`## GPU Orchestration Tiers`).
- Four Phase-0 boundary tests added to `src/runtime_parallel.rs`.
- Six runtime counters wired into `src/main.rs`; emitted as `orch_counters/shutdown` at exit.

**Phase 1 complete** (2026-06-05):
- `OrchestrationState`, `DeviceCommand`, `DeviceReceiptExt` structs added to `.cu` and mirrored as `repr(C)` in Rust.
- `OrchestrationBuffers` added to `TensorQuantaleWorld`; allocated and init'd at world construction.
- Kernels: `orchestration_state_init`, `orchestration_state_snapshot`, `device_command_ring_push`, `device_receipt_ext_ring_push`, `device_receipt_ext_drain`.
- Rust wrappers: `orch_state_init`, `orch_state_snapshot`, `push_device_command`, `drain_device_commands`, `push_device_receipt_ext`, `drain_device_receipt_ext`.
- CUDA smoke tests: `orchestration_state_init_sets_zero_state`, `command_ring_push_pop_fifo`, `receipt_ext_ring_push_pop_fifo`.

**Phase 2 complete** (2026-06-05):
- `TensorWorldBundleHost` struct packs the seven tensor/frontier device pointers for the scheduler kernel.
- `OrchStepStatus` enum + `ORCH_*` / `DISPATCH_KIND_*` constants exported from `tensor.rs`.
- `tensor_quantale_orchestrate_step` kernel: drains ext receipt ring, selects ready singleton, commits state, emits `DeviceCommand` for external nodes, writes `ORCH_*` status.
- Rust wrappers: `orchestrate_step()`, `set_dispatch_kinds()`.
- Phase-2 dispatch kind table (`dispatch_kinds`) and step status buffer (`step_status`) added to `OrchestrationBuffers`.
- CUDA smoke tests: `scheduler_selects_first_ready_singleton`, `scheduler_emits_external_command_for_process_node`, `cpu_gpu_scheduler_equivalence_singleton`.

**Phase 3 complete** (2026-06-05):
- `DeviceCommand` extended with `operator_name_id`, `timeout_ticks`, `retry_budget` fields.
- `src/orch_service.rs`: `service_external_commands` polls the command ring, resolves operator names, executes process/IO via `UniversalExecutor`, and pushes `DeviceReceiptExt` receipts back to the GPU.
- 7 unit tests covering outcome mapping, name resolution, dispatch kind routing, and the single-receipt-per-command invariant.

**Phase 4 complete** (2026-06-05):
- `OrchestrationState` extended with `star_counter` / `star_bound` (Phase-4 bounded-star fields).
- CUDA: `ControlEdge`, `EffectTable` structs; `CONTROL_OP_*` defines; `effects_independent`, `find_matching_control_edge`, `star_counter_advance` device helpers; `control_flow_advance` and `check_effects_independent` kernels.
- Rust: `ControlEdge`, `EffectTable` repr(C) structs + `DeviceRepr`; `CONTROL_OP_*` constants; `OrchestrationBuffers` extended with `control_edges`, `effect_table`, `control_op_out`; `load_control_table`, `control_flow_advance`, `check_effects_independent` wrapper methods.
- `src/control_flow_lowering.rs`: `lower_patterns_from_json` compiles CKA pattern expressions (`seq`, `par`, `choice`, `star`) into flat `ControlEdge` arrays; 8 unit tests covering all four control constructs plus effect independence.
- CUDA smoke tests: `control_flow_advance_seq_edge_matched`, `control_flow_par_effects_independent`, `control_flow_par_effects_conflict`, `control_flow_star_bounded_advances_counter`.
- Total: 182 tests passing.

**Phase 5 complete** (2026-06-05):
- `FailureClass` / `FailureAction` / `FailurePolicy` / `FailureClassifyRequest` structs added to `.cu` and mirrored as `repr(C)` in Rust.
- `FAILURE_CLASS_*`, `FAILURE_ACTION_*`, `DISPATCH_KIND_REPAIR` constants exported from `tensor.rs`.
- `OrchestrationState` extended with `consecutive_blocks`, `block_threshold`, `hard_reset_requested`, `rollback_available`, `failure_action`.
- `OrchestrationBuffers` extended with `failure_policies`, `rollback_consumed`, `rollback_active`, `failure_action_out`.
- Kernels: `failure_policy_init`, `failure_policy_classify_and_emit`, `failure_policy_set_rollback_marker`, `failure_policy_apply_rollback`.
- Rust wrappers: `failure_policy_init`, `failure_policy_classify_and_emit`, `set_rollback_marker`, `apply_rollback`.
- CUDA smoke tests: `failure_policy_retries_until_budget`, `failure_policy_blocks_after_budget`, `failure_policy_rollback_marker_round_trip`.

**Phase 6 complete** (2026-06-05):
- `LearnedDelta` struct added to `.cu` and mirrored as `repr(C)` in Rust; `LEARNED_DELTA_RING_SIZE` constant exported.
- `OrchestrationBuffers` extended with `receipt_priors`, `learned_delta_ring`, `learned_delta_head`, `learned_delta_tail`, `receipt_prior_snapshot_buf`.
- Kernels: `learned_delta_init`, `learned_delta_fold_receipt`, `learned_delta_apply`, `receipt_prior_snapshot`.
- Rust wrappers: `learned_delta_init`, `learned_delta_fold_receipt`, `learned_delta_apply`, `export_receipt_priors`.
- Receipt prior table updated on-device on success; exploration seeding reads it without CPU round-trip.
- Delta ring produced on-device; CPU service drains for durable JSONL/state persistence.
- CUDA smoke tests: `learned_delta_fold_success_updates_prior`, `learned_delta_fold_failure_does_not_update_prior`, `learned_delta_apply_updates_tensor`.

**Phase 7 complete** (2026-06-05):
- Fixed pre-existing `TensorWorldBundle` by-value kernel bug: `tensor_quantale_orchestrate_step` now takes `const TensorWorldBundle* world` (pointer), fixing CUDA_ERROR_ILLEGAL_ADDRESS in all Phase-2 scheduler tests.
- Added `orchestrate_until_wait_or_halt(max_steps)` to `TensorQuantaleWorld`: runs GPU scheduler in a burst loop, returning on `WaitExternal`, `Halted`, `Error`, or blocked/exhausted (`Continue`).
- Added `gpu_native_supervisor_loop` function in `main.rs` (`#[cfg(feature = "cuda")]`): implements the Phase-7 supervisor pattern with injected `service_fn` for external command handling; CPU no longer in per-step hot path.
- 4 previously-SIGABRTing Phase-2 tests now pass: `scheduler_selects_first_ready_singleton`, `scheduler_emits_external_command_for_process_node`, `cpu_gpu_scheduler_equivalence_singleton`, `gpu_dispatch_region_uses_device_slot_registry`.
- CUDA smoke tests: `orchestrate_step_no_longer_sigabrt`, `orchestrate_until_wait_or_halt_respects_max_steps`, `gpu_native_loop_advances_graph_state`.
- Total: 210 tests passing.

**Phase 8 complete** (2026-06-05):
- `OrchestrationState` extended with `selected_src` / `selected_dst` (Phase-8 edge tracking).
- `OrchestrationEvent` repr(C) struct added; `ORCH_TRACE_RING_SIZE` / `ORCH_EVENT_*` constants exported.
- `OrchestrationBuffers` extended with `trace_ring`, `trace_head`, `trace_tail`, `trace_drain_buf`, `trace_drain_count`, `orch_violation_out`, `replay_state`, `replay_consumed`, `replay_active`.
- Kernels: `orch_event_trace_push`, `orch_event_trace_drain`, `orch_check_no_duplicate_receipts`, `orch_check_frontier_valid`, `orch_check_no_command_without_receipt`, `orch_replay_snapshot`, `orch_replay_restore`.
- Rust wrappers: `push_trace_event`, `drain_trace_events`, `check_no_duplicate_receipts`, `check_frontier_valid`, `check_no_command_without_receipt`, `replay_snapshot`, `replay_restore`.
- CUDA smoke tests: `trace_ring_push_drain_round_trip`, `invariant_frontier_valid_passes_on_init`, `invariant_no_duplicate_receipts_passes_on_empty_ring`, `replay_snapshot_restore_is_identity`.
- Total: 214 tests passing.

---

## 1. Par-tier keeps explicit process/IO fallback orchestration on CPU

Current state is a GPU-selected parallel dispatch tier, not fully GPU-native orchestration.

```text
G_s = 1   GPU selects the par group.
G_c = 1   GPU commits consumed/active state. Readiness scans are block-parallel
          inside the par kernel; the first passing group is still selected
          deterministically by group order.
E_g = 1   Eligibility is computed on-device from per-member table flags.
D_h = 2/3 Static/generated H_f and manifest-covered abstract-device members run
          or receipt-dispatch on device. Fully device-dispatched par groups skip
          fusion lookup and host dispatch scheduling after kernel commit.
          Process/IO and unsupported fallback members remain host-bound and are
          excluded from GPU par selection.
R_d = 1   GPU-dispatched par-member successes route exactly one device-ring
          receipt. Failed or explicit process/IO host fallbacks still use the
          CPU lattice queue path.
R_k = 1   Per-member region id and dispatch kind are emitted by the kernel.
H_o = 1   Par-group node names and fusion-entry metadata are pre-resolved once
          per epoch. Fully device-dispatched groups use a device-only fast path;
          host orchestration remains only for explicit fallback/failure work.
```

Remaining intentional boundary:

1. **Host orchestration remains outside fully GPU-native scope for process/IO.**
   Process/IO dispatch, broader runtime orchestration, and fallback operator execution are still CPU-owned. This is intentional, but it means the correct label is:

```text
GPU-selected parallel dispatch tier with block-parallel readiness, in-kernel
static/generated H_f dispatch, manifest-covered abstract-device receipt dispatch,
device-only fast path for fully device-dispatched par groups, and host-bound
process/IO members excluded from GPU par selection.
```

Not:

```text
Fully GPU-native orchestration.
```
