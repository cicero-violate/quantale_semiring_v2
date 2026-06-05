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

**GPU-native SEQ/PAR/CHOICE/STAR scheduler integration complete** (2026-06-05):
- `tensor_quantale_orchestrate_step` now owns all control-flow decisions; no CPU involvement in hot-path routing.
- `OrchestrationState` extended with 7 control-flow fields: `selected_control_edge`, `selected_control_op`, `selected_control_lhs`, `selected_control_rhs`, `control_epoch`, `star_counter_epoch`, `last_block_reason`.
- `TensorWorldBundle` extended with 6 new fields: `control_edges`, `control_edge_count`, `effects`, `effect_count`, `star_counters`, `star_counter_count`.
- Device helpers added: `ready_seq`, `ready_par`, `ready_choice`, `ready_star`, `score_choice_branch`, `select_control_decision`, `commit_selected_node_or_command`.
- New kernel `star_counters_init`; per-edge star counter buffer allocated in `OrchestrationBuffers`.
- `orch_replay_snapshot` / `orch_replay_restore` extended to include star counters.
- `OrchestrationEvent` extended with 4 trace fields: `selected_control_op`, `selected_control_edge`, `branch_count`, `star_counter_val`.
- Rust: `ORCH_BLOCK_REASON_*`, `CONTROL_*`, `MAX_CONTROL_EDGES`, `STAR_COUNTERS_INIT_KERNEL` constants; `star_counters_reset()` wrapper; `load_control_table` auto-resizes and reinits star counters.
- Plan file: `plan.gpu.native.seq.par.choice.star.md` marked **Implemented**.
- Current tier label: **GPU-native orchestration with external command service** (`S_g=1 P_g=1`).

**Phase 9 complete** (2026-06-05):
- Scheduler-integrated tests added for SEQ, PAR, CHOICE, STAR_BOUNDED via `orchestrate_step`.
- `control_flow_advance` marked `#[deprecated]`; legacy Phase-4 tests suppressed with `#[allow(deprecated)]`.
- Fixed: PAR missing from `select_control_decision` priority chain.
- Fixed: STAR_EXIT now consumes the back-edge and holds frontier at exit node.
- Plan `plan.gpu.native.seq.par.choice.star.md` fully complete.
- Current tier: **GPU-native orchestration with external command service** — all nine phases done.

---

## 2. Process/IO dispatch remains an explicit external command service

Process/IO work is explicit — the GPU emits `DeviceCommand`; the CPU services it and
returns `DeviceReceiptExt`. This is intentional and correct (CPU owns OS process
execution). The boundary is now explicit rather than silent fallback.

```text
S_g=1  P_g=1  E_g=1  C_g=1  R_g=1  F_g=1  H_g=0
D_g=0 for process/IO (explicit command/receipt protocol — not a regression)
```
