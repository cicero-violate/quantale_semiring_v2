# Completed: Delete Legacy Code

## Context

GPU-native orchestration is now the default CUDA runtime path. The host remains
as an external service provider for GPU-emitted `DeviceCommand`s, but it no
longer owns hot-path scheduling, fallback routing, par dispatch, or receipt
application in the default path.

This file lists code that can likely be deleted once we decide to remove the
`legacy-cpu-orchestration` compatibility feature and stop supporting the
non-CUDA CPU scheduler path as a runtime mode.

## Highest-Confidence Deletions

### 1. `legacy-cpu-orchestration` feature flag

Current location:

- `Cargo.toml`
- `src/main.rs`
- `src/runtime_epoch.rs`
- `src/runtime_dispatch.rs`
- `src/runtime_parallel.rs`
- `src/runtime_reset.rs`

Delete when:

- We no longer need `rtk cargo check --features legacy-cpu-orchestration,cuda`.
- We accept that CPU orchestration is not a compatibility/debug runtime.

Expected deletion:

- Remove `legacy-cpu-orchestration = []` from `Cargo.toml`.
- Remove every `#[cfg(feature = "legacy-cpu-orchestration")]` branch.
- Remove every `#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]`
  branch if non-CUDA runtime orchestration is also being retired.

Verification after deletion:

```bash
rtk cargo check --features cuda
rtk cargo test --features cuda
rtk cargo run --features cuda
```

### 2. `src/runtime_parallel.rs`

Why it is legacy:

- It implements the old CPU par-group dispatch bridge.
- It still routes host fallback receipts through `queue_lattice_update`.
- Default CUDA runtime no longer calls this module.

Delete when:

- `legacy-cpu-orchestration` is removed.

Likely follow-up edits:

- Remove `mod runtime_parallel;` and imports from `src/main.rs`.
- Remove par-tier counters from `src/main.rs`:
  - `gpu_selected_groups`
  - `gpu_device_only_groups`
  - `host_fallback_groups`
  - `device_ring_receipts`
  - `cpu_queue_receipts`
  - `external_io_commands`
- Move any still-useful tests to GPU-native tests only if they cover current
  behavior.

### 3. CPU frontier/runtime loop in `src/main.rs`

Why it is legacy:

- The default CUDA path exits through `gpu_native_supervisor_loop`.
- The large CPU loop still contains:
  - reload polling
  - CPU `frontier_step`
  - CPU executor dispatch
  - CPU hard reset policy
  - CPU lattice queue updates
  - legacy par tier

Delete when:

- Non-CUDA runtime mode is explicitly retired, or moved into a separate example
  binary for tests/debugging.

Candidate deletion boundary:

- Everything behind:

```rust
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
```

Keep:

- `gpu_native_supervisor_loop`.
- CLI handling.
- epoch construction needed by GPU-native runtime.
- host external service callback using `orch_service::service_external_commands`.

### 4. `src/runtime_reset.rs`

Why it is legacy:

- Implements CPU hard-reset behavior around the CPU frontier loop.
- GPU-native runtime now exits blocked/no-progress states through supervisor
  state and should eventually use device-visible failure policy/rollback.

Delete when:

- CPU runtime loop is removed.
- Hard reset is either device-native or handled by a smaller GPU supervisor
  restart function.

### 5. Most of `src/runtime_dispatch.rs`

Why it is legacy:

- `execute_active_node_blocking`
- `queue_execution_lattice_updates`
- `record_learning_edges`
- `update_execution_receipt_priors`
- `apply_hot_dispatch_if_needed`

These are CPU-frontier-loop helpers. GPU-native orchestration should instead
use:

- `orch_service::service_external_commands`
- `drain_device_receipt_ext`
- learned delta / receipt prior device paths

Delete when:

- CPU frontier loop is gone.

Possible survivors:

- Small utility helpers may be kept only if still used outside CPU runtime.

## Completed Medium-Confidence Deletions

### 6. Host-side lattice receipt queue

Current location:

- `TensorQuantaleWorld::event_queue`
- `TensorQuantaleWorld::queue_lattice_update`
- `TensorQuantaleWorld::drain_lattice_queue`
- CUDA `tensor_quantale_drain_queue` path, if no longer referenced

Why it is legacy:

- It is the old CPU receipt queue path.
- GPU-native external work now returns through `DeviceReceiptExt`.
- GPU-native device work returns through device receipt rings.

Do not delete until:

- Tests that intentionally validate tensor receipt math are migrated to direct
  GPU receipt/ext receipt helpers.
- `tests/tensor_quantale.rs` and any CPU-only tests no longer call it.

Migration target:

- Replace CPU queue tests with:
  - `push_device_receipt_ext` + `drain_device_receipt_ext`
  - `push_device_receipt` + `drain_device_receipts`

### 7. CPU `frontier_step`/`tick` wrappers

Current location:

- `TensorQuantaleWorld::frontier_step`
- `TensorQuantaleWorld::tick`
- `FRONTIER_STEP_KERNEL`
- `TICK_KERNEL`

Why it may be legacy:

- GPU-native scheduler uses `orchestrate_step` /
  `orchestrate_until_wait_or_halt`, not host-driven `frontier_step`.
- Some tests still compare GPU scheduler selection to `frontier_step`.

Do not delete until:

- Equivalence tests are rewritten to compare against fixed expected decisions
  or a pure Rust reference projection that does not mutate runtime state.
- No runtime path calls `frontier_step`.

### 8. Split topology runtime views

Status: implemented.

Landed:

- Removed `SplitTopologyRuntime`, `ControlTopologyRuntime`, and `HotTopologyRuntime` from `src/topology.rs`.
- Removed `SystemConfig::split_topology` and startup/reload loading of `assets/topology.control.json` and `assets/topology.hot.json`.
- Stopped `topology_core::build_overlay_assets` from emitting split topology view artifacts.
- Deleted tracked split topology artifacts:
  - `assets/topology.control.json`
  - `assets/topology.hot.json`
- Migrated hot-region validation tests to active runtime inputs:
  - `assets/topology.generated.json`
  - `assets/regions.hot.json`


## Keep For Now

### `src/orch_service.rs`

Keep. This is not legacy CPU scheduling. It is the current host service layer:

```text
GPU scheduler -> DeviceCommand -> host service -> DeviceReceiptExt -> GPU drain
```

### Operator Python files

Keep. External IO/process nodes still execute through Python/service scripts,
but only after GPU emits explicit commands.

### CPU math/reference tests

Keep unless they are specifically tied to the old scheduler. CPU reference
tests are still useful for semiring laws and deterministic projection checks.

### `assets/operators.json`

Keep. It is still the source of truth for external process/IO metadata and
dispatch-kind ingress.

## Suggested Deletion Order

1. Remove `legacy-cpu-orchestration` feature and `src/runtime_parallel.rs`.
2. Remove legacy par-tier block/counters from `src/main.rs`.
3. Remove CPU runtime loop from `src/main.rs`, leaving only GPU-native runtime.
4. Remove `src/runtime_reset.rs`.
5. Remove CPU-frontier helpers from `src/runtime_dispatch.rs`.
6. Migrate tests off `queue_lattice_update` / `drain_lattice_queue`.
7. Remove host-side lattice queue from `TensorQuantaleWorld`.
8. Re-evaluate `frontier_step`, `tick`, and split topology views.

## Acceptance Criteria

Deletion is complete when:

- `rg "legacy-cpu-orchestration"` returns no code references.
- `rg "queue_lattice_update|drain_lattice_queue"` returns no runtime references.
- Default runtime contains no CPU scheduler loop.
- GPU-native run still shows:

```text
[gpu_native] [INFO] dispatch_kinds_uploaded ...
[gpu_native] [INFO] burst_complete ...
[gpu_native] [INFO] external_commands_serviced count=...
```

- Verification passes:

```bash
rtk cargo check --features cuda
rtk cargo test --features cuda
rtk cargo run --features cuda
```

## Implementation Status

Status: high-confidence and medium-confidence legacy deletion pass implemented.

Landed:

- Removed the `legacy-cpu-orchestration` feature flag from `Cargo.toml`.
- Removed the legacy CPU runtime loop from `src/main.rs`; the binary now enters the GPU-native supervisor path directly.
- Deleted legacy CPU orchestration helper modules:
  - `src/runtime_dispatch.rs`
  - `src/runtime_parallel.rs`
  - `src/runtime_reset.rs`
- Simplified `src/runtime_epoch.rs` to the GPU-native epoch construction surface.
- Migrated tests off the host-side lattice receipt queue.
- Removed host-side lattice receipt queue state and APIs:
  - `TensorQuantaleWorld::event_queue`
  - `TensorQuantaleWorld::queue_lattice_update`
  - `TensorQuantaleWorld::drain_lattice_queue`
  - `tensor_quantale_drain_queue`
  - `ExecutionReceipt`
- Migrated tests off `frontier_step` and `tick`.
- Removed host-driven frontier/tick compatibility APIs and kernels:
  - `TensorQuantaleWorld::frontier_step`
  - `TensorQuantaleWorld::tick`
  - `tensor_quantale_frontier_step`
  - `tensor_quantale_tick`

Verification completed:

```bash
rtk cargo check --features cuda
rtk cargo check --tests --features cuda
rtk cargo test --features cuda dispatch_kind -- --nocapture
rtk cargo test --features cuda scheduler_emits_external_command -- --nocapture
rtk cargo test --features cuda default_dispatch_table_reaches_market_feed_external_command -- --nocapture
rtk cargo test --features cuda receipt_ext_ring_push_pop_fifo -- --nocapture
rtk cargo test --features cuda gpu_native_scheduler_advances_active_state -- --nocapture
rtk cargo test --features cuda gpu_closure_then_native_scheduler_advances_frontier -- --nocapture
QUANTALE_MAX_TICKS=4 rtk cargo run --features cuda
```

Runtime evidence:

```text
[gpu_native] [INFO] dispatch_kinds_uploaded hf_device=10 abstract_device=18 external_process=25 external_io=1 unsupported=7
[gpu_native] [INFO] burst_complete status=WaitExternal ...
[gpu_native] [INFO] external_commands_serviced count=1
[gpu_native] [INFO] supervisor_exit total_steps=4
```

Remaining deferred cleanup:

- None in this deletion plan. Split topology view artifacts have been retired.

## Final Implementation Status

Status: implemented.

Additional landed cleanup:

- Retired split topology runtime views as active runtime/generated artifacts.
- Removed remaining code/test/generator references to `topology.control.json`, `topology.hot.json`, `SplitTopologyRuntime`, `split_topology`, and `SYNTHETIC_HOT_NODES`.
- Deleted tracked `assets/topology.control.json` and `assets/topology.hot.json`.

Additional verification completed:

```bash
rtk cargo check --features cuda
rtk cargo check --tests --features cuda
```

Active-code cleanup checks now return no matches for:

```text
legacy-cpu-orchestration
runtime_parallel
runtime_reset
runtime_dispatch
queue_lattice_update
drain_lattice_queue
event_queue
frontier_step
tensor_quantale_tick
topology.control.json
topology.hot.json
split_topology
SplitTopologyRuntime
SYNTHETIC_HOT_NODES
```

## Closure Status

Status: closed. No pending implementation work remains in this deletion plan.

The active runtime, tests, CUDA kernels, and topology generator no longer reference the retired legacy surfaces listed in this document. The file is retained as the completion record for the deletion campaign, not as an active pending work queue.

Post-closure cleanup check:

```text
legacy-cpu-orchestration: none in active code
runtime_parallel/runtime_dispatch/runtime_reset: deleted
queue_lattice_update/drain_lattice_queue/event_queue: deleted
frontier_step/tick compatibility path: deleted
topology.control.json/topology.hot.json split views: deleted
SplitTopologyRuntime/SystemConfig::split_topology: deleted
```
