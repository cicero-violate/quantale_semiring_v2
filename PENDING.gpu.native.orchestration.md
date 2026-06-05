# GPU-Native Orchestration — Complete

Phases 0–8 are complete.  The GPU-native SEQ/PAR/CHOICE/STAR scheduler
integration (plan.gpu.native.seq.par.choice.star.md) is also complete as of
2026-06-05.

**Current tier: GPU-native orchestration with external command service.**

```text
S_g=1  P_g=1  E_g=1  C_g=1  R_g=1  F_g=1  H_g=0
D_g=0 for process/IO (explicit DeviceCommand/DeviceReceiptExt protocol)
```

`tensor_quantale_orchestrate_step` now owns all control-flow decisions.
SEQ, PAR, CHOICE, and bounded STAR are selected and committed on-device.
The ControlEdge + EffectTable device tables drive deterministic selection.
Per-edge star counters are GPU-resident and included in replay snapshots.
Process/IO work is explicit: the GPU emits DeviceCommand; the CPU services it.

All nine phases are complete as of 2026-06-05.  Phase 9 added
scheduler-integrated tests for SEQ, PAR, CHOICE, and STAR_BOUNDED through
`orchestrate_step`; `control_flow_advance` is now `#[deprecated]`.
Two correctness fixes were applied: PAR was missing from
`select_control_decision` and STAR_EXIT did not consume its back-edge.

---

The earlier sections below document the full design for reference.

---

## Target state

A fully GPU-native orchestration design should satisfy:

```text
S_g = 1   GPU owns step scheduling and frontier progression.
P_g = 1   GPU owns par/seq/choice/star control-flow decisions.
E_g = 1   GPU owns dispatch eligibility and fallback classification.
D_g = 1   GPU dispatches all device-capable work without CPU scheduling.
C_g = 1   GPU commits tensor, frontier, receipt, and learning updates.
R_g = 1   GPU owns receipt production, routing, and fold-in.
F_g = 1   GPU owns failure classification and retry/block/rollback decisions.
H_g = 0   CPU is not in the hot orchestration path.
```

A CPU process may still exist as a supervisor, loader, logger, or external IO
service, but it must not decide the next runtime step in the hot path.

---

## Core invariants

1. **Determinism**
   Device-side orchestration must preserve the existing deterministic ordering
   rules:

   ```text
   selected = min_ready(candidate_set)
   par_selected = min { g | group_pass[g] = 1 }
   ```

2. **Single receipt per committed unit**
   Each committed node/member emits exactly one receipt:

   ```text
   committed(i) => count(receipt[i]) = 1
   ```

3. **No silent host fallback**
   Device orchestration must never silently execute CPU fallback. Unsupported
   work must become an explicit device-visible command:

   ```text
   unsupported(i) => external_command(i) ∧ pending_receipt(i)
   ```

4. **No duplicate tensor fold**
   A receipt may update tensor confidence/cost/safety exactly once:

   ```text
   receipt.valid ∧ !receipt.consumed => fold(receipt), receipt.consumed = 1
   ```

5. **Safety gates remain first-class**
   Gate decisions must be represented in device state, not bypassed by kernel
   shortcuts.

---

## Phase 0 — Baseline and freeze current contract

### Goal

Create a hard baseline for the current GPU-selected par tier before changing the
runtime orchestration ABI.

### Work

1. Add a design note that distinguishes:
   - GPU-selected parallel dispatch tier.
   - GPU-native par dispatch.
   - Fully GPU-native orchestration.

2. Add explicit tests for current boundaries:
   - Process/IO members are excluded from GPU par selection.
   - Host fallback remains CPU-owned.
   - Fully device-dispatched par groups do not enter host dispatch scheduling.
   - GPU-dispatched members produce exactly one device-ring receipt.

3. Add runtime counters:

   ```text
   gpu_selected_groups
   gpu_device_only_groups
   host_fallback_groups
   device_ring_receipts
   cpu_queue_receipts
   external_io_commands
   ```

### Acceptance

```text
cargo test
cuda_smoke passes
PENDING.md accurately states the current non-fully-native label
```

---

## Phase 1 — Device orchestration state block

### Goal

Move runtime step state into a persistent GPU-resident orchestration block.

### New device structures

```c
typedef struct OrchestrationState {
    int step;
    int halted;
    int blocked;
    int current_frontier_epoch;
    int selected_group;
    int selected_node;
    int pending_external_count;
    int pending_receipt_count;
    int failure_count;
    int rollback_requested;
} OrchestrationState;
```

```c
typedef struct DeviceCommand {
    int valid;
    int command_id;
    int node_id;
    int src;
    int dst;
    int dispatch_kind;
    int payload_offset;
    int payload_len;
} DeviceCommand;
```

```c
typedef struct DeviceReceiptExt {
    int valid;
    int consumed;
    int command_id;
    int node_id;
    int src;
    int dst;
    int outcome;
    int receipt_kind;
    int output_flags;
    float latency;
} DeviceReceiptExt;
```

### Work

1. Add device buffers:
   - `OrchestrationState` singleton.
   - Device command ring.
   - Extended receipt ring.
   - Pending external command counters.

2. Add kernels:
   - `orchestration_state_init`.
   - `orchestration_state_snapshot` for tests/debug.
   - `device_command_ring_push`.
   - `device_receipt_ext_ring_push`.
   - `device_receipt_ext_drain`.

3. Add Rust wrappers in `TensorQuantaleWorld`.

### Acceptance

- State block initializes on GPU.
- Command and receipt rings pass FIFO, overflow, and wraparound tests.
- Existing par-tier behavior remains unchanged.

---

## Phase 2 — Device-owned scheduler kernel

### Goal

Introduce a single GPU scheduler kernel that chooses the next action instead of
letting the CPU runtime loop decide every step.

### Kernel

```c
extern "C" __global__ void tensor_quantale_orchestrate_step(
    TensorWorld world,
    OrchestrationState* state,
    DeviceCommand* command_ring,
    DeviceReceiptExt* receipt_ring,
    const DispatchTable* dispatch_table,
    const PatternControlTable* control_table
);
```

### Responsibilities

For each orchestration step:

1. Drain completed device receipts.
2. Apply receipt tensor updates.
3. Evaluate active frontier.
4. Select one of:
   - fully device-native par group,
   - single GPU-native node,
   - external process/IO command,
   - block/retry/rollback/halt.
5. Commit consumed/active state.
6. Emit receipt or command.

### Scheduling equation

```text
next_action = argmin_ordered(
    ready_gpu_par_groups,
    ready_gpu_singletons,
    ready_external_commands,
    control_fallbacks
)
```

### Work

1. Refactor current `par_group_step` into a callable device routine or a kernel
   branch inside `tensor_quantale_orchestrate_step`.

2. Add singleton GPU-node scheduling using the same readiness scoring as the
   frontier/project kernels.

3. Encode dispatch kind table on device:

   ```text
   dispatch_kind[node] ∈ {
       HF_DEVICE,
       ABSTRACT_DEVICE,
       FUSION_ENTRY,
       EXTERNAL_PROCESS,
       EXTERNAL_IO,
       UNSUPPORTED
   }
   ```

4. Return only a small CPU-visible status:

   ```text
   ORCH_CONTINUE
   ORCH_WAIT_EXTERNAL
   ORCH_HALTED
   ORCH_ERROR
   ```

### Acceptance

- A pure GPU graph can run multiple steps without CPU deciding each step.
- Deterministic selected nodes match existing CPU-loop behavior on equivalent
  graphs.
- Par group behavior remains bit-for-bit equivalent for selected group/member
  decisions.

---

## Phase 3 — External command protocol for process/IO

### Goal

Make process/IO host work explicit and GPU-owned at the orchestration level.
The GPU decides that external work is needed; the CPU only services a command
queue and returns receipts.

### Model

```text
GPU scheduler -> DeviceCommand ring -> CPU/IO service -> DeviceReceiptExt ring -> GPU drain
```

The CPU no longer decides fallback routing. It only executes commands that the
GPU has emitted.

### Work

1. Define command ABI:

   ```text
   command_id
   node_id
   src
   dst
   operator_name_id
   input_payload_ref
   timeout_policy
   retry_policy
   ```

2. Add a host service loop:
   - Poll command ring.
   - Resolve operator by `operator_name_id`.
   - Execute process/IO.
   - Push extended receipt into device ring.

3. Add backpressure semantics:

   ```text
   command_ring_full => scheduler_state = WAIT_EXTERNAL_CAPACITY
   pending_external_count > 0 => scheduler may continue only independent work
   ```

4. Add timeout semantics:
   - Device-visible deadline ticks.
   - Host-visible timeout enforcement.
   - Device-side timeout receipt if service fails to respond.

### Acceptance

- Process/IO work is initiated by GPU command emission.
- CPU fallback queue is removed from the orchestration hot path.
- Tests prove command/receipt pairing:

   ```text
   every command_id has exactly one terminal receipt
   ```

---

## Phase 4 — Device-side control-flow lowering

### Goal

Represent pattern control flow directly on GPU so the scheduler can orchestrate
`seq`, `par`, `choice`, and bounded `star` without CPU interpretation.

### Device tables

```text
ControlOp = SEQ | PAR | CHOICE | STAR_BOUNDED | GATE | HALT
ControlEdge = { op, lhs, rhs, guard, order, bound }
EffectTable = { reads, writes, locks, safety_class }
```

### Work

1. Lower compiled patterns into compact device tables.
2. Encode effect independence checks for par eligibility.
3. Encode choice guards and gate predicates as device-evaluable predicates.
4. Add bounded-star progress counters in `OrchestrationState`.
5. Add tests for each control construct.

### Acceptance

- Existing compiled patterns can be replayed from device tables.
- CPU pattern interpreter is not required for hot orchestration.
- Device and CPU lowering agree on selected edge sequences.

---

## Phase 5 — Device-native failure policy

### Goal

Move retry/block/rollback/hard-reset decisions from CPU policy into GPU-visible
state and kernels.

### Device policy

```text
FailureClass = SPAWN_FAILURE | TIMEOUT | SAFETY | CONTRACT | GPU_ERROR | UNKNOWN
FailureAction = RETRY | BLOCK | ROLLBACK | HALT | EXTERNAL_REPAIR
```

### Work

1. Add device failure classifier for receipt outcomes.
2. Add retry budget counters by node and edge.
3. Add rollback markers for consumed/active state.
4. Add repair-command emission for failures requiring CPU/LLM repair.
5. Move hard-reset trigger into GPU-visible policy state.

### Acceptance

- Consecutive block logic is represented in `OrchestrationState`.
- Retry/block/rollback choices are deterministic on device.
- CPU no longer decides failure action except when servicing explicit external
  repair commands.

---

## Phase 6 — Device-side learning and exploration updates

### Goal

Move receipt-prior updates, learned-edge deltas, and exploration priors into GPU
state where possible.

### Work

1. Add GPU-resident receipt prior table.
2. Add learned-edge delta ring:

   ```text
   LearnedDelta = { src, dst, confidence_delta, cost_delta, safety_delta }
   ```

3. Fold successful execution receipts into learned deltas on device.
4. Keep persistence as CPU service:
   - GPU emits learned-delta snapshots.
   - CPU writes durable JSONL/state files.

### Acceptance

- Runtime scoring sees updated priors without CPU intervention.
- Durable persistence remains eventually consistent through snapshot/export.

---

## Phase 7 — Host loop demotion

### Goal

Demote the CPU runtime loop into a supervisor and service multiplexer.

### New CPU responsibilities

1. Load assets and initialize GPU state.
2. Service external command rings.
3. Persist logs, snapshots, and learned deltas.
4. Handle terminal errors and process shutdown.
5. Optionally display observability events.

### Removed CPU responsibilities

1. Per-step action selection.
2. Par-group selection.
3. Receipt route selection.
4. Retry/block/rollback decision-making.
5. Hot-path fusion lookup.
6. Hot-path tensor update queueing.

### Acceptance

The main runtime loop becomes:

```rust
loop {
    let status = world.orchestrate_until_wait_or_halt(max_device_steps)?;
    match status {
        Continue => continue,
        WaitExternal => service_external_commands(),
        Halted => break,
        Error => snapshot_and_exit(),
    }
}
```

---

## Phase 8 — Observability, debug, and replay

### Goal

Make GPU-native orchestration debuggable and replayable.

### Work

1. Add device event trace ring:

   ```text
   OrchestrationEvent = {
       step,
       event_kind,
       selected_node,
       selected_group,
       src,
       dst,
       outcome
   }
   ```

2. Add host drain for trace ring.
3. Add deterministic replay mode from snapshots.
4. Add invariant checker kernels:
   - no duplicate receipts,
   - no consumed edge without receipt,
   - no active frontier corruption,
   - no command without terminal receipt.

### Acceptance

- A failing GPU-native run can be replayed from a saved state snapshot.
- Trace output explains scheduler decisions without CPU-side reconstruction.

---

## Phase 9 — Decommission legacy CPU hot-path pieces

### Goal

Remove or quarantine CPU-only orchestration paths after GPU-native parity is
proven.

### Work

1. Keep CPU orchestration behind a feature flag:

   ```text
   --features legacy-cpu-orchestration
   ```

2. Move CPU receipt queue to compatibility module.
3. Remove CPU par hot-path dispatch from default runtime.
4. Keep external service code for process/IO commands.
5. Update architecture docs and labels.

### Acceptance

Default runtime uses GPU-native orchestration for scheduling and state mutation.
CPU orchestration remains only as fallback/debug compatibility.

---

## Test matrix

### Unit tests

```text
orchestration_state_init_sets_zero_state
command_ring_push_pop_fifo
receipt_ext_ring_push_pop_fifo
scheduler_selects_first_ready_singleton
scheduler_selects_first_ready_par_group
scheduler_emits_external_command_for_process_io
scheduler_waits_on_external_dependency
receipt_drain_consumes_once
failure_policy_retries_until_budget
failure_policy_blocks_after_budget
```

### CUDA smoke tests

```text
gpu_orchestrates_pure_hf_graph_to_halt
gpu_orchestrates_abstract_device_graph_to_halt
gpu_orchestrates_par_group_without_host_dispatch
gpu_emits_process_command_and_resumes_after_receipt
gpu_handles_timeout_receipt
gpu_trace_ring_records_selected_steps
```

### Equivalence tests

```text
cpu_gpu_scheduler_equivalence_seq
cpu_gpu_scheduler_equivalence_par
cpu_gpu_scheduler_equivalence_choice
cpu_gpu_scheduler_equivalence_bounded_star
cpu_gpu_receipt_fold_equivalence
```

### Invariant tests

```text
no_duplicate_receipts
no_unconsumed_terminal_receipts
no_command_without_terminal_receipt
frontier_has_valid_node_ids
consumed_edges_match_committed_decisions
```

---

## Migration order

Recommended implementation sequence:

1. Add device orchestration state and command/receipt rings.
2. Add scheduler kernel in observe-only mode.
3. Run CPU and GPU schedulers side-by-side and compare decisions.
4. Enable GPU scheduler for pure GPU graphs.
5. Add external command protocol for process/IO.
6. Enable GPU scheduler for mixed GPU + external graphs.
7. Move failure policy to GPU-visible state.
8. Move learning/exploration priors to GPU state.
9. Demote CPU loop to supervisor/service role.
10. Put legacy CPU orchestration behind a feature flag.

---

## Definition of done

Fully GPU-native orchestration is complete only when:

```text
1. GPU owns next-step scheduling.
2. GPU owns par/seq/choice/star control progression.
3. GPU owns receipt routing and tensor fold-in.
4. GPU owns retry/block/rollback/halt decisions.
5. CPU process/IO work is invoked only via GPU-emitted commands.
6. CPU does not choose fallback routes in the hot path.
7. Pure GPU graphs run to halt without CPU step decisions.
8. Mixed graphs run through GPU command/receipt protocol.
9. Deterministic replay exists for scheduler decisions.
10. Legacy CPU orchestration is optional compatibility, not default behavior.
```

---

## Non-goals

The following are not required for fully GPU-native orchestration:

1. Running arbitrary OS process execution on GPU.
2. Removing the CPU process entirely.
3. Persisting files directly from GPU.
4. Replacing external APIs with CUDA kernels.
5. Making unsafe or unsupported operators GPU-native by default.

The CPU may remain as a service provider. The key requirement is that the GPU
owns orchestration decisions and treats CPU work as explicit asynchronous
commands with device-visible receipts.
