# Plan: GPU-Native Seq / Par / Choice / Star Orchestration

## Status

Planned.

This plan defines the remaining work to move device-side control-flow support
from a callable side path into the GPU-owned orchestration scheduler.

The current system already has these foundations:

- `ControlEdge` device table.
- `EffectTable` device table.
- `control_flow_advance` CUDA kernel and Rust wrapper.
- `tensor_quantale_orchestrate_step` CUDA scheduler kernel.
- Persistent `OrchestrationState`.
- Device command ring.
- `DeviceReceiptExt` ring and drain path.
- GPU-native dispatch kind table.
- Reentrant mask upload.
- Legacy CPU frontier/tick/queue/split-topology paths removed.

The missing step is integration: `seq`, `par`, `choice`, and bounded `star`
control flow must become first-class branches inside the GPU-native scheduler,
with deterministic state transitions, receipt accounting, replayability, and
failure policy interaction.

---

## Target State

The runtime should satisfy:

```text
S_g = 1   GPU owns seq progression.
P_g = 1   GPU owns par readiness, independence, commit, and receipt routing.
C_g = 1   GPU owns choice scoring and branch commit.
*_g = 1   GPU owns bounded star iteration counters and termination.
H_g = 0   CPU does not choose control-flow steps.
```

Host participation remains allowed only for explicit external services:

```text
GPU scheduler -> DeviceCommand -> host service -> DeviceReceiptExt -> GPU drain
```

The CPU must not silently decide fallback routing, branch choice, par-group
membership, star continuation, or frontier advancement.

---

## Core Model

### Control Table

Each lowered control edge is device-visible:

```c
struct ControlEdge {
    int op;     // SEQ, PAR, CHOICE, STAR_BOUNDED, GATE, HALT
    int lhs;    // source / guard / loop node
    int rhs;    // destination / branch / loop body
    int guard;  // optional guard node or predicate id
    int order;  // deterministic ordering key
    int bound;  // star iteration bound or -1
};
```

### Effect Table

Each node has a compact effect summary:

```c
struct EffectTable {
    int reads;
    int writes;
    int locks;
};
```

Independence remains device-computable:

```text
independent(a,b) = disjoint(writes[a], reads[b] ∪ writes[b] ∪ locks[b])
                ∧ disjoint(writes[b], reads[a] ∪ writes[a] ∪ locks[a])
                ∧ disjoint(locks[a], locks[b])
```

### Scheduler Outcome

Each scheduler step returns exactly one state transition category:

```text
Continue | WaitExternal | Halted | Error
```

No category may partially commit work without producing either:

```text
receipt(node) OR command(node) OR blocked_reason(node)
```

---

## Invariants

### I1. Deterministic Selection

For any ready control candidates:

```text
selected = min_by(order, node_id, edge_index)(ready_candidates)
```

No warp/block scheduling nondeterminism may affect the selected control edge.

### I2. Single Commit Per Step

For singleton seq/choice/star-body steps:

```text
committed_node_count(step) <= 1
```

For par steps:

```text
committed_node_count(step) = size(selected_independent_group)
```

### I3. Single Receipt Per Committed Unit

```text
committed(node, step) => exactly_one(receipt(node, step) OR command(node, step))
```

### I4. No Silent Host Fallback

```text
unsupported(node) => emit(DeviceCommand) OR block_with_reason
```

### I5. Star Bound Safety

```text
star_counter(edge) <= edge.bound
```

A star edge with exhausted bound must not schedule its body again.

### I6. Replayability

All scheduler decisions must be reconstructable from:

```text
OrchestrationState
consumed matrix
active frontier
ControlEdge table
EffectTable table
dispatch kind table
receipt/command rings
trace ring
```

---

## Phase 1 — Control-State ABI

### Goal

Extend device state so seq/par/choice/star decisions can be persisted,
snapshotted, and replayed.

### Work

1. Extend `OrchestrationState` with compact control-flow fields:

   ```c
   int selected_control_edge;
   int selected_control_op;
   int selected_control_lhs;
   int selected_control_rhs;
   int control_epoch;
   int star_counter_epoch;
   int last_block_reason;
   ```

2. Add a device-side bounded star counter table:

   ```c
   int* star_counters; // indexed by control edge id
   ```

3. Add Rust-side buffers in `OrchestrationBuffers`.

4. Extend snapshot/restore/replay structs to include control state and star
   counters.

### Acceptance

- `orchestration_state_init` zeroes all new fields.
- Snapshot/restore round-trips new fields.
- Replay snapshot includes active, consumed, orchestration state, and star
  counters.

---

## Phase 2 — Scheduler Integration Boundary

### Goal

Make `tensor_quantale_orchestrate_step` consult the control table before using
plain tensor frontier selection.

### Work

1. Add device helper:

   ```c
   __device__ ControlDecision select_control_decision(
       const ControlEdge* edges,
       int edge_count,
       const int* active,
       const int* consumed,
       const int* star_counters,
       const EffectTable* effects,
       const float* tensor,
       const int* witness
   );
   ```

2. Return one of:

   ```text
   CONTROL_NONE
   CONTROL_SEQ_READY
   CONTROL_PAR_READY
   CONTROL_CHOICE_READY
   CONTROL_STAR_BODY_READY
   CONTROL_STAR_EXIT_READY
   CONTROL_HALT_READY
   CONTROL_BLOCKED
   ```

3. Update `tensor_quantale_orchestrate_step` order:

   ```text
   drain receipts
   select control decision
   if control decision exists: execute control branch
   else: use default GPU singleton scheduler
   ```

### Acceptance

- Existing singleton scheduler tests still pass.
- Empty control table preserves current behavior.
- Non-empty control table changes selected path only when a matching active
  control edge is ready.

---

## Phase 3 — SEQ Native Lowering

### Goal

A `SEQ(lhs, rhs)` edge advances deterministically from `lhs` to `rhs` on GPU.

### Device Semantics

```text
ready_seq(e) = active[lhs] = 1
             ∧ consumed[lhs,rhs] = 0
             ∧ dispatch_allowed(rhs)
```

Commit:

```text
consumed[lhs,rhs] = 1
active = one_hot(rhs)
selected_node = rhs
selected_control_op = SEQ
```

### Work

1. Implement `ready_seq` device helper.
2. Implement `commit_seq` device helper.
3. Integrate dispatch-kind handling:
   - `HF_DEVICE` -> execute/receipt on device.
   - `ABSTRACT_DEVICE` -> receipt on device.
   - `EXTERNAL_PROCESS` / `EXTERNAL_IO` -> emit command and return
     `WaitExternal`.
   - `UNSUPPORTED` -> block or explicit command, never silent fallback.

### Tests

- `seq_advances_active_frontier_on_device`.
- `seq_external_node_emits_device_command`.
- `seq_unsupported_node_blocks_with_reason`.
- `seq_reentrant_node_can_be_revisited_when_masked`.

### Acceptance

```text
SEQ path uses no CPU frontier/tick/project fallback.
```

---

## Phase 4 — PAR Native Lowering

### Goal

A `PAR(a,b,...)` region is selected, checked for independence, committed, and
receipted entirely on GPU when all members are device-capable or explicit
external commands.

### Device Semantics

```text
ready_par(group) = all(active[parent])
                 ∧ all(unconsumed(parent, member))
                 ∧ pairwise_independent(members)
```

Commit for device-capable members:

```text
for member in deterministic_order(group):
    consumed[parent,member] = 1
    emit receipt(member)
active = join(successors(group))
```

External member handling:

```text
external(member) => emit DeviceCommand(member), pending_receipt_count += 1
```

### Work

1. Represent par groups as control-edge spans or a compact par-group table.
2. Reuse `EffectTable` for device-side independence checks.
3. Add deterministic par selection:

   ```text
   selected_group = min_ready_group_id
   ```

4. Ensure receipt ring writes one receipt per committed device member.
5. Ensure external par members are commands, not host fallback routes.

### Tests

- `par_independent_members_commit_on_device`.
- `par_conflicting_effects_block`.
- `par_mixed_external_members_wait_external`.
- `par_member_receipts_are_unique`.
- `par_group_selection_is_deterministic`.

### Acceptance

```text
No host par dispatch scheduling remains.
Each device par member produces exactly one receipt.
```

---

## Phase 5 — CHOICE Native Lowering

### Goal

A `CHOICE(root, branch...)` evaluates branch candidates on GPU and commits one
branch deterministically.

### Device Semantics

For each branch:

```text
score(branch) = α * confidence(root, branch)
              - β * cost(root, branch)
              + γ * safety(root, branch)
              + receipt_prior(branch)
              - failure_penalty(branch)
```

Select:

```text
selected_branch = argmax(score, tie_break=min(order,node_id))
```

Commit:

```text
consumed[root, selected_branch] = 1
active = one_hot(selected_branch)
selected_control_op = CHOICE
```

### Work

1. Add device-side branch scoring helper.
2. Include receipt priors and failure counters if available.
3. Preserve deterministic tie-breaking.
4. Emit trace event with all considered branches and selected branch.

### Tests

- `choice_selects_highest_score_branch`.
- `choice_tie_breaks_by_order_then_node_id`.
- `choice_external_branch_emits_command`.
- `choice_all_branches_blocked_returns_blocked`.

### Acceptance

```text
CHOICE does not call CPU projection/reference logic.
```

---

## Phase 6 — Bounded STAR Native Lowering

### Goal

A bounded `STAR(body)` loop executes body iterations on GPU until exit condition
or bound exhaustion.

### Device Semantics

```text
ready_star_body(e) = active[lhs] = 1
                   ∧ star_counters[e] < bound
                   ∧ guard_allows_continue(e)
```

Commit body:

```text
star_counters[e] += 1
consumed[lhs,rhs] = reentrant ? keep_available : mark_consumed
active = one_hot(rhs)
selected_control_op = STAR_BOUNDED
```

Exit:

```text
star_counters[e] >= bound OR guard_blocks_continue(e)
=> active = one_hot(exit_node)
```

### Work

1. Add star counter device buffer.
2. Add reset semantics for entering a new star scope.
3. Respect reentrant mask for loop bodies.
4. Add failure-policy interaction:
   - retry inside bound when retryable.
   - block when retry budget exhausted.
   - rollback when policy requests rollback.

### Tests

- `star_iterates_until_bound`.
- `star_respects_reentrant_body`.
- `star_exits_when_bound_exhausted`.
- `star_failure_policy_blocks_after_budget`.
- `star_replay_snapshot_restores_counter`.

### Acceptance

```text
No CPU loop manages star continuation or bound checks.
```

---

## Phase 7 — Unified Commit Protocol

### Goal

All control-flow operations share one device-side commit and receipt protocol.

### Work

1. Add device helper:

   ```c
   __device__ int commit_selected_node_or_command(
       TensorWorld world,
       OrchestrationState* state,
       int src,
       int dst,
       int dispatch_kind,
       DeviceCommand* command_ring,
       DeviceReceiptExt* receipt_ring
   );
   ```

2. Standardize state updates:

   ```text
   selected_src
   selected_dst
   selected_node
   selected_control_op
   pending_external_count
   pending_receipt_count
   failure_count
   step
   ```

3. Standardize outcomes:

   ```text
   committed_device_receipt
   emitted_external_command
   blocked
   halted
   error
   ```

### Acceptance

- Seq/par/choice/star all use the same command/receipt path.
- Invariant checker detects duplicate receipts and command-without-terminal
  receipt conditions.

---

## Phase 8 — Trace and Replay

### Goal

Make seq/par/choice/star decisions reproducible and inspectable.

### Work

1. Extend trace event payload:

   ```text
   step
   selected_control_op
   selected_control_edge
   selected_node
   selected_group
   branch_count
   star_counter
   outcome
   ```

2. Extend replay snapshot/restore to include:
   - control state fields.
   - star counters.
   - active frontier.
   - consumed matrix.
   - command/receipt ring heads/tails.

3. Add deterministic replay tests for each control op.

### Acceptance

```text
recorded_trace + replay_snapshot => same selected_control_op and selected_node
```

---

## Phase 9 — Remove Side-Path Control API

### Goal

Once scheduler-integrated control-flow is stable, retire standalone side-path
control APIs that are no longer needed outside tests.

### Candidate Removals

- Standalone `control_flow_advance` host wrapper if all runtime/test coverage
  moves to `orchestrate_step`.
- Any test that validates control flow by directly calling the side kernel
  instead of observing scheduler state.

### Keep if Useful

A pure test-only control evaluator may remain if it is clearly marked as a
reference model and does not mutate runtime state.

### Acceptance

```text
Runtime control-flow entrypoint = tensor_quantale_orchestrate_step only.
```

---

## Validation Matrix

### Build

```bash
rtk cargo check --features cuda
rtk cargo check --tests --features cuda
```

### Focused Tests

```bash
rtk cargo test --features cuda control_flow -- --nocapture
rtk cargo test --features cuda orchestrate_step -- --nocapture
rtk cargo test --features cuda scheduler -- --nocapture
rtk cargo test --features cuda receipt -- --nocapture
rtk cargo test --features cuda replay -- --nocapture
```

### Runtime Smoke

```bash
QUANTALE_MAX_TICKS=4 rtk cargo run --features cuda
```

Expected runtime evidence:

```text
[gpu_native] [INFO] dispatch_kinds_uploaded ...
[gpu_native] [INFO] burst_complete status=...
[gpu_native] [INFO] external_commands_serviced count=...
[gpu_native] [INFO] supervisor_exit total_steps=...
```

---

## Definition of Done

This plan is complete when:

- `tensor_quantale_orchestrate_step` owns seq/par/choice/star decisions.
- The CPU never decides control-flow progression in the hot path.
- Seq, par, choice, and bounded star each have scheduler-integrated tests.
- Par dispatch uses device-side effect independence and emits unique receipts.
- Choice scoring is deterministic and device-owned.
- Star counters are device-owned, bounded, snapshotted, and replayable.
- External work is represented only by `DeviceCommand` / `DeviceReceiptExt`.
- Runtime smoke passes with GPU-native burst logs.

---

## Implementation Order

1. Extend orchestration state and replay snapshot for control state.
2. Integrate control-table lookup into `tensor_quantale_orchestrate_step`.
3. Implement SEQ commit branch.
4. Implement PAR group branch with device independence checks.
5. Implement CHOICE scoring branch.
6. Implement bounded STAR counter branch.
7. Unify command/receipt commit protocol.
8. Extend trace/replay/invariant checks.
9. Remove side-path control-flow API if no longer needed.
