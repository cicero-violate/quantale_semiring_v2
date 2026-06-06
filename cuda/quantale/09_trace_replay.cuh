// ── Phase-8: observability, debug, and replay ─────────────────────────────────
//
// OrchestrationEvent: one record in the device trace ring.  The scheduler
//   emits events that explain every decision without CPU reconstruction.
// orch_event_trace_push: push one event derived from the current state.
// orch_event_trace_drain: copy all pending events to an output buffer.
// orch_check_no_duplicate_receipts: scan for consumed receipts sharing a
//   command_id — a receipt-fold-duplication invariant.
// orch_check_frontier_valid: all entries in active[] are 0 or 1.
// orch_check_no_command_without_receipt: every valid command in cmd_ring has
//   a consumed receipt in ext_ring.
// orch_replay_snapshot: block-parallel copy of state + consumed + active for
//   deterministic replay.
// orch_replay_restore: restore a previously saved replay snapshot.

#define ORCH_EVENT_STEP_COMMITTED  0  // GPU-native node dispatched
#define ORCH_EVENT_WAIT_EXTERNAL   1  // external command emitted
#define ORCH_EVENT_HALTED          2  // halt node reached
#define ORCH_EVENT_BLOCKED         3  // no ready node found
#define ORCH_EVENT_RECEIPT_DRAINED 4  // receipt folded from ext ring

#define ORCH_TRACE_RING_SIZE 512

struct OrchestrationEvent {
    int step;
    int event_kind;           // ORCH_EVENT_* constant
    int selected_node;
    int selected_group;
    int src;
    int dst;
    int outcome;
    // Phase 8 — control trace fields
    int selected_control_op;  // CONTROL_OP_* or -1
    int selected_control_edge;// ControlEdge table index or -1
    int branch_count;         // number of par members or choice candidates considered
    int star_counter_val;     // star counter epoch at commit time, or 0
};

// Push one event built from the current OrchestrationState into the trace ring.
// Thread 0, block 0 only.
extern "C" __global__ void orch_event_trace_push(
    const OrchestrationState* state,
    int                       event_kind,
    int                       outcome,
    OrchestrationEvent*       trace_ring,
    int*                      trace_tail,
    const int*                trace_head,
    int                       ring_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!trace_ring || !trace_tail || !trace_head || ring_size <= 0) return;
    int t = *trace_tail, h = *trace_head;
    if (t - h >= ring_size) return; // ring full — drop oldest-first on next push
    OrchestrationEvent ev;
    ev.step                  = state ? state->step                  : 0;
    ev.event_kind            = event_kind;
    ev.selected_node         = state ? state->selected_node         : -1;
    ev.selected_group        = state ? state->selected_group        : -1;
    ev.src                   = state ? state->selected_src          : -1;
    ev.dst                   = state ? state->selected_dst          : -1;
    ev.outcome               = outcome;
    ev.selected_control_op   = state ? state->selected_control_op   : -1;
    ev.selected_control_edge = state ? state->selected_control_edge : -1;
    ev.branch_count          = 0; // filled by caller when known
    ev.star_counter_val      = state ? state->star_counter_epoch    : 0;
    trace_ring[t % ring_size] = ev;
    *trace_tail = t + 1;
}

// Drain all pending trace events into out_buf.  Advances *trace_head.
// Writes the number of events copied to *out_count.  Thread 0, block 0 only.
extern "C" __global__ void orch_event_trace_drain(
    OrchestrationEvent*       trace_ring,
    int*                      trace_head,
    const int*                trace_tail,
    int                       ring_size,
    OrchestrationEvent*       out_buf,
    int*                      out_count,
    int                       max_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *trace_head;
    int t = *trace_tail;
    int count = 0;
    while (h != t && count < max_count) {
        out_buf[count] = trace_ring[h % ring_size];
        h++;
        count++;
    }
    *trace_head = h;
    *out_count  = count;
}

// Invariant: no two consumed DeviceReceiptExt entries share a (command_id, node_id).
// Writes 1 to *violation_out if a duplicate is found, 0 otherwise.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_no_duplicate_receipts(
    const DeviceReceiptExt* ring,
    int                     size,
    int*                    violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int i = 0; i < size; ++i) {
        if (!ring[i].valid || !ring[i].consumed) continue;
        for (int j = i + 1; j < size; ++j) {
            if (!ring[j].valid || !ring[j].consumed) continue;
            if (ring[i].command_id == ring[j].command_id
                    && ring[i].node_id == ring[j].node_id) {
                *violation_out = 1;
                return;
            }
        }
    }
}

// Invariant: every entry in active[] is 0 or 1.
// Writes 1 to *violation_out if a corrupt value is found.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_frontier_valid(
    const int* active,
    int        n,
    int*       violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int i = 0; i < n; ++i) {
        if (active[i] != 0 && active[i] != 1) {
            *violation_out = 1;
            return;
        }
    }
}

// Invariant: every valid DeviceCommand in cmd_ring has a consumed receipt in
// ext_ring with a matching command_id.
// Writes 1 to *violation_out if a command without a terminal receipt is found.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_no_command_without_receipt(
    const DeviceCommand*    cmd_ring,
    int                     cmd_size,
    const DeviceReceiptExt* ext_ring,
    int                     ext_size,
    int*                    violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int j = 0; j < cmd_size; ++j) {
        if (!cmd_ring[j].valid) continue;
        int cid = cmd_ring[j].command_id;
        int found = 0;
        for (int i = 0; i < ext_size; ++i) {
            if (ext_ring[i].valid && ext_ring[i].consumed
                    && ext_ring[i].command_id == cid) {
                found = 1;
                break;
            }
        }
        if (!found) {
            *violation_out = 1;
            return;
        }
    }
}

// Deterministic replay: block-parallel snapshot of state + consumed + active +
// per-edge star counters.
extern "C" __global__ void orch_replay_snapshot(
    const OrchestrationState* state,
    OrchestrationState*       out_state,
    const int*                consumed,
    int*                      out_consumed,
    const int*                active,
    int*                      out_active,
    const int*                star_counters,
    int*                      out_star_counters,
    int                       star_counter_count
) {
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        out_consumed[i] = consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        out_active[i] = active[i];
    if (star_counters && out_star_counters)
        for (int i = tid; i < star_counter_count; i += blockDim.x)
            out_star_counters[i] = star_counters[i];
    __syncthreads();
    if (tid == 0 && state && out_state)
        *out_state = *state;
}

// Deterministic replay: restore state + consumed + active + star counters from
// a replay snapshot.
extern "C" __global__ void orch_replay_restore(
    OrchestrationState*       state,
    const OrchestrationState* snap_state,
    int*                      consumed,
    const int*                snap_consumed,
    int*                      active,
    const int*                snap_active,
    int*                      star_counters,
    const int*                snap_star_counters,
    int                       star_counter_count
) {
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        consumed[i] = snap_consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        active[i] = snap_active[i];
    if (star_counters && snap_star_counters)
        for (int i = tid; i < star_counter_count; i += blockDim.x)
            star_counters[i] = snap_star_counters[i];
    __syncthreads();
    if (tid == 0 && state && snap_state)
        *state = *snap_state;
}

