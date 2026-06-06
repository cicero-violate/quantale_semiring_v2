// ── Device-side receipts ──────────────────────────────────────────────────────
//
// DeviceReceipt is the GPU-native receipt produced by the hot execution path.
// It is written entirely on-device by tensor_quantale_gpu_dispatch and drained
// by tensor_quantale_drain_device_receipts without any CPU hop.

struct DeviceReceipt {
    int region_id;    // GPU region that produced this receipt
    int src;          // src node id from the dispatch decision
    int dst;          // dst node id
    int outcome;      // 0=success, 1=failure, 2=timeout, 3=safety_violation
    float latency;    // relative execution latency hint (0=unknown)
    int valid;        // 1 if this ring slot is populated
    int output_flags; // bitmask of written slot indices
};

#define DEVICE_RECEIPT_RING_SIZE 256

// ── Phase-1 orchestration state block ────────────────────────────────────────
//
// OrchestrationState: persistent GPU-resident step scheduler state.
// DeviceCommand: GPU → CPU/IO service command (Phase 3 protocol).
// DeviceReceiptExt: extended receipt returned from CPU/IO services (Phase 3).
//
// These structures are allocate-on-device now and remain zeroed/unused until
// Phase 2+ kernels begin writing to them.  Adding them here locks the ABI so
// later kernels can be added without changing struct layouts.

#define DEVICE_COMMAND_RING_SIZE     64
#define DEVICE_RECEIPT_EXT_RING_SIZE 256

struct OrchestrationState {
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
    int star_bound;     // Phase 4: current star bound (0 = no star active)
    int consecutive_blocks;    // Phase 5: count of consecutive blocked scheduler steps
    int block_threshold;       // Phase 5: hard-reset threshold in consecutive blocks (0=disabled)
    int hard_reset_requested;  // Phase 5: set to 1 by failure policy when HALT fires
    int rollback_available;    // Phase 5: 1 when a rollback marker snapshot is saved
    int failure_action;        // Phase 5: last FAILURE_ACTION_* decision (observability)
    int selected_src;          // Phase 8: src of the last committed edge (for trace)
    int selected_dst;          // Phase 8: dst of the last committed edge (for trace)
    // Phase 1 (new plan): control-state ABI fields
    int selected_control_edge;  // index into ControlEdge table, or -1
    int selected_control_op;    // CONTROL_OP_* for last committed control op, or -1
    int selected_control_lhs;   // lhs of selected control edge, or -1
    int selected_control_rhs;   // rhs of selected control edge, or -1
    int control_epoch;          // incremented on each control decision commit
    int star_counter_epoch;     // incremented when any per-edge star counter advances
    int last_block_reason;      // ORCH_BLOCK_REASON_* code
};

struct DeviceCommand {
    int valid;
    int command_id;
    int node_id;
    int src;
    int dst;
    int dispatch_kind;
    int operator_name_id;  // index into host operator name table
    int timeout_ticks;     // deadline in scheduler ticks; 0 = no timeout
    int retry_budget;      // remaining retries; 0 = no retry
    int payload_offset;    // reserved for Phase 5+
    int payload_len;       // reserved for Phase 5+
};

struct DeviceReceiptExt {
    int   valid;
    int   consumed;
    int   command_id;
    int   node_id;
    int   src;
    int   dst;
    int   outcome;
    int   receipt_kind;
    int   output_flags;
    float latency;
};

// ── Phase-1 orchestration state kernels ───────────────────────────────────────

extern "C" __global__ void orchestration_state_init(OrchestrationState* state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    state->step                  = 0;
    state->halted                = 0;
    state->blocked               = 0;
    state->current_frontier_epoch = 0;
    state->selected_group        = -1;
    state->selected_node         = -1;
    state->pending_external_count = 0;
    state->pending_receipt_count = 0;
    state->failure_count         = 0;
    state->rollback_requested    = 0;
    state->star_bound            = 0;
    state->consecutive_blocks    = 0;
    state->block_threshold       = 0;
    state->hard_reset_requested  = 0;
    state->rollback_available    = 0;
    state->failure_action        = 0;
    state->selected_src          = -1;
    state->selected_dst          = -1;
    state->selected_control_edge = -1;
    state->selected_control_op   = -1;
    state->selected_control_lhs  = -1;
    state->selected_control_rhs  = -1;
    state->control_epoch         = 0;
    state->star_counter_epoch    = 0;
    state->last_block_reason     = 0;
}

// Copies the live device state into a separate snapshot buffer for tests/debug.
extern "C" __global__ void orchestration_state_snapshot(
    const OrchestrationState* state,
    OrchestrationState*       out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out = *state;
}

// ── Block-reason codes (last_block_reason field) ──────────────────────────────
#define ORCH_BLOCK_REASON_NONE           0
#define ORCH_BLOCK_REASON_NO_READY_NODE  1
#define ORCH_BLOCK_REASON_STAR_EXHAUSTED 2
#define ORCH_BLOCK_REASON_UNSUPPORTED    3
#define ORCH_BLOCK_REASON_ALL_CONSUMED   4

// Zero-initialise the per-edge star counter table.
// Launch with <<<ceil(count/256), 256>>> for parallel init.
extern "C" __global__ void star_counters_init(int* star_counters, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) star_counters[tid] = 0;
}

// Push one DeviceCommand into the command ring (single-threaded, thread 0 only).
// Returns without writing if the ring is full (tail - head >= ring_size).
extern "C" __global__ void device_command_ring_push(
    DeviceCommand* ring,
    int*           tail,
    const int*     head,
    int            ring_size,
    DeviceCommand  cmd
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = *tail, h = *head;
    if (t - h >= ring_size) return; // full; caller must retry
    ring[t % ring_size] = cmd;
    *tail = t + 1;
}

// Push one DeviceReceiptExt into the extended receipt ring (thread 0 only).
// Returns without writing if the ring is full.
extern "C" __global__ void device_receipt_ext_ring_push(
    DeviceReceiptExt* ring,
    int*              tail,
    const int*        head,
    int               ring_size,
    DeviceReceiptExt  receipt,
    OrchestrationState* state
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = *tail, h = *head;
    if (t - h >= ring_size) return; // full
    ring[t % ring_size] = receipt;
    *tail = t + 1;
    if (state && receipt.valid && !receipt.consumed) {
        atomicAdd(&state->pending_receipt_count, 1);
    }
}

// Drain the extended receipt ring, applying tensor updates for each valid entry.
// Marks each drained receipt as consumed (receipt.consumed = 1) in the ring.
// Advances *head to *tail when done.
extern "C" __global__ void device_receipt_ext_drain(
    float*            tensor,
    DeviceReceiptExt* ring,
    int               ring_size,
    int*              head,
    const int*        tail,
    OrchestrationState* state
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *head;
    int t = *tail;
    while (h != t) {
        int slot = h % ring_size;
        DeviceReceiptExt r = ring[slot];
        if (r.valid && !r.consumed
                && r.src >= 0 && r.src < N && r.dst >= 0 && r.dst < N && r.src != r.dst) {
            int cidx = tensor_idx(LAYER_CONFIDENCE, r.src, r.dst);
            int eidx = tensor_idx(LAYER_COST,       r.src, r.dst);
            int sidx = tensor_idx(LAYER_SAFETY,     r.src, r.dst);
            if (r.outcome == 0) {
                atomicExch(&tensor[cidx], 1.0f);
                {
                    int* ptr = (int*)&tensor[eidx];
                    int ob, nb;
                    do {
                        ob = *((volatile int*)ptr);
                        float v = __int_as_float(ob);
                        if (v <= 0.01f) break;
                        nb = __float_as_int(v * 0.5f);
                    } while (atomicCAS(ptr, ob, nb) != ob);
                }
                atomicExch(&tensor[sidx], 1.0f);
            } else if (r.outcome == 1) {
                atomic_float_mul(&tensor[cidx], 0.1f);
                atomic_float_add_capped(&tensor[eidx], 10.0f);
                atomic_float_mul(&tensor[sidx], 0.5f);
                if (state) atomicAdd(&state->failure_count, 1);
            } else if (r.outcome == 2) {
                atomic_float_mul(&tensor[cidx], 0.5f);
                atomic_float_add_capped(&tensor[eidx], 100.0f);
            } else if (r.outcome == 3) {
                atomic_float_mul(&tensor[cidx], 0.25f);
                atomic_float_add_capped(&tensor[eidx], 25.0f);
                atomicExch(&tensor[sidx], 0.0f);
            }
            ring[slot].consumed = 1;
            if (state) {
                atomicAdd(&state->pending_receipt_count, -1);
                if (state->pending_external_count > 0) {
                    atomicAdd(&state->pending_external_count, -1);
                }
            }
        }
        h++;
    }
    *head = h;
}

