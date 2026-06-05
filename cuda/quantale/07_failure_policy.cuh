// ── Phase-5: device-native failure policy ─────────────────────────────────────
//
// FailureClass: why a receipt outcome was non-zero.
// FailureAction: corrective action chosen on-device.
// FailurePolicy: per-node retry/repair configuration stored in GPU memory.
//
// classify_failure maps a raw outcome code to a FailureClass.
// failure_policy_classify_and_emit consults the per-node FailurePolicy, updates
// the retry budget, emits a repair DeviceCommand when needed, and writes the
// chosen FAILURE_ACTION_* code.
// failure_policy_set_rollback_marker snapshots consumed/active for later restore.
// failure_policy_apply_rollback restores that snapshot.

#define FAILURE_CLASS_SPAWN_FAILURE  0
#define FAILURE_CLASS_TIMEOUT        1
#define FAILURE_CLASS_SAFETY         2
#define FAILURE_CLASS_CONTRACT       3
#define FAILURE_CLASS_GPU_ERROR      4
#define FAILURE_CLASS_UNKNOWN        5

#define FAILURE_ACTION_RETRY           0
#define FAILURE_ACTION_BLOCK           1
#define FAILURE_ACTION_ROLLBACK        2
#define FAILURE_ACTION_HALT            3
#define FAILURE_ACTION_EXTERNAL_REPAIR 4

// Extends the Phase-2 DISPATCH_KIND_* table.
#define DISPATCH_KIND_REPAIR  8

struct FailurePolicy {
    int retry_budget;         // remaining retries; -1 = unlimited; 0 = exhausted
    int block_threshold;      // consecutive failures before BLOCK; -1 = disabled
    int rollback_on_failure;  // 1 = save rollback marker on any failure
    int repair_on_block;      // 1 = emit repair DeviceCommand when budget exhausted
};

// Packed classification target: passed as a single struct to stay within the
// cudarc LaunchAsync arity limit.
struct FailureClassifyRequest {
    int outcome;
    int node_id;
    int src;
    int dst;
    int command_id;
};

__device__ int classify_failure(int outcome) {
    if (outcome == 2) return FAILURE_CLASS_TIMEOUT;
    if (outcome == 3) return FAILURE_CLASS_SAFETY;
    if (outcome == 1) return FAILURE_CLASS_SPAWN_FAILURE;
    return FAILURE_CLASS_UNKNOWN;
}

// Initialise all per-node FailurePolicy entries to the given defaults.
// Launch with <<<ceil(n_nodes/256), 256>>> for parallel init.
extern "C" __global__ void failure_policy_init(
    FailurePolicy* policies,
    int            n_nodes,
    int            default_budget,
    int            default_block_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes) return;
    policies[tid].retry_budget        = default_budget;
    policies[tid].block_threshold     = default_block_threshold;
    policies[tid].rollback_on_failure = 0;
    policies[tid].repair_on_block     = 0;
}

// Classify a receipt failure, update the per-node retry budget, choose the
// corrective action, and emit a repair DeviceCommand when the action is
// EXTERNAL_REPAIR.  Writes the chosen FAILURE_ACTION_* code into *action_out.
// Thread 0, block 0 only.
extern "C" __global__ void failure_policy_classify_and_emit(
    const FailureClassifyRequest* req,
    FailurePolicy*                policies,
    int                           n_policies,
    OrchestrationState*           state,
    DeviceCommand*                cmd_ring,
    int*                          cmd_tail,
    const int*                    cmd_head,
    int                           cmd_ring_size,
    int*                          action_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int outcome    = req->outcome;
    int node_id    = req->node_id;
    int src        = req->src;
    int dst        = req->dst;
    int command_id = req->command_id;

    if (!policies || node_id < 0 || node_id >= n_policies) {
        *action_out = FAILURE_ACTION_BLOCK;
        return;
    }

    int failure_class = classify_failure(outcome);
    FailurePolicy* p = &policies[node_id];

    int action;
    if (failure_class == FAILURE_CLASS_SAFETY) {
        // Safety violations escalate to HALT regardless of retry budget.
        action = FAILURE_ACTION_HALT;
    } else if (p->retry_budget > 0) {
        p->retry_budget -= 1;
        action = FAILURE_ACTION_RETRY;
    } else if (p->retry_budget == -1) {
        action = FAILURE_ACTION_RETRY;  // unlimited
    } else {
        // Budget == 0: exhausted.
        action = p->repair_on_block ? FAILURE_ACTION_EXTERNAL_REPAIR : FAILURE_ACTION_BLOCK;
    }

    if (state) {
        state->failure_action = action;
        if (action == FAILURE_ACTION_HALT) {
            state->hard_reset_requested = 1;
            state->halted               = 1;
        } else if (action == FAILURE_ACTION_BLOCK || action == FAILURE_ACTION_EXTERNAL_REPAIR) {
            state->consecutive_blocks += 1;
            if (state->block_threshold > 0
                    && state->consecutive_blocks >= state->block_threshold)
                state->hard_reset_requested = 1;
        } else {
            state->consecutive_blocks = 0;  // retry clears the run of blocks
        }
    }

    if (action == FAILURE_ACTION_EXTERNAL_REPAIR && cmd_ring && cmd_tail && cmd_head) {
        int t = *cmd_tail, h = *cmd_head;
        if (t - h < cmd_ring_size) {
            DeviceCommand cmd;
            cmd.valid            = 1;
            cmd.command_id       = command_id;
            cmd.node_id          = node_id;
            cmd.src              = src;
            cmd.dst              = dst;
            cmd.dispatch_kind    = DISPATCH_KIND_REPAIR;
            cmd.operator_name_id = node_id;
            cmd.timeout_ticks    = 0;
            cmd.retry_budget     = 0;
            cmd.payload_offset   = 0;
            cmd.payload_len      = 0;
            cmd_ring[t % cmd_ring_size] = cmd;
            *cmd_tail = t + 1;
            if (state) atomicAdd(&state->pending_external_count, 1);
        }
    }

    *action_out = action;
}

// Snapshot consumed[] and active[] as a rollback marker.
// Sets state->rollback_available = 1.  Block-parallel copy.
extern "C" __global__ void failure_policy_set_rollback_marker(
    const int*          consumed,
    const int*          active,
    int*                rollback_consumed,
    int*                rollback_active,
    OrchestrationState* state
) {
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        rollback_consumed[i] = consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        rollback_active[i] = active[i];
    __syncthreads();
    if (tid == 0 && state)
        state->rollback_available = 1;
}

// Restore consumed[] and active[] from the rollback marker.
// No-op if state->rollback_available == 0.  Block-parallel copy.
extern "C" __global__ void failure_policy_apply_rollback(
    int*                consumed,
    int*                active,
    const int*          rollback_consumed,
    const int*          rollback_active,
    OrchestrationState* state
) {
    if (!state || !state->rollback_available) return;
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        consumed[i] = rollback_consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        active[i] = rollback_active[i];
    __syncthreads();
    if (tid == 0) {
        state->rollback_available = 0;
        state->rollback_requested = 0;
        state->consecutive_blocks = 0;
    }
}

