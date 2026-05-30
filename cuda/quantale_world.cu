struct TransitionEdge {
    int src;
    int dst;
    float value;
};

struct QuantaleCudaReport {
    int step;
    int best_src;
    int best_dst;
    float best_value;
    int event_count;
    float goal_to_execute;
    float goal_to_learn;
};

struct DecisionReport {
    int step;
    int selected_src;
    int selected_dst;
    int first_hop;
    float selected_value;
    int halted;
    int blocked;
};

#define STATE_NODE_COUNT 13
#define CONTROL_NODE_COUNT 13
#define EVENT_NODE_COUNT 18

#define STATE_OFFSET 0
#define CONTROL_OFFSET (STATE_OFFSET + STATE_NODE_COUNT)
#define EVENT_OFFSET (CONTROL_OFFSET + CONTROL_NODE_COUNT)

#define N (STATE_NODE_COUNT + CONTROL_NODE_COUNT + EVENT_NODE_COUNT)
#define MATRIX_LEN (N * N)

#define Q_BOTTOM 0.0f
#define Q_UNIT 1.0f

#define S_Goal 0
#define S_Input 1
#define S_Parse 2
#define S_Map 3
#define S_Search 4
#define S_Score 5
#define S_Select 6
#define S_Plan 7
#define S_Optimize 8
#define S_Execute 9
#define S_Validate 10
#define S_Memory 11
#define S_Learn 12

#define C_Allow 0
#define C_Block 1
#define C_Retry 2
#define C_Repair 3
#define C_Commit 4
#define C_Rollback 5
#define C_Halt 6
#define C_GateInput 7
#define C_GateExecution 8
#define C_GateReceipt 9
#define C_GateMemory 10
#define C_GateLearn 11
#define C_ChooseBest 12

#define E_FactArrived 0
#define E_InputAccepted 1
#define E_ParseOk 2
#define E_ParseErr 3
#define E_MapReady 4
#define E_CandidateFound 5
#define E_ScoreReady 6
#define E_TopKSelected 7
#define E_PlanReady 8
#define E_OptimizeReady 9
#define E_ExecuteStarted 10
#define E_ExecuteFinished 11
#define E_ReceiptAttached 12
#define E_ReceiptAccepted 13
#define E_ReceiptRejected 14
#define E_HashNonzero 15
#define E_MemoryWritten 16
#define E_LearnUpdated 17

#define STATE_ID(x) (STATE_OFFSET + (x))
#define CONTROL_ID(x) (CONTROL_OFFSET + (x))
#define EVENT_ID(x) (EVENT_OFFSET + (x))

#define START_NODE STATE_ID(S_Goal)
#define EXECUTE_PROBE_NODE STATE_ID(S_Execute)
#define LEARN_PROBE_NODE STATE_ID(S_Learn)
#define HALT_NODE CONTROL_ID(C_Halt)

__device__ __forceinline__ float q_clamp(float value) {
    if (!(value == value) || value <= Q_BOTTOM) {
        return Q_BOTTOM;
    }
    if (value >= Q_UNIT) {
        return Q_UNIT;
    }
    return value;
}

__device__ __forceinline__ float q_join(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float q_mul(float a, float b) {
    if (a <= Q_BOTTOM || b <= Q_BOTTOM) {
        return Q_BOTTOM;
    }
    return q_clamp(a * b);
}


__device__ __forceinline__ void choose_best(
    float candidate_value,
    int candidate_src,
    int candidate_dst,
    int candidate_hop,
    float& best_value,
    int& best_src,
    int& best_dst,
    int& best_hop
) {
    if (candidate_value > best_value) {
        best_value = candidate_value;
        best_src = candidate_src;
        best_dst = candidate_dst;
        best_hop = candidate_hop;
    }
}

__device__ __forceinline__ void warp_reduce_best(
    float& value,
    int& src,
    int& dst,
    int& hop
) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_value = __shfl_down_sync(mask, value, offset);
        int other_src = __shfl_down_sync(mask, src, offset);
        int other_dst = __shfl_down_sync(mask, dst, offset);
        int other_hop = __shfl_down_sync(mask, hop, offset);
        choose_best(other_value, other_src, other_dst, other_hop, value, src, dst, hop);
    }
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ __forceinline__ void quantale_copy(float* dst, const float* src) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
}

__device__ __forceinline__ void quantale_copy_i32(int* dst, const int* src) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
}

// A* closure over max-times quantale paths.
// transition holds A / A* values; next_hop is W, the first-hop witness matrix.
__device__ void quantale_closure(float* transition, float* scratch, int* next_hop) {
    quantale_copy(scratch, transition);
    __syncthreads();

    for (int k = 0; k < N; ++k) {
        for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
            int i = idx / N;
            int j = idx % N;
            float through_k = q_mul(scratch[i * N + k], scratch[k * N + j]);
            if (through_k > scratch[idx]) {
                scratch[idx] = through_k;
                next_hop[idx] = next_hop[i * N + k]; // W[i,j] = W[i,k]
            }
        }
        __syncthreads();
    }

    quantale_copy(transition, scratch);
    __syncthreads();
}

extern "C" __global__ void quantale_reset(
    float* transition,
    float* scratch,
    float* previous,
    int* next_hop,
    int* scratch_next_hop,
    int* active,
    int* next_active,
    int* event_counts,
    QuantaleCudaReport* out,
    DecisionReport* decision
) {
    int tid = threadIdx.x;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        transition[idx] = Q_BOTTOM;
        scratch[idx] = Q_BOTTOM;
        previous[idx] = Q_BOTTOM;
        next_hop[idx] = -1;
        scratch_next_hop[idx] = -1;
    }
    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = 0;
        next_active[i] = 0;
    }
    for (int i = tid; i < blockDim.x; i += blockDim.x) {
        event_counts[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < N; ++i) {
            transition[i * N + i] = Q_UNIT;
            next_hop[i * N + i] = i;
        }
        active[START_NODE] = 1;
        out->step = 0;
        out->best_src = -1;
        out->best_dst = -1;
        out->best_value = Q_BOTTOM;
        out->event_count = 0;
        out->goal_to_execute = Q_BOTTOM;
        out->goal_to_learn = Q_BOTTOM;

        decision->step = 0;
        decision->selected_src = -1;
        decision->selected_dst = -1;
        decision->first_hop = -1;
        decision->selected_value = Q_BOTTOM;
        decision->halted = 0;
        decision->blocked = 0;
    }
}

extern "C" __global__ void quantale_load_edges(
    float* transition,
    int* next_hop,
    const TransitionEdge* edges,
    int edge_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int e = 0; e < edge_count; ++e) {
            int src = edges[e].src;
            int dst = edges[e].dst;
            float value = q_clamp(edges[e].value);
            if (src >= 0 && src < N && dst >= 0 && dst < N && value > Q_BOTTOM) {
                int idx = src * N + dst;
                if (value > transition[idx]) {
                    transition[idx] = value;
                    next_hop[idx] = dst; // direct edge witness: W[src,dst] = dst
                }
            }
        }
    }
}

extern "C" __global__ void quantale_join_assign(
    float* lhs,
    int* lhs_next_hop,
    const float* rhs,
    const int* rhs_next_hop
) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        if (rhs[idx] > lhs[idx]) {
            lhs[idx] = rhs[idx];
            lhs_next_hop[idx] = rhs_next_hop[idx];
        }
    }
}

extern "C" __global__ void quantale_mul_assign(
    float* lhs,
    int* lhs_next_hop,
    const float* rhs,
    const int* rhs_next_hop,
    float* scratch,
    int* scratch_next_hop
) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;
        float acc = Q_BOTTOM;
        int hop = -1;
        for (int k = 0; k < N; ++k) {
            float candidate = q_mul(lhs[i * N + k], rhs[k * N + j]);
            if (candidate > acc) {
                acc = candidate;
                hop = lhs_next_hop[i * N + k];
            }
        }
        scratch[idx] = acc;
        scratch_next_hop[idx] = hop;
    }
    __syncthreads();
    quantale_copy(lhs, scratch);
    quantale_copy_i32(lhs_next_hop, scratch_next_hop);
}

extern "C" __global__ void quantale_closure_assign(
    float* transition,
    float* scratch,
    int* next_hop
) {
    quantale_closure(transition, scratch, next_hop);
}

extern "C" __global__ void quantale_step(
    float* transition,
    float* scratch,
    float* previous,
    int* next_hop,
    int* active,
    int* next_active,
    int* event_counts,
    QuantaleCudaReport* out
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warp_count = (blockDim.x + 31) >> 5;

    __shared__ float warp_values[32];
    __shared__ int warp_srcs[32];
    __shared__ int warp_dsts[32];
    __shared__ int warp_counts[32];

    if (tid < blockDim.x) {
        event_counts[tid] = 0;
    }
    for (int i = tid; i < N; i += blockDim.x) {
        next_active[i] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        previous[idx] = transition[idx];
    }
    __syncthreads();

    quantale_closure(transition, scratch, next_hop);

    int local_events = 0;
    float local_best_value = Q_BOTTOM;
    int local_best_src = -1;
    int local_best_dst = -1;
    int local_best_hop = -1;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;
        float before = previous[idx];
        float after = transition[idx];
        if (i != j && active[i] != 0 && after > before && after > Q_BOTTOM) {
            local_events += 1;
        }
        if (i != j && after > local_best_value) {
            local_best_value = after;
            local_best_src = i;
            local_best_dst = j;
            local_best_hop = next_hop[idx];
        }
    }

    int warp_events = warp_reduce_sum(local_events);
    warp_reduce_best(local_best_value, local_best_src, local_best_dst, local_best_hop);

    if (lane == 0) {
        event_counts[warp_id] = warp_events;
        warp_counts[warp_id] = warp_events;
        warp_values[warp_id] = local_best_value;
        warp_srcs[warp_id] = local_best_src;
        warp_dsts[warp_id] = local_best_dst;
    }
    __syncthreads();

    for (int dst = tid; dst < N; dst += blockDim.x) {
        int reachable = 0;
        for (int src = 0; src < N; ++src) {
            if (active[src] != 0 && transition[src * N + dst] > Q_BOTTOM) {
                reachable = 1;
                break;
            }
        }
        next_active[dst] = reachable;
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = next_active[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane] : Q_BOTTOM;
        int block_best_src = lane < warp_count ? warp_srcs[lane] : -1;
        int block_best_dst = lane < warp_count ? warp_dsts[lane] : -1;
        int block_best_hop = -1;
        int block_events = lane < warp_count ? warp_counts[lane] : 0;

        block_events = warp_reduce_sum(block_events);
        warp_reduce_best(block_best_value, block_best_src, block_best_dst, block_best_hop);

        if (lane == 0) {
            out->step += 1;
            out->best_src = block_best_src;
            out->best_dst = block_best_dst;
            out->best_value = block_best_value;
            out->event_count = block_events;
            out->goal_to_execute = transition[START_NODE * N + EXECUTE_PROBE_NODE];
            out->goal_to_learn = transition[START_NODE * N + LEARN_PROBE_NODE];
        }
    }
}

// π(A*) decision projection: choose the best reachable destination from
// the active frontier using matrix structure only, then emit W[src,dst] as the next executable hop.
// Expects `transition` to already contain the closed matrix A*.
extern "C" __global__ void quantale_decide_path(
    const float* transition,
    const int* next_hop,
    const int* active,
    DecisionReport* decision
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warp_count = (blockDim.x + 31) >> 5;

    __shared__ float warp_values[32];
    __shared__ int warp_srcs[32];
    __shared__ int warp_dsts[32];
    __shared__ int warp_hops[32];
    __shared__ int warp_candidates[32];

    float local_best_value = Q_BOTTOM;
    int local_best_src = -1;
    int local_best_dst = -1;
    int local_best_hop = -1;
    int local_candidates = 0;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N;
        int dst = idx % N;
        if (src == dst || active[src] == 0) {
            continue;
        }
        float value = transition[idx];
        if (value <= Q_BOTTOM) {
            continue;
        }
        local_candidates += 1;
        choose_best(
            value,
            src,
            dst,
            next_hop[idx],
            local_best_value,
            local_best_src,
            local_best_dst,
            local_best_hop
        );
    }

    int warp_candidate_count = warp_reduce_sum(local_candidates);
    warp_reduce_best(local_best_value, local_best_src, local_best_dst, local_best_hop);

    if (lane == 0) {
        warp_values[warp_id] = local_best_value;
        warp_srcs[warp_id] = local_best_src;
        warp_dsts[warp_id] = local_best_dst;
        warp_hops[warp_id] = local_best_hop;
        warp_candidates[warp_id] = warp_candidate_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane] : Q_BOTTOM;
        int block_best_src = lane < warp_count ? warp_srcs[lane] : -1;
        int block_best_dst = lane < warp_count ? warp_dsts[lane] : -1;
        int block_best_hop = lane < warp_count ? warp_hops[lane] : -1;
        int block_candidates = lane < warp_count ? warp_candidates[lane] : 0;

        block_candidates = warp_reduce_sum(block_candidates);
        warp_reduce_best(block_best_value, block_best_src, block_best_dst, block_best_hop);

        if (lane == 0) {
            decision->step += 1;
            decision->selected_src = block_best_src;
            decision->selected_dst = block_best_dst;
            decision->first_hop = block_best_hop;
            decision->selected_value = block_best_value;
            decision->halted = block_best_dst == HALT_NODE ? 1 : 0;
            decision->blocked = block_candidates == 0 ? 1 : 0;
        }
    }
}
