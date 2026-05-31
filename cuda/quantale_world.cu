struct LatticeEdge {
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

#define BOTTOM 0.0f
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
    if (!(value == value) || value <= BOTTOM) {
        return BOTTOM;
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
    if (a <= BOTTOM || b <= BOTTOM) {
        return BOTTOM;
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
// transition holds A / A* values; witness_matrix is W, the first-hop witness matrix.
__device__ void quantale_closure(float* transition, float* scratch, int* witness_matrix) {
    quantale_copy(scratch, transition);
    __syncthreads();

    for (int k = 0; k < N; ++k) {
        for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
            int i = idx / N;
            int j = idx % N;
            float through_k = q_mul(scratch[i * N + k], scratch[k * N + j]);
            if (through_k > scratch[idx]) {
                scratch[idx] = through_k;
                witness_matrix[idx] = witness_matrix[i * N + k]; // W[i,j] = W[i,k]
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
    int* witness_matrix,
    int* scratch_witness,
    int* consumed,
    int* active,
    int* next_active,
    int* event_counts,
    QuantaleCudaReport* out,
    DecisionReport* decision
) {
    int tid = threadIdx.x;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        transition[idx] = BOTTOM;
        scratch[idx] = BOTTOM;
        previous[idx] = BOTTOM;
        witness_matrix[idx] = -1;
        scratch_witness[idx] = -1;
        consumed[idx] = 0;
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
            witness_matrix[i * N + i] = i;
        }
        active[START_NODE] = 1;
        out->step = 0;
        out->best_src = -1;
        out->best_dst = -1;
        out->best_value = BOTTOM;
        out->event_count = 0;
        out->goal_to_execute = BOTTOM;
        out->goal_to_learn = BOTTOM;

        decision->step = 0;
        decision->selected_src = -1;
        decision->selected_dst = -1;
        decision->first_hop = -1;
        decision->selected_value = BOTTOM;
        decision->halted = 0;
        decision->blocked = 0;
    }
}

extern "C" __global__ void quantale_embed_elements(
    float* transition,
    int* witness_matrix,
    const LatticeEdge* edges,
    int edge_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int e = 0; e < edge_count; ++e) {
            int src = edges[e].src;
            int dst = edges[e].dst;
            float value = q_clamp(edges[e].value);
            if (src >= 0 && src < N && dst >= 0 && dst < N && value > BOTTOM) {
                int idx = src * N + dst;
                if (value > transition[idx]) {
                    transition[idx] = value;
                    witness_matrix[idx] = dst; // direct edge witness: W[src,dst] = dst
                }
            }
        }
    }
}

extern "C" __global__ void quantale_supremum_assign(
    float* lhs,
    int* lhs_witness_matrix,
    const float* rhs,
    const int* rhs_witness_matrix
) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        if (rhs[idx] > lhs[idx]) {
            lhs[idx] = rhs[idx];
            lhs_witness_matrix[idx] = rhs_witness_matrix[idx];
        }
    }
}

extern "C" __global__ void quantale_tensor_assign(
    float* lhs,
    int* lhs_witness_matrix,
    const float* rhs,
    const int* rhs_witness_matrix,
    float* scratch,
    int* scratch_witness
) {
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;
        float acc = BOTTOM;
        int hop = -1;
        for (int k = 0; k < N; ++k) {
            float candidate = q_mul(lhs[i * N + k], rhs[k * N + j]);
            if (candidate > acc) {
                acc = candidate;
                hop = lhs_witness_matrix[i * N + k];
            }
        }
        scratch[idx] = acc;
        scratch_witness[idx] = hop;
    }
    __syncthreads();
    quantale_copy(lhs, scratch);
    quantale_copy_i32(lhs_witness_matrix, scratch_witness);
}

extern "C" __global__ void quantale_least_fixed_point(
    float* transition,
    float* scratch,
    int* witness_matrix
) {
    quantale_closure(transition, scratch, witness_matrix);
}

extern "C" __global__ void quantale_step(
    float* transition,
    float* scratch,
    float* previous,
    int* witness_matrix,
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

    quantale_closure(transition, scratch, witness_matrix);

    int local_events = 0;
    float local_best_value = BOTTOM;
    int local_best_src = -1;
    int local_best_dst = -1;
    int local_best_hop = -1;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;
        float before = previous[idx];
        float after = transition[idx];
        if (i != j && active[i] != 0 && after > before && after > BOTTOM) {
            local_events += 1;
        }
        if (i != j && after > local_best_value) {
            local_best_value = after;
            local_best_src = i;
            local_best_dst = j;
            local_best_hop = witness_matrix[idx];
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

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane] : BOTTOM;
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
extern "C" __global__ void quantale_morphism(
    const float* transition,
    const int* witness_matrix,
    const int* consumed,
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

    float local_best_value = BOTTOM;
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
        if (value <= BOTTOM) {
            continue;
        }
        int hop = witness_matrix[idx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) {
            continue;
        }
        local_candidates += 1;
        choose_best(
            value,
            src,
            dst,
            hop,
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
        float block_best_value = lane < warp_count ? warp_values[lane] : BOTTOM;
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

// Fused Option-B frontier step:
//   S_frontier = (S ⊗ A*) ⊙ H_mask
//   argmax(S_frontier)
//   H_mask[src, first_hop] := 0
//   S := one_hot(first_hop)
//
// The matrix stays static. Dynamic execution state lives in the one-hot
// frontier vector and consumed/history mask. Host code receives only the compact
// DecisionReport scalar record.
extern "C" __global__ void quantale_frontier_step(
    const float* transition,
    const int* witness_matrix,
    int* consumed,
    int* active,
    int* next_active,
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

    for (int i = tid; i < N; i += blockDim.x) {
        next_active[i] = 0;
    }
    __syncthreads();

    float local_best_value = BOTTOM;
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

        int hop = witness_matrix[idx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) {
            continue;
        }

        float value = transition[idx];
        if (value <= BOTTOM) {
            continue;
        }

        local_candidates += 1;
        choose_best(
            value,
            src,
            dst,
            hop,
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
        float block_best_value = lane < warp_count ? warp_values[lane] : BOTTOM;
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
            decision->halted = block_best_hop == HALT_NODE ? 1 : 0;
            decision->blocked = block_candidates == 0 ? 1 : 0;

            if (block_candidates > 0 && block_best_src >= 0 && block_best_hop >= 0) {
                consumed[block_best_src * N + block_best_hop] = 1;
                next_active[block_best_hop] = 1;
            } else {
                for (int i = 0; i < N; ++i) {
                    next_active[i] = active[i];
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = next_active[i];
    }
}

// Fused runtime tick:
//   quantale_step closure/report work
//   quantale_frontier_step projection/frontier update work
//
// This removes the host-side launch boundary between `quantale_step` and
// `quantale_frontier_step` while keeping the existing standalone kernels for
// tests, benchmarks, and compatibility.
extern "C" __global__ void quantale_tick(
    float* transition,
    float* scratch,
    float* previous,
    int* witness_matrix,
    int* consumed,
    int* active,
    int* next_active,
    int* event_counts,
    QuantaleCudaReport* out,
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
    __shared__ int warp_counts[32];

    if (tid < blockDim.x) {
        event_counts[tid] = 0;
    }
    for (int i = tid; i < N; i += blockDim.x) {
        next_active[i] = 0;
    }
    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        previous[idx] = transition[idx];
    }
    __syncthreads();

    quantale_closure(transition, scratch, witness_matrix);

    int local_events = 0;
    float local_best_value = BOTTOM;
    int local_best_src = -1;
    int local_best_dst = -1;
    int local_best_hop = -1;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;
        float before = previous[idx];
        float after = transition[idx];
        if (i != j && active[i] != 0 && after > before && after > BOTTOM) {
            local_events += 1;
        }
        if (i != j && after > local_best_value) {
            local_best_value = after;
            local_best_src = i;
            local_best_dst = j;
            local_best_hop = witness_matrix[idx];
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
        warp_hops[warp_id] = local_best_hop;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane] : BOTTOM;
        int block_best_src = lane < warp_count ? warp_srcs[lane] : -1;
        int block_best_dst = lane < warp_count ? warp_dsts[lane] : -1;
        int block_best_hop = lane < warp_count ? warp_hops[lane] : -1;
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
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        next_active[i] = 0;
    }
    __syncthreads();

    float local_frontier_value = BOTTOM;
    int local_frontier_src = -1;
    int local_frontier_dst = -1;
    int local_frontier_hop = -1;
    int local_candidates = 0;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N;
        int dst = idx % N;
        if (src == dst || active[src] == 0) {
            continue;
        }

        int hop = witness_matrix[idx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) {
            continue;
        }

        float value = transition[idx];
        if (value <= BOTTOM) {
            continue;
        }

        local_candidates += 1;
        choose_best(
            value,
            src,
            dst,
            hop,
            local_frontier_value,
            local_frontier_src,
            local_frontier_dst,
            local_frontier_hop
        );
    }

    int warp_candidate_count = warp_reduce_sum(local_candidates);
    warp_reduce_best(local_frontier_value, local_frontier_src, local_frontier_dst, local_frontier_hop);

    if (lane == 0) {
        warp_values[warp_id] = local_frontier_value;
        warp_srcs[warp_id] = local_frontier_src;
        warp_dsts[warp_id] = local_frontier_dst;
        warp_hops[warp_id] = local_frontier_hop;
        warp_counts[warp_id] = warp_candidate_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane] : BOTTOM;
        int block_best_src = lane < warp_count ? warp_srcs[lane] : -1;
        int block_best_dst = lane < warp_count ? warp_dsts[lane] : -1;
        int block_best_hop = lane < warp_count ? warp_hops[lane] : -1;
        int block_candidates = lane < warp_count ? warp_counts[lane] : 0;

        block_candidates = warp_reduce_sum(block_candidates);
        warp_reduce_best(block_best_value, block_best_src, block_best_dst, block_best_hop);

        if (lane == 0) {
            decision->step += 1;
            decision->selected_src = block_best_src;
            decision->selected_dst = block_best_dst;
            decision->first_hop = block_best_hop;
            decision->selected_value = block_best_value;
            decision->halted = block_best_hop == HALT_NODE ? 1 : 0;
            decision->blocked = block_candidates == 0 ? 1 : 0;

            if (block_candidates > 0 && block_best_src >= 0 && block_best_hop >= 0) {
                consumed[block_best_src * N + block_best_hop] = 1;
                next_active[block_best_hop] = 1;
            } else {
                for (int i = 0; i < N; ++i) {
                    next_active[i] = active[i];
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = next_active[i];
    }
}

// Commit π(A*) dynamically: mark the selected first hop as consumed and advance
// the one-hot frontier vector S to that first hop. The transition matrix remains
// unchanged; only dynamic frontier/mask state is mutated.
extern "C" __global__ void quantale_commit_decision(
    int* active,
    int* next_active,
    int* consumed,
    const DecisionReport* decision
) {
    int tid = threadIdx.x;

    for (int i = tid; i < N; i += blockDim.x) {
        next_active[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        int src = decision->selected_src;
        int hop = decision->first_hop;
        if (decision->blocked == 0 && src >= 0 && src < N && hop >= 0 && hop < N) {
            consumed[src * N + hop] = 1;
            next_active[hop] = 1;
        } else {
            for (int i = 0; i < N; ++i) {
                next_active[i] = active[i];
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = next_active[i];
    }
}


// === Three-layer tensor quantale kernels ===
// Layers:
//   0 confidence/correctness: max-times
//   1 compute/time cost:      min-plus
//   2 security/safety:        max-min

struct TensorEdge {
    int src;
    int dst;
    float confidence;
    float cost;
    float safety;
};

struct ProjectionBias {
    float confidence;
    float cost;
    float safety;
};

#define TENSOR_LAYER_COUNT 3
#define TENSOR_LEN (TENSOR_LAYER_COUNT * MATRIX_LEN)
#define LAYER_CONFIDENCE 0
#define LAYER_COST 1
#define LAYER_SAFETY 2
#define COST_INFINITY 1.0e20f

__device__ __forceinline__ int tensor_idx(int layer, int src, int dst) {
    return layer * MATRIX_LEN + src * N + dst;
}

__device__ __forceinline__ float tensor_cost_clamp(float value) {
    if (!(value == value) || value < 0.0f) {
        return COST_INFINITY;
    }
    if (value > COST_INFINITY) {
        return COST_INFINITY;
    }
    return value;
}

__device__ __forceinline__ float tensor_safety_clamp(float value) {
    return q_clamp(value);
}

__device__ __forceinline__ float tensor_conf_compose(float a, float b) {
    return q_mul(a, b);
}

__device__ __forceinline__ float tensor_cost_compose(float a, float b) {
    if (a >= COST_INFINITY || b >= COST_INFINITY) {
        return COST_INFINITY;
    }
    float out = a + b;
    return out >= COST_INFINITY ? COST_INFINITY : out;
}

__device__ __forceinline__ float tensor_safety_compose(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ bool tensor_better(int layer, float candidate, float current) {
    if (layer == LAYER_COST) {
        return candidate < current;
    }
    return candidate > current;
}

extern "C" __global__ void tensor_quantale_reset(
    float* tensor,
    float* scratch,
    int* witness,
    int* scratch_witness,
    int* consumed,
    int* active,
    DecisionReport* decision
) {
    int tid = threadIdx.x;
    for (int idx = tid; idx < TENSOR_LEN; idx += blockDim.x) {
        int layer = idx / MATRIX_LEN;
        tensor[idx] = layer == LAYER_COST ? COST_INFINITY : BOTTOM;
        scratch[idx] = tensor[idx];
        witness[idx] = -1;
        scratch_witness[idx] = -1;
    }
    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        consumed[idx] = 0;
    }
    for (int i = tid; i < N; i += blockDim.x) {
        active[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < N; ++i) {
            tensor[tensor_idx(LAYER_CONFIDENCE, i, i)] = Q_UNIT;
            tensor[tensor_idx(LAYER_COST, i, i)] = 0.0f;
            tensor[tensor_idx(LAYER_SAFETY, i, i)] = Q_UNIT;
            witness[tensor_idx(LAYER_CONFIDENCE, i, i)] = i;
            witness[tensor_idx(LAYER_COST, i, i)] = i;
            witness[tensor_idx(LAYER_SAFETY, i, i)] = i;
        }
        active[START_NODE] = 1;
        decision->step = 0;
        decision->selected_src = -1;
        decision->selected_dst = -1;
        decision->first_hop = -1;
        decision->selected_value = BOTTOM;
        decision->halted = 0;
        decision->blocked = 0;
    }
}

extern "C" __global__ void tensor_quantale_embed_edges(
    float* tensor,
    int* witness,
    const TensorEdge* edges,
    int edge_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int e = 0; e < edge_count; ++e) {
            int src = edges[e].src;
            int dst = edges[e].dst;
            if (src < 0 || src >= N || dst < 0 || dst >= N) {
                continue;
            }

            float confidence = q_clamp(edges[e].confidence);
            float cost = tensor_cost_clamp(edges[e].cost);
            float safety = tensor_safety_clamp(edges[e].safety);

            int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
            int eidx = tensor_idx(LAYER_COST, src, dst);
            int sidx = tensor_idx(LAYER_SAFETY, src, dst);

            if (confidence > tensor[cidx]) {
                tensor[cidx] = confidence;
                witness[cidx] = dst;
            }
            if (cost < tensor[eidx]) {
                tensor[eidx] = cost;
                witness[eidx] = dst;
            }
            if (safety > tensor[sidx]) {
                tensor[sidx] = safety;
                witness[sidx] = dst;
            }
        }
    }
}

extern "C" __global__ void tensor_quantale_closure(
    float* tensor,
    float* scratch,
    int* witness
) {
    for (int idx = threadIdx.x; idx < TENSOR_LEN; idx += blockDim.x) {
        scratch[idx] = tensor[idx];
    }
    __syncthreads();

    for (int k = 0; k < N; ++k) {
        for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
            int i = idx / N;
            int j = idx % N;

            int c_ij = tensor_idx(LAYER_CONFIDENCE, i, j);
            int c_ik = tensor_idx(LAYER_CONFIDENCE, i, k);
            int c_kj = tensor_idx(LAYER_CONFIDENCE, k, j);
            float c_candidate = tensor_conf_compose(scratch[c_ik], scratch[c_kj]);
            if (c_candidate > scratch[c_ij]) {
                scratch[c_ij] = c_candidate;
                witness[c_ij] = witness[c_ik];
            }

            int e_ij = tensor_idx(LAYER_COST, i, j);
            int e_ik = tensor_idx(LAYER_COST, i, k);
            int e_kj = tensor_idx(LAYER_COST, k, j);
            float e_candidate = tensor_cost_compose(scratch[e_ik], scratch[e_kj]);
            if (e_candidate < scratch[e_ij]) {
                scratch[e_ij] = e_candidate;
                witness[e_ij] = witness[e_ik];
            }

            int s_ij = tensor_idx(LAYER_SAFETY, i, j);
            int s_ik = tensor_idx(LAYER_SAFETY, i, k);
            int s_kj = tensor_idx(LAYER_SAFETY, k, j);
            float s_candidate = tensor_safety_compose(scratch[s_ik], scratch[s_kj]);
            if (s_candidate > scratch[s_ij]) {
                scratch[s_ij] = s_candidate;
                witness[s_ij] = witness[s_ik];
            }
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < TENSOR_LEN; idx += blockDim.x) {
        tensor[idx] = scratch[idx];
    }
}

extern "C" __global__ void tensor_quantale_project(
    const float* tensor,
    const int* witness,
    const int* consumed,
    const int* active,
    const ProjectionBias* bias,
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

    float local_best_value = -COST_INFINITY;
    int local_best_src = -1;
    int local_best_dst = -1;
    int local_best_hop = -1;
    int local_candidates = 0;

    float alpha = bias[0].confidence;
    float beta = bias[0].cost;
    float gamma = bias[0].safety;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N;
        int dst = idx % N;
        if (src == dst || active[src] == 0) {
            continue;
        }

        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST, src, dst);
        int sidx = tensor_idx(LAYER_SAFETY, src, dst);

        float confidence = tensor[cidx];
        float cost = tensor[eidx];
        float safety = tensor[sidx];
        if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) {
            continue;
        }

        int hop = witness[cidx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) {
            continue;
        }

        float score = alpha * confidence - beta * cost + gamma * safety;
        local_candidates += 1;
        choose_best(score, src, dst, hop, local_best_value, local_best_src, local_best_dst, local_best_hop);
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
        float block_best_value = lane < warp_count ? warp_values[lane] : -COST_INFINITY;
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
            decision->halted = block_best_hop == HALT_NODE ? 1 : 0;
            decision->blocked = block_candidates == 0 ? 1 : 0;
        }
    }
}

extern "C" __global__ void tensor_quantale_update_edge(
    float* tensor,
    int src,
    int dst,
    int outcome
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (src < 0 || src >= N || dst < 0 || dst >= N || src == dst) {
            return;
        }
        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST, src, dst);
        int sidx = tensor_idx(LAYER_SAFETY, src, dst);

        if (outcome == 0) { // success
            tensor[cidx] = q_join(tensor[cidx], 1.0f);
            tensor[eidx] = tensor[eidx] > 0.01f ? tensor[eidx] * 0.5f : tensor[eidx];
            tensor[sidx] = q_join(tensor[sidx], 1.0f);
        } else if (outcome == 1) { // failure
            tensor[cidx] = tensor[cidx] * 0.1f;
            tensor[eidx] = tensor_cost_compose(tensor[eidx], 10.0f);
            tensor[sidx] = tensor[sidx] * 0.5f;
        } else if (outcome == 2) { // timeout
            tensor[cidx] = tensor[cidx] * 0.5f;
            tensor[eidx] = tensor_cost_compose(tensor[eidx], 100.0f);
        } else if (outcome == 3) { // safety violation
            tensor[cidx] = tensor[cidx] * 0.25f;
            tensor[eidx] = tensor_cost_compose(tensor[eidx], 25.0f);
            tensor[sidx] = BOTTOM;
        }
    }
}

extern "C" __global__ void tensor_quantale_decay(
    float* tensor,
    float factor
) {
    float decay = q_clamp(factor);
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N;
        int dst = idx % N;
        if (src == dst) {
            continue;
        }
        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST, src, dst);
        int sidx = tensor_idx(LAYER_SAFETY, src, dst);
        tensor[cidx] *= decay;
        if (tensor[eidx] < COST_INFINITY) {
            tensor[eidx] = tensor_cost_compose(tensor[eidx], 1.0f - decay);
        }
        tensor[sidx] *= decay;
    }
}
