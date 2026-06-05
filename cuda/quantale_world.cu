// Three-layer tensor quantale world.
//
// Layers:
//   0  confidence/correctness  max-times
//   1  compute/time cost       min-plus
//   2  security/safety         max-min
//
// N must match TENSOR_NODE_COUNT in src/tensor.rs.
// HALT_NODE must match id_of("Control::Halt") in the generated topology.

struct DecisionReport {
    int step;
    int selected_src;
    int selected_dst;
    int first_hop;
    float selected_value;
    int halted;
    int blocked;
};

#ifndef N
#error "N must be generated from the topology asset by build.rs"
#endif
#define MATRIX_LEN (N * N)

#define BOTTOM   0.0f
#define Q_UNIT   1.0f

// Special node IDs derived from the generated topology.
#define START_NODE  0   // State::Goal
#define HALT_NODE  19   // Control::Halt  (CONTROL_OFFSET=13, C_Halt=6)

// ── shared device helpers ──────────────────────────────────────────────────

__device__ __forceinline__ float q_clamp(float value) {
    if (!(value == value) || value <= BOTTOM) return BOTTOM;
    if (value >= Q_UNIT) return Q_UNIT;
    return value;
}

__device__ __forceinline__ float q_join(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float q_mul(float a, float b) {
    if (a <= BOTTOM || b <= BOTTOM) return BOTTOM;
    return q_clamp(a * b);
}

__device__ __forceinline__ void choose_best(
    float candidate_value, int candidate_src, int candidate_dst, int candidate_hop,
    float& best_value, int& best_src, int& best_dst, int& best_hop
) {
    if (candidate_value > best_value) {
        best_value = candidate_value;
        best_src   = candidate_src;
        best_dst   = candidate_dst;
        best_hop   = candidate_hop;
    }
}

__device__ __forceinline__ void warp_reduce_best(
    float& value, int& src, int& dst, int& hop
) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_value = __shfl_down_sync(mask, value,  offset);
        int   other_src   = __shfl_down_sync(mask, src,    offset);
        int   other_dst   = __shfl_down_sync(mask, dst,    offset);
        int   other_hop   = __shfl_down_sync(mask, hop,    offset);
        choose_best(other_value, other_src, other_dst, other_hop, value, src, dst, hop);
    }
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset);
    return value;
}

// ── tensor structs and layer constants ────────────────────────────────────

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
#define TENSOR_LEN  (TENSOR_LAYER_COUNT * MATRIX_LEN)
#define LAYER_CONFIDENCE 0
#define LAYER_COST       1
#define LAYER_SAFETY     2
#define COST_INFINITY    1.0e20f

__device__ __forceinline__ int tensor_idx(int layer, int src, int dst) {
    return layer * MATRIX_LEN + src * N + dst;
}

__device__ __forceinline__ float tensor_cost_clamp(float value) {
    if (!(value == value) || value < 0.0f) return COST_INFINITY;
    if (value > COST_INFINITY) return COST_INFINITY;
    return value;
}

__device__ __forceinline__ float tensor_safety_clamp(float value) {
    return q_clamp(value);
}

__device__ __forceinline__ float tensor_conf_compose(float a, float b) {
    return q_mul(a, b);
}

__device__ __forceinline__ float tensor_cost_compose(float a, float b) {
    if (a >= COST_INFINITY || b >= COST_INFINITY) return COST_INFINITY;
    float out = a + b;
    return out >= COST_INFINITY ? COST_INFINITY : out;
}

__device__ __forceinline__ float tensor_safety_compose(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ bool tensor_better(int layer, float candidate, float current) {
    return layer == LAYER_COST ? candidate < current : candidate > current;
}

// ── atomic float helpers ──────────────────────────────────────────────────

// CAS-loop multiply: atomically applies tensor[addr] *= factor.
__device__ __forceinline__ void atomic_float_mul(float* addr, float factor) {
    int* iptr = (int*)addr;
    int ob, nb;
    do {
        ob = *((volatile int*)iptr);
        nb = __float_as_int(__int_as_float(ob) * factor);
    } while (atomicCAS(iptr, ob, nb) != ob);
}

// CAS-loop capped add: atomically applies tensor[addr] += delta, capped at COST_INFINITY.
__device__ __forceinline__ void atomic_float_add_capped(float* addr, float delta) {
    int* iptr = (int*)addr;
    int ob, nb;
    do {
        ob = *((volatile int*)iptr);
        float v = __int_as_float(ob);
        float nv = (v >= COST_INFINITY) ? COST_INFINITY : v + delta;
        if (nv > COST_INFINITY) nv = COST_INFINITY;
        nb = __float_as_int(nv);
    } while (atomicCAS(iptr, ob, nb) != ob);
}

struct ExecutionReceipt {
    int src;
    int dst;
    int outcome; // 0=success, 1=failure, 2=timeout, 3=safety violation
};

// ── Device-side receipts ──────────────────────────────────────────────────────
//
// DeviceReceipt is the GPU-native receipt produced by the hot execution path.
// Unlike ExecutionReceipt (built on the CPU from process stdout), a
// DeviceReceipt is written entirely on-device by tensor_quantale_gpu_dispatch
// and drained by tensor_quantale_drain_device_receipts without any CPU hop.

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

// Mailbox used by the hot dispatch path: the host fills pending_region_id,
// src_node, dst_node, and outcome (from JIT exit code), then launches
// tensor_quantale_gpu_dispatch which writes a DeviceReceipt with the correct
// outcome — never hard-codes success.
struct GpuDispatchMailbox {
    int pending_region_id; // -1 = empty
    int src_node;
    int dst_node;
    int outcome;           // 0=success,1=failure,2=timeout,3=safety_violation
    int dispatched;        // set to 1 by tensor_quantale_gpu_dispatch
};

// ── kernels ───────────────────────────────────────────────────────────────

extern "C" __global__ void tensor_quantale_reset(
    float* tensor,
    float* scratch,
    int*   witness,
    int*   scratch_witness,
    int*   consumed,
    int*   active,
    DecisionReport* decision
) {
    int tid = threadIdx.x;
    for (int idx = tid; idx < TENSOR_LEN; idx += blockDim.x) {
        int layer    = idx / MATRIX_LEN;
        tensor[idx]  = layer == LAYER_COST ? COST_INFINITY : BOTTOM;
        scratch[idx] = tensor[idx];
        witness[idx] = -1;
        scratch_witness[idx] = -1;
    }
    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x)
        consumed[idx] = 0;
    for (int i = tid; i < N; i += blockDim.x)
        active[i] = 0;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < N; ++i) {
            tensor[tensor_idx(LAYER_CONFIDENCE, i, i)] = Q_UNIT;
            tensor[tensor_idx(LAYER_COST,       i, i)] = 0.0f;
            tensor[tensor_idx(LAYER_SAFETY,     i, i)] = Q_UNIT;
            witness[tensor_idx(LAYER_CONFIDENCE, i, i)] = i;
            witness[tensor_idx(LAYER_COST,       i, i)] = i;
            witness[tensor_idx(LAYER_SAFETY,     i, i)] = i;
        }
        active[START_NODE] = 1;
        decision->step           = 0;
        decision->selected_src   = -1;
        decision->selected_dst   = -1;
        decision->first_hop      = -1;
        decision->selected_value = BOTTOM;
        decision->halted         = 0;
        decision->blocked        = 0;
    }
}

extern "C" __global__ void tensor_quantale_embed_edges(
    float* tensor,
    int*   witness,
    const TensorEdge* edges,
    int edge_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int e = 0; e < edge_count; ++e) {
        int src = edges[e].src;
        int dst = edges[e].dst;
        if (src < 0 || src >= N || dst < 0 || dst >= N) continue;

        float confidence = q_clamp(edges[e].confidence);
        float cost       = tensor_cost_clamp(edges[e].cost);
        float safety     = tensor_safety_clamp(edges[e].safety);

        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST,       src, dst);
        int sidx = tensor_idx(LAYER_SAFETY,     src, dst);

        if (confidence > tensor[cidx]) { tensor[cidx] = confidence; witness[cidx] = dst; }
        if (cost       < tensor[eidx]) { tensor[eidx] = cost;       witness[eidx] = dst; }
        if (safety     > tensor[sidx]) { tensor[sidx] = safety;     witness[sidx] = dst; }
    }
}

extern "C" __global__ void tensor_quantale_closure(
    float* tensor,
    float* scratch,  // unused; retained so tensor_quantale_tick can call with same args
    int*   witness
) {
    int tid = threadIdx.x;

    // Floyd-Warshall transitive closure.
    // Invariant: during step k, row k and column k are unchanged (diagonal = identity),
    // so parallel reads within a step are data-race-free.
    for (int k = 0; k < N; ++k) {
        for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
            int i = idx / N, j = idx % N;
            int ij = i * N + j;
            int ik = i * N + k;
            int kj = k * N + j;

            // Layer 0: confidence (max-times)
            float c_cand = tensor[ik + LAYER_CONFIDENCE * MATRIX_LEN]
                         * tensor[kj + LAYER_CONFIDENCE * MATRIX_LEN];
            if (c_cand > 1.0f) c_cand = 1.0f;
            if (c_cand > tensor[ij + LAYER_CONFIDENCE * MATRIX_LEN]) {
                tensor[ij + LAYER_CONFIDENCE * MATRIX_LEN] = c_cand;
                witness[tensor_idx(LAYER_CONFIDENCE, i, j)] =
                    witness[tensor_idx(LAYER_CONFIDENCE, i, k)];
            }

            // Layer 1: cost (min-plus)
            float ik_cost = tensor[ik + LAYER_COST * MATRIX_LEN];
            float kj_cost = tensor[kj + LAYER_COST * MATRIX_LEN];
            float e_cand = (ik_cost >= COST_INFINITY || kj_cost >= COST_INFINITY)
                           ? COST_INFINITY : ik_cost + kj_cost;
            if (e_cand < tensor[ij + LAYER_COST * MATRIX_LEN]) {
                tensor[ij + LAYER_COST * MATRIX_LEN] = e_cand;
                witness[tensor_idx(LAYER_COST, i, j)] =
                    witness[tensor_idx(LAYER_COST, i, k)];
            }

            // Layer 2: safety (max-min)
            float ik_safe = tensor[ik + LAYER_SAFETY * MATRIX_LEN];
            float kj_safe = tensor[kj + LAYER_SAFETY * MATRIX_LEN];
            float s_cand = ik_safe < kj_safe ? ik_safe : kj_safe;
            if (s_cand > tensor[ij + LAYER_SAFETY * MATRIX_LEN]) {
                tensor[ij + LAYER_SAFETY * MATRIX_LEN] = s_cand;
                witness[tensor_idx(LAYER_SAFETY, i, j)] =
                    witness[tensor_idx(LAYER_SAFETY, i, k)];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void tensor_quantale_project(
    const float* tensor,
    const int*   witness,
    const int*   consumed,
    const int*   active,
    const ProjectionBias* bias,
    DecisionReport* decision
) {
    int tid = threadIdx.x, lane = tid & 31, warp_id = tid >> 5;
    int warp_count = (blockDim.x + 31) >> 5;

    __shared__ float warp_values[32];
    __shared__ int   warp_srcs[32], warp_dsts[32], warp_hops[32], warp_candidates[32];

    float local_best_value = -COST_INFINITY;
    int local_best_src = -1, local_best_dst = -1, local_best_hop = -1, local_candidates = 0;
    float alpha = bias[0].confidence, beta = bias[0].cost, gamma = bias[0].safety;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N, dst = idx % N;
        if (src == dst || active[src] == 0) continue;

        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST,       src, dst);
        int sidx = tensor_idx(LAYER_SAFETY,     src, dst);
        float confidence = tensor[cidx], cost = tensor[eidx], safety = tensor[sidx];
        if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

        int hop = witness[cidx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) continue;

        float score = alpha * confidence - beta * cost + gamma * safety;
        local_candidates += 1;
        choose_best(score, src, dst, hop, local_best_value, local_best_src, local_best_dst, local_best_hop);
    }

    int warp_candidate_count = warp_reduce_sum(local_candidates);
    warp_reduce_best(local_best_value, local_best_src, local_best_dst, local_best_hop);

    if (lane == 0) {
        warp_values[warp_id]     = local_best_value;
        warp_srcs[warp_id]       = local_best_src;
        warp_dsts[warp_id]       = local_best_dst;
        warp_hops[warp_id]       = local_best_hop;
        warp_candidates[warp_id] = warp_candidate_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane]     : -COST_INFINITY;
        int block_best_src     = lane < warp_count ? warp_srcs[lane]       : -1;
        int block_best_dst     = lane < warp_count ? warp_dsts[lane]       : -1;
        int block_best_hop     = lane < warp_count ? warp_hops[lane]       : -1;
        int block_candidates   = lane < warp_count ? warp_candidates[lane] : 0;

        block_candidates = warp_reduce_sum(block_candidates);
        warp_reduce_best(block_best_value, block_best_src, block_best_dst, block_best_hop);

        if (lane == 0) {
            decision->step         += 1;
            decision->selected_src  = block_best_src;
            decision->selected_dst  = block_best_dst;
            decision->first_hop     = block_best_hop;
            decision->selected_value = block_best_value;
            decision->halted        = block_best_hop == HALT_NODE ? 1 : 0;
            decision->blocked       = block_candidates == 0 ? 1 : 0;
        }
    }
}

extern "C" __global__ void tensor_quantale_project_batch(
    const float* tensor,
    const int*   witness,
    const int*   consumed,
    const int*   active,
    const ProjectionBias* bias,
    const int*   group_nodes,
    int          group_len,
    const DecisionReport* current_decision,
    DecisionReport* out_decisions
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float alpha = bias[0].confidence, beta = bias[0].cost, gamma = bias[0].safety;
    int next_step = current_decision[0].step + 1;

    for (int g = 0; g < group_len; ++g) {
        int target_hop = group_nodes[g];
        float best_value = -COST_INFINITY;
        int best_src = -1, best_dst = -1, best_hop = -1, candidates = 0;

        if (target_hop >= 0 && target_hop < N) {
            for (int idx = 0; idx < MATRIX_LEN; ++idx) {
                int src = idx / N, dst = idx % N;
                if (src == dst || active[src] == 0) continue;

                int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
                int eidx = tensor_idx(LAYER_COST,       src, dst);
                int sidx = tensor_idx(LAYER_SAFETY,     src, dst);
                float confidence = tensor[cidx], cost = tensor[eidx], safety = tensor[sidx];
                if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

                int hop = witness[cidx];
                if (hop != target_hop || consumed[src * N + hop] != 0) continue;

                float score = alpha * confidence - beta * cost + gamma * safety;
                candidates += 1;
                if (score > best_value) {
                    best_value = score; best_src = src; best_dst = dst; best_hop = hop;
                }
            }
        }

        out_decisions[g].step           = next_step;
        out_decisions[g].selected_src   = best_src;
        out_decisions[g].selected_dst   = best_dst;
        out_decisions[g].first_hop      = best_hop;
        out_decisions[g].selected_value = best_value;
        out_decisions[g].halted         = best_hop == HALT_NODE ? 1 : 0;
        out_decisions[g].blocked        = candidates == 0 ? 1 : 0;
    }
}

extern "C" __global__ void tensor_quantale_commit_batch(
    int* consumed,
    int* active,
    int* next_active,
    const DecisionReport* decisions,
    int  decision_count,
    DecisionReport* current_decision
) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) next_active[i] = 0;
    __syncthreads();

    if (tid == 0) {
        int committed = 0;
        for (int i = 0; i < decision_count; ++i) {
            int src = decisions[i].selected_src, hop = decisions[i].first_hop;
            if (decisions[i].blocked == 0 && decisions[i].halted == 0
                    && src >= 0 && src < N && hop >= 0 && hop < N) {
                consumed[src * N + hop] = 1;
                next_active[hop] = 1;
                committed += 1;
            }
        }
        if (committed == 0) {
            for (int i = 0; i < N; ++i) next_active[i] = active[i];
        }
        current_decision[0].step += committed > 0 ? 1 : 0;
        if (committed > 0) {
            current_decision[0].selected_src   = decisions[0].selected_src;
            current_decision[0].selected_dst   = decisions[0].selected_dst;
            current_decision[0].first_hop      = decisions[0].first_hop;
            current_decision[0].selected_value = decisions[0].selected_value;
            current_decision[0].halted         = 0;
            current_decision[0].blocked        = 0;
        }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) active[i] = next_active[i];
}

extern "C" __global__ void tensor_quantale_drain_queue(
    float* tensor,
    const ExecutionReceipt* receipts,
    int count
) {
    for (int event_idx = threadIdx.x; event_idx < count; event_idx += blockDim.x) {
        ExecutionReceipt r = receipts[event_idx];
        if (r.src < 0 || r.src >= N || r.dst < 0 || r.dst >= N || r.src == r.dst) continue;

        int cidx = tensor_idx(LAYER_CONFIDENCE, r.src, r.dst);
        int eidx = tensor_idx(LAYER_COST,       r.src, r.dst);
        int sidx = tensor_idx(LAYER_SAFETY,     r.src, r.dst);

        if (r.outcome == 0) { // success
            // q_join(x, 1.0) = 1.0 always — safe to atomicExch.
            atomicExch(&tensor[cidx], 1.0f);
            // Halve cost only if above threshold; CAS loop preserves atomicity.
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
        } else if (r.outcome == 1) { // failure
            atomic_float_mul(&tensor[cidx], 0.1f);
            atomic_float_add_capped(&tensor[eidx], 10.0f);
            atomic_float_mul(&tensor[sidx], 0.5f);
        } else if (r.outcome == 2) { // timeout
            atomic_float_mul(&tensor[cidx], 0.5f);
            atomic_float_add_capped(&tensor[eidx], 100.0f);
        } else if (r.outcome == 3) { // safety violation
            atomic_float_mul(&tensor[cidx], 0.25f);
            atomic_float_add_capped(&tensor[eidx], 25.0f);
            atomicExch(&tensor[sidx], BOTTOM);
        }
    }
}

extern "C" __global__ void tensor_quantale_decay(float* tensor, float factor) {
    float decay = q_clamp(factor);
    for (int idx = threadIdx.x; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N, dst = idx % N;
        if (src == dst) continue;
        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST,       src, dst);
        int sidx = tensor_idx(LAYER_SAFETY,     src, dst);
        tensor[cidx] *= decay;
        if (tensor[eidx] < COST_INFINITY)
            tensor[eidx] = tensor_cost_compose(tensor[eidx], 1.0f - decay);
        tensor[sidx] *= decay;
    }
}

extern "C" __global__ void tensor_quantale_frontier_step(
    const float* tensor,
    const int*   witness,
    int*   consumed,
    int*   active,
    int*   next_active,
    const ProjectionBias* bias,
    DecisionReport* decision
) {
    int tid = threadIdx.x, lane = tid & 31, warp_id = tid >> 5;
    int warp_count = (blockDim.x + 31) >> 5;

    __shared__ float warp_values[32];
    __shared__ int   warp_srcs[32], warp_dsts[32], warp_hops[32], warp_candidates[32];

    for (int i = tid; i < N; i += blockDim.x) next_active[i] = 0;
    __syncthreads();

    float local_best_value = -COST_INFINITY;
    int local_best_src = -1, local_best_dst = -1, local_best_hop = -1, local_candidates = 0;
    float alpha = bias[0].confidence, beta = bias[0].cost, gamma = bias[0].safety;

    for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
        int src = idx / N, dst = idx % N;
        if (src == dst || active[src] == 0) continue;

        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST,       src, dst);
        int sidx = tensor_idx(LAYER_SAFETY,     src, dst);
        float confidence = tensor[cidx], cost = tensor[eidx], safety = tensor[sidx];
        if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

        int hop = witness[cidx];
        if (hop < 0 || hop >= N || consumed[src * N + hop] != 0) continue;

        float score = alpha * confidence - beta * cost + gamma * safety;
        local_candidates += 1;
        choose_best(score, src, dst, hop, local_best_value, local_best_src, local_best_dst, local_best_hop);
    }

    int warp_candidate_count = warp_reduce_sum(local_candidates);
    warp_reduce_best(local_best_value, local_best_src, local_best_dst, local_best_hop);

    if (lane == 0) {
        warp_values[warp_id]     = local_best_value;
        warp_srcs[warp_id]       = local_best_src;
        warp_dsts[warp_id]       = local_best_dst;
        warp_hops[warp_id]       = local_best_hop;
        warp_candidates[warp_id] = warp_candidate_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_value = lane < warp_count ? warp_values[lane]     : -COST_INFINITY;
        int block_best_src     = lane < warp_count ? warp_srcs[lane]       : -1;
        int block_best_dst     = lane < warp_count ? warp_dsts[lane]       : -1;
        int block_best_hop     = lane < warp_count ? warp_hops[lane]       : -1;
        int block_candidates   = lane < warp_count ? warp_candidates[lane] : 0;

        block_candidates = warp_reduce_sum(block_candidates);
        warp_reduce_best(block_best_value, block_best_src, block_best_dst, block_best_hop);

        if (lane == 0) {
            decision->step         += 1;
            decision->selected_src  = block_best_src;
            decision->selected_dst  = block_best_dst;
            decision->first_hop     = block_best_hop;
            decision->selected_value = block_best_value;
            decision->halted        = block_best_hop == HALT_NODE ? 1 : 0;
            decision->blocked       = block_candidates == 0 ? 1 : 0;

            if (block_candidates > 0 && block_best_src >= 0 && block_best_hop >= 0) {
                consumed[block_best_src * N + block_best_hop] = 1;
                next_active[block_best_hop] = 1;
            } else {
                for (int i = 0; i < N; ++i) next_active[i] = active[i];
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) active[i] = next_active[i];
}

// ── exploration ───────────────────────────────────────────────────────────

struct ExplorationToken {
    int strategy_id;
    int node;
    int depth;
    float confidence, cost, safety, novelty, receipt_prior, entropy;
    int parent;
};

struct ExplorationCandidate {
    int token_id;
    int first_hop;
    int terminal_node;
    float value;
};

__device__ __forceinline__ float exploration_novelty(int node) {
    return ((node % 7) + 1) * 0.1f;
}

__device__ __forceinline__ float exploration_entropy(int node) {
    return ((node % 5) + 1) * 0.1f;
}

__device__ __forceinline__ float exploration_token_value(
    const ExplorationToken& token,
    float novelty_weight, float receipt_weight, float entropy_penalty
) {
    float depth = (float)(token.depth + 1);
    return (token.confidence / depth) - (token.cost / depth) + (token.safety / depth)
        + novelty_weight * (token.novelty / depth)
        + receipt_weight * (token.receipt_prior / depth)
        - entropy_penalty * (token.entropy / depth);
}

__device__ int exploration_first_hop(const ExplorationToken* tokens, int token_id) {
    int current = token_id, parent = tokens[current].parent;
    while (parent >= 0) {
        int grandparent = tokens[parent].parent;
        if (grandparent < 0) return tokens[current].node;
        current = parent;
        parent = grandparent;
    }
    return tokens[token_id].node;
}

extern "C" __global__ void tensor_quantale_seed_exploration(
    const float* tensor,
    const int*   strategy_nodes,
    const ProjectionBias* strategy_biases,
    const float* receipt_priors,
    int  strategy_count,
    int  max_tokens,
    ExplorationToken* tokens,
    float* scores,
    int*   parents,
    int*   token_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    token_count[0] = 0;
    for (int i = 0; i < max_tokens; ++i) {
        tokens[i].strategy_id = -1; tokens[i].node = -1; tokens[i].depth = -1;
        tokens[i].confidence = 0.0f; tokens[i].cost = COST_INFINITY; tokens[i].safety = 0.0f;
        tokens[i].novelty = 0.0f; tokens[i].receipt_prior = 0.0f; tokens[i].entropy = 0.0f;
        tokens[i].parent = -1;
        scores[i] = -COST_INFINITY;
        parents[i] = -1;
    }
    for (int sid = 0; sid < strategy_count && token_count[0] < max_tokens; ++sid) {
        int node = strategy_nodes[sid];
        if (node < 0 || node >= N) continue;
        float confidence = tensor[tensor_idx(LAYER_CONFIDENCE, START_NODE, node)];
        float safety     = tensor[tensor_idx(LAYER_SAFETY,     START_NODE, node)];
        float cost       = tensor[tensor_idx(LAYER_COST,       START_NODE, node)];
        if (confidence <= BOTTOM && safety <= BOTTOM) continue;
        int out = token_count[0]++;
        tokens[out].strategy_id   = sid;
        tokens[out].node          = node;
        tokens[out].depth         = 0;
        tokens[out].confidence    = confidence * strategy_biases[sid].confidence;
        tokens[out].cost          = cost       * strategy_biases[sid].cost;
        tokens[out].safety        = safety     * strategy_biases[sid].safety;
        tokens[out].novelty       = exploration_novelty(node);
        tokens[out].receipt_prior = receipt_priors[node];
        tokens[out].entropy       = exploration_entropy(node);
        tokens[out].parent        = -1;
        parents[out] = -1;
    }
}

extern "C" __global__ void tensor_quantale_expand_tokens(
    const float* tensor,
    int* token_count,
    int  source_depth, int max_depth, int max_tokens,
    ExplorationToken* tokens,
    int* parents
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int original_count = token_count[0];
    for (int parent_id = 0; parent_id < original_count && token_count[0] < max_tokens; ++parent_id) {
        ExplorationToken parent = tokens[parent_id];
        if (parent.depth != source_depth || parent.depth >= max_depth) continue;
        int src = parent.node;
        if (src < 0 || src >= N) continue;
        for (int dst = 0; dst < N && token_count[0] < max_tokens; ++dst) {
            if (dst == src) continue;
            float confidence = tensor[tensor_idx(LAYER_CONFIDENCE, src, dst)];
            float safety     = tensor[tensor_idx(LAYER_SAFETY,     src, dst)];
            float cost       = tensor[tensor_idx(LAYER_COST,       src, dst)];
            if (confidence <= BOTTOM && safety <= BOTTOM) continue;
            if (cost >= COST_INFINITY) cost = 0.0f;
            int out = token_count[0]++;
            tokens[out].strategy_id   = parent.strategy_id;
            tokens[out].node          = dst;
            tokens[out].depth         = parent.depth + 1;
            tokens[out].confidence    = parent.confidence + confidence;
            tokens[out].cost          = parent.cost + cost;
            tokens[out].safety        = parent.safety + safety;
            tokens[out].novelty       = parent.novelty + exploration_novelty(dst);
            tokens[out].receipt_prior = parent.receipt_prior;
            tokens[out].entropy       = parent.entropy + exploration_entropy(dst);
            tokens[out].parent        = parent_id;
            parents[out] = parent_id;
        }
    }
}

extern "C" __global__ void tensor_quantale_score_tokens(
    const ExplorationToken* tokens,
    const int* token_count,
    float novelty_weight, float receipt_weight, float entropy_penalty,
    float* scores
) {
    int count = token_count[0];
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x)
        scores[idx] = exploration_token_value(tokens[idx], novelty_weight, receipt_weight, entropy_penalty);
}

extern "C" __global__ void tensor_quantale_select_topk_tokens(
    const ExplorationToken* tokens,
    const float* scores,
    const int*   token_count,
    int  beam_width,
    float repeat_penalty,
    int  max_terminal_visits,
    int  max_first_hop_visits,
    const int* terminal_visits,
    const int* first_hop_visits,
    ExplorationCandidate* selected,
    int* selected_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int count = token_count[0];
    int limit = beam_width < N ? beam_width : N;
    selected_count[0] = 0;
    for (int slot = 0; slot < limit; ++slot) {
        float best = -COST_INFINITY;
        int best_id = -1, best_hop = -1;
        for (int i = 0; i < count; ++i) {
            int terminal = tokens[i].node;
            if (terminal < 0 || terminal >= N) continue;
            int hop = exploration_first_hop(tokens, i);
            if (hop < 0 || hop >= N) continue;
            if (terminal_visits[terminal] >= max_terminal_visits
                    || first_hop_visits[hop] >= max_first_hop_visits) continue;
            bool already = false;
            for (int j = 0; j < selected_count[0]; ++j) {
                if (selected[j].terminal_node == terminal || selected[j].first_hop == hop) {
                    already = true; break;
                }
            }
            if (already) continue;
            float adjusted = scores[i]
                - repeat_penalty * (float)(terminal_visits[terminal] + first_hop_visits[hop]);
            if (adjusted > best) { best = adjusted; best_id = i; best_hop = hop; }
        }
        if (best_id < 0) break;
        int out = selected_count[0]++;
        selected[out].token_id     = best_id;
        selected[out].first_hop    = best_hop;
        selected[out].terminal_node = tokens[best_id].node;
        selected[out].value        = best;
    }
}

extern "C" __global__ void tensor_quantale_commit_exploration(
    int*   consumed,
    int*   active,
    int*   next_active,
    const ExplorationCandidate* candidate,
    DecisionReport* decision
) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) next_active[i] = 0;
    __syncthreads();
    if (tid == 0) {
        int hop      = candidate[0].first_hop;
        int terminal = candidate[0].terminal_node;
        int blocked  = hop < 0 || hop >= N;
        decision[0].step         += blocked ? 0 : 1;
        decision[0].selected_src  = START_NODE;
        decision[0].selected_dst  = terminal;
        decision[0].first_hop     = hop;
        decision[0].selected_value = candidate[0].value;
        decision[0].halted        = hop == HALT_NODE ? 1 : 0;
        decision[0].blocked       = blocked;
        if (!blocked) {
            consumed[START_NODE * N + hop] = 1;
            next_active[hop] = 1;
        } else {
            for (int i = 0; i < N; ++i) next_active[i] = active[i];
        }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) active[i] = next_active[i];
}

// ── JIT chain scoring ──────────────────────────────────────────────────────
//
// Scores dynamically detected chain descriptors. No operator names or slot names
// appear here; all policy is numeric chain metadata supplied by Rust.

struct JitChainMetadata {
    int chain_len;
    int input_count;
    int output_count;
    float estimated_savings;
    int target_node_id;
};

extern "C" __global__ void jit_chain_score_embed(
    float* tensor,
    const JitChainMetadata* chains,
    int   chain_count,
    int   src_node
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chain_count) return;
    if (src_node < 0 || src_node >= N) return;

    JitChainMetadata chain = chains[idx];
    int target = chain.target_node_id;
    if (target < 0 || target >= N || chain.chain_len <= 0) return;

    float base_conf = tensor[tensor_idx(LAYER_CONFIDENCE, src_node, target)];
    float base_cost = tensor[tensor_idx(LAYER_COST,       src_node, target)];
    float base_safe = tensor[tensor_idx(LAYER_SAFETY,     src_node, target)];
    if (base_conf <= BOTTOM) base_conf = 0.55f;
    if (base_safe <= BOTTOM) base_safe = 0.75f;
    if (base_cost >= COST_INFINITY) base_cost = 10.0f;

    float savings = fmaxf(0.0f, chain.estimated_savings);
    float fused_conf   = q_clamp(base_conf + 0.02f * savings);
    float fused_safety = q_clamp(base_safe + 0.01f * savings);
    float fused_cost   = fmaxf(0.001f, base_cost / (1.0f + savings));

    int cidx = tensor_idx(LAYER_CONFIDENCE, src_node, target);
    int eidx = tensor_idx(LAYER_COST,       src_node, target);
    int sidx = tensor_idx(LAYER_SAFETY,     src_node, target);

    if (fused_conf > tensor[cidx]) {
        tensor[cidx] = fused_conf;
        tensor[sidx] = fused_safety;
    }
    // Cost uses min-plus: only write if genuinely cheaper.
    if (fused_cost < tensor[eidx]) {
        tensor[eidx] = fused_cost;
    }
}

extern "C" __global__ void tensor_quantale_tick(
    float* tensor,
    float* scratch,
    int*   witness,
    int*   consumed,
    int*   active,
    int*   next_active,
    const ProjectionBias* bias,
    DecisionReport* decision
) {
    tensor_quantale_closure(tensor, scratch, witness);
    __syncthreads();
    tensor_quantale_frontier_step(tensor, witness, consumed, active, next_active, bias, decision);
}

// ── Device-side receipt drain ─────────────────────────────────────────────────
//
// Processes completed DeviceReceipts from the ring buffer entirely on-device,
// applying the same tensor updates as tensor_quantale_drain_queue but without
// any CPU round-trip.  ring_head is the consumer index (advances per receipt
// processed); ring_tail is the producer index written by gpu_dispatch.

extern "C" __global__ void tensor_quantale_drain_device_receipts(
    float*               tensor,
    const DeviceReceipt* receipt_ring,
    int                  ring_size,
    int*                 ring_head,
    const int*           ring_tail
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int head = *ring_head;
    int tail = *ring_tail;

    while (head != tail) {
        int slot = head % ring_size;
        DeviceReceipt r = receipt_ring[slot];

        if (r.valid && r.src >= 0 && r.src < N && r.dst >= 0 && r.dst < N && r.src != r.dst) {
            int cidx = tensor_idx(LAYER_CONFIDENCE, r.src, r.dst);
            int eidx = tensor_idx(LAYER_COST,       r.src, r.dst);
            int sidx = tensor_idx(LAYER_SAFETY,     r.src, r.dst);

            if (r.outcome == 0) { // success
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
            } else if (r.outcome == 1) { // failure
                atomic_float_mul(&tensor[cidx], 0.1f);
                atomic_float_add_capped(&tensor[eidx], 10.0f);
                atomic_float_mul(&tensor[sidx], 0.5f);
            } else if (r.outcome == 2) { // timeout
                atomic_float_mul(&tensor[cidx], 0.5f);
                atomic_float_add_capped(&tensor[eidx], 100.0f);
            } else if (r.outcome == 3) { // safety violation
                atomic_float_mul(&tensor[cidx], 0.25f);
                atomic_float_add_capped(&tensor[eidx], 50.0f);
                atomicExch(&tensor[sidx], 0.0f);
            }
        }

        head++;
    }

    *ring_head = head;
}

// ── Device-ring push / pop ────────────────────────────────────────────────────

// Write n floats into the ring from src (single-threaded for head/tail safety).
extern "C" __global__ void device_ring_push(
    float* ring, int* tail, int capacity,
    const float* src, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = *tail;
    for (int i = 0; i < n; ++i) {
        ring[t % capacity] = src[i];
        ++t;
    }
    *tail = t;
}

// Read n floats from the ring into dst (single-threaded for head/tail safety).
extern "C" __global__ void device_ring_pop(
    const float* ring, int* head, int capacity,
    float* dst, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *head;
    for (int i = 0; i < n; ++i) {
        dst[i] = ring[h % capacity];
        ++h;
    }
    *head = h;
}

// ── GPU region __device__ functions ──────────────────────────────────────────
//
// One __device__ function per hot region.  The dispatch kernel selects the
// right function via a switch table and calls it.  When slot_ptrs == NULL the
// function runs in receipt-only mode: it records output_flags but performs no
// element-wise work. Passing a DeviceSlotRegistry-built float** table enables
// true in-kernel computation.
//
// Slot layout per region matches regions.hot.json / operators.generated.json.

__device__ void region_vector_add(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 0);
    if (!slot_ptrs || n <= 0) return;
    float* a   = slot_ptrs[0];  // math.a
    float* b   = slot_ptrs[1];  // math.b
    float* out = slot_ptrs[2];  // math.add_out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = a[i] + b[i];
}

__device__ void region_vector_scale(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 1);
    if (!slot_ptrs || n <= 0) return;
    float* x   = slot_ptrs[0];  // math.add_out
    float* s   = slot_ptrs[1];  // math.scale
    float* out = slot_ptrs[2];  // math.out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = x[i] * s[i];
}

__device__ void region_fused_add_scale(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 2);
    if (!slot_ptrs || n <= 0) return;
    float* a   = slot_ptrs[0];  // math.a
    float* b   = slot_ptrs[1];  // math.b
    float* s   = slot_ptrs[2];  // math.scale
    float* out = slot_ptrs[3];  // math.out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (a[i] + b[i]) * s[i];
}

__device__ void region_analysis_return1(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 3);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* open  = slot_ptrs[1];  // market.open
    float* out   = slot_ptrs[2];  // analysis.return
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (price[i] - open[i]) / (open[i] + 1e-8f);
}

__device__ void region_analysis_volatility(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 4);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* ret   = slot_ptrs[1];  // analysis.return
    float* out   = slot_ptrs[2];  // analysis.volatility
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = fabsf(price[i] - ret[i]) / (ret[i] + 1e-8f);
}

__device__ void region_analysis_signal_score(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 5);
    if (!slot_ptrs || n <= 0) return;
    float* ret = slot_ptrs[0];  // analysis.return
    float* vol = slot_ptrs[1];  // analysis.volatility
    float* out = slot_ptrs[2];  // analysis.signal_score
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = ret[i] / (1.0f + fabsf(vol[i]));
}

__device__ void region_commit_receipt(float** slot_ptrs, int n, DeviceReceipt* r) {
    (void)slot_ptrs; (void)n;
    r->output_flags |= (1 << 6);
}

// ── GPU-side region dispatch ──────────────────────────────────────────────────
//
// Selects the appropriate __device__ region function via a switch table and
// writes a DeviceReceipt to the ring — all without returning to the CPU.
//
// When slot_ptrs == NULL the region functions run in receipt-only mode; pass
// actual device slot pointer arrays plus element_count to enable true in-kernel
// computation.

extern "C" __global__ void tensor_quantale_gpu_dispatch(
    const GpuDispatchMailbox* mailbox,
    DeviceReceipt*            receipt_ring,
    int*                      ring_tail,
    int                       ring_size,
    int                       region_count,
    float**                   slot_ptrs,
    int                       element_count
) {
    int rid = mailbox->pending_region_id;
    if (rid < 0 || rid >= region_count) return;

    DeviceReceipt r;
    r.region_id    = rid;
    r.src          = mailbox->src_node;
    r.dst          = mailbox->dst_node;
    r.outcome      = mailbox->outcome;
    r.latency      = 0.0f;
    r.valid        = 1;
    r.output_flags = 0;

    switch (rid) {
        case 0: region_vector_add           (slot_ptrs, element_count, &r); break;
        case 1: region_vector_scale         (slot_ptrs, element_count, &r); break;
        case 2: region_fused_add_scale      (slot_ptrs, element_count, &r); break;
        case 3: region_analysis_return1     (slot_ptrs, element_count, &r); break;
        case 4: region_analysis_volatility  (slot_ptrs, element_count, &r); break;
        case 5: region_analysis_signal_score(slot_ptrs, element_count, &r); break;
        case 6: region_commit_receipt       (slot_ptrs, element_count, &r); break;
        default: break;
    }

    __syncthreads();

    // Only thread 0 appends to the receipt ring.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int tail = *ring_tail;
        int slot = tail % ring_size;
        receipt_ring[slot] = r;
        *ring_tail = tail + 1;
    }
}

// ── GPU-native parallel group selection + commit ──────────────────────────────
//
// tensor_quantale_par_group_step: single kernel that selects the first eligible,
// all-ready CKA par group, commits it atomically, and writes the result without
// any CPU round-trip for group selection or effect validation.
//
// Eligibility (jit_cuda/fusion/hot) is precomputed on the CPU at epoch start
// and passed as a static int[] mask — 0=ineligible, 1=eligible.
//
// Table layout: [g0_size, g0_n0, g0_r0, g0_d0, g0_n1, g0_r1, g0_d1, ..., g1_size, ...]
// Each member is a (node_id, region_id, is_gpu_dispatchable) triple.
//   region_id            = -1 for non-hot members (fusion-entry and pure-CPU alike)
//   is_gpu_dispatchable  = 1 for hot-region or fusion-entry; 0 for pure-CPU
// (num_groups is passed as a separate int)
//
// Eligibility is computed on-device: a group is eligible iff all members have
// is_gpu_dispatchable == 1.  No separate eligible[] array is required.
//
// Output: ParGroupStepOutput written to *result.
//   selected_group_idx = -1  → no group was ready; tensor state unchanged.
//   selected_group_idx >= 0  → group committed; consumed/active/decision updated.
//   region_ids[i]            → hot-region id for member i (-1 if not a hot region).
//
// Single-thread kernel (tid==0, bid==0).  N is at most 256 in practice so the
// O(N²) inner loop is cheap relative to kernel-launch overhead.

#define MAX_PAR_GROUP_SIZE 8
#define PAR_DISPATCH_NONE 0
#define PAR_DISPATCH_HF_DEVICE 1
#define PAR_DISPATCH_HOST_FALLBACK 2

struct ParDispatchDescriptor {
    int member_index;
    int node_id;
    int region_id;
    int dispatch_kind;
    int src_node;
    int dst_node;
};

struct ParGroupStepOutput {
    int selected_group_idx;
    int group_size;
    struct DecisionReport decisions[MAX_PAR_GROUP_SIZE];
    int region_ids[MAX_PAR_GROUP_SIZE];         // hot-region id per member; -1 if not hot
    int dispatched_on_device[MAX_PAR_GROUP_SIZE]; // 1 = dispatched in-kernel via H_f path
    struct ParDispatchDescriptor dispatch_descriptors[MAX_PAR_GROUP_SIZE];
};

// ── H_f dispatch parameter bundle ────────────────────────────────────────────
//
// Packs the six inline-dispatch parameters into a single device struct so that
// tensor_quantale_par_group_step stays within the cudarc LaunchAsync arity
// limit (max 13-tuple).

struct ParGroupHfParams {
    const unsigned long long* slot_table_ptrs;  // [num_groups * MAX_PAR_GROUP_SIZE] float** addrs
    const int*                element_counts;   // [num_groups * MAX_PAR_GROUP_SIZE]
    DeviceReceipt*            receipt_ring;
    int*                      ring_tail;
    int                       ring_size;
    int                       region_count;
};

// ── tensor_quantale_par_group_step ───────────────────────────────────────────
//
// Two-phase single-block kernel:
//
//   Phase 1 (thread 0 only): iterate the par_table, find the first eligible
//   all-ready group, commit it (consumed/active/decision updated), populate
//   shared memory with the selected group's region_ids, src, dst.
//
//   Phase 2 (all 512 threads, block 0 only): for each committed member with
//   a registered hot-region id, call the precompiled __device__ region function
//   with per-member slot_ptrs (H_f path).  Thread 0 writes the DeviceReceipt to
//   the ring.  Members without per-member slot tables are dispatched in
//   receipt-only mode (slot_ptrs == NULL); the CPU skips execute_*_blocking for
//   those members as well since the receipt captures the success outcome.
//
// This closes D_h for hot-region par members without CUDA dynamic parallelism.
// Fusion/abstract par members (region_id == -1) are not dispatched in Phase 2;
// the CPU still calls execute_*_blocking for them (D_h partial).

extern "C" __global__ void tensor_quantale_par_group_step(
    const float*            tensor,
    const int*              witness,
    int*                    consumed,
    int*                    active,
    int*                    next_active,
    const ProjectionBias*   bias,
    DecisionReport*         global_decision,
    const int*              par_table,
    int                     num_groups,
    ParGroupStepOutput*     result,
    const ParGroupHfParams* hf    // H_f inline dispatch config (slot tables + receipt ring)
) {
    // Shared memory: Phase 1 (thread 0) writes, Phase 2 (all threads) reads.
    __shared__ int sh_selected_g;
    __shared__ int sh_sz;
    __shared__ int sh_region_ids[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_srcs[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_dsts[MAX_PAR_GROUP_SIZE];

    // ── init ──────────────────────────────────────────────────────────────────
    if (threadIdx.x == 0) {
        sh_selected_g = -1;
        sh_sz = 0;
        result->selected_group_idx = -1;
        result->group_size = 0;
        for (int i = 0; i < MAX_PAR_GROUP_SIZE; i++) {
            sh_region_ids[i] = -1;
            sh_srcs[i] = -1;
            sh_dsts[i] = -1;
            result->dispatched_on_device[i] = 0;
            result->dispatch_descriptors[i].member_index = -1;
            result->dispatch_descriptors[i].node_id = -1;
            result->dispatch_descriptors[i].region_id = -1;
            result->dispatch_descriptors[i].dispatch_kind = PAR_DISPATCH_NONE;
            result->dispatch_descriptors[i].src_node = -1;
            result->dispatch_descriptors[i].dst_node = -1;
        }
    }
    __syncthreads();

    // ── Phase 1: group selection and commit (thread 0 only) ───────────────────
    if (threadIdx.x == 0) {
        float alpha = bias[0].confidence;
        float beta  = bias[0].cost;
        float gamma = bias[0].safety;
        int next_step = global_decision[0].step + 1;

        int ptr = 0;
        for (int g = 0; g < num_groups; g++) {
            int sz = par_table[ptr++];
            int group_start = ptr;
            ptr += sz * 3;

            if (sz < 2 || sz > MAX_PAR_GROUP_SIZE) continue;

            // On-device eligibility: all members must be GPU-dispatchable.
            int all_eligible = 1;
            for (int i = 0; i < sz; i++) {
                if (!par_table[group_start + i * 3 + 2]) { all_eligible = 0; break; }
            }
            if (!all_eligible) continue;

            bool all_ready = true;
            struct DecisionReport decisions[MAX_PAR_GROUP_SIZE];
            int region_ids[MAX_PAR_GROUP_SIZE];
            int node_ids[MAX_PAR_GROUP_SIZE];

            for (int i = 0; i < sz; i++) {
                int target_hop = par_table[group_start + i * 3];
                node_ids[i]    = target_hop;
                region_ids[i]  = par_table[group_start + i * 3 + 1];
                float best_value = -1.0e30f;
                int best_src = -1, best_dst = -1, best_hop = -1, candidates = 0;

                if (target_hop >= 0 && target_hop < N) {
                    for (int idx = 0; idx < MATRIX_LEN; idx++) {
                        int src = idx / N, dst = idx % N;
                        if (src == dst || active[src] == 0) continue;

                        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
                        float confidence = tensor[cidx];
                        float cost       = tensor[tensor_idx(LAYER_COST,   src, dst)];
                        float safety     = tensor[tensor_idx(LAYER_SAFETY, src, dst)];
                        if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

                        int hop = witness[cidx];
                        if (hop != target_hop || consumed[src * N + hop] != 0) continue;

                        float score = alpha * confidence - beta * cost + gamma * safety;
                        candidates++;
                        if (score > best_value) {
                            best_value = score;
                            best_src = src; best_dst = dst; best_hop = hop;
                        }
                    }
                }

                decisions[i].step           = next_step;
                decisions[i].selected_src   = best_src;
                decisions[i].selected_dst   = best_dst;
                decisions[i].first_hop      = best_hop;
                decisions[i].selected_value = best_value;
                decisions[i].halted         = (best_hop == HALT_NODE) ? 1 : 0;
                decisions[i].blocked        = (candidates == 0) ? 1 : 0;

                if (decisions[i].blocked || decisions[i].halted) { all_ready = false; break; }
            }

            if (!all_ready) continue;

            // Commit atomically.
            for (int i = 0; i < N; i++) next_active[i] = 0;
            for (int i = 0; i < sz; i++) {
                int src = decisions[i].selected_src;
                int hop = decisions[i].first_hop;
                if (src >= 0 && src < N && hop >= 0 && hop < N) {
                    consumed[src * N + hop] = 1;
                    next_active[hop] = 1;
                }
            }
            for (int i = 0; i < N; i++) active[i] = next_active[i];

            global_decision[0].step           = next_step;
            global_decision[0].selected_src   = decisions[0].selected_src;
            global_decision[0].selected_dst   = decisions[0].selected_dst;
            global_decision[0].first_hop      = decisions[0].first_hop;
            global_decision[0].selected_value = decisions[0].selected_value;
            global_decision[0].halted         = 0;
            global_decision[0].blocked        = 0;

            result->selected_group_idx = g;
            result->group_size = sz;
            for (int i = 0; i < sz; i++) {
                result->decisions[i]  = decisions[i];
                result->region_ids[i] = region_ids[i];
                result->dispatch_descriptors[i].member_index = i;
                result->dispatch_descriptors[i].node_id = node_ids[i];
                result->dispatch_descriptors[i].region_id = region_ids[i];
                result->dispatch_descriptors[i].dispatch_kind = PAR_DISPATCH_HOST_FALLBACK;
                result->dispatch_descriptors[i].src_node = decisions[i].selected_src;
                result->dispatch_descriptors[i].dst_node = decisions[i].first_hop;
            }

            // Broadcast to shared memory for Phase 2.
            sh_selected_g = g;
            sh_sz = sz;
            for (int i = 0; i < sz; i++) {
                sh_region_ids[i] = region_ids[i];
                sh_srcs[i] = decisions[i].selected_src;
                sh_dsts[i] = decisions[i].selected_dst;
            }

            break;
        }
    }
    __syncthreads();

    // ── Phase 2: hot-region inline dispatch (H_f path, all threads) ──────────
    //
    // Only runs when a group was selected and the receipt ring is available.
    // All threads see the same sh_* values → no warp divergence in the loop
    // guard checks.  The region __device__ functions use threadIdx.x for
    // element-wise parallelism; thread 0 writes the receipt to the ring.

    if (sh_selected_g < 0) return;
    if (!hf || !hf->receipt_ring || !hf->ring_tail || hf->ring_size <= 0 || hf->region_count <= 0) return;

    for (int i = 0; i < sh_sz; i++) {
        int rid = sh_region_ids[i];
        if (rid < 0 || rid >= hf->region_count) continue;

        int table_idx = sh_selected_g * MAX_PAR_GROUP_SIZE + i;
        unsigned long long slot_table_ptr = hf->slot_table_ptrs[table_idx];
        int elem_count = hf->element_counts[table_idx];
        float** slot_ptrs = slot_table_ptr ? (float**)slot_table_ptr : (float**)0;

        // Each thread keeps a local receipt; only thread 0's copy is written to
        // the ring (same pattern as tensor_quantale_gpu_dispatch).
        DeviceReceipt r;
        r.region_id    = rid;
        r.src          = sh_srcs[i];
        r.dst          = sh_dsts[i];
        r.outcome      = 0; // inline execution always succeeds at the GPU level
        r.latency      = 0.0f;
        r.valid        = 1;
        r.output_flags = 0;

        int handled = 1;
        switch (rid) {
            case 0: region_vector_add           (slot_ptrs, elem_count, &r); break;
            case 1: region_vector_scale         (slot_ptrs, elem_count, &r); break;
            case 2: region_fused_add_scale      (slot_ptrs, elem_count, &r); break;
            case 3: region_analysis_return1     (slot_ptrs, elem_count, &r); break;
            case 4: region_analysis_volatility  (slot_ptrs, elem_count, &r); break;
            case 5: region_analysis_signal_score(slot_ptrs, elem_count, &r); break;
            case 6: region_commit_receipt       (slot_ptrs, elem_count, &r); break;
            default: handled = 0; break;
        }
        __syncthreads();

        if (threadIdx.x == 0 && handled) {
            int tail = *hf->ring_tail;
            hf->receipt_ring[tail % hf->ring_size] = r;
            *hf->ring_tail = tail + 1;
            result->dispatched_on_device[i] = 1;
            result->dispatch_descriptors[i].dispatch_kind = PAR_DISPATCH_HF_DEVICE;
        }
        __syncthreads();
    }
}

// ── Device helper kernels for complex IR ops ──────────────────────────────────
//
// These kernels are called directly (not via the JIT element-wise framework)
// when TypedIrOp::Reduce or TypedIrOp::TopK must be lowered to GPU code.

// Parallel warp-shuffle + shared-memory reduction (sum).
// Launch with <<<1, THREADS>>> where THREADS is a power of two <= 1024.
extern "C" __global__ void quantale_parallel_reduce(
    const float* in, float* out, int n, float init
) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float acc = init;
    for (int i = tid; i < n; i += blockDim.x)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

// Bitonic sort top-k selection: sorts the first BLOCK_SIZE elements in-place
// using shared memory and writes the top-k results to `out`.
// Assumes n <= 1024.  For production use Thrust or CUB.
extern "C" __global__ void quantale_topk_bitonic(
    float* data, float* out, int n, int k
) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? data[tid] : -1.0e30f;
    __syncthreads();

    // Bitonic sort (ascending).
    for (int size = 2; size <= blockDim.x; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = tid ^ stride;
            if (partner > tid) {
                bool asc = ((tid & size) == 0);
                if (asc ? sdata[tid] > sdata[partner]
                        : sdata[tid] < sdata[partner]) {
                    float tmp = sdata[tid];
                    sdata[tid] = sdata[partner];
                    sdata[partner] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // Write top-k (largest = end of ascending sort) to output.
    if (tid < k && (blockDim.x - 1 - tid) < n)
        out[tid] = sdata[blockDim.x - 1 - tid];
}
