
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
    float* scratch,
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

