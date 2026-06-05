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

