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

// ── Device-side receipt drain ─────────────────────────────────────────────────
//
// Processes completed DeviceReceipts from the ring buffer entirely on-device,
// applying tensor updates without any CPU round-trip. ring_head is the consumer index
// (advances per receipt
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

extern "C" __global__ void tensor_quantale_push_device_receipt(
    DeviceReceipt* receipt_ring,
    int*           ring_tail,
    int            ring_size,
    int            region_id,
    int            src,
    int            dst,
    int            outcome
) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || ring_size <= 0) return;

    DeviceReceipt r;
    r.region_id    = region_id;
    r.src          = src;
    r.dst          = dst;
    r.outcome      = outcome;
    r.latency      = 0.0f;
    r.valid        = 1;
    r.output_flags = 0;

    int tail = *ring_tail;
    receipt_ring[tail % ring_size] = r;
    *ring_tail = tail + 1;
}

