// ── Phase-6: device-side learning and exploration updates ─────────────────────
//
// receipt_priors[N]: per-node float updated on-device when receipts arrive.
//   Used directly by tensor_quantale_seed_exploration for exploration scoring
//   without a CPU round-trip.
// LearnedDelta: compact edge-delta record emitted to a ring for CPU persistence.
//   The CPU service drains this ring and writes durable JSONL/state files.
// learned_delta_init:        zero-initializes receipt_priors (block-parallel).
// learned_delta_fold_receipt: folds one receipt into priors + delta ring.
// learned_delta_apply:       drains the delta ring, applying soft updates to tensor.
// receipt_prior_snapshot:    block-parallel copy of priors to an export buffer.

#define LEARNED_DELTA_RING_SIZE 256

struct LearnedDelta {
    int   src;
    int   dst;
    float confidence_delta;
    float cost_delta;
    float safety_delta;
};

// Zero-initialise the per-node receipt_priors table.
// Launch with <<<ceil(n/256), 256>>>.
extern "C" __global__ void learned_delta_init(
    float* receipt_priors,
    int    n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) receipt_priors[tid] = 0.0f;
}

// Fold one receipt into the per-node prior table and push a LearnedDelta entry.
// Updates receipt_priors[node_id] on success only.  Thread 0, block 0 only.
extern "C" __global__ void learned_delta_fold_receipt(
    int           src,
    int           dst,
    int           node_id,
    int           outcome,
    float*        receipt_priors,
    int           n_nodes,
    LearnedDelta* delta_ring,
    int*          delta_tail,
    const int*    delta_head,
    int           ring_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    LearnedDelta d;
    d.src = src;
    d.dst = dst;

    if (outcome == 0) { // success
        if (receipt_priors && node_id >= 0 && node_id < n_nodes)
            receipt_priors[node_id] = fminf(receipt_priors[node_id] + 0.1f, 1.0f);
        d.confidence_delta =  0.1f;
        d.cost_delta       = -0.05f;
        d.safety_delta     =  0.1f;
    } else if (outcome == 1) { // failure
        d.confidence_delta = -0.05f;
        d.cost_delta       =  0.1f;
        d.safety_delta     = -0.05f;
    } else if (outcome == 2) { // timeout
        d.confidence_delta = -0.025f;
        d.cost_delta       =  0.05f;
        d.safety_delta     =  0.0f;
    } else if (outcome == 3) { // safety violation
        d.confidence_delta = -0.1f;
        d.cost_delta       =  0.0f;
        d.safety_delta     = -0.2f;
    } else {
        d.confidence_delta = 0.0f;
        d.cost_delta       = 0.0f;
        d.safety_delta     = 0.0f;
    }

    if (delta_ring && delta_tail && delta_head && ring_size > 0) {
        int t = *delta_tail, h = *delta_head;
        if (t - h < ring_size) {
            delta_ring[t % ring_size] = d;
            *delta_tail = t + 1;
        }
    }
}

// Drain all pending learned deltas from the ring and apply soft updates to the
// live tensor (clamped to [0,1] for confidence/safety, >= 0 for cost).
// Advances *delta_head.  Thread 0, block 0 only.
extern "C" __global__ void learned_delta_apply(
    float*        tensor,
    LearnedDelta* delta_ring,
    int*          delta_head,
    const int*    delta_tail,
    int           ring_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *delta_head;
    int t = *delta_tail;
    while (h != t) {
        LearnedDelta d = delta_ring[h % ring_size];
        if (d.src >= 0 && d.src < N && d.dst >= 0 && d.dst < N && d.src != d.dst) {
            int cidx = tensor_idx(LAYER_CONFIDENCE, d.src, d.dst);
            int eidx = tensor_idx(LAYER_COST,       d.src, d.dst);
            int sidx = tensor_idx(LAYER_SAFETY,     d.src, d.dst);
            tensor[cidx] = fmaxf(0.0f, fminf(1.0f, tensor[cidx] + d.confidence_delta));
            tensor[eidx] = fmaxf(0.0f,              tensor[eidx] + d.cost_delta);
            tensor[sidx] = fmaxf(0.0f, fminf(1.0f, tensor[sidx] + d.safety_delta));
        }
        h++;
    }
    *delta_head = h;
}

// Block-parallel copy of receipt_priors into out_snapshot for CPU export.
extern "C" __global__ void receipt_prior_snapshot(
    const float* receipt_priors,
    float*       out_snapshot,
    int          n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out_snapshot[tid] = receipt_priors[tid];
}

