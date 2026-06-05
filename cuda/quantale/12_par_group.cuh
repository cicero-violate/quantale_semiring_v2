// ── GPU-native parallel group selection + commit ──────────────────────────────
//
// tensor_quantale_par_group_step: single kernel that selects the first eligible,
// all-ready CKA par group, commits it on-device, and writes the result without
// any CPU round-trip for group selection or effect validation.
//
// Eligibility (GPU-native H_f / abstract-device) is encoded per member at epoch start and
// checked by the kernel — 0=ineligible, 1=eligible.
//
// Table layout: [g0_size, g0_n0, g0_r0, g0_e0, g0_k0, g0_n1, ...]
// Each member is a (node_id, region_id, is_gpu_dispatchable, dispatch_kind) tuple.
//   region_id            = -1 for non-hot members (fusion-entry and pure-CPU alike)
//   is_gpu_dispatchable  = 1 for GPU-native dispatch kinds; 0 for host fallback
//   dispatch_kind        = initial descriptor kind; H_f may upgrade to device
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
// Selection remains deterministic on thread 0.  Commit writeback is block-
// parallel: lanes clear next_active, atomically mark consumed edges, set the
// next active frontier, and copy it back to active.

#define MAX_PAR_GROUP_SIZE 8
#define PAR_DISPATCH_NONE 0
#define PAR_DISPATCH_HF_DEVICE 1
#define PAR_DISPATCH_HOST_FALLBACK 2
#define PAR_DISPATCH_FUSION_ENTRY 3
#define PAR_DISPATCH_ABSTRACT_DEVICE 4

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
// Three-phase single-block kernel:
//
//   Phase 1 (all threads): iterate groups deterministically, evaluate each
//   member's readiness with block-parallel reductions over MATRIX_LEN, and
//   publish the first eligible all-ready group.  Thread 0 only materializes the
//   selected result after all lanes have computed readiness; it no longer owns
//   the readiness scan or group-selection predicate.
//
//   Phase 2 (all 512 threads, block 0 only): commit selected member effects by
//   clearing next_active, atomically marking consumed edges, setting next_active,
//   and copying the new frontier back to active.
//
//   Phase 3 (all 512 threads, block 0 only): for each committed member with
//   a registered hot-region id, call the precompiled __device__ region function
//   with per-member slot_ptrs (H_f path).  Thread 0 writes the DeviceReceipt to
//   the ring.  Members without per-member slot tables are dispatched in
//   receipt-only mode (slot_ptrs == NULL); the CPU skips execute_*_blocking for
//   those members as well since the receipt captures the success outcome.
//
// This closes D_h for hot-region par members without CUDA dynamic parallelism.
// Fusion/abstract par members (region_id == -1) are not dispatched in Phase 3;
// the CPU still calls execute_*_blocking for them (D_h partial).

extern "C" __global__ void tensor_quantale_par_group_step(
    const float*            tensor,
    const int*              witness,
    int*                    consumed,
    int*                    active,
    int*                    next_active,
    const ProjectionBias*   bias,
    DecisionReport*         global_decision,
    const int*              group_offsets,
    const int*              par_table,
    int                     num_groups,
    ParGroupStepOutput*     result,
    const ParGroupHfParams* hf    // H_f inline dispatch config (slot tables + receipt ring)
) {
    // Shared memory: Phase 1 writes; commit and H_f lanes read.
    __shared__ int sh_selected_g;
    __shared__ int sh_sz;
    __shared__ int sh_region_ids[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_srcs[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_dsts[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_hops[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_node_ids[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_dispatch_kinds[MAX_PAR_GROUP_SIZE];
    __shared__ int sh_member_ready[MAX_PAR_GROUP_SIZE];
    __shared__ float sh_best_values[MAX_PAR_GROUP_SIZE];
    __shared__ float warp_values[32];
    __shared__ int warp_srcs[32];
    __shared__ int warp_dsts[32];
    __shared__ int warp_hops[32];
    __shared__ int warp_candidates[32];

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
            sh_hops[i] = -1;
            sh_node_ids[i] = -1;
            sh_dispatch_kinds[i] = PAR_DISPATCH_NONE;
            sh_member_ready[i] = 0;
            sh_best_values[i] = -COST_INFINITY;
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

    // ── Phase 1: group selection with block-parallel readiness ───────────────
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warp_count = (blockDim.x + 31) >> 5;
    float alpha = bias[0].confidence;
    float beta  = bias[0].cost;
    float gamma = bias[0].safety;
    int next_step = global_decision[0].step + 1;

    for (int g = 0; g < num_groups; g++) {
        int record_start = group_offsets[g];
        int sz = par_table[record_start];
        int group_start = record_start + 1;

        if (threadIdx.x == 0) {
            sh_sz = (sz >= 2 && sz <= MAX_PAR_GROUP_SIZE) ? sz : 0;
            for (int i = 0; i < MAX_PAR_GROUP_SIZE; i++) {
                sh_region_ids[i] = -1;
                sh_srcs[i] = -1;
                sh_dsts[i] = -1;
                sh_hops[i] = -1;
                sh_node_ids[i] = -1;
                sh_dispatch_kinds[i] = PAR_DISPATCH_NONE;
                sh_member_ready[i] = 0;
                sh_best_values[i] = -COST_INFINITY;
            }
        }
        __syncthreads();

        if (sh_sz == 0) {
            __syncthreads();
            continue;
        }

        for (int m = 0; m < sh_sz; m++) {
            int target_hop = par_table[group_start + m * 4];
            int region_id = par_table[group_start + m * 4 + 1];
            int is_dispatchable = par_table[group_start + m * 4 + 2];
            int dispatch_kind = par_table[group_start + m * 4 + 3];

            float local_best_value = -COST_INFINITY;
            int local_best_src = -1;
            int local_best_dst = -1;
            int local_best_hop = -1;
            int local_candidates = 0;

            if (is_dispatchable && target_hop >= 0 && target_hop < N) {
                for (int idx = tid; idx < MATRIX_LEN; idx += blockDim.x) {
                    int src = idx / N;
                    int dst = idx % N;
                    if (src == dst || active[src] == 0) continue;

                    int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
                    float confidence = tensor[cidx];
                    float cost       = tensor[tensor_idx(LAYER_COST,   src, dst)];
                    float safety     = tensor[tensor_idx(LAYER_SAFETY, src, dst)];
                    if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

                    int hop = witness[cidx];
                    if (hop != target_hop || consumed[src * N + hop] != 0) continue;

                    float score = alpha * confidence - beta * cost + gamma * safety;
                    local_candidates++;
                    choose_best(score, src, dst, hop, local_best_value, local_best_src, local_best_dst, local_best_hop);
                }
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
                    sh_node_ids[m] = target_hop;
                    sh_region_ids[m] = region_id;
                    sh_dispatch_kinds[m] = dispatch_kind;
                    sh_srcs[m] = block_best_src;
                    sh_dsts[m] = block_best_dst;
                    sh_hops[m] = block_best_hop;
                    sh_best_values[m] = block_best_value;
                    sh_member_ready[m] = (is_dispatchable && block_candidates > 0 && block_best_hop != HALT_NODE) ? 1 : 0;
                }
            }
            __syncthreads();
        }

        int local_group_ready = 1;
        for (int m = tid; m < sh_sz; m += blockDim.x) {
            if (!sh_member_ready[m]) local_group_ready = 0;
        }
        int group_ready = warp_reduce_sum(local_group_ready == 0 ? 1 : 0);
        if (lane == 0) warp_candidates[warp_id] = group_ready;
        __syncthreads();
        if (warp_id == 0) {
            int failed_count = lane < warp_count ? warp_candidates[lane] : 0;
            failed_count = warp_reduce_sum(failed_count);
            if (lane == 0 && failed_count == 0) {
                sh_selected_g = g;
                global_decision[0].step           = next_step;
                global_decision[0].selected_src   = sh_srcs[0];
                global_decision[0].selected_dst   = sh_dsts[0];
                global_decision[0].first_hop      = sh_hops[0];
                global_decision[0].selected_value = sh_best_values[0];
                global_decision[0].halted         = 0;
                global_decision[0].blocked        = 0;

                result->selected_group_idx = g;
                result->group_size = sh_sz;
                for (int i = 0; i < sh_sz; i++) {
                    result->decisions[i].step = next_step;
                    result->decisions[i].selected_src = sh_srcs[i];
                    result->decisions[i].selected_dst = sh_dsts[i];
                    result->decisions[i].first_hop = sh_hops[i];
                    result->decisions[i].selected_value = sh_best_values[i];
                    result->decisions[i].halted = 0;
                    result->decisions[i].blocked = 0;
                    result->region_ids[i] = sh_region_ids[i];
                    result->dispatch_descriptors[i].member_index = i;
                    result->dispatch_descriptors[i].node_id = sh_node_ids[i];
                    result->dispatch_descriptors[i].region_id = sh_region_ids[i];
                    result->dispatch_descriptors[i].dispatch_kind = sh_dispatch_kinds[i];
                    result->dispatch_descriptors[i].src_node = sh_srcs[i];
                    result->dispatch_descriptors[i].dst_node = sh_hops[i];
                }
            }
        }
        __syncthreads();
        if (sh_selected_g >= 0) break;
    }
    __syncthreads();

    // ── Phase 2: block-parallel commit writeback ─────────────────────────────
    //
    // Thread 0 selected a conflict-free par group and published member effects
    // to shared memory.  All lanes participate in the state writeback so commit
    // no longer serializes the consumed/active memory loops through thread 0.

    if (sh_selected_g >= 0) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            next_active[i] = 0;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < sh_sz; i += blockDim.x) {
            int src = sh_srcs[i];
            int hop = sh_hops[i];
            if (src >= 0 && src < N && hop >= 0 && hop < N) {
                atomicExch(&consumed[src * N + hop], 1);
                atomicExch(&next_active[hop], 1);
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            active[i] = next_active[i];
        }
    }
    __syncthreads();

    // ── Phase 3: hot-region inline dispatch (H_f path, all threads) ──────────
    //
    // Only runs when a group was selected and the receipt ring is available.
    // All threads see the same sh_* values → no warp divergence in the loop
    // guard checks.  The region __device__ functions use threadIdx.x for
    // element-wise parallelism; thread 0 writes the receipt to the ring.

    if (sh_selected_g < 0) return;
    if (!hf || !hf->receipt_ring || !hf->ring_tail || hf->ring_size <= 0 || hf->region_count <= 0) return;

    for (int i = 0; i < sh_sz; i++) {
        int rid = sh_region_ids[i];
        int table_idx = sh_selected_g * MAX_PAR_GROUP_SIZE + i;
        int dispatch_kind = result->dispatch_descriptors[i].dispatch_kind;

        if (dispatch_kind == PAR_DISPATCH_ABSTRACT_DEVICE) {
            if (threadIdx.x == 0) {
                DeviceReceipt r;
                r.region_id    = -1;
                r.src          = sh_srcs[i];
                r.dst          = sh_dsts[i];
                r.outcome      = 0;
                r.latency      = 0.0f;
                r.valid        = 1;
                r.output_flags = 0;
                int tail = *hf->ring_tail;
                hf->receipt_ring[tail % hf->ring_size] = r;
                *hf->ring_tail = tail + 1;
                result->dispatched_on_device[i] = 1;
            }
            __syncthreads();
            continue;
        }

        if (rid < 0 || rid >= hf->region_count) continue;

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
            case 7: region_analysis_fused_signal_score(slot_ptrs, elem_count, &r); break;
            // @@FUSION_HF_GENERATED_PAR_CASES@@
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
