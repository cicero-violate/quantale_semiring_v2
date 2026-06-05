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
    int star_counter;   // Phase 4: bounded-star iteration count (legacy single-counter)
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
    state->star_counter          = 0;
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

// ── Phase-4 control-flow structures and device helpers ───────────────────────

#define CONTROL_OP_SEQ          0
#define CONTROL_OP_PAR          1
#define CONTROL_OP_CHOICE       2
#define CONTROL_OP_STAR_BOUNDED 3
#define CONTROL_OP_GATE         4
#define CONTROL_OP_HALT         5

// ── ControlDecision: outcome of select_control_decision ──────────────────────
#define CONTROL_NONE            0
#define CONTROL_SEQ_READY       1
#define CONTROL_PAR_READY       2
#define CONTROL_CHOICE_READY    3
#define CONTROL_STAR_BODY_READY 4
#define CONTROL_STAR_EXIT_READY 5
#define CONTROL_HALT_READY      6
#define CONTROL_BLOCKED         7

struct ControlDecision {
    int kind;       // CONTROL_* constant above
    int edge_idx;   // index into ControlEdge table, or -1
    int lhs;        // selected lhs node, or -1
    int rhs;        // selected rhs node, or -1
    int order;      // order key of selected edge
};

// Maximum number of par-group members for the inline PAR commit path.
#define MAX_PAR_INLINE_SIZE 8

// One edge in a lowered pattern control-flow table.
struct ControlEdge {
    int op;     // CONTROL_OP_*
    int lhs;    // left-hand node id (or -1 = sentinel)
    int rhs;    // right-hand node id (or -1 = sentinel)
    int guard;  // guard predicate id (0 = always-true; non-zero reserved for Phase 5)
    int order;  // sequence position index (for SEQ edges)
    int bound;  // max iterations (for STAR_BOUNDED; 0 = no limit enforced)
};

// Per-node effect entry for par-eligibility checks.
// Bitmasks address up to 32 named resources (e.g. market.price = bit 0).
struct EffectTable {
    int reads;        // resources this node reads
    int writes;       // resources this node writes
    int locks;        // resources this node locks exclusively
    int safety_class; // 0=safe, 1=risky, 2=unsafe
};

// Returns 1 if nodes a and b are effect-independent (par-eligible).
// Two nodes are independent if:
//   writes(a) ∩ (reads(b) ∪ writes(b)) = ∅   AND
//   writes(b) ∩ (reads(a) ∪ writes(a)) = ∅
__device__ int effects_independent(
    const EffectTable* effects, int n_effects, int a, int b
) {
    if (!effects || a < 0 || a >= n_effects || b < 0 || b >= n_effects) return 1;
    const EffectTable* ea = &effects[a];
    const EffectTable* eb = &effects[b];
    if (ea->writes & (eb->reads | eb->writes)) return 0;
    if (eb->writes & (ea->reads | ea->writes)) return 0;
    return 1;
}

// Returns the CONTROL_OP_* for the first edge where lhs==src and rhs==dst and
// guard==0 (always-true).  Returns -1 if no matching edge is found.
__device__ int find_matching_control_edge(
    const ControlEdge* edges, int edge_count, int src, int dst
) {
    for (int i = 0; i < edge_count; ++i) {
        const ControlEdge* e = &edges[i];
        if (e->lhs == src && e->rhs == dst && e->guard == 0) return e->op;
    }
    return -1;
}

// Advance the bounded-star counter for the STAR_BOUNDED edge matching (src, dst).
// Sets state->halted = 1 when star_counter reaches star_bound.
// No-op if no matching STAR_BOUNDED edge is found.
__device__ void star_counter_advance(
    OrchestrationState*  state,
    const ControlEdge*   edges,
    int                  edge_count,
    int                  src,
    int                  dst
) {
    for (int i = 0; i < edge_count; ++i) {
        const ControlEdge* e = &edges[i];
        if (e->lhs == src && e->rhs == dst && e->op == CONTROL_OP_STAR_BOUNDED) {
            if (e->bound > 0) {
                state->star_bound   = e->bound;
                state->star_counter = state->star_counter + 1;
                if (state->star_counter >= e->bound) state->halted = 1;
            }
            return;
        }
    }
}

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

// ── Phase-2: dispatch kind table and scheduler kernel ─────────────────────────
//
// dispatch_kind[node_id] encodes which execution tier owns each node.
// Values match the PAR_DISPATCH_* constants used by the par-group kernel.

#define DISPATCH_KIND_NONE             0
#define DISPATCH_KIND_HF_DEVICE        1
#define DISPATCH_KIND_HOST_FALLBACK    2
#define DISPATCH_KIND_FUSION_ENTRY     3
#define DISPATCH_KIND_ABSTRACT_DEVICE  4
#define DISPATCH_KIND_EXTERNAL_PROCESS 5
#define DISPATCH_KIND_EXTERNAL_IO      6
#define DISPATCH_KIND_UNSUPPORTED      7

// Orchestration step status codes returned to the host.
#define ORCH_CONTINUE       0
#define ORCH_WAIT_EXTERNAL  1
#define ORCH_HALTED         2
#define ORCH_ERROR          3

// TensorWorldBundle: packs tensor/frontier/control pointers into a single
// struct so the LaunchAsync tuple stays within the cudarc arity limit.
struct TensorWorldBundle {
    float* tensor;
    int*   witness;
    int*   consumed;
    int*   active;
    int*   next_active;
    const int* reentrant_mask;
    const ProjectionBias* bias;
    DecisionReport* decision;
    // Phase 1/2: control-flow table pointers (null-safe; NULL = no control table loaded)
    const ControlEdge* control_edges;
    int                control_edge_count;
    const EffectTable* effects;
    int                effect_count;
    int*               star_counters;      // per-edge bounded-star counter table
    int                star_counter_count;
};

// Select the highest-scoring ready singleton node on-device, using the same
// readiness predicate used by the GPU-native scheduler.
// Writes the selected src/dst/hop/score into *out.  Returns 1 if a node was
// found, 0 if the frontier is empty or halted.
__device__ int select_ready_singleton(
    const TensorWorldBundle* w,
    const int* dispatch_kinds,
    int*  out_src, int* out_dst, int* out_hop, float* out_score
) {
    float alpha = w->bias[0].confidence, beta = w->bias[0].cost, gamma = w->bias[0].safety;
    float best_value = -COST_INFINITY;
    int best_src = -1, best_dst = -1, best_hop = -1, candidates = 0;

    for (int idx = 0; idx < MATRIX_LEN; ++idx) {
        int src = idx / N, dst = idx % N;
        if (src == dst || w->active[src] == 0) continue;

        int cidx = tensor_idx(LAYER_CONFIDENCE, src, dst);
        int eidx = tensor_idx(LAYER_COST,       src, dst);
        int sidx = tensor_idx(LAYER_SAFETY,     src, dst);
        float confidence = w->tensor[cidx];
        float cost       = w->tensor[eidx];
        float safety     = w->tensor[sidx];
        if (confidence <= BOTTOM || safety <= BOTTOM || cost >= COST_INFINITY) continue;

        int hop = w->witness[cidx];
        if (hop < 0 || hop >= N) continue;
        int reusable_edge = w->reentrant_mask && (w->reentrant_mask[src] != 0 || w->reentrant_mask[hop] != 0);
        if (!reusable_edge && w->consumed[src * N + hop] != 0) continue;
        if (dispatch_kinds && dispatch_kinds[hop] == DISPATCH_KIND_UNSUPPORTED) continue;

        float score = alpha * confidence - beta * cost + gamma * safety;
        candidates++;
        if (score > best_value) {
            best_value = score;
            best_src = src; best_dst = dst; best_hop = hop;
        }
    }
    *out_src = best_src; *out_dst = best_dst; *out_hop = best_hop;
    *out_score = best_value;
    return candidates > 0 ? 1 : 0;
}

// ── Phase 3: SEQ device helpers ───────────────────────────────────────────────

// Scan the control edge table for the SEQ edge with minimum (order, rhs, idx)
// whose lhs is active and whose (lhs,rhs) edge has not been consumed.
// Returns 1 if found; fills *out_idx, *out_lhs, *out_rhs.
__device__ int ready_seq(
    const TensorWorldBundle* w,
    int* out_idx, int* out_lhs, int* out_rhs
) {
    int best_order = 0x7fffffff, best_rhs = 0x7fffffff, best_idx = -1;
    for (int i = 0; i < w->control_edge_count; ++i) {
        const ControlEdge* e = &w->control_edges[i];
        if (e->op != CONTROL_OP_SEQ) continue;
        if (e->lhs < 0 || e->lhs >= N || e->rhs < 0 || e->rhs >= N) continue;
        if (!w->active[e->lhs]) continue;
        int reusable = w->reentrant_mask
            && (w->reentrant_mask[e->lhs] || w->reentrant_mask[e->rhs]);
        if (!reusable && w->consumed[e->lhs * N + e->rhs]) continue;
        if (e->order < best_order || (e->order == best_order && e->rhs < best_rhs)) {
            best_order = e->order; best_rhs = e->rhs; best_idx = i;
        }
    }
    if (best_idx < 0) return 0;
    *out_idx = best_idx;
    *out_lhs = w->control_edges[best_idx].lhs;
    *out_rhs = w->control_edges[best_idx].rhs;
    return 1;
}

// ── Phase 5: CHOICE device helpers ───────────────────────────────────────────

// Score one CHOICE branch.  Uses tensor layers for confidence/cost/safety.
// receipt_prior and failure_penalty are not yet wired (passed as 0.0).
__device__ float score_choice_branch(
    const TensorWorldBundle* w,
    int root, int branch
) {
    if (root < 0 || root >= N || branch < 0 || branch >= N) return -COST_INFINITY;
    float alpha = w->bias[0].confidence, beta = w->bias[0].cost, gamma = w->bias[0].safety;
    float conf = w->tensor[tensor_idx(LAYER_CONFIDENCE, root, branch)];
    float cost = w->tensor[tensor_idx(LAYER_COST,       root, branch)];
    float safe = w->tensor[tensor_idx(LAYER_SAFETY,     root, branch)];
    if (conf <= BOTTOM || safe <= BOTTOM || cost >= COST_INFINITY) return -COST_INFINITY;
    return alpha * conf - beta * cost + gamma * safe;
}

// Scan all CHOICE edges from the same root (lhs) that share the lowest
// order key and whose root is active.  Select the branch with the highest
// score; tie-break by (order, rhs, edge_idx).
// Returns 1 if a choice was made; fills *out_idx, *out_lhs, *out_rhs.
__device__ int ready_choice(
    const TensorWorldBundle* w,
    int* out_idx, int* out_lhs, int* out_rhs
) {
    // Scan once to find active roots with CHOICE edges.
    float best_score = -COST_INFINITY;
    int best_idx = -1, best_lhs = -1, best_rhs = -1;
    for (int i = 0; i < w->control_edge_count; ++i) {
        const ControlEdge* e = &w->control_edges[i];
        if (e->op != CONTROL_OP_CHOICE) continue;
        if (e->lhs < 0 || e->lhs >= N || e->rhs < 0 || e->rhs >= N) continue;
        if (!w->active[e->lhs]) continue;
        if (w->consumed[e->lhs * N + e->rhs]) continue;
        float sc = score_choice_branch(w, e->lhs, e->rhs);
        if (sc > best_score
                || (sc == best_score && (e->order < w->control_edges[best_idx].order
                    || (e->order == w->control_edges[best_idx].order && e->rhs < best_rhs)))) {
            best_score = sc; best_idx = i; best_lhs = e->lhs; best_rhs = e->rhs;
        }
    }
    if (best_idx < 0) return 0;
    *out_idx = best_idx; *out_lhs = best_lhs; *out_rhs = best_rhs;
    return 1;
}

// ── Phase 6: Bounded STAR device helpers ─────────────────────────────────────

// Find the lowest-order ready STAR_BOUNDED edge where:
//   active[lhs], !consumed[lhs,rhs], star_counters[edge_idx] < bound (if bound>0).
// Returns CONTROL_STAR_BODY_READY or CONTROL_STAR_EXIT_READY.
__device__ int ready_star(
    const TensorWorldBundle* w,
    int* out_idx, int* out_lhs, int* out_rhs
) {
    int best_order = 0x7fffffff, best_idx = -1;
    for (int i = 0; i < w->control_edge_count; ++i) {
        const ControlEdge* e = &w->control_edges[i];
        if (e->op != CONTROL_OP_STAR_BOUNDED) continue;
        if (e->lhs < 0 || e->lhs >= N || e->rhs < 0 || e->rhs >= N) continue;
        if (!w->active[e->lhs]) continue;
        if (e->order < best_order) { best_order = e->order; best_idx = i; }
    }
    if (best_idx < 0) return CONTROL_NONE;
    const ControlEdge* e = &w->control_edges[best_idx];
    int counter = (w->star_counters && best_idx < w->star_counter_count)
                  ? w->star_counters[best_idx] : 0;
    *out_idx = best_idx; *out_lhs = e->lhs; *out_rhs = e->rhs;
    if (e->bound > 0 && counter >= e->bound) return CONTROL_STAR_EXIT_READY;
    return CONTROL_STAR_BODY_READY;
}

// ── Phase 4: PAR inline device helper ────────────────────────────────────────

// Scan for a PAR group: all PAR edges with active lhs and all members
// mutually independent.  A "group" is any active node that has at least one
// PAR edge and is effect-independent of all other active PAR-paired nodes.
// Fills par_lhs[]/par_rhs[] (up to MAX_PAR_INLINE_SIZE), returns group size.
__device__ int ready_par(
    const TensorWorldBundle* w,
    int* par_edge_idx, int* par_rhs, int max_size
) {
    // Collect all distinct (active lhs, unconsumed rhs) PAR pairs.
    int sz = 0;
    for (int i = 0; i < w->control_edge_count && sz < max_size; ++i) {
        const ControlEdge* e = &w->control_edges[i];
        if (e->op != CONTROL_OP_PAR) continue;
        if (e->lhs < 0 || e->lhs >= N || e->rhs < 0 || e->rhs >= N) continue;
        if (!w->active[e->lhs]) continue;
        if (w->consumed[e->lhs * N + e->rhs]) continue;
        // Check effect independence with all already-selected members.
        int ok = 1;
        for (int j = 0; j < sz && ok; ++j)
            ok = effects_independent(w->effects, w->effect_count, e->rhs, par_rhs[j]);
        if (!ok) continue;
        par_edge_idx[sz] = i;
        par_rhs[sz] = e->rhs;
        sz++;
    }
    return sz;
}

// ── Phase 2: unified select_control_decision ─────────────────────────────────

// Consult the control-flow table and return the highest-priority ready decision.
// Priority: SEQ > STAR_BODY > CHOICE > PAR > HALT.
// Returns CONTROL_NONE when the control table is empty or no edge is ready.
__device__ ControlDecision select_control_decision(const TensorWorldBundle* w) {
    ControlDecision none = { CONTROL_NONE, -1, -1, -1, 0 };
    if (!w->control_edges || w->control_edge_count == 0) return none;

    int idx, lhs, rhs;

    // SEQ
    if (ready_seq(w, &idx, &lhs, &rhs)) {
        ControlDecision d = { CONTROL_SEQ_READY, idx, lhs, rhs, w->control_edges[idx].order };
        return d;
    }
    // STAR body
    int star_kind = ready_star(w, &idx, &lhs, &rhs);
    if (star_kind == CONTROL_STAR_BODY_READY) {
        ControlDecision d = { CONTROL_STAR_BODY_READY, idx, lhs, rhs, w->control_edges[idx].order };
        return d;
    }
    if (star_kind == CONTROL_STAR_EXIT_READY) {
        ControlDecision d = { CONTROL_STAR_EXIT_READY, idx, lhs, rhs, w->control_edges[idx].order };
        return d;
    }
    // CHOICE
    if (ready_choice(w, &idx, &lhs, &rhs)) {
        ControlDecision d = { CONTROL_CHOICE_READY, idx, lhs, rhs, w->control_edges[idx].order };
        return d;
    }
    // PAR (lower priority than SEQ/STAR/CHOICE, higher than HALT)
    {
        int par_idxs[MAX_PAR_INLINE_SIZE], par_rhs_arr[MAX_PAR_INLINE_SIZE];
        int par_sz = ready_par(w, par_idxs, par_rhs_arr, MAX_PAR_INLINE_SIZE);
        if (par_sz > 0) {
            const ControlEdge* e0 = &w->control_edges[par_idxs[0]];
            ControlDecision d = { CONTROL_PAR_READY, par_idxs[0], e0->lhs, par_rhs_arr[0], e0->order };
            return d;
        }
    }
    // HALT control edge
    for (int i = 0; i < w->control_edge_count; ++i) {
        const ControlEdge* e = &w->control_edges[i];
        if (e->op == CONTROL_OP_HALT && e->lhs >= 0 && e->lhs < N && w->active[e->lhs]) {
            ControlDecision d = { CONTROL_HALT_READY, i, e->lhs, e->rhs, e->order };
            return d;
        }
    }
    return none;
}

// ── Phase 7: unified commit_selected_node_or_command ─────────────────────────
//
// Common commit path for all control-flow ops and the singleton scheduler.
// Commits the consumed/active state transition for node dst (hopped to from src).
// For external dispatch kinds, emits a DeviceCommand and returns ORCH_WAIT_EXTERNAL.
// For GPU-native dispatch kinds, updates frontier and returns ORCH_CONTINUE.
// For HALT_NODE, sets state->halted and returns ORCH_HALTED.
// Updates OrchestrationState fields (step, selected_node, selected_src/dst).

__device__ int commit_selected_node_or_command(
    const TensorWorldBundle* w,
    OrchestrationState*      state,
    int src, int dst,
    int dispatch_kind,
    int sel_ctrl_edge, int sel_ctrl_op, int sel_ctrl_lhs, int sel_ctrl_rhs,
    DeviceCommand* cmd_ring, int* cmd_tail, const int* cmd_head, int cmd_ring_size
) {
    bool gpu_native = (dispatch_kind == DISPATCH_KIND_HF_DEVICE
                    || dispatch_kind == DISPATCH_KIND_FUSION_ENTRY
                    || dispatch_kind == DISPATCH_KIND_ABSTRACT_DEVICE
                    || dispatch_kind == DISPATCH_KIND_NONE);
    bool external   = (dispatch_kind == DISPATCH_KIND_EXTERNAL_PROCESS
                    || dispatch_kind == DISPATCH_KIND_EXTERNAL_IO);

    // Commit frontier.
    if (src >= 0 && src < N && dst >= 0 && dst < N) {
        int reusable = w->reentrant_mask
            && (w->reentrant_mask[src] || w->reentrant_mask[dst]);
        if (!reusable) atomicExch(&w->consumed[src * N + dst], 1);
        for (int i = 0; i < N; ++i) w->next_active[i] = 0;
        w->next_active[dst] = 1;
        for (int i = 0; i < N; ++i) w->active[i] = w->next_active[i];
    }

    // Update orchestration state.
    if (state) {
        state->step          += 1;
        state->selected_node  = dst;
        state->selected_src   = src;
        state->selected_dst   = dst;
        state->blocked        = 0;
        state->halted         = (dst == HALT_NODE) ? 1 : 0;
        state->selected_control_edge = sel_ctrl_edge;
        state->selected_control_op   = sel_ctrl_op;
        state->selected_control_lhs  = sel_ctrl_lhs;
        state->selected_control_rhs  = sel_ctrl_rhs;
        if (sel_ctrl_op >= 0) state->control_epoch += 1;
    }
    if (w->decision) {
        w->decision->step          += 1;
        w->decision->selected_src   = src;
        w->decision->selected_dst   = dst;
        w->decision->first_hop      = dst;
        w->decision->halted         = (dst == HALT_NODE) ? 1 : 0;
        w->decision->blocked        = 0;
    }

    if (dst == HALT_NODE) return ORCH_HALTED;

    if (external) {
        if (state && state->pending_external_count >= cmd_ring_size)
            return ORCH_WAIT_EXTERNAL;
        int t = *cmd_tail, h = *cmd_head;
        if (t - h < cmd_ring_size) {
            DeviceCommand cmd;
            cmd.valid            = 1;
            cmd.command_id       = state ? state->step : 0;
            cmd.node_id          = dst;
            cmd.src              = src;
            cmd.dst              = dst;
            cmd.dispatch_kind    = dispatch_kind;
            cmd.operator_name_id = dst;
            cmd.timeout_ticks    = 0;
            cmd.retry_budget     = 1;
            cmd.payload_offset   = 0;
            cmd.payload_len      = 0;
            cmd_ring[t % cmd_ring_size] = cmd;
            *cmd_tail = t + 1;
            if (state) atomicAdd(&state->pending_external_count, 1);
        }
        return ORCH_WAIT_EXTERNAL;
    }
    if (!gpu_native) return ORCH_CONTINUE; // UNSUPPORTED treated as continue/blocked
    return ORCH_CONTINUE;
}

// ── tensor_quantale_orchestrate_step ─────────────────────────────────────────
//
// Single-block, single-step GPU scheduler kernel.  On each call it:
//   1. Drains the extended receipt ring (applies tensor updates).
//   2. Checks control table: SEQ / STAR / CHOICE / PAR / HALT.
//   3. If a control decision is ready, commits it via commit_selected_node_or_command.
//   4. Else falls back to the GPU singleton scheduler.
//   5. Writes one of ORCH_CONTINUE / ORCH_WAIT_EXTERNAL / ORCH_HALTED / ORCH_ERROR
//      into *status_out.

extern "C" __global__ void tensor_quantale_orchestrate_step(
    const TensorWorldBundle* world,
    OrchestrationState*      state,
    DeviceCommand*          cmd_ring,
    int*                    cmd_tail,
    const int*              cmd_head,
    int                     cmd_ring_size,
    DeviceReceiptExt*       ext_ring,
    int*                    ext_head,
    const int*              ext_tail,
    int                     ext_ring_size,
    const int*              dispatch_kinds,   // int[N]: dispatch kind per node id
    int*                    status_out        // ORCH_* result written by thread 0
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // ── 1. Drain extended receipt ring ───────────────────────────────────────
    {
        int h = *ext_head, t = *ext_tail;
        while (h != t) {
            int slot = h % ext_ring_size;
            DeviceReceiptExt r = ext_ring[slot];
            if (r.valid && !r.consumed
                    && r.src >= 0 && r.src < N && r.dst >= 0 && r.dst < N && r.src != r.dst) {
                int cidx = tensor_idx(LAYER_CONFIDENCE, r.src, r.dst);
                int eidx = tensor_idx(LAYER_COST,       r.src, r.dst);
                int sidx = tensor_idx(LAYER_SAFETY,     r.src, r.dst);
                if (r.outcome == 0) {
                    atomicExch(&world->tensor[cidx], 1.0f);
                    { int* ptr = (int*)&world->tensor[eidx]; int ob, nb;
                      do { ob = *((volatile int*)ptr); float v = __int_as_float(ob);
                           if (v <= 0.01f) break; nb = __float_as_int(v * 0.5f);
                      } while (atomicCAS(ptr, ob, nb) != ob); }
                    atomicExch(&world->tensor[sidx], 1.0f);
                } else if (r.outcome == 1) {
                    atomic_float_mul(&world->tensor[cidx], 0.1f);
                    atomic_float_add_capped(&world->tensor[eidx], 10.0f);
                    atomic_float_mul(&world->tensor[sidx], 0.5f);
                    if (state) atomicAdd(&state->failure_count, 1);
                } else if (r.outcome == 2) {
                    atomic_float_mul(&world->tensor[cidx], 0.5f);
                    atomic_float_add_capped(&world->tensor[eidx], 100.0f);
                } else if (r.outcome == 3) {
                    atomic_float_mul(&world->tensor[cidx], 0.25f);
                    atomic_float_add_capped(&world->tensor[eidx], 25.0f);
                    atomicExch(&world->tensor[sidx], 0.0f);
                }
                ext_ring[slot].consumed = 1;
                if (state) {
                    atomicAdd(&state->pending_receipt_count, -1);
                    if (state->pending_external_count > 0) {
                        atomicAdd(&state->pending_external_count, -1);
                    }
                }
            }
            h++;
        }
        *ext_head = h;
    }

    // ── 2. Check halt ─────────────────────────────────────────────────────────
    if (state && state->halted) {
        *status_out = ORCH_HALTED;
        return;
    }

    // ── 3. Select control decision (GPU owns control flow) ───────────────────
    ControlDecision ctrl = select_control_decision(world);

    if (ctrl.kind != CONTROL_NONE) {
        int sel_dst = ctrl.rhs;
        int sel_src = ctrl.lhs;
        int dkind = (dispatch_kinds && sel_dst >= 0 && sel_dst < N)
                    ? dispatch_kinds[sel_dst] : DISPATCH_KIND_HF_DEVICE;

        // HALT edge fires immediately.
        if (ctrl.kind == CONTROL_HALT_READY) {
            if (state) { state->halted = 1; state->step += 1;
                         state->selected_control_edge = ctrl.edge_idx;
                         state->selected_control_op   = CONTROL_OP_HALT;
                         state->selected_control_lhs  = ctrl.lhs;
                         state->selected_control_rhs  = ctrl.rhs;
                         state->control_epoch += 1; }
            *status_out = ORCH_HALTED;
            return;
        }

        // STAR exit: counter exhausted — consume the back-edge so it cannot
        // re-fire, advance active to lhs (the exit node), let subsequent SEQ/
        // CHOICE edges from lhs route to the loop continuation.
        if (ctrl.kind == CONTROL_STAR_EXIT_READY) {
            if (ctrl.lhs >= 0 && ctrl.lhs < N && ctrl.rhs >= 0 && ctrl.rhs < N)
                atomicExch(&world->consumed[ctrl.lhs * N + ctrl.rhs], 1);
            if (state) { state->step += 1; state->selected_node = ctrl.lhs;
                         state->selected_src = ctrl.lhs; state->selected_dst = ctrl.lhs;
                         state->selected_control_edge = ctrl.edge_idx;
                         state->selected_control_op   = CONTROL_OP_STAR_BOUNDED;
                         state->selected_control_lhs  = ctrl.lhs;
                         state->selected_control_rhs  = ctrl.rhs;
                         state->control_epoch += 1;
                         state->blocked = 0; }
            for (int i = 0; i < N; ++i) world->next_active[i] = 0;
            if (ctrl.lhs >= 0 && ctrl.lhs < N) world->next_active[ctrl.lhs] = 1;
            for (int i = 0; i < N; ++i) world->active[i] = world->next_active[i];
            *status_out = ORCH_CONTINUE;
            return;
        }

        // STAR body: increment per-edge counter, then commit normally.
        if (ctrl.kind == CONTROL_STAR_BODY_READY) {
            if (world->star_counters && ctrl.edge_idx >= 0
                    && ctrl.edge_idx < world->star_counter_count) {
                world->star_counters[ctrl.edge_idx] += 1;
                // Also update legacy single counter for backward compat.
                if (state) {
                    state->star_counter      += 1;
                    state->star_counter_epoch += 1;
                    const ControlEdge* e = &world->control_edges[ctrl.edge_idx];
                    state->star_bound = e->bound;
                }
            }
        }

        // PAR: find all ready, independent PAR members and commit them together.
        if (ctrl.kind == CONTROL_PAR_READY) {
            int par_edge_idx[MAX_PAR_INLINE_SIZE];
            int par_rhs[MAX_PAR_INLINE_SIZE];
            int par_sz = ready_par(world, par_edge_idx, par_rhs, MAX_PAR_INLINE_SIZE);
            if (par_sz == 0) {
                if (state) { state->blocked = 1;
                             state->last_block_reason = ORCH_BLOCK_REASON_NO_READY_NODE; }
                *status_out = ORCH_CONTINUE;
                return;
            }
            // Commit all independent par members.
            for (int i = 0; i < N; ++i) world->next_active[i] = 0;
            int any_external = 0;
            for (int m = 0; m < par_sz; ++m) {
                int ei = par_edge_idx[m];
                int mlhs = world->control_edges[ei].lhs;
                int mrhs = par_rhs[m];
                if (mlhs >= 0 && mlhs < N && mrhs >= 0 && mrhs < N)
                    atomicExch(&world->consumed[mlhs * N + mrhs], 1);
                if (mrhs >= 0 && mrhs < N) world->next_active[mrhs] = 1;
                int mk = (dispatch_kinds && mrhs >= 0 && mrhs < N)
                         ? dispatch_kinds[mrhs] : DISPATCH_KIND_HF_DEVICE;
                if (mk == DISPATCH_KIND_EXTERNAL_PROCESS || mk == DISPATCH_KIND_EXTERNAL_IO) {
                    int t = *cmd_tail, hh = *cmd_head;
                    if (t - hh < cmd_ring_size) {
                        DeviceCommand cmd; cmd.valid = 1;
                        cmd.command_id = state ? state->step + m : m;
                        cmd.node_id = mrhs; cmd.src = mlhs; cmd.dst = mrhs;
                        cmd.dispatch_kind = mk; cmd.operator_name_id = mrhs;
                        cmd.timeout_ticks = 0; cmd.retry_budget = 1;
                        cmd.payload_offset = 0; cmd.payload_len = 0;
                        cmd_ring[t % cmd_ring_size] = cmd;
                        *cmd_tail = t + 1;
                        if (state) atomicAdd(&state->pending_external_count, 1);
                    }
                    any_external = 1;
                }
            }
            for (int i = 0; i < N; ++i) world->active[i] = world->next_active[i];
            if (state) {
                state->step += 1; state->blocked = 0;
                state->selected_control_edge = ctrl.edge_idx;
                state->selected_control_op   = CONTROL_OP_PAR;
                state->selected_control_lhs  = ctrl.lhs;
                state->selected_control_rhs  = ctrl.rhs;
                state->control_epoch += 1;
            }
            *status_out = any_external ? ORCH_WAIT_EXTERNAL : ORCH_CONTINUE;
            return;
        }

        // SEQ, STAR_BODY, CHOICE: single-node commit via unified path.
        int ctrl_op = (ctrl.kind == CONTROL_SEQ_READY)       ? CONTROL_OP_SEQ
                    : (ctrl.kind == CONTROL_STAR_BODY_READY)  ? CONTROL_OP_STAR_BOUNDED
                    : CONTROL_OP_CHOICE;
        *status_out = commit_selected_node_or_command(
            world, state, sel_src, sel_dst, dkind,
            ctrl.edge_idx, ctrl_op, ctrl.lhs, ctrl.rhs,
            cmd_ring, cmd_tail, cmd_head, cmd_ring_size);
        return;
    }

    // ── 4. Fall back to GPU singleton scheduler ───────────────────────────────
    int sel_src = -1, sel_dst = -1, sel_hop = -1;
    float sel_score = -COST_INFINITY;
    int found = select_ready_singleton(world, dispatch_kinds,
                                       &sel_src, &sel_dst, &sel_hop, &sel_score);

    if (!found) {
        if (state) { state->blocked = 1;
                     state->last_block_reason = ORCH_BLOCK_REASON_NO_READY_NODE; }
        *status_out = ORCH_CONTINUE;
        return;
    }

    int dkind = (dispatch_kinds && sel_hop >= 0 && sel_hop < N)
                ? dispatch_kinds[sel_hop] : DISPATCH_KIND_HF_DEVICE;
    if (dkind == DISPATCH_KIND_UNSUPPORTED || dkind == DISPATCH_KIND_NONE) {
        if (state) { state->blocked = 1;
                     state->last_block_reason = ORCH_BLOCK_REASON_UNSUPPORTED; }
        *status_out = ORCH_CONTINUE;
        return;
    }

    if (world->decision) world->decision->selected_value = sel_score;
    *status_out = commit_selected_node_or_command(
        world, state, sel_src, sel_hop, dkind,
        -1, -1, sel_src, sel_hop,
        cmd_ring, cmd_tail, cmd_head, cmd_ring_size);
}

// ── Phase-4 kernels: control-flow advance and par-eligibility check ───────────

// Look up the matching CONTROL_OP_* for a selected (src, dst) edge.
// Advances the bounded-star counter if the edge is STAR_BOUNDED.
// Writes the matched op (or -1 if no match) into *out_op.
extern "C" __global__ void control_flow_advance(
    const ControlEdge*   edges,
    int                  edge_count,
    const EffectTable*   effects,
    int                  effect_count,
    OrchestrationState*  state,
    int                  src,
    int                  dst,
    int*                 out_op
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int op = find_matching_control_edge(edges, edge_count, src, dst);
    if (op == CONTROL_OP_STAR_BOUNDED && state) {
        star_counter_advance(state, edges, edge_count, src, dst);
    }
    *out_op = op;
    (void)effects; (void)effect_count; // reserved for Phase-5 guard evaluation
}

// Returns 1 into *out if nodes a and b are effect-independent (par-eligible),
// 0 otherwise.  Single-thread; used by smoke tests and the par tier.
extern "C" __global__ void check_effects_independent(
    const EffectTable* effects,
    int                n_effects,
    int                a,
    int                b,
    int*               out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out = effects_independent(effects, n_effects, a, b);
}

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

// ── Phase-8: observability, debug, and replay ─────────────────────────────────
//
// OrchestrationEvent: one record in the device trace ring.  The scheduler
//   emits events that explain every decision without CPU reconstruction.
// orch_event_trace_push: push one event derived from the current state.
// orch_event_trace_drain: copy all pending events to an output buffer.
// orch_check_no_duplicate_receipts: scan for consumed receipts sharing a
//   command_id — a receipt-fold-duplication invariant.
// orch_check_frontier_valid: all entries in active[] are 0 or 1.
// orch_check_no_command_without_receipt: every valid command in cmd_ring has
//   a consumed receipt in ext_ring.
// orch_replay_snapshot: block-parallel copy of state + consumed + active for
//   deterministic replay.
// orch_replay_restore: restore a previously saved replay snapshot.

#define ORCH_EVENT_STEP_COMMITTED  0  // GPU-native node dispatched
#define ORCH_EVENT_WAIT_EXTERNAL   1  // external command emitted
#define ORCH_EVENT_HALTED          2  // halt node reached
#define ORCH_EVENT_BLOCKED         3  // no ready node found
#define ORCH_EVENT_RECEIPT_DRAINED 4  // receipt folded from ext ring

#define ORCH_TRACE_RING_SIZE 512

struct OrchestrationEvent {
    int step;
    int event_kind;           // ORCH_EVENT_* constant
    int selected_node;
    int selected_group;
    int src;
    int dst;
    int outcome;
    // Phase 8 — control trace fields
    int selected_control_op;  // CONTROL_OP_* or -1
    int selected_control_edge;// ControlEdge table index or -1
    int branch_count;         // number of par members or choice candidates considered
    int star_counter_val;     // per-edge star counter at commit time, or 0
};

// Push one event built from the current OrchestrationState into the trace ring.
// Thread 0, block 0 only.
extern "C" __global__ void orch_event_trace_push(
    const OrchestrationState* state,
    int                       event_kind,
    int                       outcome,
    OrchestrationEvent*       trace_ring,
    int*                      trace_tail,
    const int*                trace_head,
    int                       ring_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!trace_ring || !trace_tail || !trace_head || ring_size <= 0) return;
    int t = *trace_tail, h = *trace_head;
    if (t - h >= ring_size) return; // ring full — drop oldest-first on next push
    OrchestrationEvent ev;
    ev.step                  = state ? state->step                  : 0;
    ev.event_kind            = event_kind;
    ev.selected_node         = state ? state->selected_node         : -1;
    ev.selected_group        = state ? state->selected_group        : -1;
    ev.src                   = state ? state->selected_src          : -1;
    ev.dst                   = state ? state->selected_dst          : -1;
    ev.outcome               = outcome;
    ev.selected_control_op   = state ? state->selected_control_op   : -1;
    ev.selected_control_edge = state ? state->selected_control_edge : -1;
    ev.branch_count          = 0; // filled by caller when known
    ev.star_counter_val      = state ? state->star_counter          : 0;
    trace_ring[t % ring_size] = ev;
    *trace_tail = t + 1;
}

// Drain all pending trace events into out_buf.  Advances *trace_head.
// Writes the number of events copied to *out_count.  Thread 0, block 0 only.
extern "C" __global__ void orch_event_trace_drain(
    OrchestrationEvent*       trace_ring,
    int*                      trace_head,
    const int*                trace_tail,
    int                       ring_size,
    OrchestrationEvent*       out_buf,
    int*                      out_count,
    int                       max_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *trace_head;
    int t = *trace_tail;
    int count = 0;
    while (h != t && count < max_count) {
        out_buf[count] = trace_ring[h % ring_size];
        h++;
        count++;
    }
    *trace_head = h;
    *out_count  = count;
}

// Invariant: no two consumed DeviceReceiptExt entries share a (command_id, node_id).
// Writes 1 to *violation_out if a duplicate is found, 0 otherwise.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_no_duplicate_receipts(
    const DeviceReceiptExt* ring,
    int                     size,
    int*                    violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int i = 0; i < size; ++i) {
        if (!ring[i].valid || !ring[i].consumed) continue;
        for (int j = i + 1; j < size; ++j) {
            if (!ring[j].valid || !ring[j].consumed) continue;
            if (ring[i].command_id == ring[j].command_id
                    && ring[i].node_id == ring[j].node_id) {
                *violation_out = 1;
                return;
            }
        }
    }
}

// Invariant: every entry in active[] is 0 or 1.
// Writes 1 to *violation_out if a corrupt value is found.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_frontier_valid(
    const int* active,
    int        n,
    int*       violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int i = 0; i < n; ++i) {
        if (active[i] != 0 && active[i] != 1) {
            *violation_out = 1;
            return;
        }
    }
}

// Invariant: every valid DeviceCommand in cmd_ring has a consumed receipt in
// ext_ring with a matching command_id.
// Writes 1 to *violation_out if a command without a terminal receipt is found.
// Thread 0, block 0 only.
extern "C" __global__ void orch_check_no_command_without_receipt(
    const DeviceCommand*    cmd_ring,
    int                     cmd_size,
    const DeviceReceiptExt* ext_ring,
    int                     ext_size,
    int*                    violation_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *violation_out = 0;
    for (int j = 0; j < cmd_size; ++j) {
        if (!cmd_ring[j].valid) continue;
        int cid = cmd_ring[j].command_id;
        int found = 0;
        for (int i = 0; i < ext_size; ++i) {
            if (ext_ring[i].valid && ext_ring[i].consumed
                    && ext_ring[i].command_id == cid) {
                found = 1;
                break;
            }
        }
        if (!found) {
            *violation_out = 1;
            return;
        }
    }
}

// Deterministic replay: block-parallel snapshot of state + consumed + active +
// per-edge star counters.
extern "C" __global__ void orch_replay_snapshot(
    const OrchestrationState* state,
    OrchestrationState*       out_state,
    const int*                consumed,
    int*                      out_consumed,
    const int*                active,
    int*                      out_active,
    const int*                star_counters,
    int*                      out_star_counters,
    int                       star_counter_count
) {
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        out_consumed[i] = consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        out_active[i] = active[i];
    if (star_counters && out_star_counters)
        for (int i = tid; i < star_counter_count; i += blockDim.x)
            out_star_counters[i] = star_counters[i];
    __syncthreads();
    if (tid == 0 && state && out_state)
        *out_state = *state;
}

// Deterministic replay: restore state + consumed + active + star counters from
// a replay snapshot.
extern "C" __global__ void orch_replay_restore(
    OrchestrationState*       state,
    const OrchestrationState* snap_state,
    int*                      consumed,
    const int*                snap_consumed,
    int*                      active,
    const int*                snap_active,
    int*                      star_counters,
    const int*                snap_star_counters,
    int                       star_counter_count
) {
    int tid = threadIdx.x;
    for (int i = tid; i < MATRIX_LEN; i += blockDim.x)
        consumed[i] = snap_consumed[i];
    for (int i = tid; i < N; i += blockDim.x)
        active[i] = snap_active[i];
    if (star_counters && snap_star_counters)
        for (int i = tid; i < star_counter_count; i += blockDim.x)
            star_counters[i] = snap_star_counters[i];
    __syncthreads();
    if (tid == 0 && state && snap_state)
        *state = *snap_state;
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

__device__ void region_analysis_fused_signal_score(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 7);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* open  = slot_ptrs[1];  // market.open
    float* out   = slot_ptrs[2];  // analysis.signal_score
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float ret = (price[i] - open[i]) / (open[i] + 1.0e-8f);
        float vol = fabsf(price[i] - ret) / (ret + 1.0e-8f);
        out[i] = ret / (1.0f + fabsf(vol));
    }
}

__device__ void region_commit_receipt(float** slot_ptrs, int n, DeviceReceipt* r) {
    (void)slot_ptrs; (void)n;
    r->output_flags |= (1 << 6);
}

// ── Generated fusion H_f handlers ─────────────────────────────────────────────
// The Rust kernel-source assembler replaces this marker with the contents of
// assets/fusion_hf.stubs.cu before PTX compilation.
// @@FUSION_HF_GENERATED_FUNCTIONS@@

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
        case 7: region_analysis_fused_signal_score(slot_ptrs, element_count, &r); break;
        // @@FUSION_HF_GENERATED_GPU_DISPATCH_CASES@@
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
