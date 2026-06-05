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

