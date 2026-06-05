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
