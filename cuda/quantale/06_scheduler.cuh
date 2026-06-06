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
                if (state) {
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
