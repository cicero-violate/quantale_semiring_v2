# Plan: Extended Topology and Semiring Invariants

## Goal

Extend the static checker (`src/topology_check.rs`) with invariants that cover
identity integrity, weight domain safety, gate dominance, SCC progress, and
semiring algebraic laws.

The five existing passes cover structural liveness:

```
valid endpoints + reachable + no dead rows + path-to-halt
```

These new passes cover:

```
identity + weights + determinism + gate dominance + progress
```

---

## Background

The five existing checks prevent topologies that cannot execute.
These new checks prevent topologies that execute incorrectly — wrong order,
bad weights, aliased indices, bypassed gates, trapped cycles.

Each invariant is listed with its mathematical statement, what it detects,
and its implementation shape.

---

## New Invariants

### 1. Unique Node Identity

```
∀ i ≠ j : name(vᵢ) ≠ name(vⱼ)
```

**Detects:** duplicate node names causing index aliasing and silent overwrites
in the `NodeRegistry` `by_name` map.

**Check:**

```rust
node_names.len() == node_names.iter().collect::<HashSet>().len()
```

Also check dense IDs are unique:

```rust
node_ids.len() == node_ids.iter().collect::<HashSet>().len()
```

---

### 2. Stable Index Mapping

```
index(name(vᵢ)) = i
name(index(vᵢ)) = name(vᵢ)
```

**Detects:** round-trip corruption in `NodeRegistry` — the root cause of the
`Unknown(-1)` class of failures where a node ID decodes to the wrong name or
to nothing.

**Check (both directions):**

```rust
for node in nodes:
    id   = registry.id_of(node.name)    // name -> index
    name = registry.name_of(id)          // index -> name
    assert id == node.id
    assert name == node.name
```

---

### 3. Start Node Validity

```
s ∈ V   and   s = v₀
```

**Detects:** silent re-ordering of the node array that changes which node the
BFS starts from without any compile error. Since the start node is always
`nodes[0]` by convention, node 0 must be the declared entry point.

**Check:**

```rust
assert nodes[0].name == "State::Goal"   // or configurable start name
```

If start node is configurable, store it explicitly and verify it is present.

---

### 4. Halt Node Validity

```
H ≠ ∅
∀ h ∈ H : h ∈ V
```

**Detects:** topologies with no halt node (path-to-halt check passes vacuously
if H is empty because every node trivially satisfies the empty-halt base case
in some BFS implementations) or halt nodes that reference non-existent names.

Also check:

```
∀ h ∈ H : outdeg(h) = 0
```

unless intentional halt loops exist (and are declared as such).

**Check:**

```rust
assert halt_ids.len() >= 1
for h in halt_ids:
    assert forward[h].is_empty()
```

---

### 5. Edge Uniqueness

```
(u,v) ∈ E ⟹ exactly one transition u→v
```

**Detects:** duplicate `from/to` pairs in `transitions` that cause
non-deterministic weight overwrite or accidental double-weighting during
`embed_tensor_edges`.

**Check:**

```rust
let mut seen: HashSet<(String, String)> = HashSet::new();
for t in transitions:
    assert seen.insert((t.from, t.to)),
        "duplicate edge {t.from} -> {t.to}"
```

---

### 6. Weight Domain Validity

```
∀ (u,v) ∈ E :
    0 ≤ confidence ≤ 1
    0 ≤ safety ≤ 1
    cost ≥ 0
    none of {confidence, safety, cost} is NaN or ∞
```

**Detects:** weights that leave the semiring domain before GPU execution.
NaN propagates silently through matrix multiplication; a weight of `∞` or
`-0.1` produces undefined quantale behavior.

**Check:**

```rust
fn valid_unit(x: f32) -> bool { x >= 0.0 && x <= 1.0 && x.is_finite() }
fn valid_cost(x: f32) -> bool { x >= 0.0 && x.is_finite() }

assert valid_unit(t.confidence)
assert valid_unit(t.safety)
assert valid_cost(t.cost)
```

---

### 7. Bottom Means Missing Edge

```
W[i,j] = ⊥ ⟺ (vᵢ,vⱼ) ∉ E
```

**Detects:** zero-confidence edges that are structurally declared but
semantically absent, causing the matrix to treat a declared transition as
impossible. In practice this catches:

```json
{"confidence": 0.0, "safety": 0.0, "cost": 0.0}
```

when `0.0` confidence is intended to mean "no edge."

**Check:** warn (not hard-reject) when a declared edge has `confidence == 0.0`.
The invariant is that only missing transitions should produce bottom rows, not
explicitly declared ones.

---

### 8. Known Executable Action Binding

```
∀ v ∈ V_exec : op(v) ≠ Unknown
```

Where `V_exec` is the set of `State::*` and `Control::*` nodes whose names
do not start with `Event::`.

**Detects:** State and Control nodes that have no entry in `operators.json`,
meaning the executor will return exit code 127 with no useful error.

**Check (cross-file):**

```rust
for node in nodes where node.type in ["State", "Control"]:
    if operator_registry.get(node.name).is_none():
        warn("[missing_operator] {node.name} has no operator binding")
```

This is a warning, not a hard failure, because some Control nodes are
intentional no-ops (`true` binary). Elevate to error for any node whose
operator is the `true` binary but whose name implies real I/O.

---

### 9. Gate Dominance

```
x requires gate g ⟹ g dominates x
```

A node `g` dominates node `x` if every path from start to `x` passes through
`g`. The specific required dominance relationships in this topology:

```
Control::Commit     is dominated by  State::Validate
State::Memory       is dominated by  Control::Commit
Event::ReceiptAccepted  dominated by Control::GateReceipt
Event::HashNonzero      dominated by Event::ReceiptAccepted
```

**Detects:** bypass edges that allow reaching commit without validation, or
memory writes without commit. This is the strongest safety property in the
checker.

**Algorithm:** standard dominator tree (Lengauer-Tarjan or simple iterative
dataflow). For a topology of ≤ 100 nodes the simple O(n²) iterative algorithm
is sufficient.

```
dom(start) = {start}
dom(v)     = {v} ∪ ⋂_{p ∈ pred(v)} dom(p)

iterate until fixed point
```

**Declared required pairs** (stored in a config struct, not hardcoded strings
scattered through logic):

```rust
const REQUIRED_DOMINATORS: &[(&str, &str)] = &[
    ("State::Validate",        "Control::Commit"),
    ("Control::GateReceipt",   "Event::ReceiptAccepted"),
    ("Event::ReceiptAccepted", "Event::HashNonzero"),
    ("Event::HashNonzero",     "State::Validate"),
];
```

---

### 10. Receipt Cutset Invariant

Every path from `Event::ExecuteFinished` to `Control::Commit` must contain
all nodes in the receipt chain:

```
{Event::ReceiptAttached, Control::GateReceipt,
 Event::ReceiptAccepted, Event::HashNonzero, State::Validate}
```

**Detects:** topology edits that create a shortcut from execution completion
to commit that bypasses receipt validation — for example accidentally adding:

```
Event::ExecuteFinished -> Control::Commit
```

**Algorithm:** enumerate all simple paths from `Event::ExecuteFinished` to
`Control::Commit` (feasible for small topologies) and assert every path
contains every required cutset member. Alternatively, verify that each cutset
member dominates `Control::Commit` when reachability is restricted to the
subgraph between `Event::ExecuteFinished` and `Control::Commit`.

---

### 11. SCC Progress Invariant

```
∀ reachable SCC S : S ∩ H = ∅ ⟹ ∃ (u,v) ∈ E : u ∈ S, v ∉ S
```

**Detects:** non-halt cycles with no exit edge — the planner can loop
indefinitely inside the cycle and never make progress toward halt.

**Algorithm:** Tarjan's or Kosaraju's SCC algorithm, then for each non-trivial
SCC (size > 1) verify at least one outgoing edge to a node outside the SCC.

Self-loops on a halt node are allowed. Self-loops on non-halt nodes are
reported unless they have at least one non-self outgoing edge.

---

### 12. No Zero-Cost Infinite Cycle

```
∀ reachable cycle C : Σ_{e∈C} cost(e) > 0
```

**Detects:** cycles where every edge has `cost = 0`, which can dominate
semiring closure and cause the planner to prefer an infinite internal loop
over any productive path.

**Algorithm:** for each SCC identified in invariant 11, flag the SCC if
every internal edge (both endpoints inside the SCC) has cost=0.  A single
non-zero-cost internal edge is sufficient for progress; the check conservatively
covers all possible cycles in the SCC without enumerating them.

---

### 13. Deterministic Tie-Break

When two outgoing edges have equal `default_weight`:

```
score(e₁) = score(e₂) ⟹ tie(e₁, e₂) = deterministic
```

**Detects:** non-deterministic next-hop selection producing different execution
paths on different runs from the same topology and payload.

**Required canonical ordering:**

```
sort by default_weight desc
then safety desc
then cost asc
then to_node name lexicographic asc
```

**Check:** for each node with multiple outgoing edges, assert no two have
identical `(default_weight, safety, cost, to)` tuples. If identical tuples
exist, flag as indeterminate.

---

### 14. Semiring Identity Law

```
I ⊗ W = W
W ⊗ I = W
```

**Detects:** broken identity matrix semantics where the diagonal is not
correctly initialized to the semiring unit before closure.

**Check (unit test, not topology check):**

```rust
let identity = build_identity_matrix();
assert_matrices_equal(identity.mul(&W), W);
assert_matrices_equal(W.mul(&identity), W);
```

---

### 15. Semiring Bottom Laws

```
⊥ ⊗ x = ⊥
x ⊗ ⊥ = ⊥
x ⊕ ⊥ = x
```

**Detects:** impossible paths that become possible through composition — the
most dangerous semiring bug because it silently routes the frontier into
invalid states.

**Check (unit test):**

```rust
let bottom = QuantaleWeight::BOTTOM;
assert_eq!(bottom.mul(any_weight), bottom);
assert_eq!(any_weight.mul(bottom), bottom);
assert_eq!(any_weight.join(bottom), any_weight);
```

---

### 16. Frontier One-Hot Validity (runtime)

```
fₜ ∈ {e₀, e₁, …, eₙ₋₁}
```

**Detects:** the `Unknown(-1)` frontier class at the point it is created
rather than when it is used.

**Implementation:** add a debug assertion in `frontier_step` and
`commit_exploration_candidate` that the returned `first_hop` is in `[0, n)`.

```rust
debug_assert!(
    (0..n).contains(&decision.first_hop),
    "frontier returned invalid node id: {}",
    decision.first_hop
);
```

In release builds this is a no-op. In test builds it panics immediately at
the source rather than propagating a bad ID through subsequent calls.

---

### 17. Reset Restores Valid Frontier

```
reset() ⟹ f_{t+1} = eₛ  or  f_{t+1} ∈ V_recovery
```

**Detects:** hard resets that land on `Unknown(-1)` (as seen when
`world.decay(0.97)` was called on an all-⊥ world without re-embedding the
topology edges first).

**Implementation:** after any hard reset, assert that the next `frontier_step`
returns a valid node ID before continuing execution.

```rust
if did_hard_reset {
    let post_reset = world.project(projection_bias)?;
    assert!(post_reset.blocked == 0,
        "hard reset did not restore a valid frontier (first_hop={})",
        post_reset.first_hop);
}
```

---

## Implementation — Status: COMPLETE

All five phases shipped in one commit.  88 tests pass.
`cargo run --bin quantale_semiring_v2 -- --check-topology` exits 0.

---

### Phase 1 — Identity and Weight (invariants 1–7, 13) ✅

Added to `topology_check::check()` in `src/topology_check.rs`.

New `ViolationKind` variants:

```rust
DuplicateNodeName,
DuplicateNodeId,
IndexMappingBroken,
InvalidStartNode,
NoHaltNode,
HaltNodeHasSuccessors,
DuplicateEdge,
WeightOutOfDomain,
ZeroConfidenceEdge,
IndeterminateOrdering,   // invariant 13 — added here (structural, no new deps)
```

---

### Phase 2 — Operator Binding (invariant 8) ✅

New entry point in `src/topology_check.rs`:

```rust
pub fn check_with_operators(
    topology: &GraphTopology,
    operator_registry: &OperatorRegistry,
) -> Vec<TopologyViolation>
```

New `ViolationKind`:

```rust
MissingOperator,
```

`OperatorRegistry` imported from `crate::config`; no new files.

---

### Phase 3 — Dominator Checks (invariants 9–12) ✅

Added to `src/topology_check.rs`:

- `compute_dominators()` — iterative O(n²) dataflow fixpoint using reverse
  adjacency
- `kosaraju_sccs()` + iterative DFS helpers (`dfs_finish`, `dfs_collect`)
- `check_receipt_cutset()` — restricted subgraph dominance check for
  invariant 10

New `ViolationKind` variants:

```rust
DominanceViolation,
ReceiptCutsetViolation,
UnsafeSCC,
ZeroCostCycle,
```

**Deviation from plan:** `("Control::Commit", "State::Memory")` was removed
from `REQUIRED_DOMINATORS`.  The market trading path
(`Event::PaperTradeFilled` / `Event::PaperTradeRejected` → `State::Memory`)
is a legitimate bypass of the commit gate added after this plan was written.
All other four required pairs are present and verified by the regression test.

---

### Phase 4 — Semiring Laws (invariants 14–15) ✅

New file `tests/semiring_laws.rs` — pure-Rust, no CUDA device required:

```rust
semiring_identity_left()
semiring_identity_right()
semiring_identity_is_idempotent()   // bonus
semiring_bottom_absorb_left()
semiring_bottom_absorb_right()
semiring_bottom_join_identity()
```

---

### Phase 5 — Runtime Assertions (invariants 16–17) ✅

`src/tensor.rs` — `frontier_step()`:
```rust
debug_assert!(
    report.blocked != 0 || (0..TENSOR_NODE_COUNT as i32).contains(&report.first_hop),
    "frontier_step returned invalid node id: {}",
    report.first_hop
);
```

`src/main.rs` — hard reset path:
```rust
if let Ok(post_reset) = world.project(projection_bias) {
    debug_assert!(post_reset.blocked == 0, "hard reset did not restore valid frontier …");
    if post_reset.blocked != 0 {
        eprintln!("[WARN] hard reset did not restore a valid frontier …");
    }
}
```

Uses `project()` (read-only) so the active set is not advanced by the check.

---

## Files Changed

| File | Change |
|------|--------|
| `src/topology_check.rs` | Full rewrite — phases 1, 2, 3 |
| `src/tensor.rs` | `frontier_step`: debug_assert (phase 5) |
| `src/main.rs` | Hard reset: post-reset project check (phase 5) |
| `tests/semiring_laws.rs` | **New** — algebraic law unit tests (phase 4) |
| `tests/topology_check.rs` | New tests for all phase 1 + phase 3 invariants |

---

## Acceptance Criteria — All Met ✅

- `cargo check` passes
- `cargo test --no-default-features` passes — 88 tests (32 topology_check,
  6 semiring_laws, 12 tensor_quantale, 8 exploration, 28 lib, + others)
- `--check-topology` exits 0 on bundled topology.json (60 nodes, 70 transitions)
- Injecting a bypass edge `State::Start → Control::Commit` is caught
  (`DominanceViolation` on `Control::Commit`)
- Injecting a duplicate node name is caught (`DuplicateNodeName`)
- Injecting `"confidence": 1.5` is caught (`WeightOutOfDomain`)
- A trapped two-node cycle with no exit is caught (`UnsafeSCC`)
- A cycle with all-zero costs is caught (`ZeroCostCycle`)
- Semiring identity and bottom law tests pass (6/6)
- Frontier one-hot `debug_assert` fires on invalid `first_hop` in debug builds

---

## Priority Order — All Shipped

```
1.  unique node names                (Phase 1) ✅
2.  stable index mapping             (Phase 1) ✅
3.  start / halt validity            (Phase 1) ✅
4.  weight domain validity           (Phase 1) ✅
5.  duplicate edge rejection         (Phase 1) ✅
6.  known executable action binding  (Phase 2) ✅
7.  commit dominated by validate     (Phase 3) ✅
8.  memory dominated by commit       (Phase 3) ⚠ skipped — market path bypass
9.  receipt cutset before commit     (Phase 3) ✅
10. SCC progress / cycle exit        (Phase 3) ✅
11. semiring identity + bottom laws  (Phase 4) ✅
12. frontier one-hot assertion       (Phase 5) ✅
13. reset restores valid frontier    (Phase 5) ✅
```

---

## Non-Goals

- Runtime weight learning policy enforcement
- Operator semantic validation (what the operator does, not whether it exists)
- Cross-session invariant discovery / mining
- Automatic topology repair
- Fixing cycle structure (cycles are valid as long as they have exit edges)

---

# Phase 6 — Runtime Tensor Invariants (planned)

## Motivation

The static checker validates the **declared** graph `A`.  At runtime the
effective tensor matrix `W_t` diverges from `A` through learning, decay, and
LLM overlay:

```
A ≠ W_t
```

The failures at steps 92–97 are not graph-structure violations.  They are
**runtime tensor violations** — the effective matrix at step-time had bottom
rows that the static checker never saw.

---

## Variables

| Symbol | Meaning |
|--------|---------|
| `A` | static topology adjacency (declared transitions) |
| `W_0` | base tensor matrix after initial embed (immutable reference copy) |
| `W_t` | effective runtime tensor at step `t` (after decay / learn / overlay) |
| `P_t` | printed projection `(u → v)` at step `t` |
| `h_t` | selected `first_hop` node id |
| `s_t` | selected score value |
| `⊥` | bottom — invalid / no live transition |
| `B` | blocked flag |
| `L` | learned LLM overlay edge set |

---

## New Invariants

### 18. Score-Bottom Implies Blocked

```
s_t = ⊥  ⟹  blocked = 1  ∧  h_t ∈ V_recovery ∪ {-1}
```

**Observed violation (steps 92–94):**

```
projection=(State::Learn->Control::Block)
first_hop=State::Validate  score=⊥  blocked=0
```

Score is bottom but `blocked=0` and `first_hop` is a real node.  The kernel
selected a hop on a bottom score — this is illegal.

**Check:** in `frontier_step`, after the kernel returns, assert:

```rust
if report.selected_value <= BOTTOM {
    assert!(report.blocked != 0 && report.first_hop < 0,
        "score=⊥ but blocked={} and first_hop={}: \
         kernel advanced frontier on bottom score",
        report.blocked, report.first_hop);
}
```

---

### 19. Projection–First-Hop Consistency

```
P_t = (u → v)  ∧  s_t ≠ ⊥  ⟹  h_t = v
```

**Detects:** cases where the printed projection destination and the committed
`first_hop` disagree — a sign the decision report was read from a stale buffer
or two kernels raced.

**Check:**

```rust
if report.selected_value > BOTTOM {
    assert_eq!(report.first_hop, report.selected_dst,
        "projection says {} but first_hop is {}",
        report.selected_dst, report.first_hop);
}
```

---

### 20. No Frontier Advance on Bottom

```
s_t = ⊥  ⟹  f_{t+1} = f_t  ∨  f_{t+1} = r
```

where `r` is a declared recovery node.

**Observed violation:**

```
score=⊥
[STEP] Tensor frontier advanced to node: State::Validate
```

**Biggest missing check.**  The executor must not be called when score is
bottom.  Add a guard in the main loop before `executor.execute_abstract_node`:

```rust
if decision.selected_value <= BOTTOM && decision.blocked == 0 {
    eprintln!("[WARN] score=⊥ but not blocked; skipping executor call");
    consecutive_blocks += 1;
    continue;
}
```

---

### 21. Runtime Matrix Liveness After Overlay

```
∀ v ∈ Reach(W_t), v ∉ H  ⟹  ∃ u : W_t[v, u] ≠ ⊥
```

**Detects:** LLM overlay or decay that creates bottom rows for reachable
non-halt nodes — the runtime equivalent of the static `DeadEnd` check, but
applied to `W_t` rather than `A`.

**Implementation:** after any `embed_tensor_edges` or `decay` call, download
the confidence layer and verify no reachable non-halt node has an all-zero
row.  Expensive in the hot path; run on a debug flag or after every N steps.

---

### 22. LLM Overlay Topology Check Before VRAM

```
L must pass check(L | W_0) = true  before  W_t = W_0 ⊕ L
```

**Observed trigger:**

```
[ALGEBRA] Tensor LLM plan: 20 edge(s) → VRAM
```

Learned edges are currently uploaded unconditionally.  Required pipeline:

```
LLM plan edges
  → endpoint validity (all names in node registry)
  → no bottom weights (confidence > 0, safety ≥ 0, cost ≥ 0, all finite)
  → no required-edge deletion (edges in W_0 must not be zeroed by L)
  → dominance-safe (L must not add bypass edges that violate REQUIRED_DOMINATORS)
  → receipt cutset safe (L must not create ExecuteFinished→Commit shortcuts)
  → then upload to VRAM
```

New entry point needed:

```rust
pub fn check_overlay(
    overlay_edges: &[TensorEdge],
    base_topology: &GraphTopology,
    registry: &NodeRegistry,
) -> Vec<OverlayViolation>
```

---

### 23. Base Matrix Preserved on Reset

```
reset()  ⟹  W_t := W_0
reset()  ⟹  f_t := e_start
```

**Root cause of the hard-reset failure:** `decay(0.97) + embed` tried to
recover through an already-bottomed `W_t`.  The fix (calling `world.reset()`
before `embed`) was shipped in the previous commit.  The invariant to codify:

Keep an immutable copy `W_base` at startup (after initial embed, before any
decay or learning).  Hard reset must restore from `W_base`, not try to lift
the current broken matrix.

**Required change:** in `TensorQuantaleWorld`, store a `base_tensor: Vec<f32>`
snapshot taken immediately after the first `embed_tensor_edges` call.  Hard
reset copies `base_tensor` back to the device rather than calling
`embed_tensor_edges` again from the host.

---

### 24. Block Node Enforces Blocked Semantics

```
node = Control::Block  ⟹  blocked = 1  ∨  halted = 1  ∨  f_{t+1} = recovery
```

**Observed violation:**

```
operator=Control::Block  exit=0  outcome=Success  blocked=0  score=⊥
```

`Control::Block` executing with `outcome=Success` and `blocked=0` is
semantically contradictory.  A block node must either set the blocked flag,
transition to halt, or transition to a declared recovery node.  It must not
silently succeed and return to normal flow on a bottom score.

**Check:** after executing a node whose name starts with `Control::Block`,
assert the resulting decision has `blocked=1` or `halted=1`.

---

### 25. Executable Nodes Have Known Action Semantics

```
v ∈ State ∪ Control  ⟹  action(v) ≠ "unknown"
```

**Observed in logs:**

```
action="unknown"  for  State::Learn, State::Validate, Control::Block, State::Input
```

The existing `MissingOperator` check (Phase 2) only verifies that an operator
entry exists in `operators.json`.  It does not verify the operator has a
declared `action` field that maps to a known semantic.

**Stronger check:** in `check_with_operators`, additionally verify that every
`State` / `Control` node's operator entry contains a non-empty `action` or
`output_mode` field.

---

## Canonical Fix Equation

```
W_t = W_0 ⊕ L_t

with:

check(W_0) = true           (static topology checker, phases 1–3)
check(L_t | W_0) = true     (overlay checker, invariant 22)
reset(W_t) = W_0            (base matrix preservation, invariant 23)
s_t = ⊥  ⟹  blocked = 1   (score-bottom implies blocked, invariant 18)
```

The learned runtime matrix can extend the base topology, but it cannot erase
recovery paths, bypass gates, or advance on bottom.

---

## Diagnosis

```
Missing layer = runtime tensor invariant checker
```

The static graph checker is correct and complete for `A`.  The current failure
class is:

```
runtime selected a hop while score was ⊥
→ executor ran on an invalid node
→ reset tried to recover through a bottomed tensor
→ active[] stayed at -1
→ Unknown(-1) loop
```

The next fix layer is **not** more static topology checks.  It is a runtime
gate applied before each executor call:

```rust
// Before every executor.execute_abstract_node call:
assert!(decision.selected_value > BOTTOM || decision.blocked != 0,
    "invariant 18: score=⊥ with blocked=0; refusing to advance frontier");
assert!(decision.blocked == 0 || decision.first_hop < 0,
    "invariant 18/19: inconsistent decision report");
```

---

## Phase 6 Implementation Plan

### New `ViolationKind` variants (runtime)

```rust
ScoreBottomNotBlocked,          // invariant 18
ProjectionFirstHopMismatch,     // invariant 19
FrontierAdvancedOnBottom,       // invariant 20
RuntimeDeadRow,                 // invariant 21
OverlayEndpointInvalid,         // invariant 22
OverlayWeightOutOfDomain,       // invariant 22
OverlayDominanceViolation,      // invariant 22
BaseMatrixCorrupted,            // invariant 23
BlockNodeNotBlocked,            // invariant 24
UnknownActionSemantics,         // invariant 25
```

### New files

| File | Purpose |
|------|---------|
| `src/overlay_check.rs` | `check_overlay()` — validates LLM edge sets before VRAM upload |
| `src/runtime_check.rs` | `check_decision()` — validates `DecisionReport` before executor |

### Changes to existing files

| File | Change |
|------|--------|
| `src/tensor.rs` | Store `base_tensor` snapshot; `reset()` restores from it |
| `src/main.rs` | Call `check_decision()` before every `execute_abstract_node` |
| `src/main.rs` | Call `check_overlay()` before every `embed_tensor_edges` from LLM plan |

### Priority Order

```
18. score_bottom_blocks                    (main.rs guard — fast)
19. projection_first_hop_consistency       (tensor.rs assertion)
20. no_frontier_advance_on_bottom          (main.rs guard — fast)
22. llm_overlay_topology_check_before_vram (new overlay_check.rs)
23. base_matrix_preserved_on_reset         (tensor.rs base snapshot)
24. block_node_sets_blocked_or_halted      (main.rs post-exec check)
21. runtime_matrix_liveness_after_overlay  (debug flag — expensive)
25. executable_nodes_have_known_action     (extend check_with_operators)
```
