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
    ("State::Validate",         "Control::Commit"),
    ("Control::Commit",         "State::Memory"),
    ("Control::GateReceipt",    "Event::ReceiptAccepted"),
    ("Event::ReceiptAccepted",  "Event::HashNonzero"),
    ("Event::HashNonzero",      "State::Validate"),
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

**Algorithm:** for each SCC identified in invariant 11, check whether the
total edge cost around the cycle is zero. Flag if so.

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
    let first_step = world.frontier_step(bias)?;
    assert_ne!(first_step.first_hop, -1,
        "hard reset did not restore a valid frontier");
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
