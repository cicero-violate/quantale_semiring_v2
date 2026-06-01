# Plan: Static Topology Checker

## Goal

Add a static graph validator that rejects invalid topology before execution starts.
The checker runs on `assets/topology.json` at startup (and optionally as a standalone
CLI tool) and enforces the five graph invariants below.

No runtime invariant machinery, no invariant-discovery pass. Static check only.
If the topology fails, the process exits with a clear error before any tensor
operations run.

---

## Background

The class of failures seen at runtime:

```
State::Memory -> Unknown(-1)   blocked=1   score=⊥
```

is fully predictable from the topology file alone:

```
∀ v ∈ V : v ∉ H ⇒ ∃ u ∈ V : W[v,u] ≠ ⊥
```

`State::Memory` had zero outgoing edges in the compiled tensor world because the
only edge the LLM produced (`State::Memory → State::Learn`) was not declared in
`topology.json` and was therefore stripped by `filter_static_topology_edges`.
A static checker would have caught this at startup.

---

## Invariants to Check

### 1. Endpoint Validity

```
∀ (u,v) ∈ E : u ∈ V ∧ v ∈ V
```

Every `from` and `to` name in `transitions` must exist in `nodes`.

**Detects:** typos, deleted nodes, copy-paste errors.

---

### 2. Non-Terminal Closure (dead-end check)

```
∀ v ∈ V : v ∉ H ⇒ outdeg(v) > 0
```

Every node that is not a declared halt node must have at least one outgoing
transition.

**Detects:** the exact `State::Memory → Unknown(-1)` class of failure.

---

### 3. Reachability from Start

```
Reach(s) = { v : s ⇝ v }
∀ v ∈ V_required : v ∈ Reach(s)
```

Every node declared in `pages[0].node_names` must be reachable from
`State::Goal` (node id 0, or whichever node carries `"type": "State"` and
appears first).

**Detects:** disconnected subgraphs, forgotten link-up after adding new nodes.

---

### 4. Path to Halt

```
∀ v ∈ V : v ⇝ h for some h ∈ H
```

Every node must lie on some path that eventually reaches a halt node.

**Detects:** cycles or branches that can never terminate.

---

### 5. No Bottom Row Except Halt

```
W[v,*] = ⊥ ⇒ v ∈ H
```

In the compiled weight matrix, only halt nodes may have an all-zero / all-⊥ row.
This is the runtime-equivalent restatement of invariant 2; catching it at
compile-time costs nothing.

**Detects:** nodes added to topology without any outgoing transition.

---

## What It Does NOT Check

- Weight values (those belong to the tensor learning system)
- Whether operators exist for every node (that is `egress.rs` responsibility)
- Cycle detection beyond "can reach halt"
- Semantic correctness of jit_body strings

---

## Implementation

### New file: `src/topology_check.rs`

```
pub struct TopologyViolation {
    pub kind: ViolationKind,
    pub node: String,
    pub detail: String,
}

pub enum ViolationKind {
    UnknownEndpoint,
    DeadEnd,
    Unreachable,
    CannotReachHalt,
}

pub fn check(topology: &GraphTopology) -> Vec<TopologyViolation>
```

`check()` performs all five invariants on a compiled `GraphTopology` and returns
every violation found. The caller decides whether to abort or warn.

Algorithm:

```
1. build node_set: HashSet<&str> from nodes
2. build halt_set: HashSet<usize>  — nodes with action == "halt"
3. build adjacency: HashMap<usize, Vec<usize>>  — from transitions
4. check endpoints   — both names in node_set
5. check dead ends   — outdeg(v) > 0 for v ∉ halt_set
6. BFS from start    — collect Reach(s)
7. check reachability — every page node in Reach(s)
8. reverse BFS from halts — collect CanReachHalt
9. check path-to-halt — every node in CanReachHalt
```

All five passes run before returning; the caller receives all violations at once,
not just the first.

---

### Integration: `src/main.rs`

After `GraphTopology::default_asset()` and `.compile()` succeed, and before
`TensorQuantaleWorld::from_tensor_edges`:

```rust
let violations = topology_check::check(&topology_asset);
if !violations.is_empty() {
    for v in &violations {
        eprintln!("[topology] {} in {}: {}", v.kind, v.node, v.detail);
    }
    std::process::exit(1);
}
```

This means a bad topology.json is caught before any GPU memory is allocated.

---

### Integration: `build.rs`

Optionally run the check as a build-time step by executing a small Rust binary
(or calling the check from a proc-macro). This is a stretch goal; the startup
check is sufficient for now.

---

### CLI option

Add a `--check-topology` flag to the binary:

```bash
./run.sh --check-topology
```

Runs the static check, prints all violations or `OK`, then exits.
Useful in CI without starting the full agent loop.

---

## New Tests: `tests/topology_check.rs`

Each invariant gets at least one passing and one failing case.

```
check_passes_on_valid_topology()
check_rejects_unknown_endpoint()
check_rejects_dead_end_non_halt_node()
check_rejects_unreachable_node()
check_rejects_node_with_no_path_to_halt()
check_reports_all_violations_not_just_first()
current_topology_passes_all_checks()   ← regression guard
```

`current_topology_passes_all_checks` loads the bundled `topology.json` and
asserts zero violations. This is the primary regression guard: any future
topology edit that introduces a dead end fails CI before it reaches execution.

---

## Export in `src/lib.rs`

```rust
pub mod topology_check;
pub use topology_check::*;
```

---

## Acceptance Criteria

- `cargo check` passes
- `cargo test --no-default-features` passes (including all new topology_check tests)
- `current_topology_passes_all_checks` passes on the current topology.json
- `./run.sh --check-topology` prints `OK` and exits 0 on a valid topology
- `./run.sh --check-topology` prints a descriptive violation and exits 1 when a
  dead-end node is injected into topology.json
- `State::Memory` dead-end is permanently caught before execution, not at runtime

---

## Non-Goals

- Runtime invariant enforcement
- Automatic topology repair
- Invariant discovery / mining
- Weight validation
- Cross-file consistency between operators.json and topology.json
