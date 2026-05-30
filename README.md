# quantale_semiring_v2

CUDA-first max-times quantale/semiring orchestrator prototype.

The crate uses one unified matrix node universe:

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
Q = ([0,1], max, ×, 0, 1)
M ∈ Q^(N × N)
```

Current node counts:

```text
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 44 × 44 = 1936
```

Every state, control, event, gate, receipt, commit, rollback, memory, learn, and halt step is addressable as a matrix node.

## Current Rust surface

```text
load_edges()
join_policy_edges()
apply_execution_policy()    # policy edges
join_receipt_edges()
join_assign()
mul_assign()
closure_assign()
step()
decide()                    # compatibility wrapper for project_decision_path()
project_decision_path()     # π(A*) decision projection
next_hop_matrix()
reconstruct_path()
reconstruct_projected_path()
```

Edge builders:

```text
edge(src, dst, value)
edge_eval(src, dst, Eval { confidence, utility, risk, cost })
build_policy_edges(policy)
build_receipt_edges(receipt)
```

## Algebra

```text
join:      a ∨ b = max(a,b)
compose:  a ⊗ b = a × b
bottom:   0
unit:     1
```

Matrix composition:

```text
(A × B)[i,j] = max_k(A[i,k] × B[k,j])
```

Closure/projection split:

```text
Quantale pathing:      A*[i,j] = ⋁_{p:i→j} product(edge weights on p)
Decision projection:   D = π(A*)
Witness extraction:    first_hop = W[D.src,D.dst]
Path reconstruction:   src → W[src,dst] → ... → dst
```

`A*` is CUDA-resident closure over composed graph paths. `π(A*)` is the operational projection that chooses a gated destination from the active frontier. `W` is the CUDA-resident first-hop witness matrix.

## CUDA-owned state

```text
transition[44 × 44]
scratch[44 × 44]
previous[44 × 44]
next_hop[44 × 44]
scratch_next_hop[44 × 44]
active[44]
next_active[44]
event_counts[512]
report[1]
decision_report[1]
```

The CPU does not own or mirror the quantale matrix. Rust may upload transition-edge deltas and may download compact reports or the `next_hop` witness matrix for path decoding. It does not run a planner, reference engine, closure engine, or matrix mirror.

## Node universe

### StateNode

```text
Goal, Input, Parse, Map, Search, Score, Select, Plan, Optimize,
Execute, Validate, Memory, Learn
```

### ControlNode

```text
Allow, Block, Retry, Repair, Commit, Rollback, Halt, GateInput,
GateExecution, GateReceipt, GateMemory, GateLearn, ChooseBest
```

### EventNode

```text
FactArrived, InputAccepted, ParseOk, ParseErr, MapReady,
CandidateFound, ScoreReady, TopKSelected, PlanReady, OptimizeReady,
ExecuteStarted, ExecuteFinished, ReceiptAttached, ReceiptAccepted,
ReceiptRejected, HashNonzero, MemoryWritten, LearnUpdated
```

## Canonical path

```text
State::Goal
→ Control::GateInput
→ Event::FactArrived
→ State::Input
→ Event::InputAccepted
→ State::Parse
→ Event::ParseOk
→ State::Map
→ Event::MapReady
→ State::Search
→ Event::CandidateFound
→ State::Score
→ Event::ScoreReady
→ State::Select
→ Event::TopKSelected
→ State::Plan
→ Event::PlanReady
→ Control::ChooseBest
→ State::Execute
→ Control::GateExecution
→ Event::ExecuteStarted
→ Event::ExecuteFinished
→ Event::ReceiptAttached
→ Control::GateReceipt
→ Event::ReceiptAccepted
→ Event::HashNonzero
→ State::Validate
→ Control::Commit
→ Control::GateMemory
→ State::Memory
→ Event::MemoryWritten
→ Control::GateLearn
→ State::Learn
→ Event::LearnUpdated
→ Control::Halt
```

Every arrow above is a scored matrix edge.

## CUDA kernels

```text
quantale_reset
quantale_load_edges
quantale_join_assign
quantale_mul_assign
quantale_closure_assign
quantale_step
quantale_decide_path
```

`quantale_decide_path` computes `π(A*)` and returns:

```text
selected_src
selected_dst
first_hop
selected_value
halted
blocked
```

`reconstruct_projected_path()` uses `selected_src`, `selected_dst`, and downloaded `next_hop[44 × 44]` to decode the selected path.

## Policy and receipt updates

Policy is represented as weighted Control/Event edges:

```text
policy condition -> weighted Control/Event edges
M := M ∨ M_policy
```

Runtime receipts are represented as transition-edge deltas:

```text
receipt -> TransitionEdge updates
M := M ∨ M_receipt
```

Policy and receipt changes are matrix-edge updates; there is no projection gate-mask side channel in the live source.

## Build

```bash
cargo fmt
cargo check
cargo test --no-run
cargo test
cargo run --release
```

`cargo run --release` requires a CUDA-capable system supported by `cudarc`/NVRTC. There is no `build.rs`, no `nvcc` build script, and no CPU fallback engine.

## Correct mental model

```text
Every system-relevant thing is a node.
Every allowed movement is an edge.
Every edge has an eval/cost.
The GPU matrix computes composed movement.
The CPU only decodes selected paths and performs external side effects.
```

Do not add a CPU planner, CPU reference engine, CPU matrix mirror, or hidden imperative control path outside the matrix, except temporary execution I/O boundaries.
