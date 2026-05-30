# Pending: Unified State / Control / Event Quantale Matrix

## Current implemented shape

The crate now uses a single matrix node universe:

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
```

Every state, control, event, gate, receipt, commit, rollback, memory, learn, and halt step is addressable as a matrix node.

```text
M ∈ Q^(N × N)
Q = ([0,1], max, ×, 0, 1)
```

Current size:

```text
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 44 × 44 = 1936
```

Current working surface:

```text
load_edges()
set_gates()              # compatibility projection mask only
set_execution_gates()    # compatibility projection mask only
join_assign()
mul_assign()
closure_assign()
step()
decide()
project_decision_path()
```

Current CUDA kernels:

```text
quantale_reset
quantale_load_edges
quantale_set_gates
quantale_join_assign
quantale_mul_assign
quantale_closure_assign
quantale_step
quantale_decide_path
```

Current validation status:

```text
cargo fmt                 => pass
cargo check               => pass
cargo test --no-run       => pass
cargo run --release       => pending runtime CUDA validation
```

## Implemented node sets

```text
StateNode:
Goal, Input, Parse, Map, Search, Score, Select, Plan, Optimize,
Execute, Validate, Memory, Learn
```

```text
ControlNode:
Allow, Block, Retry, Repair, Commit, Rollback, Halt, GateInput,
GateExecution, GateReceipt, GateMemory, GateLearn, ChooseBest
```

```text
EventNode:
FactArrived, InputAccepted, ParseOk, ParseErr, MapReady,
CandidateFound, ScoreReady, TopKSelected, PlanReady, OptimizeReady,
ExecuteStarted, ExecuteFinished, ReceiptAttached, ReceiptAccepted,
ReceiptRejected, HashNonzero, MemoryWritten, LearnUpdated
```

## Implemented canonical path

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

## Still pending, ordered

### 1. Runtime CUDA validation

Run on CUDA-capable hardware:

```text
cargo run --release
```

Why:

```text
cargo check and cargo test --no-run prove Rust builds, but do not prove NVRTC compilation, PTX module loading, or kernel launch behavior at runtime.
```

Expected output should show decoded node names rather than product-state tuples.

---

### 2. Replace compatibility `gate_mask` with pure matrix-edge policy updates

Status:

```text
implemented matrix-edge policy builder and CUDA join bridge;
projection mask retained until quantale_decide_path is edge-only
```

Current:

```text
gate_mask[N]
ExecutionGatePolicy::to_gate_mask()
quantale_set_gates()
```

This is still a projection mask side-channel.

Target:

```text
policy condition -> weighted Control/Event edges
```

Example:

```text
Control::GateExecution -> Event::ExecuteStarted = high score when allowed
Control::GateExecution -> Control::Block        = high score when blocked
Control::GateReceipt   -> Event::ReceiptAccepted = high score when accepted
Control::GateReceipt   -> Event::ReceiptRejected = high score when rejected
```

Needed API:

```text
build_policy_edges(policy) -> Vec<TransitionEdge>
```

Implemented API:

```text
build_policy_edges(policy) -> Vec<TransitionEdge>
GpuQuantaleMatrix::join_policy_edges(policy)
GpuQuantaleMatrix::apply_execution_policy(policy)
```

Then policy is joined into the same matrix as ordinary movement:

```text
M := M ∨ M_policy
```

Do not remove `gate_mask` until the projection path can be driven entirely by matrix edges.

---

### 3. Runtime receipt-driven edge insertion

Status:

```text
implemented runtime receipt value object, receipt-to-edge builder,
and CUDA matrix join bridge
```

Current default graph includes static receipt edges.

Needed:

```text
receipt -> TransitionEdge updates
```

Example accepted receipt:

```text
Event::ReceiptAttached -> Control::GateReceipt     = 0.97
Control::GateReceipt   -> Event::ReceiptAccepted   = receipt_confidence
Event::ReceiptAccepted -> Event::HashNonzero       = hash_score
Event::HashNonzero     -> State::Validate          = validation_score
```

Example rejected receipt:

```text
Control::GateReceipt   -> Event::ReceiptRejected = rejection_score
Event::ReceiptRejected -> Control::Rollback      = rollback_score
Control::Rollback      -> Control::Repair        = repair_score
```

Needed API:

```text
build_receipt_edges(receipt) -> Vec<TransitionEdge>
```

Implemented API:

```text
ExecutionReceipt
build_receipt_edges(receipt) -> Vec<TransitionEdge>
GpuQuantaleMatrix::join_receipt_edges(receipt)
```

---

### 4. Full path reconstruction

Status:

```text
implemented next-hop matrix download and CPU path reconstruction
```

Current projection returns:

```text
selected_src
selected_dst
first_hop
selected_value
```

Needed:

```text
path = selected_src -> first_hop -> ... -> selected_dst
```

CUDA already maintains:

```text
next_hop[N × N]
```

Needed Rust method:

```text
reconstruct_path(src, dst) -> Vec<Node>
```

Implemented API:

```text
reconstruct_path_from_next_hop(next_hop, src, dst) -> Result<Vec<Node>, CudaError>
GpuQuantaleMatrix::next_hop_matrix() -> Result<Vec<i32>, CudaError>
GpuQuantaleMatrix::reconstruct_path(src, dst) -> Result<Vec<Node>, CudaError>
GpuQuantaleMatrix::reconstruct_projected_path() -> Result<Vec<Node>, CudaError>
```

This may download only the `next_hop` matrix, not the quantale value matrix.

---

### 5. Edge/eval builder cleanup

Status:

```text
implemented edge_eval builder and converted default/persistence graph edges
to explainable Eval tuples while preserving previous scalar weights
```

Current:

```text
Eval { confidence, utility, risk, cost }
Eval::weight()
```

But default edges are still hand-authored scalar weights.

Target:

```text
edge_eval(src, dst, Eval { ... }) -> TransitionEdge
```

Implemented API:

```text
edge_eval(src, dst, eval) -> TransitionEdge
default_transition_edges() -> Vec<TransitionEdge>
persistence_transition_edges() -> Vec<TransitionEdge>
```

Then all default edges should be expressed as explainable eval tuples, not raw floats.

---

### 6. Sparse/tiled path for larger node universes

Current dense matrix is fine:

```text
44 × 44 = 1936 cells
```

If node count grows:

```text
N > 512 or N > 1024
```

consider:

```text
sparse edge propagation
blocked/tiled matrix closure
frontier-only projection
```

Do not optimize prematurely while `NODE_COUNT = 44`.

---

### 7. Lean / cLean spec update

Current Lean spec is stale relative to the implementation.

Needed spec:

```text
Node = StateNode ⊔ ControlNode ⊔ EventNode
encode : Node -> Fin NODE_COUNT
Q = ([0,1], max, ×, 0, 1)
matrixJoin
matrixMul
closureSpec
projectionSpec
```

Bridge targets:

```text
quantale_join_assign        ↔ matrixJoin
quantale_mul_assign         ↔ matrixMul
quantale_closure_assign     ↔ closureSpec
quantale_decide_path        ↔ projectionSpec + first-hop witness
```

Do not add fake proof scaffolding. Only update once the actual Lean/cLean toolchain is available.

---

### 8. Documentation cleanup

Status:

```text
completed README.md, ARCHITECTURE.md, and lean/README.md refresh for the
44-node unified State/Control/Event max-times matrix shape
```

Update these to match the current matrix shape:

```text
README.md
ARCHITECTURE.md
lean/README.md
```

Removed stale references to:

```text
20-node hardcoded state graph
product-state-only matrix
max-plus if the current implementation remains max-times
```

---

## Do not add

```text
No CPU planner.
No CPU reference engine.
No CPU mirror of the quantale matrix.
No hidden imperative controls outside the matrix, except temporary execution I/O boundaries.
No separate search layer unless it compiles into matrix edges or CUDA matrix operations.
```

## North star

```text
Every system-relevant thing is a node.
Every allowed movement is an edge.
Every edge has an eval/cost.
The GPU matrix computes composed movement.
The CPU only decodes the selected path and performs external side effects.
```
