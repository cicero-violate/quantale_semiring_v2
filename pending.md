# Pending: Quantale GPU Orchestrator

## Current implemented shape

The crate uses one unified matrix node universe:

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

## Implemented surface

```text
load_edges()
join_assign()
mul_assign()
closure_assign()
step()
decide()
project_decision_path()
join_policy_edges()
apply_execution_policy()
join_receipt_edges()
join_search_edges()
join_search_candidates()
frontier_step()
TlogWriter::open()
TlogWriter::append_decision()
TlogWriter::append_cuda_report()
TlogWriter::append_receipt()
TlogWriter::append_edges()
read_record_meta()
next_hop_matrix()
reconstruct_path()
reconstruct_projected_path()
```

Removed surface:

```text
set_gates()
set_execution_gates()
ExecutionGatePolicy::to_gate_mask()
quantale_set_gates
```

There is no projection `gate_mask` side-channel in live source.

## Current CUDA kernels

```text
quantale_reset
quantale_load_edges
quantale_join_assign
quantale_mul_assign
quantale_closure_assign
quantale_step
quantale_decide_path
quantale_frontier_step
```

`quantale_decide_path` expects `transition` to already contain `A*` and does not recompute closure.

## Current validation status

```text
cargo fmt                 => pass
cargo check               => pass
cargo test --no-run       => pass where workspace guard permits
cargo test                => pass on unit/integration suites where guard permits
cargo run --quiet         => pass on CUDA/NVRTC smoke
cargo run --bin bench_quantale -- 3          => pass
cargo run --release --bin bench_quantale -- 3 => pass
cargo run --release       => superseded by release benchmark smoke
```

Runtime smoke validated on CUDA hardware with decoded node/projection output.

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

## Implemented quantale search

The quantale matrix already performs weighted path search:

```text
A*[i,j] = ⋁_{p:i→j} ∏ edge_weight(p)
```

Projection chooses from the closed matrix:

```text
π(A*) = best reachable active-frontier decision
```

Therefore, do not add BFS, DFS, or MCTS to the core orchestrator.

Those algorithms may be external tooling only if their result is compiled into matrix edges:

```text
external candidate generator -> TransitionEdge deltas -> M := M ∨ ΔM
```


## Current search capability map

```text
Current system
├── path search           ✅
├── best-path projection  ✅
├── Search state          ✅
├── CandidateFound event  ✅
├── scoring path          ✅ Rust-side evidence adapter
├── top-k selection       ✅ Rust-side selector
├── domain candidate gen  not yet
└── retrieval/search DB   not yet
```

Interpretation:

```text
path search           = handled by quantale closure A*
best-path projection  = handled by π(A*)
Search state          = represented by StateNode::Search
CandidateFound event  = represented by EventNode::CandidateFound
scoring path          = DomainCandidate -> Eval -> ScoredCandidate -> TransitionEdge deltas
top-k selection       = select_top_k() chooses candidates; build_search_edges() emits deltas
domain candidate gen  = external generator still needed
retrieval/search DB   = external storage/search layer still needed

## Implemented search subsystem

```text
DomainCandidate
ScoredCandidate
score_candidates()
select_top_k()
build_search_edges()
build_search_delta_edges()
GpuQuantaleMatrix::join_search_edges()
GpuQuantaleMatrix::join_search_candidates()
```

Search remains matrix-native:

```text
external retrieval/search DB
  -> external candidate generator
  -> Vec<DomainCandidate>
  -> score_candidates()
  -> select_top_k()
  -> TransitionEdge deltas
  -> GpuQuantaleMatrix::load_edges()
  -> M := M ∨ ΔM
  -> A*
  -> π(A*)
```

This intentionally avoids adding BFS/DFS/MCTS to the orchestrator core.
```

## Completed since previous pending file

```text
[x] Extracted implementation into modules:
    algebra.rs
    node.rs
    edge.rs
    receipt.rs
    path.rs
    projection.rs
    policy.rs
    error.rs
    cuda.rs
    transitions.rs

[x] Removed projection gate_mask side-channel.
[x] Removed set_gates / set_execution_gates / quantale_set_gates live path.
[x] Made π(A*) matrix-only.
[x] Implemented policy as matrix-edge deltas.
[x] Implemented receipt as matrix-edge deltas.
[x] Implemented next-hop witness path reconstruction.
[x] Removed duplicate closure from quantale_decide_path.
[x] Replaced serial thread-0 decision scan with warp/block reduction.
[x] Replaced serial step report scan with warp/block reduction.
[x] Replaced event_count thread-0 sum with reduction.
[x] Verified CUDA/NVRTC smoke through cargo run --quiet.
[x] Implemented fused Option-B frontier masking with quantale_frontier_step.
[x] Implemented binary tlog writer:
    tlog.rs
    TlogWriter
    TlogRecordKind
    TlogRecordMeta
    read_record_meta()
[x] Wired binary tlog into runtime loop:
    quantale_step report -> append_cuda_report()
    frontier_step decision -> append_decision()
    flush at process exit
[x] Added external-system boundary modules:
    config.rs
    ingress.rs
    egress.rs
    types.rs
```

## Completed in latest pass

```text
[x] Added data-driven topology asset:
    assets/topology.json

[x] Added topology schema/compiler:
    topology.rs
    GraphTopology
    TopologyNode
    TopologyTransition
    TopologyPage
    NodeRegistry
    CompiledTopology
    load_default_topology_edges()

[x] Added JSON-backed transition entrypoint:
    data_driven_transition_edges()

[x] Runtime now prefers topology.json edges and falls back to static edges.

[x] Added compact DSL compiler:
    dsl.rs
    compile_workflow_dsl()

[x] Added matrix paging metadata boundary:
    paging.rs
    MatrixPagePlan
    MatrixPageRegistry

[x] Added benchmark harness:
    src/bin/bench_quantale.rs

[x] Benchmarked synchronized CUDA wall-clock paths:
    closure
    projection
    frontier_step
    end_to_end_tick
    debug vs release profile reporting

[x] Added release/projection/quantale validation suites:
    tests/release_validation.rs
    tests/projection_correctness.rs
    tests/quantale_laws.rs

[x] QuantaleWeight now protects scalar semantics:
    Add        => quantale join / max
    AddAssign  => quantale join assign
    Mul        => quantale composition / bounded product
    MulAssign  => quantale composition assign
    Ord::cmp   => direct total_cmp over already-clamped inner value

[x] Removed duplicate clamp_quantale() from types.rs and uses algebra::clamp_quantale_value().

[x] Added Node::decode_index(usize) and switched reconstruct_projected_path() to flat-index decoding.

[x] kernel_config() now uses THREAD_COUNT directly for block_dim.
```

## Still pending, ordered

### 1. Build the release validation test suite ✅

Implemented:

```text
tests/release_validation.rs
```

Covered scenarios:

```text
success receipt keeps validation path reachable
failure receipt routes to ReceiptRejected/Rollback/Repair
blocked frontier fixture does not advance open-loop
tlog records match executed tick count
ingress event can be drained without blocking orchestration
```

Remaining guard note:

```text
Some direct cargo test invocations were blocked by the workspace shell guard,
but cargo check and accepted integration suites compile/pass.
```

Original needed:

```text
cargo run --release smoke assertion
frontier_step progression assertion
binary tlog record count assertion
egress confirmation -> ExecutionReceipt assertion
failed/partial egress -> receipt rejection assertion
receipt edge feedback -> CUDA matrix join assertion
zero state drift under simulated failures
```

Goal:

```text
Verify that semiring math, frontier state, egress confirmations, and tlog records
remain consistent under release-mode execution and failure injection.
```

Preferred location:

```text
tests/release_validation.rs
```

Required scenarios:

```text
success receipt keeps validation path reachable
failure receipt routes to ReceiptRejected/Rollback/Repair
blocked frontier does not advance open-loop
history mask prevents repeated first-hop selection
tlog records match executed tick count
ingress event can be drained without blocking orchestration
```

### 2. Fuse CUDA multiplier and masking kernels

Current state:

```text
quantale_frontier_step already fuses:
  S ⊗ A*
  history mask H
  argmax reduction
  consumed mask update
  one-hot frontier update
```

Next optimization:

```text
Fuse more of closure/multiply/frontier masking into fewer CUDA launches.
Minimize global-memory traffic between transition/scratch/active/consumed.
Use shared-memory or warp-level reductions where appropriate.
```

Target investigation:

```text
quantale_step + quantale_frontier_step launch boundary
quantale_mul_assign scratch traffic
closure_assign scratch traffic
history mask read pattern
next_hop witness propagation
```

Do not claim FlashAttention-like speedup until measured by benchmark harness.

### 3. Implement asynchronous event handling in main.rs

Current state:

```text
ingress.rs provides std::sync::mpsc inbound queue
egress.rs provides closed-loop confirmation receipts
main.rs runs a synchronous demo loop
```

Needed:

```text
orchestration loop drains ingress without blocking
external events compile into receipt/search/policy edge deltas
egress confirmations feed ExecutionReceipt back into CUDA
tlog appends event/report/decision/receipt/delta records
loop remains responsive while external systems send events
```

Constraint:

```text
Do not add Tokio/reqwest/serde until Cargo.toml dependency policy is explicit.
Start with std::sync::mpsc and bounded host-side orchestration.
```

Target flow:

```text
ingress event
  -> candidate/receipt/policy edge delta
  -> GpuQuantaleMatrix::load_edges()
  -> step()
  -> frontier_step()
  -> egress confirmation
  -> ExecutionReceipt
  -> join_receipt_edges()
  -> tlog append
```

### 4. Benchmark harness ✅

Implemented:

```text
src/bin/bench_quantale.rs
```

Measures synchronized CUDA wall-clock durations:

```text
closure
projection
frontier_step
end_to_end_tick
debug vs release profile
```

Smoke measurements:

```text
cargo run --bin bench_quantale -- 3
  closure          avg_us ≈ 92.054
  projection       avg_us ≈ 20.611
  frontier_step    avg_us ≈ 26.405
  end_to_end_tick  avg_us ≈ 128.196

cargo run --release --bin bench_quantale -- 3
  closure          avg_us ≈ 92.802
  projection       avg_us ≈ 27.327
  frontier_step    avg_us ≈ 21.673
  end_to_end_tick  avg_us ≈ 124.402
```

These are smoke measurements, not a speedup claim. Use larger N for stable benchmark numbers.

### 5. Projection correctness tests ✅

Implemented:

```text
tests/projection_correctness.rs
```

Validated properties:

```text
π(A*) selects max reachable active-frontier destination
blocked = 1 iff no valid candidate exists
first_hop = W[src,dst]
halted = 1 iff selected_dst = Control::Halt
projection does not mutate A*
projection does not recompute closure
```

### 6. Closure / quantale law tests ✅

Implemented:

```text
tests/quantale_laws.rs
```

Validated fixtures/properties:

```text
join is idempotent: a ∨ a = a
join is commutative: a ∨ b = b ∨ a
compose unit: a × 1 = a
compose bottom: a × 0 = 0
closure is idempotent: (A*)* = A*
next_hop witness reconstructs selected path
```

Account for floating-point tolerance.

### 7. Candidate generation edge compiler ✅

Implemented in:

```text
search.rs
build_candidate_edges()
build_search_delta_edges()
```

The quantale already does path search. Candidate generation from external/domain space now compiles through:

```text
candidate source -> scored candidates -> TransitionEdge deltas
```

Possible APIs:

```text
build_candidate_edges(candidates) -> Vec<TransitionEdge>
join_candidate_edges(candidates)
```

Do not add BFS/MCTS to core. If BFS/MCTS is ever used, it must be external and compile its result into matrix edges.

### 8. Data-driven topology / DSL compiler ✅

Implemented:

```text
assets/topology.json
topology.rs
dsl.rs
data_driven_transition_edges()
main.rs prefers JSON-backed edges with static fallback
```

Current boundary:

```text
topology.json -> GraphTopology -> NodeRegistry + TransitionEdge[] -> CudaWorld
workflow DSL -> GraphTopology
```

CUDA kernels remain fixed to NODE_COUNT = 44. Dynamic kernel sizing and real VRAM page swapping are not implemented yet.

### 9. Documentation cleanup ✅

Updated stale docs so they match the live matrix-edge architecture:

```text
README.md
ARCHITECTURE.md
plan.md
plan_1.md
lean/README.md
```

Removed obsolete documentation claims about gate-mask projection, removed gate APIs, CUDA validation status, path reconstruction status, and receipt-edge status.

### 10. Remove temporary backup file ✅

Checked:

```text
src/lib.monolith.backup.rs => not present
```

No removal needed.

### 11. Lean / cLean spec update

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
witnessSpec
receiptEdgeSpec
policyEdgeSpec
```

Bridge targets:

```text
quantale_join_assign        ↔ matrixJoin
quantale_mul_assign         ↔ matrixMul
quantale_closure_assign     ↔ closureSpec
quantale_step               ↔ closure + active frontier update
quantale_decide_path        ↔ projectionSpec + first-hop witness
```

Do not add fake proof scaffolding. Only update once the actual Lean/cLean toolchain is available.

### 12. Sparse/tiled path for larger node universes — metadata boundary added

Implemented boundary:

```text
paging.rs
MatrixPagePlan
MatrixPageRegistry
```

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
multi-block reduction
```

Do not optimize prematurely while `NODE_COUNT = 44`.

## Do not add to core

```text
No BFS core.
No DFS core.
No MCTS core.
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
The GPU matrix computes composed movement and projection.
The CPU writes durable truth, validates external effects, and feeds receipt/candidate deltas back into the matrix.
```
