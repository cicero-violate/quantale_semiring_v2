# Unified State / Control / Event Matrix Plan

## Target shape

The matrix node universe is not only system state. It is:

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
```

Every state, control, event, gate, receipt, and commit decision is matrix-addressable.

```text
NodeID -> row/column in one quantale matrix
Edge   -> scored movement between any two nodes
```

## Algebra

Use the confidence/cost quantale fragment:

```text
L = [0, 1]
∨ = max
⊗ = multiplication
⊥ = 0
e = 1
```

Propagation:

```text
x'_j = ⋁ᵢ xᵢ ⊗ Mᵢⱼ
```

Concrete CUDA operation:

```text
x'_j = maxᵢ(xᵢ * Mᵢⱼ)
```

## Matrix blocks

The single matrix can be read as a typed block matrix:

```text
          dst: State      Control      Event
src:
State       S→S          S→C          S→E
Control     C→S          C→C          C→E
Event       E→S          E→C          E→E
```

All blocks live in the same dense CUDA matrix.

## Current node counts

```text
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 44 × 44 = 1936
```

## Implemented node sets

```text
StateNode:
Goal, Input, Parse, Map, Search, Score, Select, Plan, Optimize,
Execute, Validate, Memory, Learn

ControlNode:
Allow, Block, Retry, Repair, Commit, Rollback, Halt, GateInput,
GateExecution, GateReceipt, GateMemory, GateLearn, ChooseBest

EventNode:
FactArrived, InputAccepted, ParseOk, ParseErr, MapReady,
CandidateFound, ScoreReady, TopKSelected, PlanReady, OptimizeReady,
ExecuteStarted, ExecuteFinished, ReceiptAttached, ReceiptAccepted,
ReceiptRejected, HashNonzero, MemoryWritten, LearnUpdated
```

## Example path

```text
State::Execute
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

Every arrow is a matrix cell with a cost/eval value.

## CPU boundary

CPU owns:

```text
node definitions
edge/eval construction
real side effects
receipt collection
memory persistence
```

GPU owns:

```text
M[N×N]
join
composition
closure
active frontier
witness matrix
projection
```

## Completed changes

- [x] Replaced product-state-only model with a unified `Node` enum.
- [x] Added `StateNode`, `ControlNode`, and `EventNode`.
- [x] Added offsets for a single shared node index space.
- [x] Replaced product-state edge construction with node-to-node edges.
- [x] Added control/event nodes for gates, receipts, commit, memory, learn, halt.
- [x] Kept compatibility aliases for `STATE_COUNT` and `state_name`.
- [x] Updated CUDA `N` from product-state count to unified node count.
- [x] Updated reports to probe `Goal → Execute` and `Goal → Learn`.

## Still pending

- [ ] Clean docs/spec wording so it matches the live matrix-edge architecture.
- [ ] Harden the async-style runtime loop: continuously drain bounded ingress, compile inbound events into deltas, feed receipts back into CUDA, and append all records to `quantale.tlog`.
- [ ] Run full validation without workspace guard blocks, including `release_validation` and large-N benchmarks where hardware permits.
- [ ] Optimize CUDA launch/memory behavior by fusing the `quantale_step` / `quantale_frontier_step` boundary and reducing scratch/global-memory traffic.
- [ ] Update Lean/cLean spec only when the Lean/cLean toolchain is available; do not add fake proof scaffolding.
- [ ] Implement real sparse/tiled GPU propagation when scaling beyond the current 44-node dense matrix.
