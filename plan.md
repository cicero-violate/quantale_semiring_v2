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

- [ ] Replace the projection `gate_mask` compatibility path with pure matrix-edge updates.
- [ ] Add runtime receipt-driven edge updates.
- [ ] Add path reconstruction beyond first-hop witness.
- [ ] Update Lean spec from product state to unified node algebra.
- [ ] Runtime-test CUDA with `cargo run --release` on CUDA hardware.
