# Architecture

`quantale_semiring_v2` is a CUDA-first max-times matrix prototype over one unified node universe:

```text
N = StateNode ⊔ ControlNode ⊔ EventNode
Q = ([0,1], max, ×, 0, 1)
M ∈ Q^(44 × 44)
```

## Node groups

```text
StateNode   = 13
ControlNode = 13
EventNode   = 18
NODE_COUNT  = 44
MATRIX_LEN  = 1936
```

### State nodes

```text
Goal       = objective / initial active node
Input      = accepted input surface
Parse      = structured input parsing
Map        = map parsed input into search space
Search     = candidate discovery
Score      = candidate scoring
Select     = top-k or best candidate selection
Plan       = plan construction
Optimize   = optional plan optimization
Execute    = external execution boundary
Validate   = receipt/hash validation point
Memory     = persistence point
Learn      = learning/update point
```

### Control nodes

```text
Allow, Block, Retry, Repair, Commit, Rollback, Halt,
GateInput, GateExecution, GateReceipt, GateMemory, GateLearn,
ChooseBest
```

Control nodes are ordinary matrix nodes. Gates, commits, rollbacks, repair, and halt are not side channels.

### Event nodes

```text
FactArrived, InputAccepted, ParseOk, ParseErr, MapReady,
CandidateFound, ScoreReady, TopKSelected, PlanReady, OptimizeReady,
ExecuteStarted, ExecuteFinished, ReceiptAttached, ReceiptAccepted,
ReceiptRejected, HashNonzero, MemoryWritten, LearnUpdated
```

Events are ordinary matrix nodes. Receipt and memory events are represented directly in the graph.

## Ownership boundary

```text
Rust = launcher / transition-edge upload / compact report decoding / path decoding
CUDA = transition matrix / closure / composition / witness matrix / projection
NVRTC = runtime CUDA compilation path through cudarc
```

Rust does not mirror `M` and does not implement a CPU planner or CPU closure engine. Rust may upload edge batches and may download compact reports or `next_hop[44 × 44]` for path reconstruction.

## Matrix state

```text
transition[44 × 44]          # A / A*
scratch[44 × 44]             # composition and closure scratch
previous[44 × 44]            # step comparison state
next_hop[44 × 44]            # W witness matrix
scratch_next_hop[44 × 44]    # witness scratch
action/frontier buffers      # active[44], next_active[44]
gate_mask[44]                # temporary compatibility projection boundary
report[1]
decision_report[1]
```

## Path and decision flow

```text
1. Rust uploads TransitionEdge batches.
2. CUDA joins edges into M.
3. CUDA computes max-times closure/composition.
4. CUDA projects π(A*) into DecisionReport.
5. Rust decodes selected nodes and, when needed, downloads W to reconstruct a path.
```

Path reconstruction uses only the witness matrix:

```text
path = src → W[src,dst] → W[W[src,dst],dst] → ... → dst
```

## Policy and receipt flow

Policy and runtime receipts compile into matrix-edge deltas:

```text
build_policy_edges(policy)   -> Vec<TransitionEdge>
build_receipt_edges(receipt) -> Vec<TransitionEdge>
```

These deltas are joined into the same matrix:

```text
M := M ∨ M_policy
M := M ∨ M_receipt
```

`gate_mask[44]` remains only as a compatibility boundary for the current projection kernel.

## Non-goals

```text
No CPU planner.
No CPU reference engine.
No CPU mirror of the quantale matrix.
No hidden imperative control outside the matrix, except temporary execution I/O boundaries.
No sparse/tiled implementation while NODE_COUNT = 44.
```
