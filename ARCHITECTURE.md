# Architecture

`quantale_semiring_v2` is a CUDA-first tensor quantale orchestrator with a data-driven CKA structural compiler, effect-safe parallel scheduler, and a topology DSL that compiles algebraic orchestration programs into quantale-valued tensor graphs and JIT kernel fusion regions.

## Core invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Tensor substrate

```text
T ∈ R^(3 × N × N)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

Node universe:

```text
Current topology: 74 nodes
Source of truth:  assets/topology.source.json
Generated graph:  assets/topology.generated.json  (runtime input)
Rust constant:    TENSOR_NODE_COUNT is derived from generated topology at build time
MATRIX_LEN        = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT
TENSOR_LEN        = 3 * MATRIX_LEN
```

No Rust code encodes the node list. `topology.rs::NodeRegistry` loads it from the generated JSON at startup. Adding a node requires only a topology source edit followed by `cargo run -- topology build-overlay`.

## Data-driven node registry

`topology.rs::NodeRegistry` is the primary registry:

```rust
registry.id_of("State::Execute")   // → Option<usize>
registry.name_of(9)                 // → Option<&str>
registry.action_of(17)              // → Option<&str> from JSON "action" field
registry.len()                      // → node count
registry.matrix_len()               // → len * len
```

`SystemConfig`, `ExplorationEngine`, `batch.rs`, and `egress.rs` all derive dimensions from the registry. No component holds a hard-coded `NODE_COUNT` constant.

## Topology DSL pipeline

Orchestration is declared in source topology, compiled into generated artifacts, and consumed exclusively by the runtime:

```text
assets/topology.source.json          (authoritative DSL source)
  ├─ slots: declared tensor/state/config/log slots with kinds
  ├─ resources: declared lock/gpu resources
  ├─ quantale: layer declarations (join, compose, bottom, unit per layer)
  ├─ nodes: 74 typed nodes (kernel|host_node|boundary_node|policy_node|event_node)
  └─ programs: CKA expressions with quantale weights

  ↓  cargo run -- topology build-overlay

assets/topology.generated.json       (flat quantale-valued tensor graph)
assets/operators.generated.json      (compiled operator registry)
assets/patterns.source.json          (CKA tensor-weight patterns; replaces patterns.json)
assets/topology.fusion.json          (maximal GPU-safe kernel fusion regions)
```

Validators run at every build-overlay:

```text
validate_unique_source_node_names    no duplicate node names
validate_known_backends              runtime.backend ∈ {cuda,python,noop,patch,cargo,...}
validate_source_node_effects         every reads/writes/locks slot/resource exists
validate_boundary_governance         every boundary_node has a governance object
validate_kernel_slot_purity          kernel reads/writes only tensor-kind slots
validate_quantale_layers             algebraic laws (unit/bottom/compose/join)
validate_par_independence            par branches are effect-independent
compile_source_programs              CKA → flat transitions + parallel_groups
partition_fusible_regions            Fusable(F) = B∧K∧S∧A∧R
```

## GPU ownership

CUDA owns:

```text
tensor[3 × N × N]
scratch[3 × N × N]
witness[3 × N × N]
scratch_witness[3 × N × N]
consumed[N × N]
active[N]
next_active[N]
decision[1]
exploration_tokens[N × N]
exploration_scores[N × N]
exploration_parents[N × N]
exploration_selected[N]
```

Rust owns:

```text
JSON asset loading and NodeRegistry
Topology DSL compiler (crates/topology_core)
CKA pattern compilation
FusionDispatch: fusion region loading and JitChain construction
JitCache: NVRTC/PTX compilation and kernel caching (#[cfg(feature="cuda")])
Static topology invariant checking
Runtime decision invariant checking
Base tensor CPU snapshot for hard reset
Operator effect validation
Batch scheduling policy
Host-side operator execution (process and jit_cuda backends)
Edge-delta upload
Compact report decoding
Append-only transaction logging
```

Rust does not own a CPU planner or a live CPU mirror of the tensor.

## CUDA kernel split

Three distinct compilation and loading paths — all dispatch through cudarc:

```text
cuda/quantale_world.cu
  Compiled: NVRTC at runtime (cudarc::nvrtc::compile_ptx)
  Kernels:  tensor_quantale_reset, embed_edges, closure, project,
            project_batch, commit_batch, frontier_step, tick,
            update_edge, decay, seed_exploration, expand_tokens,
            score_tokens, select_topk_tokens, commit_exploration

JIT fusion kernels (src/jit_kernel_fusion/)
  Compiled: NVRTC at runtime via JitCache::get_or_compile
  Source:   synthesize_kernel(&JitChain, &registry) → CUDA C
  Regions:  loaded from assets/topology.fusion.json via FusionDispatch
```

`quantale_world.cu` provides the deterministic tensor execution core. Fusion kernels use NVRTC to specialize operator chains at runtime once fusion regions are identified.

## Fusion architecture

The partition and dispatch pipeline:

```text
topology.fusion.json
  → FusionDispatch::load(path, &operator_registry)
      → detect_jit_chains(region.nodes, registry)    [jit_kernel_fusion::chain]
          checks executable=jit_cuda, validates data-flow linkage
      → JitChain { operators, inputs, outputs, internals }
      → chain_metadata → JitChainMetadata { estimated_savings, ... }

  dispatch lookup (O(1)):
      is_fusion_entry(node)    → true if node starts a fusion region
      get_by_entry(node)       → &FusionEntry
      get_by_member(node)      → &FusionEntry (any node in chain)

  synthesis (no device needed):
      synthesize_all(&registry)  → CUDA C source strings (startup dry-run)

  compilation (cfg(feature="cuda")):
      JitCache::get_or_compile(device, &chain, registry)
          → synthesize_kernel → CUDA C
          → compile_ptx (NVRTC)
          → load_ptx + cache by operator sequence
          → CudaFunction
```

Current emitted region:

```text
Analysis::Return1 → Analysis::Volatility → Analysis::SignalScore
  backend:   cuda_jit
  fusion:    linear_chain
  inputs:    [market.open, market.price]   (external reads)
  outputs:   [analysis.signal_score]       (external writes)
  internals: [analysis.return, analysis.volatility]
```

## Data-flow architecture

```text
assets/topology.source.json
  ──▶ topology build-overlay
        ├─ topology.generated.json    (flat quantale transitions)
        ├─ operators.generated.json   (compiled operator registry)
        ├─ patterns.source.json       (CKA patterns from programs)
        └─ topology.fusion.json       (fusible kernel regions)

assets/topology.generated.json
  → NodeRegistry
  → TensorEdge[]

assets/patterns.source.json
  → CkaExpr
  → CompiledCkaPattern { edges, parallel_groups }
  → TensorEdge[]

assets/topology.fusion.json
  → FusionDispatch { entries, by_entry, by_member }
  → JitChain per region
  → synthesize_kernel → CUDA C
  → JitCache (NVRTC) → CudaFunction (cfg(feature="cuda"))

TensorEdge[]
  → TensorQuantaleWorld
  → tensor_quantale_closure
  → exploration scheduler
      → tensor_quantale_seed/expand/score/topk/commit
  → CKA scheduler fallback
      → tensor_quantale_project_batch
      → validate_parallel_group_effects
      → tensor_quantale_commit_batch
      → dispatch_decision_batch_blocking
  → single-step fallback
      → tensor_quantale_frontier_step
  → ProcessReceipt
  → tensor_quantale_update_edge
  → T := T ∨ ΔT
```

## CKA layer

CKA is an edge-delta compiler above the tensor substrate, not a replacement runtime.

```text
CKA = { 0, 1, +, ;, *, || }
```

Rust model (CkaExpr in src/pattern.rs):

```text
Zero   — produces no executable edges (bottom)
One    — identity/skip
Node   — atomic endpoint
Seq    — composes adjacent endpoints
Choice — quantale join; no cross-edges between alternatives
Star   — bounded finite unroll only
Par    — independent branches + parallel group metadata + effect check
```

Source topology programs compile the same algebra via `crates/topology_core/src/programs.rs`. Node names are validated against the registry at compile time.

## Exploration layer

```text
assets/exploration.json
  → ExplorationConfig
  → ExplorationEngine (node_count from NodeRegistry)
  → TensorQuantaleWorld::expand_exploration
  → ExplorationCandidate top-K
  → effect validation
  → TensorQuantaleWorld::commit_exploration_candidate
```

Scoring:

```text
V(H) = confidence - cost + safety + η·novelty + ρ·receipt_prior - λ·entropy
```

Anti-repeat state (host-owned, uploaded per tick):

```text
terminal_visits[N], first_hop_visits[N]
repeat_penalty, max_terminal_visits, max_first_hop_visits
```

## Parallel scheduler

```text
project_ready_batch_plan(...)
  → TensorQuantaleWorld::project_parallel_group(...)
  → tensor_quantale_project_batch          (read-only)
  → validate_parallel_group_effects(...)   (effect safety)
  → prepare_parallel_batch_plan(...)
  → TensorQuantaleWorld::commit_decision_batch(...)
  → tensor_quantale_commit_batch           (mutates frontier)
  → dispatch_decision_batch_blocking(...)
```

CUDA projection is read-only. Commit occurs only after whole-group validation. All backends go through a single uniform dispatch path.

## Operator dispatch

`egress.rs::execute_abstract_node_blocking` routes by the `executable` field in `operators.generated.json`:

```text
executable = "jit_cuda"       → jit_cuda chain (jit_kernel_fusion; NVRTC)
executable = "cuda_ptx"       → load precompiled PTX module
executable = anything else    → Command::new(binary)
```

`jit_cuda` operators are chained into `JitChain`s by `FusionDispatch`. Without `--features cuda` both GPU paths return an explicit capability error — no process-spawn fallback.

## Operator coverage

Every topology node has an operator contract. Symbolic Control/Event nodes use safe `true` no-op contracts. Real operators keep concrete contracts. The `validate_known_backends` pass enforces that every declared node backend is a recognised value.

## Effect safety contract

```text
safe_parallel(a,b) =
  writes(a) ∩ writes(b) = ∅
  ∧ writes(a) ∩ reads(b) = ∅
  ∧ reads(a) ∩ writes(b) = ∅
  ∧ locks(a) ∩ locks(b) = ∅
```

Effects are declared in `topology.source.json` node entries and crosschecked at build-overlay time.

## CUDA kernels (quantale world)

Tensor kernels (NVRTC, `cuda/quantale_world.cu`):

```text
tensor_quantale_reset
tensor_quantale_embed_edges
tensor_quantale_closure
tensor_quantale_project
tensor_quantale_project_batch
tensor_quantale_commit_batch
tensor_quantale_seed_exploration
tensor_quantale_expand_tokens
tensor_quantale_score_tokens
tensor_quantale_select_topk_tokens
tensor_quantale_commit_exploration
tensor_quantale_frontier_step
tensor_quantale_tick
tensor_quantale_update_edge
tensor_quantale_decay
```

## Trace logging

`tlog.rs` records append-only JSONL events:

```text
Decision, Receipt, TensorEdges, AgentStep
ExplorationSeed, ExplorationExpand, ExplorationTopK, ExplorationCommit
ExplorationReceipt, BatchPlan
fusion::regions_loaded, fusion::region, fusion::kernel_synthesized  (startup)
```

## Static topology invariant checker

`crates/topology_core/src/check.rs` validates `GraphTopology` before any tensor operations.

```text
Phase 1 — identity and weight
  unique node names, unique node IDs, stable index round-trip,
  start node id=0, halt node exists+outdeg=0, no duplicate edges,
  weight domain validity, zero-confidence edge warning,
  deterministic tie-break

Phase 2 — operator binding (check_with_operators)
  operator entry exists, action/output_mode field set

Phase 3 — dominator and cycle checks
  gate dominance, receipt cutset, SCC progress/exit, no zero-cost cycle
```

Required dominator pairs:

```text
State::Validate        → dominates → Control::Commit
Control::GateReceipt   → dominates → Event::ReceiptAccepted
Event::ReceiptAccepted → dominates → Event::HashNonzero
Event::HashNonzero     → dominates → State::Validate
```

## Runtime decision invariant checker

`src/runtime_check.rs` validates each `DecisionReport` before execution:

```text
decision_is_safe(report) → bool
  Invariant 20: false when score=⊥ with blocked=0, or first_hop out of range.

check_decision_with_policy(report, node_name, policy) → Vec<RuntimeViolation>
  Invariant 18: score=⊥  ⟹  blocked=1  ∧  first_hop < 0
  Invariant 19: score≠⊥, blocked=0  ⟹  first_hop = selected_dst
  Invariant 24: Control::Block  ⟹  blocked=1  ∨  halted=1
```

## Legacy removals

Removed and forbidden from runtime reintroduction:

```text
scalar CUDA world
scalar LLM plan format
CPU routing planner
policy side-channel files
legacy assets/patterns.json      (deleted; replaced by generated patterns.source.json)
search/ingress/dsl/paging layers
PyTorch/JAX/Triton runtime
hard-coded StateNode/ControlNode/EventNode constants
NODE_COUNT / MATRIX_LEN / THREAD_COUNT in src/node.rs
separate kernel_fusion crate or addons/ directory
runtime PTX stitching or FusionPlan types
fake CUDA planned-success receipts
QuantaleAction enum / selected_action()
batch_contains_cuda_ptx / CUDA-specific batch branching
runtime fallback to legacy topology assets
runtime fallback to legacy pattern assets
```

## Benchmark baseline

Recorded on 44-node topology; recapture after the 74-node graph is exercise-tested.

```text
profile=release
iterations=10
edge_count=45
tensor_closure     avg_us=217.630
tensor_projection  avg_us=33.412
tensor_decay       avg_us=12.782
```
