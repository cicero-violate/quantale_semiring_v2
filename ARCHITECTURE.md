# Architecture

`quantale_semiring_v2` is a CUDA-first tensor quantale orchestrator with a data-driven CKA structural compiler, and a topology DSL that compiles algebraic orchestration programs into quantale-valued tensor graphs and JIT kernel fusion regions.

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
TENSOR_NODE_COUNT derived from generated topology at build time
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

`SystemConfig`, `ExplorationEngine`, and `egress.rs` all derive dimensions from the registry. No component holds a hard-coded `NODE_COUNT` constant.

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
compile_source_programs              CKA → flat transitions
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
CKA pattern compilation (build time; edges embedded at epoch start)
FusionDispatch: fusion region loading and JitChain construction
JitCache: NVRTC/PTX compilation and kernel caching (#[cfg(feature="cuda")])
ParGroupGpuData: par group table [(node_id, region_id, is_gpu_dispatchable, dispatch_kind) tuples] uploaded at epoch start
Static topology invariant checking
Runtime decision invariant checking
Base tensor CPU snapshot for hard reset
Per-member is_gpu_dispatchable and region_id table construction (kernel validates eligibility on-device)
Host-side operator dispatch after GPU commit (jit_cuda and fusion only)
Device-ring receipt routing for H_f par members; CPU hot-region fallback still uses gpu_dispatch_region + drain_device_receipts
Edge-delta upload
Compact report decoding
Append-only transaction logging
Learned edge buffer and flush to state/learned_edges.jsonl
```

Rust does not own a CPU planner, a CPU group-selection loop, or a live CPU mirror of the tensor.

## GPU Orchestration Tiers

Three distinct levels of GPU involvement, not to be conflated:

```text
GPU-selected parallel dispatch tier (superseded)
  The GPU selects a par group, commits consumed/active state, and dispatches
  H_f / abstract-device members in-kernel.  Process/IO members are excluded from
  GPU par selection; the CPU still owns fallback routing, failure recovery, and
  the runtime step loop.

  Metrics: G_s=1  G_c=1  E_g=1  D_h=2/3  R_d=1  R_k=1  H_o=1

GPU-native par dispatch (superseded)
  All eligible par-group members — including fusion-entry members — are
  dispatched in-kernel. The CPU dispatches no operators in the par hot path.
  The runtime step loop is still CPU-owned.

  Adds: D_h=1  (no host operator calls in the par path)

GPU-native orchestration with external command service (current)
  tensor_quantale_orchestrate_step owns all control-flow decisions: SEQ, PAR,
  CHOICE, and bounded STAR progression are selected and committed on-device
  without CPU involvement.  The ControlEdge + EffectTable device tables drive
  deterministic selection; per-edge star counters are GPU-resident and
  snapshotted for replay.  Process/IO work is explicit: the GPU emits
  DeviceCommand entries; the CPU services them and returns DeviceReceiptExt
  entries.  No silent CPU fallback for control flow.

  Implemented by: plan.gpu.native.seq.par.choice.star.md (2026-06-05)
  Metrics: S_g=1  P_g=1  E_g=1  C_g=1  R_g=1  F_g=1  H_g=0
           D_g=0 for process/IO (explicit command/receipt protocol)

Fully GPU-native orchestration (complete — all nine phases done 2026-06-05)
  Same as above plus: standalone control_flow_advance side-path retired;
  all runtime test coverage migrated to orchestrate_step observations;
  legacy CPU orchestration feature-gated rather than default.

  Metrics: S_g=1  P_g=1  E_g=1  D_g=1  C_g=1  R_g=1  F_g=1  H_g=0
```

The correct current label is **GPU-native orchestration with external command
service** (`S_g=1 P_g=1`). The CPU is a supervisor and IO service, not a
control-flow decision maker.

## CUDA kernel split

Three distinct compilation and loading paths — all dispatch through cudarc:

```text
cuda/quantale_world.cu
  Compiled: NVRTC at runtime (cudarc::nvrtc::compile_ptx)
  Tensor core:
    tensor_quantale_reset, embed_edges, closure, project,
    project_batch, commit_batch, frontier_step, tick,
    update_edge, decay, seed_exploration, expand_tokens,
    score_tokens, select_topk_tokens, commit_exploration,
    par_group_step
  Orchestration scheduler:
    orchestration_state_init, orchestration_state_snapshot,
    tensor_quantale_orchestrate_step, star_counters_init,
    control_flow_advance, check_effects_independent,
    failure_policy_init, failure_policy_classify_and_emit,
    failure_policy_set_rollback_marker, failure_policy_apply_rollback,
    learned_delta_init, learned_delta_fold_receipt,
    learned_delta_apply, receipt_prior_snapshot,
    orch_event_trace_push, orch_event_trace_drain,
    orch_check_no_duplicate_receipts, orch_check_frontier_valid,
    orch_check_no_command_without_receipt,
    orch_replay_snapshot, orch_replay_restore

JIT fusion kernels (src/jit_kernel_fusion/)
  Compiled: NVRTC at runtime via JitCache::get_or_compile
  Source:   synthesize_kernel(&JitChain, &registry) → CUDA C
  Regions:  loaded from assets/topology.fusion.json via FusionDispatch
```

`quantale_world.cu` provides the deterministic tensor execution core and the
GPU-native orchestration scheduler. `tensor_quantale_orchestrate_step` is the
primary per-step entry point: it consults the ControlEdge + EffectTable device
tables to execute SEQ, PAR, CHOICE, and bounded STAR without CPU involvement,
then falls back to singleton tensor scoring if no control edge is active.
`par_group_step` remains for the legacy par-dispatch path. Fusion kernels use
NVRTC to specialize operator chains at startup.

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
  → CompiledCkaPattern { edges }
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
  → GPU-native parallel tier
      → par_group_step(ParGroupGpuData)
          tensor_quantale_par_group_step  (select + validate + commit)
      → dispatch_gpu_parallel_group       (concurrent operator launch)
  → single-step fallback
      → tensor_quantale_frontier_step
  → ProcessReceipt
  → tensor_quantale_update_edge
  → T := T ∨ ΔT
```

## Dispatch loop

```text
each tick:
  1. CUDA close tensor
  2. ExplorationEngine: expand + score + topk
  3. if best candidate:
       commit_exploration_candidate
       execute_active_node_blocking     ← fusion-first, then hot, then process
       update receipt priors + lattice
       continue
  4. par_group_step(par_group_data, bias)   ← one kernel: select + validate + commit
       → iterates GPU-resident par group table [(node_id, region_id) pairs]
       → projects each member toward target node on device
       → first all-ready, eligible group committed atomically on device
       → CPU reads (group_idx, decisions, per-member region_ids)
       dispatch_gpu_parallel_group           ← concurrent: jit_cuda / fusion / hot
       for hot-region members: gpu_dispatch_region → device receipt ring
       for other members:      queue_lattice_update → CPU drain
       drain_device_receipts + drain_lattice_queue
       update receipt priors + tlog
       continue  (if group was selected)
  5. frontier_step
       execute_active_node_blocking     ← same single dispatch path
       update receipt priors + lattice
```

`execute_active_node_blocking` checks `FusionDispatch.get_by_entry` first, then `HotRegionRegistry`, then falls back to `UniversalExecutor::execute_abstract_node_blocking`. All paths emit a `ProcessReceipt`.

The GPU-native parallel tier (step 4) is active when `par_group_data` is present (CUDA device available and at least one eligible par group declared in the topology). The packed table stores `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuples per member. The kernel computes eligibility on-device: a group is selected only when all members have `is_gpu_dispatchable == 1`. The kernel emits per-member dispatch descriptors in `ParGroupStepOutput`, including `region_id`, source/destination, and dispatch kind, so the CPU uses kernel-native routing info. H_f hot-region par members write receipts directly to the device ring; successful batched fusion receipts are host-detected, pushed into the same ring, and drained on-device. Abstract-node and failed fallback members use the CPU lattice queue.

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
Par    — independent branches + effect check
```

Source topology programs compile the same algebra via `crates/topology_core/src/programs.rs`. Node names are validated against the registry at compile time. Pattern edges are embedded into the tensor world at epoch start; generated `parallel_groups` are resolved into `TopologyRuntime.parallel_groups` and uploaded into `ParGroupGpuData` for the GPU-selected par tier.

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
tensor_quantale_project_batch       (used internally by par_group_step logic)
tensor_quantale_commit_batch        (used internally by par_group_step logic)
tensor_quantale_par_group_step      (GPU-native: select + validate + commit par group)
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
Decision, TensorEdges, AgentStep
ExplorationSeed, ExplorationTopK, ExplorationCommit, ExplorationReceipt
parallel::gpu_group_committed, parallel::operator_receipt  (par tier)
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
CPU group-selection loop for par dispatch (deleted; GPU kernel selects)
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
runtime fallback to legacy topology or pattern assets
CPU batch scheduler (batch.rs)
TypedIR lowering scaffold (ir.rs)
bench binaries (src/bin/bench_*)
```
