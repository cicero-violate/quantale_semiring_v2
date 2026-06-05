# Goal

Data defines execution; CPU only launches kernels, never decides orchestration. GPU runs the show.

Build a CUDA-resident tensor quantale orchestrator where workflow routing is algebraic, data-driven, and receipt-grounded. Orchestration is declared in a quantale DSL, compiled into a tensor graph and kernel fusion regions, and executed by a generic runtime VM.

## Canonical model

```text
T ∈ R^(3 × N × N)
```

Layers:

```text
confidence/correctness: max-times
compute/time cost:      min-plus
security/safety:        max-min
```

## Structural layer

```text
CKA = { 0, 1, +, ;, *, || }
```

CKA patterns describe executable structure. They compile into `TensorEdge` deltas that seed the tensor world at epoch start. They do not replace the tensor runtime.

## Runtime invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Non-negotiables

- Tensor state remains GPU-resident.
- Tensor edges carry confidence, cost, and safety directly.
- Source topology (`topology.source.json`) is the only orchestration source of truth.
- Generated artifacts are consumed by the runtime; hand-authored JSON is build input only.
- CKA and exploration remain data-driven through JSON assets.
- `par` requires effect independence.
- Runtime feedback updates tensor cells directly.
- Receipts remain the canonical truth gate.
- Boundary nodes are explicit, governed, and always barriers — never fused.
- No scalar sidecar metadata model.
- No CPU routing planner.
- No hidden imperative graph traversal.
- Par group selection happens on the GPU when all members are GPU-eligible; CPU does not iterate groups to decide.

## Current milestone

Implemented:

```text
Tensor engine (74-node topology)
Tensor topology compilation
Tensor rule deltas
Tensor frontier step
Tensor tick
Tensor runtime loop
Full CKA pattern compiler (edges embedded at epoch start)
Effect-gated par validation
CUDA exploration seed/expand/score/top-k/commit kernels
Exploration-first scheduler integration
  → exploration candidate → execute_active_node_blocking → receipt → tensor update
  → no candidate → frontier_step → same dispatch path
Receipt-prior feedback into exploration
Static topology invariant checker — phases 1–3 (topology_check.rs)
  identity, weight domain, operator binding, gate dominance,
  receipt cutset, SCC progress, zero-cost cycle detection
Semiring law unit tests (tests/semiring_laws.rs)
Runtime frontier assertions (tensor.rs, main.rs)
Runtime decision invariant checker (runtime_check.rs)
  decision_is_safe() guard (inv 20), check_decision() diagnostics (inv 18/19/24)
  base_tensor CPU snapshot for hard-reset groundwork (inv 23)
  action/output_mode operator field check (inv 25)
Single canonical dispatch path through execute_active_node_blocking
  FusionDispatch checked first, hot region second, process fallback last
Runtime split into sub-modules: cli, runtime_dispatch, runtime_epoch, runtime_parallel, runtime_reset
GPU-native parallel tier (Phase 8):
  tensor_quantale_par_group_step kernel: GPU selects, validates, and commits
    the first ready CKA par group in one round trip
  ParGroupGpuData: static par group tuple table, built at epoch start; eligibility is computed on-device from per-member is_gpu_dispatchable flags
  Eligibility = all group operators are jit_cuda / fusion-entry / hot-region
  CPU dispatches committed operators concurrently via dispatch_gpu_parallel_group
  CPU does not iterate groups or validate effects at runtime

Topology DSL — Phases 1–7 (plan.topology.md):
  Phase 1: topology.source.json with programs compiler
  Phase 2: slots, resources, declared node effects
  Phase 3: algebraic branching (seq/choice/par/star/zero/one), patterns.source.json
  Phase 4: quantale layer declarations, source edge weight validation
  Phase 5: 74 nodes declared (kernel/host/boundary/policy/event),
           validate_boundary_governance, validate_kernel_slot_purity
  Phase 6: validate_unique_source_node_names, validate_known_backends,
           partition_fusible_regions, topology.fusion.json emitted
  Phase 7: FusionDispatch (src/fusion_dispatch.rs), JitChain construction,
           synthesize_all (dry-run CUDA C), JitCache (NVRTC, cfg(feature="cuda")),
           patterns.source.json as runtime source, patterns.json deleted
```

Validated test count: **138 passed** (10 suites, `--no-default-features`).

Fusion region active for:

```text
Analysis::Return1 → Analysis::Volatility → Analysis::SignalScore
  (linear_chain, cuda_jit, reads=[market.open,market.price], writes=[analysis.signal_score])
```
