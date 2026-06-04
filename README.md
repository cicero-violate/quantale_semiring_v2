# quantale_semiring_v2

CUDA-first tensor quantale orchestrator with a data-driven topology DSL, Concurrent Kleene Algebra pattern layer, and JIT kernel fusion pipeline.

The runtime compiles declarative JSON assets into quantale-valued tensor-edge graphs and kernel fusion regions, runs tensor closure and projection on CUDA, dispatches selected operators (process, jit_cuda, cuda_ptx), and feeds execution receipts back into the tensor as ordinary edge deltas.

## Runtime invariant

```text
CKA constrains possible thought.
Tensor quantale scores possible thought.
Exploration searches competing thought.
CUDA commits selected safe thought.
Receipts validate actual thought.
```

## Canonical pipeline

```text
assets/topology.source.json              (authoritative DSL source)
  ──▶  cargo run -- topology build-overlay
         ├─ topology.generated.json      (flat quantale transitions; runtime input)
         ├─ operators.generated.json     (compiled operator registry)
         ├─ patterns.source.json         (CKA tensor-weight patterns)
         └─ topology.fusion.json         (maximal GPU-safe kernel regions)

assets/topology.generated.json
  → NodeRegistry + TensorEdge[]
assets/patterns.source.json
  → CompiledCkaPattern { edges } → TensorEdge[]
assets/topology.fusion.json
  → FusionDispatch → JitChain → synthesize_kernel → JitCache (NVRTC)

TensorEdge[]
  → TensorQuantaleWorld → tensor_quantale_closure
  → Exploration scheduler  (seed / expand / score / topk / commit)
  → Single-step fallback   (frontier_step)
  → ProcessReceipt
  → tensor_quantale_update_edge → T := T ∨ ΔT
```

`main.rs` uses the generated artifact path only. The scalar CUDA matrix runtime, scalar LLM plan format, CPU routing planner, and DSL/search/ingress demo layer must not be reintroduced.

## Tensor model

```text
T ∈ R^(3 × N × N)
```

Layers:

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

Projection score:

```text
score = α·confidence - β·cost + γ·safety
```

## Data assets

```text
assets/topology.source.json      DSL source: 74 nodes, 52 slots, 11 resources, 11 programs
assets/topology.generated.json   Generated: flat quantale-valued tensor graph (runtime input)
assets/operators.generated.json  Generated: compiled operator registry (runtime input)
assets/patterns.source.json      Generated: CKA tensor-weight patterns (replaces patterns.json)
assets/topology.fusion.json      Generated: fusible kernel regions (runtime input)
assets/operators.json            Build input: operator contracts with jit_body and effects
assets/exploration.json          Token-value exploration policy, scoring and anti-repeat weights
state/learned_edges.jsonl        Learned sparse tensor-edge checkpoints
state/quantale.tlog              Append-only JSONL execution trace
```

Quantale-valued transition fields:

```json
{
  "from": "Analysis::Return1",
  "to": "Analysis::Volatility",
  "confidence": 0.9,
  "cost": 1.0,
  "safety": 0.9,
  "policy_effect": "market_analysis_cycle"
}
```

## Node registry

The node universe is declared in `assets/topology.source.json` and compiled into `topology.generated.json`. There are no hard-coded node constants in Rust.

`topology.rs::NodeRegistry` is the single source of truth:

```rust
registry.id_of("State::Execute")   // Option<usize>
registry.name_of(9)                 // Option<&str>
registry.action_of(17)              // Option<&str>  — from JSON "action" field
registry.len()                      // node count (74)
registry.matrix_len()               // len * len
```

To add a node: edit `assets/topology.source.json`, then run `cargo run -- topology build-overlay`.

## CUDA kernel split

```text
cuda/quantale_world.cu
  → compiled at runtime via NVRTC (cudarc::nvrtc::compile_ptx)
  → kernels: closure, projection, exploration, frontier, tick, decay

JIT fusion kernels (src/jit_kernel_fusion/)
  → synthesized from topology.fusion.json + operators.json at startup
  → compiled via NVRTC through JitCache::get_or_compile
  → Analysis::Return1 + Volatility + SignalScore → single fused kernel
```

## JIT fusion dispatch

```text
FusionDispatch::load("assets/topology.fusion.json", &operator_registry)
  → detect_jit_chains(region.nodes, registry)   validates jit_cuda linkage
  → JitChain { operators, inputs, outputs, internals }
  → synthesize_all()                            dry-run at startup (no device)
  → JitCache::get_or_compile(device, chain)     NVRTC → PTX (cfg(feature="cuda"))

Runtime lookup:
  config.fusion_dispatch.is_fusion_entry(node)   O(1)
  config.fusion_dispatch.get_by_entry(node)       → &FusionEntry
```

## Operator dispatch

`egress.rs` routes by `executable` in `operators.generated.json`:

```text
jit_cuda      → JIT fusion chain (NVRTC via JitCache)
cuda_ptx      → precompiled PTX module
anything else → Command::new(binary)
```

## Concurrent Kleene Algebra

Source topology programs and runtime patterns use the same algebra:

```text
zero   blocked / impossible (bottom)
one    identity / skip
node   atomic endpoint
seq    a ; b
choice a + b  (quantale join; no cross-edges)
star   bounded Kleene repetition
par    a || b  (effect independence required)
```

Compilation paths:

```text
topology.source.json programs
  → crates/topology_core/src/programs.rs
  → flat transitions
  → topology.generated.json

assets/patterns.source.json (generated)
  → src/pattern.rs
  → CompiledCkaPattern { edges }
  → TensorEdge deltas
```

## Effect safety

```text
safe_parallel(a,b) =
  writes(a) ∩ writes(b) = ∅
  ∧ writes(a) ∩ reads(b) = ∅
  ∧ reads(a) ∩ writes(b) = ∅
  ∧ locks(a) ∩ locks(b) = ∅
```

Effects are declared in `topology.source.json` node entries and validated at build-overlay time.

## Main Rust surfaces

```text
src/main.rs                runtime loop: exploration-first → frontier fallback
src/cli.rs                 CLI command parsing (topology build-overlay, --check-topology)
src/runtime_dispatch.rs    execute_active_node_blocking, hot/fusion dispatch helpers
src/runtime_epoch.rs       RuntimeEpoch struct, build_runtime_epoch, asset fingerprint
src/runtime_reset.rs       maybe_hard_reset_after_blocks
src/tensor.rs              CUDA tensor world, TensorEdge API, projection/commit
src/topology.rs            topology.generated.json parser, NodeRegistry
src/fusion_dispatch.rs     FusionDispatch: load topology.fusion.json → JitChain
src/jit_kernel_fusion/     chain detection, kernel synthesis, NVRTC cache, slot buffers
src/pattern.rs             CKA pattern compiler
src/egress.rs              data-driven executor: process / jit_cuda / cuda_ptx
src/config.rs              SystemConfig: operator registry + FusionDispatch + runtime config
src/learning.rs            learned_edges.jsonl checkpoint loader
src/plan.rs                tensor LLM plan compiler
src/tlog.rs                append-only JSONL trace log
src/exploration.rs         ExplorationEngine, token management, anti-repeat policy
src/runtime_check.rs       runtime decision invariant checker
crates/topology_core/      DSL compiler: programs, validators, fusion partitioner
  src/programs.rs          CKA compiler + slot/resource/quantale validators
  src/fusion.rs            partition_fusible_regions → topology.fusion.json
  src/overlay.rs           build_overlay_assets pipeline
```

## Execution loop

1. `cargo run -- topology build-overlay` generates all runtime artifacts from `topology.source.json`.
2. At startup: learned edge checkpoints + topology + CKA pattern edges are embedded. `FusionDispatch` loads `topology.fusion.json` and builds `JitChain`s; fusion kernels are synthesized (and compiled via NVRTC if `--features cuda`).
3. CUDA closes the tensor.
4. Exploration seeds strategies from `assets/exploration.json`, expands CUDA tokens, scores and selects top-K.
5. If a best exploration candidate is available, it is committed and dispatched.
6. If no exploration candidate is ready, CUDA runs a normal single frontier step.
7. `runtime_check::decision_is_safe()` guards every execution step.
8. Process results become `ProcessReceipt` evidence; tensor edge feedback and exploration receipt priors are updated.
9. Asset fingerprint changes (from `assets/reload_policy.json`) trigger epoch reload.

## Validation

```bash
cargo run -- topology build-overlay
cargo run -- --check-topology
cargo fmt --check
cargo check
cargo test
cargo check --no-default-features
cargo test --no-default-features
```

Current validated test counts:

```text
cargo test                         135 passed (10 suites)
cargo test --no-default-features   135 passed (10 suites)
```

## Non-goals

Do not add or reintroduce:

```text
scalar CUDA world or scalar LLM plan format
CPU routing planner
assets/patterns.json as a runtime source (deleted; generated by build-overlay)
policy side-channel files outside declared runtime assets
PyTorch/JAX/Triton alternate runtime
hard-coded node ID constants (StateNode, ControlNode, EventNode, NODE_COUNT)
separate kernel_fusion crate or addons/ directory
runtime PTX stitching or FusionPlan types
fake CUDA planned-success receipts
QuantaleAction enum / selected_action()
TypedIR lowering scaffold (ir.rs — deleted)
CPU batch scheduler (batch.rs — deleted)
bench binaries in src/bin/ (deleted)
```

## Proof boundary

Lean/cLean artifacts live under `lean/`. They name the proof boundary for tensor closure, projection, frontier, tick, and batch projection/commit behavior. Specification artifacts unless a local Lean toolchain is installed.
