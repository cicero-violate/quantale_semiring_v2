# Topology DSL Migration Plan

## Status

All phases complete. The pipeline runs end-to-end:

```text
topology.source.json
  ──▶ topology build-overlay
        ├── topology.generated.json   (flat quantale-valued tensor graph)
        ├── operators.generated.json  (compiled operator registry)
        ├── patterns.source.json      (CKA patterns, replaces patterns.json)
        └── topology.fusion.json      (maximal GPU-safe kernel regions)
  ──▶ runtime startup
        ├── FusionDispatch::load(topology.fusion.json, registry)
        │     └── detect_jit_chains → JitChain per region
        ├── JitCache::get_or_compile (NVRTC/PTX, #[cfg(feature="cuda")])
        └── tick loop: quantale projection → dispatch (fused or barrier)
```

---

## Goal

Move orchestration into the topology itself so the source topology becomes an algebraic program, while runtime execution is handled by a small generic interpreter and pure kernels/operators.

Original split:

```text
topology.json   = flat node/transition graph   (build input; still exists)
patterns.json   = algebraic CKA expressions    (deleted; replaced by patterns.source.json)
operators.json  = node → executable metadata   (build input; still exists)
```

Achieved split:

```text
topology.source.json     = algebraic topology DSL (source of truth)
topology.generated.json  = compiled flat tensor graph
operators.generated.json = compiled runtime operator registry
patterns.source.json     = generated CKA patterns (replaces patterns.json)
topology.fusion.json     = generated fusible kernel regions
```

The fixed point:

```text
Topology = orchestration program
Generated topology = tensor-executable graph
Runtime = generic VM/interpreter
Nodes = pure kernels or explicit boundary adapters
Receipts = algebraic feedback into tensor weights
```

---

## Scope Boundary: What the DSL Replaces

The topology DSL replaces orchestration, not every imperative implementation detail.

```text
Python orchestration -> topology DSL
Python control/I/O   -> Rust boundary operators
Python numeric loops -> CUDA kernels
```

The correct split:

```text
Topology DSL = what may happen, algebraic branch structure, slot/resource dependencies
Rust VM      = generic interpreter, scheduler, quantale projection, receipts
Rust boundary= files, JSONL, HTTP, LLM, cargo/git/patch, governed mutations
CUDA kernels = pure tensor/vector/market/risk compute
```

The DSL does not open files, call HTTP APIs, spawn subprocesses, or mutate source files. Those effects are declared as boundary nodes.

---

## I/O Boundary Model

All I/O is explicit, declared, governed, and receipted.

```text
IO := file read/write | JSONL append | HTTP | LLM | subprocess | cargo | git | patch | asset mutation
```

I/O node shape:

```json
{
  "name": "State::MarketFeed",
  "kind": "boundary_node",
  "runtime": {
    "backend": "python",
    "script": "crates/operators_lib/market_feed.py"
  },
  "reads": ["market.config"],
  "writes": ["market.feed", "state/market_feed.jsonl"],
  "locks": [],
  "governance": {
    "network": true,
    "allowed_hosts": ["coingecko", "stooq"]
  }
}
```

The invariant:

```text
forall IO: declared(IO) and governed(IO) and receipted(IO)
```

No I/O is hidden in CUDA kernels, untracked Python scripts, or undeclared Rust helper code.

---

## Core Principle

Branching belongs in the DSL as algebraic forms, not imperative `if/else` host logic.

```text
seq(A, B, C)       = composition
choice(A, B)       = quantale join / alternative path
par(A, B)          = concurrent independent composition
star(A, n)         = bounded iteration
zero               = algebraic impossibility / bottom
one                = identity / skip
```

The runtime compiles these into tensor edges with quantale values. Unavailable paths are not skipped by host conditionals — they become bottom through the quantale algebra itself.

---

## Quantale Algebra

The tensor T ∈ ℝ^{3×N×N} carries three quantale layers:

```text
T[0] = confidence / correctness   (max-times semiring)
T[1] = cost / time                (min-plus semiring)
T[2] = safety / security          (max-min semiring)
```

### Edge value

Each transition carries a quantale element:

```text
q_ij = (c_ij, t_ij, s_ij)

c_ij ∈ [0,1]       confidence
t_ij ∈ [0,∞]       cost
s_ij ∈ [0,1]       safety
```

### Composition (path i → j → k)

```text
q_ik = q_ij ⊗ q_jk

c_ik = c_ij · c_jk       (multiply)
t_ik = t_ij + t_jk       (accumulate)
s_ik = min(s_ij, s_jk)   (bottleneck)
```

### Join (competing paths)

```text
q = q_a ∨ q_b

c = max(c_a, c_b)         (higher confidence wins)
t = min(t_a, t_b)         (lower cost wins)
s = max(s_a, s_b)         (higher safety wins)
```

### Bottom (algebraic impossibility)

```text
⊥ = (0, ∞, 0)   — additive identity of the semiring
```

### Unit (identity transition)

```text
e = (1, 0, 1)   — identity for composition
```

---

## Source Topology Shape

See `assets/topology.source.json` for the full current declaration. Key structure:

```json
{
  "matrix_name": "quantale_semiring_v2",
  "version": 1,
  "quantale": { "layers": [...] },
  "slots":     { "slot.name": { "type": "f32[]|json|jsonl|bytes", "kind": "tensor|state|config|..." } },
  "resources": { "resource.name": { "kind": "gpu|lock", "capacity": 1 } },
  "nodes":     [ { "name": "...", "kind": "kernel|host_node|boundary_node|policy_node|event_node", ... } ],
  "programs":  [ { "name": "...", "expr": { "seq|choice|par|star": [...] }, "confidence": 0.9, ... } ]
}
```

Current counts: **74 nodes**, **52 slots**, **11 resources**, **11 programs**.

---

## Generated Runtime Shape

The compiler emits quantale-valued transitions:

```json
{
  "matrix_name": "quantale_semiring_v2",
  "nodes": [...],
  "transitions": [
    {
      "from": "Analysis::Volatility",
      "to": "Analysis::SignalScore",
      "confidence": 0.9,
      "cost": 1.0,
      "safety": 0.9,
      "policy_effect": "market_analysis_cycle"
    }
  ],
  "parallel_groups": [
    ["State::Map", "State::Parse"]
  ]
}
```

Program-compiled transitions carry the explicit quantale triple. Hand-authored transitions in `topology.json` carry `default_weight` for backward compat; the runtime always prefers the explicit triple.

The compiler also emits `topology.fusion.json`. See the Fusion Architecture section.

---

## Fusion Architecture

The topology graph runs entirely under JIT orchestration. Only maximal GPU-resident subgraphs fuse into single CUDA kernels:

```text
Graph JIT:      G_t → G_{t+1}              (dynamically select next executable region)
Kernel fusion:  K_1;K_2;K_3 ⟹ K_{123}    (coalesce GPU ops into one CUDA kernel)
```

### Fusion Condition

A subgraph F ⊆ G is fusible iff:

```text
Fusable(F) = B(F) ∧ K(F) ∧ S(F) ∧ A(F) ∧ R(F)

B(F) = same backend       (all nodes: runtime.backend = "cuda")
K(F) = kernel-compatible  (all nodes: kind = "kernel")
S(F) = static shapes      (tensor slot layouts known at compile time)
A(F) = associative algebra (seq or par within region; no choice boundary inside)
R(F) = no interior barrier (no boundary_node, governance gate, or lock conflict)
```

Edge legality within a region:

```text
producer-consumer:  writes(A_i) ∩ reads(A_{i+1}) ≠ ∅
independent par:    writes(A_i) ∩ writes(A_j) = ∅
```

### Kernel Rule

```text
Fuse algebra, not control flow.

CKA gives structure.
Quantale gives score/composition.
Effects give legality.
JIT gives specialization.
CUDA gives execution.
```

### Non-Fusible Barriers

```text
boundary_node                 (external I/O, LLM, network, git/cargo)
host_node with I/O effects    (purity = deterministic_io)
governance gate               (policy check required before proceeding)
resource lock conflict        (exclusive locks across candidate nodes)
shape unknown                 (dynamic slot types at compile time)
```

### Fusible Regions (current topology)

```text
Analysis::Return1 → Analysis::Volatility → Analysis::SignalScore
```

Emitted in `topology.fusion.json`:

```json
{
  "regions": [
    {
      "region": "Analysis::Return1__Analysis::Volatility__Analysis::SignalScore",
      "backend": "cuda_jit",
      "fusion": "linear_chain",
      "nodes": ["Analysis::Return1", "Analysis::Volatility", "Analysis::SignalScore"],
      "reads": ["market.open", "market.price"],
      "writes": ["analysis.signal_score"],
      "locks": [],
      "quantale": {
        "compose": ["times", "plus", "min"],
        "join":    ["max",   "min",  "max"]
      }
    }
  ]
}
```

`analysis.return` and `analysis.volatility` are internal (produced and consumed within the region). The fused kernel signature: `(market.price[], market.open[]) → analysis.signal_score[]`.

### Compiler Pipeline

```text
topology.source.json
  → validate: unique node names, known backends
  → validate: quantale layer declarations (join/compose/bottom/unit laws)
  → validate: slot/resource references per node
  → validate: boundary nodes have governance
  → validate: kernel nodes read/write only tensor slots
  → compile: algebraic programs → flat transitions
  → validate: par branch effect independence
  → validate: program weight domains per quantale layer
  → partition: effect-aware fusible region detector (Fusable(F) condition)
  → emit: topology.generated.json
  → emit: topology.fusion.json
  → emit: operators.generated.json
  → emit: patterns.source.json
```

### Dispatch Pipeline

```text
FusionDispatch::load(topology.fusion.json, operator_registry)
  → detect_jit_chains(region.nodes, registry)   [jit_kernel_fusion::chain]
  → synthesize_kernel(&chain, registry)          [jit_kernel_fusion::synth]  ← CUDA C
  → JitCache::get_or_compile(device, &chain)     [jit_kernel_fusion::cache]  ← PTX via NVRTC
  → runtime: is_fusion_entry(node) → dispatch fused kernel
             else → dispatch boundary/host operator
```

---

## Migration Phases

### Phase 1 — Consolidate Algebra Into Topology ✅

- [x] Add `assets/topology.source.json`.
- [x] Move `patterns.json` expressions into `topology.source.json` programs.
- [x] Add a compiler step that produces `topology.generated.json` from source programs.
- [x] Preserve all current flat transitions.

```text
cargo run -- topology build-overlay   ✅
cargo run -- --check-topology         ✅
cargo test                            ✅
```

---

### Phase 2 — Add Slots and Resources ✅

- [x] Add `slots` and `resources` to `topology.source.json`.
- [x] Declare effects (reads/writes/locks) on nodes.
- [x] Validate every node read/write references a declared slot.
- [x] Validate every lock references a declared resource.

```text
no undeclared read slots    ✅
no undeclared write slots   ✅
no undeclared locks         ✅
```

---

### Phase 3 — Algebraic Branching ✅

- [x] `choice` = quantale join (no cross-edges).
- [x] `seq` = sequential composition.
- [x] `par` = concurrent composition + effect independence check.
- [x] `star` = bounded unroll.
- [x] `zero`/`blocked`/`impossible` = bottom.
- [x] `one`/`identity`/`skip` = identity.
- [x] `"kind": "cka_pattern"` programs are runtime tensor-weight patterns only.
- [x] Emit `patterns.source.json` from topology programs.

```text
Compiler rules:
  compile(node)      → endpoint(node)
  compile(seq xs)    → compose adjacent endpoints
  compile(choice xs) → union endpoints, no cross-edges
  compile(par xs)    → independent branches + parallel group metadata + effect check
  compile(star x n)  → bounded unroll of x
  compile(zero)      → no edge / bottom
  compile(one)       → identity / no-op endpoint
```

---

### Phase 4 — Quantale Source Semantics ✅

**Correction:** The original Phase 4 proposed boolean masks (`m=false ⟹ ⊥`). This is a boolean control guard pretending to be algebra. The correct approach declares the quantale structure in the source topology and compiles every edge into a proper quantale element.

- [x] Add `quantale.layers` with declared `join`, `compose`, `bottom`, `unit` per layer.
- [x] Validate each layer's algebraic laws.
- [x] Validate every program weight against layer domains.
- [x] Compile `weight` fields into quantale-valued transitions.
- [x] Remove `default_weight` from generated transitions (use explicit triple).

```text
every source edge compiles to a valid (c, t, s) quantale element  ✅
bottom = (0, ∞, 0) — no program edge compiles to this             ✅
unit = (1, 0, 1) — identity transitions valid                      ✅
```

---

### Phase 5 — Pure Node Boundary ✅

Five node kinds:

```text
kernel_node    = pure CUDA compute over tensor slots
host_node      = deterministic CPU transform over slots
boundary_node  = external I/O, LLM, filesystem, network, git/cargo
policy_node    = governance or routing decision (no slot output)
event_node     = receipt/state transition marker (no I/O)
```

- [x] All 74 nodes declared: **6 kernel, 16 host_node, 15 boundary_node, 8 policy_node, 29 event_node**.
- [x] 7 new slots added: `introspect.report`, `topology.plan`, `operator.plan`, `pattern.plan`, `select.result`, `mutation.review`, `rollback.record`.
- [x] `validate_boundary_governance` — every `boundary_node` must declare a `governance` object.
- [x] `validate_kernel_slot_purity` — every `kernel` reads/writes must be `kind: "tensor"` slots only.

```text
no node has implicit side effects                           ✅
all side effects declared in topology/operator metadata     ✅
```

---

### Phase 6 — Compiler and Validators ✅

- [x] `validate_unique_source_node_names` — no duplicate names in `nodes[]`.
- [x] `validate_known_backends` — `runtime.backend` must be a recognised value.
- [x] `partition_fusible_regions` in `crates/topology_core/src/fusion.rs` — maximal linear-chain subgraphs satisfying `Fusable(F)`.
- [x] `build_overlay_assets` emits `topology.fusion.json`.

Emitted region:

```text
Analysis::Return1 → Analysis::Volatility → Analysis::SignalScore
  backend: cuda_jit  |  fusion: linear_chain
  reads:   [market.open, market.price]
  writes:  [analysis.signal_score]
  internals (produced+consumed within): analysis.return, analysis.volatility
```

```text
invalid topology.source.json never emits generated artifacts  ✅
topology.fusion.json contains only regions where Fusable(F)   ✅
```

---

### Phase 7 — Runtime Simplification and JIT Execution ✅

- [x] `src/fusion_dispatch.rs` — `FusionDispatch::load(path, registry)` builds `JitChain`s from fusion regions via `detect_jit_chains` (existing `jit_kernel_fusion::chain`).
- [x] `FusionDispatch` indexed by entry node (`is_fusion_entry`) and by member (`get_by_member`) for O(1) dispatch lookup.
- [x] `FusionDispatch` wired into `SystemConfig` — loaded and reloaded alongside the operator registry.
- [x] `synthesize_all(registry)` produces CUDA C source for all regions without a CUDA device (dry-run / startup verification).
- [x] `#[cfg(feature = "cuda")] JitCache::get_or_compile` compiles PTX via NVRTC on demand.
- [x] Fusion dispatch logged at every epoch build: `fusion::regions_loaded`, `fusion::region` (per region with chain_len + estimated_savings), `fusion::kernel_synthesized`.
- [x] `GraphTopology::default_asset()` — `topology.generated.json` first, bundled constant fallback only. `topology.json` runtime fallback removed.
- [x] `load_default_patterns()` — `patterns.source.json` first, bundled constant fallback only. `patterns.json` runtime fallback removed.
- [x] `DEFAULT_PATTERNS_JSON` embed switched from `patterns.json` to `patterns.source.json`.
- [x] `assets/patterns.json` deleted — replaced by `assets/patterns.source.json`.
- [x] `assets/reload_policy.json` updated: `patterns.source.json` and `topology.fusion.json` added; `patterns.json` removed.

```text
adding a new workflow: topology.source.json changes only, no Rust edits required  ✅
Analysis::Return1→Volatility→SignalScore: JitChain built, NVRTC deferred to JitCache  ✅
boundary nodes are always barriers, never fused                                     ✅
117 tests pass                                                                      ✅
```

---

## Non-Goals

- Do not put external I/O inside CUDA kernels.
- Do not make the topology DSL a general-purpose programming language.
- Do not hide file/network/subprocess effects inside ungoverned operator implementations.
- Do not make topology execute itself without a runtime VM.
- Do not encode policy as arbitrary host-language scripts.
- Do not introduce boolean `if/else` guards into the DSL — use quantale bottom for impossible paths.
- Do not remove Rust; shrink it into a generic topology VM.
- Do not fuse the whole graph into one kernel — only maximal GPU-safe subgraphs fuse.
- Do not fuse control flow — fuse algebra (associative quantale composition over pure tensor nodes only).

---

## Target Invariants

```text
1.  Source topology is the only orchestration source of truth.          ✅
2.  Flat transitions are generated, not hand-authored.                  ✅
3.  Branching is algebraic: choice/par/star/seq/zero/one.               ✅
4.  Every edge carries a quantale element (confidence, cost, safety).   ✅
5.  Internal nodes are pure transforms over slots.                      ✅
6.  Boundary nodes are explicit and governed.                           ✅
7.  Runtime is a generic interpreter, not a workflow-specific orchestrator. ✅
8.  All generated topology passes existing topology invariants.         ✅
9.  The whole graph is a JIT-compiled execution fabric: G_t → G_{t+1}. ✅
10. Only maximal GPU-safe quantale regions (Fusable(F)) become fused CUDA kernels. ✅
```

---

## Final Fixed Point ✅

```text
Source topology    = CKA / quantale DSL program          (topology.source.json)
Generated topology = quantale-valued tensor edge graph   (topology.generated.json)
Generated fusion   = maximal GPU-safe kernel regions     (topology.fusion.json)
Generated operators= executable backend metadata         (operators.generated.json)
Generated patterns = CKA tensor-weight patterns          (patterns.source.json)
Runtime            = generic topology VM + JIT dispatcher (src/fusion_dispatch.rs)
Fused compute      = NVRTC/PTX via JitCache              (#[cfg(feature="cuda")])
Boundary           = governed adapters, always barriers  (never fused)
Receipts           = quantale feedback into T[3,N,N]
```
