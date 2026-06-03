# Topology DSL Migration Plan

## Goal

Move orchestration into the topology itself so the source topology becomes an algebraic program, while runtime execution is handled by a small generic interpreter and pure kernels/operators.

Current split:

```text
topology.json   = flat node/transition graph
patterns.json   = algebraic CKA expressions
operators.json  = node -> executable metadata
```

Target split:

```text
topology.source.json     = algebraic topology DSL
operators.source.json    = optional operator declarations / runtime backends
topology.generated.json  = compiled flat tensor graph
operators.generated.json = compiled runtime operator registry
```

The fixed point is:

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

The correct split is:

```text
Topology DSL = what may happen, algebraic branch structure, slot/resource dependencies
Rust VM      = generic interpreter, scheduler, quantale projection, receipts
Rust boundary= files, JSONL, HTTP, LLM, cargo/git/patch, governed mutations
CUDA kernels = pure tensor/vector/market/risk compute
```

The DSL should not attempt to open files, call HTTP APIs, spawn subprocesses, or mutate source files. It should declare those effects as boundary nodes.

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
    "backend": "rust_host",
    "handler": "market_feed"
  },
  "reads": ["assets/market_feed.json"],
  "writes": ["market.feed", "state/market_feed.jsonl"],
  "locks": [],
  "governance": {
    "network": true,
    "allowed_hosts": ["coingecko", "stooq"]
  }
}
```

The invariant is:

```text
forall IO: declared(IO) and governed(IO) and receipted(IO)
```

No I/O should be hidden in CUDA kernels, untracked Python scripts, or undeclared Rust helper code.

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

The runtime compiles these into tensor edges with quantale values. Unavailable paths are not skipped by host conditionals — they become bottom through the quantale algebra itself. The algebraic structure determines routing; no boolean guards are needed.

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

Path confidence multiplies, cost accumulates, safety bottlenecks.

### Join (competing paths)

```text
q = q_a ∨ q_b

c = max(c_a, c_b)         (higher confidence wins)
t = min(t_a, t_b)         (lower cost wins)
s = max(s_a, s_b)         (higher safety wins)
```

### Bottom (algebraic impossibility)

```text
⊥ = (0, ∞, 0)

no confidence
infinite cost
no safety
```

Bottom is not a boolean guard. It is the additive identity of the semiring — a path that can never be preferred over any other.

### Unit (identity transition)

```text
e = (1, 0, 1)

perfect confidence
zero cost
full safety
```

---

## Desired Source Topology Shape

```json
{
  "matrix_name": "quantale_semiring_v2",
  "version": 1,

  "quantale": {
    "layers": [
      {
        "name": "confidence",
        "join": "max",
        "compose": "times",
        "bottom": 0.0,
        "unit": 1.0
      },
      {
        "name": "cost",
        "join": "min",
        "compose": "plus",
        "bottom": "inf",
        "unit": 0.0
      },
      {
        "name": "safety",
        "join": "max",
        "compose": "min",
        "bottom": 0.0,
        "unit": 1.0
      }
    ]
  },

  "slots": {
    "task.context": { "type": "json", "kind": "state" },
    "market.feed": { "type": "json", "kind": "state" },
    "analysis.return": { "type": "f32[]", "kind": "tensor" },
    "analysis.volatility": { "type": "f32[]", "kind": "tensor" },
    "analysis.signal_score": { "type": "f32[]", "kind": "tensor" }
  },

  "resources": {
    "cuda": { "kind": "gpu", "capacity": 1 },
    "memory": { "kind": "lock", "capacity": 1 },
    "workspace": { "kind": "lock", "capacity": 1 },
    "paper_broker": { "kind": "lock", "capacity": 1 }
  },

  "nodes": [
    {
      "name": "Analysis::SignalScore",
      "kind": "kernel",
      "runtime": {
        "backend": "cuda",
        "module": "analysis",
        "kernel": "analysis_signal_score"
      },
      "reads": ["analysis.return", "analysis.volatility"],
      "writes": ["analysis.signal_score"],
      "locks": []
    }
  ],

  "programs": [
    {
      "name": "market_analysis_cycle",
      "entry": "State::Input",
      "expr": {
        "seq": [
          "State::MarketFeed",
          "Analysis::Return1",
          "Analysis::Volatility",
          "Analysis::SignalScore",
          {
            "choice": [
              "State::TradePlan",
              "Control::Block"
            ]
          }
        ]
      },
      "weight": {
        "confidence": 0.95,
        "cost": 1.0,
        "safety": 0.98
      }
    }
  ]
}
```

---

## Generated Runtime Shape

The compiler emits quantale-valued transitions:

```json
{
  "matrix_name": "quantale_semiring_v2",
  "nodes": [],
  "transitions": [
    {
      "from": "Analysis::Volatility",
      "to": "Analysis::SignalScore",
      "confidence": 0.97,
      "cost": 0.2,
      "safety": 0.99,
      "default_weight": 0.97,
      "policy_effect": "AnalysisSignalScore"
    }
  ],
  "pages": [],
  "parallel_groups": [
    ["Analysis::Return1", "Analysis::Volatility"]
  ]
}
```

The `confidence`, `cost`, and `safety` fields on each transition are the quantale element for that edge. The existing `default_weight` field is `confidence` by convention. Long-term, `default_weight` should be removed in favour of the explicit quantale triple.

The Rust runtime should consume generated artifacts only.

---

## Migration Phases

### Phase 1 — Consolidate Algebra Into Topology ✅

- [x] Add `assets/topology.source.json`.
- [x] Move existing `assets/patterns.json` expressions into `topology.source.json.programs`.
- [x] Keep `assets/patterns.json` temporarily as a compatibility artifact.
- [x] Add a compiler step that can produce the current `topology.generated.json` from source programs.
- [x] Preserve all current flat transitions during migration.

Success condition:

```text
cargo run -- topology build-overlay   ✅
cargo run -- --check-topology         ✅
cargo test                            ✅
```

---

### Phase 2 — Add Slots and Resources ✅

- [x] Add top-level `slots` to `topology.source.json`.
- [x] Add top-level `resources` to `topology.source.json`.
- [x] Move operator `effects.reads`, `effects.writes`, and `effects.locks` into node declarations (19 nodes declared).
- [x] Validate that every node read/write references a declared slot.
- [x] Validate that every lock references a declared resource.

Success condition:

```text
no undeclared read slots    ✅
no undeclared write slots   ✅
no undeclared locks         ✅
```

---

### Phase 3 — Algebraic Branching ✅

- [x] Treat `choice` as quantale join, not imperative branching.
- [x] Treat `seq` as composition.
- [x] Treat `par` as parallel composition requiring effect independence.
- [x] Treat `star` as bounded unroll.
- [x] Treat `zero`, `blocked`, and `impossible` as bottom.
- [x] Treat `one`, `identity`, and `skip` as identity.
- [x] Programs with `"kind": "cka_pattern"` are runtime tensor-weight patterns only (not compiled to flat transitions).

Compiler rules implemented:

```text
compile(node)      -> endpoint(node)
compile(seq xs)    -> compose adjacent endpoints
compile(choice xs) -> union endpoints / competing alternatives
compile(par xs)    -> independent branches + parallel group metadata + effect check
compile(star x n)  -> bounded unroll of x
compile(zero)      -> no edge / bottom
compile(one)       -> identity / no-op endpoint
```

Success condition:

```text
patterns.json generated from topology.source.json    ✅  (assets/patterns.source.json)
```

---

### Phase 4 — Quantale Source Semantics

Replace the stub `quantale` field with a full quantale layer declaration that drives compilation of source edges into T[3,N,N].

**Correction note:** The original Phase 4 proposed boolean masks (`m ∈ {0,1}`, `m=false ⟹ ⊥`). This is wrong — it is a boolean control guard pretending to be algebra. The correct approach is to declare the quantale structure directly in the source topology and compile every edge into a proper quantale element.

- [ ] Add `quantale.layers` to `topology.source.json` with declared `join`, `compose`, `bottom`, `unit` per layer.
- [ ] Validate that every source edge's `weight` (or `value`) has a component for each declared layer.
- [ ] Validate `bottom` and `unit` values satisfy each layer's algebra (unit is the identity for compose, bottom is the absorbing element for compose).
- [ ] Compile source program `weight` fields directly into quantale-valued transitions using layer laws.
- [ ] Make CKA program compiler emit quantale-valued edge deltas (not just `default_weight`).
- [ ] Validate that `choice` in programs emits join (not cross-edges), `seq` emits composition, `zero` emits bottom, `one` emits unit.
- [ ] Remove `default_weight` from generated transitions in favour of explicit `confidence`/`cost`/`safety` triple (the existing T[3,N,N] structure already carries this).

Success condition:

```text
every source edge compiles to a valid quantale element (c ∈ [0,1], t ∈ [0,∞], s ∈ [0,1])
bottom = (0, ∞, 0) — no program edge should compile to this
unit = (1, 0, 1) — identity transitions are valid
quantale.layers declaration validates composition and join laws
```

---

### Phase 5 — Pure Node Boundary

Classify every node:

```text
kernel_node    = pure compute over slots
host_node      = deterministic CPU transform over slots
boundary_node  = external I/O, LLM, filesystem, network, git/cargo
policy_node    = governance or contract producer
event_node     = receipt/state transition marker
```

Node target shape:

```json
{
  "name": "State::Memory",
  "kind": "host_node",
  "purity": "deterministic_io",
  "reads": ["task.context"],
  "writes": ["memory.store"],
  "locks": ["memory"],
  "runtime": {
    "backend": "resident_worker",
    "worker": "operator-memory",
    "protocol": "jsonl"
  }
}
```

Long-term target:

```text
internal nodes -> pure Rust/CUDA functions
boundary nodes -> explicit governed adapters
```

Remaining work:

- [ ] Declare the remaining ~55 nodes in `topology.source.json` (event nodes, control nodes, state nodes not yet declared).
- [ ] Validate that every boundary node has a `governance` field.
- [ ] Validate that every kernel node has no I/O effects (reads/writes are tensor slots only).

Success condition:

```text
no node has implicit side effects
all side effects are declared in topology/operator metadata
```

---

### Phase 6 — Compiler and Validators

Add a complete topology compiler pipeline:

```text
topology.source.json
  -> parse
  -> validate quantale layer declarations
  -> validate slots/resources/nodes/programs
  -> compile algebraic programs
  -> validate effect independence for par
  -> validate quantale values per edge
  -> emit topology.generated.json
  -> emit operators.generated.json
  -> emit patterns.source.json
  -> emit parallel_groups artifact
```

Required validators not yet wired:

- [ ] Unique node names (across source topology nodes).
- [ ] Unique slot names.
- [ ] Unique resource names.
- [ ] Every transition endpoint exists in the node set.
- [ ] Every node runtime backend is a known backend.
- [ ] Every boundary node has governance coverage.
- [ ] Every generated topology passes existing topology invariants.
- [ ] Every quantale edge value is within domain per layer.

Already wired (Phases 2–3):

- [x] Every declared node read/write slot exists.
- [x] Every declared node lock resource exists.
- [x] Every `par` branch is effect-independent.

Success condition:

```text
invalid topology.source.json never emits generated artifacts
```

---

### Phase 7 — Runtime Simplification

After generated topology and quantale compilation are reliable:

- [ ] Make Rust runtime consume generated topology only.
- [ ] Remove hardcoded watched asset path list and use `assets/reload_policy.json`.
- [ ] Remove direct `patterns.json` loading from runtime — load compiled programs from generated topology instead.
- [ ] Make runtime execute only these generic steps:

```text
load generated artifacts
project tensor frontier using quantale composition
execute selected node/backend
collect receipt
update quantale weights via receipt feedback
append log
reload on asset fingerprint change
```

Success condition:

```text
adding a new workflow requires topology.source.json/operator metadata changes, not Rust runtime changes
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

---

## Target Invariants

```text
1. Source topology is the only orchestration source of truth.
2. Flat transitions are generated, not hand-authored long-term.
3. Branching is algebraic: choice/par/star/seq/zero/one.
4. Every edge carries a quantale element (confidence, cost, safety).
5. Internal nodes are pure transforms over slots.
6. Boundary nodes are explicit and governed.
7. Runtime is a generic interpreter, not a workflow-specific orchestrator.
8. All generated topology passes existing topology invariants.
```

---

## Immediate Implementation Checklist

Done:

- [x] Create `assets/topology.source.json` from current `topology.json` + `patterns.json`.
- [x] Add `programs` support to `topology_core`.
- [x] Move `CkaExpr` parsing/compilation into topology compiler (`programs.rs`).
- [x] Emit `patterns.source.json` compatibility output during `topology build-overlay`.
- [x] Generate flat `transitions` from source programs.
- [x] Generate `parallel_groups` from `par` expressions.
- [x] Add slot/resource declarations.
- [x] Validate operator effects against slots/resources.
- [x] Classify 19 key nodes as `kernel`, `host_node`, or `boundary_node`.
- [x] Add governance fields to boundary nodes.
- [x] Add par effect-independence checking.
- [x] Remove `masks` stub — masks are not the correct abstraction.

Next:

- [ ] Add `quantale.layers` declaration to `topology.source.json`.
- [ ] Validate every source edge `weight` has components matching all declared layers.
- [ ] Compile source `weight` fields into proper quantale triples using declared layer laws.
- [ ] Validate `bottom` and `unit` per layer.
- [ ] Remove `default_weight` from generated transitions (use explicit triple).
- [ ] Classify remaining ~55 nodes in `topology.source.json`.
- [ ] Add remaining Phase 6 validators (unique names, known backends, boundary governance).
- [ ] Convert runtime to consume generated topology + load patterns from source.

---

## Final Fixed Point

```text
Source topology    = CKA / quantale DSL program
Generated topology = quantale-valued tensor edge graph
Generated operators= executable backend metadata
Runtime            = generic topology VM
Compute            = pure Rust/CUDA kernels
Boundary           = governed adapters
Receipts           = quantale feedback into T[3,N,N]
```
