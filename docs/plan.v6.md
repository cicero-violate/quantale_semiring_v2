# Plan v6: Data-Driven Config Extraction

## Objective

Extract all tuning knobs, policy tables, and structural constants from Rust and
CUDA source into versioned JSON assets under `assets/`.  The code becomes
structural; the numbers become data.

Motivation:
- Runtime behavior (decay rates, thresholds, penalty weights) should be tunable
  without a recompile.
- The same constant living in both `src/tensor.rs` and `cuda/quantale_world.cu`
  will eventually drift; generated headers eliminate that class of bug.
- Validator policy (`REQUIRED_DOMINATORS`, `RECEIPT_CUTSET`) is part of the
  domain model, not the compiler unit.

---

## Design Principles

**Ownership:** each module owns its config file.  A module's asset is loaded by
that module's `default_asset()` / `from_json_str()` methods, not by `main.rs`
threading values through.

**Orchestration layer:** `SystemConfig` (`config.rs`) is the CUDA quantale
orchestrator config.  Fields that are directly about tick-loop behavior (decay
rates, hard-reset thresholds) belong there, not in separate files.

**Unify, don't multiply files:** the 9-item naive split collapses to 5 files plus
`SystemConfig` extensions by merging items with the same owner.

---

## Work Items

### A  `SystemConfig` extension  *(items 1 + 8 merged)*

Fields added to `SystemConfig`:

| Field                 | Default | Replaces |
|-----------------------|---------|----------|
| `decay_normal`        | 0.995   | `world.decay(0.995)` literals in `main.rs` |
| `decay_blocked`       | 0.97    | `world.decay(0.97)` literal |
| `hard_reset_blocks`   | 3       | `consecutive_blocks >= 3` literal |
| `hard_reset_sleep_ms` | 200     | `Duration::from_millis(200)` literal |

`DEFAULT_BLOCK_SIZE` stays as a const; `max_ticks` and `tick_sleep_ms` were
already fields.  `ProjectionBias` is left in `main.rs` because moving it into
`config.rs` would create a circular dep (`tensor.rs` → `config.rs` →
`tensor.rs`).

`maybe_hard_reset_after_blocks` in `main.rs` gains explicit `hard_reset_blocks`
and `hard_reset_sleep` parameters to match.

No new JSON file — the defaults live in `Default::default()`.

**Risk:** low.  Pure substitution.

---

### B  `assets/topology_invariants.json`  *(item 2)*

Owner: `topology_check.rs`

The module-level statics become a deserializable struct embedded via
`include_str!`:

```json
{
  "required_dominators": [
    { "gate": "State::Validate",       "protected": "Control::Commit" },
    { "gate": "Control::GateReceipt",  "protected": "Event::ReceiptAccepted" },
    { "gate": "Event::ReceiptAccepted","protected": "Event::HashNonzero" },
    { "gate": "Event::HashNonzero",    "protected": "State::Validate" }
  ],
  "receipt_cutset": [
    "Event::ReceiptAttached",
    "Control::GateReceipt",
    "Event::ReceiptAccepted",
    "Event::HashNonzero",
    "State::Validate"
  ]
}
```

`pub fn check(topology, invariants: &TopologyInvariants)` takes the invariants as
a parameter.  Call sites pass `&TopologyInvariants::default()` which embeds the
file above.  The statics `REQUIRED_DOMINATORS` and `RECEIPT_CUTSET` are deleted.

**Risk:** low.  Only the call-site signature changes; checker logic is untouched.

---

### C  `assets/exploration.json` extensions  *(items 5 + 6 merged)*

Owner: `ExplorationConfig` in `exploration.rs`

Two new top-level keys added to the existing `exploration.json`:

```json
{
  "receipt_policy": {
    "exit_observations": { "0": 1.0, "124": -0.5 },
    "default_observation": -0.25,
    "ema_current":      0.8,
    "ema_observation":  0.2
  },
  "node_features": {
    "State::Goal":    { "novelty": 0.1, "entropy": 0.1 },
    "State::Plan":    { "novelty": 0.1, "entropy": 0.3 },
    ...
  }
}
```

`ExplorationConfig` gains `receipt_policy: ReceiptPolicy` and
`node_features: HashMap<String, NodeFeatures>`.  Both fields use
`#[serde(default)]` so old `exploration.json` files that predate these keys
continue to load without error.

`update_receipt_prior` reads from `self.config.receipt_policy` instead of the
inline `match`.  `novelty_for_node` / `entropy_for_node` become private engine
methods that look up `self.config.node_features` and fall back to the modulo
formula when a node name is absent.  This fallback is intentional — new nodes
added to `topology.json` continue to work until their entry is added to
`node_features`.

**Risk:** low-medium.  Behavior is identical to current for any node present in
`node_features`; absent nodes fall back to the modulo formula unchanged.

---

### D  `assets/learning_policy.json`  *(item 7)*

Owner: `learning.rs`

```json
{
  "learned_edge_cost_floor": 0.001,
  "confidence_clamp": [0.0, 1.0],
  "safety_clamp":     [0.0, 1.0]
}
```

`LearningPolicy` is added to `learning.rs` with `from_json_str` / `default_asset`.
`load_learned_tensor_edges` gains a `policy: &LearningPolicy` parameter.
`main.rs` loads the policy before loading learned edges.
`LEARNED_EDGE_COST_FLOOR` const is deleted.

**Risk:** very low.

---

## Deferred Items

The following require CUDA build-system changes and are deferred to a separate
patch after the Rust-only items above are stable:

| Item | File(s) | Blocker |
|------|---------|---------|
| Shared tensor constants (node count, COST_INFINITY) | `tensor.rs`, `quantale_world.cu`, `build.rs` | requires generated `.cuh` header + `build.rs` emit |
| CUDA scoring/penalty constants | `quantale_world.cu` | `__constant__` struct layout, device-side alignment |
| JIT synth policy | `jit_kernel_fusion/synth.rs` | allowlist validation adds non-trivial logic |

---

## Final Asset Layout

```
assets/
  topology.json               existing
  operators.json              existing
  patterns.json               existing
  exploration.json            extended (items C)

  topology_invariants.json    new (item B)
  learning_policy.json        new (item D)
```

---

## Invariant

All extracted values must match the current hardcoded values exactly at the time
of extraction.  No behavior change is introduced in this plan.
