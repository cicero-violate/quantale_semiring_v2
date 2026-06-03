# Declarative Runtime Plan

## Goal

Make the runtime scalable by moving agent behavior out of `src` and into data
assets. Rust should become the stable interpreter/enforcer. Topology, operators,
payload contracts, side effects, exploration policy, and mutation policy should
be declared in assets.

## Meaning of Declarative

Imperative behavior is special-case code:

```text
if node == Control::WriteOperator:
  require filename/source
  block direct exploration
  only run after State::OperatorPlan
```

Declarative behavior is data:

```json
{
  "node": "Control::WriteOperator",
  "requires_payload": ["filename", "source"],
  "allowed_after": ["State::OperatorPlan"],
  "side_effect": "repo_write",
  "mutation_mode": "staged"
}
```

The runtime then applies one generic rule to every node:

```text
load contract -> validate payload -> validate predecessor -> validate side effect -> execute or block
```

## Current State

Already data-driven:

- Topology nodes and transitions: `assets/topology.json`, `assets/topology.generated.json`
- Operator commands: `assets/operators.json`, `assets/operators.generated.json`
- LLM templates: `assets/call_llm_templates.json`
- Learning policy: `assets/learning_policy.json`
- CKA patterns: `assets/patterns.json`
- Tensor node count: derived from topology at build time
- Process execution: generic `UniversalExecutor`

Not yet declarative enough:

- Payload requirements are implicit in operator scripts.
- Side-effect permissions are implicit in operator behavior.
- Exploration can jump directly to payload-dependent or mutating nodes.
- File mutations are applied immediately, not staged for review.
- Runtime policy is still concentrated in `src/main.rs`.
- CUDA tensor semantics are fixed and require rebuild after topology size changes.

## Implementation Status

Done in the first implementation slice:

- Added `assets/node_contracts.json`.
- Added a Rust contract loader/enforcer in `src/contracts.rs`.
- Runtime now reloads node contracts with topology assets.
- Exploration is blocked from directly executing declared unsafe mutating nodes.
- `Control::WriteOperator` now requires `filename` and `source` payload keys and
  must follow `State::OperatorPlan`.
- Contract failures are logged separately from operator execution failures.
- Added `assets/side_effect_policy.json`.
- Added `state/mutation_queue.jsonl` staging support through
  `crates/operators_lib/mutation_policy.py`.
- `Control::WriteOperator`, `Control::TopologyMutate`, and
  `Control::PatternMutate` now stage validated mutations by default instead of
  writing repo assets.
- Added `crates/operators_lib/apply_mutations.py` as the explicit apply path for
  pending queue records.
- Added `quantale_semiring_v2 mutations list|apply [mutation_id]` CLI wiring.
- Added `assets/runtime_policy.json` for runtime tick, decay, and hard-reset
  thresholds.
- Added `assets/mutation_review_policy.json`.
- Mutation queue review now supports preview diffs and selective rejection by
  id through `crates/operators_lib/apply_mutations.py`.
- Added read-only topology wiring for `State::MutationReview` and
  `Event::MutationReviewed`.
- Added `assets/governance_policy.json` and governance checks for queued
  topology, pattern, and operator mutations before apply.

Still pending:

- No implementation items remain in this plan. Future work should add new
  policy assets or operators without changing Rust unless the interpreter itself
  needs a new generic capability.

## Target Assets

### 1. Node Contracts

Add an asset such as:

```text
assets/node_contracts.json
```

Example:

```json
{
  "contracts": [
    {
      "node": "Control::WriteOperator",
      "requires_payload": ["filename", "source"],
      "allowed_after": ["State::OperatorPlan"],
      "side_effects": ["repo_write", "operator_registry_write"],
      "mutation_mode": "staged",
      "exploration": {
        "allow_direct": false
      }
    }
  ]
}
```

Purpose:

- Prevent side-effecting nodes from running without valid input.
- Make node preconditions explicit.
- Let new operators declare requirements without Rust edits.

### 2. Side-Effect Policy

Add:

```text
assets/side_effect_policy.json
```

Example:

```json
{
  "default": "deny",
  "effects": {
    "read_state": "allow",
    "write_state": "allow",
    "repo_write": "stage",
    "operator_registry_write": "stage",
    "topology_write": "stage"
  }
}
```

Purpose:

- Separate execution from file mutation.
- Make permanent repo changes pass through a review/apply step.
- Allow safe continuous runs without accidental asset churn.

### 3. Mutation Queue

Mutating operators should write proposals to:

```text
state/mutation_queue.jsonl
```

instead of directly editing assets.

Example record:

```json
{
  "kind": "topology_patch",
  "source_node": "Control::TopologyMutate",
  "ops": [
    {
      "op": "create_edge",
      "from": "Control::WriteOperator",
      "to": "Control::Repair"
    }
  ],
  "reason": "repair fallback for failed write operator"
}
```

Then a separate apply command can review and apply queued mutations.

### 4. Exploration Policy

Add:

```text
assets/exploration_policy.json
```

Example:

```json
{
  "default_allow_direct": true,
  "deny_direct": [
    "Control::WriteOperator",
    "Control::TopologyMutate",
    "Control::PatternMutate"
  ],
  "require_payload_for": [
    "Control::WriteOperator"
  ],
  "side_effecting_nodes": "contract_only"
}
```

Purpose:

- Stop exploration from selecting mutating nodes as standalone actions.
- Keep exploration useful for safe nodes.
- Avoid LLM repair loops caused by invalid direct execution.

### 5. Payload Schemas

Use simple JSON schemas or a minimal local schema format.

Example:

```json
{
  "schemas": {
    "operator_write_payload": {
      "required": {
        "filename": "string",
        "source": "string"
      },
      "optional": {
        "node_name": "string",
        "operator_contract_ops": "array"
      }
    }
  }
}
```

Contracts can reference schemas:

```json
{
  "node": "Control::WriteOperator",
  "payload_schema": "operator_write_payload"
}
```

## Runtime Enforcement

Before executing any node, the runtime should:

1. Load the node contract.
2. Validate required payload/schema.
3. Validate predecessor constraints.
4. Validate side-effect policy.
5. If mutation mode is `staged`, write a proposal instead of applying directly.
6. Execute only if all checks pass.
7. Log blocked contract failures distinctly from execution failures.

This creates a clean distinction:

- Contract failure: node should not have been selected.
- Execution failure: node had valid input but failed while running.
- Mutation proposal: side effect requested, awaiting apply.

## Suggested Phases

### Phase 1 — Observation Only

- Add contracts as data.
- Load and log contract violations.
- Do not block execution yet.

Expected result:

- See which nodes are being executed without valid payloads.
- Confirm `Control::WriteOperator` direct exploration is a contract violation.

### Phase 2 — Block Unsafe Direct Execution

- Enforce `allow_direct=false`.
- Enforce required payload keys.
- Treat contract blocks as routing feedback, not operator failures.

Expected result:

- Exploration no longer creates false failures for side-effecting nodes.
- LLM topology planner stops adding repair edges for induced failures.

### Phase 3 — Stage Mutations

- Change topology/pattern/operator mutators to write proposals.
- Add an explicit apply path.
- Keep normal runtime runs from changing repo assets automatically.

Expected result:

- Continuous runs become safe to inspect.
- Repo diffs reflect reviewed changes only.

### Phase 4 — Asset-Driven Governance

- Move retry/reset thresholds and side-effect permissions into assets.
- Make mutation approval policy data-driven.
- Add tests for contract enforcement.

Expected result:

- New operators can be added by assets and scripts without Rust changes.
- Rust becomes mostly stable infrastructure.

## Success Criteria

- Running the agent does not create unreviewed repo diffs by default.
- `Control::WriteOperator` cannot execute unless `filename` and `source` exist.
- `Control::TopologyMutate` and `Control::PatternMutate` stage changes unless policy says apply.
- Topology changes do not require Rust source edits.
- New node behavior can be added by updating assets and operator scripts.
- Contract violations are logged separately from operator execution failures.

## Current Priority

Do not add more topology first.

The next scalable step is:

```text
node contracts + side-effect policy + staged mutation queue
```

That is the layer that lets the system keep evolving without constantly editing
`src`.
