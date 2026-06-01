# Plan: Data-Driven Dynamic Kernel Fusion

## Goal

Begin using dynamic GPU kernel fusion without adding CPU routing policy or Rust-side graph shortcuts.

The runtime contract stays:

- CPU launches kernels and external operators.
- GPU tensor stores path weights and learned edge state.
- JSON assets define static topology, operators, effects, and fusion patterns.
- JSONL state stores evidence/checkpoints, not alternate policy.

## Current Baseline

- `assets/topology.json` defines the canonical graph.
- `assets/operators.json` defines external operators and CUDA PTX operator metadata.
- `assets/patterns.json` defines CKA/static scheduling patterns.
- External `tensor_plan` output is filtered to static topology edges before GPU embedding.
- `state/learned_edges.jsonl` is loaded as sparse learned weights only for topology edges.
- Runtime has demonstrated:

```text
Control::Commit
  -> Control::GateMemory
  -> State::Memory
  -> Event::MemoryWritten
  -> Control::GateLearn
  -> State::Learn
  -> Event::LearnUpdated
  -> Control::Halt
```

## Implementation Steps

1. Add a fusion asset file.

Create `assets/fusion_patterns.json` with declarative records only:

```json
{
  "patterns": [
    {
      "name": "math.add_then_scale",
      "inputs": ["math.a", "math.b", "math.scale"],
      "outputs": ["math.out"],
      "operators": [
        "Execution::VectorAdd",
        "Execution::VectorScale"
      ],
      "fused_operator": "Execution::FusedVectorAddScale",
      "constraints": {
        "same_shape": true,
        "dtype": "f32",
        "max_elements": 1048576
      }
    }
  ]
}
```

Do not encode routing decisions in this file. It should describe legal fusion opportunities only.

2. Add simple test math kernels.

Add or extend CUDA kernels under `cuda/`:

- `vector_add_kernel`
- `vector_scale_kernel`
- `fused_vector_add_scale_kernel`

These are test math operators, not domain policy. They provide a small measurable target for dynamic fusion.

3. Register math operators in JSON.

Add operator entries to `assets/operators.json`:

- `Execution::VectorAdd`
- `Execution::VectorScale`
- `Execution::FusedVectorAddScale`

Each entry should use existing `cuda_ptx` metadata shape:

- `module`
- `module_name`
- `kernel`
- `plane`
- `category`
- `scheduler_contract`

Reads/writes/locks must remain declarative effect metadata.

4. Compile fusion candidates from assets.

Add a Rust loader/compiler for `assets/fusion_patterns.json`.

Allowed responsibilities:

- Parse JSON.
- Validate referenced operators exist in `assets/operators.json`.
- Validate fused operator exists.
- Validate declared effects are compatible.
- Emit a fusion candidate descriptor for the scheduler.

Forbidden responsibilities:

- Choosing paths outside the GPU tensor.
- Creating new topology edges.
- Reading JSONL state as policy.
- Special-casing operator names in Rust.

5. Add a GPU-native fusion selection signal.

Represent fusion availability as tensor-side weights, not CPU branch policy.

Minimum acceptable first pass:

- CPU detects legal fusion candidates from JSON.
- CPU launches a small CUDA kernel that scores/marks candidates using operator/effect metadata encoded in buffers.
- Resulting candidate scores are embedded as tensor weights for existing topology/operator nodes only.

The CPU may launch the kernel and dispatch the selected operator. It must not own graph policy.

6. Prefer fused operator execution when tensor-selected.

When the tensor frontier selects a fused operator node, the existing executor launches the fused CUDA operator declared in `assets/operators.json`.

Do not add a Rust `if VectorAdd + VectorScale then FusedVectorAddScale` path. Fusion must come from the asset compiler plus GPU tensor score.

7. Add verification.

Add tests that prove:

- `assets/fusion_patterns.json` parses.
- Unknown operator references fail validation.
- Fused math kernel output matches unfused math kernel output.
- Runtime can select the fused operator through existing tensor/operator machinery.
- No fusion pattern creates non-topology policy edges.

8. Add docs.

Update `README.md` and `ARCHITECTURE.md` with:

- Fusion assets are static declarative legality metadata.
- Dynamic fusion scores live in GPU tensors.
- JSONL records only evidence/checkpoints.
- Rust launches kernels and validates assets; it does not route graph policy.

## Acceptance Criteria

- `cargo check` passes.
- `cargo test --no-default-features` passes.
- A GPU-enabled run shows a fused math operator selected/executed.
- Learned edge checkpoints still contain only `assets/topology.json` edges.
- Grep finds no reintroduced `rule_delta`, `ExecutionReceipt`, CPU routing planner, or side-channel policy file.

## Non-Goals

- No Triton/PyTorch/JAX alternate engine.
- No runtime PTX string generation.
- No CPU graph traversal planner.
- No JSONL alternate topology or policy.
- No hard-coded Rust fusion shortcuts for specific operator names.
