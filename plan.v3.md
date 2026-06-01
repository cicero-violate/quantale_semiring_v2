# Plan: JIT Tensor Kernel Fusion

## Goal

Replace static pre-compiled kernel fusion with a fully data-driven JIT system where:

- Operator math bodies are declared as CUDA C expressions in `assets/operators.json`
- The runtime detects fusable chains dynamically from the `effects` dependency graph
- Fused kernels are synthesized at runtime and compiled via NVRTC (already present)
- Intermediate slot values stay device-resident — no CPU round-trips inside a chain
- No Rust code names, hardcodes, or special-cases any operator or fusion rule

The runtime contract stays:

- CPU launches kernels and detects chains from JSON metadata
- GPU tensor stores path weights and learned edge state
- JSON assets define operators, effects, topology, and JIT kernel bodies
- JSONL state stores evidence/checkpoints only

---

## Current Baseline

Completed in the previous phase (Data-Driven Dynamic Kernel Fusion):

- `assets/topology.json` has `Execution::VectorAdd`, `Execution::VectorScale`,
  `Execution::FusedVectorAddScale` as topology nodes (IDs 44–46)
- `assets/operators.json` has all three math operators with `cuda_ptx` metadata
- `assets/fusion_patterns.json` declares the static fusion opportunity
- `src/fusion.rs` compiles static patterns against the operator registry and topology
- `cuda/trading_execution_kernels.cu` contains hand-written math kernels
- `cuda/quantale_world.cu` contains `fusion_score_embed` which scores pre-declared candidates
- `src/tensor.rs` calls `embed_fusion_scores` with `FusionCandidate` descriptors
- `state/learned_edges.jsonl` can trigger fused execution for live testing

Known limitations of this baseline:

- Each kernel still round-trips through CPU (`htod → kernel → dtoh → ProcessReceipt`)
- Fused kernels are manually written and must be hand-maintained
- Fusion candidates are statically declared, not derived from the effects graph
- No device-resident slot buffers between operator calls
- `build.rs` requires an `nvcc` installation to compile math kernels AOT

---

## What Gets Deleted

| Artifact | Why |
|---|---|
| `cuda/trading_execution_kernels.cu` math kernels | Generated at runtime, not hand-written |
| `build.rs` nvcc compile step | No more AOT compilation for math operators |
| `assets/fusion_patterns.json` | Static declarations replaced by dynamic chain detection |
| `src/fusion.rs` | Static asset compiler replaced by runtime graph analysis |
| `KERNEL_NAMES` list in `egress.rs` | Static registration replaced by JIT cache |
| `PTX_BYTES = include_bytes!` in `egress.rs` | Replaced by NVRTC-compiled JIT kernels |
| `execute_cuda_ptx_blocking` CUDA branch | Replaced by JIT chain executor |
| `FUSION_SCORE_KERNEL` constant and `embed_fusion_scores` in `tensor.rs` | Replaced by chain-aware scoring kernel |
| `FusionCandidate`, `FusionPatterns`, `compile_fusion_patterns` from `main.rs` | Replaced by `JitChain` descriptors |

---

## Implementation Steps

### 1. Add `jit_body` to operators.json math entries

Extend each `jit_cuda` operator entry with a positional kernel body expression.
Input slots map in `effects.reads` order to `in0`, `in1`, ... Output maps to `out`.

```json
{
  "node_name": "Execution::VectorAdd",
  "executable": "jit_cuda",
  "jit_body": "out[i] = in0[i] + in1[i];",
  "effects": {
    "reads":  ["math.a", "math.b"],
    "writes": ["math.add_out"],
    "locks":  []
  }
}
```

Remove the `input_mapping` fields that reference pre-compiled PTX kernel names.
The `effects` slots are the only data-flow declaration needed.

No Rust changes in this step — pure asset edit.

### 2. Add slot buffer registry

Create `src/slot_buffers.rs`:

- `SlotBuffers`: `HashMap<String, CudaSlice<f32>>` keyed by slot name
- Written when a `jit_cuda` operator completes and its `effects.writes` slot stays on device
- Read instead of `htod_copy` when the next operator's `effects.reads` names a resident slot
- Evicted (copied to CPU, removed) only when a non-`jit_cuda` consumer needs the value
- Cleared at each full loop iteration boundary

This is the mechanism that eliminates CPU round-trips inside chains.
No operator names appear in this module — only slot name strings from the JSON.

### 3. Add JIT chain detector

Create `src/jit_chain.rs` to replace `src/fusion.rs`:

- Accepts a sequence of operator names (the active frontier decisions for the tick)
- Reads each operator's `effects.reads` and `effects.writes` from the registry
- Groups consecutive `jit_cuda` operators where `A.writes ∩ B.reads ≠ ∅`
- Emits `Vec<JitChain>` — ordered groups of fusable operators

```rust
pub struct JitChain {
    pub operators: Vec<String>,   // operator names in execution order
    pub inputs:    Vec<String>,   // chain-level input slots (not produced internally)
    pub outputs:   Vec<String>,   // chain-level output slots (not consumed internally)
    pub internals: Vec<String>,   // slots that stay device-resident as registers
}
```

No operator names are hardcoded in this module.
The chain boundary rule is entirely data-driven from the effects graph.

### 4. Add JIT kernel synthesizer

Create `src/jit_synth.rs`:

- Accepts a `&JitChain` and the operator registry
- Reads each operator's `jit_body` and `effects` fields
- Performs slot name → positional variable substitution across the chain
- Eliminates intermediate slots (they become register variables in the loop body)
- Emits a complete `__global__` CUDA C kernel string

Example synthesis for `VectorAdd → VectorScale`:

```cuda
extern "C" __global__ void jit_fused(
    const float* in0, const float* in1, const float* in2,
    float* out0, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float reg_math_add_out = in0[i] + in1[i];  // VectorAdd body, slot stays in register
    out0[i] = reg_math_add_out * in2[i];        // VectorScale body, writes output slot
}
```

The kernel string is fully data-driven: if `jit_body` changes in `operators.json`,
the synthesized kernel changes automatically.

### 5. Add JIT kernel cache

Add `src/jit_cache.rs`:

- Key: `Vec<String>` of operator names (the chain signature)
- Value: compiled `CudaModule` + `CudaFunction` (via NVRTC `compile_ptx`, already present)
- First call for a chain: synthesize → compile → store
- Subsequent calls: retrieve compiled function directly

NVRTC compilation is already used for `quantale_world.cu` in `tensor.rs`.
The cache uses the same `compile_ptx` call path — no new CUDA dependency.

Compilation happens once per unique operator sequence per process lifetime.
Hot chains pay zero recompilation overhead after the first call.

### 6. Add JIT executor path in `egress.rs`

Replace the `execute_cuda_ptx_blocking` CUDA branch with a JIT chain executor:

Allowed responsibilities:

- Check `"executable": "jit_cuda"` on the operator entry
- Look up device-resident input slots from `SlotBuffers`; `htod_copy` only for slots not yet on device
- Look up or compile the chain kernel via `JitCache`
- Launch the fused kernel
- Write output slots back into `SlotBuffers`
- `dtoh_sync_copy` only for chain-terminal outputs (first slot with a non-`jit_cuda` downstream consumer)

Forbidden responsibilities:

- Naming specific operators or slots in Rust
- Choosing which slots to fuse (that is the chain detector's job)
- Owning any kernel source strings (those come from `operators.json`)

### 7. Update GPU fusion scoring kernel

Replace `fusion_score_embed` in `quantale_world.cu` with a kernel that scores
dynamically detected `JitChain` descriptors:

- Input: chain metadata buffer (chain length, total input slot count, estimated savings)
- No operator names in the kernel — only numeric chain properties
- Scores based on: memory round-trips eliminated, kernel launches saved
- Writes score to the fused operator node's tensor entry

Update `tensor.rs` to call this kernel with `JitChain` metadata instead of `FusionCandidate`.

### 8. Delete static artifacts

In order:

1. Remove math kernel bodies from `cuda/trading_execution_kernels.cu`
   (keep the file if other non-JIT kernels remain; delete if empty)
2. Remove the nvcc compile step from `build.rs`
3. Delete `assets/fusion_patterns.json`
4. Delete `src/fusion.rs`
5. Remove `KERNEL_NAMES` static list and `PTX_BYTES` from `egress.rs`
6. Remove `FusionPatterns`, `compile_fusion_patterns`, `FusionCandidate` imports from `main.rs`
7. Remove `FUSION_SCORE_KERNEL` and `embed_fusion_scores` from `tensor.rs`
8. Replace fusion loading block in `main.rs` with JIT chain detection call

### 9. Verification

Tests that must pass:

- JIT-synthesized kernel output for `VectorAdd → VectorScale` matches reference values
- Same chain on second call hits the cache (compile_ptx called exactly once)
- Slot buffers remain device-resident across operators in a chain
- Non-`jit_cuda` operator forces dtoh of upstream slot
- `cargo check` passes
- `cargo test --no-default-features` passes
- Grep finds no operator or slot names hardcoded in Rust source

Runtime verification:

- A GPU-enabled run shows `jit_fused` kernel selected and executed
- No CPU round-trip appears in the log for intra-chain slot transfers
- `state/learned_edges.jsonl` still triggers the fused execution path
- Learned edge checkpoints contain only `assets/topology.json` edges

---

## Acceptance Criteria

- `cargo check` passes
- `cargo test --no-default-features` passes
- A GPU-enabled run executes a JIT-synthesized fused kernel (not a pre-compiled one)
- Intermediate slot values do not appear in `ProcessReceipt.stdout_payload` for chained ops
- The chain kernel is compiled exactly once per operator sequence per run
- `assets/fusion_patterns.json` and `src/fusion.rs` are deleted
- `cuda/trading_execution_kernels.cu` contains no hand-written math kernels
- `build.rs` has no nvcc step for math operators
- Grep finds no `FusionCandidate`, `FusionPatterns`, or `compile_fusion_patterns` in the codebase

---

## Non-Goals

- No transformer implementation
- No cuBLAS or WMMA integration
- No multi-GPU
- No quantization
- No KV cache or autoregressive generation
- No Triton or external JIT compiler — NVRTC only (already present)
- No runtime PTX string generation for topology or control operators
  (only `jit_cuda` math operators are JIT-synthesized)
- No CPU graph traversal planner
- No JSONL alternate topology or policy
- No Rust hardcoding of operator names, slot names, or chain rules
