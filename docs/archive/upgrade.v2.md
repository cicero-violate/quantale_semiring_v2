## State after upgrade.v1

The following items from `upgrade.v1` were completed or partially completed:

```
topology split assets created        ✓
HotRegionRegistry + regions.hot.json ✓
DeviceSlotRegistry, DeviceBufferPool ✓
DeviceRingBuffer (allocation only)   ✓
DeviceReceipt + ring buffer          ✓
tensor_quantale_drain_device_receipts ✓
tensor_quantale_gpu_dispatch (stub)  ✓
TypedIR + IrPipeline                 ✓ (scalar ops only)
FusionDispatch::entry_from_ir_pipeline ✓
is_hot_node(), hot dispatch routing  ✓
fusion-before-hot ordering           ✓
receipt outcome propagation          ✓
validate_slots()                     ✓
build passes --no-default-features   ✓
```

The following items are **not yet done** and are the scope of this session.

---

## Variables

[
G_{control}=\texttt{assets/topology.control.json}
]

[
G_{hot}=\texttt{assets/topology.hot.json}
]

[
R_{hot}=\texttt{assets/regions.hot.json}
]

[
F=\texttt{assets/topology.fusion.json}
]

[
Q_G=\text{GPU quantale runtime}
]

[
D_G=\text{device slot registry}
]

[
Receipt_G=\text{device receipt ring}
]

---

## Remaining Work

---

### 1. Make the runtime load G_control and G_hot explicitly

**Current state:**

```
TopologyRuntime::load_checked_default()
  → topology.generated.json          ← still the single source
  → operators.generated.json
```

The split files exist as assets but are not the active runtime source.

**Target:**

```rust
struct SplitTopologyRuntime {
    control: ControlTopologyRuntime,   // loads topology.control.json
    hot:     HotTopologyRuntime,       // loads topology.hot.json
    unified: TopologyRuntime,          // keeps topology.generated.json for tensor
}
```

The unified topology stays as-is for the CUDA quantale tensor (node IDs must not change). The split files drive routing logic only:

- `control.contains(node)` → CPU operator path
- `hot.contains(node)` → GPU region path

Enforce at startup:

```rust
assert!(control.nodes ∩ hot.nodes == ∅);
assert!(hot.nodes ⊆ hot_region_registry.names());
assert!(control.nodes ∩ io_types == control.nodes);  // no GPU compute in control
```

**Files to modify:**

```
src/topology.rs            add SplitTopologyRuntime
src/config.rs              load split files alongside unified
src/main.rs                use split topology for routing decisions
```

---

### 2. Make topology.hot.json executable (add region graph)

**Current state:**

```json
{ "nodes": [...6 nodes...], "transitions": [] }
```

`topology.hot.json` is a node list only. There is no graph structure describing how hot regions sequence.

**Target:**

Add region-level transitions that describe the natural execution chains. These are not CUDA tensor edges — they are declarative graph structure for the region scheduler:

```json
{
  "transitions": [
    { "from": "Analysis::Return1",    "to": "Analysis::Volatility",    "confidence": 1.0, "cost": 0.1, "safety": 1.0 },
    { "from": "Analysis::Volatility", "to": "Analysis::SignalScore",   "confidence": 1.0, "cost": 0.1, "safety": 1.0 },
    { "from": "Execution::VectorAdd", "to": "Execution::VectorScale",  "confidence": 1.0, "cost": 0.1, "safety": 1.0 }
  ]
}
```

Also add a `Region::CommitReceipt` terminal node:

```json
{ "id": 6, "name": "Region::CommitReceipt", "type": "gpu_region" }
```

with edges from every leaf region → `Region::CommitReceipt`.

**Rule:**

[
G_{hot}=\text{executable region graph},\quad G_{hot}\neq\text{node list}
]

**Files to modify:**

```
assets/topology.hot.json   add transitions, add Region::CommitReceipt
assets/regions.hot.json    add Region::CommitReceipt entry (region_id=6)
```

---

### 3. Fix region slot names to match declared operator slots

**Current mismatch:**

`regions.hot.json` uses:

```
execution.a, execution.b, execution.sum
execution.input, execution.scale, execution.scaled
execution.result
```

The operator contracts in `operators.generated.json` for `Execution::VectorAdd` etc. use:

```
math.a, math.b, math.scale, math.add_out, math.out
```

**Target:**

Align `regions.hot.json` slot names with `operators.generated.json` effects declarations. Run `validate_slots()` at startup and surface any mismatches as a fatal error (not a silent pass).

**Rule:**

[
\forall r\in R_{hot},\quad reads(r)\cup writes(r)\subseteq declared\_slots
]

**Files to modify:**

```
assets/regions.hot.json    rename execution.* slots to match operators.generated.json
src/config.rs              call hot_region_registry.validate_slots() at startup,
                           fatal on violation
```

---

### 4. DeviceRingBuffer: implement push and pop kernels

**Current state:**

```rust
pub struct DeviceRingBuffer {
    pub data: CudaSlice<f32>,
    pub head: CudaSlice<i32>,
    pub tail: CudaSlice<i32>,
    pub capacity: usize,
}
```

Allocation only. No GPU-side write path, no CPU-side read path, no async ingress.

**Target:**

```cuda
// kernel: write n floats to the ring from a source buffer
extern "C" __global__ void device_ring_push(
    float* ring, int* tail, int capacity,
    const float* src, int n
);

// kernel: read n floats from the ring into a destination buffer
extern "C" __global__ void device_ring_pop(
    const float* ring, int* head, int capacity,
    float* dst, int n
);
```

Rust side:

```rust
impl DeviceRingBuffer {
    pub fn push(&mut self, world: &TensorQuantaleWorld, src: &CudaSlice<f32>) -> Result<(), CudaError>;
    pub fn pop(&mut self, world: &TensorQuantaleWorld, n: usize) -> Result<CudaSlice<f32>, CudaError>;
}
```

**Files to modify:**

```
cuda/quantale_world.cu     add device_ring_push, device_ring_pop kernels
src/tensor.rs              register new kernels in load_ptx
src/device_slots.rs        add push/pop methods to DeviceRingBuffer
```

---

### 5. CPU ingress pipeline: pinned host buffer → async upload → DeviceRingBuffer

**Current state:**

External data enters via `current_payload: Value` (JSON blob passed as operator stdin). There is no zero-copy or async upload path.

**Target:**

```
ExternalData (CPU)
  → PinnedHostBuffer (page-locked alloc)
  → cudaMemcpyAsync (H2D, non-blocking)
  → DeviceRingBuffer (D_G)
  → GPU region reads from D_G
```

New types:

```rust
/// Page-locked host buffer for async DMA.
pub struct PinnedHostBuffer {
    ptr: *mut f32,
    len: usize,
}

/// Async upload queue: stages CPU data for H2D transfer.
pub struct AsyncUploadQueue {
    staged: Vec<(String, PinnedHostBuffer)>,  // slot_name → buffer
    stream: CudaStream,
}

impl AsyncUploadQueue {
    pub fn stage(&mut self, slot: &str, data: &[f32]) -> Result<(), CudaError>;
    pub fn flush(&mut self, registry: &mut DeviceSlotRegistry) -> Result<(), CudaError>;
}
```

**Files to create / modify:**

```
src/device_slots.rs        add PinnedHostBuffer, AsyncUploadQueue
src/egress.rs              add async ingress path alongside blocking path
src/main.rs                wire ingress into epoch.world before tick
```

---

### 6. True GPU-side region dispatch (replace receipt-only stub)

**Current state:**

`tensor_quantale_gpu_dispatch` writes a receipt but does not execute a region. Execution still flows:

```
CPU decides → CPU launches JIT kernel → GPU runs → CPU writes mailbox → GPU writes receipt
```

**Target:**

```
GPU quantale selects region_id
→ device-side dispatch table
→ device function or dynamic parallelism (CDP) call
→ GPU executes region
→ GPU writes DeviceReceipt
→ tensor_quantale_drain_device_receipts
```

**Two viable approaches:**

**A. Device function table (no CDP required):**

```cuda
// Each hot region is a __device__ function, not a __global__ kernel.
// The dispatch kernel calls them via switch.

__device__ void region_vector_add(float** slots, DeviceReceipt* r);
__device__ void region_analysis_return1(float** slots, DeviceReceipt* r);
// ...

extern "C" __global__ void tensor_quantale_gpu_dispatch(
    int region_id,
    float** slot_ptrs,   // device pointer array indexed by slot id
    DeviceReceipt* ring, int* ring_tail, int ring_size
) {
    switch (region_id) {
        case 0: region_vector_add(slot_ptrs, &r);       break;
        case 3: region_analysis_return1(slot_ptrs, &r); break;
        // ...
    }
}
```

This is the recommended approach. It requires:
- JIT-synthesized bodies moved from `__global__` to `__device__` helper functions
- The dispatch kernel reconstructs inputs from the slot pointer array

**B. CUDA Dynamic Parallelism (CDP):**

```cuda
extern "C" __global__ void tensor_quantale_gpu_dispatch(...) {
    jit_fused_kernel<<<grid, block>>>(slots, out, n);
    cudaDeviceSynchronize();
}
```

Requires `sm_35+`, `--rdc=true`, `cudaDeviceSynchronize()` inside kernel. More flexible but more complex.

**Recommended approach: A (device function table)**

**Files to modify:**

```
cuda/quantale_world.cu     refactor jit_fused kernels to __device__ functions,
                           add device function dispatch switch
src/jit_kernel_fusion/synth.rs
                           generate __device__ function signature instead of
                           __global__ when used as a region body
src/tensor.rs              update gpu_dispatch_region to pass slot pointer array
```

---

### 7. Complete TypedIR lowering

**Current state — implemented (scalar-correct):**

```
Map, Filter, Window, Verify
simple scalar Reduce (per-element fake)
```

**Current state — placeholder only:**

```
TopK     → copies input
MatMul   → element-wise multiply (not GEMM)
Embed    → simplistic lookup
Reduce   → per-element, not a true parallel reduction
Join     → not lowerable (returns Err)
Sort     → not lowerable (returns Err)
GraphTraverse → not lowerable (returns Err)
```

**Target for each:**

```
Reduce    → parallel reduction tree (warp shuffle + shared memory)
TopK      → radix sort + head slice (or bitonic sort for small k)
MatMul    → tiled shared-memory GEMM for square matrices;
            cuBLAS binding for production
Embed     → coalesced row read from embedding table slot
Join      → hash join on key slot (device hash table)
Sort      → bitonic sort or thrust::sort binding
GraphTraverse → BFS frontier kernel over CSR adjacency
```

IR ops that remain non-lowerable should return a clear compile-time error, not a placeholder body that silently produces wrong results.

**Files to modify:**

```
src/ir.rs                  replace placeholder match arms with real CUDA C bodies
                           or explicit Err for non-scalar ops pending GEMM/sort
cuda/quantale_world.cu     add parallel_reduce, topk_bitonic, embed_coalesced
                           device helper kernels
```

---

## Priority order for this session

```
1  Fix region slot names (blocks correctness validation)
2  Activate split topology at runtime (blocks runtime/hot separation)
3  Add hot topology transitions + Region::CommitReceipt (blocks hot graph execution)
4  DeviceRingBuffer push/pop kernels (enables streaming)
5  CPU ingress pipeline (enables async data flow)
6  True GPU-side dispatch via device function table (enables full GPU hot loop)
7  TypedIR: Reduce, TopK, MatMul (enables real data pipelines)
```

---

## Invariants to enforce before session ends

```rust
// 1. No IO nodes in hot topology
assert!(G_hot.nodes.iter().all(|n| !n.is_control_io()));

// 2. All hot topology nodes have region metadata
assert!(G_hot.nodes.iter().all(|n| R_hot.is_hot(&n.name)));

// 3. All region slots are declared in operators.generated.json effects
assert!(R_hot.validate_slots(&declared_slots).is_empty());

// 4. Hot topology has at least one transition
assert!(!G_hot.transitions.is_empty());

// 5. build passes without cuda feature
// cargo check --no-default-features  →  0 errors
```

---

## What does NOT change

```
topology.generated.json    N=79 stays fixed, tensor unchanged
CUDA_TENSOR_NODE_COUNT     unchanged
existing CKA/batch dispatch unchanged
existing exploration path  unchanged
existing fusion chain      Analysis::Return1→Volatility→SignalScore still works
```
