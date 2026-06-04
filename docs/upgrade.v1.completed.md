## Variables

[
G_{old}=\text{current mixed CPU/GPU workflow graph}
]

[
G_{control}=\text{cold CPU/SYS graph}
]

[
G_{hot}=\text{GPU-only execution graph}
]

[
D_G=\text{GPU-resident data / VRAM state}
]

[
R_i=\text{GPU region}
]

[
Q_G=\text{GPU quantale orchestrator}
]

[
K_i=\text{kernel / fused kernel}
]

[
Receipt_G=\text{device-side receipt}
]

---

# Upgrade Goal

[
\boxed{
G_{old}
\rightarrow
G_{control} \cup G_{hot}
}
]

[
\boxed{
CPU = SYS/IO
}
]

[
\boxed{
GPU = data + logic + compute + orchestration
}
]

One-line explanation:

[
\text{Move the hot execution system from CPU-dispatched operators to GPU-resident regions.}
]

---

# Upgrade List

## 1. Split hot graph from control graph

Current problem:

[
topology = IO + CPU operators + GPU compute + workflow
]

Target:

[
G_{control}={IO, files, LLM, compile, logs, syscalls}
]

[
G_{hot}={GPU regions, device control flow, receipts, tensor updates}
]

**Upgrade:**

Create separate topology layer:

```text
assets/topology.control.json
assets/topology.hot.json
```

---

## 2. Remove IO from hot topology

Remove from hot graph:

[
FileRead,\ FileWrite,\ HTTP,\ WebSocket,\ LLM,\ Shell,\ PythonProcess
]

Keep IO only at system boundary:

[
ExternalWorld \rightarrow CPU_{sys} \rightarrow D_G
]

**Upgrade:**

Hot graph only consumes GPU buffers.

---

## 3. Replace operator nodes with GPU region nodes

Current:

[
State::Plan \rightarrow PythonOperator
]

[
State::Execute \rightarrow CPU dispatch
]

Target:

[
Region::FusedExpr
]

[
Region::Transform
]

[
Region::Reduce
]

[
Region::VerifyDevice
]

[
Region::CommitReceipt
]

**Upgrade:**

Topology nodes become **GPU regions**, not CPU tasks.

---

## 4. Add GPU slot system

Need generalized GPU memory objects:

[
Slot=(name,kind,dtype,shape,layout,device_ptr)
]

Examples:

[
price_ring,\ embeddings,\ table_columns,\ graph_csr,\ image_tensor,\ event_log
]

**Upgrade:**

Create a GPU slot registry:

```text
DeviceSlotRegistry
DeviceBufferPool
DeviceRingBuffer
DeviceReceiptBuffer
```

---

## 5. Add region metadata table

Each region needs:

[
R_i=
{
id,
inputs,
outputs,
kernel,
preconditions,
postconditions,
receipt
}
]

Example:

```json
{
  "name": "Region::WindowAggregate",
  "kind": "gpu_region",
  "reads": ["events.normalized"],
  "writes": ["features.windowed"],
  "kernel": "window_aggregate_fused",
  "pure": true
}
```

**Upgrade:**

Build:

```text
assets/regions.hot.json
```

or extend topology with region metadata.

---

## 6. Move quantale dispatch fully GPU-side

Current:

[
Q_G \rightarrow DecisionReport \rightarrow CPU \rightarrow operator
]

Target:

[
Q_G \rightarrow region_id \rightarrow DeviceDispatch(region_id)
]

**Upgrade:**

Add one of:

[
\text{persistent GPU runtime kernel}
]

or:

[
\text{CUDA graph epoch runner}
]

or:

[
\text{device-side region dispatch table}
]

Main missing piece:

[
\boxed{
GPU\ decision \rightarrow GPU\ region execution
}
]

---

## 7. Add device-side receipts

Current:

[
Receipt \rightarrow CPU \rightarrow tensor update
]

Target:

[
Receipt_G \rightarrow TensorUpdate_G
]

Receipt fields:

[
status,\ latency,\ valid,\ error,\ rows,\ risk,\ output_flags
]

**Upgrade:**

Create:

```text
DeviceReceipt
DeviceReceiptBuffer
tensor_quantale_drain_device_receipts
```

---

## 8. Generalize beyond finance

Target data kinds:

[
DataKind=
{
Tensor,
Table,
Stream,
Graph,
Text,
Embedding,
Image,
Audio,
SparseMatrix,
EventLog,
KeyValue,
TimeSeries
}
]

**Upgrade:**

The system becomes:

[
Data \rightarrow TypedDeviceSlots \rightarrow GPURegions \rightarrow Receipts
]

not:

[
FinanceOnly \rightarrow QuantStrategy
]

---

## 9. Add typed data IR

Need an IR that describes data transformations:

[
IR=
Map
\mid Filter
\mid Reduce
\mid Join
\mid Sort
\mid Window
\mid TopK
\mid MatMul
\mid GraphTraverse
\mid Embed
\mid Verify
]

**Upgrade:**

Compiler path:

[
Input \rightarrow TypedIR \rightarrow FusionPlan \rightarrow GPURegion
]

---

## 10. Use JIT fusion as region compiler

Current JIT fusion fuses declared `jit_cuda` operators.

Upgrade it to consume IR:

[
TypedIR \rightarrow jit_cuda\ operator\ chain \rightarrow fused\ kernel
]

Target:

[
Add/Sub/Mul \in IR
]

[
Region::FusedExpr \in topology
]

---

## 11. Reduce CPU to SYS

CPU responsibilities:

[
CPU_{sys}=
IO
+
Driver
+
Compile
+
Allocate
+
Load
+
Persist
+
Observe
]

Not CPU responsibilities:

[
schedule,\ branch,\ compute,\ verify,\ update\ tensor
]

**Upgrade:**

CPU becomes ingress/egress, not orchestrator.

---

## 12. Performance target

Current bottleneck:

[
host/sync/launch\ bound
]

Target:

[
GPU\ hot\ loop
]

[
DeviceBuffers
\rightarrow Q_G
\rightarrow Region
\rightarrow Kernel
\rightarrow Receipt_G
\rightarrow TensorUpdate_G
\rightarrow Q_G
]

No CPU between steps.

---

# Final Upgrade Spec

```text
1. Split control topology from hot topology.
2. Remove IO, LLM, shell, files, Python processes from hot graph.
3. Convert hot nodes into GPU region nodes.
4. Add typed GPU slot registry for generalized data.
5. Add GPU region metadata table.
6. Make quantale select region_id, not CPU operator name.
7. Add device-side dispatcher or persistent runtime kernel.
8. Add device-side receipts and tensor update.
9. Add generalized TypedIR for data manipulation.
10. Connect TypedIR → FusionPlan → JIT fused CUDA region.
11. Keep CPU as SYS/IO boundary only.
12. Make hot path GPU → GPU → GPU.
```

---

# Canonical End State

[
\boxed{
ExternalData
\rightarrow CPU_{sys}
\rightarrow D_G
\rightarrow Q_G
\rightarrow R_i
\rightarrow K_i
\rightarrow Receipt_G
\rightarrow T'
}
]

Plain English:

We want to upgrade from a **GPU-assisted CPU workflow runner** into a **GPU-resident data operating fabric**.

The final system captures any data, stores it in typed GPU slots, manipulates it through fused GPU regions, schedules those regions with the quantale tensor, and updates itself through device receipts.
