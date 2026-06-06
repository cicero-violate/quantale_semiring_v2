# PENDING — Streaming Quantale Integration

## Status

```text
implemented
```

## Purpose

Connect the existing streaming ingress layer to the quantale runtime so streamed
events do more than update host-side queues. The target is:

```text
async source
  -> RawStreamEvent
  -> SlotUpdate / NodeDelta / EdgeDelta
  -> bounded queues
  -> single GPU-owner applier
  -> DeviceSlotRegistry + TensorQuantaleWorld
  -> active quantale event frontier
  -> scheduler burst
  -> receipts
```

The existing streamer already defines the input side:

```text
RawStreamEvent
SlotUpdate
StreamReceipt
StreamIngress
StreamWorkers
AsyncStreamBridge
SlotApplier
PinnedSlotApplier
BackpressurePolicy
compact_latest_wins
normalize_event
```

The missing bridge is:

```text
applied stream update -> quantale-visible node/edge state mutation
```

One-line rule:

```text
Streaming data changes GPU state; applied stream receipts activate quantale nodes.
```

## Core Model

Let:

```math
G=(V,E)
```

Where:

| Symbol | Meaning |
|--------|---------|
| `V` | quantale nodes |
| `E` | quantale edges |
| `D_t` | GPU device slot state at time `t` |
| `Q_r` | raw event queue |
| `Q_u` | slot update queue |
| `u_t` | typed stream update |
| `R_t` | stream receipt |
| `F_t` | active frontier |
| `W_t` | tensor edge matrix |

Raw event normalization:

```math
x_t \rightarrow Q_r \rightarrow u_t \rightarrow Q_u
```

Device-slot update:

```math
D_{t+1}=Apply(D_t, compact(Q_u))
```

Receipt-to-frontier activation:

```math
F_{t+1}=F_t \cup EventNodes(R_t)
```

Streamed topology update:

```math
W_{t+1}=W_t \oplus \Delta E_t
```

## Current Gap

`src/streaming.rs` can already normalize and apply slot batches, but the main
runtime does not yet make streamed updates part of the epoch-owned quantale
execution state.

Current effective path:

```text
RawStreamEvent -> SlotUpdate -> DeviceSlotRegistry
```

Needed path:

```text
RawStreamEvent
  -> SlotUpdate
  -> DeviceSlotRegistry
  -> StreamReceipt
  -> Event::StreamUpdated / Event::MarketFeedUpdated
  -> active[] frontier
  -> TensorQuantaleWorld scheduler
```

Without the receipt-to-frontier bridge, data may be uploaded while the scheduler
has no quantale-visible reason to execute the affected nodes.

## Runtime Ownership Invariants

### 1. Single GPU Mutator

Only one runtime owner may mutate GPU state:

```text
many producers -> bounded queues -> one GPU owner -> device state
```

Allowed:

```text
StreamWorkers::apply_pending called from the orchestrator/GPU-owner thread
```

Forbidden:

```text
background stream reader directly mutates TensorQuantaleWorld or DeviceSlotRegistry
```

### 2. Slot Updates Must Be Typed

Raw JSON must not enter device memory directly.

Required path:

```text
RawStreamEvent -> normalize_event -> SlotUpdate -> validate_update -> apply_batch
```

### 3. Stream Receipts Are Control Signals

Every applied or dropped update produces a receipt. Applied receipts can activate
quantale event nodes. Dropped receipts must be append-only diagnostic records.

```text
StreamReceipt.applied == true  -> may activate event node
StreamReceipt.dropped == true  -> log only, do not activate normal event path
```

### 4. Edges Are Separate From Slots

Slot update:

```text
market.price -> DeviceSlotRegistry
```

Edge update:

```text
src/dst/confidence/cost/safety -> TensorEdge -> embed_tensor_edges
```

Do not overload `SlotUpdate` to mean graph topology mutation.

## Required Code Changes

## 1. Extend `RuntimeEpoch`

File:

```text
src/app/runtime_epoch.rs
```

Current shape:

```rust
pub(crate) struct RuntimeEpoch {
    pub(crate) topology: TopologyRuntime,
    pub(crate) executor: UniversalExecutor,
    pub(crate) world: TensorQuantaleWorld,
    pub(crate) learning_buffer: LearningBuffer,
}
```

Target shape:

```rust
pub(crate) struct RuntimeEpoch {
    pub(crate) topology: TopologyRuntime,
    pub(crate) executor: UniversalExecutor,
    pub(crate) world: TensorQuantaleWorld,
    pub(crate) learning_buffer: LearningBuffer,

    pub(crate) slot_registry: DeviceSlotRegistry,
    pub(crate) stream_workers: Option<StreamWorkers>,
    pub(crate) stream_receipts: StreamReceiptWriter,
}
```

Reason:

```text
The epoch must own the quantale world, stream workers, slot registry, and receipt writer together.
```

## 2. Add GPU-owner Stream Batch Method

File:

```text
src/tensor/mod.rs or src/tensor/orchestration.rs
```

Add a method that keeps CUDA device ownership inside `TensorQuantaleWorld`:

```rust
impl TensorQuantaleWorld {
    pub fn apply_stream_batch(
        &mut self,
        stream: &mut StreamWorkers,
        slots: &mut DeviceSlotRegistry,
    ) -> Vec<StreamReceipt> {
        stream.apply_pending(slots, &self.dev)
    }
}
```

Do not expose arbitrary mutable CUDA internals unless necessary.

Fallback acceptable form:

```rust
impl TensorQuantaleWorld {
    pub fn device(&self) -> std::sync::Arc<cudarc::driver::CudaDevice> {
        self.dev.clone()
    }
}
```

Preferred invariant:

```text
TensorQuantaleWorld remains the GPU-owner boundary.
```

## 3. Add `SlotSchema::iter`

File:

```text
src/streaming.rs
```

Needed for preallocating device slots from the validated schema.

```rust
impl SlotSchema {
    pub fn iter(&self) -> impl Iterator<Item = &DeviceSlot> {
        self.slots.values()
    }
}
```

## 4. Build Stream Slot Schema At Startup

File:

```text
src/app/runtime_epoch.rs
```

Minimal initial schema:

```rust
fn build_stream_slot_schema() -> SlotSchema {
    let mut schema = SlotSchema::new();

    schema.register(DeviceSlot::tensor_f32("market.price", vec![3]));
    schema.register(DeviceSlot::tensor_f32("market.open", vec![3]));
    schema.register(DeviceSlot::tensor_f32("market.high", vec![3]));
    schema.register(DeviceSlot::tensor_f32("market.low", vec![3]));
    schema.register(DeviceSlot::tensor_f32("market.volume", vec![3]));

    schema.register(DeviceSlot::tensor_f32("analysis.return", vec![3]));
    schema.register(DeviceSlot::tensor_f32("analysis.volatility", vec![3]));
    schema.register(DeviceSlot::tensor_f32("analysis.signal_score", vec![3]));

    schema
}
```

Later upgrade:

```text
build schema from operator effects reads/writes + configured symbol count
```

## 5. Preallocate `DeviceSlotRegistry`

File:

```text
src/app/runtime_epoch.rs
```

During epoch build:

```rust
let schema = build_stream_slot_schema();
let mut slot_registry = DeviceSlotRegistry::new();

for slot in schema.iter() {
    let buf = world
        .device()
        .htod_copy(vec![0.0_f32; slot.len()])
        .map_err(|error| error.to_string())?;
    slot_registry.register(slot.clone(), buf);
}
```

If using `TensorQuantaleWorld::apply_stream_batch` and not exposing `device()`,
add a small helper method for slot allocation instead.

## 6. Preserve Slot Metadata During Upload

File:

```text
src/device_slots.rs
src/streaming.rs
```

Current `UploadQueue::flush` path can synthesize metadata from buffer length:

```rust
registry.insert(slot.clone(), device_buf);
```

Target behavior should preserve schema metadata:

```rust
registry.register(meta.clone(), device_buf);
```

Recommended representation:

```rust
staged: Vec<(DeviceSlot, HostStagingBuffer)>
```

instead of:

```rust
staged: Vec<(String, HostStagingBuffer)>
```

Reason:

```text
Shape and dtype are part of the stream contract, not disposable upload metadata.
```

## 7. Wire Stream Batch Before Each Orchestrator Burst

File:

```text
src/main.rs
src/app/supervisor.rs
```

Current control shape:

```text
orchestrate_until_wait_or_halt
  -> maybe service external commands
```

Target control shape:

```text
apply stream batch
write stream receipts
activate stream event nodes
orchestrate_until_wait_or_halt
maybe service external commands
```

Sketch:

```rust
if let Some(stream) = epoch.stream_workers.as_mut() {
    let receipts = epoch.world.apply_stream_batch(stream, &mut epoch.slot_registry);
    epoch.stream_receipts.append(&receipts)?;
    activate_stream_event_nodes(&mut epoch.world, &epoch.topology, &receipts)?;
}

let status = epoch.world.orchestrate_until_wait_or_halt(max_device_steps)?;
```

## 8. Add Active-Node Mutation API

File:

```text
src/tensor/orchestration.rs
cuda/quantale/03_tensor_core.cuh or cuda/quantale/06_scheduler.cuh
```

Needed host method:

```rust
impl TensorQuantaleWorld {
    pub fn mark_node_active(&mut self, node_id: i32) -> Result<(), CudaError> {
        // Launch small kernel or copy/update active[] safely.
    }
}
```

Needed CUDA kernel shape:

```cuda
extern "C" __global__ void quantale_mark_node_active(
    int* active,
    int node_id,
    int n
) {
    if (threadIdx.x == 0 && node_id >= 0 && node_id < n) {
        active[node_id] = 1;
    }
}
```

Reason:

```text
Stream receipts must become scheduler-visible frontier state.
```

## 9. Map Receipts To Event Nodes

File:

```text
src/app/supervisor.rs or new src/app/stream_quantale.rs
```

Sketch:

```rust
fn activate_stream_event_nodes(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    receipts: &[StreamReceipt],
) -> Result<(), CudaError> {
    if receipts.iter().any(|r| r.applied) {
        if let Some(id) = topology.registry().id_of("Event::StreamUpdated") {
            world.mark_node_active(id as i32)?;
        }
    }

    if receipts
        .iter()
        .any(|r| r.applied && r.slot.starts_with("market."))
    {
        if let Some(id) = topology.registry().id_of("Event::MarketFeedUpdated") {
            world.mark_node_active(id as i32)?;
        }
    }

    Ok(())
}
```

## 10. Add Stream Event Nodes To Topology

Files:

```text
assets/topology.source.json
assets/operators.json
crates/topology_core/src/overlay.rs
```

Minimum nodes:

| Node | Role |
|------|------|
| `Stream::Poll` | source polling / async source boundary |
| `Stream::Normalize` | raw event to typed update |
| `Stream::SlotUpdate` | typed slot update creation |
| `Stream::ApplySlot` | device slot mutation |
| `Stream::Backpressure` | pressure/drop/compact policy |
| `Event::StreamUpdated` | generic applied stream receipt event |
| `Event::MarketFeedUpdated` | applied market slot event |

Minimum edges:

```text
Stream::Poll -> Stream::Normalize
Stream::Normalize -> Stream::SlotUpdate
Stream::SlotUpdate -> Stream::ApplySlot
Stream::ApplySlot -> Event::StreamUpdated
Event::MarketFeedUpdated -> Analysis::Return1
Event::MarketFeedUpdated -> Analysis::Volatility
Event::MarketFeedUpdated -> Analysis::SignalScore
```

If stream nodes are host-owned rather than GPU-dispatched, mark them as external
or abstract-device nodes with explicit receipts.

## 11. Pass Slot Registry Into Hot Region / Par Group Data

Existing API already supports this:

```rust
world.make_par_group_data(
    groups,
    region_ids,
    is_gpu_dispatchable,
    dispatch_kinds,
    Some(&slot_registry),
    Some(&config.fusion_hf_coverage),
)
```

Ensure runtime construction passes `Some(&epoch.slot_registry)` instead of
`None` when building par-group GPU data.

Reason:

```text
Without DeviceSlotRegistry, hot regions run receipt-only and do not consume real stream data.
```

## 12. Add Streamed Edge Deltas

File:

```text
src/streaming.rs or new src/streaming_quantale.rs
```

Add separate topology-delta type:

```rust
pub struct EdgeDelta {
    pub source: String,
    pub src: String,
    pub dst: String,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
    pub observed_at: String,
    pub event_hash: String,
}
```

Normalize from envelope:

```json
{
  "kind": "edge_delta",
  "src": "Event::MarketFeedUpdated",
  "dst": "Analysis::Return1",
  "confidence": 0.95,
  "cost": 0.02,
  "safety": 1.0
}
```

Apply:

```rust
fn apply_edge_delta(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    delta: EdgeDelta,
) -> Result<(), CudaError> {
    let src = topology
        .registry()
        .id_of(&delta.src)
        .ok_or_else(|| CudaError::invalid_input(format!("unknown src '{}'", delta.src)))?;
    let dst = topology
        .registry()
        .id_of(&delta.dst)
        .ok_or_else(|| CudaError::invalid_input(format!("unknown dst '{}'", delta.dst)))?;

    let edge = TensorEdge::new(
        src as i32,
        dst as i32,
        delta.confidence,
        delta.cost,
        delta.safety,
    );

    world.embed_tensor_edges(&[edge])
}
```

## 13. Add Streamed Node Deltas

Use node deltas for frontier/state events that do not change edge weights.

```rust
pub struct NodeDelta {
    pub source: String,
    pub node: String,
    pub active: bool,
    pub observed_at: String,
    pub event_hash: String,
}
```

Apply:

```text
NodeDelta(active=true) -> mark_node_active(node_id)
```

Do not use `NodeDelta` for slot data payloads. Slot payloads remain
`SlotUpdate`.

## Recommended File Layout

```text
src/streaming.rs                  existing queue/normalizer/applier layer
src/streaming_quantale.rs         new node/edge delta + receipt activation bridge
src/app/stream_runtime.rs         optional runtime wiring helpers
src/app/runtime_epoch.rs          epoch-owned stream/slot construction
src/app/supervisor.rs             stream-before-burst integration
```

Export from `src/lib.rs`:

```rust
pub mod streaming_quantale;
pub use streaming_quantale::*;
```

## Test Plan

Add or extend tests:

| Test | Purpose |
|------|---------|
| `stream_receipt_activates_event_node` | applied receipt marks event node active |
| `dropped_stream_receipt_does_not_activate_event_node` | safety for rejected updates |
| `market_slot_receipt_activates_market_event` | market slot maps to `Event::MarketFeedUpdated` |
| `edge_delta_embeds_tensor_edge` | streamed edge delta mutates `W` |
| `unknown_edge_delta_node_rejected` | topology safety |
| `slot_schema_preallocates_registry` | all schema slots exist in `DeviceSlotRegistry` |
| `hot_region_receives_streamed_slots` | par/hot region uses real slot pointers |
| `stream_batch_runs_before_orchestrator_burst` | ordering invariant |
| `single_gpu_owner_applies_stream_updates` | ownership invariant |

## Acceptance Criteria

The integration is complete when:

```text
1. RuntimeEpoch owns stream workers, stream receipts, and DeviceSlotRegistry.
2. Stream batches are applied before every scheduler burst.
3. Applied stream receipts activate quantale event nodes.
4. Market updates can trigger analysis nodes without restarting the runtime.
5. Hot GPU regions receive real streamed slot pointers.
6. Edge deltas can be validated and embedded into TensorQuantaleWorld.
7. Dropped updates are logged but do not activate normal event nodes.
8. Background stream workers never directly mutate GPU state.
```

## Non-Goals

Do not add these in the first pass:

```text
full Tokio rewrite of the GPU owner
unbounded async channels
direct GPU writes from network tasks
raw JSON -> CUDA memory bypass
dynamic unknown slot creation during hot path
edge deltas without topology validation
```

## Final Target

```text
streaming input
  -> typed algebraic deltas
  -> bounded queues
  -> single GPU owner
  -> slot/node/edge mutation
  -> quantale frontier activation
  -> GPU scheduler burst
  -> receipts and replayable trace
```

The key invariant is:

```text
Async streaming is ingress; quantale execution remains algebraic and GPU-owned.
```