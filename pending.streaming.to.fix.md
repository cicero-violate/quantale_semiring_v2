# PENDING — Streaming Fix Plan

## Status

```text
implemented
```

## Purpose

Stabilize the streaming-to-quantale integration without regressing the GPU-native orchestrator.

Target shape:

```text
stream ingress
  -> typed slot / node / edge deltas
  -> bounded queues
  -> GPU-owner boundary
  -> TensorQuantaleWorld mutation APIs
  -> active quantale frontier
  -> GPU-native scheduler burst
  -> CPU command service only on WaitExternal
  -> receipt foldback
```

Core rule:

```text
CPU services boundaries; GPU owns graph scheduling.
```

## Non-Regression Guard

Do not replace the active runtime path:

```rust
gpu_native_supervisor_loop(...)
world.orchestrate_until_wait_or_halt(max_device_steps)?
```

The runtime must remain:

```text
GPU-native orchestrator with CPU boundary services
```

Not:

```text
CPU-driven node scheduler
```

## Current Known State

Observed source direction:

```text
RuntimeEpoch owns slot_registry / stream_workers / stream_receipts
main.rs calls gpu_native_supervisor_loop
supervisor calls world.orchestrate_until_wait_or_halt
streaming_quantale.rs defines EdgeDelta / NodeDelta helpers
streaming.rs has started adding TopologyDelta routing
topology check has been fixed to include stream/event nodes
```

Important distinction:

```text
GPU-native orchestration did not regress.
Streaming integration is the unstable area to finish and verify.
```

## Required Verification Commands

Run these after every fix:

```bash
cargo check -q
cargo test -q --lib
cargo test -q
cargo run -q -- --check-topology
```

Expected final result:

```text
cargo check: pass
lib tests: pass
full default tests: pass
topology check: ok
```

## Fix Order

### 1. Keep GPU-Native Orchestrator Untouched

Files:

```text
src/main.rs
src/app/supervisor.rs
src/tensor/orchestration.rs
```

Must remain true:

```rust
gpu_native_supervisor_loop(...)
```

must call:

```rust
world.orchestrate_until_wait_or_halt(max_device_steps)?
```

CPU work must remain only in these boundary hooks:

```text
pre_burst_fn
service_fn on WaitExternal
burst-boundary persistence/logging
```

Do not add CPU-side graph-step selection.

### 2. Make TopologyDelta Routing Compile-Clean

Files:

```text
src/streaming.rs
src/streaming_quantale.rs
src/main.rs
```

Intended edge-delta pipeline:

```text
RawStreamEvent(kind=edge_delta)
  -> try_normalize_topology_delta
  -> TopologyDelta::Edge
  -> StreamWorkers.topology_delta_rx
  -> drain_topology_deltas()
  -> apply_topology_delta(...)
  -> world.embed_tensor_edges(...)
```

Intended node-delta pipeline:

```text
RawStreamEvent(kind=node_delta)
  -> try_normalize_topology_delta
  -> TopologyDelta::Node
  -> StreamWorkers.topology_delta_rx
  -> drain_topology_deltas()
  -> apply_topology_delta(...)
  -> world.mark_node_active(...)
```

Acceptance checks:

```text
run_normalizer_worker accepts the topology-delta sender
normalizer routes edge_delta/node_delta into the delta channel
normalizer does not emit fake SlotUpdate for topology deltas
drain_topology_deltas is callable only from the GPU-owner boundary
```

### 3. Preserve SlotUpdate Path

Files:

```text
src/streaming.rs
src/device_slots.rs
```

Do not break the existing slot path:

```text
RawStreamEvent(kind=market_tick)
  -> normalize_event
  -> SlotUpdate
  -> StreamWorkers.update_rx
  -> apply_pending
  -> SlotApplier
  -> DeviceSlotRegistry
  -> StreamReceipt
```

Rules:

```text
slot updates mutate data slots
topology deltas mutate nodes/edges
do not encode topology deltas as pseudo-slots
do not let background workers touch GPU state
```

### 4. Ensure Runtime Applies Both Stream Products

File:

```text
src/main.rs
```

The pre-burst hook should do this order:

```text
1. apply streamed slot updates
2. append slot receipts
3. activate receipt event nodes
4. drain topology deltas
5. apply topology deltas through TensorQuantaleWorld APIs
6. start GPU orchestration burst
```

Canonical shape:

```rust
if let Some(stream) = epoch.stream_workers.as_mut() {
    let receipts = world.apply_stream_batch(stream, &mut epoch.slot_registry);
    epoch.stream_receipts.append(&receipts)?;
    activate_stream_event_nodes(world, &epoch.topology, &receipts)?;

    for delta in stream.drain_topology_deltas() {
        apply_topology_delta(world, &epoch.topology, delta)?;
    }
}
```

Do not run this inside the GPU step kernel. It is a boundary operation.

### 5. Decide Stream Source Startup Policy

File:

```text
src/app/runtime_epoch.rs
```

Current shape may be:

```rust
stream_workers: None
```

This is acceptable only if runtime has an explicit source-attachment policy.

Choose one:

#### Option A — Disabled by default

```text
stream_workers: None
```

Document this as intentional and provide a future CLI/config hook.

#### Option B — File source by config

```text
if config.stream_file exists:
    StreamWorkers::spawn(StreamConfig, schema, FileLineSource)
else:
    None
```

#### Option C — Channel bridge by API

```text
construct ChannelStreamSource / AsyncStreamBridge pair
return bridge to caller
```

Do not silently claim streaming is live if `stream_workers` is always `None`.

Acceptance rule:

```text
The runtime must make clear whether streaming is enabled, disabled, or externally attached.
```

### 6. Keep Par-Group Slot Registry Item Scoped

Current main runtime appears to use:

```text
orchestrate_until_wait_or_halt
```

not direct host-side:

```rust
make_par_group_data(...)
par_group_step(...)
```

Therefore:

```text
Do not block current streaming stabilization on par_group_step wiring unless main runtime starts using it.
```

If a path later calls `make_par_group_data`, then pass:

```rust
Some(&epoch.slot_registry)
```

instead of:

```rust
None
```

Acceptance rule:

```text
Mark this item N/A for the current main runtime path, but keep it as a guard for future par-group execution paths.
```

### 7. Test Coverage To Keep

Required tests:

```text
slot_update path still normalizes market_tick
edge_delta normalizes into TopologyDelta::Edge
node_delta normalizes into TopologyDelta::Node
drain_topology_deltas returns routed deltas
applied stream receipt activates Event::StreamUpdated
market slot receipt activates Event::MarketFeedUpdated
dropped receipt does not activate normal event path
unknown edge node is rejected
unknown node delta target is rejected
topology check passes with stream nodes reachable or explicitly exempted
```

Avoid tests that only assert helper structs exist. Prefer end-to-end boundary tests.

### 8. Failure Modes To Prevent

Do not allow:

```text
background normalizer mutating TensorQuantaleWorld
background reader mutating DeviceSlotRegistry
raw JSON directly mutating GPU memory
CPU choosing next graph node
topology deltas encoded as magic slot names
stream nodes unreachable in checked topology
full tests blocked by unused imports
```

## Definition Of Done

This plan is complete when:

```text
1. cargo check -q passes
2. cargo test -q --lib passes
3. cargo test -q passes
4. cargo run -q -- --check-topology reports ok
5. GPU-native supervisor path remains active
6. Slot updates and topology deltas are routed separately
7. CPU services only run at pre-burst or WaitExternal boundaries
8. Stream source startup policy is explicit
9. pending.streaming.quantale.md can be marked implemented or split into remaining future items
```

## Final Target Shape

```text
CPU stream/input boundary
  -> bounded queues
  -> slot updates + topology deltas
  -> TensorQuantaleWorld boundary APIs
  -> GPU-native orchestrator burst
  -> WaitExternal command ring
  -> CPU service
  -> receipt ring
  -> GPU receipt foldback
```

Final invariant:

```text
Streaming feeds the GPU-native orchestrator; it does not replace it.
```