# PENDING — Async Streamer

## Status

```text
complete
```

## Purpose

Add a first-class async streaming ingress layer that can feed typed slot updates into the topology quantale orchestrator while GPU/CPU parallel work continues.

The current system already supports:

```text
append-only JSONL state
market feed polling
external IO/process nodes
GPU orchestration bursts
WaitExternal handoff
device command / receipt rings
parallel_groups metadata
safe_parallel effect checks
```

The missing layer is:

```text
async stream ingress
bounded slot-update queues
single GPU slot applier
backpressure policy
optional async CUDA upload
```

## Core Model

```math
S \rightarrow E_t \rightarrow U_t \rightarrow D_G \rightarrow Q(D_G) \rightarrow R_t
```

Where:

| Symbol | Meaning                        |
|--------+--------------------------------|
| `S`    | streaming source               |
| `E_t`  | raw event at time `t`          |
| `U_t`  | typed slot update              |
| `D_G`  | GPU-resident tensor state      |
| `Q`    | quantale topology orchestrator |
| `R_t`  | receipt / trace                |

One-line rule:

```text
Streaming data changes state; the quantale scheduler chooses transitions over that state.
```

## Target Runtime Shape

```text
StreamReader(s)
  -> RawEventQueue
  -> Normalizer
  -> SlotUpdateQueue
  -> SlotApplier
  -> GPU Tensor Slots
  -> QuantaleOrchestrator
  -> CommandRing / ReceiptRing
  -> ReceiptWriter / TlogWriter
```

Parallel structure:

```text
stream ingress      runs on IO/runtime workers
normalization       runs on CPU workers
slot application    runs on one GPU-owner worker
orchestration       runs in GPU bursts
external commands   run on CPU service workers
receipts/logs       append to JSONL / tlog
```

## Critical Invariants

### 1. Single GPU Mutator

Only one component may mutate GPU slot state.

```text
many producers -> bounded queue -> one SlotApplier -> D_G
```

Do not allow arbitrary threads to write device slots directly.

### 2. Typed Slot Updates Only

Raw events must not enter the GPU state directly.

```text
RawEvent -> TypedEvent -> SlotUpdate -> DeviceSlot
```

Example:

```json
{
  "kind": "slot_update",
  "source": "market_feed",
  "slot": "market.price",
  "dtype": "f32",
  "shape": [3],
  "value": [60934.0, 1597.11, 64.08],
  "observed_at": "2026-06-05T00:00:00Z"
}
```

### 3. Effect-Safe Parallelism

Parallel nodes are valid only when their effects are independent.

```math
writes(A) \cap writes(B) = \emptyset
```

```math
writes(A) \cap reads(B) = \emptyset
```

```math
writes(B) \cap reads(A) = \emptyset
```

### 4. Backpressure Is Explicit

```math
rate_{ingest} \leq rate_{consume}
```

If this fails, apply source-specific policy.

| Data class           | Policy                      |
|----------------------+-----------------------------|
| market price         | latest-wins                 |
| high-frequency ticks | window / sample             |
| receipts             | never drop                  |
| safety events        | priority queue              |
| duplicate facts      | hash + dedupe               |
| embeddings           | batch                       |
| learned edges        | compact / latest-wins merge |

### 5. Receipts Are Append-Only

Every applied update and every orchestrator decision must have a receipt.

```text
SlotUpdateReceipt
CommandReceipt
DeviceReceiptExt
TraceEvent
TlogRecord
```

## Minimal Types

### RawStreamEvent

```rust
pub struct RawStreamEvent {
    pub source: String,
    pub kind: String,
    pub observed_at: String,
    pub payload: serde_json::Value,
    pub hash: String,
}
```

### SlotUpdate

```rust
pub struct SlotUpdate {
    pub source: String,
    pub slot: String,
    pub dtype: SlotDType,
    pub shape: Vec<usize>,
    pub values_f32: Vec<f32>,
    pub observed_at: String,
    pub policy: BackpressurePolicy,
}
```

### StreamReceipt

```rust
pub struct StreamReceipt {
    pub source: String,
    pub event_hash: String,
    pub slot: String,
    pub applied: bool,
    pub dropped: bool,
    pub reason: Option<String>,
}
```

## Minimal Operators / Nodes

Add these as operator-topology nodes, not world-state nodes.

| Node                   | Input                | Output                          |
|------------------------+----------------------+---------------------------------|
| `Stream::Poll`         | stream config        | raw stream events               |
| `Stream::Normalize`    | raw stream event     | typed event                     |
| `Stream::SlotUpdate`   | typed event          | slot update                     |
| `Stream::ApplySlot`    | slot update          | device slot mutation + receipt  |
| `Stream::Backpressure` | queue stats          | drop / batch / compact decision |
| `Event::StreamUpdated` | applied slot receipt | observed event flag             |

## First Sources

### 1. Market Feed

Current path:

```text
State::MarketFeed -> state/market_feed.jsonl -> market.feed
```

Target path:

```text
MarketFeedStream
  -> RawStreamEvent
  -> SlotUpdate(market.price, market.open, market.high, market.low, market.volume)
  -> GPU slots
  -> Event::MarketFeedUpdated
```

### 2. File Tail

```text
FileTail(state/events.jsonl)
  -> RawStreamEvent
  -> TypedEvent
  -> SlotUpdate
```

### 3. External HTTP / SSE / WebSocket

Later:

```text
HTTP/SSE/WebSocket
  -> RawStreamEventQueue
  -> Normalizer
```

## Integration With Quantale Orchestrator

The orchestrator should not be interrupted for every incoming event.

Use micro-batches:

```text
collect updates for window_ms or max_batch_size
apply batch to GPU slots
run orchestrator burst
service external commands if WaitExternal
write receipts
repeat
```

Suggested first defaults:

```json
{
  "stream_window_ms": 50,
  "max_batch_size": 1024,
  "slot_update_queue_capacity": 8192,
  "raw_event_queue_capacity": 8192,
  "latest_wins_slots": [
    "market.price",
    "market.open",
    "market.high",
    "market.low",
    "market.volume"
  ]
}
```

## Current Implementation Gap

Current `UploadQueue` stages via CPU `Vec` and flushes with synchronous `htod_copy`.

That is acceptable for the first implementation.

Future upgrade:

```text
PinnedHostBuffer
cudaMemcpyAsync
CUDA stream
ring-buffered device slots
double buffering
```

## Phases

### Phase 1 — Synchronous Micro-Batch Streamer

- Add `src/streaming.rs`.
- Add `RawStreamEvent`, `SlotUpdate`, `BackpressurePolicy`, `StreamReceipt`.
- Add bounded `std::sync::mpsc` or simple in-process queues.
- Add normalizer for market feed records.
- Add `SlotApplier` that batches updates and calls current synchronous upload path.
- Append stream receipts to `state/events.jsonl` or `state/quantale.tlog`.

Acceptance:

```text
market feed updates can be normalized into slot updates
slot updates are applied in batches
orchestrator can run after each batch
all applied/dropped updates produce receipts
```

### Phase 2 — Parallel CPU Workers

- Add one stream reader worker.
- Add one normalizer worker.
- Keep one GPU slot applier.
- Keep orchestrator ownership explicit.
- Add queue length metrics.
- Add backpressure decisions.

Acceptance:

```text
stream reader continues while GPU orchestrator runs bursts
no direct multi-writer GPU mutation
bounded queues prevent unbounded memory growth
```

### Phase 3 — External Command Worker Pool

- Service independent external commands concurrently when effect-safe.
- Preserve one receipt per command id.
- Preserve deterministic receipt ordering by command id or enqueue time.

Acceptance:

```text
WaitExternal can drain multiple commands
independent IO/process commands run in parallel
receipts return deterministically
```

### Phase 4 — Async Runtime

- Introduce async IO runtime only after Phase 1-3 are stable.
- Candidate: `tokio` for WebSocket/SSE/HTTP streams.
- Keep CUDA worker separate from async IO runtime.

Acceptance:

```text
async stream sources feed bounded queues
orchestrator remains deterministic
shutdown drains queues cleanly
```

### Phase 5 — Async CUDA Upload

- Add pinned host buffers.
- Add CUDA streams.
- Add double-buffered slot updates.
- Overlap H2D copy with previous GPU compute where safe.

Acceptance:

```text
H2D transfer can overlap with non-conflicting GPU work
slot versioning prevents read/write races
fallback synchronous path remains available
```

## Device Slot Versioning

Add version counters per slot:

```text
slot_version[slot] += 1 on apply
receipt records version
orchestrator snapshot records versions read
```

This enables deterministic replay:

```text
receipt says node read market.price@v12 and market.open@v12
```

## Failure Modes

| Failure                 | Handling                                 |
|-------------------------+------------------------------------------|
| queue overflow          | backpressure policy                      |
| malformed event         | reject + receipt                         |
| unknown slot            | reject + receipt                         |
| dtype mismatch          | reject + receipt                         |
| GPU upload failure      | stop applying, emit fatal receipt        |
| external source timeout | timeout receipt, keep orchestrator alive |
| duplicate event         | dedupe or latest-wins                    |
| orchestrator blocked    | preserve queue, apply reset policy       |

## Tests To Add

| Test                                          | Purpose                        |
|-----------------------------------------------+--------------------------------|
| `stream_event_normalizes_market_tick`         | raw tick -> slot updates       |
| `slot_update_rejects_unknown_slot`            | schema safety                  |
| `latest_wins_compacts_market_price`           | backpressure correctness       |
| `receipts_are_emitted_for_dropped_updates`    | auditability                   |
| `single_gpu_applier_owns_device_slots`        | mutation invariant             |
| `orchestrator_runs_after_stream_batch`        | integration                    |
| `parallel_commands_preserve_receipt_identity` | deterministic external service |

## Non-Goals For First Pass

```text
no WebSocket-first rewrite
no Tokio dependency until sync micro-batch works
no multi-writer GPU slot mutation
no direct raw JSON into CUDA state
no unbounded channels
no dropping receipts
```

## Final Target

```text
async sources feed typed slot updates
slot updates mutate GPU state through one owner
quantale topology runs parallel-safe work over current state
external IO returns receipts
receipts update learning/logging/replay
```

Final equation:

```math
Stream_t \oplus State_t \xrightarrow{Q} State_{t+1}, Receipt_t
```
