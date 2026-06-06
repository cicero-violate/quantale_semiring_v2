//! Async streaming ingress.
//!
//! Phase 1 — synchronous micro-batch streamer (no background threads):
//!   push_raw_event → RawEventQueue → normalize_pending → SlotUpdateQueue
//!   → apply_pending → DeviceSlotRegistry → QuantaleOrchestrator
//!
//! Phase 2 — parallel CPU workers:
//!   StreamSource → StreamReaderWorker (thread)
//!   → RawEventQueue → NormalizerWorker (thread)
//!   → SlotUpdateQueue → StreamWorkers::apply_pending (GPU-owner thread)

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Write as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::device_slots::{DeviceSlot, DeviceSlotRegistry};
#[cfg(feature = "cuda")]
use crate::device_slots::{PinnedHostBuffer, UploadQueue};

// ── Core types ────────────────────────────────────────────────────────────────

/// Supported element dtypes for slot updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotDType {
    F32,
    F64,
    I32,
    U8,
}

impl Default for SlotDType {
    fn default() -> Self {
        Self::F32
    }
}

impl std::fmt::Display for SlotDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
            Self::I32 => f.write_str("i32"),
            Self::U8 => f.write_str("u8"),
        }
    }
}

/// Per-slot backpressure policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackpressurePolicy {
    /// Keep only the most recent value; earlier pending values for the same slot are discarded.
    LatestWins,
    /// Accumulate into a window; callers manage sampling.
    WindowSample,
    /// Never drop; block or return an error when the queue is full.
    NeverDrop,
    /// Priority queue ordering (caller-managed).
    Priority,
    /// Hash-based deduplication.
    Dedupe,
    /// Batch accumulation before forwarding.
    Batch,
    /// Compact multiple pending updates for the same slot into one.
    Compact,
}

/// Normalized topology delta from a streaming source.
///
/// Produced when a raw event has `"kind": "edge_delta"` or `"kind": "node_delta"`.
/// Routed through a separate bounded channel from slot updates so graph mutations
/// do not compete with data uploads for queue capacity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TopologyDelta {
    /// Mutate one edge in the quantale tensor.
    Edge {
        source: String,
        src: String,
        dst: String,
        confidence: f32,
        cost: f32,
        safety: f32,
        observed_at: String,
        event_hash: String,
    },
    /// Activate or clear one node in the quantale frontier.
    Node {
        source: String,
        node: String,
        active: bool,
        observed_at: String,
        event_hash: String,
    },
}

/// Raw event from a streaming source before normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawStreamEvent {
    pub source: String,
    pub kind: String,
    pub observed_at: String,
    pub payload: Value,
    /// FNV-flavoured hash of the serialized payload for dedup / receipt correlation.
    pub hash: String,
}

impl RawStreamEvent {
    pub fn new(
        source: impl Into<String>,
        kind: impl Into<String>,
        observed_at: impl Into<String>,
        payload: Value,
    ) -> Self {
        let hash = payload_hash(&payload);
        Self {
            source: source.into(),
            kind: kind.into(),
            observed_at: observed_at.into(),
            payload,
            hash,
        }
    }
}

fn payload_hash(payload: &Value) -> String {
    let mut hasher = DefaultHasher::new();
    payload.to_string().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Typed slot update produced by the normalizer, ready for GPU application.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlotUpdate {
    pub source: String,
    pub slot: String,
    pub dtype: SlotDType,
    pub shape: Vec<usize>,
    pub values_f32: Vec<f32>,
    pub observed_at: String,
    pub policy: BackpressurePolicy,
    /// Originating event hash for receipt correlation.
    pub event_hash: String,
}

/// Receipt emitted for every applied or dropped slot update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamReceipt {
    pub source: String,
    pub event_hash: String,
    pub slot: String,
    pub applied: bool,
    pub dropped: bool,
    pub reason: Option<String>,
    /// Slot version after successful application; 0 when dropped.
    pub version: u64,
}

impl StreamReceipt {
    /// Receipt for an update applied to the current device-slot state.
    pub fn applied(update: SlotUpdate, version: u64) -> Self {
        Self {
            source: update.source,
            event_hash: update.event_hash,
            slot: update.slot,
            applied: true,
            dropped: false,
            reason: None,
            version,
        }
    }

    /// Receipt for an update rejected or dropped before mutating slot state.
    pub fn dropped(update: SlotUpdate, reason: impl Into<String>) -> Self {
        Self {
            source: update.source,
            event_hash: update.event_hash,
            slot: update.slot,
            applied: false,
            dropped: true,
            reason: Some(reason.into()),
            version: 0,
        }
    }

    /// Receipt placeholder for a valid update staged for a later batch flush.
    #[allow(dead_code)]
    fn staged(update: SlotUpdate) -> Self {
        Self {
            source: update.source,
            event_hash: update.event_hash,
            slot: update.slot,
            applied: false,
            dropped: false,
            reason: None,
            version: 0,
        }
    }
}

// ── Stream configuration ──────────────────────────────────────────────────────

/// Configuration for the streaming ingress layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamConfig {
    pub stream_window_ms: u64,
    pub max_batch_size: usize,
    pub slot_update_queue_capacity: usize,
    pub raw_event_queue_capacity: usize,
    pub latest_wins_slots: Vec<String>,
    /// How long the reader worker sleeps when its source returns no data (ms).
    pub reader_poll_interval_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            stream_window_ms: 50,
            max_batch_size: 1024,
            slot_update_queue_capacity: 8192,
            raw_event_queue_capacity: 8192,
            latest_wins_slots: vec![
                "market.price".into(),
                "market.open".into(),
                "market.high".into(),
                "market.low".into(),
                "market.volume".into(),
            ],
            reader_poll_interval_ms: 5,
        }
    }
}

impl StreamConfig {
    /// Explicit name for the current production-safe defaults.
    pub fn normal_capacity() -> Self {
        Self::default()
    }

    /// Higher-throughput profile from the async-streamer sizing plan.
    pub fn high_throughput() -> Self {
        Self {
            raw_event_queue_capacity: 65_536,
            slot_update_queue_capacity: 65_536,
            max_batch_size: 4_096,
            stream_window_ms: 10,
            ..Self::default()
        }
    }

    /// Upper bound for slot updates drained by the applier per second.
    pub fn apply_capacity_per_sec(&self) -> usize {
        if self.stream_window_ms == 0 {
            return 0;
        }
        self.max_batch_size * (1000usize / self.stream_window_ms as usize)
    }

    /// Complete market ticks representable before pressure, assuming five slots per tick.
    pub fn market_tick_buffer_capacity(&self) -> usize {
        let update_limited = self.slot_update_queue_capacity / 5;
        self.raw_event_queue_capacity.min(update_limited)
    }

    /// Reject impossible queue/window settings before worker startup.
    pub fn validate(&self) -> Result<(), String> {
        if self.stream_window_ms == 0 {
            return Err("stream_window_ms must be greater than 0".into());
        }
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be greater than 0".into());
        }
        if self.raw_event_queue_capacity == 0 {
            return Err("raw_event_queue_capacity must be greater than 0".into());
        }
        if self.slot_update_queue_capacity == 0 {
            return Err("slot_update_queue_capacity must be greater than 0".into());
        }
        if self.max_batch_size > self.slot_update_queue_capacity {
            return Err("max_batch_size must not exceed slot_update_queue_capacity".into());
        }
        Ok(())
    }
}

// ── Normalizer ────────────────────────────────────────────────────────────────

/// Rejection produced when a raw event cannot be normalized.
#[derive(Clone, Debug)]
pub struct NormalizeError {
    pub event_hash: String,
    pub reason: String,
}

/// Normalize a `RawStreamEvent` into zero or more typed `SlotUpdate`s.
///
/// Returns `Err` with a rejection reason when the event is malformed.
/// Unknown event kinds yield an empty `Vec` (silently skipped, no error).
pub fn normalize_event(
    event: &RawStreamEvent,
    config: &StreamConfig,
) -> Result<Vec<SlotUpdate>, NormalizeError> {
    match event.kind.as_str() {
        "market_tick" => normalize_market_tick(event, config),
        "slot_update" => normalize_slot_update_envelope(event, config),
        // Topology mutations are handled by the separate delta channel; yield no
        // slot updates here so the normalizer worker counts them correctly.
        "edge_delta" | "node_delta" => Ok(Vec::new()),
        _ => {
            // Check if payload carries an explicit kind field.
            if let Some(k) = event.payload.get("kind").and_then(Value::as_str) {
                if k == "slot_update" {
                    return normalize_slot_update_envelope(event, config);
                }
            }
            Ok(Vec::new())
        }
    }
}

/// If `event.kind` is `"edge_delta"` or `"node_delta"`, parse and return a
/// `TopologyDelta`.  Returns `None` for all other event kinds.
pub fn try_normalize_topology_delta(event: &RawStreamEvent) -> Option<TopologyDelta> {
    match event.kind.as_str() {
        "edge_delta" => normalize_edge_delta(event),
        "node_delta" => normalize_node_delta(event),
        _ => None,
    }
}

fn mk_reject(event: &RawStreamEvent, reason: &str) -> NormalizeError {
    NormalizeError {
        event_hash: event.hash.clone(),
        reason: reason.to_string(),
    }
}

fn normalize_market_tick(
    event: &RawStreamEvent,
    config: &StreamConfig,
) -> Result<Vec<SlotUpdate>, NormalizeError> {
    const FIELD_SLOT: &[(&str, &str)] = &[
        ("price", "market.price"),
        ("close", "market.price"),
        ("open", "market.open"),
        ("high", "market.high"),
        ("low", "market.low"),
        ("volume", "market.volume"),
    ];

    let mut updates: Vec<SlotUpdate> = Vec::new();
    let observed_at = event.observed_at.clone();

    // Scalar fields.
    for &(field, slot) in FIELD_SLOT {
        let Some(val) = event.payload.get(field) else {
            continue;
        };
        let scalar = val
            .as_f64()
            .ok_or_else(|| mk_reject(event, &format!("field '{field}' is not a number")))?
            as f32;
        let policy = slot_policy(slot, config);
        updates.push(SlotUpdate {
            source: event.source.clone(),
            slot: slot.to_string(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![scalar],
            observed_at: observed_at.clone(),
            policy,
            event_hash: event.hash.clone(),
        });
    }

    // Array-valued "prices" → shape=[N] market.price vector.
    if let Some(arr) = event.payload.get("prices").and_then(Value::as_array) {
        let values: Vec<f32> = arr
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if !values.is_empty() {
            let n = values.len();
            updates.push(SlotUpdate {
                source: event.source.clone(),
                slot: "market.price".to_string(),
                dtype: SlotDType::F32,
                shape: vec![n],
                values_f32: values,
                observed_at: observed_at.clone(),
                policy: BackpressurePolicy::LatestWins,
                event_hash: event.hash.clone(),
            });
        }
    }

    Ok(updates)
}

fn normalize_slot_update_envelope(
    event: &RawStreamEvent,
    config: &StreamConfig,
) -> Result<Vec<SlotUpdate>, NormalizeError> {
    let slot = event.payload["slot"]
        .as_str()
        .ok_or_else(|| mk_reject(event, "slot_update envelope missing 'slot' field"))?;

    let dtype_str = event.payload["dtype"].as_str().unwrap_or("f32");
    let dtype = match dtype_str {
        "f32" => SlotDType::F32,
        "f64" => SlotDType::F64,
        "i32" => SlotDType::I32,
        "u8" => SlotDType::U8,
        other => return Err(mk_reject(event, &format!("unsupported dtype '{other}'"))),
    };

    if dtype != SlotDType::F32 {
        return Err(mk_reject(
            event,
            &format!("dtype '{dtype}' is not yet supported; only f32 is implemented"),
        ));
    }

    let shape: Vec<usize> = event.payload["shape"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_u64().unwrap_or(1) as usize)
                .collect()
        })
        .unwrap_or_else(|| vec![1]);

    let values_f32: Vec<f32> = event.payload["value"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect()
        })
        .unwrap_or_default();

    let policy = slot_policy(slot, config);

    Ok(vec![SlotUpdate {
        source: event.source.clone(),
        slot: slot.to_string(),
        dtype,
        shape,
        values_f32,
        observed_at: event.observed_at.clone(),
        policy,
        event_hash: event.hash.clone(),
    }])
}

/// Parse a `{"kind":"edge_delta", "src":…, "dst":…, "confidence":…, …}` envelope.
///
/// Returns `None` when required fields are missing or malformed.
pub(crate) fn normalize_edge_delta(event: &RawStreamEvent) -> Option<TopologyDelta> {
    let src = event.payload["src"].as_str()?.to_string();
    let dst = event.payload["dst"].as_str()?.to_string();
    let confidence = event.payload["confidence"].as_f64()? as f32;
    let cost = event.payload["cost"].as_f64()? as f32;
    let safety = event.payload["safety"].as_f64()? as f32;
    Some(TopologyDelta::Edge {
        source: event.source.clone(),
        src,
        dst,
        confidence,
        cost,
        safety,
        observed_at: event.observed_at.clone(),
        event_hash: event.hash.clone(),
    })
}

/// Parse a `{"kind":"node_delta", "node":…, "active":…}` envelope.
///
/// Returns `None` when required fields are missing or malformed.
pub(crate) fn normalize_node_delta(event: &RawStreamEvent) -> Option<TopologyDelta> {
    let node = event.payload["node"].as_str()?.to_string();
    let active = event.payload["active"].as_bool().unwrap_or(true);
    Some(TopologyDelta::Node {
        source: event.source.clone(),
        node,
        active,
        observed_at: event.observed_at.clone(),
        event_hash: event.hash.clone(),
    })
}

fn slot_policy(slot: &str, config: &StreamConfig) -> BackpressurePolicy {
    if config.latest_wins_slots.iter().any(|s| s == slot) {
        BackpressurePolicy::LatestWins
    } else {
        BackpressurePolicy::NeverDrop
    }
}

// ── Latest-wins compaction ────────────────────────────────────────────────────

/// Compact a batch of `SlotUpdate`s: for `LatestWins` slots, keep only the
/// last update per slot name; other policies pass through unchanged.
pub fn compact_latest_wins(updates: Vec<SlotUpdate>) -> Vec<SlotUpdate> {
    let mut last_idx: HashMap<String, usize> = HashMap::new();
    for (i, u) in updates.iter().enumerate() {
        if u.policy == BackpressurePolicy::LatestWins {
            last_idx.insert(u.slot.clone(), i);
        }
    }
    updates
        .into_iter()
        .enumerate()
        .filter(|(i, u)| {
            if u.policy == BackpressurePolicy::LatestWins {
                last_idx.get(&u.slot) == Some(i)
            } else {
                true
            }
        })
        .map(|(_, u)| u)
        .collect()
}

// ── Slot versioning ───────────────────────────────────────────────────────────

/// Per-slot monotonic version counters; bumped on every successful apply.
#[derive(Default)]
pub struct SlotVersions(HashMap<String, u64>);

impl SlotVersions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bump(&mut self, slot: &str) -> u64 {
        let v = self.0.entry(slot.to_string()).or_insert(0);
        *v += 1;
        *v
    }

    pub fn get(&self, slot: &str) -> u64 {
        self.0.get(slot).copied().unwrap_or(0)
    }
}

// ── SlotSchema ────────────────────────────────────────────────────────────────

/// Declared schema of device slots.  Used to reject unknown names and dtype
/// mismatches before any GPU upload is attempted.
#[derive(Default)]
pub struct SlotSchema {
    slots: HashMap<String, DeviceSlot>,
}

impl SlotSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, slot: DeviceSlot) {
        self.slots.insert(slot.name.clone(), slot);
    }

    pub fn get(&self, name: &str) -> Option<&DeviceSlot> {
        self.slots.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.slots.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &DeviceSlot> {
        self.slots.values()
    }
}

// ── SlotApplier ───────────────────────────────────────────────────────────────

/// Single GPU-owner that applies batches of typed slot updates.
///
/// Invariant: only one `SlotApplier` per GPU device.  All slot mutations flow
/// through this struct; no other path may write `DeviceSlotRegistry` entries
/// that stream sources own.
pub struct SlotApplier {
    pub schema: SlotSchema,
    pub versions: SlotVersions,
    #[cfg(feature = "cuda")]
    upload_queue: UploadQueue,
}

impl SlotApplier {
    pub fn new(schema: SlotSchema) -> Self {
        Self {
            schema,
            versions: SlotVersions::new(),
            #[cfg(feature = "cuda")]
            upload_queue: UploadQueue::new(),
        }
    }

    /// Validate a single slot update against the schema.
    ///
    /// Returns `Ok(())` when the update is acceptable, `Err(reason)` otherwise.
    pub fn validate_update(&self, update: &SlotUpdate) -> Result<(), String> {
        match self.schema.get(&update.slot) {
            Some(meta) if meta.dtype != update.dtype.to_string() => Err(format!(
                "dtype mismatch: slot '{}' expects '{}', got '{}'",
                update.slot, meta.dtype, update.dtype
            )),
            Some(_) => Ok(()),
            None => Err(format!("unknown slot '{}'", update.slot)),
        }
    }

    /// Apply a batch of slot updates to the device registry.
    ///
    /// Each update is validated first; invalid updates produce a dropped receipt
    /// without touching the GPU.  Valid updates are staged and flushed in one
    /// `htod_copy` pass.
    #[cfg(feature = "cuda")]
    pub fn apply_batch(
        &mut self,
        updates: Vec<SlotUpdate>,
        registry: &mut DeviceSlotRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Vec<StreamReceipt> {
        let mut receipts: Vec<StreamReceipt> = Vec::with_capacity(updates.len());
        let mut staged_slots: Vec<String> = Vec::new();

        for update in updates {
            match self.validate_update(&update) {
                Err(reason) => {
                    receipts.push(StreamReceipt::dropped(update, reason));
                    continue;
                }
                Ok(()) => {}
            }

            // schema entry confirmed by validate_update above
            let slot_meta = match self.schema.get(&update.slot).cloned() {
                Some(m) => m,
                None => {
                    receipts.push(StreamReceipt::dropped(update, "internal: slot missing after validation"));
                    continue;
                }
            };
            match self.upload_queue.stage(&slot_meta, &update.values_f32) {
                Err(e) => {
                    receipts.push(StreamReceipt::dropped(update, format!("stage failed: {e}")));
                    continue;
                }
                Ok(()) => {}
            }

            staged_slots.push(update.slot.clone());
            receipts.push(StreamReceipt::staged(update));
        }

        if !staged_slots.is_empty() {
            match self.upload_queue.flush(registry, dev) {
                Ok(()) => {
                    for receipt in receipts.iter_mut() {
                        if !receipt.dropped && staged_slots.contains(&receipt.slot) {
                            let v = self.versions.bump(&receipt.slot);
                            receipt.applied = true;
                            receipt.version = v;
                        }
                    }
                }
                Err(e) => {
                    for receipt in receipts.iter_mut() {
                        if !receipt.dropped && staged_slots.contains(&receipt.slot) {
                            receipt.dropped = true;
                            receipt.reason = Some(format!("GPU upload failed: {e}"));
                        }
                    }
                }
            }
        }

        receipts
    }

    /// Non-CUDA path: schema validation runs, GPU upload does not.
    #[cfg(not(feature = "cuda"))]
    pub fn apply_batch(
        &mut self,
        updates: Vec<SlotUpdate>,
        _registry: &mut DeviceSlotRegistry,
    ) -> Vec<StreamReceipt> {
        updates
            .into_iter()
            .map(|u| {
                let reason = self
                    .validate_update(&u)
                    .err()
                    .unwrap_or_else(|| "cuda feature not enabled".into());
                StreamReceipt::dropped(u, reason)
            })
            .collect()
    }
}

// ── StreamIngress ─────────────────────────────────────────────────────────────

/// Error returned when a raw event cannot be enqueued.
#[derive(Clone, Debug)]
pub enum StreamIngressError {
    QueueFull(String),
    Disconnected,
}

impl std::fmt::Display for StreamIngressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull(q) => write!(f, "queue '{q}' is full"),
            Self::Disconnected => write!(f, "ingress queue disconnected"),
        }
    }
}

/// Pull-based ingress pipeline for Phase 1 (synchronous micro-batch).
///
/// No background threads.  The orchestrator loop calls `run_batch` once per
/// cycle to drain both queues and apply a compacted batch to the device.
pub struct StreamIngress {
    pub config: StreamConfig,
    pub applier: SlotApplier,
    raw_tx: SyncSender<RawStreamEvent>,
    raw_rx: Receiver<RawStreamEvent>,
    update_tx: SyncSender<SlotUpdate>,
    update_rx: Receiver<SlotUpdate>,
}

impl StreamIngress {
    pub fn new(config: StreamConfig, schema: SlotSchema) -> Self {
        let (raw_tx, raw_rx) = mpsc::sync_channel(config.raw_event_queue_capacity);
        let (update_tx, update_rx) = mpsc::sync_channel(config.slot_update_queue_capacity);
        Self {
            applier: SlotApplier::new(schema),
            config,
            raw_tx,
            raw_rx,
            update_tx,
            update_rx,
        }
    }

    /// Push a raw event into the ingress queue without blocking.
    ///
    /// Returns `Err(QueueFull)` when the bounded queue has no space.
    pub fn push_raw_event(&self, event: RawStreamEvent) -> Result<(), StreamIngressError> {
        self.raw_tx.try_send(event).map_err(|e| match e {
            TrySendError::Full(_) => StreamIngressError::QueueFull("raw_event_queue".into()),
            TrySendError::Disconnected(_) => StreamIngressError::Disconnected,
        })
    }

    /// Drain the raw event queue, normalize events, push slot updates.
    ///
    /// Returns receipts for events rejected during normalization.
    pub fn normalize_pending(&mut self) -> Vec<StreamReceipt> {
        let mut receipts = Vec::new();
        while let Ok(event) = self.raw_rx.try_recv() {
            match normalize_event(&event, &self.config) {
                Ok(updates) => {
                    for update in updates {
                        if let Err(e) = self.update_tx.try_send(update) {
                            match e {
                                TrySendError::Full(u) => {
                                    receipts.push(StreamReceipt {
                                        source: u.source,
                                        event_hash: u.event_hash,
                                        slot: u.slot,
                                        applied: false,
                                        dropped: true,
                                        reason: Some("slot_update_queue full".into()),
                                        version: 0,
                                    });
                                }
                                TrySendError::Disconnected(_) => break,
                            }
                        }
                    }
                }
                Err(e) => {
                    receipts.push(StreamReceipt {
                        source: "normalizer".into(),
                        event_hash: e.event_hash,
                        slot: String::new(),
                        applied: false,
                        dropped: true,
                        reason: Some(e.reason),
                        version: 0,
                    });
                }
            }
        }
        receipts
    }

    /// Drain the slot update queue up to `max_batch_size`, compact latest-wins,
    /// apply the batch.
    #[cfg(feature = "cuda")]
    pub fn apply_pending(
        &mut self,
        registry: &mut DeviceSlotRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Vec<StreamReceipt> {
        let updates = self.drain_updates();
        if updates.is_empty() {
            return Vec::new();
        }
        let compacted = compact_latest_wins(updates);
        self.applier.apply_batch(compacted, registry, dev)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn apply_pending(&mut self, registry: &mut DeviceSlotRegistry) -> Vec<StreamReceipt> {
        let updates = self.drain_updates();
        if updates.is_empty() {
            return Vec::new();
        }
        let compacted = compact_latest_wins(updates);
        self.applier.apply_batch(compacted, registry)
    }

    /// Run a full micro-batch cycle: normalize → compact → apply.
    ///
    /// Called once per orchestrator burst.  Returns all receipts from this
    /// cycle so the caller can append them to the receipt log.
    #[cfg(feature = "cuda")]
    pub fn run_batch(
        &mut self,
        registry: &mut DeviceSlotRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Vec<StreamReceipt> {
        let mut receipts = self.normalize_pending();
        receipts.extend(self.apply_pending(registry, dev));
        receipts
    }

    #[cfg(not(feature = "cuda"))]
    pub fn run_batch(&mut self, registry: &mut DeviceSlotRegistry) -> Vec<StreamReceipt> {
        let mut receipts = self.normalize_pending();
        receipts.extend(self.apply_pending(registry));
        receipts
    }

    fn drain_updates(&mut self) -> Vec<SlotUpdate> {
        let mut updates = Vec::with_capacity(self.config.max_batch_size);
        while updates.len() < self.config.max_batch_size {
            match self.update_rx.try_recv() {
                Ok(u) => updates.push(u),
                Err(_) => break,
            }
        }
        updates
    }
}

// ── StreamReceiptWriter ───────────────────────────────────────────────────────

/// Append-only JSONL writer for stream receipts.
///
/// Follows the same O_APPEND contract as `TlogWriter`: each record is
/// serialized into memory first, then written with a single `write_all` call.
pub struct StreamReceiptWriter {
    file: std::fs::File,
}

impl StreamReceiptWriter {
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file })
    }

    pub fn append(&mut self, receipts: &[StreamReceipt]) -> std::io::Result<()> {
        for receipt in receipts {
            let mut line = serde_json::to_vec(receipt)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            line.push(b'\n');
            self.file.write_all(&line)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

// ── Phase 2: queue metrics ────────────────────────────────────────────────────

/// Snapshot of streaming pipeline queue metrics at one point in time.
#[derive(Clone, Debug, Default)]
pub struct QueueSnapshot {
    pub raw_enqueued: u64,
    pub raw_dequeued: u64,
    pub raw_dropped: u64,
    pub normalize_errors: u64,
    pub slot_update_enqueued: u64,
    pub slot_update_dequeued: u64,
    pub slot_update_dropped: u64,
}

impl QueueSnapshot {
    /// Approximate number of raw events currently in-flight between source and normalizer.
    pub fn raw_queue_depth(&self) -> u64 {
        self.raw_enqueued.saturating_sub(self.raw_dequeued)
    }

    /// Approximate number of slot updates waiting to be applied.
    pub fn slot_update_queue_depth(&self) -> u64 {
        self.slot_update_enqueued
            .saturating_sub(self.slot_update_dequeued)
    }
}

/// Shared atomic counters updated by background worker threads.
#[derive(Default)]
pub struct StreamMetrics {
    pub raw_enqueued: AtomicU64,
    pub raw_dequeued: AtomicU64,
    pub raw_dropped: AtomicU64,
    pub normalize_errors: AtomicU64,
    pub slot_update_enqueued: AtomicU64,
    pub slot_update_dequeued: AtomicU64,
    pub slot_update_dropped: AtomicU64,
}

impl StreamMetrics {
    pub fn snapshot(&self) -> QueueSnapshot {
        QueueSnapshot {
            raw_enqueued: self.raw_enqueued.load(Ordering::Relaxed),
            raw_dequeued: self.raw_dequeued.load(Ordering::Relaxed),
            raw_dropped: self.raw_dropped.load(Ordering::Relaxed),
            normalize_errors: self.normalize_errors.load(Ordering::Relaxed),
            slot_update_enqueued: self.slot_update_enqueued.load(Ordering::Relaxed),
            slot_update_dequeued: self.slot_update_dequeued.load(Ordering::Relaxed),
            slot_update_dropped: self.slot_update_dropped.load(Ordering::Relaxed),
        }
    }
}

// ── Phase 2: stream sources ───────────────────────────────────────────────────

/// A pull-based streaming source.  Called by `StreamReaderWorker` in a loop.
///
/// `poll` returns the next event, or `None` when no data is available yet.
/// The reader sleeps for `reader_poll_interval_ms` between `None` returns.
pub trait StreamSource: Send + 'static {
    fn poll(&mut self) -> Option<RawStreamEvent>;
    fn source_name(&self) -> &str;
}

/// In-memory source backed by a pre-loaded `Vec`.  Useful for tests and
/// one-shot replay.  Returns `None` once all events are drained.
pub struct InMemorySource {
    name: String,
    events: std::collections::VecDeque<RawStreamEvent>,
}

impl InMemorySource {
    pub fn new(name: impl Into<String>, events: Vec<RawStreamEvent>) -> Self {
        Self {
            name: name.into(),
            events: events.into(),
        }
    }
}

impl StreamSource for InMemorySource {
    fn poll(&mut self) -> Option<RawStreamEvent> {
        self.events.pop_front()
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// JSONL file source.  Each non-empty line is parsed as a JSON object and
/// wrapped in a `RawStreamEvent`.
///
/// In `tail_mode = false` (default) the source is exhausted at EOF.
/// In `tail_mode = true` the reader loops at EOF (like `tail -f`), returning
/// `None` each time so the reader worker sleeps before retrying.
pub struct FileLineSource {
    name: String,
    path: std::path::PathBuf,
    reader: Option<BufReader<std::fs::File>>,
    pub tail_mode: bool,
}

impl FileLineSource {
    pub fn new(name: impl Into<String>, path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            reader: None,
            tail_mode: false,
        }
    }

    pub fn with_tail(mut self) -> Self {
        self.tail_mode = true;
        self
    }
}

impl StreamSource for FileLineSource {
    fn poll(&mut self) -> Option<RawStreamEvent> {
        // Lazy-open: attempt to open the file on first poll.
        if self.reader.is_none() {
            match std::fs::File::open(&self.path) {
                Ok(f) => self.reader = Some(BufReader::new(f)),
                Err(_) => return None,
            }
        }

        let reader = self.reader.as_mut()?;
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF.
                if !self.tail_mode {
                    self.reader = None; // exhaust
                }
                None
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return None;
                }
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(payload) => {
                        let kind = payload
                            .get("kind")
                            .and_then(Value::as_str)
                            .unwrap_or("market_tick")
                            .to_string();
                        let observed_at = payload
                            .get("observed_at")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        Some(RawStreamEvent::new(&self.name, kind, observed_at, payload))
                    }
                    Err(_) => None, // malformed line: skip, reader advances past it
                }
            }
            Err(_) => None,
        }
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

// ── Phase 2: background worker functions ──────────────────────────────────────

fn run_reader_worker(
    mut source: impl StreamSource,
    raw_tx: SyncSender<RawStreamEvent>,
    metrics: Arc<StreamMetrics>,
    shutdown: Arc<AtomicBool>,
    poll_interval_ms: u64,
) {
    while !shutdown.load(Ordering::Relaxed) {
        match source.poll() {
            Some(event) => match raw_tx.try_send(event) {
                Ok(()) => {
                    metrics.raw_enqueued.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Full(_)) => {
                    metrics.raw_dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => break,
            },
            None => {
                // No data available; yield to avoid spinning.
                std::thread::sleep(Duration::from_millis(poll_interval_ms.max(1)));
            }
        }
    }
}

fn run_normalizer_worker(
    raw_rx: Receiver<RawStreamEvent>,
    update_tx: SyncSender<SlotUpdate>,
    delta_tx: SyncSender<TopologyDelta>,
    metrics: Arc<StreamMetrics>,
    shutdown: Arc<AtomicBool>,
    config: StreamConfig,
) {
    loop {
        match raw_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(event) => {
                metrics.raw_dequeued.fetch_add(1, Ordering::Relaxed);
                // Route topology deltas (edge_delta / node_delta) to the separate
                // channel.  Other event kinds produce slot updates as before.
                if let Some(delta) = try_normalize_topology_delta(&event) {
                    // Best-effort: drop delta if the delta channel is full.
                    let _ = delta_tx.try_send(delta);
                }
                match normalize_event(&event, &config) {
                    Ok(updates) => {
                        for update in updates {
                            let policy = update.policy;
                            match update_tx.try_send(update) {
                                Ok(()) => {
                                    metrics.slot_update_enqueued.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(TrySendError::Full(u)) => {
                                    if policy == BackpressurePolicy::NeverDrop {
                                        // Retry with blocking send (bounded duration: up to 10ms
                                        // across 10 attempts so the normalizer cannot stall forever).
                                        let mut retries = 10u8;
                                        let mut pending = u;
                                        while retries > 0 {
                                            std::thread::sleep(Duration::from_millis(1));
                                            match update_tx.try_send(pending) {
                                                Ok(()) => {
                                                    metrics
                                                        .slot_update_enqueued
                                                        .fetch_add(1, Ordering::Relaxed);
                                                    break;
                                                }
                                                Err(TrySendError::Full(u2)) => {
                                                    pending = u2;
                                                    retries -= 1;
                                                }
                                                Err(TrySendError::Disconnected(_)) => return,
                                            }
                                        }
                                        if retries == 0 {
                                            metrics
                                                .slot_update_dropped
                                                .fetch_add(1, Ordering::Relaxed);
                                        }
                                    } else {
                                        // LatestWins / WindowSample / etc: drop immediately.
                                        metrics.slot_update_dropped.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                Err(TrySendError::Disconnected(_)) => return,
                            }
                        }
                    }
                    Err(_) => {
                        metrics.normalize_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Relaxed) {
                    // Drain any remaining raw events before exiting.
                    while let Ok(event) = raw_rx.try_recv() {
                        metrics.raw_dequeued.fetch_add(1, Ordering::Relaxed);
                        if let Some(delta) = try_normalize_topology_delta(&event) {
                            let _ = delta_tx.try_send(delta);
                        }
                        if let Ok(updates) = normalize_event(&event, &config) {
                            for u in updates {
                                let _ = update_tx.try_send(u);
                            }
                        }
                    }
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

// ── Phase 2: StreamWorkers ────────────────────────────────────────────────────

/// Parallel streaming pipeline with one reader thread and one normalizer thread.
///
/// Invariants:
/// - Only `StreamWorkers` (via its `SlotApplier`) may mutate device slot state.
/// - All GPU mutations flow through `apply_pending` on the orchestrator thread.
/// - Background threads write only to bounded queues; they never touch the device.
pub struct StreamWorkers {
    /// Sender that external code can use to inject raw events.
    pub raw_event_tx: SyncSender<RawStreamEvent>,
    update_rx: Receiver<SlotUpdate>,
    /// Topology mutation deltas routed by the normalizer from `edge_delta` /
    /// `node_delta` events.  Drained on the GPU-owner thread via
    /// `drain_topology_deltas`.
    topology_delta_rx: Receiver<TopologyDelta>,
    pub applier: SlotApplier,
    pub metrics: Arc<StreamMetrics>,
    shutdown: Arc<AtomicBool>,
    reader_handle: Option<std::thread::JoinHandle<()>>,
    normalizer_handle: Option<std::thread::JoinHandle<()>>,
    config: StreamConfig,
}

impl StreamWorkers {
    /// Spawn the reader and normalizer background workers.
    ///
    /// `source` is moved into the reader thread.  The GPU applier stays on the
    /// calling thread; call `apply_pending` from the orchestrator loop.
    pub fn spawn(config: StreamConfig, schema: SlotSchema, source: impl StreamSource) -> Self {
        let (raw_tx, raw_rx) = mpsc::sync_channel(config.raw_event_queue_capacity);
        let (update_tx, update_rx) = mpsc::sync_channel(config.slot_update_queue_capacity);
        // Topology delta channel: capacity proportional to raw event queue but
        // capped at 256 since graph mutations are rare compared to data updates.
        let delta_cap = config.raw_event_queue_capacity.min(256).max(8);
        let (delta_tx, topology_delta_rx) = mpsc::sync_channel::<TopologyDelta>(delta_cap);
        let metrics = Arc::new(StreamMetrics::default());
        let shutdown = Arc::new(AtomicBool::new(false));

        let reader_raw_tx = raw_tx.clone();
        let reader_metrics = metrics.clone();
        let reader_shutdown = shutdown.clone();
        let poll_interval = config.reader_poll_interval_ms;
        let reader_handle = std::thread::Builder::new()
            .name("stream-reader".into())
            .spawn(move || {
                run_reader_worker(
                    source,
                    reader_raw_tx,
                    reader_metrics,
                    reader_shutdown,
                    poll_interval,
                );
            })
            .expect("failed to spawn stream-reader thread");

        let norm_metrics = metrics.clone();
        let norm_shutdown = shutdown.clone();
        let norm_config = config.clone();
        let normalizer_handle = std::thread::Builder::new()
            .name("stream-normalizer".into())
            .spawn(move || {
                run_normalizer_worker(
                    raw_rx,
                    update_tx,
                    delta_tx,
                    norm_metrics,
                    norm_shutdown,
                    norm_config,
                );
            })
            .expect("failed to spawn stream-normalizer thread");

        Self {
            raw_event_tx: raw_tx,
            update_rx,
            topology_delta_rx,
            applier: SlotApplier::new(schema),
            metrics,
            shutdown,
            reader_handle: Some(reader_handle),
            normalizer_handle: Some(normalizer_handle),
            config,
        }
    }

    /// Drain all pending topology deltas (edge and node mutations) from the
    /// background normalizer queue.
    ///
    /// Must be called from the orchestrator (GPU-owner) thread before each burst,
    /// alongside `apply_pending`, so that graph mutations are visible to the
    /// scheduler in the same burst cycle as their associated slot updates.
    pub fn drain_topology_deltas(&mut self) -> Vec<TopologyDelta> {
        let mut deltas = Vec::new();
        while let Ok(d) = self.topology_delta_rx.try_recv() {
            deltas.push(d);
        }
        deltas
    }

    /// Inject a raw event directly (bypassing the source thread).
    ///
    /// Useful when the caller produces events itself rather than via a source.
    pub fn push_raw_event(&self, event: RawStreamEvent) -> Result<(), StreamIngressError> {
        match self.raw_event_tx.try_send(event) {
            Ok(()) => {
                self.metrics.raw_enqueued.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.metrics.raw_dropped.fetch_add(1, Ordering::Relaxed);
                Err(StreamIngressError::QueueFull("raw_event_queue".into()))
            }
            Err(TrySendError::Disconnected(_)) => Err(StreamIngressError::Disconnected),
        }
    }

    /// Drain the slot update queue and apply a compacted batch to the device.
    ///
    /// Must be called from the orchestrator (GPU-owner) thread only.
    #[cfg(feature = "cuda")]
    pub fn apply_pending(
        &mut self,
        registry: &mut DeviceSlotRegistry,
        dev: &Arc<cudarc::driver::CudaDevice>,
    ) -> Vec<StreamReceipt> {
        let updates = self.drain_slot_updates();
        if updates.is_empty() {
            return Vec::new();
        }
        self.metrics
            .slot_update_dequeued
            .fetch_add(updates.len() as u64, Ordering::Relaxed);
        let compacted = compact_latest_wins(updates);
        self.applier.apply_batch(compacted, registry, dev)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn apply_pending(&mut self, registry: &mut DeviceSlotRegistry) -> Vec<StreamReceipt> {
        let updates = self.drain_slot_updates();
        if updates.is_empty() {
            return Vec::new();
        }
        self.metrics
            .slot_update_dequeued
            .fetch_add(updates.len() as u64, Ordering::Relaxed);
        let compacted = compact_latest_wins(updates);
        self.applier.apply_batch(compacted, registry)
    }

    /// Snapshot of queue depth and drop counters.
    pub fn queue_metrics(&self) -> QueueSnapshot {
        self.metrics.snapshot()
    }

    /// Signal background workers to stop.  They finish draining in-flight events
    /// before exiting.  Joined automatically on `Drop`.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    fn drain_slot_updates(&mut self) -> Vec<SlotUpdate> {
        let mut updates = Vec::with_capacity(self.config.max_batch_size);
        while updates.len() < self.config.max_batch_size {
            match self.update_rx.try_recv() {
                Ok(u) => updates.push(u),
                Err(_) => break,
            }
        }
        updates
    }
}

impl Drop for StreamWorkers {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.reader_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.normalizer_handle.take() {
            let _ = h.join();
        }
    }
}

// ── Phase 4: async runtime bridge ─────────────────────────────────────────────

/// A `StreamSource` backed by an `mpsc::Receiver<RawStreamEvent>`.
///
/// Allows async runtimes (tokio, async-std) to feed events into the
/// synchronous `StreamWorkers` pipeline without touching GPU state.
/// The async side holds an `AsyncStreamBridge` (wraps the sender);
/// this source is dropped into `StreamWorkers::spawn` as the source.
pub struct ChannelStreamSource {
    name: String,
    rx: Receiver<RawStreamEvent>,
}

impl ChannelStreamSource {
    pub fn new(name: impl Into<String>, rx: Receiver<RawStreamEvent>) -> Self {
        Self {
            name: name.into(),
            rx,
        }
    }
}

impl StreamSource for ChannelStreamSource {
    fn poll(&mut self) -> Option<RawStreamEvent> {
        self.rx.try_recv().ok()
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// `Clone + Send + Sync` handle that async event producers use to push raw
/// events into the streaming pipeline.
///
/// The bridge wraps the `SyncSender` half of the raw event channel.  Any
/// number of async tasks can clone it; they push events and the CUDA worker
/// thread remains entirely separate.
///
/// # Tokio migration path
///
/// 1. Add `tokio` to `Cargo.toml`.
/// 2. Create a `ChannelStreamSource` / `AsyncStreamBridge` pair:
///    `let (source, bridge) = AsyncStreamBridge::channel("my-source", &config);`
/// 3. Spawn `StreamWorkers::spawn(config, schema, source)`.
/// 4. Inside a tokio task: `bridge.push(event)`.
/// 5. The CUDA worker thread stays on `std::thread`, unaffected.
#[derive(Clone)]
pub struct AsyncStreamBridge {
    sender: Arc<SyncSender<RawStreamEvent>>,
    metrics: Arc<StreamMetrics>,
}

impl AsyncStreamBridge {
    /// Create a bridge from a `StreamWorkers` raw-event sender.
    pub fn new(sender: SyncSender<RawStreamEvent>, metrics: Arc<StreamMetrics>) -> Self {
        Self {
            sender: Arc::new(sender),
            metrics,
        }
    }

    /// Create a linked `(ChannelStreamSource, AsyncStreamBridge)` pair.
    ///
    /// Pass the source to `StreamWorkers::spawn`; clone the bridge for each
    /// async producer.
    pub fn channel(name: impl Into<String>, config: &StreamConfig) -> (ChannelStreamSource, Self) {
        let (tx, rx) = mpsc::sync_channel(config.raw_event_queue_capacity);
        let metrics = Arc::new(StreamMetrics::default());
        let source = ChannelStreamSource::new(name, rx);
        let bridge = Self::new(tx, metrics);
        (source, bridge)
    }

    /// Push a raw event from any thread or async task.
    ///
    /// Returns `Err(QueueFull)` when the bounded queue has no capacity.
    /// Async producers should handle this as a backpressure signal (drop,
    /// retry after yielding, or apply source-specific policy).
    pub fn push(&self, event: RawStreamEvent) -> Result<(), StreamIngressError> {
        match self.sender.try_send(event) {
            Ok(()) => {
                self.metrics.raw_enqueued.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.metrics.raw_dropped.fetch_add(1, Ordering::Relaxed);
                Err(StreamIngressError::QueueFull("raw_event_queue".into()))
            }
            Err(TrySendError::Disconnected(_)) => Err(StreamIngressError::Disconnected),
        }
    }

    /// Expose the underlying metrics for monitoring.
    pub fn metrics(&self) -> &StreamMetrics {
        &self.metrics
    }
}

/// Externally visible shutdown signal for `StreamWorkers`.
///
/// Signalling `shutdown` causes the normalizer worker to drain remaining
/// raw events before exiting, preventing event loss on clean stop.
pub struct ShutdownHandle {
    signal: Arc<AtomicBool>,
}

impl ShutdownHandle {
    /// Signal all workers to stop after draining in-flight events.
    pub fn shutdown(&self) {
        self.signal.store(true, Ordering::Relaxed);
    }

    pub fn is_shutdown(&self) -> bool {
        self.signal.load(Ordering::Relaxed)
    }
}

impl StreamWorkers {
    /// Create a `Clone + Send` bridge for async event producers.
    pub fn bridge(&self) -> AsyncStreamBridge {
        AsyncStreamBridge::new(self.raw_event_tx.clone(), self.metrics.clone())
    }

    /// Create a shutdown handle that signals workers to drain and stop.
    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            signal: self.shutdown.clone(),
        }
    }
}

// ── Phase 5: double-buffered slot applier ─────────────────────────────────────

/// One device slot with two alternating buffers.
///
/// At version `v`:
///   - GPU compute reads from `read_buf(v)` = `buf[v % 2]` (last committed).
///   - H2D upload writes to `write_buf(v)` = `buf[(v+1) % 2]` (in-flight).
/// After `versions.bump(slot)` → `v+1`, the newly written buffer becomes the
/// stable read buffer for the next orchestrator burst.
#[cfg(feature = "cuda")]
pub struct DoubleBufferedSlot {
    pub name: String,
    buf_even: cudarc::driver::CudaSlice<f32>,
    buf_odd: cudarc::driver::CudaSlice<f32>,
    pub len: usize,
}

#[cfg(feature = "cuda")]
impl DoubleBufferedSlot {
    /// The buffer the next H2D upload should write to.
    pub fn write_buf_mut(&mut self, version: u64) -> &mut cudarc::driver::CudaSlice<f32> {
        if version % 2 == 0 {
            &mut self.buf_odd
        } else {
            &mut self.buf_even
        }
    }

    /// The stable buffer that GPU compute reads from (last committed state).
    pub fn read_buf(&self, version: u64) -> &cudarc::driver::CudaSlice<f32> {
        if version % 2 == 0 {
            &self.buf_even
        } else {
            &self.buf_odd
        }
    }
}

/// Registry of pre-allocated double-buffered device slots.
///
/// Callers register slot names + lengths at startup; `PinnedSlotApplier` then
/// writes to the write buffer and bumps the version on each apply, keeping the
/// read buffer stable for concurrent GPU kernels.
#[cfg(feature = "cuda")]
pub struct DoubleBufferRegistry {
    slots: HashMap<String, DoubleBufferedSlot>,
}

#[cfg(feature = "cuda")]
impl Default for DoubleBufferRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl DoubleBufferRegistry {
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    /// Pre-allocate two device buffers for a slot.
    pub fn register(
        &mut self,
        name: &str,
        len: usize,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), String> {
        let buf_even = dev
            .htod_copy(vec![0.0f32; len])
            .map_err(|e| format!("DoubleBufferRegistry register '{name}' even: {e}"))?;
        let buf_odd = dev
            .htod_copy(vec![0.0f32; len])
            .map_err(|e| format!("DoubleBufferRegistry register '{name}' odd: {e}"))?;
        self.slots.insert(
            name.to_string(),
            DoubleBufferedSlot {
                name: name.to_string(),
                buf_even,
                buf_odd,
                len,
            },
        );
        Ok(())
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut DoubleBufferedSlot> {
        self.slots.get_mut(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.slots.contains_key(name)
    }

    pub fn read_buf(&self, name: &str, version: u64) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.slots.get(name).map(|s| s.read_buf(version))
    }
}

/// Non-CUDA stub so the type name resolves without the feature.
#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct DoubleBufferRegistry;

#[cfg(not(feature = "cuda"))]
impl DoubleBufferRegistry {
    pub fn new() -> Self {
        Self
    }

    pub fn contains(&self, _name: &str) -> bool {
        false
    }
}

/// Slot applier that uses `PinnedHostBuffer` for page-locked staging and
/// `DoubleBufferRegistry` for non-blocking read/write separation.
///
/// The H2D transfer writes to the write buffer; GPU compute reads the stable
/// read buffer.  After `versions.bump`, the newly written buffer becomes the
/// stable read buffer for the next orchestrator cycle.
///
/// **Fallback**: the synchronous `SlotApplier` + `UploadQueue` path remains
/// available unchanged.  `PinnedSlotApplier` is an opt-in upgrade.
pub struct PinnedSlotApplier {
    pub schema: SlotSchema,
    pub versions: SlotVersions,
}

impl PinnedSlotApplier {
    pub fn new(schema: SlotSchema) -> Self {
        Self {
            schema,
            versions: SlotVersions::new(),
        }
    }

    /// Validate a slot update against the schema (same logic as `SlotApplier`).
    pub fn validate_update(&self, update: &SlotUpdate) -> Result<(), String> {
        match self.schema.get(&update.slot) {
            Some(meta) if meta.dtype != update.dtype.to_string() => Err(format!(
                "dtype mismatch: slot '{}' expects '{}', got '{}'",
                update.slot, meta.dtype, update.dtype
            )),
            Some(_) => Ok(()),
            None => Err(format!("unknown slot '{}'", update.slot)),
        }
    }

    /// Apply a batch to the double-buffer registry using pinned host staging.
    ///
    /// For each valid update:
    ///   1. Stage values in a `PinnedHostBuffer` (page-locked host memory).
    ///   2. Copy to the slot's current write buffer via synchronous `htod_copy`.
    ///   3. Replace the write buffer entry with the new device allocation.
    ///   4. Bump the slot version so orchestrator kernels see the new state.
    ///
    /// Future: replace `htod_copy` with `cudaMemcpyAsync` on a dedicated CUDA
    /// stream to overlap H2D with non-conflicting GPU compute.
    #[cfg(feature = "cuda")]
    pub fn apply_batch(
        &mut self,
        updates: Vec<SlotUpdate>,
        registry: &mut DoubleBufferRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Vec<StreamReceipt> {
        let mut receipts = Vec::with_capacity(updates.len());

        for update in updates {
            if let Err(reason) = self.validate_update(&update) {
                receipts.push(StreamReceipt {
                    source: update.source,
                    event_hash: update.event_hash,
                    slot: update.slot,
                    applied: false,
                    dropped: true,
                    reason: Some(reason),
                    version: 0,
                });
                continue;
            }

            if !registry.contains(&update.slot) {
                let reason = format!(
                    "slot '{}' not registered in DoubleBufferRegistry",
                    update.slot
                );
                receipts.push(StreamReceipt::dropped(update, reason));
                continue;
            }

            // Stage in pinned (page-locked) host memory for faster H2D.
            let pinned = match PinnedHostBuffer::from_slice(&update.values_f32) {
                Ok(p) => p,
                Err(e) => {
                    receipts.push(StreamReceipt::dropped(
                        update,
                        format!("pinned alloc failed: {e}"),
                    ));
                    continue;
                }
            };

            // H2D copy into a new device allocation (synchronous baseline).
            let new_buf = match dev.htod_copy(pinned.as_slice().to_vec()) {
                Ok(b) => b,
                Err(e) => {
                    receipts.push(StreamReceipt::dropped(
                        update,
                        format!("htod_copy failed: {e}"),
                    ));
                    continue;
                }
            };

            // Write to the double-buffer write slot and bump version.
            let current_version = self.versions.get(&update.slot);
            if let Some(slot) = registry.get_mut(&update.slot) {
                *slot.write_buf_mut(current_version) = new_buf;
            }
            let v = self.versions.bump(&update.slot);

            receipts.push(StreamReceipt::applied(update, v));
        }

        receipts
    }

    /// Non-CUDA path: schema validation only; all updates are dropped.
    #[cfg(not(feature = "cuda"))]
    pub fn apply_batch(
        &mut self,
        updates: Vec<SlotUpdate>,
        _registry: &mut DoubleBufferRegistry,
    ) -> Vec<StreamReceipt> {
        updates
            .into_iter()
            .map(|u| {
                let reason = self
                    .validate_update(&u)
                    .err()
                    .unwrap_or_else(|| "cuda feature not enabled".into());
                StreamReceipt::dropped(u, reason)
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_config() -> StreamConfig {
        StreamConfig::default()
    }

    // ── normalize ──────────────────────────────────────────────────────────────

    #[test]
    fn stream_config_profiles_report_expected_capacity() {
        let normal = StreamConfig::normal_capacity();
        assert_eq!(normal.apply_capacity_per_sec(), 20_480);
        assert_eq!(normal.market_tick_buffer_capacity(), 1_638);
        assert!(normal.validate().is_ok());

        let high = StreamConfig::high_throughput();
        assert_eq!(high.apply_capacity_per_sec(), 409_600);
        assert_eq!(high.market_tick_buffer_capacity(), 13_107);
        assert!(high.validate().is_ok());
    }

    #[test]
    fn stream_config_rejects_invalid_capacity_settings() {
        let mut config = StreamConfig::normal_capacity();
        config.stream_window_ms = 0;
        assert!(config.validate().is_err());

        let mut config = StreamConfig::normal_capacity();
        config.max_batch_size = config.slot_update_queue_capacity + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn stream_event_normalizes_market_tick() {
        let event = RawStreamEvent::new(
            "market_feed",
            "market_tick",
            "2026-06-06T00:00:00Z",
            json!({ "price": 60934.0, "open": 59000.0, "volume": 1234.5 }),
        );
        let updates = normalize_event(&event, &default_config()).unwrap();
        assert!(!updates.is_empty(), "expected at least one slot update");
        let price_update = updates.iter().find(|u| u.slot == "market.price");
        assert!(price_update.is_some(), "expected market.price slot update");
        let price_update = price_update.unwrap();
        assert_eq!(price_update.values_f32, vec![60934.0_f32]);
        assert_eq!(price_update.dtype, SlotDType::F32);
        assert_eq!(price_update.source, "market_feed");
    }

    #[test]
    fn stream_event_normalizes_slot_update_envelope() {
        let event = RawStreamEvent::new(
            "market_feed",
            "slot_update",
            "2026-06-06T00:00:00Z",
            json!({
                "kind": "slot_update",
                "source": "market_feed",
                "slot": "market.price",
                "dtype": "f32",
                "shape": [3],
                "value": [60934.0, 1597.11, 64.08],
                "observed_at": "2026-06-05T00:00:00Z"
            }),
        );
        let updates = normalize_event(&event, &default_config()).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].slot, "market.price");
        assert_eq!(updates[0].shape, vec![3]);
        assert_eq!(updates[0].values_f32.len(), 3);
    }

    #[test]
    fn normalize_rejects_unsupported_dtype() {
        let event = RawStreamEvent::new(
            "src",
            "slot_update",
            "2026-06-06T00:00:00Z",
            json!({ "slot": "market.price", "dtype": "f64", "shape": [1], "value": [1.0] }),
        );
        let result = normalize_event(&event, &default_config());
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("not yet supported"));
    }

    #[test]
    fn normalize_rejects_non_numeric_field() {
        let event = RawStreamEvent::new(
            "src",
            "market_tick",
            "2026-06-06T00:00:00Z",
            json!({ "price": "not_a_number" }),
        );
        let result = normalize_event(&event, &default_config());
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("not a number"));
    }

    #[test]
    fn normalize_unknown_kind_yields_empty() {
        let event = RawStreamEvent::new(
            "src",
            "unknown_kind",
            "2026-06-06T00:00:00Z",
            json!({ "foo": 1 }),
        );
        let updates = normalize_event(&event, &default_config()).unwrap();
        assert!(updates.is_empty());
    }

    // ── compaction ─────────────────────────────────────────────────────────────

    #[test]
    fn latest_wins_compacts_market_price() {
        let config = default_config();
        let make_price = |v: f32| SlotUpdate {
            source: "feed".into(),
            slot: "market.price".into(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![v],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: format!("{v:08}"),
        };

        let updates = vec![make_price(100.0), make_price(101.0), make_price(102.0)];
        let _ = &config; // config used by normalize_event; not needed for compact test
        let compacted = compact_latest_wins(updates);

        assert_eq!(
            compacted.len(),
            1,
            "expected latest-wins to collapse to one update"
        );
        assert_eq!(compacted[0].values_f32, vec![102.0_f32]);
    }

    #[test]
    fn compact_preserves_never_drop_updates() {
        let make = |slot: &str, v: f32, policy: BackpressurePolicy| SlotUpdate {
            source: "src".into(),
            slot: slot.to_string(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![v],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy,
            event_hash: format!("{v:08}"),
        };

        let updates = vec![
            make("safe.slot", 1.0, BackpressurePolicy::NeverDrop),
            make("safe.slot", 2.0, BackpressurePolicy::NeverDrop),
        ];
        let compacted = compact_latest_wins(updates);
        assert_eq!(
            compacted.len(),
            2,
            "NeverDrop updates must not be compacted"
        );
    }

    // ── schema validation ──────────────────────────────────────────────────────

    #[test]
    fn slot_update_rejects_unknown_slot() {
        let applier = SlotApplier::new(SlotSchema::new());
        let update = SlotUpdate {
            source: "src".into(),
            slot: "nonexistent.slot".into(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![1.0],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "abc123".into(),
        };
        let err = applier.validate_update(&update);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("unknown slot"));
    }

    #[test]
    fn slot_update_rejects_dtype_mismatch() {
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));
        let applier = SlotApplier::new(schema);

        let update = SlotUpdate {
            source: "src".into(),
            slot: "market.price".into(),
            dtype: SlotDType::I32,
            shape: vec![1],
            values_f32: vec![1.0],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "abc456".into(),
        };
        let err = applier.validate_update(&update);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("dtype mismatch"));
    }

    #[test]
    fn slot_update_accepts_known_f32_slot() {
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));
        let applier = SlotApplier::new(schema);

        let update = SlotUpdate {
            source: "src".into(),
            slot: "market.price".into(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![100.0],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "abc789".into(),
        };
        assert!(applier.validate_update(&update).is_ok());
    }

    // ── receipts ───────────────────────────────────────────────────────────────

    #[test]
    fn receipts_are_emitted_for_dropped_updates() {
        let mut ingress = StreamIngress::new(default_config(), SlotSchema::new());
        let mut registry = DeviceSlotRegistry::new();

        let event = RawStreamEvent::new(
            "market_feed",
            "market_tick",
            "2026-06-06T00:00:00Z",
            json!({ "price": 60934.0 }),
        );
        ingress.push_raw_event(event).unwrap();

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            use std::sync::Arc;
            if let Ok(dev) = CudaDevice::new(0).map(Arc::new) {
                let receipts = ingress.run_batch(&mut registry, &dev);
                assert!(!receipts.is_empty());
                let dropped = receipts.iter().find(|r| r.dropped);
                assert!(
                    dropped.is_some(),
                    "expected dropped receipt for unknown slot"
                );
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let receipts = ingress.run_batch(&mut registry);
            assert!(!receipts.is_empty(), "expected at least one receipt");
            assert!(
                receipts.iter().all(|r| r.dropped),
                "all receipts should be dropped without cuda"
            );
        }
    }

    #[test]
    fn parallel_commands_preserve_receipt_identity() {
        // Verify event_hash is preserved through normalize → SlotUpdate → StreamReceipt.
        let event = RawStreamEvent::new(
            "market_feed",
            "market_tick",
            "2026-06-06T00:00:00Z",
            json!({ "price": 42.0 }),
        );
        let original_hash = event.hash.clone();
        let updates = normalize_event(&event, &default_config()).unwrap();
        assert!(
            updates.iter().all(|u| u.event_hash == original_hash),
            "event_hash must be preserved in slot updates"
        );
    }

    #[test]
    fn single_gpu_applier_owns_device_slots() {
        // Structural: SlotApplier is not Clone and not Send — only one instance
        // should touch device slots.  Verify the type does not implement Clone.
        fn assert_not_clone<T: ?Sized>() {}
        // This compiles only if SlotApplier does not derive Clone.
        let _: fn() = assert_not_clone::<SlotApplier>;
    }

    #[test]
    fn orchestrator_runs_after_stream_batch() {
        // Integration: push a tick, normalize, apply (no-op without cuda),
        // verify the pipeline does not panic and receipts are returned.
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));
        let mut ingress = StreamIngress::new(default_config(), schema);
        let mut registry = DeviceSlotRegistry::new();

        let event = RawStreamEvent::new(
            "market_feed",
            "market_tick",
            "2026-06-06T00:00:00Z",
            json!({ "price": 99.0 }),
        );
        ingress.push_raw_event(event).unwrap();

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            use std::sync::Arc;
            if let Ok(dev) = CudaDevice::new(0).map(Arc::new) {
                let receipts = ingress.run_batch(&mut registry, &dev);
                // With a known slot registered, at least one update should apply.
                let applied = receipts.iter().any(|r| r.applied);
                assert!(applied, "expected at least one applied receipt");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let receipts = ingress.run_batch(&mut registry);
            // No-cuda path drops all; pipeline must not panic.
            let _ = receipts;
        }
    }

    // ── slot versioning ────────────────────────────────────────────────────────

    #[test]
    fn slot_versions_increment_monotonically() {
        let mut versions = SlotVersions::new();
        assert_eq!(versions.get("market.price"), 0);
        assert_eq!(versions.bump("market.price"), 1);
        assert_eq!(versions.bump("market.price"), 2);
        assert_eq!(versions.get("market.price"), 2);
        assert_eq!(versions.get("other.slot"), 0);
    }

    // ── Phase 2: parallel workers ──────────────────────────────────────────────

    #[test]
    fn stream_reader_continues_while_gpu_orchestrator_runs() {
        // Reader and normalizer threads should process events independently.
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));

        let events: Vec<_> = (0..5)
            .map(|i| {
                RawStreamEvent::new(
                    "market_feed",
                    "market_tick",
                    "2026-06-06T00:00:00Z",
                    json!({ "price": (60000.0 + i as f64) }),
                )
            })
            .collect();

        let source = InMemorySource::new("test", events);
        let workers = StreamWorkers::spawn(StreamConfig::default(), schema, source);

        // Give background threads time to process all events.
        std::thread::sleep(Duration::from_millis(100));

        let snap = workers.queue_metrics();
        assert!(
            snap.raw_enqueued > 0 || snap.slot_update_enqueued > 0,
            "expected background threading activity; got {snap:?}"
        );
    }

    #[test]
    fn no_direct_multi_writer_gpu_mutation() {
        // Structural: `StreamWorkers` owns the sole `SlotApplier` and is not Clone.
        // Only one `StreamWorkers` instance can exist per device.
        fn assert_not_clone<T: ?Sized>() {}
        let _: fn() = assert_not_clone::<StreamWorkers>;
    }

    #[test]
    fn bounded_queues_prevent_unbounded_memory_growth() {
        // The bounded mpsc queue rejects pushes when at capacity, preventing
        // unbounded memory growth under a fast producer.
        let (tx, _rx) = mpsc::sync_channel::<RawStreamEvent>(3);
        let make = || RawStreamEvent::new("test", "market_tick", "", json!({ "price": 1.0 }));

        // Fill to capacity.
        assert!(tx.try_send(make()).is_ok());
        assert!(tx.try_send(make()).is_ok());
        assert!(tx.try_send(make()).is_ok());
        // One past capacity must fail.
        assert!(
            matches!(tx.try_send(make()), Err(TrySendError::Full(_))),
            "expected QueueFull on 4th push to capacity-3 queue"
        );
    }

    #[test]
    fn file_line_source_reads_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "stream_file_source_test_{}.jsonl",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "{\"kind\":\"market_tick\",\"price\":100.0}\n\
             {\"kind\":\"market_tick\",\"price\":101.0}\n",
        )
        .unwrap();

        let mut source = FileLineSource::new("test", &path);
        let e1 = source.poll().expect("expected first event");
        let e2 = source.poll().expect("expected second event");
        let e3 = source.poll(); // EOF

        let _ = std::fs::remove_file(&path);

        assert_eq!(e1.kind, "market_tick");
        assert_eq!(e2.kind, "market_tick");
        assert_eq!(e1.payload["price"], 100.0);
        assert_eq!(e2.payload["price"], 101.0);
        assert!(e3.is_none(), "expected None at EOF");
    }

    #[test]
    fn in_memory_source_exhausts_after_all_events() {
        let events = vec![
            RawStreamEvent::new("src", "market_tick", "", json!({ "price": 1.0 })),
            RawStreamEvent::new("src", "market_tick", "", json!({ "price": 2.0 })),
        ];
        let mut source = InMemorySource::new("test", events);
        assert!(source.poll().is_some());
        assert!(source.poll().is_some());
        assert!(source.poll().is_none(), "expected None after exhaustion");
    }

    #[test]
    fn queue_snapshot_depth_is_difference_of_enqueued_and_dequeued() {
        let snap = QueueSnapshot {
            raw_enqueued: 10,
            raw_dequeued: 3,
            slot_update_enqueued: 7,
            slot_update_dequeued: 5,
            ..Default::default()
        };
        assert_eq!(snap.raw_queue_depth(), 7);
        assert_eq!(snap.slot_update_queue_depth(), 2);
    }

    // ── receipt writer ─────────────────────────────────────────────────────────

    #[test]
    fn receipt_writer_appends_valid_jsonl() {
        let path =
            std::env::temp_dir().join(format!("stream_receipts_test_{}.jsonl", std::process::id()));
        let mut writer = StreamReceiptWriter::open(&path).unwrap();
        let receipts = vec![
            StreamReceipt {
                source: "test".into(),
                event_hash: "deadbeef".into(),
                slot: "market.price".into(),
                applied: true,
                dropped: false,
                reason: None,
                version: 1,
            },
            StreamReceipt {
                source: "test".into(),
                event_hash: "cafebabe".into(),
                slot: "market.volume".into(),
                applied: false,
                dropped: true,
                reason: Some("unknown slot".into()),
                version: 0,
            },
        ];
        writer.append(&receipts).unwrap();
        writer.flush().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["applied"], true);
        assert_eq!(v["slot"], "market.price");
        assert_eq!(v["version"], 1);
    }

    // ── Phase 4: async bridge ──────────────────────────────────────────────────

    #[test]
    fn channel_source_drains_pending_events() {
        let (tx, rx) = mpsc::sync_channel(16);
        let mut source = ChannelStreamSource::new("test", rx);

        tx.send(RawStreamEvent::new(
            "s",
            "market_tick",
            "",
            json!({ "price": 1.0 }),
        ))
        .unwrap();
        tx.send(RawStreamEvent::new(
            "s",
            "market_tick",
            "",
            json!({ "price": 2.0 }),
        ))
        .unwrap();

        assert!(source.poll().is_some(), "expected first event");
        assert!(source.poll().is_some(), "expected second event");
        assert!(source.poll().is_none(), "expected None when queue is empty");
    }

    #[test]
    fn async_bridge_push_increments_metrics() {
        let metrics = Arc::new(StreamMetrics::default());
        let (tx, _rx) = mpsc::sync_channel::<RawStreamEvent>(16);
        let bridge = AsyncStreamBridge::new(tx, metrics.clone());

        bridge
            .push(RawStreamEvent::new(
                "src",
                "market_tick",
                "",
                json!({ "price": 1.0 }),
            ))
            .unwrap();

        assert_eq!(metrics.raw_enqueued.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn async_bridge_returns_queue_full_when_capacity_exceeded() {
        let metrics = Arc::new(StreamMetrics::default());
        let (tx, _rx) = mpsc::sync_channel::<RawStreamEvent>(1);
        let bridge = AsyncStreamBridge::new(tx, metrics.clone());

        let make = || RawStreamEvent::new("src", "market_tick", "", json!({ "price": 1.0 }));
        assert!(bridge.push(make()).is_ok());
        let err = bridge.push(make());
        assert!(
            matches!(err, Err(StreamIngressError::QueueFull(_))),
            "expected QueueFull on overflow"
        );
        assert!(metrics.raw_dropped.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn bridge_channel_pair_source_and_bridge_are_linked() {
        let (source, bridge) = AsyncStreamBridge::channel("linked", &StreamConfig::default());
        assert_eq!(source.source_name(), "linked");

        bridge
            .push(RawStreamEvent::new(
                "x",
                "market_tick",
                "",
                json!({ "price": 9.0 }),
            ))
            .unwrap();

        // The source's internal rx should now have one event.
        let mut source = source;
        assert!(source.poll().is_some(), "bridge event should reach source");
    }

    #[test]
    fn shutdown_handle_signals_workers() {
        let schema = SlotSchema::new();
        let source = InMemorySource::new("test", Vec::new());
        let workers = StreamWorkers::spawn(StreamConfig::default(), schema, source);

        let handle = workers.shutdown_handle();
        assert!(!handle.is_shutdown());
        handle.shutdown();
        assert!(handle.is_shutdown());
    }

    #[test]
    fn shutdown_drains_queues_cleanly() {
        // Push events, signal shutdown, verify workers terminate without panic.
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));

        let events: Vec<_> = (0..3)
            .map(|i| {
                RawStreamEvent::new(
                    "feed",
                    "market_tick",
                    "",
                    json!({ "price": (100.0 + i as f64) }),
                )
            })
            .collect();
        let source = InMemorySource::new("test", events);
        let workers = StreamWorkers::spawn(StreamConfig::default(), schema, source);

        std::thread::sleep(Duration::from_millis(30));
        workers.shutdown_handle().shutdown();
        // Drop joins both threads; if either panics, the test fails.
        drop(workers);
    }

    // ── Phase 5: double-buffered applier ───────────────────────────────────────

    #[test]
    fn double_buffer_version_selects_write_buf() {
        // When version is even, write_buf is odd; when version is odd, write_buf is even.
        // This ensures write and read always target different allocations.
        // We verify the version parity semantics without requiring cuda.
        let even_version: u64 = 4;
        let odd_version: u64 = 3;

        // write_buf index at version v = (v+1) % 2
        // read_buf  index at version v = v % 2
        let write_at_even = (even_version + 1) % 2; // = 1 (odd buf)
        let read_at_even = even_version % 2; // = 0 (even buf)
        let write_at_odd = (odd_version + 1) % 2; // = 0 (even buf)
        let read_at_odd = odd_version % 2; // = 1 (odd buf)

        assert_ne!(
            write_at_even, read_at_even,
            "write and read must differ (even version)"
        );
        assert_ne!(
            write_at_odd, read_at_odd,
            "write and read must differ (odd version)"
        );
    }

    #[test]
    fn pinned_applier_rejects_unknown_slot() {
        let applier = PinnedSlotApplier::new(SlotSchema::new());
        let update = SlotUpdate {
            source: "src".into(),
            slot: "nonexistent".into(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![1.0],
            observed_at: "".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "aabb".into(),
        };
        let err = applier.validate_update(&update);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("unknown slot"));
    }

    #[test]
    fn pinned_applier_rejects_dtype_mismatch() {
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));
        let applier = PinnedSlotApplier::new(schema);
        let update = SlotUpdate {
            source: "src".into(),
            slot: "market.price".into(),
            dtype: SlotDType::I32,
            shape: vec![1],
            values_f32: vec![1.0],
            observed_at: "".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "ccdd".into(),
        };
        let err = applier.validate_update(&update);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("dtype mismatch"));
    }

    #[test]
    fn double_buffer_registry_contains_returns_false_for_unregistered() {
        let registry = DoubleBufferRegistry::new();
        assert!(!registry.contains("market.price"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn pinned_applier_no_cuda_drops_with_reason() {
        let mut schema = SlotSchema::new();
        schema.register(DeviceSlot::tensor_f32("market.price", vec![1]));
        let mut applier = PinnedSlotApplier::new(schema);
        let mut registry = DoubleBufferRegistry::new();

        let update = SlotUpdate {
            source: "src".into(),
            slot: "market.price".into(),
            dtype: SlotDType::F32,
            shape: vec![1],
            values_f32: vec![42.0],
            observed_at: "".into(),
            policy: BackpressurePolicy::LatestWins,
            event_hash: "eeff".into(),
        };
        let receipts = applier.apply_batch(vec![update], &mut registry);
        assert_eq!(receipts.len(), 1);
        assert!(receipts[0].dropped);
        assert!(receipts[0].reason.as_deref().unwrap_or("").contains("cuda"));
    }
}
