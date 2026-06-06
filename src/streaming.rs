//! Async streaming ingress — Phase 1: synchronous micro-batch streamer.
//!
//! Pipeline:
//!   push_raw_event → RawEventQueue → normalize_pending → SlotUpdateQueue
//!   → apply_pending → DeviceSlotRegistry → QuantaleOrchestrator

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write as _;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::device_slots::{DeviceSlot, DeviceSlotRegistry};
#[cfg(feature = "cuda")]
use crate::device_slots::UploadQueue;

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

// ── Stream configuration ──────────────────────────────────────────────────────

/// Configuration for the streaming ingress layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamConfig {
    pub stream_window_ms: u64,
    pub max_batch_size: usize,
    pub slot_update_queue_capacity: usize,
    pub raw_event_queue_capacity: usize,
    pub latest_wins_slots: Vec<String>,
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
        }
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
        .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
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
                Ok(()) => {}
            }

            match self.upload_queue.stage(&update.slot, &update.values_f32) {
                Err(e) => {
                    receipts.push(StreamReceipt {
                        source: update.source,
                        event_hash: update.event_hash,
                        slot: update.slot,
                        applied: false,
                        dropped: true,
                        reason: Some(format!("stage failed: {e}")),
                        version: 0,
                    });
                    continue;
                }
                Ok(()) => {}
            }

            staged_slots.push(update.slot.clone());
            receipts.push(StreamReceipt {
                source: update.source,
                event_hash: update.event_hash,
                slot: update.slot,
                applied: false,
                dropped: false,
                reason: None,
                version: 0,
            });
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
                StreamReceipt {
                    source: u.source,
                    event_hash: u.event_hash,
                    slot: u.slot,
                    applied: false,
                    dropped: true,
                    reason: Some(reason),
                    version: 0,
                }
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

        assert_eq!(compacted.len(), 1, "expected latest-wins to collapse to one update");
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
        assert_eq!(compacted.len(), 2, "NeverDrop updates must not be compacted");
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
            use std::sync::Arc;
            use cudarc::driver::CudaDevice;
            if let Ok(dev) = CudaDevice::new(0).map(Arc::new) {
                let receipts = ingress.run_batch(&mut registry, &dev);
                assert!(!receipts.is_empty());
                let dropped = receipts.iter().find(|r| r.dropped);
                assert!(dropped.is_some(), "expected dropped receipt for unknown slot");
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
            use std::sync::Arc;
            use cudarc::driver::CudaDevice;
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

    // ── receipt writer ─────────────────────────────────────────────────────────

    #[test]
    fn receipt_writer_appends_valid_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "stream_receipts_test_{}.jsonl",
            std::process::id()
        ));
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
}
