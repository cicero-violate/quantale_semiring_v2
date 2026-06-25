//! Opt-in Canon runtime transition tracing.
//!
//! Tensor payloads are written as little-endian `f32` binary. JSONL records
//! contain versioned metadata and typed byte ranges into the binary file.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    DeviceReceipt, DeviceSlotRegistry, OrchestrationState, StreamReceipt, TENSOR_LAYER_COUNT,
    TENSOR_LEN,
};

pub const TRACE_SCHEMA_VERSION: &str = "canon.runtime.transition.v1";
pub const TRACE_DTYPE: &str = "float32_le";
pub const TRACE_RECORDS_FILE: &str = "records.jsonl";
pub const TRACE_ARRAYS_FILE: &str = "arrays.f32le.bin";
pub const TRACE_MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionTraceConfig {
    pub enabled: bool,
    pub output_path: PathBuf,
    pub sample_every: u64,
    pub max_records: usize,
    pub flush_every: usize,
    pub include_slot_values: bool,
}

impl Default for TransitionTraceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_path: PathBuf::from("state/canon_runtime_trace"),
            sample_every: 1,
            max_records: 10_000,
            flush_every: 32,
            include_slot_values: true,
        }
    }
}

impl TransitionTraceConfig {
    pub fn from_env() -> Self {
        let default = Self::default();
        Self {
            enabled: env_bool("CANON_TRACE_ENABLED", default.enabled),
            output_path: std::env::var_os("CANON_TRACE_OUTPUT_PATH")
                .map(PathBuf::from)
                .unwrap_or(default.output_path),
            sample_every: env_u64("CANON_TRACE_SAMPLE_EVERY", default.sample_every).max(1),
            max_records: env_usize("CANON_TRACE_MAX_RECORDS", default.max_records),
            flush_every: env_usize("CANON_TRACE_FLUSH_EVERY", default.flush_every).max(1),
            include_slot_values: env_bool(
                "CANON_TRACE_INCLUDE_SLOT_VALUES",
                default.include_slot_values,
            ),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.output_path.as_os_str().is_empty() {
            return Err("transition trace output_path must be non-empty".to_string());
        }
        if self.sample_every == 0 {
            return Err("transition trace sample_every must be >= 1".to_string());
        }
        if self.flush_every == 0 {
            return Err("transition trace flush_every must be >= 1".to_string());
        }
        Ok(())
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryArrayRef {
    pub file: String,
    pub offset_bytes: u64,
    pub byte_len: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub order: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationStateSnapshot {
    pub step: i32,
    pub halted: i32,
    pub blocked: i32,
    pub current_frontier_epoch: i32,
    pub selected_group: i32,
    pub selected_node: i32,
    pub pending_external_count: i32,
    pub pending_receipt_count: i32,
    pub failure_count: i32,
    pub rollback_requested: i32,
    pub star_bound: i32,
    pub consecutive_blocks: i32,
    pub block_threshold: i32,
    pub hard_reset_requested: i32,
    pub rollback_available: i32,
    pub failure_action: i32,
    pub selected_src: i32,
    pub selected_dst: i32,
    pub selected_control_edge: i32,
    pub selected_control_op: i32,
    pub selected_control_lhs: i32,
    pub selected_control_rhs: i32,
    pub control_epoch: i32,
    pub star_counter_epoch: i32,
    pub last_block_reason: i32,
}

impl From<OrchestrationState> for OrchestrationStateSnapshot {
    fn from(state: OrchestrationState) -> Self {
        Self {
            step: state.step,
            halted: state.halted,
            blocked: state.blocked,
            current_frontier_epoch: state.current_frontier_epoch,
            selected_group: state.selected_group,
            selected_node: state.selected_node,
            pending_external_count: state.pending_external_count,
            pending_receipt_count: state.pending_receipt_count,
            failure_count: state.failure_count,
            rollback_requested: state.rollback_requested,
            star_bound: state.star_bound,
            consecutive_blocks: state.consecutive_blocks,
            block_threshold: state.block_threshold,
            hard_reset_requested: state.hard_reset_requested,
            rollback_available: state.rollback_available,
            failure_action: state.failure_action,
            selected_src: state.selected_src,
            selected_dst: state.selected_dst,
            selected_control_edge: state.selected_control_edge,
            selected_control_op: state.selected_control_op,
            selected_control_lhs: state.selected_control_lhs,
            selected_control_rhs: state.selected_control_rhs,
            control_epoch: state.control_epoch,
            star_counter_epoch: state.star_counter_epoch,
            last_block_reason: state.last_block_reason,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceReceiptSnapshot {
    pub region_id: i32,
    pub src: i32,
    pub dst: i32,
    pub outcome: i32,
    pub latency: f32,
    pub output_flags: i32,
}

impl From<DeviceReceipt> for DeviceReceiptSnapshot {
    fn from(receipt: DeviceReceipt) -> Self {
        Self {
            region_id: receipt.region_id,
            src: receipt.src,
            dst: receipt.dst,
            outcome: receipt.outcome,
            latency: receipt.latency,
            output_flags: receipt.output_flags,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReceiptSnapshot {
    pub command_id: i32,
    pub node_id: i32,
    pub outcome: i32,
    pub node_name: String,
    pub exit_code: i32,
    pub stdout_payload_sha256: String,
    pub stderr_payload_sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlotValueSnapshot {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedArrayRef {
    pub name: String,
    pub source_dtype: String,
    pub logical_shape: Vec<usize>,
    pub values: BinaryArrayRef,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantResults {
    pub no_duplicate_receipts: bool,
    pub frontier_valid: bool,
    pub no_command_without_receipt: bool,
    pub finite_phi_t: bool,
    pub finite_phi_exact: bool,
    pub finite_phi_next: bool,
    pub action_step_matches: bool,
    pub hard_violation_count: u32,
}

#[derive(Clone, Debug)]
pub struct TransitionCapture {
    pub step_id: u64,
    pub phi_t: Vec<f32>,
    pub phi_exact: Vec<f32>,
    pub phi_next: Vec<f32>,
    pub selected_src: i32,
    pub selected_dst: i32,
    pub selected_value: f32,
    pub selected_control_op: i32,
    pub selected_control_edge: i32,
    pub dispatch_kind: i32,
    pub orchestration_state_before: OrchestrationStateSnapshot,
    pub orchestration_state_after: OrchestrationStateSnapshot,
    pub stream_receipts: Vec<StreamReceipt>,
    pub device_receipts: Vec<DeviceReceiptSnapshot>,
    pub external_receipts: Vec<ExternalReceiptSnapshot>,
    pub receipt_priors: Vec<f32>,
    pub changed_slot_values: Vec<SlotValueSnapshot>,
    pub projection_applied: bool,
    pub invariant_results: InvariantResults,
    pub reset_boundary: bool,
    pub exact_transition_ns: u64,
    pub observation_apply_ns: u64,
    pub projection_ns: u64,
    pub trace_capture_ns: u64,
    pub total_step_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionRecord {
    pub schema_version: String,
    pub run_id: String,
    pub step_id: u64,
    pub monotonic_timestamp_ns: u64,
    pub topology_sha256: String,
    pub node_count: usize,
    pub transition_count: usize,
    pub phi_t: BinaryArrayRef,
    pub phi_exact: BinaryArrayRef,
    pub phi_next: BinaryArrayRef,
    pub selected_src: i32,
    pub selected_dst: i32,
    pub selected_value: f32,
    pub selected_control_op: i32,
    pub selected_control_edge: i32,
    pub dispatch_kind: i32,
    pub orchestration_state_before: OrchestrationStateSnapshot,
    pub orchestration_state_after: OrchestrationStateSnapshot,
    pub stream_receipts: Vec<StreamReceipt>,
    pub device_receipts: Vec<DeviceReceiptSnapshot>,
    pub external_receipts: Vec<ExternalReceiptSnapshot>,
    pub receipt_priors: Option<BinaryArrayRef>,
    pub changed_slot_names: Vec<String>,
    pub changed_slot_values: Vec<NamedArrayRef>,
    pub projection_applied: bool,
    pub invariant_results: InvariantResults,
    pub reset_boundary: bool,
    pub exact_transition_ns: u64,
    pub observation_apply_ns: u64,
    pub projection_ns: u64,
    pub trace_capture_ns: u64,
    pub total_step_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRunManifest {
    pub schema_version: String,
    pub run_id: String,
    pub topology_sha256: String,
    pub node_count: usize,
    pub transition_count: usize,
    pub node_ordering: Vec<String>,
    pub layer_order: Vec<String>,
    pub tensor_shape: Vec<usize>,
    pub dtype: String,
    pub arrays_file: String,
    pub records_file: String,
    pub reset_boundary_semantics: String,
    pub missing_observation_semantics: String,
    pub projection_timing_semantics: String,
    pub config: TransitionTraceConfig,
    pub started_unix_timestamp_ns: u64,
    pub ended_unix_timestamp_ns: Option<u64>,
    pub attempted_steps: u64,
    pub records_written: usize,
    pub dropped_records: usize,
    pub invalid_records: usize,
    pub reset_boundaries: usize,
    pub trace_capture_ns_sum: u64,
    pub trace_write_ns_sum: u64,
}

pub struct TransitionTraceWriter {
    config: TransitionTraceConfig,
    output_dir: PathBuf,
    arrays: BufWriter<File>,
    records: BufWriter<File>,
    array_offset: u64,
    started: Instant,
    manifest: TraceRunManifest,
    finished: bool,
}

impl TransitionTraceWriter {
    pub fn open(
        config: TransitionTraceConfig,
        topology_sha256: String,
        node_count: usize,
        transition_count: usize,
        node_ordering: Vec<String>,
    ) -> Result<Self, String> {
        config.validate()?;
        if node_ordering.len() != node_count {
            return Err(format!(
                "node ordering length {} does not match node_count {node_count}",
                node_ordering.len()
            ));
        }

        fs::create_dir_all(&config.output_path).map_err(|error| {
            format!(
                "create transition trace directory '{}': {error}",
                config.output_path.display()
            )
        })?;

        let run_id = std::env::var("CANON_TRACE_RUN_ID").unwrap_or_else(|_| default_run_id());
        let arrays = BufWriter::new(
            File::create(config.output_path.join(TRACE_ARRAYS_FILE))
                .map_err(|error| format!("create trace array file: {error}"))?,
        );
        let records = BufWriter::new(
            File::create(config.output_path.join(TRACE_RECORDS_FILE))
                .map_err(|error| format!("create trace record file: {error}"))?,
        );

        let manifest = TraceRunManifest {
            schema_version: TRACE_SCHEMA_VERSION.to_string(),
            run_id,
            topology_sha256,
            node_count,
            transition_count,
            node_ordering,
            layer_order: vec![
                "confidence_max_times".to_string(),
                "cost_min_plus".to_string(),
                "safety_max_min".to_string(),
            ],
            tensor_shape: vec![TENSOR_LAYER_COUNT, node_count, node_count],
            dtype: TRACE_DTYPE.to_string(),
            arrays_file: TRACE_ARRAYS_FILE.to_string(),
            records_file: TRACE_RECORDS_FILE.to_string(),
            reset_boundary_semantics:
                "phi_next[t] must equal phi_t[t+1] unless reset_boundary=true; dynamic topology deltas and explicit runtime resets establish reset boundaries"
                    .to_string(),
            missing_observation_semantics:
                "empty receipt and changed-slot collections mean no observation was committed during the transition; missing values are never imputed"
                    .to_string(),
            projection_timing_semantics:
                "projection_ns is zero when projection remains fused inside the orchestration kernel; exact_transition_ns then includes projection"
                    .to_string(),
            config: config.clone(),
            started_unix_timestamp_ns: unix_now_ns(),
            ended_unix_timestamp_ns: None,
            attempted_steps: 0,
            records_written: 0,
            dropped_records: 0,
            invalid_records: 0,
            reset_boundaries: 0,
            trace_capture_ns_sum: 0,
            trace_write_ns_sum: 0,
        };

        let mut writer = Self {
            config: config.clone(),
            output_dir: config.output_path.clone(),
            arrays,
            records,
            array_offset: 0,
            started: Instant::now(),
            manifest,
            finished: false,
        };
        writer.write_manifest()?;
        Ok(writer)
    }

    pub fn run_id(&self) -> &str {
        &self.manifest.run_id
    }

    pub fn records_written(&self) -> usize {
        self.manifest.records_written
    }

    pub fn should_sample(&mut self, step_id: u64) -> bool {
        self.manifest.attempted_steps = self.manifest.attempted_steps.saturating_add(1);
        if self.config.max_records > 0
            && self.manifest.records_written >= self.config.max_records
        {
            self.manifest.dropped_records = self.manifest.dropped_records.saturating_add(1);
            return false;
        }
        step_id % self.config.sample_every == 0
    }

    pub fn append(&mut self, capture: TransitionCapture) -> Result<(), String> {
        let write_started = Instant::now();
        if let Err(error) = validate_capture(&capture, self.manifest.node_count) {
            self.manifest.invalid_records = self.manifest.invalid_records.saturating_add(1);
            self.write_manifest()?;
            return Err(error);
        }

        let tensor_shape = vec![
            TENSOR_LAYER_COUNT,
            self.manifest.node_count,
            self.manifest.node_count,
        ];
        let phi_t = self.write_f32_array(&capture.phi_t, tensor_shape.clone())?;
        let phi_exact = self.write_f32_array(&capture.phi_exact, tensor_shape.clone())?;
        let phi_next = self.write_f32_array(&capture.phi_next, tensor_shape)?;

        let receipt_priors = if capture.receipt_priors.is_empty() {
            None
        } else {
            Some(self.write_f32_array(
                &capture.receipt_priors,
                vec![capture.receipt_priors.len()],
            )?)
        };

        let changed_slot_names = capture
            .changed_slot_values
            .iter()
            .map(|slot| slot.name.clone())
            .collect();
        let mut changed_slot_values = Vec::with_capacity(capture.changed_slot_values.len());
        for slot in &capture.changed_slot_values {
            changed_slot_values.push(NamedArrayRef {
                name: slot.name.clone(),
                source_dtype: slot.dtype.clone(),
                logical_shape: slot.shape.clone(),
                values: self.write_f32_array(&slot.values, slot.shape.clone())?,
            });
        }

        let record = TransitionRecord {
            schema_version: TRACE_SCHEMA_VERSION.to_string(),
            run_id: self.manifest.run_id.clone(),
            step_id: capture.step_id,
            monotonic_timestamp_ns: saturating_u64(self.started.elapsed().as_nanos()),
            topology_sha256: self.manifest.topology_sha256.clone(),
            node_count: self.manifest.node_count,
            transition_count: self.manifest.transition_count,
            phi_t,
            phi_exact,
            phi_next,
            selected_src: capture.selected_src,
            selected_dst: capture.selected_dst,
            selected_value: capture.selected_value,
            selected_control_op: capture.selected_control_op,
            selected_control_edge: capture.selected_control_edge,
            dispatch_kind: capture.dispatch_kind,
            orchestration_state_before: capture.orchestration_state_before,
            orchestration_state_after: capture.orchestration_state_after,
            stream_receipts: capture.stream_receipts,
            device_receipts: capture.device_receipts,
            external_receipts: capture.external_receipts,
            receipt_priors,
            changed_slot_names,
            changed_slot_values,
            projection_applied: capture.projection_applied,
            invariant_results: capture.invariant_results,
            reset_boundary: capture.reset_boundary,
            exact_transition_ns: capture.exact_transition_ns,
            observation_apply_ns: capture.observation_apply_ns,
            projection_ns: capture.projection_ns,
            trace_capture_ns: capture.trace_capture_ns,
            total_step_ns: capture.total_step_ns,
        };

        serde_json::to_writer(&mut self.records, &record)
            .map_err(|error| format!("serialize transition record: {error}"))?;
        self.records
            .write_all(b"\n")
            .map_err(|error| format!("write transition record newline: {error}"))?;

        self.manifest.records_written = self.manifest.records_written.saturating_add(1);
        self.manifest.trace_capture_ns_sum = self
            .manifest
            .trace_capture_ns_sum
            .saturating_add(capture.trace_capture_ns);
        self.manifest.trace_write_ns_sum = self
            .manifest
            .trace_write_ns_sum
            .saturating_add(saturating_u64(write_started.elapsed().as_nanos()));
        if capture.reset_boundary {
            self.manifest.reset_boundaries = self.manifest.reset_boundaries.saturating_add(1);
        }

        if self.manifest.records_written % self.config.flush_every == 0 {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.arrays
            .flush()
            .map_err(|error| format!("flush transition arrays: {error}"))?;
        self.records
            .flush()
            .map_err(|error| format!("flush transition records: {error}"))?;
        self.write_manifest()
    }

    pub fn finish(&mut self) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        self.manifest.ended_unix_timestamp_ns = Some(unix_now_ns());
        self.flush()?;
        self.finished = true;
        Ok(())
    }

    fn write_f32_array(
        &mut self,
        values: &[f32],
        shape: Vec<usize>,
    ) -> Result<BinaryArrayRef, String> {
        let expected_len = shape.iter().copied().product::<usize>();
        if expected_len != values.len() {
            return Err(format!(
                "array shape {shape:?} implies {expected_len} values, got {}",
                values.len()
            ));
        }
        let offset_bytes = self.array_offset;
        for value in values {
            self.arrays
                .write_all(&value.to_le_bytes())
                .map_err(|error| format!("write transition array: {error}"))?;
        }
        let byte_len = (values.len() * std::mem::size_of::<f32>()) as u64;
        self.array_offset = self.array_offset.saturating_add(byte_len);
        Ok(BinaryArrayRef {
            file: TRACE_ARRAYS_FILE.to_string(),
            offset_bytes,
            byte_len,
            dtype: TRACE_DTYPE.to_string(),
            shape,
            order: "C".to_string(),
        })
    }

    fn write_manifest(&self) -> Result<(), String> {
        let path = self.output_dir.join(TRACE_MANIFEST_FILE);
        let temporary = self.output_dir.join("manifest.json.tmp");
        let bytes = serde_json::to_vec_pretty(&self.manifest)
            .map_err(|error| format!("serialize transition manifest: {error}"))?;
        fs::write(&temporary, bytes).map_err(|error| {
            format!(
                "write temporary transition manifest '{}': {error}",
                temporary.display()
            )
        })?;
        fs::rename(&temporary, &path).map_err(|error| {
            format!(
                "replace transition manifest '{}' from '{}': {error}",
                path.display(),
                temporary.display()
            )
        })
    }
}

impl Drop for TransitionTraceWriter {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub fn validate_capture(capture: &TransitionCapture, node_count: usize) -> Result<(), String> {
    let expected = TENSOR_LAYER_COUNT * node_count * node_count;
    for (name, values) in [
        ("phi_t", &capture.phi_t),
        ("phi_exact", &capture.phi_exact),
        ("phi_next", &capture.phi_next),
    ] {
        if values.len() != expected || values.len() != TENSOR_LEN {
            return Err(format!(
                "{name} length {} does not equal 3N^2={expected} and generated TENSOR_LEN={TENSOR_LEN}",
                values.len()
            ));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(format!("{name} contains NaN or infinity"));
        }
    }
    if !capture.selected_value.is_finite() {
        return Err("selected_value contains NaN or infinity".to_string());
    }
    if !capture.receipt_priors.iter().all(|value| value.is_finite()) {
        return Err("receipt_priors contains NaN or infinity".to_string());
    }
    if capture
        .changed_slot_values
        .iter()
        .flat_map(|slot| slot.values.iter())
        .any(|value| !value.is_finite())
    {
        return Err("changed_slot_values contains NaN or infinity".to_string());
    }
    if capture.invariant_results.hard_violation_count != 0 {
        return Err(format!(
            "hard invariant violations are not accepted: {}",
            capture.invariant_results.hard_violation_count
        ));
    }
    Ok(())
}

pub fn validate_continuity(
    previous_phi_next: &[f32],
    current_phi_t: &[f32],
    reset_boundary: bool,
) -> Result<(), String> {
    if reset_boundary || previous_phi_next == current_phi_t {
        Ok(())
    } else {
        Err("phi_next[t] != phi_t[t+1] outside an explicit reset boundary".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn snapshot_slot_values(
    registry: &DeviceSlotRegistry,
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    include_values: bool,
) -> Result<Vec<SlotValueSnapshot>, String> {
    if !include_values {
        return Ok(Vec::new());
    }
    let mut slots = Vec::new();
    for (slot, buffer) in registry.iter() {
        let values = dev
            .dtoh_sync_copy(buffer)
            .map_err(|error| format!("read device slot '{}': {error}", slot.name))?;
        if values.len() != slot.len() {
            return Err(format!(
                "slot '{}' metadata/readback length mismatch: {} != {}",
                slot.name,
                values.len(),
                slot.len()
            ));
        }
        slots.push(SlotValueSnapshot {
            name: slot.name.clone(),
            dtype: slot.dtype.clone(),
            shape: slot.shape.clone(),
            values,
        });
    }
    slots.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(slots)
}

pub fn changed_slots(
    before: &[SlotValueSnapshot],
    after: &[SlotValueSnapshot],
) -> Vec<SlotValueSnapshot> {
    let prior: BTreeMap<&str, &SlotValueSnapshot> = before
        .iter()
        .map(|slot| (slot.name.as_str(), slot))
        .collect();
    after
        .iter()
        .filter(|slot| {
            prior.get(slot.name.as_str()).is_none_or(|previous| {
                previous.dtype != slot.dtype
                    || previous.shape != slot.shape
                    || previous.values != slot.values
            })
        })
        .cloned()
        .collect()
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, String> {
    let path = path.as_ref();
    let mut file =
        File::open(path).map_err(|error| format!("open '{}' for sha256: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read '{}' for sha256: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn sha256_text(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

pub fn read_binary_array(
    directory: impl AsRef<Path>,
    array: &BinaryArrayRef,
) -> Result<Vec<f32>, String> {
    if array.dtype != TRACE_DTYPE {
        return Err(format!("unsupported array dtype '{}'", array.dtype));
    }
    if array.byte_len % std::mem::size_of::<f32>() as u64 != 0 {
        return Err(format!("array byte length {} is not f32-aligned", array.byte_len));
    }
    let mut file = File::open(directory.as_ref().join(&array.file))
        .map_err(|error| format!("open trace array file: {error}"))?;
    file.seek(SeekFrom::Start(array.offset_bytes))
        .map_err(|error| format!("seek trace array file: {error}"))?;
    let mut bytes = vec![0_u8; array.byte_len as usize];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read trace array: {error}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect())
}

fn unix_now_ns() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    saturating_u64(nanos)
}

fn default_run_id() -> String {
    format!("canon-{}-{}", std::process::id(), unix_now_ns())
}

fn saturating_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TENSOR_NODE_COUNT;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "canon-runtime-trace-{name}-{}-{}",
            std::process::id(),
            unix_now_ns()
        ))
    }

    fn valid_capture() -> TransitionCapture {
        let values = TENSOR_LEN;
        TransitionCapture {
            step_id: 7,
            phi_t: vec![1.0; values],
            phi_exact: vec![2.0; values],
            phi_next: vec![3.0; values],
            selected_src: 1,
            selected_dst: 2,
            selected_value: 0.5,
            selected_control_op: -1,
            selected_control_edge: -1,
            dispatch_kind: 0,
            orchestration_state_before: OrchestrationStateSnapshot::default(),
            orchestration_state_after: OrchestrationStateSnapshot {
                step: 7,
                selected_src: 1,
                selected_dst: 2,
                ..OrchestrationStateSnapshot::default()
            },
            stream_receipts: Vec::new(),
            device_receipts: Vec::new(),
            external_receipts: Vec::new(),
            receipt_priors: vec![0.0; TENSOR_NODE_COUNT],
            changed_slot_values: Vec::new(),
            projection_applied: true,
            invariant_results: InvariantResults {
                no_duplicate_receipts: true,
                frontier_valid: true,
                no_command_without_receipt: true,
                finite_phi_t: true,
                finite_phi_exact: true,
                finite_phi_next: true,
                action_step_matches: true,
                hard_violation_count: 0,
            },
            reset_boundary: false,
            exact_transition_ns: 10,
            observation_apply_ns: 20,
            projection_ns: 0,
            trace_capture_ns: 30,
            total_step_ns: 60,
        }
    }

    fn writer(dir: &Path) -> TransitionTraceWriter {
        TransitionTraceWriter::open(
            TransitionTraceConfig {
                enabled: true,
                output_path: dir.to_path_buf(),
                sample_every: 1,
                max_records: 10,
                flush_every: 1,
                include_slot_values: true,
            },
            "topology-hash".to_string(),
            TENSOR_NODE_COUNT,
            0,
            (0..TENSOR_NODE_COUNT)
                .map(|index| format!("node-{index}"))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn tracing_is_disabled_by_default() {
        assert!(!TransitionTraceConfig::default().enabled);
    }

    #[test]
    fn writer_owns_binary_values_and_does_not_alias_capture_storage() {
        let dir = temp_dir("alias");
        let mut capture = valid_capture();
        let expected = capture.phi_t[0];
        let retained = capture.clone();
        let mut writer = writer(&dir);
        writer.append(capture.clone()).unwrap();
        capture.phi_t[0] = 999.0;
        writer.finish().unwrap();

        let records = fs::read_to_string(dir.join(TRACE_RECORDS_FILE)).unwrap();
        let record: TransitionRecord = serde_json::from_str(records.lines().next().unwrap()).unwrap();
        let observed = read_binary_array(&dir, &record.phi_t).unwrap();
        assert_eq!(observed[0], expected);
        assert_eq!(retained.phi_t[0], expected);
        assert_ne!(observed[0], capture.phi_t[0]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_non_finite_arrays_and_hard_invariant_failures() {
        let mut non_finite = valid_capture();
        non_finite.phi_next[0] = f32::NAN;
        assert!(validate_capture(&non_finite, TENSOR_NODE_COUNT).is_err());

        let mut violation = valid_capture();
        violation.invariant_results.hard_violation_count = 1;
        assert!(validate_capture(&violation, TENSOR_NODE_COUNT).is_err());
    }

    #[test]
    fn validates_continuity_except_at_reset_boundaries() {
        assert!(validate_continuity(&[1.0, 2.0], &[1.0, 2.0], false).is_ok());
        assert!(validate_continuity(&[1.0], &[2.0], false).is_err());
        assert!(validate_continuity(&[1.0], &[2.0], true).is_ok());
    }

    #[test]
    fn drop_flushes_a_complete_record_and_final_manifest() {
        let dir = temp_dir("shutdown");
        {
            let mut writer = writer(&dir);
            writer.append(valid_capture()).unwrap();
        }
        let records = fs::read_to_string(dir.join(TRACE_RECORDS_FILE)).unwrap();
        assert_eq!(records.lines().count(), 1);
        let manifest: TraceRunManifest =
            serde_json::from_slice(&fs::read(dir.join(TRACE_MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(manifest.records_written, 1);
        assert!(manifest.ended_unix_timestamp_ns.is_some());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn changed_slot_snapshots_are_owned() {
        let before = vec![SlotValueSnapshot {
            name: "slot".to_string(),
            dtype: "f32".to_string(),
            shape: vec![1],
            values: vec![1.0],
        }];
        let mut after = before.clone();
        after[0].values[0] = 2.0;
        let changed = changed_slots(&before, &after);
        after[0].values[0] = 3.0;
        assert_eq!(changed[0].values, vec![2.0]);
    }
}
