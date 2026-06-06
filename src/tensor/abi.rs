use cudarc::driver::DeviceRepr;
use serde::{Deserialize, Serialize};

use super::{ORCH_CONTINUE, ORCH_HALTED, ORCH_WAIT_EXTERNAL};
use crate::types::ProcessReceipt;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TensorEdge {
    pub src: i32,
    pub dst: i32,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
}

unsafe impl DeviceRepr for TensorEdge {}

impl TensorEdge {
    pub const fn new(src: i32, dst: i32, confidence: f32, cost: f32, safety: f32) -> Self {
        Self {
            src,
            dst,
            confidence,
            cost,
            safety,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionBias {
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
}

unsafe impl DeviceRepr for ProjectionBias {}

impl Default for ProjectionBias {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            cost: 1.0,
            safety: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionOutcome {
    Success,
    Failure,
    Timeout,
    SafetyViolation,
}

impl ExecutionOutcome {
    pub fn code(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::Failure => 1,
            Self::Timeout => 2,
            Self::SafetyViolation => 3,
        }
    }
}

/// GPU-native receipt produced by the hot dispatch path.
///
/// Written entirely on-device by `tensor_quantale_gpu_dispatch`; drained by
/// `tensor_quantale_drain_device_receipts` without any CPU tensor-update hop.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceReceipt {
    pub region_id: i32,
    pub src: i32,
    pub dst: i32,
    pub outcome: i32,
    pub latency: f32,
    pub valid: i32,
    pub output_flags: i32,
}

unsafe impl DeviceRepr for DeviceReceipt {}

/// Host-side mirror of the CUDA `GpuDispatchMailbox` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDispatchMailboxHost {
    pub pending_region_id: i32,
    pub src_node: i32,
    pub dst_node: i32,
    /// Outcome derived from the JIT kernel exit code: 0=success, 1=failure,
    /// 2=timeout, 3=safety_violation.  Must be set correctly before launching
    /// tensor_quantale_gpu_dispatch so the receipt truth is preserved.
    pub outcome: i32,
    pub dispatched: i32,
}

unsafe impl DeviceRepr for GpuDispatchMailboxHost {}

// ── Phase-1 device structs ────────────────────────────────────────────────────

/// Persistent GPU-resident orchestration step state.
/// Mirrors `OrchestrationState` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrchestrationState {
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
    pub star_bound: i32, // Phase 4: current star bound (0 = no star active)
    pub consecutive_blocks: i32, // Phase 5: count of consecutive blocked steps
    pub block_threshold: i32, // Phase 5: hard-reset threshold (0 = disabled)
    pub hard_reset_requested: i32, // Phase 5: set to 1 when HALT action fires
    pub rollback_available: i32, // Phase 5: 1 when rollback marker is saved
    pub failure_action: i32, // Phase 5: last FAILURE_ACTION_* decision
    pub selected_src: i32, // Phase 8: src of last committed edge
    pub selected_dst: i32, // Phase 8: dst of last committed edge
    // GPU-native control-flow (Plan: gpu-native seq/par/choice/star)
    pub selected_control_edge: i32, // ControlEdge table index, or -1
    pub selected_control_op: i32,   // CONTROL_OP_* for last committed control op, or -1
    pub selected_control_lhs: i32,  // lhs of selected control edge, or -1
    pub selected_control_rhs: i32,  // rhs of selected control edge, or -1
    pub control_epoch: i32,         // incremented on each control decision commit
    pub star_counter_epoch: i32,    // incremented when any per-edge star counter advances
    pub last_block_reason: i32,     // ORCH_BLOCK_REASON_* code
}

unsafe impl DeviceRepr for OrchestrationState {}

/// GPU → CPU/IO service command (Phase-3 protocol).
/// Mirrors `DeviceCommand` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceCommand {
    pub valid: i32,
    pub command_id: i32,
    pub node_id: i32,
    pub src: i32,
    pub dst: i32,
    pub dispatch_kind: i32,
    /// Index into the host operator name table; host resolves to executable path.
    pub operator_name_id: i32,
    /// Deadline in scheduler ticks; 0 = no timeout.
    pub timeout_ticks: i32,
    /// Remaining retries; 0 = no retry.
    pub retry_budget: i32,
    pub payload_offset: i32,
    pub payload_len: i32,
}

unsafe impl DeviceRepr for DeviceCommand {}

/// Extended receipt returned from CPU/IO services (Phase-3 protocol).
/// Mirrors `DeviceReceiptExt` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceReceiptExt {
    pub valid: i32,
    pub consumed: i32,
    pub command_id: i32,
    pub node_id: i32,
    pub src: i32,
    pub dst: i32,
    pub outcome: i32,
    pub receipt_kind: i32,
    pub output_flags: i32,
    pub latency: f32,
}

unsafe impl DeviceRepr for DeviceReceiptExt {}

// ── Phase-4 device structs ────────────────────────────────────────────────────

/// One edge in a lowered pattern control-flow table.
/// Mirrors `ControlEdge` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlEdge {
    pub op: i32,
    pub lhs: i32,
    pub rhs: i32,
    pub guard: i32,
    pub order: i32,
    pub bound: i32,
}

unsafe impl DeviceRepr for ControlEdge {}

/// Per-node effect entry for par-eligibility checks.
/// Mirrors `EffectTable` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectTable {
    pub reads: i32,
    pub writes: i32,
    pub locks: i32,
    pub safety_class: i32,
}

unsafe impl DeviceRepr for EffectTable {}

// ── Phase-5 device struct ─────────────────────────────────────────────────────

/// Per-node failure policy stored on GPU.
/// Mirrors `FailurePolicy` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FailurePolicy {
    /// Remaining retries; -1 = unlimited; 0 = exhausted → BLOCK/REPAIR.
    pub retry_budget: i32,
    /// Consecutive failures before scheduler-level BLOCK; -1 = disabled.
    pub block_threshold: i32,
    /// 1 = save rollback marker on any failure.
    pub rollback_on_failure: i32,
    /// 1 = emit repair DeviceCommand when budget exhausted.
    pub repair_on_block: i32,
}

unsafe impl DeviceRepr for FailurePolicy {}

/// Packed classification target passed to `failure_policy_classify_and_emit`.
/// Using a struct keeps the kernel within cudarc's LaunchAsync arity limit.
/// Mirrors `FailureClassifyRequest` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FailureClassifyRequest {
    pub outcome: i32,
    pub node_id: i32,
    pub src: i32,
    pub dst: i32,
    pub command_id: i32,
}

unsafe impl DeviceRepr for FailureClassifyRequest {}

// ── Phase-6 device struct ─────────────────────────────────────────────────────

/// One entry in the learned-delta ring, recording what was learned from a
/// receipt.  Emitted on-device; drained by the CPU service for persistence.
/// Mirrors `LearnedDelta` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LearnedDelta {
    pub src: i32,
    pub dst: i32,
    pub confidence_delta: f32,
    pub cost_delta: f32,
    pub safety_delta: f32,
}

unsafe impl DeviceRepr for LearnedDelta {}

// ── Phase-8 device structs ────────────────────────────────────────────────────

/// One entry in the GPU-resident orchestration event trace ring.
/// Mirrors `OrchestrationEvent` in `cuda/quantale_world.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrchestrationEvent {
    pub step: i32,
    pub event_kind: i32,
    pub selected_node: i32,
    pub selected_group: i32,
    pub src: i32,
    pub dst: i32,
    pub outcome: i32,
    // Phase 8 control-trace fields
    pub selected_control_op: i32,
    pub selected_control_edge: i32,
    pub branch_count: i32,
    pub star_counter_val: i32,
}

unsafe impl DeviceRepr for OrchestrationEvent {}

/// Decoded result of a `tensor_quantale_orchestrate_step` launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrchStepStatus {
    Continue,
    WaitExternal,
    Halted,
    Error,
}

impl OrchStepStatus {
    pub fn from_code(code: i32) -> Self {
        match code {
            ORCH_CONTINUE => Self::Continue,
            ORCH_WAIT_EXTERNAL => Self::WaitExternal,
            ORCH_HALTED => Self::Halted,
            _ => Self::Error,
        }
    }
}

impl From<&ProcessReceipt> for ExecutionOutcome {
    fn from(receipt: &ProcessReceipt) -> Self {
        match receipt.exit_code {
            0 => Self::Success,
            124 => Self::Timeout,
            _ => Self::Failure,
        }
    }
}
