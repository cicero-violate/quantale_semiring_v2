use cudarc::driver::{CudaSlice, DeviceRepr};

use super::{
    ControlEdge, DeviceCommand, DeviceReceipt, DeviceReceiptExt, EffectTable, FailurePolicy,
    LearnedDelta, OrchestrationEvent, OrchestrationState, ProjectionBias,
};

/// Device-resident ring buffers for the Phase-1 orchestration state block,
/// plus the Phase-2 dispatch-kind table and step-status scratch buffer.
pub struct OrchestrationBuffers {
    pub state: CudaSlice<OrchestrationState>,
    pub command_ring: CudaSlice<DeviceCommand>,
    pub command_head: CudaSlice<i32>,
    pub command_tail: CudaSlice<i32>,
    pub receipt_ext_ring: CudaSlice<DeviceReceiptExt>,
    pub receipt_ext_head: CudaSlice<i32>,
    pub receipt_ext_tail: CudaSlice<i32>,
    /// Phase-2: per-node dispatch kind table (length N). Defaults to
    /// `DISPATCH_KIND_HF_DEVICE` for all nodes; callers can overwrite entries
    /// for EXTERNAL_PROCESS / EXTERNAL_IO nodes via `set_dispatch_kinds`.
    pub dispatch_kinds: CudaSlice<i32>,
    /// Phase-9: per-node reentrant-consumption mask.  Edges incident to nodes
    /// with mask=1 are reusable across continuous orchestration cycles.
    pub reentrant_mask: CudaSlice<i32>,
    /// Phase-2: single-element scratch buffer for the ORCH_* step status.
    pub step_status: CudaSlice<i32>,
    /// Phase-2: device copy of the default `ProjectionBias` used by the
    /// scheduler kernel.  Avoids per-call host allocations.
    pub default_bias: CudaSlice<ProjectionBias>,
    /// Phase-4: lowered pattern control-flow edge table.
    /// Starts with a single sentinel entry (lhs=-1, rhs=-1); replaced by
    /// `load_control_table` after patterns are compiled.
    pub control_edges: CudaSlice<ControlEdge>,
    /// Phase-4: per-node effect table indexed by node id.
    pub effect_table: CudaSlice<EffectTable>,
    /// Phase-5: per-node failure policy table (length N).
    /// Defaults to all-zero (budget=0 → immediate BLOCK); callers invoke
    /// `failure_policy_init` to set non-zero retry budgets.
    pub failure_policies: CudaSlice<FailurePolicy>,
    /// Phase-5: rollback snapshot of `consumed[]` (length MATRIX_LEN).
    pub rollback_consumed: CudaSlice<i32>,
    /// Phase-5: rollback snapshot of `active[]` (length N).
    pub rollback_active: CudaSlice<i32>,
    /// Phase-5: single-element scratch buffer for `failure_policy_classify_and_emit` output.
    pub failure_action_out: CudaSlice<i32>,
    /// Phase-6: per-node receipt prior table (float, length N).
    /// Updated on-device by `learned_delta_fold_receipt`; read by exploration
    /// seeding without a CPU round-trip.
    pub receipt_priors: CudaSlice<f32>,
    /// Phase-6: learned-edge delta ring (length LEARNED_DELTA_RING_SIZE).
    /// Produced on-device; drained by the CPU service for durable persistence.
    pub learned_delta_ring: CudaSlice<LearnedDelta>,
    pub learned_delta_head: CudaSlice<i32>,
    pub learned_delta_tail: CudaSlice<i32>,
    /// Phase-6: export buffer for `receipt_prior_snapshot` (length N).
    pub receipt_prior_snapshot_buf: CudaSlice<f32>,
    /// Phase-8: device event trace ring (length ORCH_TRACE_RING_SIZE).
    pub trace_ring: CudaSlice<OrchestrationEvent>,
    pub trace_head: CudaSlice<i32>,
    pub trace_tail: CudaSlice<i32>,
    /// Phase-8: host drain buffer for `orch_event_trace_drain` (length ORCH_TRACE_RING_SIZE).
    pub trace_drain_buf: CudaSlice<OrchestrationEvent>,
    /// Phase-8: single-element count output from `orch_event_trace_drain`.
    pub trace_drain_count: CudaSlice<i32>,
    /// Phase-8: single-element violation output for invariant checker kernels.
    pub orch_violation_out: CudaSlice<i32>,
    /// Phase-8: replay snapshot buffers (state + consumed + active).
    pub replay_state: CudaSlice<OrchestrationState>,
    pub replay_consumed: CudaSlice<i32>,
    pub replay_active: CudaSlice<i32>,
    /// GPU-native control-flow: per-edge star counter buffer (length MAX_CONTROL_EDGES).
    /// Indexed by ControlEdge table index.
    pub star_counters: CudaSlice<i32>,
    /// Replay snapshot of star_counters (same length as star_counters).
    pub replay_star_counters: CudaSlice<i32>,
}

/// Phase-2: host-side mirror of the CUDA `TensorWorldBundle` struct.
/// All fields are raw device addresses (u64 = CUdeviceptr).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TensorWorldBundleHost {
    pub tensor_dev: u64,
    pub witness_dev: u64,
    pub consumed_dev: u64,
    pub active_dev: u64,
    pub next_active_dev: u64,
    pub reentrant_mask_dev: u64,
    pub bias_dev: u64,
    pub decision_dev: u64,
    // GPU-native control-flow fields (Plan: gpu-native seq/par/choice/star)
    pub control_edges_dev: u64,
    pub control_edge_count: i32,
    pub effects_dev: u64,
    pub effect_count: i32,
    pub star_counters_dev: u64,
    pub star_counter_count: i32,
}

unsafe impl DeviceRepr for TensorWorldBundleHost {}

/// Device-resident ring buffer for `DeviceReceipt`s.
///
/// `ring` holds `DEVICE_RECEIPT_RING_SIZE` slots on the GPU.
/// `head` and `tail` are single-element slices so GPU kernels can advance
/// them atomically.  The CPU only reads `tail` to know when new receipts
/// arrived, and only writes `region_id` + mailbox data.
pub struct DeviceReceiptBuffer {
    pub ring: CudaSlice<DeviceReceipt>,
    pub head: CudaSlice<i32>,
    pub tail: CudaSlice<i32>,
}
