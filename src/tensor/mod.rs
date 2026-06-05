//! Three-layer tensor quantale engine.
//!
//! Layers:
//! - confidence/correctness: max-times
//! - compute/time cost: min-plus
//! - security/safety: max-min

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::config::{DEFAULT_BLOCK_SIZE, RuntimeContext};
use crate::device_slots::DeviceSlotRegistry;
use crate::error::CudaError;
use crate::exploration::{ExplorationCandidate, ExplorationEngine, ExplorationToken};
use crate::graph::{DecisionReport, Node, reconstruct_path_from_witness_matrix};
use crate::topology::{GraphTopology, NodeRegistry};

mod abi;
mod coverage;
mod kernel_source;

pub use abi::{
    ControlEdge, DeviceCommand, DeviceReceipt, DeviceReceiptExt, EffectTable, ExecutionOutcome,
    FailureClassifyRequest, FailurePolicy, GpuDispatchMailboxHost, LearnedDelta, OrchStepStatus,
    OrchestrationEvent, OrchestrationState, ProjectionBias, TensorEdge,
};
pub use coverage::{
    AbstractDeviceCoverage, AbstractDeviceCoverageEntry, DEFAULT_PAR_SLOT_ELEMENTS,
    FusionHfCoverage, FusionHfCoverageEntry, fusion_hf_region_id, gpu_region_slots,
    static_hf_symbol,
};
use kernel_source::{assemble_kernel_source, assemble_kernel_source_with_generated};

pub const TENSOR_LAYER_COUNT: usize = 3;
include!(concat!(env!("OUT_DIR"), "/topology_constants.rs"));
pub const MATRIX_LEN: usize = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT;
pub const TENSOR_LEN: usize = TENSOR_LAYER_COUNT * MATRIX_LEN;
pub const COST_INFINITY: f32 = 1.0e20;

pub const LAYER_CONFIDENCE: i32 = 0;
pub const LAYER_COST: i32 = 1;
pub const LAYER_SAFETY: i32 = 2;

const MODULE_NAME: &str = "quantale_semiring_v2_tensor";
const RESET_KERNEL: &str = "tensor_quantale_reset";
const EMBED_KERNEL: &str = "tensor_quantale_embed_edges";
const CLOSURE_KERNEL: &str = "tensor_quantale_closure";
const PROJECT_KERNEL: &str = "tensor_quantale_project";
const PROJECT_BATCH_KERNEL: &str = "tensor_quantale_project_batch";
const COMMIT_BATCH_KERNEL: &str = "tensor_quantale_commit_batch";
const DECAY_KERNEL: &str = "tensor_quantale_decay";
const EXPLORATION_SEED_KERNEL: &str = "tensor_quantale_seed_exploration";
const EXPLORATION_EXPAND_KERNEL: &str = "tensor_quantale_expand_tokens";
const EXPLORATION_SCORE_KERNEL: &str = "tensor_quantale_score_tokens";
const EXPLORATION_TOPK_KERNEL: &str = "tensor_quantale_select_topk_tokens";
const EXPLORATION_COMMIT_KERNEL: &str = "tensor_quantale_commit_exploration";
const JIT_CHAIN_SCORE_KERNEL: &str = "jit_chain_score_embed";
const DRAIN_DEVICE_RECEIPTS_KERNEL: &str = "tensor_quantale_drain_device_receipts";
const PUSH_DEVICE_RECEIPT_KERNEL: &str = "tensor_quantale_push_device_receipt";
const GPU_DISPATCH_KERNEL: &str = "tensor_quantale_gpu_dispatch";
const RING_PUSH_KERNEL: &str = "device_ring_push";
const RING_POP_KERNEL: &str = "device_ring_pop";
const PARALLEL_REDUCE_KERNEL: &str = "quantale_parallel_reduce";
const TOPK_BITONIC_KERNEL: &str = "quantale_topk_bitonic";
const PAR_GROUP_STEP_KERNEL: &str = "tensor_quantale_par_group_step";
const ORCH_STATE_INIT_KERNEL: &str = "orchestration_state_init";
const ORCH_STATE_SNAPSHOT_KERNEL: &str = "orchestration_state_snapshot";
const DEVICE_CMD_RING_PUSH_KERNEL: &str = "device_command_ring_push";
const DEVICE_RECEIPT_EXT_PUSH_KERNEL: &str = "device_receipt_ext_ring_push";
const DEVICE_RECEIPT_EXT_DRAIN_KERNEL: &str = "device_receipt_ext_drain";
const ORCHESTRATE_STEP_KERNEL: &str = "tensor_quantale_orchestrate_step";
const CONTROL_FLOW_ADVANCE_KERNEL: &str = "control_flow_advance";
const CHECK_EFFECTS_INDEPENDENT_KERNEL: &str = "check_effects_independent";
const FAILURE_POLICY_INIT_KERNEL: &str = "failure_policy_init";
const FAILURE_POLICY_CLASSIFY_KERNEL: &str = "failure_policy_classify_and_emit";
const FAILURE_POLICY_SET_ROLLBACK_KERNEL: &str = "failure_policy_set_rollback_marker";
const FAILURE_POLICY_APPLY_ROLLBACK_KERNEL: &str = "failure_policy_apply_rollback";
const LEARNED_DELTA_INIT_KERNEL: &str = "learned_delta_init";
const LEARNED_DELTA_FOLD_KERNEL: &str = "learned_delta_fold_receipt";
const LEARNED_DELTA_APPLY_KERNEL: &str = "learned_delta_apply";
const RECEIPT_PRIOR_SNAPSHOT_KERNEL: &str = "receipt_prior_snapshot";
const ORCH_TRACE_PUSH_KERNEL: &str = "orch_event_trace_push";
const ORCH_TRACE_DRAIN_KERNEL: &str = "orch_event_trace_drain";
const ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL: &str = "orch_check_no_duplicate_receipts";
const ORCH_CHECK_FRONTIER_VALID_KERNEL: &str = "orch_check_frontier_valid";
const ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL: &str = "orch_check_no_command_without_receipt";
const ORCH_REPLAY_SNAPSHOT_KERNEL: &str = "orch_replay_snapshot";
const ORCH_REPLAY_RESTORE_KERNEL: &str = "orch_replay_restore";

// Phase-2 orchestration step status codes (mirror ORCH_* defines in .cu).
pub const ORCH_CONTINUE: i32 = 0;
pub const ORCH_WAIT_EXTERNAL: i32 = 1;
pub const ORCH_HALTED: i32 = 2;
pub const ORCH_ERROR: i32 = 3;

// Phase-2 dispatch kind codes (mirror DISPATCH_KIND_* defines in .cu).
pub const DISPATCH_KIND_NONE: i32 = 0;
pub const DISPATCH_KIND_HF_DEVICE: i32 = 1;
pub const DISPATCH_KIND_HOST_FALLBACK: i32 = 2;
pub const DISPATCH_KIND_FUSION_ENTRY: i32 = 3;
pub const DISPATCH_KIND_ABSTRACT_DEVICE: i32 = 4;
pub const DISPATCH_KIND_EXTERNAL_PROCESS: i32 = 5;
pub const DISPATCH_KIND_EXTERNAL_IO: i32 = 6;
pub const DISPATCH_KIND_UNSUPPORTED: i32 = 7;

// Phase-4 control-flow operation codes (mirror CONTROL_OP_* in .cu).
pub const CONTROL_OP_SEQ: i32 = 0;
pub const CONTROL_OP_PAR: i32 = 1;
pub const CONTROL_OP_CHOICE: i32 = 2;
pub const CONTROL_OP_STAR_BOUNDED: i32 = 3;
pub const CONTROL_OP_GATE: i32 = 4;
pub const CONTROL_OP_HALT_OP: i32 = 5;

// GPU-native scheduler decision kind codes (mirror CONTROL_* in .cu).
pub const CONTROL_NONE: i32 = 0;
pub const CONTROL_SEQ_READY: i32 = 1;
pub const CONTROL_PAR_READY: i32 = 2;
pub const CONTROL_CHOICE_READY: i32 = 3;
pub const CONTROL_STAR_BODY_READY: i32 = 4;
pub const CONTROL_STAR_EXIT_READY: i32 = 5;
pub const CONTROL_HALT_READY: i32 = 6;
pub const CONTROL_BLOCKED: i32 = 7;

// Block-reason codes set on OrchestrationState (mirror ORCH_BLOCK_REASON_* in .cu).
pub const ORCH_BLOCK_REASON_NONE: i32 = 0;
pub const ORCH_BLOCK_REASON_NO_READY_NODE: i32 = 1;
pub const ORCH_BLOCK_REASON_STAR_EXHAUSTED: i32 = 2;
pub const ORCH_BLOCK_REASON_UNSUPPORTED: i32 = 3;
pub const ORCH_BLOCK_REASON_ALL_CONSUMED: i32 = 4;

/// Maximum number of control edges in a loaded control table.
pub const MAX_CONTROL_EDGES: usize = 1024;

/// Kernel name for zeroing the per-edge star counter buffer.
pub const STAR_COUNTERS_INIT_KERNEL: &str = "star_counters_init";

// Phase-5 failure classification codes (mirror FAILURE_CLASS_* in .cu).
pub const FAILURE_CLASS_SPAWN_FAILURE: i32 = 0;
pub const FAILURE_CLASS_TIMEOUT: i32 = 1;
pub const FAILURE_CLASS_SAFETY: i32 = 2;
pub const FAILURE_CLASS_CONTRACT: i32 = 3;
pub const FAILURE_CLASS_GPU_ERROR: i32 = 4;
pub const FAILURE_CLASS_UNKNOWN: i32 = 5;

// Phase-5 failure action codes (mirror FAILURE_ACTION_* in .cu).
pub const FAILURE_ACTION_RETRY: i32 = 0;
pub const FAILURE_ACTION_BLOCK: i32 = 1;
pub const FAILURE_ACTION_ROLLBACK: i32 = 2;
pub const FAILURE_ACTION_HALT: i32 = 3;
pub const FAILURE_ACTION_EXTERNAL_REPAIR: i32 = 4;

// Phase-5 repair dispatch kind (extends DISPATCH_KIND_* in .cu).
pub const DISPATCH_KIND_REPAIR: i32 = 8;

// Phase-6 learned-delta ring capacity (mirrors LEARNED_DELTA_RING_SIZE in .cu).
pub const LEARNED_DELTA_RING_SIZE: usize = 256;

// Phase-8 trace ring capacity and event kind codes (mirror ORCH_TRACE_RING_SIZE
// and ORCH_EVENT_* defines in .cu).
pub const ORCH_TRACE_RING_SIZE: usize = 512;
pub const ORCH_EVENT_STEP_COMMITTED: i32 = 0;
pub const ORCH_EVENT_WAIT_EXTERNAL: i32 = 1;
pub const ORCH_EVENT_HALTED: i32 = 2;
pub const ORCH_EVENT_BLOCKED: i32 = 3;
pub const ORCH_EVENT_RECEIPT_DRAINED: i32 = 4;

pub const MAX_PAR_GROUP_SIZE: usize = 8;

pub const DEVICE_RECEIPT_RING_SIZE: usize = 256;
pub const DEVICE_COMMAND_RING_SIZE: usize = 64;
pub const DEVICE_RECEIPT_EXT_RING_SIZE: usize = 256;
pub const GPU_HOT_REGION_COUNT: i32 = 8;
pub const PAR_DISPATCH_NONE: i32 = 0;
pub const PAR_DISPATCH_HF_DEVICE: i32 = 1;
pub const PAR_DISPATCH_HOST_FALLBACK: i32 = 2;
pub const PAR_DISPATCH_FUSION_ENTRY: i32 = 3;
pub const PAR_DISPATCH_ABSTRACT_DEVICE: i32 = 4;

pub const EXPLORATION_MAX_TOKENS: usize = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT;
pub const EXPLORATION_MAX_SELECTED: usize = TENSOR_NODE_COUNT;
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
    /// Phase-4: single-element scratch buffer for `control_flow_advance` output.
    pub control_op_out: CudaSlice<i32>,
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

/// GPU-resident data for the par-group-step kernel.
///
/// Built once at epoch start from the topology's compiled par groups and the
/// per-member dispatch metadata. Uploaded to the GPU device at construction.
pub struct ParGroupGpuData {
    pub(crate) table_buf: CudaSlice<i32>,
    /// Offset of each packed group record inside `table_buf`.
    ///
    /// `table_buf[group_offsets[g]]` is the group size and
    /// `table_buf[group_offsets[g] + 1]` is the first member tuple.  This lets
    /// the par-step kernel index groups directly instead of walking the
    /// variable-length table serially.
    pub(crate) group_offsets_buf: CudaSlice<i32>,
    pub num_groups: usize,
    /// Per-member slot table pointer array (shape: num_groups × MAX_PAR_GROUP_SIZE).
    /// Each entry is the device address of the `float**` pointer table for that
    /// member's hot region.  0 = no slot table (receipt-only / non-hot member).
    pub(crate) member_slot_table_ptrs: CudaSlice<u64>,
    /// Element count per member slot table (same shape as member_slot_table_ptrs).
    pub(crate) member_element_counts: CudaSlice<i32>,
    pub(crate) region_count: i32,
    /// Keeps the per-member `float**` device arrays alive for the epoch lifetime.
    #[allow(dead_code)]
    pub(crate) _slot_table_storage: Vec<CudaSlice<u64>>,
}

impl ParGroupGpuData {
    /// Build and upload par group data.
    ///
    /// `region_ids[g][i]` is the hot-region id for member `i` of group `g`, or
    /// `-1` when the member is not a hot-region operator.
    ///
    /// `is_gpu_dispatchable[g][i]` is `true` when the member has a GPU-native
    /// dispatch kind (H_f device or abstract-device receipt). The table is packed as
    /// `[g0_size, g0_n0, g0_r0, g0_e0, g0_k0, g0_n1, ...]`
    /// — `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuples.  The kernel computes
    /// eligibility on-device from the `is_gpu_dispatchable` flags rather than from
    /// a separate CPU-precomputed mask.
    ///
    /// `slot_registry` is used to build per-member `float**` slot pointer tables for
    /// hot-region members.  When a member's slots are registered, the kernel runs the
    /// `__device__` region function with real slot data (H_f path, D_h closed for that
    /// member).  When slots are absent the entry is 0 (receipt-only).
    pub fn build(
        dev: &Arc<CudaDevice>,
        groups: &[Vec<i32>],
        region_ids: &[Vec<i32>],
        is_gpu_dispatchable: &[Vec<bool>],
        dispatch_kinds: &[Vec<i32>],
        slot_registry: Option<&DeviceSlotRegistry>,
        fusion_hf_coverage: Option<&FusionHfCoverage>,
    ) -> Result<Self, CudaError> {
        use cudarc::driver::DevicePtr;

        assert_eq!(groups.len(), region_ids.len());
        assert_eq!(groups.len(), is_gpu_dispatchable.len());
        assert_eq!(groups.len(), dispatch_kinds.len());

        let num_groups = groups.len();
        let flat_size = (num_groups * MAX_PAR_GROUP_SIZE).max(1);

        // Allocate host-side flat arrays for per-member slot table pointers and
        // element counts.  Index: g * MAX_PAR_GROUP_SIZE + i.
        let mut slot_ptrs_host = vec![0u64; flat_size];
        let mut elem_counts_host = vec![0i32; flat_size];
        let mut slot_table_storage: Vec<CudaSlice<u64>> = Vec::new();

        if let Some(registry) = slot_registry {
            for (g, rids) in region_ids.iter().enumerate() {
                for (i, &rid) in rids.iter().enumerate() {
                    if rid < 0 {
                        continue;
                    }
                    let static_slots = gpu_region_slots(rid);
                    let generated_slots =
                        fusion_hf_coverage.and_then(|coverage| coverage.slots_for_region_id(rid));
                    let slot_refs: Vec<&str> = match (static_slots, generated_slots) {
                        (Some(slots), _) => slots.to_vec(),
                        (None, Some(slots)) => slots.iter().map(String::as_str).collect(),
                        (None, None) => continue,
                    };
                    if slot_refs.is_empty() {
                        continue;
                    }
                    match registry.device_slot_ptr_table(dev, &slot_refs) {
                        Ok((ptr_table, elem_count)) => {
                            let device_addr = *ptr_table.device_ptr();
                            slot_ptrs_host[g * MAX_PAR_GROUP_SIZE + i] = device_addr;
                            elem_counts_host[g * MAX_PAR_GROUP_SIZE + i] = elem_count;
                            // Transmute: CudaSlice<CUdeviceptr> is layout-equivalent to
                            // CudaSlice<u64> since CUdeviceptr = u64.
                            // Safety: cudarc guarantees CUdeviceptr = u64.
                            let raw: CudaSlice<u64> = unsafe { std::mem::transmute(ptr_table) };
                            slot_table_storage.push(raw);
                        }
                        Err(_) => { /* slots not registered; stays 0 (receipt-only) */ }
                    }
                }
            }
        }

        let member_slot_table_ptrs = dev
            .htod_copy(slot_ptrs_host)
            .map_err(|e| CudaError::new("htod par_group slot_table_ptrs", e))?;
        let member_element_counts = dev
            .htod_copy(elem_counts_host)
            .map_err(|e| CudaError::new("htod par_group element_counts", e))?;

        if groups.is_empty() {
            let table_buf = dev
                .htod_copy(vec![0_i32])
                .map_err(|e| CudaError::new("htod par_group table empty", e))?;
            let group_offsets_buf = dev
                .htod_copy(vec![0_i32])
                .map_err(|e| CudaError::new("htod par_group offsets empty", e))?;
            return Ok(Self {
                table_buf,
                group_offsets_buf,
                num_groups: 0,
                member_slot_table_ptrs,
                member_element_counts,
                region_count: fusion_hf_coverage
                    .map(FusionHfCoverage::region_count)
                    .unwrap_or(GPU_HOT_REGION_COUNT),
                _slot_table_storage: slot_table_storage,
            });
        }

        // Packed table: [g0_size, g0_n0, g0_r0, g0_e0, g0_k0, g0_n1, ...]
        let mut table: Vec<i32> = Vec::new();
        let mut group_offsets: Vec<i32> = Vec::with_capacity(num_groups);
        for (((group, rids), dispatchable), kinds) in groups
            .iter()
            .zip(region_ids.iter())
            .zip(is_gpu_dispatchable.iter())
            .zip(dispatch_kinds.iter())
        {
            group_offsets.push(table.len() as i32);
            table.push(group.len() as i32);
            for (((&node_id, &rid), &disp), &kind) in group
                .iter()
                .zip(rids.iter())
                .zip(dispatchable.iter())
                .zip(kinds.iter())
            {
                table.push(node_id);
                table.push(rid);
                table.push(disp as i32);
                table.push(kind);
            }
        }
        let table_buf = dev
            .htod_copy(table)
            .map_err(|e| CudaError::new("htod par_group table", e))?;
        let group_offsets_buf = dev
            .htod_copy(group_offsets)
            .map_err(|e| CudaError::new("htod par_group offsets", e))?;
        Ok(Self {
            table_buf,
            group_offsets_buf,
            num_groups,
            member_slot_table_ptrs,
            member_element_counts,
            region_count: fusion_hf_coverage
                .map(FusionHfCoverage::region_count)
                .unwrap_or(GPU_HOT_REGION_COUNT),
            _slot_table_storage: slot_table_storage,
        })
    }
}

/// C-compatible descriptor for one committed par-group member.
/// Must match the CUDA `ParDispatchDescriptor` definition exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParDispatchDescriptor {
    pub member_index: i32,
    pub node_id: i32,
    pub region_id: i32,
    pub dispatch_kind: i32,
    pub src_node: i32,
    pub dst_node: i32,
}

unsafe impl DeviceRepr for ParDispatchDescriptor {}

/// C-compatible output struct for `tensor_quantale_par_group_step`.
/// Must match the CUDA `ParGroupStepOutput` definition exactly.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct ParGroupStepOutputRaw {
    pub selected_group_idx: i32,
    pub group_size: i32,
    pub decisions: [DecisionReport; MAX_PAR_GROUP_SIZE],
    /// Hot-region id for each committed member; -1 when the member is not a hot region.
    pub region_ids: [i32; MAX_PAR_GROUP_SIZE],
    /// 1 when the member was dispatched in-kernel via the H_f path (Phase 2).
    /// CPU must skip execute_*_blocking and gpu_dispatch_region for those members.
    pub dispatched_on_device: [i32; MAX_PAR_GROUP_SIZE],
    /// Per-member dispatch descriptors emitted by the GPU. Non-H_f members keep
    /// explicit dispatch kinds so future tiers can consume descriptors without
    /// re-deriving member routing on the host.
    pub dispatch_descriptors: [ParDispatchDescriptor; MAX_PAR_GROUP_SIZE],
}

impl Default for ParGroupStepOutputRaw {
    fn default() -> Self {
        Self {
            selected_group_idx: -1,
            group_size: 0,
            decisions: [DecisionReport::default(); MAX_PAR_GROUP_SIZE],
            region_ids: [-1_i32; MAX_PAR_GROUP_SIZE],
            dispatched_on_device: [0_i32; MAX_PAR_GROUP_SIZE],
            dispatch_descriptors: [ParDispatchDescriptor::default(); MAX_PAR_GROUP_SIZE],
        }
    }
}

unsafe impl DeviceRepr for ParGroupStepOutputRaw {}

/// Host-side mirror of the CUDA `ParGroupHfParams` struct.
///
/// Uploaded as a single device word so the kernel parameter count stays within
/// the cudarc `LaunchAsync` arity limit.  All pointer fields are raw device
/// addresses (u64 = CUdeviceptr).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ParGroupHfParamsHost {
    pub slot_table_ptrs_dev: u64,
    pub element_counts_dev: u64,
    pub receipt_ring_dev: u64,
    pub ring_tail_dev: u64,
    pub ring_size: i32,
    pub region_count: i32,
}

unsafe impl DeviceRepr for ParGroupHfParamsHost {}

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

pub struct TensorQuantaleWorld {
    dev: Arc<CudaDevice>,
    tensor: CudaSlice<f32>,
    scratch: CudaSlice<f32>,
    witness: CudaSlice<i32>,
    scratch_witness: CudaSlice<i32>,
    consumed: CudaSlice<i32>,
    active: CudaSlice<i32>,
    next_active: CudaSlice<i32>,
    decision: CudaSlice<DecisionReport>,
    exploration_tokens: CudaSlice<ExplorationToken>,
    exploration_scores: CudaSlice<f32>,
    exploration_parents: CudaSlice<i32>,
    exploration_selected: CudaSlice<ExplorationCandidate>,
    exploration_token_count: CudaSlice<i32>,
    exploration_selected_count: CudaSlice<i32>,
    /// Invariant 23: CPU snapshot of the tensor taken immediately after the
    /// first embed_tensor_edges call.  Hard reset restores from this rather
    /// than re-uploading from the host edge list, so it works even when the
    /// original edge list is no longer in scope.
    base_tensor: Vec<f32>,
    /// Device-resident receipt ring for the GPU hot-dispatch path.
    device_receipt_buffer: DeviceReceiptBuffer,
    /// Phase-1 orchestration state block: persistent scheduler state, command
    /// ring, and extended receipt ring.  Zeroed at world creation; written by
    /// Phase-2+ orchestration kernels.
    orch_buffers: OrchestrationBuffers,
}

impl TensorQuantaleWorld {
    pub fn empty() -> Result<Self, CudaError> {
        Self::empty_with_kernel_source(assemble_kernel_source())
    }

    #[doc(hidden)]
    pub fn empty_with_generated_fusion_hf_fragments(
        generated_functions: &str,
    ) -> Result<Self, CudaError> {
        Self::empty_with_kernel_source(assemble_kernel_source_with_generated(generated_functions))
    }

    fn empty_with_kernel_source(kernel_source: String) -> Result<Self, CudaError> {
        let dev = CudaDevice::new(0).map_err(|error| CudaError::new("CudaDevice::new", error))?;
        let ptx =
            compile_ptx(kernel_source).map_err(|error| CudaError::new("compile_ptx", error))?;
        dev.load_ptx(
            ptx,
            MODULE_NAME,
            &[
                RESET_KERNEL,
                EMBED_KERNEL,
                CLOSURE_KERNEL,
                PROJECT_KERNEL,
                PROJECT_BATCH_KERNEL,
                COMMIT_BATCH_KERNEL,
                DECAY_KERNEL,
                EXPLORATION_SEED_KERNEL,
                EXPLORATION_EXPAND_KERNEL,
                EXPLORATION_SCORE_KERNEL,
                EXPLORATION_TOPK_KERNEL,
                EXPLORATION_COMMIT_KERNEL,
                JIT_CHAIN_SCORE_KERNEL,
                DRAIN_DEVICE_RECEIPTS_KERNEL,
                PUSH_DEVICE_RECEIPT_KERNEL,
                GPU_DISPATCH_KERNEL,
                RING_PUSH_KERNEL,
                RING_POP_KERNEL,
                PARALLEL_REDUCE_KERNEL,
                TOPK_BITONIC_KERNEL,
                PAR_GROUP_STEP_KERNEL,
                ORCH_STATE_INIT_KERNEL,
                ORCH_STATE_SNAPSHOT_KERNEL,
                DEVICE_CMD_RING_PUSH_KERNEL,
                DEVICE_RECEIPT_EXT_PUSH_KERNEL,
                DEVICE_RECEIPT_EXT_DRAIN_KERNEL,
                ORCHESTRATE_STEP_KERNEL,
                CONTROL_FLOW_ADVANCE_KERNEL,
                CHECK_EFFECTS_INDEPENDENT_KERNEL,
                FAILURE_POLICY_INIT_KERNEL,
                FAILURE_POLICY_CLASSIFY_KERNEL,
                FAILURE_POLICY_SET_ROLLBACK_KERNEL,
                FAILURE_POLICY_APPLY_ROLLBACK_KERNEL,
                LEARNED_DELTA_INIT_KERNEL,
                LEARNED_DELTA_FOLD_KERNEL,
                LEARNED_DELTA_APPLY_KERNEL,
                RECEIPT_PRIOR_SNAPSHOT_KERNEL,
                ORCH_TRACE_PUSH_KERNEL,
                ORCH_TRACE_DRAIN_KERNEL,
                ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL,
                ORCH_CHECK_FRONTIER_VALID_KERNEL,
                ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL,
                ORCH_REPLAY_SNAPSHOT_KERNEL,
                ORCH_REPLAY_RESTORE_KERNEL,
            ],
        )
        .map_err(|error| CudaError::new("load_ptx tensor", error))?;

        let tensor = dev
            .htod_copy(vec![0.0; TENSOR_LEN])
            .map_err(|error| CudaError::new("htod_copy tensor", error))?;
        let scratch = dev
            .htod_copy(vec![0.0; TENSOR_LEN])
            .map_err(|error| CudaError::new("htod_copy tensor scratch", error))?;
        let witness = dev
            .htod_copy(vec![-1_i32; TENSOR_LEN])
            .map_err(|error| CudaError::new("htod_copy tensor witness", error))?;
        let scratch_witness = dev
            .htod_copy(vec![-1_i32; TENSOR_LEN])
            .map_err(|error| CudaError::new("htod_copy tensor scratch_witness", error))?;
        let consumed = dev
            .htod_copy(vec![0_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy tensor consumed", error))?;
        let active = dev
            .htod_copy(vec![0_i32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy tensor active", error))?;
        let next_active = dev
            .htod_copy(vec![0_i32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy tensor next_active", error))?;
        let decision = dev
            .htod_copy(vec![DecisionReport::default()])
            .map_err(|error| CudaError::new("htod_copy tensor decision", error))?;
        let exploration_tokens = dev
            .htod_copy(vec![ExplorationToken::default(); EXPLORATION_MAX_TOKENS])
            .map_err(|error| CudaError::new("htod_copy exploration tokens", error))?;
        let exploration_scores = dev
            .htod_copy(vec![-COST_INFINITY; EXPLORATION_MAX_TOKENS])
            .map_err(|error| CudaError::new("htod_copy exploration scores", error))?;
        let exploration_parents = dev
            .htod_copy(vec![-1_i32; EXPLORATION_MAX_TOKENS])
            .map_err(|error| CudaError::new("htod_copy exploration parents", error))?;
        let exploration_selected = dev
            .htod_copy(vec![
                ExplorationCandidate::default();
                EXPLORATION_MAX_SELECTED
            ])
            .map_err(|error| CudaError::new("htod_copy exploration selected", error))?;
        let exploration_token_count = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy exploration token_count", error))?;
        let exploration_selected_count = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy exploration selected_count", error))?;

        let device_receipt_ring = dev
            .htod_copy(vec![DeviceReceipt::default(); DEVICE_RECEIPT_RING_SIZE])
            .map_err(|error| CudaError::new("htod_copy device_receipt_ring", error))?;
        let device_receipt_head = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy device_receipt_head", error))?;
        let device_receipt_tail = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy device_receipt_tail", error))?;

        // Phase-1: orchestration state block + command ring + extended receipt ring.
        let orch_state = dev
            .htod_copy(vec![OrchestrationState::default()])
            .map_err(|error| CudaError::new("htod_copy orch_state", error))?;
        let command_ring = dev
            .htod_copy(vec![DeviceCommand::default(); DEVICE_COMMAND_RING_SIZE])
            .map_err(|error| CudaError::new("htod_copy command_ring", error))?;
        let command_head = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy command_head", error))?;
        let command_tail = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy command_tail", error))?;
        let receipt_ext_ring = dev
            .htod_copy(vec![
                DeviceReceiptExt::default();
                DEVICE_RECEIPT_EXT_RING_SIZE
            ])
            .map_err(|error| CudaError::new("htod_copy receipt_ext_ring", error))?;
        let receipt_ext_head = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy receipt_ext_head", error))?;
        let receipt_ext_tail = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy receipt_ext_tail", error))?;

        // Phase-2: dispatch kind table (default HF_DEVICE for all nodes) + step status.
        let dispatch_kinds = dev
            .htod_copy(vec![DISPATCH_KIND_HF_DEVICE; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy dispatch_kinds", error))?;
        let reentrant_mask = dev
            .htod_copy(vec![0_i32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy reentrant_mask", error))?;
        let step_status = dev
            .htod_copy(vec![ORCH_CONTINUE])
            .map_err(|error| CudaError::new("htod_copy step_status", error))?;
        let default_bias = dev
            .htod_copy(vec![ProjectionBias::default()])
            .map_err(|error| CudaError::new("htod_copy default_bias", error))?;

        // Phase-4: control-flow tables. Sentinel edge (lhs=-1, rhs=-1) ensures
        // find_matching_control_edge returns -1 until real patterns are loaded.
        let ctrl_sentinel = ControlEdge {
            op: CONTROL_OP_HALT_OP,
            lhs: -1,
            rhs: -1,
            guard: 0,
            order: 0,
            bound: 0,
        };
        let control_edges = dev
            .htod_copy(vec![ctrl_sentinel])
            .map_err(|error| CudaError::new("htod_copy control_edges", error))?;
        let effect_table = dev
            .htod_copy(vec![EffectTable::default()])
            .map_err(|error| CudaError::new("htod_copy effect_table", error))?;
        let control_op_out = dev
            .htod_copy(vec![-1_i32])
            .map_err(|error| CudaError::new("htod_copy control_op_out", error))?;

        // Phase-5: failure policy table, rollback snapshot buffers, action scratch.
        let failure_policies = dev
            .htod_copy(vec![FailurePolicy::default(); TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy failure_policies", error))?;
        let rollback_consumed = dev
            .htod_copy(vec![0_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy rollback_consumed", error))?;
        let rollback_active = dev
            .htod_copy(vec![0_i32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy rollback_active", error))?;
        let failure_action_out = dev
            .htod_copy(vec![FAILURE_ACTION_BLOCK])
            .map_err(|error| CudaError::new("htod_copy failure_action_out", error))?;

        // Phase-6: receipt prior table, learned-delta ring, export snapshot.
        let receipt_priors = dev
            .htod_copy(vec![0.0_f32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy receipt_priors", error))?;
        let learned_delta_ring = dev
            .htod_copy(vec![LearnedDelta::default(); LEARNED_DELTA_RING_SIZE])
            .map_err(|error| CudaError::new("htod_copy learned_delta_ring", error))?;
        let learned_delta_head = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy learned_delta_head", error))?;
        let learned_delta_tail = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy learned_delta_tail", error))?;
        let receipt_prior_snapshot_buf = dev
            .htod_copy(vec![0.0_f32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy receipt_prior_snapshot_buf", error))?;

        // Phase-8 allocations.
        let trace_ring = dev
            .htod_copy(vec![OrchestrationEvent::default(); ORCH_TRACE_RING_SIZE])
            .map_err(|error| CudaError::new("htod_copy trace_ring", error))?;
        let trace_head = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy trace_head", error))?;
        let trace_tail = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy trace_tail", error))?;
        let trace_drain_buf = dev
            .htod_copy(vec![OrchestrationEvent::default(); ORCH_TRACE_RING_SIZE])
            .map_err(|error| CudaError::new("htod_copy trace_drain_buf", error))?;
        let trace_drain_count = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy trace_drain_count", error))?;
        let orch_violation_out = dev
            .htod_copy(vec![0_i32])
            .map_err(|error| CudaError::new("htod_copy orch_violation_out", error))?;
        let replay_state = dev
            .htod_copy(vec![OrchestrationState::default()])
            .map_err(|error| CudaError::new("htod_copy replay_state", error))?;
        let replay_consumed = dev
            .htod_copy(vec![0_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy replay_consumed", error))?;
        let replay_active = dev
            .htod_copy(vec![0_i32; TENSOR_NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy replay_active", error))?;
        // GPU-native control-flow: per-edge star counter buffers.
        let star_counters = dev
            .htod_copy(vec![0_i32; MAX_CONTROL_EDGES])
            .map_err(|error| CudaError::new("htod_copy star_counters", error))?;
        let replay_star_counters = dev
            .htod_copy(vec![0_i32; MAX_CONTROL_EDGES])
            .map_err(|error| CudaError::new("htod_copy replay_star_counters", error))?;

        let mut world = Self {
            dev,
            tensor,
            scratch,
            witness,
            scratch_witness,
            consumed,
            active,
            next_active,
            decision,
            exploration_tokens,
            exploration_scores,
            exploration_parents,
            exploration_selected,
            exploration_token_count,
            exploration_selected_count,
            base_tensor: Vec::new(),
            device_receipt_buffer: DeviceReceiptBuffer {
                ring: device_receipt_ring,
                head: device_receipt_head,
                tail: device_receipt_tail,
            },
            orch_buffers: OrchestrationBuffers {
                state: orch_state,
                command_ring,
                command_head,
                command_tail,
                receipt_ext_ring,
                receipt_ext_head,
                receipt_ext_tail,
                dispatch_kinds,
                reentrant_mask,
                step_status,
                default_bias,
                control_edges,
                effect_table,
                control_op_out,
                failure_policies,
                rollback_consumed,
                rollback_active,
                failure_action_out,
                receipt_priors,
                learned_delta_ring,
                learned_delta_head,
                learned_delta_tail,
                receipt_prior_snapshot_buf,
                trace_ring,
                trace_head,
                trace_tail,
                trace_drain_buf,
                trace_drain_count,
                orch_violation_out,
                replay_state,
                replay_consumed,
                replay_active,
                star_counters,
                replay_star_counters,
            },
        };
        world.orch_state_init()?;
        world.reset()?;
        Ok(world)
    }

    pub fn from_tensor_edges(edges: &[TensorEdge]) -> Result<Self, CudaError> {
        let mut world = Self::empty()?;
        world.embed_tensor_edges(edges)?;
        world.snapshot_base_tensor()?;
        Ok(world)
    }

    /// Invariant 23: take a CPU snapshot of the current tensor state.
    ///
    /// Called once by `from_tensor_edges` after the initial embed.  The
    /// snapshot is used by `restore_base_tensor` to perform a clean hard reset
    /// without needing the original edge list in scope.
    pub fn snapshot_base_tensor(&mut self) -> Result<(), CudaError> {
        self.base_tensor = self.tensor()?;
        Ok(())
    }

    /// Invariant 23: restore the tensor to its post-embed baseline and reset
    /// all runtime state (active[], consumed[], decision[]).
    ///
    /// Prefer this over `reset() + embed_tensor_edges()` for hard resets
    /// because it restores from a known-good snapshot rather than trying to
    /// lift a potentially broken `W_t`.
    pub fn restore_base_tensor(&mut self) -> Result<(), CudaError> {
        if self.base_tensor.is_empty() {
            return Err(CudaError::invalid_input(
                "no base tensor snapshot; use from_tensor_edges to create the world",
            ));
        }
        // Reset clears active[], consumed[], decision[], scratch, witness.
        self.reset()?;
        // Overwrite the zeroed tensor with the base snapshot.
        self.tensor = self
            .dev
            .htod_copy(self.base_tensor.clone())
            .map_err(|error| CudaError::new("htod_copy tensor base restore", error))?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, RESET_KERNEL)
            .ok_or(CudaError::missing_function(RESET_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.scratch,
                    &mut self.witness,
                    &mut self.scratch_witness,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_reset", error))
    }

    pub fn embed_tensor_edges(&mut self, edges: &[TensorEdge]) -> Result<(), CudaError> {
        let edge_count = i32::try_from(edges.len())
            .map_err(|_| CudaError::invalid_input("too many tensor edges"))?;
        let edge_buffer = self
            .dev
            .htod_copy(edges.to_vec())
            .map_err(|error| CudaError::new("htod_copy tensor edges", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, EMBED_KERNEL)
            .ok_or(CudaError::missing_function(EMBED_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.witness,
                    &edge_buffer,
                    edge_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_embed_edges", error))
    }

    pub fn close(&mut self) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, CLOSURE_KERNEL)
            .ok_or(CudaError::missing_function(CLOSURE_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (&mut self.tensor, &mut self.scratch, &mut self.witness),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_closure", error))
    }

    pub fn project(&mut self, bias: ProjectionBias) -> Result<DecisionReport, CudaError> {
        let bias_buffer = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|error| CudaError::new("htod_copy projection bias", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PROJECT_KERNEL)
            .ok_or(CudaError::missing_function(PROJECT_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &self.witness,
                    &self.consumed,
                    &self.active,
                    &bias_buffer,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_project", error))?;
        self.decision_report()
    }

    pub fn project_parallel_group(
        &mut self,
        group_nodes: &[i32],
        bias: ProjectionBias,
    ) -> Result<Vec<DecisionReport>, CudaError> {
        if group_nodes.len() < 2 {
            return Err(CudaError::invalid_input(
                "parallel projection requires at least two group nodes",
            ));
        }
        if group_nodes.len() > TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(
                "parallel projection group too large",
            ));
        }
        for node in group_nodes {
            if Node::decode(*node, &bundled_registry()?).is_none() {
                return Err(CudaError::invalid_input(format!(
                    "invalid parallel projection node {node}"
                )));
            }
        }

        let group_len = i32::try_from(group_nodes.len())
            .map_err(|_| CudaError::invalid_input("parallel projection group too large"))?;
        let bias_buffer = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|error| CudaError::new("htod_copy tensor batch bias", error))?;
        let group_buffer = self
            .dev
            .htod_copy(group_nodes.to_vec())
            .map_err(|error| CudaError::new("htod_copy tensor batch group", error))?;
        let mut out_decisions = self
            .dev
            .htod_copy(vec![DecisionReport::default(); group_nodes.len()])
            .map_err(|error| CudaError::new("htod_copy tensor batch decisions", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PROJECT_BATCH_KERNEL)
            .ok_or(CudaError::missing_function(PROJECT_BATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &self.witness,
                    &self.consumed,
                    &self.active,
                    &bias_buffer,
                    &group_buffer,
                    group_len,
                    &self.decision,
                    &mut out_decisions,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_project_batch", error))?;
        self.dev
            .dtoh_sync_copy(&out_decisions)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor batch decisions", error))
    }

    pub fn commit_decision_batch(&mut self, decisions: &[DecisionReport]) -> Result<(), CudaError> {
        if decisions.len() < 2 {
            return Err(CudaError::invalid_input(
                "decision batch commit requires at least two decisions",
            ));
        }
        if decisions.len() > TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input("decision batch is too large"));
        }
        if decisions
            .iter()
            .any(|decision| decision.blocked != 0 || decision.halted != 0)
        {
            return Err(CudaError::invalid_input(
                "cannot commit blocked or halted decision batch",
            ));
        }
        for decision in decisions {
            let registry = bundled_registry()?;
            if Node::decode(decision.selected_src, &registry).is_none()
                || Node::decode(decision.first_hop, &registry).is_none()
            {
                return Err(CudaError::invalid_input(
                    "cannot commit decision batch with invalid node IDs",
                ));
            }
        }

        let decision_count = i32::try_from(decisions.len())
            .map_err(|_| CudaError::invalid_input("decision batch is too large"))?;
        let decision_buffer = self
            .dev
            .htod_copy(decisions.to_vec())
            .map_err(|error| CudaError::new("htod_copy tensor batch commit decisions", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, COMMIT_BATCH_KERNEL)
            .ok_or(CudaError::missing_function(COMMIT_BATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &decision_buffer,
                    decision_count,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_commit_batch", error))
    }

    /// Build and upload par group GPU data using this world's CUDA device.
    ///
    /// `groups` are the compiled par groups (node ID lists).
    /// `region_ids[g][i]` is the hot-region id for member `i` (-1 if not hot).
    /// `is_gpu_dispatchable[g][i]` is `true` for GPU-native dispatch members.
    /// `dispatch_kinds[g][i]` is the initial descriptor kind the kernel should
    /// emit for each committed member; H_f dispatch may upgrade it in-kernel.
    /// Eligibility is encoded per-member in the table and validated by the kernel.
    ///
    /// `slot_registry` provides `float**` device slot tables for hot-region members
    /// so that Phase 2 of `par_group_step` can call region functions with real data.
    /// Pass `None` to use receipt-only mode for all members.
    pub fn make_par_group_data(
        &self,
        groups: &[Vec<i32>],
        region_ids: &[Vec<i32>],
        is_gpu_dispatchable: &[Vec<bool>],
        dispatch_kinds: &[Vec<i32>],
        slot_registry: Option<&DeviceSlotRegistry>,
        fusion_hf_coverage: Option<&FusionHfCoverage>,
    ) -> Result<ParGroupGpuData, CudaError> {
        ParGroupGpuData::build(
            &self.dev,
            groups,
            region_ids,
            is_gpu_dispatchable,
            dispatch_kinds,
            slot_registry,
            fusion_hf_coverage,
        )
    }

    /// GPU-native parallel group step: select the first eligible, all-ready CKA
    /// par group, commit it on-device, and return the committed decisions.
    ///
    /// Returns `Ok(None)` when no group is ready — tensor state is unchanged.
    /// Returns `Ok(Some((group_idx, decisions, region_ids, dispatched_on_device, descriptors)))`.
    ///
    /// `dispatched_on_device[i] == 1` means member `i` was dispatched in-kernel
    /// via the H_f path: the region function ran on-device and its DeviceReceipt
    /// was written to the ring.  The CPU must skip `execute_*_blocking` and
    /// `gpu_dispatch_region` for those members (call `drain_device_receipts` only).
    pub fn par_group_step(
        &mut self,
        data: &ParGroupGpuData,
        bias: ProjectionBias,
    ) -> Result<
        Option<(
            usize,
            Vec<DecisionReport>,
            Vec<i32>,
            Vec<i32>,
            Vec<ParDispatchDescriptor>,
        )>,
        CudaError,
    > {
        if data.num_groups == 0 {
            return Ok(None);
        }
        let bias_buf = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|e| CudaError::new("htod par_group bias", e))?;
        let mut out_buf = self
            .dev
            .htod_copy(vec![ParGroupStepOutputRaw::default()])
            .map_err(|e| CudaError::new("htod par_group output", e))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PAR_GROUP_STEP_KERNEL)
            .ok_or(CudaError::missing_function(PAR_GROUP_STEP_KERNEL))?;
        use cudarc::driver::DevicePtr;
        let hf_params = ParGroupHfParamsHost {
            slot_table_ptrs_dev: *data.member_slot_table_ptrs.device_ptr(),
            element_counts_dev: *data.member_element_counts.device_ptr(),
            receipt_ring_dev: *self.device_receipt_buffer.ring.device_ptr(),
            ring_tail_dev: *self.device_receipt_buffer.tail.device_ptr(),
            ring_size: DEVICE_RECEIPT_RING_SIZE as i32,
            region_count: data.region_count,
        };
        let hf_buf = self
            .dev
            .htod_copy(vec![hf_params])
            .map_err(|e| CudaError::new("htod par_group hf_params", e))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &self.witness,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &bias_buf,
                    &mut self.decision,
                    &data.group_offsets_buf,
                    &data.table_buf,
                    data.num_groups as i32,
                    &mut out_buf,
                    &hf_buf,
                ),
            )
        }
        .map_err(|e| CudaError::new(PAR_GROUP_STEP_KERNEL, e))?;
        let output = self
            .dev
            .dtoh_sync_copy(&out_buf)
            .map_err(|e| CudaError::new("dtoh par_group output", e))?;
        let raw = &output[0];
        if raw.selected_group_idx < 0 {
            return Ok(None);
        }
        let sz = (raw.group_size as usize).min(MAX_PAR_GROUP_SIZE);
        Ok(Some((
            raw.selected_group_idx as usize,
            raw.decisions[..sz].to_vec(),
            raw.region_ids[..sz].to_vec(),
            raw.dispatched_on_device[..sz].to_vec(),
            raw.dispatch_descriptors[..sz].to_vec(),
        )))
    }

    /// Drain all `DeviceReceipt`s in the device ring buffer on-device.
    ///
    /// The GPU reads the ring and applies confidence/cost/
    /// safety atomics directly.
    pub fn drain_device_receipts(&mut self) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DRAIN_DEVICE_RECEIPTS_KERNEL)
            .ok_or(CudaError::missing_function(DRAIN_DEVICE_RECEIPTS_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &self.device_receipt_buffer.ring,
                    ring_size,
                    &mut self.device_receipt_buffer.head,
                    &self.device_receipt_buffer.tail,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_drain_device_receipts", error))
    }

    /// Push a generic execution receipt into the device receipt ring.
    ///
    /// This is for GPU-dispatched work that is not a registered hot region, such
    /// as a batched fusion JIT kernel. Call `drain_device_receipts` afterwards
    /// to apply the tensor update on-device.
    pub fn push_device_receipt(
        &mut self,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PUSH_DEVICE_RECEIPT_KERNEL)
            .ok_or(CudaError::missing_function(PUSH_DEVICE_RECEIPT_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_id,
                    src_node,
                    dst_node,
                    outcome,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_push_device_receipt", error))
    }

    // ── Phase-1 orchestration state wrappers ─────────────────────────────────

    /// Launch `orchestration_state_init` to zero the device-resident state block.
    /// Called once at world construction.
    pub fn orch_state_init(&mut self) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCH_STATE_INIT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_STATE_INIT_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.orch_buffers.state,)) }
            .map_err(|error| CudaError::new(ORCH_STATE_INIT_KERNEL, error))
    }

    /// Copy the live device orchestration state into a snapshot and return it.
    pub fn orch_state_snapshot(&mut self) -> Result<OrchestrationState, CudaError> {
        let mut snapshot = self
            .dev
            .htod_copy(vec![OrchestrationState::default()])
            .map_err(|error| CudaError::new("htod_copy orch_snapshot", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCH_STATE_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_STATE_SNAPSHOT_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&self.orch_buffers.state, &mut snapshot)) }
            .map_err(|error| CudaError::new(ORCH_STATE_SNAPSHOT_KERNEL, error))?;
        let result = self
            .dev
            .dtoh_sync_copy(&snapshot)
            .map_err(|error| CudaError::new("dtoh_sync_copy orch_snapshot", error))?;
        Ok(result[0])
    }

    /// Push one `DeviceCommand` into the device command ring.
    /// Returns `Err` if the ring is full (capacity = `DEVICE_COMMAND_RING_SIZE`).
    pub fn push_device_command(&mut self, cmd: DeviceCommand) -> Result<(), CudaError> {
        let ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_CMD_RING_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_CMD_RING_PUSH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    ring_size,
                    cmd,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_CMD_RING_PUSH_KERNEL, error))
    }

    /// Drain the device command ring to the host and return all valid commands.
    pub fn drain_device_commands(&mut self) -> Result<Vec<DeviceCommand>, CudaError> {
        let head = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_head)
            .map_err(|error| CudaError::new("dtoh command_head", error))?[0];
        let tail = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_tail)
            .map_err(|error| CudaError::new("dtoh command_tail", error))?[0];
        if head == tail {
            return Ok(Vec::new());
        }
        let ring = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_ring)
            .map_err(|error| CudaError::new("dtoh command_ring", error))?;
        let ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let mut out = Vec::new();
        let mut h = head;
        while h != tail {
            let cmd = ring[(h % ring_size) as usize];
            if cmd.valid != 0 {
                out.push(cmd);
            }
            h += 1;
        }
        // Advance head to match tail on the device.
        self.orch_buffers.command_head = self
            .dev
            .htod_copy(vec![tail])
            .map_err(|error| CudaError::new("htod command_head advance", error))?;
        Ok(out)
    }

    /// Push one `DeviceReceiptExt` into the extended receipt ring.
    pub fn push_device_receipt_ext(&mut self, receipt: DeviceReceiptExt) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_RECEIPT_EXT_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_RECEIPT_EXT_PUSH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.orch_buffers.receipt_ext_ring,
                    &mut self.orch_buffers.receipt_ext_tail,
                    &self.orch_buffers.receipt_ext_head,
                    ring_size,
                    receipt,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_RECEIPT_EXT_PUSH_KERNEL, error))
    }

    /// Drain the extended receipt ring on-device, applying tensor updates.
    pub fn drain_device_receipt_ext(&mut self) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_RECEIPT_EXT_DRAIN_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_RECEIPT_EXT_DRAIN_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.orch_buffers.receipt_ext_ring,
                    ring_size,
                    &mut self.orch_buffers.receipt_ext_head,
                    &self.orch_buffers.receipt_ext_tail,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_RECEIPT_EXT_DRAIN_KERNEL, error))
    }

    // ── Phase-2 orchestration step wrappers ──────────────────────────────────

    /// Upload a dispatch-kind table to the device.
    /// `kinds` must have length `TENSOR_NODE_COUNT`; each entry is one of the
    /// `DISPATCH_KIND_*` constants.
    pub fn set_dispatch_kinds(&mut self, kinds: &[i32]) -> Result<(), CudaError> {
        if kinds.len() != TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "set_dispatch_kinds: expected {TENSOR_NODE_COUNT} entries, got {}",
                kinds.len()
            )));
        }
        self.orch_buffers.dispatch_kinds = self
            .dev
            .htod_copy(kinds.to_vec())
            .map_err(|error| CudaError::new("htod_copy dispatch_kinds update", error))?;
        Ok(())
    }

    /// Upload a node-level reentrant-consumption mask.
    /// `mask[id] != 0` means edges incident to that node are not one-shot.
    pub fn set_reentrant_mask(&mut self, mask: &[i32]) -> Result<(), CudaError> {
        if mask.len() != TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "set_reentrant_mask: expected {TENSOR_NODE_COUNT} entries, got {}",
                mask.len()
            )));
        }
        self.orch_buffers.reentrant_mask = self
            .dev
            .htod_copy(mask.to_vec())
            .map_err(|error| CudaError::new("htod_copy reentrant_mask update", error))?;
        Ok(())
    }

    /// Launch one `tensor_quantale_orchestrate_step` and return the status.
    ///
    /// The kernel:
    ///   1. Drains the extended receipt ring.
    ///   2. Selects the next ready node (singleton path).
    ///   3. Commits consumed/active state for GPU-native nodes.
    ///   4. Emits a `DeviceCommand` for external-dispatch nodes.
    ///   5. Returns `OrchStepStatus` to the host.
    pub fn orchestrate_step(&mut self) -> Result<OrchStepStatus, CudaError> {
        use cudarc::driver::DevicePtr;

        let ctrl_edge_count = self.orch_buffers.control_edges.len() as i32;
        let effect_count = self.orch_buffers.effect_table.len() as i32;
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        let bundle = TensorWorldBundleHost {
            tensor_dev: *self.tensor.device_ptr() as u64,
            witness_dev: *self.witness.device_ptr() as u64,
            consumed_dev: *self.consumed.device_ptr() as u64,
            active_dev: *self.active.device_ptr() as u64,
            next_active_dev: *self.next_active.device_ptr() as u64,
            reentrant_mask_dev: *self.orch_buffers.reentrant_mask.device_ptr() as u64,
            bias_dev: *self.orch_buffers.default_bias.device_ptr() as u64,
            decision_dev: *self.decision.device_ptr() as u64,
            control_edges_dev: *self.orch_buffers.control_edges.device_ptr() as u64,
            control_edge_count: ctrl_edge_count,
            effects_dev: *self.orch_buffers.effect_table.device_ptr() as u64,
            effect_count,
            star_counters_dev: *self.orch_buffers.star_counters.device_ptr() as u64,
            star_counter_count,
        };

        let bundle_dev = self
            .dev
            .htod_copy(vec![bundle])
            .map_err(|error| CudaError::new("htod_copy TensorWorldBundle", error))?;

        let cmd_ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let ext_ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;

        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCHESTRATE_STEP_KERNEL)
            .ok_or(CudaError::missing_function(ORCHESTRATE_STEP_KERNEL))?;

        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &bundle_dev,
                    &mut self.orch_buffers.state,
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    cmd_ring_size,
                    &mut self.orch_buffers.receipt_ext_ring,
                    &mut self.orch_buffers.receipt_ext_head,
                    &self.orch_buffers.receipt_ext_tail,
                    ext_ring_size,
                    &self.orch_buffers.dispatch_kinds,
                    &mut self.orch_buffers.step_status,
                ),
            )
        }
        .map_err(|error| CudaError::new(ORCHESTRATE_STEP_KERNEL, error))?;

        let status_vec = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.step_status)
            .map_err(|error| CudaError::new("dtoh step_status", error))?;
        Ok(OrchStepStatus::from_code(status_vec[0]))
    }

    // ── Phase-7 supervisor loop ───────────────────────────────────────────────

    /// Run the GPU scheduler for up to `max_steps` iterations without CPU
    /// involvement in per-step decisions.
    ///
    /// Returns as soon as one of the following occurs:
    /// - `ORCH_WAIT_EXTERNAL`: GPU has emitted an external command and is
    ///   waiting for the CPU service to respond.
    /// - `ORCH_HALTED`: the graph reached its halt node or `state.halted` was
    ///   set by the failure policy.
    /// - `ORCH_ERROR`: internal kernel error (unrecoverable).
    /// - `state.blocked == 1`: no ready singleton found; returns `Continue`
    ///   so the outer supervisor loop can decide on repair or shutdown.
    ///
    /// Returns `Continue` after exhausting `max_steps` without hitting a stop
    /// condition, giving the CPU a chance to service external commands or
    /// apply learned deltas between bursts.
    ///
    /// This is the Phase-7 host-loop demotion entry point:
    /// ```text
    /// loop {
    ///     match world.orchestrate_until_wait_or_halt(max_steps)? {
    ///         Continue      => continue,
    ///         WaitExternal  => service_external_commands(&mut world)?,
    ///         Halted        => break,
    ///         Error         => { snapshot(); break; }
    ///     }
    /// }
    /// ```
    pub fn orchestrate_until_wait_or_halt(
        &mut self,
        max_steps: u32,
    ) -> Result<OrchStepStatus, CudaError> {
        for _ in 0..max_steps {
            let status = self.orchestrate_step()?;
            match status {
                OrchStepStatus::Continue => {
                    // Blocked check: if the scheduler found no ready node,
                    // yield back to the CPU rather than spin.
                    let state = self.orch_state_snapshot()?;
                    if state.blocked != 0 {
                        return Ok(OrchStepStatus::Continue);
                    }
                }
                OrchStepStatus::WaitExternal | OrchStepStatus::Halted | OrchStepStatus::Error => {
                    return Ok(status);
                }
            }
        }
        Ok(OrchStepStatus::Continue)
    }

    // ── Phase-4 control-flow methods ─────────────────────────────────────────

    /// Upload a lowered pattern control table to the device.
    ///
    /// If `edges` is empty the device control table retains its current content.
    /// If `effects` is empty the device effect table retains its current content.
    pub fn load_control_table(
        &mut self,
        edges: Vec<ControlEdge>,
        effects: Vec<EffectTable>,
    ) -> Result<(), CudaError> {
        if !edges.is_empty() {
            let edge_count = edges.len();
            self.orch_buffers.control_edges = self
                .dev
                .htod_copy(edges)
                .map_err(|e| CudaError::new("load_control_table edges", e))?;
            // Resize star_counters to match edge count (capped at MAX_CONTROL_EDGES).
            let counter_len = edge_count.min(MAX_CONTROL_EDGES);
            let init_count = counter_len as i32;
            self.orch_buffers.star_counters = self
                .dev
                .htod_copy(vec![0_i32; counter_len])
                .map_err(|e| CudaError::new("load_control_table star_counters", e))?;
            self.orch_buffers.replay_star_counters =
                self.dev
                    .htod_copy(vec![0_i32; counter_len])
                    .map_err(|e| CudaError::new("load_control_table replay_star_counters", e))?;
            // Zero-init via dedicated kernel for consistency.
            if let Some(f) = self.dev.get_func(MODULE_NAME, STAR_COUNTERS_INIT_KERNEL) {
                unsafe {
                    f.launch(
                        kernel_config(),
                        (&mut self.orch_buffers.star_counters, init_count),
                    )
                }
                .map_err(|e| CudaError::new(STAR_COUNTERS_INIT_KERNEL, e))?;
            }
        }
        if !effects.is_empty() {
            self.orch_buffers.effect_table = self
                .dev
                .htod_copy(effects)
                .map_err(|e| CudaError::new("load_control_table effects", e))?;
        }
        Ok(())
    }

    /// Look up the control-flow op for a selected (src, dst) edge.
    ///
    /// Returns the matched `CONTROL_OP_*` code, or `-1` if no matching edge is
    /// found in the current control table.  Advances the bounded-star counter
    /// in `OrchestrationState` when the matching edge is `CONTROL_OP_STAR_BOUNDED`.
    /// Zero all per-edge star counters on device.
    ///
    /// Call this when entering a new STAR loop scope so iteration counts start
    /// fresh.  Also called automatically by `load_control_table`.
    pub fn star_counters_reset(&mut self) -> Result<(), CudaError> {
        let count = self.orch_buffers.star_counters.len() as i32;
        if count == 0 {
            return Ok(());
        }
        let f = self
            .dev
            .get_func(MODULE_NAME, STAR_COUNTERS_INIT_KERNEL)
            .ok_or(CudaError::missing_function(STAR_COUNTERS_INIT_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (&mut self.orch_buffers.star_counters, count),
            )
        }
        .map_err(|e| CudaError::new(STAR_COUNTERS_INIT_KERNEL, e))
    }

    /// Look up and advance the matching control edge for (src, dst).
    ///
    /// Prefer `orchestrate_step` for all runtime use.  This side-path kernel
    /// mutates `OrchestrationState` directly and bypasses the scheduler's
    /// deterministic selection logic.  It is retained only for legacy tests;
    /// new tests should observe control-flow behavior through `orchestrate_step`.
    #[deprecated(
        since = "0.0.0",
        note = "use orchestrate_step; this side-path bypasses scheduler selection"
    )]
    pub fn control_flow_advance(&mut self, src: i32, dst: i32) -> Result<i32, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, CONTROL_FLOW_ADVANCE_KERNEL)
            .ok_or(CudaError::missing_function(CONTROL_FLOW_ADVANCE_KERNEL))?;
        self.orch_buffers.control_op_out = self
            .dev
            .htod_copy(vec![-1_i32])
            .map_err(|e| CudaError::new("reset control_op_out", e))?;
        let edge_count = self.orch_buffers.control_edges.len() as i32;
        let effect_count = self.orch_buffers.effect_table.len() as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.orch_buffers.control_edges,
                    edge_count,
                    &self.orch_buffers.effect_table,
                    effect_count,
                    &mut self.orch_buffers.state,
                    src,
                    dst,
                    &mut self.orch_buffers.control_op_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(CONTROL_FLOW_ADVANCE_KERNEL, e))?;
        let out = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.control_op_out)
            .map_err(|e| CudaError::new("dtoh control_op_out", e))?;
        Ok(out[0])
    }

    /// Check whether nodes `a` and `b` are par-eligible (effect-independent)
    /// according to the current device effect table.
    pub fn check_effects_independent(&mut self, a: i32, b: i32) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, CHECK_EFFECTS_INDEPENDENT_KERNEL)
            .ok_or(CudaError::missing_function(
                CHECK_EFFECTS_INDEPENDENT_KERNEL,
            ))?;
        let mut out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod_copy effects_out", e))?;
        let n_effects = self.orch_buffers.effect_table.len() as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (&self.orch_buffers.effect_table, n_effects, a, b, &mut out),
            )
        }
        .map_err(|e| CudaError::new(CHECK_EFFECTS_INDEPENDENT_KERNEL, e))?;
        let result = self
            .dev
            .dtoh_sync_copy(&out)
            .map_err(|e| CudaError::new("dtoh effects_out", e))?;
        Ok(result[0] != 0)
    }

    // ── Phase-5 failure policy wrappers ──────────────────────────────────────

    /// Initialise the per-node failure policy table on the GPU.
    ///
    /// Every node receives `default_budget` retries.  -1 = unlimited.
    /// `default_block_threshold` sets consecutive-block limit; -1 = disabled.
    /// Call this once after world construction to arm retry budgets.
    pub fn failure_policy_init(
        &mut self,
        default_budget: i32,
        default_block_threshold: i32,
    ) -> Result<(), CudaError> {
        let n_nodes = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_INIT_KERNEL)
            .ok_or(CudaError::missing_function(FAILURE_POLICY_INIT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (
                    &mut self.orch_buffers.failure_policies,
                    n_nodes,
                    default_budget,
                    default_block_threshold,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_INIT_KERNEL, e))
    }

    /// Classify a receipt failure, update the per-node retry budget, and choose
    /// the corrective action on-device.
    ///
    /// Returns the `FAILURE_ACTION_*` code written by the kernel.  When the
    /// action is `FAILURE_ACTION_EXTERNAL_REPAIR` a repair `DeviceCommand` is
    /// also pushed into the device command ring.
    pub fn failure_policy_classify_and_emit(
        &mut self,
        outcome: i32,
        node_id: i32,
        src: i32,
        dst: i32,
        command_id: i32,
    ) -> Result<i32, CudaError> {
        self.orch_buffers.failure_action_out = self
            .dev
            .htod_copy(vec![FAILURE_ACTION_BLOCK])
            .map_err(|e| CudaError::new("reset failure_action_out", e))?;
        let req = FailureClassifyRequest {
            outcome,
            node_id,
            src,
            dst,
            command_id,
        };
        let req_buf = self
            .dev
            .htod_copy(vec![req])
            .map_err(|e| CudaError::new("htod failure_classify_req", e))?;
        let n_policies = self.orch_buffers.failure_policies.len() as i32;
        let cmd_ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_CLASSIFY_KERNEL)
            .ok_or(CudaError::missing_function(FAILURE_POLICY_CLASSIFY_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &req_buf,
                    &mut self.orch_buffers.failure_policies,
                    n_policies,
                    &mut self.orch_buffers.state,
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    cmd_ring_size,
                    &mut self.orch_buffers.failure_action_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_CLASSIFY_KERNEL, e))?;
        let result = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.failure_action_out)
            .map_err(|e| CudaError::new("dtoh failure_action_out", e))?;
        Ok(result[0])
    }

    /// Snapshot the current `consumed[]` and `active[]` arrays as a rollback marker.
    ///
    /// Sets `OrchestrationState::rollback_available = 1`.  The snapshot can be
    /// restored by calling `apply_rollback`.
    pub fn set_rollback_marker(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_SET_ROLLBACK_KERNEL)
            .ok_or(CudaError::missing_function(
                FAILURE_POLICY_SET_ROLLBACK_KERNEL,
            ))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.consumed,
                    &self.active,
                    &mut self.orch_buffers.rollback_consumed,
                    &mut self.orch_buffers.rollback_active,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_SET_ROLLBACK_KERNEL, e))
    }

    /// Restore `consumed[]` and `active[]` from the saved rollback marker.
    ///
    /// No-op if `OrchestrationState::rollback_available == 0`.
    /// Clears `rollback_available` and `consecutive_blocks` on success.
    pub fn apply_rollback(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_APPLY_ROLLBACK_KERNEL)
            .ok_or(CudaError::missing_function(
                FAILURE_POLICY_APPLY_ROLLBACK_KERNEL,
            ))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &mut self.consumed,
                    &mut self.active,
                    &self.orch_buffers.rollback_consumed,
                    &self.orch_buffers.rollback_active,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_APPLY_ROLLBACK_KERNEL, e))
    }

    // ── Phase-6 wrappers ──────────────────────────────────────────────────────

    /// Zero-initialise the per-node receipt prior table on device.
    pub fn learned_delta_init(&mut self) -> Result<(), CudaError> {
        let n = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_INIT_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_INIT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&mut self.orch_buffers.receipt_priors, n)) }
            .map_err(|e| CudaError::new(LEARNED_DELTA_INIT_KERNEL, e))
    }

    /// Fold one receipt into the per-node prior table and push a `LearnedDelta`
    /// entry to the on-device ring.  `outcome` uses the same codes as
    /// `DeviceReceiptExt::outcome` (0 = success, 1 = failure, 2 = timeout,
    /// 3 = safety violation).
    pub fn learned_delta_fold_receipt(
        &mut self,
        src: i32,
        dst: i32,
        node_id: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let n_nodes = TENSOR_NODE_COUNT as i32;
        let ring_size = LEARNED_DELTA_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_FOLD_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_FOLD_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    src,
                    dst,
                    node_id,
                    outcome,
                    &mut self.orch_buffers.receipt_priors,
                    n_nodes,
                    &mut self.orch_buffers.learned_delta_ring,
                    &mut self.orch_buffers.learned_delta_tail,
                    &self.orch_buffers.learned_delta_head,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(LEARNED_DELTA_FOLD_KERNEL, e))
    }

    /// Drain the on-device learned-delta ring and apply soft tensor updates.
    /// Each pending `LearnedDelta` entry increments or decrements the
    /// corresponding confidence/cost/safety cell in the live tensor.
    pub fn learned_delta_apply(&mut self) -> Result<(), CudaError> {
        let ring_size = LEARNED_DELTA_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_APPLY_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_APPLY_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.orch_buffers.learned_delta_ring,
                    &mut self.orch_buffers.learned_delta_head,
                    &self.orch_buffers.learned_delta_tail,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(LEARNED_DELTA_APPLY_KERNEL, e))
    }

    /// Copy the GPU-resident receipt prior table to host and return it.
    /// Also writes the snapshot into the on-device export buffer for
    /// subsequent CPU persistence without a second kernel launch.
    pub fn export_receipt_priors(&mut self) -> Result<Vec<f32>, CudaError> {
        let n = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, RECEIPT_PRIOR_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(RECEIPT_PRIOR_SNAPSHOT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (
                    &self.orch_buffers.receipt_priors,
                    &mut self.orch_buffers.receipt_prior_snapshot_buf,
                    n,
                ),
            )
        }
        .map_err(|e| CudaError::new(RECEIPT_PRIOR_SNAPSHOT_KERNEL, e))?;
        self.dev
            .dtoh_sync_copy(&self.orch_buffers.receipt_prior_snapshot_buf)
            .map_err(|e| CudaError::new("dtoh receipt_prior_snapshot_buf", e))
    }

    // ── Phase-8 wrappers ──────────────────────────────────────────────────────

    /// Push one event onto the device trace ring.
    /// Single-thread kernel (1 block × 1 thread).
    pub fn push_trace_event(&mut self, event_kind: i32, outcome: i32) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_TRACE_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_TRACE_PUSH_KERNEL))?;
        let ring_size = ORCH_TRACE_RING_SIZE as i32;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &self.orch_buffers.state,
                    event_kind,
                    outcome,
                    &mut self.orch_buffers.trace_ring,
                    &mut self.orch_buffers.trace_tail,
                    &self.orch_buffers.trace_head,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_TRACE_PUSH_KERNEL, e))
    }

    /// Drain pending trace events to host.  Returns a `Vec` of drained events.
    /// Single-thread kernel writes entries to `trace_drain_buf`; host copies.
    pub fn drain_trace_events(&mut self) -> Result<Vec<OrchestrationEvent>, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_TRACE_DRAIN_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_TRACE_DRAIN_KERNEL))?;
        let ring_size = ORCH_TRACE_RING_SIZE as i32;
        let max_count = ORCH_TRACE_RING_SIZE as i32;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.orch_buffers.trace_ring,
                    &mut self.orch_buffers.trace_head,
                    &self.orch_buffers.trace_tail,
                    ring_size,
                    &mut self.orch_buffers.trace_drain_buf,
                    &mut self.orch_buffers.trace_drain_count,
                    max_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_TRACE_DRAIN_KERNEL, e))?;
        let count_vec = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.trace_drain_count)
            .map_err(|e| CudaError::new("dtoh trace_drain_count", e))?;
        let n = count_vec[0].max(0) as usize;
        let all = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.trace_drain_buf)
            .map_err(|e| CudaError::new("dtoh trace_drain_buf", e))?;
        Ok(all.into_iter().take(n).collect())
    }

    /// Run the no-duplicate-receipts invariant check.
    /// Returns `Ok(true)` when the invariant holds (no violation).
    pub fn check_no_duplicate_receipts(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.orch_buffers.receipt_ext_ring,
                    size,
                    &mut self.orch_buffers.orch_violation_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Run the frontier-valid invariant check.
    /// Returns `Ok(true)` when all `active[i]` ∈ {0, 1}.
    pub fn check_frontier_valid(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_FRONTIER_VALID_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_FRONTIER_VALID_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let n = TENSOR_NODE_COUNT as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (&self.active, n, &mut self.orch_buffers.orch_violation_out),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_FRONTIER_VALID_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Run the no-command-without-receipt invariant check.
    /// Returns `Ok(true)` when the invariant holds.
    pub fn check_no_command_without_receipt(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let cmd_size = DEVICE_COMMAND_RING_SIZE as i32;
        let ext_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.orch_buffers.command_ring,
                    cmd_size,
                    &self.orch_buffers.receipt_ext_ring,
                    ext_size,
                    &mut self.orch_buffers.orch_violation_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Snapshot current orchestration state (+ consumed + active) into the
    /// replay buffers.  Block-parallel copy kernel (N threads).
    pub fn replay_snapshot(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_REPLAY_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_REPLAY_SNAPSHOT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (TENSOR_NODE_COUNT as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        unsafe {
            f.launch(
                cfg,
                (
                    &self.orch_buffers.state,
                    &mut self.orch_buffers.replay_state,
                    &self.consumed,
                    &mut self.orch_buffers.replay_consumed,
                    &self.active,
                    &mut self.orch_buffers.replay_active,
                    &self.orch_buffers.star_counters,
                    &mut self.orch_buffers.replay_star_counters,
                    star_counter_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_REPLAY_SNAPSHOT_KERNEL, e))
    }

    /// Restore orchestration state from the replay buffers back to live state.
    /// Block-parallel copy kernel (N threads).
    pub fn replay_restore(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_REPLAY_RESTORE_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_REPLAY_RESTORE_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (TENSOR_NODE_COUNT as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        unsafe {
            f.launch(
                cfg,
                (
                    &mut self.orch_buffers.state,
                    &self.orch_buffers.replay_state,
                    &mut self.consumed,
                    &self.orch_buffers.replay_consumed,
                    &mut self.active,
                    &self.orch_buffers.replay_active,
                    &mut self.orch_buffers.star_counters,
                    &self.orch_buffers.replay_star_counters,
                    star_counter_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_REPLAY_RESTORE_KERNEL, e))
    }

    /// Write a region dispatch request to the device mailbox and launch
    /// `tensor_quantale_gpu_dispatch`, which records a `DeviceReceipt` in the
    /// ring buffer without returning to the CPU.
    ///
    /// The JIT kernel for the region must have been launched **before** calling
    /// this method so that `gpu_dispatch` can immediately record a success
    /// receipt.  Call `drain_device_receipts` afterwards to fold the receipt
    /// into the quantale tensor.
    pub fn gpu_dispatch_region(
        &mut self,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        use crate::tensor::GpuDispatchMailboxHost;
        let mailbox = GpuDispatchMailboxHost {
            pending_region_id: region_id,
            src_node,
            dst_node,
            outcome,
            dispatched: 0,
        };
        let mailbox_buf = self
            .dev
            .htod_copy(vec![mailbox])
            .map_err(|error| CudaError::new("htod_copy gpu dispatch mailbox", error))?;
        let region_count = GPU_HOT_REGION_COUNT;
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, GPU_DISPATCH_KERNEL)
            .ok_or(CudaError::missing_function(GPU_DISPATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mailbox_buf,
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_count,
                    0_u64,
                    0_i32,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_gpu_dispatch", error))
    }

    /// Dispatch a hot GPU region with real device-slot backing.
    ///
    /// `DeviceSlotRegistry` supplies the region's ordered `float**` slot table,
    /// so the CUDA dispatch switch calls the region function and writes output
    /// slots before appending the device receipt.
    pub fn gpu_dispatch_region_with_slots(
        &mut self,
        registry: &DeviceSlotRegistry,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let slot_names = gpu_region_slots(region_id).ok_or_else(|| {
            CudaError::invalid_input(format!("unknown GPU hot region id {region_id}"))
        })?;
        let mailbox = GpuDispatchMailboxHost {
            pending_region_id: region_id,
            src_node,
            dst_node,
            outcome,
            dispatched: 0,
        };
        let mailbox_buf = self
            .dev
            .htod_copy(vec![mailbox])
            .map_err(|error| CudaError::new("htod_copy gpu dispatch mailbox", error))?;
        let (slot_ptrs, element_count) = registry
            .device_slot_ptr_table(&self.dev, slot_names)
            .map_err(CudaError::invalid_input)?;
        let region_count = GPU_HOT_REGION_COUNT;
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, GPU_DISPATCH_KERNEL)
            .ok_or(CudaError::missing_function(GPU_DISPATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mailbox_buf,
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_count,
                    &slot_ptrs,
                    element_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_gpu_dispatch", error))?;
        self.dev
            .synchronize()
            .map_err(|error| CudaError::new("synchronize gpu dispatch", error))
    }

    pub fn decay(&mut self, factor: f32) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DECAY_KERNEL)
            .ok_or(CudaError::missing_function(DECAY_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.tensor, factor)) }
            .map_err(|error| CudaError::new("tensor_quantale_decay", error))
    }

    pub fn seed_exploration(&mut self, engine: &mut ExplorationEngine) -> Result<(), CudaError> {
        let strategy_nodes = engine.strategy_nodes()?;
        let strategy_biases = engine.strategy_biases();
        let receipt_priors = engine.receipt_prior_vector();
        let strategy_count = i32::try_from(strategy_nodes.len())
            .map_err(|_| CudaError::invalid_input("too many exploration strategies"))?;
        let strategy_node_buffer = self
            .dev
            .htod_copy(strategy_nodes)
            .map_err(|error| CudaError::new("htod_copy exploration strategy nodes", error))?;
        let strategy_bias_buffer = self
            .dev
            .htod_copy(strategy_biases)
            .map_err(|error| CudaError::new("htod_copy exploration strategy bias", error))?;
        let receipt_prior_buffer = self
            .dev
            .htod_copy(receipt_priors)
            .map_err(|error| CudaError::new("htod_copy exploration receipt priors", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_SEED_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_SEED_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &strategy_node_buffer,
                    &strategy_bias_buffer,
                    &receipt_prior_buffer,
                    strategy_count,
                    EXPLORATION_MAX_TOKENS as i32,
                    &mut self.exploration_tokens,
                    &mut self.exploration_scores,
                    &mut self.exploration_parents,
                    &mut self.exploration_token_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_seed_exploration", error))?;
        self.sync_exploration_engine(engine)
    }

    pub fn expand_exploration(
        &mut self,
        engine: &mut ExplorationEngine,
    ) -> Result<Vec<ExplorationCandidate>, CudaError> {
        self.seed_exploration(engine)?;
        let max_depth = i32::try_from(engine.config().max_depth)
            .map_err(|_| CudaError::invalid_input("exploration max_depth too large"))?;
        let beam_width = i32::try_from(engine.config().beam_width)
            .map_err(|_| CudaError::invalid_input("exploration beam_width too large"))?;
        for source_depth in 0..max_depth {
            let expand = self
                .dev
                .get_func(MODULE_NAME, EXPLORATION_EXPAND_KERNEL)
                .ok_or(CudaError::missing_function(EXPLORATION_EXPAND_KERNEL))?;
            unsafe {
                expand.launch(
                    kernel_config(),
                    (
                        &self.tensor,
                        &mut self.exploration_token_count,
                        source_depth,
                        max_depth,
                        EXPLORATION_MAX_TOKENS as i32,
                        &mut self.exploration_tokens,
                        &mut self.exploration_parents,
                    ),
                )
            }
            .map_err(|error| CudaError::new("tensor_quantale_expand_tokens", error))?;
        }
        let score = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_SCORE_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_SCORE_KERNEL))?;
        unsafe {
            score.launch(
                kernel_config(),
                (
                    &self.exploration_tokens,
                    &self.exploration_token_count,
                    engine.config().novelty_weight,
                    engine.config().receipt_weight,
                    engine.config().entropy_penalty,
                    &mut self.exploration_scores,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_score_tokens", error))?;
        let terminal_visits = self
            .dev
            .htod_copy(engine.terminal_visit_vector())
            .map_err(|error| CudaError::new("htod_copy exploration terminal visits", error))?;
        let first_hop_visits = self
            .dev
            .htod_copy(engine.first_hop_visit_vector())
            .map_err(|error| CudaError::new("htod_copy exploration first-hop visits", error))?;
        let max_terminal_visits = i32::try_from(engine.config().max_terminal_visits)
            .map_err(|_| CudaError::invalid_input("exploration max_terminal_visits too large"))?;
        let max_first_hop_visits = i32::try_from(engine.config().max_first_hop_visits)
            .map_err(|_| CudaError::invalid_input("exploration max_first_hop_visits too large"))?;
        let topk = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_TOPK_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_TOPK_KERNEL))?;
        unsafe {
            topk.launch(
                kernel_config(),
                (
                    &self.exploration_tokens,
                    &self.exploration_scores,
                    &self.exploration_token_count,
                    beam_width,
                    engine.config().repeat_penalty,
                    max_terminal_visits,
                    max_first_hop_visits,
                    &terminal_visits,
                    &first_hop_visits,
                    &mut self.exploration_selected,
                    &mut self.exploration_selected_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_select_topk_tokens", error))?;
        self.sync_exploration_engine(engine)?;
        Ok(engine.selected().to_vec())
    }

    pub fn commit_exploration_candidate(
        &mut self,
        candidate: &ExplorationCandidate,
    ) -> Result<DecisionReport, CudaError> {
        let candidate_buffer = self
            .dev
            .htod_copy(vec![*candidate])
            .map_err(|error| CudaError::new("htod_copy exploration commit candidate", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_COMMIT_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_COMMIT_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &candidate_buffer,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_commit_exploration", error))?;
        self.decision_report()
    }

    /// Score dynamically detected JIT chains on the GPU and embed results into the tensor.
    pub fn embed_jit_chain_scores(
        &mut self,
        chains: &[crate::jit_kernel_fusion::JitChainMetadata],
        src_node: i32,
    ) -> Result<(), CudaError> {
        if chains.is_empty() {
            return Ok(());
        }
        let count = i32::try_from(chains.len())
            .map_err(|_| CudaError::invalid_input("too many JIT chains"))?;
        let chain_buf = self
            .dev
            .htod_copy(chains.to_vec())
            .map_err(|error| CudaError::new("htod_copy jit chain metadata", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, JIT_CHAIN_SCORE_KERNEL)
            .ok_or(CudaError::missing_function(JIT_CHAIN_SCORE_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (&mut self.tensor, &chain_buf, count, src_node),
            )
        }
        .map_err(|error| CudaError::new("jit_chain_score_embed", error))?;
        Ok(())
    }

    fn sync_exploration_engine(&self, engine: &mut ExplorationEngine) -> Result<(), CudaError> {
        let token_count = self
            .dev
            .dtoh_sync_copy(&self.exploration_token_count)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration token_count", error))?
            .into_iter()
            .next()
            .unwrap_or(0)
            .clamp(0, EXPLORATION_MAX_TOKENS as i32) as usize;
        let selected_count = self
            .dev
            .dtoh_sync_copy(&self.exploration_selected_count)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration selected_count", error))?
            .into_iter()
            .next()
            .unwrap_or(0)
            .clamp(0, EXPLORATION_MAX_SELECTED as i32) as usize;
        let mut tokens = self
            .dev
            .dtoh_sync_copy(&self.exploration_tokens)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration tokens", error))?;
        let mut selected = self
            .dev
            .dtoh_sync_copy(&self.exploration_selected)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration selected", error))?;
        tokens.truncate(token_count);
        selected.truncate(selected_count);
        engine.load_gpu_state(tokens, selected);
        Ok(())
    }

    /// The `CudaDevice` that owns all PTX modules loaded by this world.
    /// Pass this reference when launching kernels that belong to the same module.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub fn tensor(&self) -> Result<Vec<f32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.tensor)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor", error))
    }

    pub fn witness(&self) -> Result<Vec<i32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.witness)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor witness", error))
    }

    pub fn reconstruct_tensor_path(
        &self,
        layer: i32,
        src: Node,
        dst: Node,
    ) -> Result<Vec<Node>, CudaError> {
        if !(0..TENSOR_LAYER_COUNT as i32).contains(&layer) {
            return Err(CudaError::invalid_input(format!(
                "invalid tensor layer {layer}"
            )));
        }
        let witness = self.witness()?;
        let offset = layer as usize * MATRIX_LEN;
        let registry = bundled_registry()?;
        reconstruct_path_from_witness_matrix(
            &witness[offset..offset + MATRIX_LEN],
            src,
            dst,
            &registry,
        )
    }

    pub fn reconstruct_projected_tensor_path(&self, layer: i32) -> Result<Vec<Node>, CudaError> {
        let decision = self.decision_report()?;
        let registry = bundled_registry()?;
        let src = Node::decode(decision.selected_src, &registry).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct tensor path with invalid selected_src {}",
                decision.selected_src
            ))
        })?;
        let dst = Node::decode(decision.selected_dst, &registry).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct tensor path with invalid selected_dst {}",
                decision.selected_dst
            ))
        })?;
        self.reconstruct_tensor_path(layer, src, dst)
    }

    pub fn decision_report(&self) -> Result<DecisionReport, CudaError> {
        let reports = self
            .dev
            .dtoh_sync_copy(&self.decision)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor decision", error))?;
        reports.into_iter().next().ok_or(CudaError {
            operation: "dtoh_sync_copy tensor decision",
            message: "empty tensor decision buffer".to_string(),
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.dev
            .synchronize()
            .map_err(|error| CudaError::new("CudaDevice::synchronize tensor", error))
    }
}

pub fn tensor_idx(layer: i32, src: i32, dst: i32) -> usize {
    layer as usize * MATRIX_LEN + src as usize * TENSOR_NODE_COUNT + dst as usize
}

fn kernel_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (DEFAULT_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn tensor_start_node() -> i32 {
    GraphTopology::bundled_registry()
        .ok()
        .and_then(|registry| {
            RuntimeContext::default_asset()
                .ok()
                .and_then(|context| registry.id_of(&context.start_node))
        })
        .unwrap_or(0) as i32
}

fn bundled_registry() -> Result<NodeRegistry, CudaError> {
    Ok(GraphTopology::bundled_registry()?)
}

#[cfg(test)]
mod generated_hf_tests {
    use super::*;

    #[test]
    fn generated_fusion_hf_coverage_promotes_region_id_eight() {
        let coverage = FusionHfCoverage::from_json_str(
            r#"{
                "schema":"fusion_hf_coverage.v1",
                "regions":[
                    {
                        "region":"Fixture::Add__Fixture::Scale",
                        "entry":"Fixture::Add",
                        "nodes":["Fixture::Add", "Fixture::Scale"],
                        "hf_region_id":8,
                        "covered":true,
                        "reason":"generated_hf_handler",
                        "symbol":"region_fusion_stub_fixture_add_fixture_scale",
                        "slots":["fixture.a", "fixture.b", "fixture.scale", "fixture.out"]
                    }
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(coverage.region_id("Fixture::Add__Fixture::Scale"), Some(8));
        assert!(coverage.has_handler_for_region_id(8));
        assert_eq!(coverage.region_count(), 9);
        assert_eq!(
            coverage.slots_for_region_id(8),
            Some(
                &[
                    "fixture.a".to_string(),
                    "fixture.b".to_string(),
                    "fixture.scale".to_string(),
                    "fixture.out".to_string(),
                ][..]
            )
        );
    }
}
