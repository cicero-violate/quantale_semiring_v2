//! Three-layer tensor quantale engine.
//!
//! Layers:
//! - confidence/correctness: max-times
//! - compute/time cost: min-plus
//! - security/safety: max-min

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::config::{DEFAULT_BLOCK_SIZE, RuntimeContext};
use crate::device_slots::DeviceSlotRegistry;
use crate::error::CudaError;
use crate::exploration::{ExplorationCandidate, ExplorationEngine, ExplorationToken};
use crate::graph::{DecisionReport, Node, reconstruct_path_from_witness_matrix};
use crate::topology::{GraphTopology, NodeRegistry};

mod abi;
mod buffers;
mod constants;
mod coverage;
mod exploration_ops;
mod hot_dispatch;
mod kernel_names;
mod kernel_source;
mod orchestration;
mod par_group;
mod par_group_ops;
mod readback;

pub use abi::{
    ControlEdge, DeviceCommand, DeviceReceipt, DeviceReceiptExt, EffectTable, ExecutionOutcome,
    FailureClassifyRequest, FailurePolicy, GpuDispatchMailboxHost, LearnedDelta, OrchStepStatus,
    OrchestrationEvent, OrchestrationState, ProjectionBias, TensorEdge,
};
use buffers::TensorWorldBundleHost;
pub use buffers::{DeviceReceiptBuffer, OrchestrationBuffers};
pub use constants::*;
pub use coverage::{
    AbstractDeviceCoverage, AbstractDeviceCoverageEntry, DEFAULT_PAR_SLOT_ELEMENTS,
    FusionHfCoverage, FusionHfCoverageEntry, fusion_hf_region_id, gpu_region_slots,
    static_hf_symbol,
};
use kernel_names::*;
use kernel_source::{assemble_kernel_source, assemble_kernel_source_with_generated};
pub use par_group::{ParDispatchDescriptor, ParGroupGpuData};
use par_group::{ParGroupHfParamsHost, ParGroupStepOutputRaw};

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
