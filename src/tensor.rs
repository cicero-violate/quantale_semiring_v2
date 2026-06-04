//! Three-layer tensor quantale engine.
//!
//! Layers:
//! - confidence/correctness: max-times
//! - compute/time cost: min-plus
//! - security/safety: max-min

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use serde::{Deserialize, Serialize};

use crate::config::{DEFAULT_BLOCK_SIZE, RuntimeContext};
use crate::error::CudaError;
use crate::exploration::{ExplorationCandidate, ExplorationEngine, ExplorationToken};
use crate::graph::{DecisionReport, Node, reconstruct_path_from_witness_matrix};
use crate::topology::{GraphTopology, NodeRegistry};
use crate::types::ProcessReceipt;

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
const DRAIN_KERNEL: &str = "tensor_quantale_drain_queue";
const DECAY_KERNEL: &str = "tensor_quantale_decay";
const FRONTIER_STEP_KERNEL: &str = "tensor_quantale_frontier_step";
const TICK_KERNEL: &str = "tensor_quantale_tick";
const EXPLORATION_SEED_KERNEL: &str = "tensor_quantale_seed_exploration";
const EXPLORATION_EXPAND_KERNEL: &str = "tensor_quantale_expand_tokens";
const EXPLORATION_SCORE_KERNEL: &str = "tensor_quantale_score_tokens";
const EXPLORATION_TOPK_KERNEL: &str = "tensor_quantale_select_topk_tokens";
const EXPLORATION_COMMIT_KERNEL: &str = "tensor_quantale_commit_exploration";
const JIT_CHAIN_SCORE_KERNEL: &str = "jit_chain_score_embed";
const DRAIN_DEVICE_RECEIPTS_KERNEL: &str = "tensor_quantale_drain_device_receipts";
const GPU_DISPATCH_KERNEL: &str = "tensor_quantale_gpu_dispatch";
const RING_PUSH_KERNEL: &str = "device_ring_push";
const RING_POP_KERNEL: &str = "device_ring_pop";
const PARALLEL_REDUCE_KERNEL: &str = "quantale_parallel_reduce";
const TOPK_BITONIC_KERNEL: &str = "quantale_topk_bitonic";

pub const DEVICE_RECEIPT_RING_SIZE: usize = 256;

pub const EXPLORATION_MAX_TOKENS: usize = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT;
pub const EXPLORATION_MAX_SELECTED: usize = TENSOR_NODE_COUNT;
const KERNEL_SOURCE_TEMPLATE: &str = include_str!("../cuda/quantale_world.cu");

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
    fn code(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::Failure => 1,
            Self::Timeout => 2,
            Self::SafetyViolation => 3,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecutionReceipt {
    pub src: i32,
    pub dst: i32,
    pub outcome: i32,
}

unsafe impl DeviceRepr for ExecutionReceipt {}

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

impl From<&ProcessReceipt> for ExecutionOutcome {
    fn from(receipt: &ProcessReceipt) -> Self {
        match receipt.exit_code {
            0 => Self::Success,
            124 => Self::Timeout,
            _ => Self::Failure,
        }
    }
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
    /// Host-side queue of execution receipts pending a drain_lattice_queue call.
    event_queue: Vec<ExecutionReceipt>,
    /// Device-resident receipt ring for the GPU hot-dispatch path.
    device_receipt_buffer: DeviceReceiptBuffer,
}

impl TensorQuantaleWorld {
    pub fn empty() -> Result<Self, CudaError> {
        let dev = CudaDevice::new(0).map_err(|error| CudaError::new("CudaDevice::new", error))?;
        let kernel_source = format!("{CUDA_TENSOR_NODE_COUNT_DEFINE}{KERNEL_SOURCE_TEMPLATE}");
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
                DRAIN_KERNEL,
                DECAY_KERNEL,
                FRONTIER_STEP_KERNEL,
                TICK_KERNEL,
                EXPLORATION_SEED_KERNEL,
                EXPLORATION_EXPAND_KERNEL,
                EXPLORATION_SCORE_KERNEL,
                EXPLORATION_TOPK_KERNEL,
                EXPLORATION_COMMIT_KERNEL,
                JIT_CHAIN_SCORE_KERNEL,
                DRAIN_DEVICE_RECEIPTS_KERNEL,
                GPU_DISPATCH_KERNEL,
                RING_PUSH_KERNEL,
                RING_POP_KERNEL,
                PARALLEL_REDUCE_KERNEL,
                TOPK_BITONIC_KERNEL,
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
            event_queue: Vec::new(),
            device_receipt_buffer: DeviceReceiptBuffer {
                ring: device_receipt_ring,
                head: device_receipt_head,
                tail: device_receipt_tail,
            },
        };
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

    /// Project and advance the tensor frontier on CUDA.
    pub fn frontier_step(&mut self, bias: ProjectionBias) -> Result<DecisionReport, CudaError> {
        let bias_buffer = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|error| CudaError::new("htod_copy tensor frontier bias", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, FRONTIER_STEP_KERNEL)
            .ok_or(CudaError::missing_function(FRONTIER_STEP_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &self.witness,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &bias_buffer,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_frontier_step", error))?;
        let report = self.decision_report()?;
        // Invariant 16: frontier one-hot validity — first_hop must be a valid
        // node index.  In test builds this panics at the source instead of
        // propagating Unknown(-1) through subsequent calls.
        debug_assert!(
            report.blocked != 0 || (0..TENSOR_NODE_COUNT as i32).contains(&report.first_hop),
            "frontier_step returned invalid node id: {}",
            report.first_hop
        );
        Ok(report)
    }

    /// Fused tensor closure plus frontier projection/update.
    pub fn tick(&mut self, bias: ProjectionBias) -> Result<DecisionReport, CudaError> {
        let bias_buffer = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|error| CudaError::new("htod_copy tensor tick bias", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, TICK_KERNEL)
            .ok_or(CudaError::missing_function(TICK_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.scratch,
                    &mut self.witness,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &bias_buffer,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_tick", error))?;
        self.decision_report()
    }

    /// Push an execution receipt onto the host-side event queue.
    /// Call drain_lattice_queue to flush the batch to the GPU.
    pub fn queue_lattice_update(&mut self, src: i32, dst: i32, outcome: ExecutionOutcome) {
        self.event_queue.push(ExecutionReceipt {
            src,
            dst,
            outcome: outcome.code(),
        });
    }

    /// Drain all pending execution receipts to the GPU in a single parallel kernel launch.
    /// No-ops if the queue is empty.
    pub fn drain_lattice_queue(&mut self) -> Result<(), CudaError> {
        if self.event_queue.is_empty() {
            return Ok(());
        }
        let receipts = std::mem::take(&mut self.event_queue);
        let count = i32::try_from(receipts.len())
            .map_err(|_| CudaError::invalid_input("too many queued lattice updates"))?;
        let receipt_buf = self
            .dev
            .htod_copy(receipts)
            .map_err(|error| CudaError::new("htod_copy lattice receipts", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DRAIN_KERNEL)
            .ok_or(CudaError::missing_function(DRAIN_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.tensor, &receipt_buf, count)) }
            .map_err(|error| CudaError::new("tensor_quantale_drain_queue", error))
    }

    /// Drain all `DeviceReceipt`s in the device ring buffer on-device.
    ///
    /// Unlike `drain_lattice_queue`, this path never touches the CPU for
    /// tensor updates — the GPU reads the ring and applies confidence/cost/
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
        let region_count = 64_i32; // upper bound; actual regions registered at startup
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
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_gpu_dispatch", error))
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
