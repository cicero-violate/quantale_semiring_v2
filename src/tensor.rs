//! Three-layer tensor quantale engine.
//!
//! Layers:
//! - confidence/correctness: max-times
//! - compute/time cost: min-plus
//! - security/safety: max-min

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use serde::Serialize;

use crate::error::CudaError;
use crate::node::{MATRIX_LEN, NODE_COUNT, Node, START_NODE, THREAD_COUNT};
use crate::path::reconstruct_path_from_witness_matrix;
use crate::projection::DecisionReport;
use crate::rule_delta::ProcessReceipt;

pub const TENSOR_LAYER_COUNT: usize = 3;
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
const UPDATE_KERNEL: &str = "tensor_quantale_update_edge";
const DECAY_KERNEL: &str = "tensor_quantale_decay";
const FRONTIER_STEP_KERNEL: &str = "tensor_quantale_frontier_step";
const TICK_KERNEL: &str = "tensor_quantale_tick";
const KERNEL_SOURCE: &str = include_str!("../cuda/quantale_world.cu");

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
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
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
}

impl TensorQuantaleWorld {
    pub fn empty() -> Result<Self, CudaError> {
        let dev = CudaDevice::new(0).map_err(|error| CudaError::new("CudaDevice::new", error))?;
        let ptx =
            compile_ptx(KERNEL_SOURCE).map_err(|error| CudaError::new("compile_ptx", error))?;
        dev.load_ptx(
            ptx,
            MODULE_NAME,
            &[
                RESET_KERNEL,
                EMBED_KERNEL,
                CLOSURE_KERNEL,
                PROJECT_KERNEL,
                UPDATE_KERNEL,
                DECAY_KERNEL,
                FRONTIER_STEP_KERNEL,
                TICK_KERNEL,
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
            .htod_copy(vec![0_i32; NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy tensor active", error))?;
        let next_active = dev
            .htod_copy(vec![0_i32; NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy tensor next_active", error))?;
        let decision = dev
            .htod_copy(vec![DecisionReport::default()])
            .map_err(|error| CudaError::new("htod_copy tensor decision", error))?;

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
        };
        world.reset()?;
        Ok(world)
    }

    pub fn from_tensor_edges(edges: &[TensorEdge]) -> Result<Self, CudaError> {
        let mut world = Self::empty()?;
        world.embed_tensor_edges(edges)?;
        Ok(world)
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
        self.decision_report()
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

    pub fn update_lattice_edge(
        &mut self,
        src: i32,
        dst: i32,
        outcome: ExecutionOutcome,
    ) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, UPDATE_KERNEL)
            .ok_or(CudaError::missing_function(UPDATE_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (&mut self.tensor, src, dst, outcome.code()),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_update_edge", error))
    }

    pub fn decay(&mut self, factor: f32) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DECAY_KERNEL)
            .ok_or(CudaError::missing_function(DECAY_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.tensor, factor)) }
            .map_err(|error| CudaError::new("tensor_quantale_decay", error))
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
        reconstruct_path_from_witness_matrix(&witness[offset..offset + MATRIX_LEN], src, dst)
    }

    pub fn reconstruct_projected_tensor_path(&self, layer: i32) -> Result<Vec<Node>, CudaError> {
        let decision = self.decision_report()?;
        let src = Node::decode(decision.selected_src).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct tensor path with invalid selected_src {}",
                decision.selected_src
            ))
        })?;
        let dst = Node::decode(decision.selected_dst).ok_or_else(|| {
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
    layer as usize * MATRIX_LEN + src as usize * NODE_COUNT + dst as usize
}

fn kernel_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (THREAD_COUNT as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn tensor_start_node() -> i32 {
    START_NODE.encode()
}
