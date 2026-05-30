//! CUDA-resident quantale matrix and kernels.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::algebra::Q_BOTTOM;
use crate::edge::TransitionEdge;
use crate::error::CudaError;
use crate::node::{MATRIX_LEN, NODE_COUNT, Node, THREAD_COUNT};
use crate::path::reconstruct_path_from_next_hop;
use crate::policy::{ExecutionGatePolicy, build_policy_edges};
use crate::projection::{DecisionProjection, DecisionReport, QuantaleCudaReport};
use crate::receipt::{ExecutionReceipt, build_receipt_edges};
use crate::search::{
    DomainCandidate, ScoredCandidate, build_search_delta_edges, build_search_edges,
};
use crate::transitions::default_transition_edges;

const MODULE_NAME: &str = "quantale_semiring_v2";
const RESET_KERNEL: &str = "quantale_reset";
const LOAD_EDGES_KERNEL: &str = "quantale_load_edges";
const JOIN_ASSIGN_KERNEL: &str = "quantale_join_assign";
const MUL_ASSIGN_KERNEL: &str = "quantale_mul_assign";
const CLOSURE_ASSIGN_KERNEL: &str = "quantale_closure_assign";
const STEP_KERNEL: &str = "quantale_step";
/// CUDA kernel for the operational projection π(A*) from closed quantale paths.
const DECISION_PROJECTION_KERNEL: &str = "quantale_decide_path";
const FRONTIER_STEP_KERNEL: &str = "quantale_frontier_step";
const KERNEL_SOURCE: &str = include_str!("../cuda/quantale_world.cu");

pub struct GpuQuantaleMatrix {
    dev: Arc<CudaDevice>,
    /// CUDA-resident A / A*: quantale transition matrix and closed path values.
    transition: CudaSlice<f32>,
    /// CUDA-resident scratch matrix for A* closure/composition.
    scratch: CudaSlice<f32>,
    /// CUDA-resident W witness matrix: first hop for the selected path value.
    next_hop: CudaSlice<i32>,
    /// CUDA-resident scratch witness matrix used by quantale multiplication.
    scratch_next_hop: CudaSlice<i32>,
    consumed: CudaSlice<i32>,
    previous: CudaSlice<f32>,
    active: CudaSlice<i32>,
    next_active: CudaSlice<i32>,
    event_counts: CudaSlice<i32>,
    report: CudaSlice<QuantaleCudaReport>,
    decision_report: CudaSlice<DecisionReport>,
}

pub type CudaWorld = GpuQuantaleMatrix;

impl GpuQuantaleMatrix {
    pub fn new() -> Result<Self, CudaError> {
        Self::from_edges(&default_transition_edges())
    }

    pub fn empty() -> Result<Self, CudaError> {
        let dev = CudaDevice::new(0).map_err(|error| CudaError::new("CudaDevice::new", error))?;
        let ptx =
            compile_ptx(KERNEL_SOURCE).map_err(|error| CudaError::new("compile_ptx", error))?;
        dev.load_ptx(
            ptx,
            MODULE_NAME,
            &[
                RESET_KERNEL,
                LOAD_EDGES_KERNEL,
                JOIN_ASSIGN_KERNEL,
                MUL_ASSIGN_KERNEL,
                CLOSURE_ASSIGN_KERNEL,
                STEP_KERNEL,
                DECISION_PROJECTION_KERNEL,
                FRONTIER_STEP_KERNEL,
            ],
        )
        .map_err(|error| CudaError::new("load_ptx", error))?;

        let transition = dev
            .htod_copy(vec![Q_BOTTOM; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy transition", error))?;
        let scratch = dev
            .htod_copy(vec![Q_BOTTOM; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy scratch", error))?;
        let next_hop = dev
            .htod_copy(vec![-1_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy next_hop", error))?;
        let scratch_next_hop = dev
            .htod_copy(vec![-1_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy scratch_next_hop", error))?;
        let consumed = dev
            .htod_copy(vec![0_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy consumed", error))?;
        let previous = dev
            .htod_copy(vec![Q_BOTTOM; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy previous", error))?;
        let active = dev
            .htod_copy(vec![0_i32; NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy active", error))?;
        let next_active = dev
            .htod_copy(vec![0_i32; NODE_COUNT])
            .map_err(|error| CudaError::new("htod_copy next_active", error))?;
        let event_counts = dev
            .htod_copy(vec![0_i32; THREAD_COUNT])
            .map_err(|error| CudaError::new("htod_copy event_counts", error))?;
        let report = dev
            .htod_copy(vec![QuantaleCudaReport::default()])
            .map_err(|error| CudaError::new("htod_copy report", error))?;
        let decision_report = dev
            .htod_copy(vec![DecisionReport::default()])
            .map_err(|error| CudaError::new("htod_copy decision_report", error))?;

        let mut matrix = Self {
            dev,
            transition,
            scratch,
            next_hop,
            scratch_next_hop,
            consumed,
            previous,
            active,
            next_active,
            event_counts,
            report,
            decision_report,
        };
        matrix.reset()?;
        Ok(matrix)
    }

    pub fn from_edges(edges: &[TransitionEdge]) -> Result<Self, CudaError> {
        let mut matrix = Self::empty()?;
        matrix.load_edges(edges)?;
        Ok(matrix)
    }

    pub fn reset(&mut self) -> Result<(), CudaError> {
        let reset = self
            .dev
            .get_func(MODULE_NAME, RESET_KERNEL)
            .ok_or(CudaError::missing_function(RESET_KERNEL))?;
        unsafe {
            reset.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.scratch,
                    &mut self.previous,
                    &mut self.next_hop,
                    &mut self.scratch_next_hop,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &mut self.event_counts,
                    &mut self.report,
                    &mut self.decision_report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_reset", error))
    }

    pub fn load_edges(&mut self, edges: &[TransitionEdge]) -> Result<(), CudaError> {
        let edge_count = i32::try_from(edges.len())
            .map_err(|_| CudaError::invalid_input("too many transition edges"))?;
        let edge_buffer = self
            .dev
            .htod_copy(edges.to_vec())
            .map_err(|error| CudaError::new("htod_copy edges", error))?;
        let load_edges = self
            .dev
            .get_func(MODULE_NAME, LOAD_EDGES_KERNEL)
            .ok_or(CudaError::missing_function(LOAD_EDGES_KERNEL))?;
        unsafe {
            load_edges.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.next_hop,
                    &edge_buffer,
                    edge_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_load_edges", error))
    }

    /// Join matrix-native policy edges into the current transition matrix.
    ///
    /// This joins policy into the matrix. Projection reads matrix structure only.
    pub fn join_policy_edges(&mut self, policy: ExecutionGatePolicy) -> Result<(), CudaError> {
        let policy_edges = build_policy_edges(policy);
        self.load_edges(&policy_edges)
    }

    /// Join runtime receipt evidence into the current transition matrix.
    pub fn join_receipt_edges(&mut self, receipt: ExecutionReceipt) -> Result<(), CudaError> {
        let receipt_edges = build_receipt_edges(receipt);
        self.load_edges(&receipt_edges)
    }

    /// Join selected runtime search evidence into the current transition matrix.
    pub fn join_search_edges(&mut self, top_k: &[ScoredCandidate]) -> Result<(), CudaError> {
        let search_edges = build_search_edges(top_k);
        self.load_edges(&search_edges)
    }

    /// Score external candidates, select top-k, and join search evidence.
    pub fn join_search_candidates(
        &mut self,
        candidates: impl IntoIterator<Item = DomainCandidate>,
        k: usize,
    ) -> Result<Vec<ScoredCandidate>, CudaError> {
        let (top_k, search_edges) = build_search_delta_edges(candidates, k);
        self.load_edges(&search_edges)?;
        Ok(top_k)
    }

    /// Represent policy as matrix edges. Projection reads matrix structure only.
    pub fn apply_execution_policy(&mut self, policy: ExecutionGatePolicy) -> Result<(), CudaError> {
        self.join_policy_edges(policy)
    }

    pub fn join_assign(&mut self, rhs: &Self) -> Result<(), CudaError> {
        if !Arc::ptr_eq(&self.dev, &rhs.dev) {
            return Err(CudaError::invalid_input(
                "join_assign requires both matrices to belong to the same CUDA device/context",
            ));
        }
        let join = self
            .dev
            .get_func(MODULE_NAME, JOIN_ASSIGN_KERNEL)
            .ok_or(CudaError::missing_function(JOIN_ASSIGN_KERNEL))?;
        unsafe {
            join.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.next_hop,
                    &rhs.transition,
                    &rhs.next_hop,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_join_assign", error))
    }

    pub fn mul_assign(&mut self, rhs: &Self) -> Result<(), CudaError> {
        if !Arc::ptr_eq(&self.dev, &rhs.dev) {
            return Err(CudaError::invalid_input(
                "mul_assign requires both matrices to belong to the same CUDA device/context",
            ));
        }
        let mul = self
            .dev
            .get_func(MODULE_NAME, MUL_ASSIGN_KERNEL)
            .ok_or(CudaError::missing_function(MUL_ASSIGN_KERNEL))?;
        unsafe {
            mul.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.next_hop,
                    &rhs.transition,
                    &rhs.next_hop,
                    &mut self.scratch,
                    &mut self.scratch_next_hop,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_mul_assign", error))
    }

    pub fn closure_assign(&mut self) -> Result<(), CudaError> {
        let closure = self
            .dev
            .get_func(MODULE_NAME, CLOSURE_ASSIGN_KERNEL)
            .ok_or(CudaError::missing_function(CLOSURE_ASSIGN_KERNEL))?;
        unsafe {
            closure.launch(
                kernel_config(),
                (&mut self.transition, &mut self.scratch, &mut self.next_hop),
            )
        }
        .map_err(|error| CudaError::new("quantale_closure_assign", error))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.dev
            .synchronize()
            .map_err(|error| CudaError::new("CudaDevice::synchronize", error))
    }

    pub fn step(&mut self) -> Result<QuantaleCudaReport, CudaError> {
        let step = self
            .dev
            .get_func(MODULE_NAME, STEP_KERNEL)
            .ok_or(CudaError::missing_function(STEP_KERNEL))?;
        unsafe {
            step.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.scratch,
                    &mut self.previous,
                    &mut self.next_hop,
                    &mut self.active,
                    &mut self.next_active,
                    &mut self.event_counts,
                    &mut self.report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_step", error))?;
        self.report()
    }

    /// Compute π(A*) on CUDA from the already-closed matrix: choose a destination
    /// from the active frontier and return W[src,dst] as the first executable hop.
    pub fn project_decision_path(&mut self) -> Result<DecisionProjection, CudaError> {
        let projection = self
            .dev
            .get_func(MODULE_NAME, DECISION_PROJECTION_KERNEL)
            .ok_or(CudaError::missing_function(DECISION_PROJECTION_KERNEL))?;
        unsafe {
            projection.launch(
                kernel_config(),
                (
                    &self.transition,
                    &self.next_hop,
                    &self.consumed,
                    &self.active,
                    &mut self.decision_report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_decide_path", error))?;
        self.decision_report()
    }

    /// Backwards-compatible name for the decision projection π(A*).
    pub fn decide(&mut self) -> Result<DecisionProjection, CudaError> {
        self.project_decision_path()
    }

    /// Fused Option-B frontier projection and update on GPU.
    pub fn frontier_step(&mut self) -> Result<DecisionProjection, CudaError> {
        let frontier = self
            .dev
            .get_func(MODULE_NAME, FRONTIER_STEP_KERNEL)
            .ok_or(CudaError::missing_function(FRONTIER_STEP_KERNEL))?;
        unsafe {
            frontier.launch(
                kernel_config(),
                (
                    &self.transition,
                    &self.next_hop,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &mut self.decision_report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_frontier_step", error))?;
        self.decision_report()
    }

    pub fn report(&self) -> Result<QuantaleCudaReport, CudaError> {
        let reports = self
            .dev
            .dtoh_sync_copy(&self.report)
            .map_err(|error| CudaError::new("dtoh_sync_copy report", error))?;
        reports.into_iter().next().ok_or(CudaError {
            operation: "dtoh_sync_copy report",
            message: "empty report buffer".to_string(),
        })
    }

    pub fn decision_report(&self) -> Result<DecisionReport, CudaError> {
        let reports = self
            .dev
            .dtoh_sync_copy(&self.decision_report)
            .map_err(|error| CudaError::new("dtoh_sync_copy decision_report", error))?;
        reports.into_iter().next().ok_or(CudaError {
            operation: "dtoh_sync_copy decision_report",
            message: "empty decision report buffer".to_string(),
        })
    }

    pub fn next_hop_matrix(&self) -> Result<Vec<i32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.next_hop)
            .map_err(|error| CudaError::new("dtoh_sync_copy next_hop", error))
    }

    /// Download W only and reconstruct a concrete node path from `src` to `dst`.
    pub fn reconstruct_path(&self, src: Node, dst: Node) -> Result<Vec<Node>, CudaError> {
        let next_hop = self.next_hop_matrix()?;
        reconstruct_path_from_next_hop(&next_hop, src, dst)
    }

    /// Reconstruct the last projected path using `selected_src` and
    /// `selected_dst` from `quantale_decide_path`.
    pub fn reconstruct_projected_path(&self) -> Result<Vec<Node>, CudaError> {
        let decision = self.decision_report()?;
        let src = Node::decode_index(decision.selected_src as usize).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct projected path with invalid selected_src {}",
                decision.selected_src
            ))
        })?;
        let dst = Node::decode_index(decision.selected_dst as usize).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct projected path with invalid selected_dst {}",
                decision.selected_dst
            ))
        })?;

        self.reconstruct_path(src, dst)
    }
}

fn kernel_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (THREAD_COUNT as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn run_once() -> Result<QuantaleCudaReport, CudaError> {
    let mut world = CudaWorld::new()?;
    world.step()
}
