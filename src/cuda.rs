//! CUDA-resident quantale matrix and kernels.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::algebra::BOTTOM;
use crate::edge::LatticeEdge;
use crate::error::CudaError;
use crate::node::{Node, MATRIX_LEN, NODE_COUNT, THREAD_COUNT};
use crate::path::reconstruct_path_from_witness_matrix;
use crate::policy::{build_policy_edges, ExecutionGatePolicy};
use crate::projection::{DecisionProjection, DecisionReport, QuantaleCudaReport};
use crate::receipt::{build_receipt_edges, ExecutionReceipt};
use crate::search::{
    build_search_delta_edges, build_search_edges, DomainCandidate, ScoredCandidate,
};
use crate::transitions::default_transition_edges;

const MODULE_NAME: &str = "quantale_semiring_v2";
const RESET_KERNEL: &str = "quantale_reset";
const EMBED_ELEMENTS_KERNEL: &str = "quantale_embed_elements";
const SUPREMUM_ASSIGN_KERNEL: &str = "quantale_supremum_assign";
const TENSOR_ASSIGN_KERNEL: &str = "quantale_tensor_assign";
const LEAST_FIXED_POINT_KERNEL: &str = "quantale_least_fixed_point";
const STEP_KERNEL: &str = "quantale_step";
const TICK_KERNEL: &str = "quantale_tick";
/// CUDA kernel for the operational projection π(A*) from closed quantale paths.
const QUANTALE_MORPHISM_KERNEL: &str = "quantale_morphism";
const FRONTIER_STEP_KERNEL: &str = "quantale_frontier_step";
const KERNEL_SOURCE: &str = include_str!("../cuda/quantale_world.cu");

pub struct GpuQuantaleMatrix {
    dev: Arc<CudaDevice>,
    // === Full MATRIX_LEN buffers: N × N ===
    /// CUDA-resident A / A*: quantale transition matrix and closed path values.
    transition: CudaSlice<f32>,
    /// CUDA-resident scratch matrix for A* closure/composition.
    scratch: CudaSlice<f32>,
    /// CUDA-resident W witness matrix: first hop for the selected path value.
    witness_matrix: CudaSlice<i32>,
    /// CUDA-resident scratch witness matrix used by tensor composition.
    scratch_witness: CudaSlice<i32>,
    /// CUDA-resident execution-history matrix.
    /// consumed[src, dst] prevents repeated execution of the same transition.
    consumed: CudaSlice<i32>,
    /// CUDA-resident previous A / A* matrix.
    /// Used for delta detection, convergence checks, and reporting.
    previous: CudaSlice<f32>,
    // === Non-matrix runtime buffers ===
    /// CUDA-resident active frontier vector over NODE_COUNT nodes.
    active: CudaSlice<i32>,
    /// CUDA-resident next active frontier vector over NODE_COUNT nodes.
    next_active: CudaSlice<i32>,
    /// CUDA-resident per-thread event counter scratch buffer.
    event_counts: CudaSlice<i32>,
    /// CUDA-resident compact closure telemetry report.
    report: CudaSlice<QuantaleCudaReport>,
    /// CUDA-resident compact executable decision projection report.
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
                EMBED_ELEMENTS_KERNEL,
                SUPREMUM_ASSIGN_KERNEL,
                TENSOR_ASSIGN_KERNEL,
                LEAST_FIXED_POINT_KERNEL,
                STEP_KERNEL,
                TICK_KERNEL,
                QUANTALE_MORPHISM_KERNEL,
                FRONTIER_STEP_KERNEL,
            ],
        )
        .map_err(|error| CudaError::new("load_ptx", error))?;

        let transition = dev
            .htod_copy(vec![BOTTOM; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy transition", error))?;
        let scratch = dev
            .htod_copy(vec![BOTTOM; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy scratch", error))?;
        let witness_matrix = dev
            .htod_copy(vec![-1_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy witness_matrix", error))?;
        let scratch_witness = dev
            .htod_copy(vec![-1_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy scratch_witness", error))?;
        let consumed = dev
            .htod_copy(vec![0_i32; MATRIX_LEN])
            .map_err(|error| CudaError::new("htod_copy consumed", error))?;
        let previous = dev
            .htod_copy(vec![BOTTOM; MATRIX_LEN])
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
            witness_matrix,
            scratch_witness,
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

    pub fn from_edges(edges: &[LatticeEdge]) -> Result<Self, CudaError> {
        let mut matrix = Self::empty()?;
        matrix.embed_elements(edges)?;
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
                    &mut self.witness_matrix,
                    &mut self.scratch_witness,
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

    pub fn embed_elements(&mut self, edges: &[LatticeEdge]) -> Result<(), CudaError> {
        let edge_count = i32::try_from(edges.len())
            .map_err(|_| CudaError::invalid_input("too many lattice elements"))?;
        let edge_buffer = self
            .dev
            .htod_copy(edges.to_vec())
            .map_err(|error| CudaError::new("htod_copy edges", error))?;
        let embed_elements = self
            .dev
            .get_func(MODULE_NAME, EMBED_ELEMENTS_KERNEL)
            .ok_or(CudaError::missing_function(EMBED_ELEMENTS_KERNEL))?;
        unsafe {
            embed_elements.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.witness_matrix,
                    &edge_buffer,
                    edge_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_embed_elements", error))
    }

    /// Join matrix-native policy edges into the current transition matrix.
    ///
    /// This joins policy into the matrix. Projection reads matrix structure only.
    pub fn join_policy_edges(&mut self, policy: ExecutionGatePolicy) -> Result<(), CudaError> {
        let policy_edges = build_policy_edges(policy);
        self.embed_elements(&policy_edges)
    }

    /// Write a single feedback weight from a process receipt directly into VRAM.
    ///
    /// This is the primary feedback path for the agent loop: after every operator
    /// execution, the raw Unix exit-code weight is injected back onto the selected
    /// edge so the least fixed point can route away from failed nodes on the next tick.
    pub fn join_empirical_element(
        &mut self,
        src: i32,
        dst: i32,
        weight: f32,
    ) -> Result<(), CudaError> {
        self.embed_elements(&[LatticeEdge::new(src, dst, weight)])
    }

    /// Join runtime receipt evidence into the current transition matrix.
    pub fn join_receipt_edges(&mut self, receipt: ExecutionReceipt) -> Result<(), CudaError> {
        let receipt_edges = build_receipt_edges(receipt);
        self.embed_elements(&receipt_edges)
    }

    /// Join selected runtime search evidence into the current transition matrix.
    pub fn join_search_edges(&mut self, top_k: &[ScoredCandidate]) -> Result<(), CudaError> {
        let search_edges = build_search_edges(top_k);
        self.embed_elements(&search_edges)
    }

    /// Score external candidates, select top-k, and join search evidence.
    pub fn join_search_candidates(
        &mut self,
        candidates: impl IntoIterator<Item = DomainCandidate>,
        k: usize,
    ) -> Result<Vec<ScoredCandidate>, CudaError> {
        let (top_k, search_edges) = build_search_delta_edges(candidates, k);
        self.embed_elements(&search_edges)?;
        Ok(top_k)
    }

    /// Represent policy as matrix edges. Projection reads matrix structure only.
    pub fn apply_execution_policy(&mut self, policy: ExecutionGatePolicy) -> Result<(), CudaError> {
        self.join_policy_edges(policy)
    }

    pub fn supremum_assign(&mut self, rhs: &Self) -> Result<(), CudaError> {
        if !Arc::ptr_eq(&self.dev, &rhs.dev) {
            return Err(CudaError::invalid_input(
                "supremum_assign requires both matrices to belong to the same CUDA device/context",
            ));
        }
        let join = self
            .dev
            .get_func(MODULE_NAME, SUPREMUM_ASSIGN_KERNEL)
            .ok_or(CudaError::missing_function(SUPREMUM_ASSIGN_KERNEL))?;
        unsafe {
            join.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.witness_matrix,
                    &rhs.transition,
                    &rhs.witness_matrix,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_supremum_assign", error))
    }

    pub fn tensor_assign(&mut self, rhs: &Self) -> Result<(), CudaError> {
        if !Arc::ptr_eq(&self.dev, &rhs.dev) {
            return Err(CudaError::invalid_input(
                "tensor_assign requires both matrices to belong to the same CUDA device/context",
            ));
        }
        let mul = self
            .dev
            .get_func(MODULE_NAME, TENSOR_ASSIGN_KERNEL)
            .ok_or(CudaError::missing_function(TENSOR_ASSIGN_KERNEL))?;
        unsafe {
            mul.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.witness_matrix,
                    &rhs.transition,
                    &rhs.witness_matrix,
                    &mut self.scratch,
                    &mut self.scratch_witness,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_tensor_assign", error))
    }

    pub fn star_assign(&mut self) -> Result<(), CudaError> {
        let closure = self
            .dev
            .get_func(MODULE_NAME, LEAST_FIXED_POINT_KERNEL)
            .ok_or(CudaError::missing_function(LEAST_FIXED_POINT_KERNEL))?;
        unsafe {
            closure.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.scratch,
                    &mut self.witness_matrix,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_least_fixed_point", error))
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
                    &mut self.witness_matrix,
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

    /// Fused host tick: CUDA performs closure/report and frontier projection/update
    /// in one launch. This removes the previous `step()` then `frontier_step()`
    /// host launch boundary for the main runtime loop.
    pub fn tick(&mut self) -> Result<(QuantaleCudaReport, DecisionProjection), CudaError> {
        let tick = self
            .dev
            .get_func(MODULE_NAME, TICK_KERNEL)
            .ok_or(CudaError::missing_function(TICK_KERNEL))?;
        unsafe {
            tick.launch(
                kernel_config(),
                (
                    &mut self.transition,
                    &mut self.scratch,
                    &mut self.previous,
                    &mut self.witness_matrix,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &mut self.event_counts,
                    &mut self.report,
                    &mut self.decision_report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_tick", error))?;
        Ok((self.report()?, self.decision_report()?))
    }

    /// Compute π(A*) on CUDA from the already-closed matrix: choose a destination
    /// from the active frontier and return W[src,dst] as the first executable hop.
    pub fn project_decision_path(&mut self) -> Result<DecisionProjection, CudaError> {
        let projection = self
            .dev
            .get_func(MODULE_NAME, QUANTALE_MORPHISM_KERNEL)
            .ok_or(CudaError::missing_function(QUANTALE_MORPHISM_KERNEL))?;
        unsafe {
            projection.launch(
                kernel_config(),
                (
                    &self.transition,
                    &self.witness_matrix,
                    &self.consumed,
                    &self.active,
                    &mut self.decision_report,
                ),
            )
        }
        .map_err(|error| CudaError::new("quantale_morphism", error))?;
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
                    &self.witness_matrix,
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

    pub fn witness_matrix(&self) -> Result<Vec<i32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.witness_matrix)
            .map_err(|error| CudaError::new("dtoh_sync_copy witness_matrix", error))
    }

    /// Download W only and reconstruct a concrete node path from `src` to `dst`.
    pub fn reconstruct_path(&self, src: Node, dst: Node) -> Result<Vec<Node>, CudaError> {
        let witness_matrix = self.witness_matrix()?;
        reconstruct_path_from_witness_matrix(&witness_matrix, src, dst)
    }

    /// Reconstruct the last projected path using `selected_src` and
    /// `selected_dst` from `quantale_morphism`.
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
