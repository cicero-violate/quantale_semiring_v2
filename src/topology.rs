//! Topology facade: re-exports, runtime loader, and tensor edge adapter.

use crate::error::CudaError;
use crate::tensor::TensorEdge;

pub use topology_core::{
    CompiledTopology, CompiledTransition, GraphTopology, NodeRegistry, TopologyError, TopologyNode,
    TopologyOverlay, TopologyPage, TopologyTransition,
    DominatorPair, TopologyInvariants, TopologyViolation, ViolationKind,
    check, check_with_operators, format_violations,
};

pub fn load_default_tensor_topology_edges() -> Result<Vec<TensorEdge>, CudaError> {
    Ok(TopologyRuntime::load_checked_default()?.tensor_edges().to_vec())
}

pub fn full_tensor_transition_edges() -> Vec<TensorEdge> {
    load_default_tensor_topology_edges()
        .expect("bundled assets/topology.json tensor fields must compile")
}

impl From<CompiledTransition> for TensorEdge {
    fn from(e: CompiledTransition) -> TensorEdge {
        TensorEdge::new(e.src as i32, e.dst as i32, e.confidence, e.cost, e.safety)
    }
}

#[derive(Clone, Debug)]
pub struct TopologyRuntime {
    pub document: GraphTopology,
    pub compiled: CompiledTopology,
    pub tensor_edges: Vec<TensorEdge>,
}

impl TopologyRuntime {
    pub fn load_checked_default() -> Result<Self, CudaError> {
        let document = GraphTopology::default_asset()?;
        let invariants = TopologyInvariants::default_asset();
        let violations = check(&document, &invariants);
        if !violations.is_empty() {
            return Err(CudaError::invalid_input(format!(
                "{}\n{} violation(s) found",
                format_violations(&violations),
                violations.len()
            )));
        }
        let compiled = document.compile()?;
        let tensor_edges = compiled
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        Ok(Self {
            document,
            compiled,
            tensor_edges,
        })
    }

    pub fn registry(&self) -> &NodeRegistry {
        &self.compiled.registry
    }

    pub fn node_name(&self, id: usize) -> Option<&str> {
        self.compiled.registry.name_of(id)
    }

    pub fn tensor_edges(&self) -> &[TensorEdge] {
        &self.tensor_edges
    }
}
