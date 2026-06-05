//! Topology facade: re-exports, runtime loader, and tensor edge adapter.

use crate::console;
use crate::error::CudaError;
use crate::tensor::{TENSOR_NODE_COUNT, TensorEdge};

pub use topology_core::{
    CompiledTopology, CompiledTransition, DominatorPair, GraphTopology, NodeRegistry,
    TopologyError, TopologyInvariants, TopologyNode, TopologyPage, TopologyTransition,
    TopologyViolation, ViolationKind, check, check_with_operators, format_violations,
};

pub fn load_default_tensor_topology_edges() -> Result<Vec<TensorEdge>, CudaError> {
    Ok(TopologyRuntime::load_checked_default()?
        .tensor_edges()
        .to_vec())
}

pub fn full_tensor_transition_edges() -> Vec<TensorEdge> {
    load_default_tensor_topology_edges()
        .expect("topology.generated.json or bundled topology must compile")
}

impl From<CompiledTransition> for TensorEdge {
    fn from(e: CompiledTransition) -> TensorEdge {
        TensorEdge::new(e.src as i32, e.dst as i32, e.confidence, e.cost, e.safety)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopologyRuntime {
    pub document: GraphTopology,
    pub compiled: CompiledTopology,
    pub tensor_edges: Vec<TensorEdge>,
    /// CKA `par` groups from the generated topology, with node names resolved
    /// to integer IDs.  Groups with any unknown node are silently dropped.
    /// Each group has at least two members.
    pub parallel_groups: Vec<Vec<i32>>,
}

impl TopologyRuntime {
    pub fn load_checked_default() -> Result<Self, CudaError> {
        let document = GraphTopology::default_asset()?;
        let invariants = TopologyInvariants::default_asset();
        let violations = check(&document, &invariants);
        // ConsumedBlockPoint is a structural warning, not a startup blocker —
        // the hard-reset path re-embeds accumulated edges and resets consumed[],
        // which handles re-entry correctly at runtime.  Emit as warnings only.
        let (warnings, fatal): (Vec<_>, Vec<_>) = violations
            .into_iter()
            .partition(|v| v.kind == ViolationKind::ConsumedBlockPoint);
        for v in &warnings {
            console::warn("topology", "violation", &[("detail", v.to_string())]);
        }
        if !fatal.is_empty() {
            return Err(CudaError::invalid_input(format!(
                "{}\n{} violation(s) found",
                format_violations(&fatal),
                fatal.len()
            )));
        }
        let compiled = document.compile()?;
        if compiled.node_count > TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "topology has {} nodes but generated tensor capacity is {}; rebuild after updating topology assets",
                compiled.node_count, TENSOR_NODE_COUNT
            )));
        }
        let tensor_edges = compiled
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        let parallel_groups = document
            .parallel_groups
            .iter()
            .filter_map(|group| {
                let ids: Vec<i32> = group
                    .iter()
                    .filter_map(|name| compiled.registry.id_of(name).map(|id| id as i32))
                    .collect();
                (ids.len() >= 2 && ids.len() == group.len()).then_some(ids)
            })
            .collect();
        Ok(Self {
            document,
            compiled,
            tensor_edges,
            parallel_groups,
        })
    }

    pub fn registry(&self) -> &NodeRegistry {
        &self.compiled.registry
    }

    pub fn tensor_edges(&self) -> &[TensorEdge] {
        &self.tensor_edges
    }
}
