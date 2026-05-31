//! Data-driven tensor transition graph entrypoint.

use crate::tensor::TensorEdge;

/// Complete tensor transition graph from the bundled topology asset.
pub fn full_tensor_transition_edges() -> Vec<TensorEdge> {
    crate::topology::load_default_tensor_topology_edges()
        .expect("bundled assets/topology.json tensor fields must compile")
}
