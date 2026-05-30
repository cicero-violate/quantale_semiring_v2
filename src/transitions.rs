//! Data-driven transition graph entrypoints.
//!
//! Transition topology is owned by `assets/topology.json`. This module keeps the
//! previous public API names as thin compatibility wrappers while removing the
//! hardcoded Rust edge table from the live compilation layer.

use crate::edge::TransitionEdge;

/// Load all bundled transition edges from the JSON topology asset.
pub fn data_driven_transition_edges() -> Result<Vec<TransitionEdge>, crate::CudaError> {
    crate::topology::load_default_topology_edges()
}

/// Compatibility entrypoint for callers that historically requested the static
/// default graph. The default graph is now the bundled data topology.
pub fn default_transition_edges() -> Vec<TransitionEdge> {
    full_transition_edges()
}

/// Compatibility entrypoint for the former persistence-edge suffix. Persistence
/// is now encoded directly in `assets/topology.json`, so there is no separate
/// Rust-side suffix to append.
pub fn persistence_transition_edges() -> Vec<TransitionEdge> {
    Vec::new()
}

/// Compatibility entrypoint for the complete transition graph.
pub fn full_transition_edges() -> Vec<TransitionEdge> {
    data_driven_transition_edges().expect("bundled assets/topology.json must compile")
}
