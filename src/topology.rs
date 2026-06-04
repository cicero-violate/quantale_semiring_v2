//! Topology facade: re-exports, runtime loader, and tensor edge adapter.

use std::collections::HashSet;

use crate::console;
use crate::error::CudaError;
use crate::hot_region::HotRegionRegistry;
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
        Ok(Self {
            document,
            compiled,
            tensor_edges,
        })
    }

    pub fn registry(&self) -> &NodeRegistry {
        &self.compiled.registry
    }

    pub fn tensor_edges(&self) -> &[TensorEdge] {
        &self.tensor_edges
    }
}

// ── Split topology ─────────────────────────────────────────────────────────────

/// Synthetic runtime nodes that appear in the hot topology / `regions.hot.json`
/// but are NOT declared in `topology.source.json`.
///
/// These nodes are compiler-generated sentinels used by the hot-region
/// scheduler.  They do not correspond to any operator and must not be
/// dispatched through the operator executor.
///
/// Validators must treat these as an explicit whitelist, not as missing nodes.
pub const SYNTHETIC_HOT_NODES: &[&str] = &["Region::CommitReceipt"];

/// CPU-only control/IO subgraph loaded from `topology.control.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlTopologyRuntime {
    pub node_names: HashSet<String>,
}

impl ControlTopologyRuntime {
    pub fn contains(&self, name: &str) -> bool {
        self.node_names.contains(name)
    }
}

/// GPU-resident hot subgraph loaded from `topology.hot.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct HotTopologyRuntime {
    pub node_names: HashSet<String>,
    /// Region-level execution chains for the region scheduler.
    pub transitions: Vec<TopologyTransition>,
}

impl HotTopologyRuntime {
    pub fn contains(&self, name: &str) -> bool {
        self.node_names.contains(name)
    }
}

/// Two-part routing view: hot GPU regions + cold CPU control path.
///
/// The `unified` field keeps `topology.generated.json` alive for the CUDA
/// quantale tensor — node IDs there must not change.  The split files drive
/// routing logic only.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitTopologyRuntime {
    pub control: ControlTopologyRuntime,
    pub hot: HotTopologyRuntime,
    pub unified: TopologyRuntime,
}

impl SplitTopologyRuntime {
    /// Load and validate the split topology.
    ///
    /// Enforces at startup:
    /// - control ∩ hot == ∅ (by name)
    /// - all hot nodes have region registry metadata (except virtual terminal)
    /// - no GPU regions in the control topology
    /// - hot topology has at least one transition
    pub fn load_checked(registry: &HotRegionRegistry) -> Result<Self, CudaError> {
        let unified = TopologyRuntime::load_checked_default()?;

        let control_doc = load_split_doc("assets/topology.control.json")?;
        let hot_doc = load_split_doc("assets/topology.hot.json")?;

        let control_names: HashSet<String> =
            control_doc.nodes.iter().map(|n| n.name.clone()).collect();
        let hot_names: HashSet<String> =
            hot_doc.nodes.iter().map(|n| n.name.clone()).collect();

        // Invariant 1: disjoint node sets.
        let overlap: Vec<&str> = control_names
            .iter()
            .filter(|n| hot_names.contains(n.as_str()))
            .map(String::as_str)
            .collect();
        if !overlap.is_empty() {
            return Err(CudaError::invalid_input(format!(
                "control and hot topologies share nodes: {:?}",
                overlap
            )));
        }

        // Invariant 2: all hot nodes (except synthetic sentinels) have region metadata.
        let unregistered: Vec<&str> = hot_doc
            .nodes
            .iter()
            .filter(|n| {
                !SYNTHETIC_HOT_NODES.contains(&n.name.as_str()) && !registry.is_hot(&n.name)
            })
            .map(|n| n.name.as_str())
            .collect();
        if !unregistered.is_empty() {
            return Err(CudaError::invalid_input(format!(
                "hot topology nodes missing from region registry: {:?}",
                unregistered
            )));
        }

        // Invariant 3: no GPU compute in control topology.
        let bad: Vec<&str> = control_doc
            .nodes
            .iter()
            .filter(|n| n.is_gpu_region())
            .map(|n| n.name.as_str())
            .collect();
        if !bad.is_empty() {
            return Err(CudaError::invalid_input(format!(
                "control topology contains gpu_region nodes: {:?}",
                bad
            )));
        }

        // Invariant 4: hot topology must have at least one transition.
        if hot_doc.transitions.is_empty() {
            return Err(CudaError::invalid_input(
                "hot topology has no transitions".to_string(),
            ));
        }

        Ok(Self {
            control: ControlTopologyRuntime { node_names: control_names },
            hot: HotTopologyRuntime {
                node_names: hot_names,
                transitions: hot_doc.transitions,
            },
            unified,
        })
    }
}

fn load_split_doc(path: &str) -> Result<GraphTopology, CudaError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| CudaError::invalid_input(format!("read {path}: {e}")))?;
    serde_json::from_str::<GraphTopology>(&raw)
        .map_err(|e| CudaError::invalid_input(format!("parse {path}: {e}")))
}
