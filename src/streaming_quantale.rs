use crate::error::CudaError;
use crate::streaming::{StreamReceipt, TopologyDelta};
use crate::tensor::{TensorEdge, TensorQuantaleWorld};
use crate::topology::TopologyRuntime;

// ── Delta types ──────────────────────────────────────────────────────────────

/// A streaming update that mutates one edge in the quantale tensor.
pub struct EdgeDelta {
    pub source: String,
    pub src: String,
    pub dst: String,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
    pub observed_at: String,
    pub event_hash: String,
}

/// A streaming update that activates or clears one node in the active frontier.
pub struct NodeDelta {
    pub source: String,
    pub node: String,
    pub active: bool,
    pub observed_at: String,
    pub event_hash: String,
}

// ── Delta application ────────────────────────────────────────────────────────

// ── Stream event frontier activation ────────────────────────────────────────

const EVENT_STREAM_UPDATED: &str = "Event::StreamUpdated";
const EVENT_MARKET_FEED_UPDATED: &str = "Event::MarketFeedUpdated";

/// Translate applied `StreamReceipt`s into quantale event-node activations.
///
/// - Any applied receipt triggers `Event::StreamUpdated`.
/// - Any applied receipt whose slot starts with `"market."` additionally
///   triggers `Event::MarketFeedUpdated`.
///
/// Dropped receipts are silently ignored.
pub fn activate_stream_event_nodes(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    receipts: &[StreamReceipt],
) -> Result<(), CudaError> {
    let registry = topology.registry();

    let any_applied = receipts.iter().any(|r| r.applied);
    let any_market = receipts
        .iter()
        .any(|r| r.applied && r.slot.starts_with("market."));

    if any_applied {
        if let Some(id) = registry.id_of(EVENT_STREAM_UPDATED) {
            world.mark_node_active(id as i32)?;
        }
    }

    if any_market {
        if let Some(id) = registry.id_of(EVENT_MARKET_FEED_UPDATED) {
            world.mark_node_active(id as i32)?;
        }
    }

    Ok(())
}

// ── Delta application ─────────────────────────────────────────────────────────

/// Embed `delta` into `world`'s quantale tensor.
///
/// Returns `Err` if either node name is absent from the topology registry.
pub fn apply_edge_delta(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    delta: EdgeDelta,
) -> Result<(), CudaError> {
    let registry = topology.registry();
    let src_id = registry.id_of(&delta.src).ok_or_else(|| {
        CudaError::invalid_input(format!(
            "apply_edge_delta: unknown src node '{}'",
            delta.src
        ))
    })? as i32;
    let dst_id = registry.id_of(&delta.dst).ok_or_else(|| {
        CudaError::invalid_input(format!(
            "apply_edge_delta: unknown dst node '{}'",
            delta.dst
        ))
    })? as i32;
    let edge = TensorEdge::new(src_id, dst_id, delta.confidence, delta.cost, delta.safety);
    world.embed_tensor_edges(&[edge])
}

/// Activate or skip a node in `world`'s frontier based on `delta.active`.
///
/// Returns `Err` if the node name is absent from the topology registry.
/// A delta with `active = false` is accepted but is a no-op (the kernel only
/// sets; clearing is handled by the normal scheduler reset path).
pub fn apply_node_delta(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    delta: NodeDelta,
) -> Result<(), CudaError> {
    let registry = topology.registry();
    let node_id = registry.id_of(&delta.node).ok_or_else(|| {
        CudaError::invalid_input(format!(
            "apply_node_delta: unknown node '{}'",
            delta.node
        ))
    })? as i32;
    if delta.active {
        world.mark_node_active(node_id)?;
    }
    Ok(())
}

/// Apply a `TopologyDelta` produced by the stream normalizer to `world`.
///
/// This is the main entry point called from the orchestrator's pre-burst loop
/// after draining `StreamWorkers::drain_topology_deltas`.
pub fn apply_topology_delta(
    world: &mut TensorQuantaleWorld,
    topology: &TopologyRuntime,
    delta: TopologyDelta,
) -> Result<(), CudaError> {
    match delta {
        TopologyDelta::Edge {
            src,
            dst,
            confidence,
            cost,
            safety,
            source,
            observed_at,
            event_hash,
        } => apply_edge_delta(
            world,
            topology,
            EdgeDelta {
                source,
                src,
                dst,
                confidence,
                cost,
                safety,
                observed_at,
                event_hash,
            },
        ),
        TopologyDelta::Node {
            node,
            active,
            source,
            observed_at,
            event_hash,
        } => apply_node_delta(
            world,
            topology,
            NodeDelta {
                source,
                node,
                active,
                observed_at,
                event_hash,
            },
        ),
    }
}
