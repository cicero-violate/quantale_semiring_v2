//! Compact node IDs, projection reports, and witness path reconstruction.

use cudarc::driver::DeviceRepr;
use serde::Serialize;

use crate::error::CudaError;
use crate::tensor::{MATRIX_LEN, TENSOR_NODE_COUNT};
use crate::topology::NodeRegistry;
use crate::types::{BOTTOM, Q_UNIT};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node(pub i32);

impl Node {
    pub fn encode(self) -> i32 {
        self.0
    }

    pub fn decode(id: i32, registry: &NodeRegistry) -> Option<Self> {
        if id >= 0 && registry.name_of(id as usize).is_some() {
            Some(Self(id))
        } else {
            None
        }
    }

    pub fn name(self, registry: &NodeRegistry) -> Option<&str> {
        registry.name_of(self.0 as usize)
    }
}

/// Compact report for π(A*): the gated executable projection of least fixed point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct DecisionReport {
    pub step: i32,
    pub selected_src: i32,
    pub selected_dst: i32,
    pub first_hop: i32,
    pub selected_value: f32,
    pub halted: i32,
    pub blocked: i32,
}

unsafe impl DeviceRepr for DecisionReport {}

impl DecisionReport {
    pub fn selected_node(&self, registry: &NodeRegistry) -> Option<Node> {
        Node::decode(self.selected_dst, registry)
    }

    pub fn first_hop_node(&self, registry: &NodeRegistry) -> Option<Node> {
        Node::decode(self.first_hop, registry)
    }
}

pub fn action_label(node_id: i32, registry: &NodeRegistry) -> &str {
    if node_id < 0 {
        return "blocked";
    }
    registry.action_of(node_id as usize).unwrap_or("unknown")
}

pub fn format_quantale_value(value: f32) -> String {
    if value <= BOTTOM {
        "⊥".to_string()
    } else if (value - Q_UNIT).abs() <= f32::EPSILON {
        "e".to_string()
    } else {
        format!("{value:.4}")
    }
}

pub fn reconstruct_path_from_witness_matrix(
    witness_matrix: &[i32],
    src: Node,
    dst: Node,
    registry: &NodeRegistry,
) -> Result<Vec<Node>, CudaError> {
    if witness_matrix.len() != MATRIX_LEN {
        return Err(CudaError::invalid_input(format!(
            "expected {MATRIX_LEN} witness entries, got {}",
            witness_matrix.len()
        )));
    }

    let dst_id = dst.encode();
    let mut current_id = src.encode();
    let mut path = vec![src];

    if current_id == dst_id {
        return Ok(path);
    }

    for _ in 0..TENSOR_NODE_COUNT {
        let idx = current_id as usize * TENSOR_NODE_COUNT + dst_id as usize;
        let hop_id = witness_matrix[idx];
        let Some(hop) = Node::decode(hop_id, registry) else {
            return Err(CudaError::invalid_input(format!(
                "missing witness from {} to {}",
                node_name(current_id, registry),
                node_name(dst_id, registry)
            )));
        };

        if hop_id == current_id {
            return Err(CudaError::invalid_input(format!(
                "stalled witness at {} while targeting {}",
                node_name(current_id, registry),
                node_name(dst_id, registry)
            )));
        }

        path.push(hop);
        current_id = hop_id;

        if current_id == dst_id {
            return Ok(path);
        }
    }

    Err(CudaError::invalid_input(format!(
        "witness did not converge from {} to {} within {TENSOR_NODE_COUNT} hops",
        node_name(src.encode(), registry),
        node_name(dst.encode(), registry)
    )))
}

fn node_name(node_id: i32, registry: &NodeRegistry) -> String {
    Node::decode(node_id, registry)
        .and_then(|node| node.name(registry))
        .map(str::to_string)
        .unwrap_or_else(|| format!("Unknown({node_id})"))
}
