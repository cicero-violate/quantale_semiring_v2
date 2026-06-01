//! First-hop witness path reconstruction.

use crate::error::CudaError;
use crate::node::Node;
use crate::tensor::{MATRIX_LEN, TENSOR_NODE_COUNT};
use crate::topology::NodeRegistry;

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
                "missing witness witness from {} to {}",
                node_name(current_id, registry),
                node_name(dst_id, registry)
            )));
        };

        if hop_id == current_id {
            return Err(CudaError::invalid_input(format!(
                "stalled witness witness at {} while targeting {}",
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
        "witness witness did not converge from {} to {} within {TENSOR_NODE_COUNT} hops",
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
