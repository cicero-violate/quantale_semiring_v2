//! First-hop witness path reconstruction.

use crate::error::CudaError;
use crate::node::{MATRIX_LEN, NODE_COUNT, Node, node_name};

pub fn reconstruct_path_from_witness_matrix(
    witness_matrix: &[i32],
    src: Node,
    dst: Node,
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

    for _ in 0..NODE_COUNT {
        let idx = current_id as usize * NODE_COUNT + dst_id as usize;
        let hop_id = witness_matrix[idx];
        let Some(hop) = Node::decode(hop_id) else {
            return Err(CudaError::invalid_input(format!(
                "missing witness witness from {} to {}",
                node_name(current_id),
                node_name(dst_id)
            )));
        };

        if hop_id == current_id {
            return Err(CudaError::invalid_input(format!(
                "stalled witness witness at {} while targeting {}",
                node_name(current_id),
                node_name(dst_id)
            )));
        }

        path.push(hop);
        current_id = hop_id;

        if current_id == dst_id {
            return Ok(path);
        }
    }

    Err(CudaError::invalid_input(format!(
        "witness witness did not converge from {} to {} within {NODE_COUNT} hops",
        src.name(),
        dst.name()
    )))
}
