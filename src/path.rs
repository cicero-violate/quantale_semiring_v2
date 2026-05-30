//! First-hop witness path reconstruction.

use crate::error::CudaError;
use crate::node::{MATRIX_LEN, NODE_COUNT, Node, node_name};

pub fn reconstruct_path_from_next_hop(
    next_hop: &[i32],
    src: Node,
    dst: Node,
) -> Result<Vec<Node>, CudaError> {
    if next_hop.len() != MATRIX_LEN {
        return Err(CudaError::invalid_input(format!(
            "expected {MATRIX_LEN} next-hop entries, got {}",
            next_hop.len()
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
        let hop_id = next_hop[idx];
        let Some(hop) = Node::decode(hop_id) else {
            return Err(CudaError::invalid_input(format!(
                "missing next-hop witness from {} to {}",
                node_name(current_id),
                node_name(dst_id)
            )));
        };

        if hop_id == current_id {
            return Err(CudaError::invalid_input(format!(
                "stalled next-hop witness at {} while targeting {}",
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
        "next-hop witness did not converge from {} to {} within {NODE_COUNT} hops",
        src.name(),
        dst.name()
    )))
}
