use quantale_semiring_v2::{
    MATRIX_LEN, NODE_COUNT, Node, Q_BOTTOM, Q_UNIT, QuantaleWeight, StateNode,
    reconstruct_path_from_next_hop,
};

fn close_dense(mut matrix: Vec<f32>) -> Vec<f32> {
    for node in 0..NODE_COUNT {
        let idx = node * NODE_COUNT + node;
        matrix[idx] = matrix[idx].max(Q_UNIT);
    }
    for k in 0..NODE_COUNT {
        for i in 0..NODE_COUNT {
            let ik = matrix[i * NODE_COUNT + k];
            if ik <= Q_BOTTOM {
                continue;
            }
            for j in 0..NODE_COUNT {
                let candidate = ik * matrix[k * NODE_COUNT + j];
                let idx = i * NODE_COUNT + j;
                if candidate > matrix[idx] {
                    matrix[idx] = candidate;
                }
            }
        }
    }
    matrix
}

#[test]
fn quantale_join_is_idempotent_and_commutative() {
    let a = QuantaleWeight::new(0.37);
    let b = QuantaleWeight::new(0.91);

    assert_eq!(a.join(a), a);
    assert_eq!(a.join(b), b.join(a));
}

#[test]
fn quantale_compose_has_unit_and_bottom() {
    let a = QuantaleWeight::new(0.37);

    assert_eq!(a.compose(QuantaleWeight::UNIT), a);
    assert_eq!(QuantaleWeight::UNIT.compose(a), a);
    assert_eq!(a.compose(QuantaleWeight::BOTTOM), QuantaleWeight::BOTTOM);
    assert_eq!(QuantaleWeight::BOTTOM.compose(a), QuantaleWeight::BOTTOM);
}

#[test]
fn dense_closure_fixture_is_idempotent() {
    let mut matrix = vec![Q_BOTTOM; MATRIX_LEN];
    let goal = Node::state(StateNode::Goal).encode() as usize;
    let input = Node::state(StateNode::Input).encode() as usize;
    let parse = Node::state(StateNode::Parse).encode() as usize;

    matrix[goal * NODE_COUNT + input] = 0.8;
    matrix[input * NODE_COUNT + parse] = 0.5;

    let closed = close_dense(matrix);
    let closed_again = close_dense(closed.clone());

    assert_eq!(closed, closed_again);
    assert!((closed[goal * NODE_COUNT + parse] - 0.4).abs() <= f32::EPSILON);
}

#[test]
fn next_hop_witness_reconstructs_selected_path() {
    let src = Node::state(StateNode::Goal);
    let mid = Node::state(StateNode::Input);
    let dst = Node::state(StateNode::Parse);
    let mut next_hop = vec![-1_i32; MATRIX_LEN];
    next_hop[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = mid.encode();
    next_hop[mid.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

    let path = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap();

    assert_eq!(path, vec![src, mid, dst]);
}
