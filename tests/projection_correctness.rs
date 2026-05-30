use quantale_semiring_v2::{
    ControlNode, DecisionReport, DomainCandidate, EventNode, MATRIX_LEN, NODE_COUNT, Node,
    Q_BOTTOM, QuantaleAction, StateNode, build_candidate_edges, build_receipt_edges,
    reconstruct_path_from_next_hop,
};

fn host_project(
    closed: &[f32],
    next_hop: &[i32],
    active: &[i32],
    consumed: &[i32],
) -> DecisionReport {
    let mut best_src = -1;
    let mut best_dst = -1;
    let mut best_value = Q_BOTTOM;
    let mut best_first_hop = -1;

    for src in 0..NODE_COUNT {
        if active[src] == 0 {
            continue;
        }
        for dst in 0..NODE_COUNT {
            let idx = src * NODE_COUNT + dst;
            let value = closed[idx];
            if consumed[idx] == 0 && value > best_value {
                best_src = src as i32;
                best_dst = dst as i32;
                best_value = value;
                best_first_hop = next_hop[idx];
            }
        }
    }

    DecisionReport {
        step: 0,
        selected_src: best_src,
        selected_dst: best_dst,
        first_hop: best_first_hop,
        selected_value: best_value,
        halted: i32::from(best_dst == Node::control(ControlNode::Halt).encode()),
        blocked: i32::from(best_dst < 0),
    }
}

#[test]
fn projection_selects_max_reachable_active_frontier_destination() {
    let src = Node::state(StateNode::Goal).encode() as usize;
    let low = Node::state(StateNode::Input).encode() as usize;
    let high = Node::control(ControlNode::GateInput).encode() as usize;
    let mut closed = vec![Q_BOTTOM; MATRIX_LEN];
    let mut next_hop = vec![-1_i32; MATRIX_LEN];
    let mut active = vec![0_i32; NODE_COUNT];
    let consumed = vec![0_i32; MATRIX_LEN];
    active[src] = 1;
    closed[src * NODE_COUNT + low] = 0.2;
    closed[src * NODE_COUNT + high] = 0.9;
    next_hop[src * NODE_COUNT + low] = low as i32;
    next_hop[src * NODE_COUNT + high] = high as i32;

    let before = closed.clone();
    let decision = host_project(&closed, &next_hop, &active, &consumed);

    assert_eq!(decision.blocked, 0);
    assert_eq!(decision.selected_src, src as i32);
    assert_eq!(decision.selected_dst, high as i32);
    assert_eq!(decision.first_hop, high as i32);
    assert_eq!(closed, before, "projection must not mutate A*");
}

#[test]
fn projection_blocks_when_no_valid_candidate_exists() {
    let closed = vec![Q_BOTTOM; MATRIX_LEN];
    let next_hop = vec![-1_i32; MATRIX_LEN];
    let mut active = vec![0_i32; NODE_COUNT];
    let consumed = vec![0_i32; MATRIX_LEN];
    active[Node::state(StateNode::Goal).encode() as usize] = 1;

    let decision = host_project(&closed, &next_hop, &active, &consumed);

    assert_eq!(decision.blocked, 1);
    assert_eq!(decision.halted, 0);
    assert_eq!(decision.selected_dst, -1);
}

#[test]
fn projection_marks_halted_only_for_halt_destination() {
    let src = Node::state(StateNode::Goal).encode() as usize;
    let halt = Node::control(ControlNode::Halt).encode() as usize;
    let mut closed = vec![Q_BOTTOM; MATRIX_LEN];
    let mut next_hop = vec![-1_i32; MATRIX_LEN];
    let mut active = vec![0_i32; NODE_COUNT];
    let consumed = vec![0_i32; MATRIX_LEN];
    active[src] = 1;
    closed[src * NODE_COUNT + halt] = 0.7;
    next_hop[src * NODE_COUNT + halt] = halt as i32;

    let decision = host_project(&closed, &next_hop, &active, &consumed);

    assert_eq!(decision.halted, 1);
    assert_eq!(decision.selected_action(), QuantaleAction::Stop);
}

#[test]
fn history_mask_prevents_repeated_first_hop_selection() {
    let src = Node::state(StateNode::Goal).encode() as usize;
    let first = Node::state(StateNode::Input).encode() as usize;
    let second = Node::state(StateNode::Parse).encode() as usize;
    let mut closed = vec![Q_BOTTOM; MATRIX_LEN];
    let mut next_hop = vec![-1_i32; MATRIX_LEN];
    let mut active = vec![0_i32; NODE_COUNT];
    let mut consumed = vec![0_i32; MATRIX_LEN];
    active[src] = 1;
    closed[src * NODE_COUNT + first] = 0.9;
    closed[src * NODE_COUNT + second] = 0.8;
    next_hop[src * NODE_COUNT + first] = first as i32;
    next_hop[src * NODE_COUNT + second] = second as i32;
    consumed[src * NODE_COUNT + first] = 1;

    let decision = host_project(&closed, &next_hop, &active, &consumed);

    assert_eq!(decision.selected_dst, second as i32);
    assert_eq!(decision.first_hop, second as i32);
}

#[test]
fn candidate_generation_compiles_external_candidates_to_edges() {
    let candidates = [
        DomainCandidate::new("low", "external", 0.4, 1.0, 0.0, 0.0),
        DomainCandidate::new("high", "external", 0.9, 1.0, 0.0, 0.0),
    ];

    let (top, edges) = build_candidate_edges(candidates, 1);

    assert_eq!(top.len(), 1);
    assert_eq!(top[0].candidate.id, "high");
    assert!(edges.iter().any(|edge| {
        edge.src == Node::state(StateNode::Search).encode()
            && edge.dst == Node::event(EventNode::CandidateFound).encode()
    }));
}

#[test]
fn receipt_feedback_edges_distinguish_success_and_failure() {
    let accepted = build_receipt_edges(quantale_semiring_v2::ExecutionReceipt::accepted(
        0.9, 0.9, 0.9,
    ));
    let rejected = build_receipt_edges(quantale_semiring_v2::ExecutionReceipt::rejected(
        0.9, 0.8, 0.7,
    ));

    assert!(
        accepted
            .iter()
            .any(|edge| { edge.dst == Node::event(EventNode::ReceiptAccepted).encode() })
    );
    assert!(
        rejected
            .iter()
            .any(|edge| { edge.dst == Node::event(EventNode::ReceiptRejected).encode() })
    );
    assert!(
        rejected
            .iter()
            .any(|edge| { edge.dst == Node::control(ControlNode::Rollback).encode() })
    );
}

#[test]
fn projection_first_hop_matches_witness_matrix() {
    let src = Node::state(StateNode::Goal);
    let hop = Node::control(ControlNode::GateInput);
    let dst = Node::event(EventNode::FactArrived);
    let mut next_hop = vec![-1_i32; MATRIX_LEN];
    next_hop[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = hop.encode();
    next_hop[hop.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

    let path = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap();

    assert_eq!(path[1], hop);
}
