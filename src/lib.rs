pub mod algebra;
pub mod cuda;
pub mod edge;
pub mod error;
pub mod node;
pub mod path;
pub mod policy;
pub mod projection;
pub mod receipt;
pub mod transitions;

pub use algebra::*;
pub use cuda::*;
pub use edge::*;
pub use error::*;
pub use node::*;
pub use path::*;
pub use policy::*;
pub use projection::*;
pub use receipt::*;
pub use transitions::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ControlNode, EventNode, ExecutionReceipt, MATRIX_LEN, NODE_COUNT, Node, StateNode,
        build_receipt_edges, reconstruct_path_from_next_hop,
    };

    fn has_edge(edges: &[TransitionEdge], src: Node, dst: Node) -> bool {
        edges.iter().any(|edge| {
            edge.src == src.encode() && edge.dst == dst.encode() && edge.value > Q_BOTTOM
        })
    }

    #[test]
    fn edge_eval_uses_eval_weight() {
        let edge = edge_eval(
            Node::state(StateNode::Goal),
            Node::control(ControlNode::GateInput),
            Eval::new(0.80, 0.50, 0.25, 0.10),
        );

        assert_eq!(edge.src, Node::state(StateNode::Goal).encode());
        assert_eq!(edge.dst, Node::control(ControlNode::GateInput).encode());
        assert!((edge.value - 0.27).abs() <= f32::EPSILON);
    }

    #[test]
    fn policy_edges_allow_execution_and_commit_when_receipt_is_valid() {
        let policy = ExecutionGatePolicy {
            evidence_accepted: true,
            receipt_hash_nonzero: true,
            commit_allowed: true,
            ..ExecutionGatePolicy::default()
        };

        let edges = build_policy_edges(policy);

        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateExecution),
            Node::event(EventNode::ExecuteStarted)
        ));
        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptAccepted)
        ));
        assert!(has_edge(
            &edges,
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate)
        ));
        assert!(has_edge(
            &edges,
            Node::state(StateNode::Validate),
            Node::control(ControlNode::Commit)
        ));
    }

    #[test]
    fn policy_edges_route_blocked_execution_to_block_control() {
        let policy = ExecutionGatePolicy {
            task_blocked: true,
            retry_allowed: false,
            halt_allowed: true,
            ..ExecutionGatePolicy::default()
        };

        let edges = build_policy_edges(policy);

        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateExecution),
            Node::control(ControlNode::Block)
        ));
        assert!(has_edge(
            &edges,
            Node::control(ControlNode::ChooseBest),
            Node::control(ControlNode::Block)
        ));
        assert!(has_edge(
            &edges,
            Node::control(ControlNode::Block),
            Node::control(ControlNode::Halt)
        ));
        assert!(!has_edge(
            &edges,
            Node::control(ControlNode::GateExecution),
            Node::event(EventNode::ExecuteStarted)
        ));
    }

    #[test]
    fn accepted_receipt_edges_route_to_validation() {
        let receipt = ExecutionReceipt::accepted(0.88, 0.91, 0.93);

        let edges = build_receipt_edges(receipt);

        assert!(has_edge(
            &edges,
            Node::event(EventNode::ReceiptAttached),
            Node::control(ControlNode::GateReceipt)
        ));
        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptAccepted)
        ));
        assert!(has_edge(
            &edges,
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::HashNonzero)
        ));
        assert!(has_edge(
            &edges,
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate)
        ));
        assert!(!has_edge(
            &edges,
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected)
        ));
    }

    #[test]
    fn rejected_receipt_edges_route_to_rollback_and_repair() {
        let receipt = ExecutionReceipt::rejected(0.89, 0.86, 0.81);

        let edges = build_receipt_edges(receipt);

        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected)
        ));
        assert!(has_edge(
            &edges,
            Node::event(EventNode::ReceiptRejected),
            Node::control(ControlNode::Rollback)
        ));
        assert!(has_edge(
            &edges,
            Node::control(ControlNode::Rollback),
            Node::control(ControlNode::Repair)
        ));
        assert!(!has_edge(
            &edges,
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate)
        ));
    }

    #[test]
    fn accepted_receipt_without_hash_routes_to_rejection() {
        let receipt = ExecutionReceipt::accepted_without_hash(0.90, 0.84);

        let edges = build_receipt_edges(receipt);

        assert!(has_edge(
            &edges,
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected)
        ));
        assert!(!has_edge(
            &edges,
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::HashNonzero)
        ));
    }

    #[test]
    fn reconstruct_path_from_next_hop_returns_direct_path() {
        let src = Node::state(StateNode::Goal);
        let dst = Node::control(ControlNode::GateInput);
        let mut next_hop = vec![-1_i32; MATRIX_LEN];
        next_hop[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

        let path = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap();

        assert_eq!(path, vec![src, dst]);
    }

    #[test]
    fn reconstruct_path_from_next_hop_walks_multiple_hops() {
        let src = Node::state(StateNode::Goal);
        let gate = Node::control(ControlNode::GateInput);
        let event = Node::event(EventNode::FactArrived);
        let dst = Node::state(StateNode::Input);
        let mut next_hop = vec![-1_i32; MATRIX_LEN];
        next_hop[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = gate.encode();
        next_hop[gate.encode() as usize * NODE_COUNT + dst.encode() as usize] = event.encode();
        next_hop[event.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

        let path = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap();

        assert_eq!(path, vec![src, gate, event, dst]);
    }

    #[test]
    fn reconstruct_path_from_next_hop_rejects_missing_witness() {
        let src = Node::state(StateNode::Goal);
        let dst = Node::state(StateNode::Input);
        let next_hop = vec![-1_i32; MATRIX_LEN];

        let err = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap_err();

        assert_eq!(err.operation, "input");
        assert!(err.message.contains("missing next-hop witness"));
    }

    #[test]
    fn reconstruct_path_from_next_hop_rejects_cycle() {
        let src = Node::state(StateNode::Goal);
        let mid = Node::control(ControlNode::GateInput);
        let dst = Node::state(StateNode::Input);
        let mut next_hop = vec![-1_i32; MATRIX_LEN];
        next_hop[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = mid.encode();
        next_hop[mid.encode() as usize * NODE_COUNT + dst.encode() as usize] = src.encode();

        let err = reconstruct_path_from_next_hop(&next_hop, src, dst).unwrap_err();

        assert_eq!(err.operation, "input");
        assert!(err.message.contains("did not converge"));
    }
}
