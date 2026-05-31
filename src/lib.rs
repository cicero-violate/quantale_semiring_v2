pub mod algebra;
pub mod config;
pub mod cuda;
pub mod dsl;
pub mod edge;
pub mod egress;
pub mod error;
pub mod ingress;
pub mod node;
pub mod paging;
pub mod path;
pub mod plan;
pub mod projection;
pub mod rule_delta;
pub mod search;
pub mod tensor;
pub mod tlog;
pub mod topology;
pub mod transitions;
pub mod types;

pub use algebra::*;
pub use config::*;
pub use cuda::*;
pub use dsl::*;
pub use edge::*;
pub use egress::*;
pub use error::*;
pub use ingress::*;
pub use node::*;
pub use paging::*;
pub use path::*;
pub use plan::*;
pub use projection::*;
pub use rule_delta::*;
pub use search::*;
pub use tensor::*;
pub use tlog::*;
pub use topology::*;
pub use transitions::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ControlNode, EventNode, ExecutionReceipt, MATRIX_LEN, NODE_COUNT, Node, StateNode,
        build_receipt_edges, reconstruct_path_from_witness_matrix,
    };

    fn has_edge(edges: &[LatticeEdge], src: Node, dst: Node) -> bool {
        edges
            .iter()
            .any(|edge| edge.src == src.encode() && edge.dst == dst.encode() && edge.value > BOTTOM)
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
    fn reconstruct_path_from_witness_matrix_returns_direct_path() {
        let src = Node::state(StateNode::Goal);
        let dst = Node::control(ControlNode::GateInput);
        let mut witness_matrix = vec![-1_i32; MATRIX_LEN];
        witness_matrix[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

        let path = reconstruct_path_from_witness_matrix(&witness_matrix, src, dst).unwrap();

        assert_eq!(path, vec![src, dst]);
    }

    #[test]
    fn reconstruct_path_from_witness_matrix_walks_multiple_hops() {
        let src = Node::state(StateNode::Goal);
        let gate = Node::control(ControlNode::GateInput);
        let event = Node::event(EventNode::FactArrived);
        let dst = Node::state(StateNode::Input);
        let mut witness_matrix = vec![-1_i32; MATRIX_LEN];
        witness_matrix[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = gate.encode();
        witness_matrix[gate.encode() as usize * NODE_COUNT + dst.encode() as usize] =
            event.encode();
        witness_matrix[event.encode() as usize * NODE_COUNT + dst.encode() as usize] = dst.encode();

        let path = reconstruct_path_from_witness_matrix(&witness_matrix, src, dst).unwrap();

        assert_eq!(path, vec![src, gate, event, dst]);
    }

    #[test]
    fn reconstruct_path_from_witness_matrix_rejects_missing_witness() {
        let src = Node::state(StateNode::Goal);
        let dst = Node::state(StateNode::Input);
        let witness_matrix = vec![-1_i32; MATRIX_LEN];

        let err = reconstruct_path_from_witness_matrix(&witness_matrix, src, dst).unwrap_err();

        assert_eq!(err.operation, "input");
        assert!(err.message.contains("missing witness witness"));
    }

    #[test]
    fn reconstruct_path_from_witness_matrix_rejects_cycle() {
        let src = Node::state(StateNode::Goal);
        let mid = Node::control(ControlNode::GateInput);
        let dst = Node::state(StateNode::Input);
        let mut witness_matrix = vec![-1_i32; MATRIX_LEN];
        witness_matrix[src.encode() as usize * NODE_COUNT + dst.encode() as usize] = mid.encode();
        witness_matrix[mid.encode() as usize * NODE_COUNT + dst.encode() as usize] = src.encode();

        let err = reconstruct_path_from_witness_matrix(&witness_matrix, src, dst).unwrap_err();

        assert_eq!(err.operation, "input");
        assert!(err.message.contains("did not converge"));
    }

    #[test]
    fn top_k_selection_orders_by_score_and_tie_breaks_by_id() {
        let scored = score_candidates(vec![
            DomainCandidate::new("b", "search candidate b", 0.80, 1.00, 0.00, 0.00),
            DomainCandidate::new("a", "search candidate a", 0.80, 1.00, 0.00, 0.00),
            DomainCandidate::new("c", "search candidate c", 0.50, 1.00, 0.00, 0.00),
        ]);

        let top = select_top_k(scored, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].candidate.id, "a");
        assert_eq!(top[1].candidate.id, "b");
    }

    #[test]
    fn external_candidates_compile_top_k_into_quantale_delta_edges() {
        let candidates = vec![
            DomainCandidate::new("low", "needle candidate", 0.50, 1.00, 0.00, 0.00),
            DomainCandidate::new("high", "needle candidate", 0.90, 1.00, 0.00, 0.00),
            DomainCandidate::new("miss", "other candidate", 1.00, 1.00, 0.00, 0.00),
        ]
        .into_iter()
        .filter(|candidate| candidate.label.contains("needle"));

        let (top, edges) = build_search_delta_edges(candidates, 1);

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].candidate.id, "high");
        assert!(has_edge(
            &edges,
            Node::state(StateNode::Search),
            Node::event(EventNode::CandidateFound)
        ));
        assert!(has_edge(
            &edges,
            Node::state(StateNode::Score),
            Node::event(EventNode::ScoreReady)
        ));
        assert!(has_edge(
            &edges,
            Node::state(StateNode::Select),
            Node::event(EventNode::TopKSelected)
        ));
    }

    #[test]
    fn binary_tlog_appends_typed_records_with_sequences() {
        let path = std::env::temp_dir().join(format!(
            "quantale_semiring_v2_test_{}_{}.tlog",
            std::process::id(),
            1_u64
        ));
        let _ = std::fs::remove_file(&path);

        let mut tlog = TlogWriter::open(&path).unwrap();
        let report = QuantaleCudaReport {
            step: 7,
            best_src: Node::state(StateNode::Goal).encode(),
            best_dst: Node::control(ControlNode::GateInput).encode(),
            best_value: 0.99,
            event_count: 3,
            goal_to_execute: 0.50,
            goal_to_learn: 0.25,
        };
        let decision = DecisionReport {
            step: 8,
            selected_src: Node::state(StateNode::Goal).encode(),
            selected_dst: Node::control(ControlNode::GateInput).encode(),
            first_hop: Node::control(ControlNode::GateInput).encode(),
            selected_value: 0.99,
            halted: 0,
            blocked: 0,
        };
        let receipt = ExecutionReceipt::accepted(0.88, 0.91, 0.93);
        let edges = vec![LatticeEdge::from_nodes(
            Node::state(StateNode::Search),
            Node::event(EventNode::CandidateFound),
            0.90,
        )];

        assert_eq!(tlog.append_cuda_report(&report).unwrap(), 0);
        assert_eq!(tlog.append_decision(&decision).unwrap(), 1);
        assert_eq!(tlog.append_receipt(&receipt).unwrap(), 2);
        assert_eq!(tlog.append_edges("search", &edges).unwrap(), 3);
        tlog.flush().unwrap();

        let records = read_record_meta(&path).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].kind, TlogRecordKind::CudaReport);
        assert_eq!(records[1].kind, TlogRecordKind::Decision);
        assert_eq!(records[2].kind, TlogRecordKind::Receipt);
        assert_eq!(records[3].kind, TlogRecordKind::LatticeEdges);
        assert_eq!(records[0].sequence, 0);
        assert_eq!(records[3].sequence, 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn binary_tlog_reopens_and_continues_sequence() {
        let path = std::env::temp_dir().join(format!(
            "quantale_semiring_v2_test_{}_{}.tlog",
            std::process::id(),
            2_u64
        ));
        let _ = std::fs::remove_file(&path);

        let decision = DecisionReport {
            step: 1,
            selected_src: 0,
            selected_dst: 1,
            first_hop: 1,
            selected_value: 0.5,
            halted: 0,
            blocked: 0,
        };

        {
            let mut tlog = TlogWriter::open(&path).unwrap();
            assert_eq!(tlog.append_decision(&decision).unwrap(), 0);
            tlog.flush().unwrap();
        }
        {
            let mut tlog = TlogWriter::open(&path).unwrap();
            assert_eq!(tlog.append_decision(&decision).unwrap(), 1);
            tlog.flush().unwrap();
        }

        let records = read_record_meta(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sequence, 0);
        assert_eq!(records[1].sequence, 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn system_config_matches_node_registry() {
        let config = SystemConfig::default();

        assert_eq!(config.matrix_dim, NODE_COUNT);
        assert_eq!(config.matrix_len, MATRIX_LEN);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn ingress_queue_drains_available_events_without_blocking() {
        let (server, receiver) = IngressServer::new();

        server
            .push_event(InboundEvent::new("test", "fact", b"payload".to_vec()))
            .unwrap();

        let events = drain_available(&receiver, 8);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "test");
        assert_eq!(events[0].event_name, "fact");
        assert_eq!(events[0].payload, b"payload".to_vec());
        assert!(drain_available(&receiver, 8).is_empty());
    }

    #[test]
    fn process_receipt_converts_to_closed_loop_execution_receipt() {
        let process_receipt = ProcessReceipt {
            node_name: "Control::GateExecution".to_string(),
            exit_code: 0,
            stdout_payload: String::new(),
            stderr_payload: String::new(),
        };
        let receipt = process_receipt.to_execution_receipt();

        assert!(receipt.accepted);
        assert!(receipt.hash_nonzero());
    }

    #[test]
    fn quantale_weight_clamps_and_composes() {
        let left = QuantaleWeight::new(0.5);
        let right = QuantaleWeight::new(0.8);

        assert_eq!(QuantaleWeight::new(2.0).raw(), Q_UNIT);
        assert_eq!(QuantaleWeight::new(f32::NAN).raw(), BOTTOM);
        assert_eq!(left.join(right).raw(), 0.8);
        assert!((left.compose(right).raw() - 0.4).abs() <= f32::EPSILON);
    }
}
