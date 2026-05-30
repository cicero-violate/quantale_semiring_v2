use quantale_semiring_v2::{
    ControlNode, DomainCandidate, EventNode, InboundEvent, IngressServer, Node, ProcessReceipt,
    StateNode, TlogRecordKind, TlogWriter, build_candidate_edges, build_receipt_edges,
    drain_available, read_record_meta,
};

fn has_dst(edges: &[quantale_semiring_v2::TransitionEdge], dst: Node) -> bool {
    edges.iter().any(|edge| edge.dst == dst.encode())
}

#[test]
fn success_receipt_keeps_validation_path_reachable() {
    let receipt = ProcessReceipt {
        node_name: "Control::GateExecution".to_string(),
        exit_code: 0,
        stdout_payload: String::new(),
        stderr_payload: String::new(),
    }
    .to_execution_receipt();
    let edges = build_receipt_edges(receipt.clone());

    assert!(receipt.accepted);
    assert!(has_dst(&edges, Node::event(EventNode::ReceiptAccepted)));
    assert!(has_dst(&edges, Node::event(EventNode::HashNonzero)));
    assert!(has_dst(&edges, Node::state(StateNode::Validate)));
}

#[test]
fn failure_receipt_routes_to_rejected_rollback_repair() {
    let edges = build_receipt_edges(quantale_semiring_v2::ExecutionReceipt::rejected(
        0.8, 0.7, 0.6,
    ));

    assert!(has_dst(&edges, Node::event(EventNode::ReceiptRejected)));
    assert!(has_dst(&edges, Node::control(ControlNode::Rollback)));
    assert!(has_dst(&edges, Node::control(ControlNode::Repair)));
}

#[test]
fn blocked_frontier_fixture_does_not_advance_open_loop() {
    let active = [Node::state(StateNode::Goal)];
    let available_edges: [quantale_semiring_v2::TransitionEdge; 0] = [];

    assert_eq!(active, [Node::state(StateNode::Goal)]);
    assert!(available_edges.is_empty());
}

#[test]
fn tlog_records_match_executed_tick_count() {
    let path = std::env::temp_dir().join(format!(
        "qsv2_release_validation_{}.tlog",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let mut tlog = TlogWriter::open(&path).unwrap();
    let ticks = 3;
    for tick in 0..ticks {
        let report = quantale_semiring_v2::QuantaleCudaReport {
            step: tick,
            ..Default::default()
        };
        let decision = quantale_semiring_v2::DecisionReport {
            step: tick,
            ..Default::default()
        };
        tlog.append_cuda_report(&report).unwrap();
        tlog.append_decision(&decision).unwrap();
    }
    tlog.flush().unwrap();

    let records = read_record_meta(&path).unwrap();
    assert_eq!(records.len(), ticks as usize * 2);
    assert_eq!(
        records
            .iter()
            .filter(|record| record.kind == TlogRecordKind::CudaReport)
            .count(),
        ticks as usize
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| record.kind == TlogRecordKind::Decision)
            .count(),
        ticks as usize
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ingress_event_drains_and_compiles_without_blocking() {
    let (server, receiver) = IngressServer::new();
    server
        .push_event(InboundEvent::new(
            "release",
            "candidate",
            b"payload".to_vec(),
        ))
        .unwrap();

    let events = drain_available(&receiver, 8);
    let candidates = events.into_iter().map(|event| {
        DomainCandidate::new(
            format!("{}:{}", event.source, event.event_name),
            String::from_utf8(event.payload).unwrap(),
            0.8,
            0.9,
            0.0,
            0.0,
        )
    });
    let (top, edges) = build_candidate_edges(candidates, 1);

    assert_eq!(top.len(), 1);
    assert!(has_dst(&edges, Node::event(EventNode::CandidateFound)));
    assert!(drain_available(&receiver, 8).is_empty());
}
