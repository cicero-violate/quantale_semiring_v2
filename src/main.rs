use quantale_semiring_v2::{
    CudaWorld, DomainCandidate, EgressDispatcher, ExternalAction, InboundEvent, IngressServer,
    SystemConfig, TlogWriter, build_candidate_edges, build_receipt_edges,
    data_driven_transition_edges, drain_available, format_quantale_value, node_name,
};

fn main() {
    let config = SystemConfig::default();
    let (ingress, inbound) = IngressServer::new();
    if let Err(error) = ingress.push_event(InboundEvent::new("demo", "candidate", b"seed".to_vec()))
    {
        eprintln!("{error}");
        std::process::exit(1);
    }

    let mut tlog = match TlogWriter::open(&config.tlog_path) {
        Ok(tlog) => tlog,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let transition_edges = data_driven_transition_edges()
        .unwrap_or_else(|_| quantale_semiring_v2::full_transition_edges());

    let mut world = match CudaWorld::from_edges(&transition_edges) {
        Ok(world) => world,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    for tick in 0..4 {
        for event in drain_available(&inbound, 32) {
            let candidate = DomainCandidate::new(
                format!("{}:{}:{tick}", event.source, event.event_name),
                String::from_utf8_lossy(&event.payload).to_string(),
                0.92,
                0.88,
                0.02,
                0.04,
            );
            let (_top, edges) = build_candidate_edges([candidate], 1);
            if !edges.is_empty() {
                if let Err(error) = world.load_edges(&edges) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                if let Err(error) = tlog.append_edges("ingress:candidate", &edges) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            }
        }
        let report = match world.step() {
            Ok(report) => report,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };
        let decision = match world.frontier_step() {
            Ok(decision) => decision,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };

        if let Err(error) = tlog.append_cuda_report(&report) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Err(error) = tlog.append_decision(&decision) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if matches!(
            decision.selected_action(),
            quantale_semiring_v2::QuantaleAction::RunExecutor
        ) {
            let confirmation = EgressDispatcher::dispatch_with_confirmation(ExternalAction::Noop {
                label: format!("tick-{tick}"),
            });
            if let Err(error) = world.join_receipt_edges(confirmation.receipt) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            let receipt_edges = build_receipt_edges(confirmation.receipt);
            if let Err(error) = tlog.append_receipt(&confirmation.receipt) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            if let Err(error) = tlog.append_edges("egress:receipt", &receipt_edges) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }

        println!(
            "step={} best=({}->{}) value={} events={} Goal->Execute={} Goal->Learn={} projection=({}->{}) witness_first_hop={} dvalue={} selected_action={:?} halted={} blocked={}",
            report.step,
            node_name(report.best_src),
            node_name(report.best_dst),
            format_quantale_value(report.best_value),
            report.event_count,
            format_quantale_value(report.goal_to_execute),
            format_quantale_value(report.goal_to_learn),
            node_name(decision.selected_src),
            node_name(decision.selected_dst),
            node_name(decision.first_hop),
            format_quantale_value(decision.selected_value),
            decision.selected_action(),
            decision.halted,
            decision.blocked,
        );
    }

    if let Err(error) = tlog.flush() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
