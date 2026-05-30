use serde_json::json;

use quantale_semiring_v2::{
    CudaWorld, DomainCandidate, InboundEvent, IngressServer, Node, SystemConfig, TlogWriter,
    UniversalExecutor, build_candidate_edges, build_receipt_edges, compile_llm_plan,
    drain_available, format_quantale_value, node_name,
};

fn main() {
    let config = SystemConfig::default();

    let (ingress, inbound) = IngressServer::new();
    if let Err(error) =
        ingress.push_event(InboundEvent::new("demo", "candidate", b"seed".to_vec()))
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

    let executor = UniversalExecutor::from_config(&config);

    let mut world = match CudaWorld::new() {
        Ok(world) => world,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    // Seed ingress candidates into the GPU matrix before the agent loop starts.
    for event in drain_available(&inbound, 32) {
        let candidate = DomainCandidate::new(
            format!("{}:{}", event.source, event.event_name),
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

    println!("Starting Quantale Neuro-Symbolic Agent Loop...");

    // The dynamic payload fed into the executor at each step.
    // After a successful LLM call the stdout content replaces this.
    let mut current_payload = json!({ "context": "Optimize memory allocations across threads" });

    for _ in 0..config.max_ticks {
        // Fused CUDA tick: compute quantale closure and project the frontier.
        let (report, decision) = match world.tick() {
            Ok(result) => result,
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

        println!(
            "step={} best=({}->{}) value={} Goal->Execute={} Goal->Learn={} \
             projection=({}->{}) first_hop={} dvalue={} action={:?} halted={} blocked={}",
            report.step,
            node_name(report.best_src),
            node_name(report.best_dst),
            format_quantale_value(report.best_value),
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

        if decision.halted != 0 {
            println!("System execution chain reached terminal halt safely.");
            break;
        }

        if decision.blocked != 0 {
            // GPU could not advance the frontier; no operator to run this tick.
            continue;
        }

        // Decode the frontier node to resolve its operator name.
        let Some(active_node) = Node::decode(decision.selected_dst) else {
            eprintln!("Invalid selected_dst index: {}", decision.selected_dst);
            break;
        };
        let active_node_name = active_node.name();

        println!("[STEP] Frontier advanced to node: {active_node_name}");

        // Run the operator mapped to this node (may be an LLM call, a patch, tests, etc.).
        let process_receipt =
            executor.execute_abstract_node_blocking(active_node_name, &current_payload);
        let feedback_weight = process_receipt.calculate_algebraic_feedback();

        println!(
            "[STEP] operator={} exit={} weight={} stdout_len={}",
            active_node_name,
            process_receipt.exit_code,
            feedback_weight.raw(),
            process_receipt.stdout_payload.len(),
        );
        if !process_receipt.stderr_payload.is_empty() {
            eprintln!("[STEP] stderr: {}", process_receipt.stderr_payload.trim());
        }

        // Flash the feedback weight onto the selected edge in GPU VRAM.
        if let Err(error) =
            world.inject_dynamic_weight(decision.selected_src, decision.selected_dst, feedback_weight.raw())
        {
            eprintln!("{error}");
            std::process::exit(1);
        }

        // Also apply the structured receipt routing edges (ReceiptAccepted/Rejected paths).
        let execution_receipt = process_receipt.to_execution_receipt();
        if let Err(error) = world.join_receipt_edges(execution_receipt.clone()) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        let receipt_edges = build_receipt_edges(execution_receipt.clone());
        if let Err(error) = tlog.append_receipt(&execution_receipt) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Err(error) = tlog.append_edges("egress:receipt", &receipt_edges) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        // Only compile a JSON edge plan if the operator explicitly declares output_mode=plan.
        // Other operators (cargo test, patch, etc.) produce non-plan stdout that should
        // never be parsed as an edge array or penalised for not being one.
        if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
            if executor.output_mode(active_node_name) == Some("plan") {
                match compile_llm_plan(&process_receipt.stdout_payload) {
                    Ok(plan_edges) if !plan_edges.is_empty() => {
                        println!("[ALGEBRA] LLM plan: {} edge(s) → VRAM", plan_edges.len());
                        if let Err(error) = world.load_edges(&plan_edges) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                        if let Err(error) = tlog.append_edges("plan:llm", &plan_edges) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                    }
                    Ok(_) => {}
                    Err(reason) => {
                        println!("[WARN] LLM plan invalid ({reason}); penalising edge");
                        if let Err(error) = world.inject_dynamic_weight(
                            decision.selected_src,
                            decision.selected_dst,
                            0.0,
                        ) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            // Always carry stdout forward as context for the next prompt.
            current_payload = json!({ "context": process_receipt.stdout_payload });
        }

        // Append fused step record to the transaction log.
        if let Err(error) = tlog.log_step(&process_receipt, &decision) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }

    if let Err(error) = tlog.flush() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
