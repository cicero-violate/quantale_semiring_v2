use serde_json::json;

use quantale_semiring_v2::{
    ExecutionOutcome, LAYER_CONFIDENCE, Node, ProjectionBias, SystemConfig, TensorQuantaleWorld,
    TlogWriter, UniversalExecutor, build_tensor_receipt_edges, compile_tensor_plan,
    format_quantale_value, full_tensor_transition_edges, node_name,
};

fn main() {
    let config = SystemConfig::default();
    let projection_bias = ProjectionBias::default();

    let mut tlog = match TlogWriter::open(&config.tlog_path) {
        Ok(tlog) => tlog,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let executor = UniversalExecutor::from_config(&config);
    let tensor_edges = full_tensor_transition_edges();
    let mut world = match TensorQuantaleWorld::from_tensor_edges(&tensor_edges) {
        Ok(world) => world,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = tlog.append_tensor_edges("topology:tensor", &tensor_edges) {
        eprintln!("{error}");
        std::process::exit(1);
    }

    println!("Starting Tensor Quantale Neuro-Symbolic Agent Loop...");

    let mut current_payload = json!({ "context": "Optimize memory allocations across threads" });

    for _ in 0..config.max_ticks {
        let decision = match world.tick(projection_bias) {
            Ok(decision) => decision,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };

        if let Err(error) = tlog.append_decision(&decision) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        println!(
            "step={} projection=({}->{}) first_hop={} score={} action={:?} halted={} blocked={}",
            decision.step,
            node_name(decision.selected_src),
            node_name(decision.selected_dst),
            node_name(decision.first_hop),
            format_quantale_value(decision.selected_value),
            decision.selected_action(),
            decision.halted,
            decision.blocked,
        );

        if decision.halted != 0 {
            println!("Tensor execution chain reached terminal halt safely.");
            break;
        }

        if decision.blocked != 0 {
            continue;
        }

        let Some(active_node) = Node::decode(decision.first_hop) else {
            eprintln!("Invalid first_hop index: {}", decision.first_hop);
            break;
        };
        let active_node_name = active_node.name();

        println!("[STEP] Tensor frontier advanced to node: {active_node_name}");

        let process_receipt =
            executor.execute_abstract_node_blocking(active_node_name, &current_payload);
        let outcome = ExecutionOutcome::from(&process_receipt);

        println!(
            "[STEP] operator={} exit={} outcome={:?} stdout_len={}",
            active_node_name,
            process_receipt.exit_code,
            outcome,
            process_receipt.stdout_payload.len(),
        );
        if !process_receipt.stderr_payload.is_empty() {
            eprintln!("[STEP] stderr: {}", process_receipt.stderr_payload.trim());
        }

        if let Err(error) =
            world.update_lattice_edge(decision.selected_src, decision.first_hop, outcome)
        {
            eprintln!("{error}");
            std::process::exit(1);
        }

        let execution_receipt = process_receipt.to_execution_receipt();
        let receipt_edges = build_tensor_receipt_edges(execution_receipt.clone());
        if let Err(error) = world.embed_tensor_edges(&receipt_edges) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Err(error) = tlog.append_receipt(&execution_receipt) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Err(error) = tlog.append_tensor_edges("egress:receipt", &receipt_edges) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
            if executor.output_mode(active_node_name) == Some("tensor_plan") {
                match compile_tensor_plan(&process_receipt.stdout_payload) {
                    Ok(plan_edges) if !plan_edges.is_empty() => {
                        println!(
                            "[ALGEBRA] Tensor LLM plan: {} edge(s) → VRAM",
                            plan_edges.len()
                        );
                        if let Err(error) = world.embed_tensor_edges(&plan_edges) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                        if let Err(error) = tlog.append_tensor_edges("plan:tensor_llm", &plan_edges)
                        {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                    }
                    Ok(_) => {}
                    Err(reason) => {
                        println!(
                            "[WARN] Tensor LLM plan invalid ({reason}); dampening selected edge"
                        );
                        if let Err(error) = world.update_lattice_edge(
                            decision.selected_src,
                            decision.first_hop,
                            ExecutionOutcome::Failure,
                        ) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            current_payload = json!({ "context": process_receipt.stdout_payload });
        }

        if let Err(error) = world.decay(0.995) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if let Err(error) = tlog.log_step(&process_receipt, &decision) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        let _ = world.reconstruct_projected_tensor_path(LAYER_CONFIDENCE);
    }

    if let Err(error) = tlog.flush() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
