use std::collections::BTreeSet;

use serde_json::json;

use quantale_semiring_v2::{
    ExecutionOutcome, ExplorationConfig, ExplorationEngine, GraphTopology, LAYER_CONFIDENCE, Node,
    ProjectionBias, SystemConfig, TensorQuantaleWorld, TlogWriter, UniversalExecutor, action_label,
    compile_pattern, compile_tensor_plan, dispatch_decision_batch_blocking, format_quantale_value,
    full_tensor_transition_edges, load_default_patterns, load_learned_tensor_edges,
    project_ready_batch_plan, runtime_check, topology_check,
};

fn main() {
    // --check-topology: validate topology.json and exit without running the loop.
    if std::env::args().any(|a| a == "--check-topology") {
        let topology = match GraphTopology::default_asset() {
            Ok(t) => t,
            Err(error) => {
                eprintln!("[topology] parse failed: {error}");
                std::process::exit(1);
            }
        };
        let violations = topology_check::check(&topology);
        if violations.is_empty() {
            println!("topology OK ({} nodes, {} transitions)",
                topology.nodes.len(), topology.transitions.len());
            std::process::exit(0);
        }
        eprintln!("{}", topology_check::format_violations(&violations));
        eprintln!("{} violation(s) found", violations.len());
        std::process::exit(1);
    }

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
    let mut tensor_edges = full_tensor_transition_edges();
    if let Err(error) = tlog.append_tensor_edges("topology:tensor", &tensor_edges) {
        eprintln!("{error}");
        std::process::exit(1);
    }

    let topology_asset = match GraphTopology::default_asset() {
        Ok(topology) => topology,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let violations = topology_check::check(&topology_asset);
    if !violations.is_empty() {
        eprintln!("{}", topology_check::format_violations(&violations));
        eprintln!("topology has {} violation(s); refusing to start", violations.len());
        std::process::exit(1);
    }
    let topology = match topology_asset.compile() {
        Ok(topology) => topology,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let learned_edges = match load_learned_tensor_edges(
        &config.learned_edges_path,
        &topology.registry,
        &topology.tensor_edges,
    ) {
        Ok(edges) => edges,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    if !learned_edges.is_empty() {
        if let Err(error) = tlog.append_tensor_edges("state:learned", &learned_edges) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        tensor_edges.extend(learned_edges);
    }
    let patterns = match load_default_patterns() {
        Ok(patterns) => patterns,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let mut compiled_patterns = Vec::new();
    for pattern in &patterns.patterns {
        let compiled = match compile_pattern(pattern, &topology, &config.operator_registry) {
            Ok(compiled) => compiled,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };
        if let Err(error) = tlog.append_tensor_edges("pattern:cka", &compiled.edges) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        tensor_edges.extend(compiled.edges.clone());
        compiled_patterns.push(compiled);
    }

    let mut world = match TensorQuantaleWorld::from_tensor_edges(&tensor_edges) {
        Ok(world) => world,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let exploration_config = match ExplorationConfig::default_asset() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let mut exploration_engine = match ExplorationEngine::new(
        exploration_config,
        &topology_asset,
        config.operator_registry.clone(),
    ) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    println!("Starting Tensor Quantale Neuro-Symbolic Agent Loop...");
    if config.max_ticks == 0 {
        println!("Continuous mode: running until halt or signal.");
    }

    let sleep_dur = (config.tick_sleep_ms > 0)
        .then(|| std::time::Duration::from_millis(config.tick_sleep_ms));
    let mut current_payload = json!({ "context": "market_analysis_loop" });
    let mut tick: usize = 0;
    let mut consecutive_blocks: usize = 0;

    loop {
        if config.max_ticks > 0 && tick >= config.max_ticks {
            break;
        }
        tick += 1;
        if let Err(error) = world.close() {
            eprintln!("{error}");
            std::process::exit(1);
        }

        let exploration_candidates = match world.expand_exploration(&mut exploration_engine) {
            Ok(candidates) => candidates,
            Err(error) => {
                eprintln!("[WARN] exploration unavailable; falling back to CKA/frontier: {error}");
                Vec::new()
            }
        };
        if let Err(error) = tlog.append_exploration_seed(&json!({
            "token_count": exploration_engine.tokens().len(),
            "strategy_count": exploration_engine.config().strategies.len(),
        })) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Err(error) = tlog.append_exploration_topk(&json!({
            "candidate_count": exploration_candidates.len(),
            "candidates": exploration_candidates,
        })) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        let exploration_winner = exploration_engine.best_commit_candidate();
        if let Some(candidate) = exploration_winner {
            let commit_record = exploration_engine.commit_record(candidate);
            if let Err(error) = tlog.append_exploration_commit(&commit_record) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            let decision = match world.commit_exploration_candidate(&candidate) {
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
            exploration_engine.mark_candidate_committed(&candidate);
            println!(
                "exploration projection=({}->{}) first_hop={} score={} path={:?}",
                node_name(decision.selected_src, &topology.registry),
                node_name(decision.selected_dst, &topology.registry),
                node_name(decision.first_hop, &topology.registry),
                format_quantale_value(decision.selected_value),
                commit_record.path,
            );
            if decision.blocked != 0 {
                continue;
            }
            let Some(active_node) = Node::decode(decision.first_hop, &topology.registry) else {
                eprintln!(
                    "Invalid exploration first_hop index: {}",
                    decision.first_hop
                );
                break;
            };
            let Some(active_node_name) = active_node.name(&topology.registry) else {
                eprintln!(
                    "Invalid exploration first_hop index: {}",
                    decision.first_hop
                );
                break;
            };
            let process_receipt =
                executor.execute_abstract_node_blocking(active_node_name, &current_payload);
            let outcome = ExecutionOutcome::from(&process_receipt);
            if let Err(error) =
                world.update_lattice_edge(decision.selected_src, decision.first_hop, outcome)
            {
                eprintln!("{error}");
                std::process::exit(1);
            }
            exploration_engine.update_receipt_prior(decision.first_hop, &process_receipt);
            if let Err(error) = tlog.append_exploration_receipt(&json!({
                "node": active_node_name,
                "exit_code": process_receipt.exit_code,
                "prior": exploration_engine.receipt_prior_for(decision.first_hop),
            })) {
                eprintln!("{error}");
                std::process::exit(1);
            }

            if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
                if executor.output_mode(active_node_name) == Some("tensor_plan") {
                    match compile_tensor_plan(&process_receipt.stdout_payload) {
                        Ok(plan_edges) => {
                            let plan_edges =
                                filter_static_topology_edges(plan_edges, &topology.tensor_edges);
                            if !plan_edges.is_empty() {
                                if let Err(error) = world.embed_tensor_edges(&plan_edges) {
                                    eprintln!("{error}");
                                    std::process::exit(1);
                                }
                                if let Err(error) =
                                    tlog.append_tensor_edges("exploration:plan_tensor", &plan_edges)
                                {
                                    eprintln!("{error}");
                                    std::process::exit(1);
                                }
                            }
                        }
                        Err(reason) => {
                            println!(
                                "[WARN] Exploration tensor plan invalid ({reason}); dampening selected edge"
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
            continue;
        }

        match project_ready_batch_plan(
            &mut world,
            &compiled_patterns,
            projection_bias,
            &config.operator_registry,
        ) {
            Ok(Some(batch_plan)) => {
                if let Err(error) = tlog.append_batch_plan("scheduler:cka_parallel", &batch_plan) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }

                for batch in &batch_plan.batches {
                    if let Err(error) = world.commit_decision_batch(&batch.decisions) {
                        eprintln!("{error}");
                        std::process::exit(1);
                    }

                    for decision in &batch.decisions {
                        if let Err(error) = tlog.append_decision(decision) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                        println!(
                            "batch_step={} projection=({}->{}) first_hop={} score={} action={:?}",
                            decision.step,
                            node_name(decision.selected_src, &topology.registry),
                            node_name(decision.selected_dst, &topology.registry),
                            node_name(decision.first_hop, &topology.registry),
                            format_quantale_value(decision.selected_value),
                            action_label(decision.first_hop, &topology.registry),
                        );
                    }

                    let scheduled_receipts =
                        dispatch_decision_batch_blocking(&executor, batch, &current_payload);
                    let mut batch_stdout = Vec::new();

                    for scheduled in scheduled_receipts {
                        let active_node_name = scheduled.receipt.node_name.clone();
                        let outcome = ExecutionOutcome::from(&scheduled.receipt);
                        println!(
                            "[BATCH] operator={} exit={} outcome={:?} stdout_len={}",
                            active_node_name,
                            scheduled.receipt.exit_code,
                            outcome,
                            scheduled.receipt.stdout_payload.len(),
                        );
                        if !scheduled.receipt.stderr_payload.is_empty() {
                            eprintln!(
                                "[BATCH] stderr: {}",
                                scheduled.receipt.stderr_payload.trim()
                            );
                        }

                        if let Err(error) = world.update_lattice_edge(
                            scheduled.decision.selected_src,
                            scheduled.decision.first_hop,
                            outcome,
                        ) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }

                        exploration_engine
                            .update_receipt_prior(scheduled.decision.first_hop, &scheduled.receipt);
                        if let Err(error) = tlog.append_exploration_receipt(&json!({
                            "node": active_node_name,
                            "exit_code": scheduled.receipt.exit_code,
                            "prior": exploration_engine.receipt_prior_for(scheduled.decision.first_hop),
                        })) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }

                        if scheduled.receipt.exit_code == 0
                            && !scheduled.receipt.stdout_payload.is_empty()
                        {
                            if executor.output_mode(&active_node_name) == Some("tensor_plan") {
                                match compile_tensor_plan(&scheduled.receipt.stdout_payload) {
                                    Ok(plan_edges) => {
                                        let plan_edges = filter_static_topology_edges(
                                            plan_edges,
                                            &topology.tensor_edges,
                                        );
                                        if !plan_edges.is_empty() {
                                            println!(
                                                "[ALGEBRA] Tensor batch plan: {} edge(s) → VRAM",
                                                plan_edges.len()
                                            );
                                            if let Err(error) =
                                                world.embed_tensor_edges(&plan_edges)
                                            {
                                                eprintln!("{error}");
                                                std::process::exit(1);
                                            }
                                            if let Err(error) = tlog.append_tensor_edges(
                                                "plan:tensor_batch",
                                                &plan_edges,
                                            ) {
                                                eprintln!("{error}");
                                                std::process::exit(1);
                                            }
                                        }
                                    }
                                    Err(reason) => {
                                        println!(
                                            "[WARN] Tensor batch plan invalid ({reason}); dampening selected edge"
                                        );
                                        if let Err(error) = world.update_lattice_edge(
                                            scheduled.decision.selected_src,
                                            scheduled.decision.first_hop,
                                            ExecutionOutcome::Failure,
                                        ) {
                                            eprintln!("{error}");
                                            std::process::exit(1);
                                        }
                                    }
                                }
                            }
                            batch_stdout.push(json!({
                                "node": active_node_name,
                                "stdout": scheduled.receipt.stdout_payload,
                            }));
                        }

                        if let Err(error) = tlog.log_step(&scheduled.receipt, &scheduled.decision) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                    }

                    if !batch_stdout.is_empty() {
                        current_payload = json!({ "context": batch_stdout });
                    }
                }

                if let Err(error) = world.decay(0.995) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                let _ = world.reconstruct_projected_tensor_path(LAYER_CONFIDENCE);
                continue;
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }

        let decision = match world.frontier_step(projection_bias) {
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
            node_name(decision.selected_src, &topology.registry),
            node_name(decision.selected_dst, &topology.registry),
            node_name(decision.first_hop, &topology.registry),
            format_quantale_value(decision.selected_value),
            action_label(decision.first_hop, &topology.registry),
            decision.halted,
            decision.blocked,
        );

        if decision.halted != 0 {
            if config.max_ticks == 0 {
                // Continuous mode: dampen the halt edge and restart the trading cycle.
                let _ = world.update_lattice_edge(
                    decision.selected_src,
                    decision.first_hop,
                    ExecutionOutcome::Failure,
                );
                let _ = world.decay(0.97);
                current_payload = json!({ "context": "market_analysis_loop" });
                if let Some(dur) = sleep_dur {
                    std::thread::sleep(dur);
                }
                continue;
            }
            println!("Tensor execution chain reached terminal halt safely.");
            break;
        }

        if decision.blocked != 0 {
            consecutive_blocks += 1;
            if consecutive_blocks >= 3 {
                // Hard reset: decay alone does not restore a valid frontier
                // because active[] tracks the current position and is not
                // affected by decay or embed.  world.reset() clears active[],
                // consumed[], and the tensor to a known-good initial state.
                // Re-embedding the topology edges then lifts the world above ⊥.
                // close() runs here so the invariant-17 project check below is
                // a real validation, not a pre-closure false alarm.
                eprintln!("[WARN] {consecutive_blocks} consecutive blocked steps; hard reset.");
                if let Err(error) = world.reset() {
                    eprintln!("[WARN] hard reset world.reset() failed: {error}");
                }
                if let Err(error) = world.embed_tensor_edges(&tensor_edges) {
                    eprintln!("[WARN] hard reset embed failed: {error}");
                }
                if let Err(error) = world.close() {
                    eprintln!("[WARN] hard reset close failed: {error}");
                }
                current_payload = json!({ "context": "market_analysis_loop" });
                consecutive_blocks = 0;
                let reset_dur = sleep_dur.unwrap_or(std::time::Duration::from_millis(200));
                std::thread::sleep(reset_dur);
                // Invariant 17: verify reset restored a valid frontier.
                // Uses project (read-only) so active[] is not advanced.
                if let Ok(post_reset) = world.project(projection_bias) {
                    if post_reset.blocked != 0 {
                        eprintln!(
                            "[WARN] hard reset did not restore a valid frontier \
                             (first_hop={}); reset+embed may have failed",
                            post_reset.first_hop
                        );
                    }
                }
            }
            continue;
        }
        // Invariant 20: refuse to advance the executor on a bottom score.
        // Check BEFORE resetting consecutive_blocks so that score=⊥ steps
        // accumulate toward the hard-reset threshold alongside blocked steps.
        if !runtime_check::decision_is_safe(&decision) {
            eprintln!(
                "[WARN] invariant 20: score=⊥ with blocked=0 (first_hop={}); \
                 skipping executor call",
                decision.first_hop
            );
            consecutive_blocks += 1;
            continue;
        }

        consecutive_blocks = 0;

        let Some(active_node) = Node::decode(decision.first_hop, &topology.registry) else {
            eprintln!("Invalid first_hop index: {}", decision.first_hop);
            break;
        };
        let Some(active_node_name) = active_node.name(&topology.registry) else {
            eprintln!("Invalid first_hop index: {}", decision.first_hop);
            break;
        };

        // Invariants 18 + 19: validate decision report before running executor.
        for v in runtime_check::check_decision(&decision, active_node_name) {
            eprintln!("[runtime_check] {v}");
        }

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
        // Invariant 24: Control::Block must result in blocked or halted state.
        if active_node_name.contains("Control::Block") && outcome == ExecutionOutcome::Success {
            for v in runtime_check::check_decision(&decision, active_node_name) {
                if v.kind == runtime_check::RuntimeViolationKind::BlockNodeNotBlocked {
                    eprintln!("[runtime_check] {v}");
                }
            }
        }

        if let Err(error) =
            world.update_lattice_edge(decision.selected_src, decision.first_hop, outcome)
        {
            eprintln!("{error}");
            std::process::exit(1);
        }

        exploration_engine.update_receipt_prior(decision.first_hop, &process_receipt);
        if let Err(error) = tlog.append_exploration_receipt(&json!({
            "node": active_node_name,
            "exit_code": process_receipt.exit_code,
            "prior": exploration_engine.receipt_prior_for(decision.first_hop),
        })) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
            if executor.output_mode(active_node_name) == Some("tensor_plan") {
                match compile_tensor_plan(&process_receipt.stdout_payload) {
                    Ok(plan_edges) => {
                        let plan_edges =
                            filter_static_topology_edges(plan_edges, &topology.tensor_edges);
                        if !plan_edges.is_empty() {
                            println!(
                                "[ALGEBRA] Tensor LLM plan: {} edge(s) → VRAM",
                                plan_edges.len()
                            );
                            if let Err(error) = world.embed_tensor_edges(&plan_edges) {
                                eprintln!("{error}");
                                std::process::exit(1);
                            }
                            if let Err(error) =
                                tlog.append_tensor_edges("plan:tensor_llm", &plan_edges)
                            {
                                eprintln!("{error}");
                                std::process::exit(1);
                            }
                        }
                    }
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
        if let Some(dur) = sleep_dur {
            std::thread::sleep(dur);
        }
    }

    if let Err(error) = tlog.flush() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn node_name(node_id: i32, registry: &quantale_semiring_v2::NodeRegistry) -> String {
    Node::decode(node_id, registry)
        .and_then(|node| node.name(registry))
        .map(str::to_string)
        .unwrap_or_else(|| format!("Unknown({node_id})"))
}

fn filter_static_topology_edges(
    edges: Vec<quantale_semiring_v2::TensorEdge>,
    topology_edges: &[quantale_semiring_v2::TensorEdge],
) -> Vec<quantale_semiring_v2::TensorEdge> {
    let allowed: BTreeSet<(i32, i32)> = topology_edges
        .iter()
        .map(|edge| (edge.src, edge.dst))
        .collect();
    edges
        .into_iter()
        .filter(|edge| allowed.contains(&(edge.src, edge.dst)))
        .collect()
}
