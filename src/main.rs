use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::SystemTime;

use serde_json::{Value, json};

use quantale_semiring_v2::{
    CompiledCkaPattern, ContractContext, ContractViolation, ExecutionOutcome, ExplorationConfig,
    ExplorationEngine, GraphTopology, LAYER_CONFIDENCE, LearningPolicy, Node, NodeContracts,
    ProcessReceipt, ProjectionBias, SystemConfig, TensorQuantaleWorld, TlogWriter,
    TopologyInvariants, TopologyRuntime, UniversalExecutor, ViolationKind, action_label, check,
    check_with_operators, compile_pattern, compile_tensor_plan, dispatch_decision_batch_blocking,
    format_quantale_value, format_violations, load_default_patterns, load_learned_tensor_edges,
    project_ready_batch_plan, runtime_check,
};

use topology_core::build_overlay_assets;

#[derive(Clone, Debug, PartialEq, Eq)]
struct AssetFingerprint {
    entries: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
}

struct RuntimeEpoch {
    id: usize,
    fingerprint: AssetFingerprint,
    topology: TopologyRuntime,
    executor: UniversalExecutor,
    contracts: NodeContracts,
    /// Accumulates static topology edges plus every LLM-proposed edge so that
    /// hard reset re-embeds the full learned set, not just the static baseline.
    accumulated_edges: Vec<quantale_semiring_v2::TensorEdge>,
    compiled_patterns: Vec<CompiledCkaPattern>,
    world: TensorQuantaleWorld,
    exploration_engine: ExplorationEngine,
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("topology")
        && args.get(2).map(String::as_str) == Some("build-overlay")
    {
        match build_overlay_assets(".") {
            Ok(()) => {
                println!(
                    "wrote assets/topology.generated.json and assets/operators.generated.json"
                );
                std::process::exit(0);
            }
            Err(error) => {
                eprintln!("[topology] build-overlay failed: {error}");
                std::process::exit(1);
            }
        }
    }

    // --check-topology: validate topology.json and exit without running the loop.
    if args.iter().any(|a| a == "--check-topology") {
        let topology = match GraphTopology::default_asset() {
            Ok(t) => t,
            Err(error) => {
                eprintln!("[topology] parse failed: {error}");
                std::process::exit(1);
            }
        };
        let inv = TopologyInvariants::default_asset();
        let violations = check(&topology, &inv);
        let (warnings, fatal): (Vec<_>, Vec<_>) = violations
            .into_iter()
            .partition(|v| v.kind == ViolationKind::ConsumedBlockPoint);
        for v in &warnings {
            eprintln!("[topology] [WARN] {v}");
        }
        if fatal.is_empty() {
            println!(
                "topology OK ({} nodes, {} transitions, {} known warnings)",
                topology.nodes.len(),
                topology.transitions.len(),
                warnings.len()
            );
            std::process::exit(0);
        }
        eprintln!("{}", format_violations(&fatal));
        eprintln!("{} violation(s) found", fatal.len());
        std::process::exit(1);
    }

    let mut config = SystemConfig::default();
    let projection_bias = ProjectionBias::default();
    let learning_policy = LearningPolicy::default_asset();

    let mut tlog = match TlogWriter::open(&config.tlog_path) {
        Ok(tlog) => tlog,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let mut epoch = match build_runtime_epoch(0, &mut config, &learning_policy, &mut tlog) {
        Ok(epoch) => epoch,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    println!("Starting Tensor Quantale Neuro-Symbolic Agent Loop...");
    if config.max_ticks == 0 {
        println!("Continuous mode: running until halt or signal.");
    }

    let sleep_dur =
        (config.tick_sleep_ms > 0).then(|| std::time::Duration::from_millis(config.tick_sleep_ms));
    let mut current_payload = json!({ "context": "market_analysis_loop" });
    let mut current_payload_origin: Option<String> = None;
    let mut tick: usize = 0;
    let mut consecutive_blocks: usize = 0;

    'ticks: loop {
        if config.max_ticks > 0 && tick >= config.max_ticks {
            break;
        }
        tick += 1;
        if let Some(next_fingerprint) = changed_asset_fingerprint(&epoch.fingerprint) {
            match build_runtime_epoch(epoch.id + 1, &mut config, &learning_policy, &mut tlog) {
                Ok(next_epoch) => {
                    println!(
                        "[TOPOLOGY] reloaded epoch {} -> {} ({} nodes, {} transitions)",
                        epoch.id,
                        next_epoch.id,
                        next_epoch.topology.document.nodes.len(),
                        next_epoch.topology.document.transitions.len()
                    );
                    epoch = next_epoch;
                    current_payload = json!({ "context": "market_analysis_loop" });
                    current_payload_origin = None;
                    consecutive_blocks = 0;
                }
                Err(error) => {
                    eprintln!(
                        "[TOPOLOGY] reload rejected; continuing epoch {}: {error}",
                        epoch.id
                    );
                    epoch.fingerprint = next_fingerprint;
                }
            }
        }

        if let Err(error) = epoch.world.close() {
            eprintln!("{error}");
            std::process::exit(1);
        }

        let exploration_candidates = match epoch
            .world
            .expand_exploration(&mut epoch.exploration_engine)
        {
            Ok(candidates) => candidates,
            Err(error) => {
                eprintln!("[WARN] exploration unavailable; falling back to CKA/frontier: {error}");
                Vec::new()
            }
        };
        if let Err(error) = tlog.append_exploration_seed(&json!({
            "token_count": epoch.exploration_engine.tokens().len(),
            "strategy_count": epoch.exploration_engine.config().strategies.len(),
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
        let exploration_winner = epoch.exploration_engine.best_commit_candidate();
        if let Some(candidate) = exploration_winner {
            let commit_record = epoch.exploration_engine.commit_record(candidate);
            if let Err(error) = tlog.append_exploration_commit(&commit_record) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            let decision = match epoch.world.commit_exploration_candidate(&candidate) {
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
            epoch
                .exploration_engine
                .mark_candidate_committed(&candidate);
            println!(
                "exploration projection=({}->{}) first_hop={} score={} path={:?}",
                node_name(decision.selected_src, epoch.topology.registry()),
                node_name(decision.selected_dst, epoch.topology.registry()),
                node_name(decision.first_hop, epoch.topology.registry()),
                format_quantale_value(decision.selected_value),
                commit_record.path,
            );
            if decision.blocked != 0 {
                continue;
            }
            let Some(active_node) = Node::decode(decision.first_hop, epoch.topology.registry())
            else {
                eprintln!(
                    "Invalid exploration first_hop index: {}",
                    decision.first_hop
                );
                break;
            };
            let Some(active_node_name) = active_node.name(epoch.topology.registry()) else {
                eprintln!(
                    "Invalid exploration first_hop index: {}",
                    decision.first_hop
                );
                break;
            };
            if let Err(violation) = epoch.contracts.validate(
                active_node_name,
                &current_payload,
                current_payload_origin.as_deref(),
                ContractContext::Exploration,
            ) {
                eprintln!("[contract] {violation}; skipping exploration executor call");
                let process_receipt = contract_violation_receipt(active_node_name, &violation);
                epoch.world.queue_lattice_update(
                    decision.selected_src,
                    decision.first_hop,
                    ExecutionOutcome::Failure,
                );
                epoch
                    .exploration_engine
                    .update_receipt_prior(decision.first_hop, &process_receipt);
                if let Err(error) = tlog.append_exploration_receipt(&json!({
                    "node": active_node_name,
                    "exit_code": process_receipt.exit_code,
                    "contract_violation": violation.reason,
                    "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
                })) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                if let Err(error) = epoch.world.drain_lattice_queue() {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                if let Err(error) = tlog.log_step(&process_receipt, &decision) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                consecutive_blocks += 1;
                maybe_hard_reset_after_blocks(
                    &mut consecutive_blocks,
                    &mut epoch.world,
                    &epoch.accumulated_edges,
                    &mut current_payload,
                    std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                    projection_bias,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
                }
                continue;
            }
            let process_receipt = epoch
                .executor
                .execute_abstract_node_blocking(active_node_name, &current_payload);
            let outcome = ExecutionOutcome::from(&process_receipt);
            epoch
                .world
                .queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
            epoch
                .exploration_engine
                .update_receipt_prior(decision.first_hop, &process_receipt);
            if let Err(error) = tlog.append_exploration_receipt(&json!({
                "node": active_node_name,
                "exit_code": process_receipt.exit_code,
                "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
            })) {
                eprintln!("{error}");
                std::process::exit(1);
            }

            if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
                if epoch.executor.output_mode(active_node_name) == Some("tensor_plan") {
                    match compile_tensor_plan(&process_receipt.stdout_payload) {
                        Ok(plan_edges) => {
                            let plan_edges = filter_static_topology_edges(
                                plan_edges,
                                epoch.topology.tensor_edges(),
                            );
                            if !plan_edges.is_empty() {
                                if let Err(error) = epoch.world.embed_tensor_edges(&plan_edges) {
                                    eprintln!("{error}");
                                    std::process::exit(1);
                                }
                                epoch.accumulated_edges.extend(plan_edges.clone());
                                if let Err(error) =
                                    tlog.append_tensor_edges("exploration:plan_tensor", &plan_edges)
                                {
                                    eprintln!("{error}");
                                    std::process::exit(1);
                                }
                            }
                            // Tensor plan edges are now in the GPU world — do not
                            // recycle them as context for the next operator call.
                            // Passing a JSON edge array as context creates nested
                            // JSON on each iteration that call_llm.py must unwrap.
                        }
                        Err(reason) => {
                            println!(
                                "[WARN] Exploration tensor plan invalid ({reason}); dampening selected edge"
                            );
                            epoch.world.queue_lattice_update(
                                decision.selected_src,
                                decision.first_hop,
                                ExecutionOutcome::Failure,
                            );
                        }
                    }
                } else {
                    current_payload = json!({ "context": process_receipt.stdout_payload });
                    current_payload_origin = Some(active_node_name.to_string());
                }
            }
            if let Err(error) = epoch.world.drain_lattice_queue() {
                eprintln!("{error}");
                std::process::exit(1);
            }
            if let Err(error) = tlog.log_step(&process_receipt, &decision) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            // Failed operators (e.g. jit_cuda without --features cuda) must not
            // count as normal progress — treat them as blocked steps so repeated
            // failures eventually trigger a hard reset.
            if process_receipt.exit_code != 0 {
                consecutive_blocks += 1;
                maybe_hard_reset_after_blocks(
                    &mut consecutive_blocks,
                    &mut epoch.world,
                    &epoch.accumulated_edges,
                    &mut current_payload,
                    std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                    projection_bias,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
                }
                continue;
            }
            consecutive_blocks = 0;
            if let Err(error) = epoch.world.decay(config.decay_normal) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            continue;
        }

        match project_ready_batch_plan(
            &mut epoch.world,
            &epoch.compiled_patterns,
            projection_bias,
            &config.operator_registry,
        ) {
            Ok(Some(batch_plan)) => {
                if let Err(error) = tlog.append_batch_plan("scheduler:cka_parallel", &batch_plan) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }

                let mut batch_failure_count = 0usize;
                let mut plan_stdout = Vec::new();
                for batch in &batch_plan.batches {
                    if let Err(error) = epoch.world.commit_decision_batch(&batch.decisions) {
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
                            node_name(decision.selected_src, epoch.topology.registry()),
                            node_name(decision.selected_dst, epoch.topology.registry()),
                            node_name(decision.first_hop, epoch.topology.registry()),
                            format_quantale_value(decision.selected_value),
                            action_label(decision.first_hop, epoch.topology.registry()),
                        );
                    }

                    let scheduled_receipts =
                        dispatch_decision_batch_blocking(&epoch.executor, batch, &current_payload);
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

                        epoch.world.queue_lattice_update(
                            scheduled.decision.selected_src,
                            scheduled.decision.first_hop,
                            outcome,
                        );

                        epoch
                            .exploration_engine
                            .update_receipt_prior(scheduled.decision.first_hop, &scheduled.receipt);
                        if let Err(error) = tlog.append_exploration_receipt(&json!({
                            "node": active_node_name,
                            "exit_code": scheduled.receipt.exit_code,
                            "prior": epoch.exploration_engine.receipt_prior_for(scheduled.decision.first_hop),
                        })) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }

                        if scheduled.receipt.exit_code == 0
                            && !scheduled.receipt.stdout_payload.is_empty()
                        {
                            if epoch.executor.output_mode(&active_node_name) == Some("tensor_plan")
                            {
                                match compile_tensor_plan(&scheduled.receipt.stdout_payload) {
                                    Ok(plan_edges) => {
                                        let plan_edges = filter_static_topology_edges(
                                            plan_edges,
                                            epoch.topology.tensor_edges(),
                                        );
                                        if !plan_edges.is_empty() {
                                            println!(
                                                "[ALGEBRA] Tensor batch plan: {} edge(s) → VRAM",
                                                plan_edges.len()
                                            );
                                            if let Err(error) =
                                                epoch.world.embed_tensor_edges(&plan_edges)
                                            {
                                                eprintln!("{error}");
                                                std::process::exit(1);
                                            }
                                            epoch.accumulated_edges.extend(plan_edges.clone());
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
                                        epoch.world.queue_lattice_update(
                                            scheduled.decision.selected_src,
                                            scheduled.decision.first_hop,
                                            ExecutionOutcome::Failure,
                                        );
                                    }
                                }
                            }
                            // Only pass non-tensor-plan stdout forward as
                            // context — tensor plan edges are already in the
                            // GPU world and would create nested JSON if recycled.
                            if epoch.executor.output_mode(&active_node_name) != Some("tensor_plan")
                            {
                                batch_stdout.push(json!({
                                    "node": active_node_name,
                                    "stdout": scheduled.receipt.stdout_payload,
                                }));
                            }
                        }

                        if let Err(error) = tlog.log_step(&scheduled.receipt, &scheduled.decision) {
                            eprintln!("{error}");
                            std::process::exit(1);
                        }
                        if scheduled.receipt.exit_code != 0 {
                            batch_failure_count += 1;
                        }
                    }

                    if let Err(error) = epoch.world.drain_lattice_queue() {
                        eprintln!("{error}");
                        std::process::exit(1);
                    }

                    if !batch_stdout.is_empty() {
                        plan_stdout.extend(batch_stdout);
                    }
                }

                if batch_failure_count != 0 {
                    consecutive_blocks += batch_failure_count;
                    maybe_hard_reset_after_blocks(
                        &mut consecutive_blocks,
                        &mut epoch.world,
                        &epoch.accumulated_edges,
                        &mut current_payload,
                        std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                        projection_bias,
                    );
                    if consecutive_blocks == 0 {
                        current_payload_origin = None;
                    }
                    continue 'ticks;
                }
                consecutive_blocks = 0;
                if !plan_stdout.is_empty() {
                    current_payload = json!({ "context": plan_stdout });
                    current_payload_origin = Some("scheduler:cka_parallel".to_string());
                }

                if let Err(error) = epoch.world.decay(config.decay_normal) {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
                let _ = epoch
                    .world
                    .reconstruct_projected_tensor_path(LAYER_CONFIDENCE);
                continue;
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }

        let decision = match epoch.world.frontier_step(projection_bias) {
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
            node_name(decision.selected_src, epoch.topology.registry()),
            node_name(decision.selected_dst, epoch.topology.registry()),
            node_name(decision.first_hop, epoch.topology.registry()),
            format_quantale_value(decision.selected_value),
            action_label(decision.first_hop, epoch.topology.registry()),
            decision.halted,
            decision.blocked,
        );

        if decision.halted != 0 {
            if config.max_ticks == 0 {
                // Continuous mode: dampen the halt edge and restart the trading cycle.
                epoch.world.queue_lattice_update(
                    decision.selected_src,
                    decision.first_hop,
                    ExecutionOutcome::Failure,
                );
                let _ = epoch.world.drain_lattice_queue();
                let _ = epoch.world.decay(config.decay_blocked);
                current_payload = json!({ "context": "market_analysis_loop" });
                current_payload_origin = None;
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
            if consecutive_blocks >= config.hard_reset_blocks {
                maybe_hard_reset_after_blocks(
                    &mut consecutive_blocks,
                    &mut epoch.world,
                    &epoch.accumulated_edges,
                    &mut current_payload,
                    std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                    projection_bias,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
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

        let Some(active_node) = Node::decode(decision.first_hop, epoch.topology.registry()) else {
            eprintln!("Invalid first_hop index: {}", decision.first_hop);
            break;
        };
        let Some(active_node_name) = active_node.name(epoch.topology.registry()) else {
            eprintln!("Invalid first_hop index: {}", decision.first_hop);
            break;
        };

        // Invariants 18 + 19: validate decision report before running executor.
        for v in runtime_check::check_decision(&decision, active_node_name) {
            eprintln!("[runtime_check] {v}");
        }

        println!("[STEP] Tensor frontier advanced to node: {active_node_name}");

        if let Err(violation) = epoch.contracts.validate(
            active_node_name,
            &current_payload,
            current_payload_origin.as_deref(),
            ContractContext::Frontier,
        ) {
            eprintln!("[contract] {violation}; skipping executor call");
            let process_receipt = contract_violation_receipt(active_node_name, &violation);
            epoch.world.queue_lattice_update(
                decision.selected_src,
                decision.first_hop,
                ExecutionOutcome::Failure,
            );
            epoch
                .exploration_engine
                .update_receipt_prior(decision.first_hop, &process_receipt);
            if let Err(error) = tlog.append_exploration_receipt(&json!({
                "node": active_node_name,
                "exit_code": process_receipt.exit_code,
                "contract_violation": violation.reason,
                "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
            })) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            if let Err(error) = epoch.world.drain_lattice_queue() {
                eprintln!("{error}");
                std::process::exit(1);
            }
            if let Err(error) = tlog.log_step(&process_receipt, &decision) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            consecutive_blocks += 1;
            maybe_hard_reset_after_blocks(
                &mut consecutive_blocks,
                &mut epoch.world,
                &epoch.accumulated_edges,
                &mut current_payload,
                std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                projection_bias,
            );
            if consecutive_blocks == 0 {
                current_payload_origin = None;
            }
            continue;
        }

        let process_receipt = epoch
            .executor
            .execute_abstract_node_blocking(active_node_name, &current_payload);
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

        epoch
            .world
            .queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
        epoch
            .exploration_engine
            .update_receipt_prior(decision.first_hop, &process_receipt);
        if let Err(error) = tlog.append_exploration_receipt(&json!({
            "node": active_node_name,
            "exit_code": process_receipt.exit_code,
            "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
        })) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
            if epoch.executor.output_mode(active_node_name) == Some("tensor_plan") {
                match compile_tensor_plan(&process_receipt.stdout_payload) {
                    Ok(plan_edges) => {
                        let plan_edges =
                            filter_static_topology_edges(plan_edges, epoch.topology.tensor_edges());
                        if !plan_edges.is_empty() {
                            println!(
                                "[ALGEBRA] Tensor LLM plan: {} edge(s) → VRAM",
                                plan_edges.len()
                            );
                            if let Err(error) = epoch.world.embed_tensor_edges(&plan_edges) {
                                eprintln!("{error}");
                                std::process::exit(1);
                            }
                            epoch.accumulated_edges.extend(plan_edges.clone());
                            if let Err(error) =
                                tlog.append_tensor_edges("plan:tensor_llm", &plan_edges)
                            {
                                eprintln!("{error}");
                                std::process::exit(1);
                            }
                        }
                        // Tensor plan edges are now in the GPU world — do not
                        // recycle them as context for the next operator call.
                        // Passing a JSON edge array as context creates nested
                        // JSON on each iteration that call_llm.py must unwrap.
                    }
                    Err(reason) => {
                        println!(
                            "[WARN] Tensor LLM plan invalid ({reason}); dampening selected edge"
                        );
                        epoch.world.queue_lattice_update(
                            decision.selected_src,
                            decision.first_hop,
                            ExecutionOutcome::Failure,
                        );
                    }
                }
            } else {
                current_payload = json!({ "context": process_receipt.stdout_payload });
                current_payload_origin = Some(active_node_name.to_string());
            }
        }

        if let Err(error) = epoch.world.drain_lattice_queue() {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if let Err(error) = epoch.world.decay(config.decay_normal) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if let Err(error) = tlog.log_step(&process_receipt, &decision) {
            eprintln!("{error}");
            std::process::exit(1);
        }

        if process_receipt.exit_code != 0 {
            consecutive_blocks += 1;
            maybe_hard_reset_after_blocks(
                &mut consecutive_blocks,
                &mut epoch.world,
                &epoch.accumulated_edges,
                &mut current_payload,
                std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                projection_bias,
            );
            if consecutive_blocks == 0 {
                current_payload_origin = None;
            }
            continue;
        }
        consecutive_blocks = 0;

        let _ = epoch
            .world
            .reconstruct_projected_tensor_path(LAYER_CONFIDENCE);
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

fn build_runtime_epoch(
    id: usize,
    config: &mut SystemConfig,
    learning_policy: &LearningPolicy,
    tlog: &mut TlogWriter,
) -> Result<RuntimeEpoch, String> {
    config.reload_default_operator_registry()?;

    let topology = TopologyRuntime::load_checked_default().map_err(|error| error.to_string())?;
    let invariants = TopologyInvariants::default_asset();
    let operator_violations =
        check_with_operators(&topology.document, &invariants, &config.operator_registry);
    let blocking_operator_violations: Vec<_> = operator_violations
        .into_iter()
        .filter(|violation| violation.kind == ViolationKind::MissingOperator)
        .collect();
    if !blocking_operator_violations.is_empty() {
        return Err(format!(
            "{}\n{} operator topology violation(s) found",
            format_violations(&blocking_operator_violations),
            blocking_operator_violations.len()
        ));
    }

    let executor = UniversalExecutor::from_registry(
        config.operator_registry.clone(),
        topology.registry().clone(),
    );
    let contracts = NodeContracts::default_asset();

    let mut tensor_edges = topology.tensor_edges().to_vec();
    tlog.append_tensor_edges(&format!("topology:tensor:epoch:{id}"), &tensor_edges)
        .map_err(|error| error.to_string())?;

    let learned_edges = load_learned_tensor_edges(
        &config.learned_edges_path,
        topology.registry(),
        topology.tensor_edges(),
        learning_policy,
    )
    .map_err(|error| error.to_string())?;
    if !learned_edges.is_empty() {
        tlog.append_tensor_edges(&format!("state:learned:epoch:{id}"), &learned_edges)
            .map_err(|error| error.to_string())?;
        tensor_edges.extend(learned_edges);
    }

    let patterns = load_default_patterns().map_err(|error| error.to_string())?;
    let mut compiled_patterns = Vec::new();
    for pattern in &patterns.patterns {
        let compiled = compile_pattern(pattern, &topology.compiled, &config.operator_registry)
            .map_err(|error| error.to_string())?;
        tlog.append_tensor_edges(&format!("pattern:cka:epoch:{id}"), &compiled.edges)
            .map_err(|error| error.to_string())?;
        tensor_edges.extend(compiled.edges.clone());
        compiled_patterns.push(compiled);
    }

    let world =
        TensorQuantaleWorld::from_tensor_edges(&tensor_edges).map_err(|error| error.to_string())?;

    let exploration_config =
        ExplorationConfig::default_asset().map_err(|error| error.to_string())?;
    let exploration_engine = ExplorationEngine::new(
        exploration_config,
        &topology.document,
        config.operator_registry.clone(),
    )
    .map_err(|error| error.to_string())?;

    Ok(RuntimeEpoch {
        id,
        fingerprint: current_asset_fingerprint(),
        topology,
        executor,
        contracts,
        accumulated_edges: tensor_edges,
        compiled_patterns,
        world,
        exploration_engine,
    })
}

fn changed_asset_fingerprint(previous: &AssetFingerprint) -> Option<AssetFingerprint> {
    let current = current_asset_fingerprint();
    (current != *previous).then_some(current)
}

fn current_asset_fingerprint() -> AssetFingerprint {
    AssetFingerprint {
        entries: watched_asset_paths()
            .into_iter()
            .map(|path| {
                let metadata = std::fs::metadata(&path).ok();
                let stamp = metadata.and_then(|metadata| {
                    metadata
                        .modified()
                        .ok()
                        .map(|modified| (modified, metadata.len()))
                });
                (path, stamp)
            })
            .collect(),
    }
}

fn watched_asset_paths() -> Vec<PathBuf> {
    [
        "assets/topology.generated.json",
        "assets/topology.json",
        "assets/operators.generated.json",
        "assets/operators.json",
        "assets/node_contracts.json",
        "assets/patterns.json",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect()
}

fn maybe_hard_reset_after_blocks(
    consecutive_blocks: &mut usize,
    world: &mut TensorQuantaleWorld,
    accumulated_edges: &[quantale_semiring_v2::TensorEdge],
    current_payload: &mut Value,
    hard_reset_sleep: std::time::Duration,
    projection_bias: ProjectionBias,
) {
    if *consecutive_blocks == 0 {
        return;
    }

    // Hard reset: re-embed the full accumulated edge set (static topology +
    // all LLM-proposed edges collected since startup) so learned dev-chain
    // weights survive cycle restarts.  reset() zeros the tensor to BOTTOM and
    // re-initialises the witness via embed, which is required for close() to
    // build a correct transitive closure.
    eprintln!(
        "[WARN] {} consecutive blocked/failed steps; hard reset.",
        *consecutive_blocks
    );
    if let Err(error) = world.reset() {
        eprintln!("[WARN] hard reset world.reset() failed: {error}");
    }
    if let Err(error) = world.embed_tensor_edges(accumulated_edges) {
        eprintln!("[WARN] hard reset embed failed: {error}");
    }
    if let Err(error) = world.close() {
        eprintln!("[WARN] hard reset close failed: {error}");
    }
    *current_payload = json!({ "context": "market_analysis_loop" });
    *consecutive_blocks = 0;
    std::thread::sleep(hard_reset_sleep);

    // Invariant 17: verify reset restored a valid frontier. Uses project
    // (read-only) so active[] is not advanced.
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

fn contract_violation_receipt(node_name: &str, violation: &ContractViolation) -> ProcessReceipt {
    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 125,
        stdout_payload: String::new(),
        stderr_payload: format!("contract violation: {}", violation.reason),
    }
}
