use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::SystemTime;

use serde_json::{Value, json};

use quantale_semiring_v2::{
    ContractContext, ContractViolation, DecisionReport, ExecutionOutcome, ExplorationConfig,
    ExplorationEngine, GraphTopology, LAYER_CONFIDENCE, LearningPolicy, Node, NodeContracts,
    ProcessReceipt, ProjectionBias, ReloadPolicy, RuntimeContext, SystemConfig,
    TensorQuantaleWorld, TlogWriter, TopologyInvariants, TopologyRuntime, UniversalExecutor,
    ViolationKind, action_label, check, check_with_operators, compile_and_emit_pattern_edges,
    compile_pattern, compile_tensor_plan, console, format_quantale_value, format_violations,
    load_compiled_pattern_edges, load_default_patterns, load_learned_tensor_edges, runtime_check,
};

use topology_core::build_overlay_assets;

mod cli;
mod runtime_dispatch;
mod runtime_epoch;
mod runtime_parallel;
mod runtime_reset;

use runtime_dispatch::{
    FusionLogicalAdvance, apply_hot_dispatch_if_needed, execute_active_node_blocking,
    filter_static_topology_edges, node_name, queue_execution_lattice_updates,
    record_learning_edge_for_pair, record_learning_edges, update_execution_receipt_priors,
};
use runtime_parallel::dispatch_gpu_parallel_group;
use runtime_epoch::{RuntimeEpoch, build_runtime_epoch, changed_asset_fingerprint};
use runtime_reset::maybe_hard_reset_after_blocks;

macro_rules! fatal {
    ($scope:expr, $message:expr, $error:expr) => {{
        console::error($scope, $message, &[("error", $error.to_string())]);
        std::process::exit(1);
    }};
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if let cli::CliCommand::Exit(code) = cli::handle(&args) {
        std::process::exit(code);
    }

    let mut config = SystemConfig::default();
    let mut runtime_context = match RuntimeContext::default_asset() {
        Ok(context) => context,
        Err(error) => {
            fatal!("runtime", "load_runtime_context_failed", error);
        }
    };
    let mut runtime_invariants = match runtime_check::RuntimeInvariantPolicy::default_asset() {
        Ok(policy) => policy,
        Err(error) => {
            fatal!("runtime", "load_runtime_invariants_failed", error);
        }
    };
    let projection_bias = ProjectionBias::default();
    let learning_policy = LearningPolicy::default_asset();

    let mut tlog = match TlogWriter::open(&config.tlog_path) {
        Ok(tlog) => tlog,
        Err(error) => {
            fatal!("tlog", "open_failed", error);
        }
    };

    let mut epoch = match build_runtime_epoch(0, &mut config, &learning_policy, &mut tlog) {
        Ok(epoch) => epoch,
        Err(error) => {
            fatal!("runtime", "build_epoch_failed", error);
        }
    };

    console::info(
        "runtime",
        "starting",
        &[
            (
                "mode",
                if config.max_ticks == 0 {
                    "continuous"
                } else {
                    "bounded"
                }
                .to_string(),
            ),
            ("max_ticks", config.max_ticks.to_string()),
            ("tick_sleep_ms", config.tick_sleep_ms.to_string()),
        ],
    );

    let sleep_dur =
        (config.tick_sleep_ms > 0).then(|| std::time::Duration::from_millis(config.tick_sleep_ms));
    let mut current_payload = runtime_context.default_payload();
    let mut current_payload_origin: Option<String> = None;
    let mut tick: usize = 0;
    let mut consecutive_blocks: usize = 0;

    loop {
        if config.max_ticks > 0 && tick >= config.max_ticks {
            break;
        }
        tick += 1;
        if let Some(next_fingerprint) = changed_asset_fingerprint(&epoch.fingerprint) {
            match build_runtime_epoch(epoch.id + 1, &mut config, &learning_policy, &mut tlog) {
                Ok(next_epoch) => {
                    if let Err(error) = epoch.learning_buffer.flush() {
                        console::warn("learning", "flush_failed", &[("error", error)]);
                    }
                    runtime_context = match RuntimeContext::default_asset() {
                        Ok(context) => context,
                        Err(error) => {
                            fatal!("runtime", "reload_runtime_context_failed", error);
                        }
                    };
                    runtime_invariants =
                        match runtime_check::RuntimeInvariantPolicy::default_asset() {
                            Ok(policy) => policy,
                            Err(error) => {
                                fatal!("runtime", "reload_runtime_invariants_failed", error);
                            }
                        };
                    console::info(
                        "topology",
                        "reloaded",
                        &[
                            ("from_epoch", epoch.id.to_string()),
                            ("to_epoch", next_epoch.id.to_string()),
                            (
                                "nodes",
                                next_epoch.topology.document.nodes.len().to_string(),
                            ),
                            (
                                "transitions",
                                next_epoch.topology.document.transitions.len().to_string(),
                            ),
                        ],
                    );
                    epoch = next_epoch;
                    current_payload = runtime_context.default_payload();
                    current_payload_origin = None;
                    consecutive_blocks = 0;
                }
                Err(error) => {
                    console::warn(
                        "topology",
                        "reload_rejected",
                        &[
                            ("epoch", epoch.id.to_string()),
                            ("error", error.to_string()),
                        ],
                    );
                    epoch.fingerprint = next_fingerprint;
                }
            }
        }

        if let Err(error) = epoch.world.close() {
            fatal!("tensor", "close_failed", error);
        }

        let exploration_candidates = match epoch
            .world
            .expand_exploration(&mut epoch.exploration_engine)
        {
            Ok(candidates) => candidates,
            Err(error) => {
                console::warn(
                    "exploration",
                    "unavailable",
                    &[
                        ("fallback", "cka_frontier".to_string()),
                        ("error", error.to_string()),
                    ],
                );
                Vec::new()
            }
        };
        if let Err(error) = tlog.append_exploration_seed(&json!({
            "token_count": epoch.exploration_engine.tokens().len(),
            "strategy_count": epoch.exploration_engine.config().strategies.len(),
        })) {
            fatal!("tlog", "append_exploration_seed_failed", error);
        }
        if let Err(error) = tlog.append_exploration_topk(&json!({
            "candidate_count": exploration_candidates.len(),
            "candidates": exploration_candidates,
        })) {
            fatal!("tlog", "append_exploration_topk_failed", error);
        }
        let exploration_winner = epoch.exploration_engine.best_commit_candidate();
        if let Some(candidate) = exploration_winner {
            let commit_record = epoch.exploration_engine.commit_record(candidate);
            if let Err(error) = tlog.append_exploration_commit(&commit_record) {
                fatal!("tlog", "append_exploration_commit_failed", error);
            }
            let decision = match epoch.world.commit_exploration_candidate(&candidate) {
                Ok(decision) => decision,
                Err(error) => {
                    fatal!("exploration", "commit_candidate_failed", error);
                }
            };
            if let Err(error) = tlog.append_decision(&decision) {
                fatal!("tlog", "append_decision_failed", error);
            }
            epoch
                .exploration_engine
                .mark_candidate_committed(&candidate);
            console::info(
                "exploration",
                "projection",
                &[
                    (
                        "src",
                        node_name(decision.selected_src, epoch.topology.registry()),
                    ),
                    (
                        "dst",
                        node_name(decision.selected_dst, epoch.topology.registry()),
                    ),
                    (
                        "first_hop",
                        node_name(decision.first_hop, epoch.topology.registry()),
                    ),
                    ("score", format_quantale_value(decision.selected_value)),
                    ("path", format!("{:?}", commit_record.path)),
                ],
            );
            if decision.blocked != 0 {
                continue;
            }
            let Some(active_node) = Node::decode(decision.first_hop, epoch.topology.registry())
            else {
                console::error(
                    "exploration",
                    "invalid_first_hop",
                    &[("first_hop", decision.first_hop.to_string())],
                );
                break;
            };
            let Some(active_node_name) = active_node.name(epoch.topology.registry()) else {
                console::error(
                    "exploration",
                    "invalid_first_hop",
                    &[("first_hop", decision.first_hop.to_string())],
                );
                break;
            };
            if let Err(violation) = epoch.contracts.validate(
                active_node_name,
                &current_payload,
                current_payload_origin.as_deref(),
                ContractContext::Exploration,
            ) {
                console::warn(
                    "contract",
                    "exploration_skipped",
                    &[
                        ("node", active_node_name.to_string()),
                        ("reason", violation.reason.clone()),
                    ],
                );
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
                    fatal!("tlog", "append_exploration_receipt_failed", error);
                }
                if let Err(error) = epoch.world.drain_lattice_queue() {
                    fatal!("tensor", "drain_lattice_queue_failed", error);
                }
                if let Err(error) = tlog.log_step(&process_receipt, &decision) {
                    fatal!("tlog", "log_step_failed", error);
                }
                consecutive_blocks += 1;
                maybe_hard_reset_after_blocks(
                    &mut consecutive_blocks,
                    &mut epoch.world,
                    &epoch.accumulated_edges,
                    &mut current_payload,
                    std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                    projection_bias,
                    &runtime_context,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
                }
                continue;
            }
            let execution = execute_active_node_blocking(
                &epoch,
                &config,
                &decision,
                active_node_name,
                &current_payload,
            );
            let process_receipt = &execution.receipt;
            let outcome = ExecutionOutcome::from(process_receipt);
            apply_hot_dispatch_if_needed(&mut epoch.world, &epoch.executor, &execution);
            queue_execution_lattice_updates(&mut epoch.world, &decision, &execution, outcome);
            record_learning_edges(&mut epoch.learning_buffer, &epoch.topology, &decision, &execution, &learning_policy);
            update_execution_receipt_priors(
                &mut epoch.exploration_engine,
                &decision,
                &execution,
                process_receipt,
            );
            if let Err(error) = tlog.append_exploration_receipt(&json!({
                "node": active_node_name,
                "exit_code": process_receipt.exit_code,
                "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
                "fusion": execution.fusion.as_ref().map(FusionLogicalAdvance::receipt_json),
            })) {
                fatal!("tlog", "append_exploration_receipt_failed", error);
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
                                    fatal!("tensor", "embed_tensor_edges_failed", error);
                                }
                                epoch.accumulated_edges.extend(plan_edges.clone());
                                if let Err(error) =
                                    tlog.append_tensor_edges("exploration:plan_tensor", &plan_edges)
                                {
                                    fatal!("tlog", "append_tensor_edges_failed", error);
                                }
                            }
                            // Tensor plan edges are now in the GPU world — do not
                            // recycle them as context for the next operator call.
                            // Passing a JSON edge array as context creates nested
                            // JSON on each iteration that call_llm.py must unwrap.
                        }
                        Err(reason) => {
                            console::warn(
                                "tensor",
                                "exploration_plan_invalid",
                                &[("reason", reason)],
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
                    current_payload_origin =
                        Some(execution.output_origin(active_node_name).to_string());
                }
            }
            if let Err(error) = epoch.world.drain_lattice_queue() {
                fatal!("tensor", "drain_lattice_queue_failed", error);
            }
            if let Err(error) = tlog.log_step(process_receipt, &decision) {
                fatal!("tlog", "log_step_failed", error);
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
                    &runtime_context,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
                }
                continue;
            }
            consecutive_blocks = 0;
            if let Err(error) = epoch.world.decay(config.decay_normal) {
                fatal!("tensor", "decay_failed", error);
            }
            continue;
        }

        // ── GPU-native parallel tier ──────────────────────────────────────────
        // One kernel selects, validates, and commits a par group.  CPU only
        // resolves names, dispatches eligible GPU operators, and drains receipts.
        // If par_group_data is absent (no CUDA device or no eligible groups)
        // this block falls through to frontier_step immediately.
        let parallel_committed = 'par_tier: {
            let Some(data) = &epoch.par_group_data else {
                break 'par_tier false;
            };
            let step = match epoch.world.par_group_step(data, projection_bias) {
                Err(error) => {
                    console::warn("parallel", "gpu_step_error", &[("error", error.to_string())]);
                    break 'par_tier false;
                }
                Ok(None) => break 'par_tier false,
                Ok(Some(result)) => result,
            };
            let (group_idx, par_decisions) = step;

            let par_names: Vec<String> = epoch
                .topology
                .parallel_groups
                .get(group_idx)
                .map(|g| {
                    g.iter()
                        .filter_map(|&id| {
                            epoch.topology.registry().name_of(id as usize).map(str::to_string)
                        })
                        .collect()
                })
                .unwrap_or_default();

            let fusion_entries: Vec<Option<&quantale_semiring_v2::FusionEntry>> = par_names
                .iter()
                .map(|n| config.fusion_dispatch.get_by_entry(n))
                .collect();

            let par_receipts = dispatch_gpu_parallel_group(
                &epoch.executor,
                &fusion_entries,
                &par_names,
                &current_payload,
            );

            console::info(
                "parallel",
                "gpu_group_committed",
                &[
                    ("nodes", par_names.join(" || ")),
                    ("size", par_names.len().to_string()),
                ],
            );

            let mut par_failures = 0usize;
            let mut par_stdout: Vec<Value> = Vec::new();
            for ((decision, receipt), par_node_name) in par_decisions
                .iter()
                .zip(par_receipts.iter())
                .zip(par_names.iter())
            {
                if let Err(error) = tlog.append_decision(decision) {
                    fatal!("tlog", "append_decision_failed", error);
                }
                console::info(
                    "parallel",
                    "operator_receipt",
                    &[
                        ("node", par_node_name.clone()),
                        ("exit", receipt.exit_code.to_string()),
                        ("outcome", format!("{:?}", ExecutionOutcome::from(receipt))),
                    ],
                );
                if !receipt.stderr_payload.is_empty() {
                    console::warn(
                        "parallel",
                        "operator_stderr",
                        &[
                            ("node", par_node_name.clone()),
                            ("stderr", receipt.stderr_payload.trim().to_string()),
                        ],
                    );
                }
                let par_outcome = ExecutionOutcome::from(receipt);
                epoch.world.queue_lattice_update(
                    decision.selected_src,
                    decision.first_hop,
                    par_outcome,
                );
                epoch.exploration_engine.update_receipt_prior(decision.first_hop, receipt);
                if let Err(error) = tlog.append_exploration_receipt(&json!({
                    "node": par_node_name,
                    "exit_code": receipt.exit_code,
                    "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
                })) {
                    fatal!("tlog", "append_exploration_receipt_failed", error);
                }
                if let Err(error) = tlog.log_step(receipt, decision) {
                    fatal!("tlog", "log_step_failed", error);
                }
                if receipt.exit_code != 0 {
                    par_failures += 1;
                } else {
                    record_learning_edge_for_pair(
                        &mut epoch.learning_buffer,
                        &epoch.topology,
                        decision.selected_src,
                        decision.first_hop,
                        &learning_policy,
                    );
                    if !receipt.stdout_payload.is_empty() {
                        if epoch.executor.output_mode(par_node_name) == Some("tensor_plan") {
                            match compile_tensor_plan(&receipt.stdout_payload) {
                                Ok(plan_edges) => {
                                    let plan_edges = filter_static_topology_edges(
                                        plan_edges,
                                        epoch.topology.tensor_edges(),
                                    );
                                    if !plan_edges.is_empty() {
                                        if let Err(error) =
                                            epoch.world.embed_tensor_edges(&plan_edges)
                                        {
                                            fatal!("tensor", "embed_tensor_edges_failed", error);
                                        }
                                        epoch.accumulated_edges.extend(plan_edges.clone());
                                        if let Err(error) = tlog.append_tensor_edges(
                                            "plan:tensor_parallel",
                                            &plan_edges,
                                        ) {
                                            fatal!("tlog", "append_tensor_edges_failed", error);
                                        }
                                    }
                                }
                                Err(reason) => {
                                    console::warn(
                                        "parallel",
                                        "plan_invalid",
                                        &[("reason", reason)],
                                    );
                                    epoch.world.queue_lattice_update(
                                        decision.selected_src,
                                        decision.first_hop,
                                        ExecutionOutcome::Failure,
                                    );
                                }
                            }
                        } else {
                            par_stdout.push(json!({
                                "node": par_node_name,
                                "stdout": receipt.stdout_payload,
                            }));
                        }
                    }
                }
            }
            if let Err(error) = epoch.world.drain_lattice_queue() {
                fatal!("tensor", "drain_lattice_queue_failed", error);
            }
            if par_failures > 0 {
                consecutive_blocks += par_failures;
                maybe_hard_reset_after_blocks(
                    &mut consecutive_blocks,
                    &mut epoch.world,
                    &epoch.accumulated_edges,
                    &mut current_payload,
                    std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                    projection_bias,
                    &runtime_context,
                );
                if consecutive_blocks == 0 {
                    current_payload_origin = None;
                }
                break 'par_tier true;
            }
            consecutive_blocks = 0;
            if !par_stdout.is_empty() {
                current_payload = json!({ "context": par_stdout });
                current_payload_origin = Some("parallel:cka_par".to_string());
            }
            if let Err(error) = epoch.world.decay(config.decay_normal) {
                fatal!("tensor", "decay_failed", error);
            }
            true
        };
        if parallel_committed {
            continue;
        }

        let decision = match epoch.world.frontier_step(projection_bias) {
            Ok(decision) => decision,
            Err(error) => {
                fatal!("frontier", "step_failed", error);
            }
        };

        if let Err(error) = tlog.append_decision(&decision) {
            fatal!("tlog", "append_decision_failed", error);
        }

        console::info(
            "frontier",
            "projection",
            &[
                ("step", decision.step.to_string()),
                (
                    "src",
                    node_name(decision.selected_src, epoch.topology.registry()),
                ),
                (
                    "dst",
                    node_name(decision.selected_dst, epoch.topology.registry()),
                ),
                (
                    "first_hop",
                    node_name(decision.first_hop, epoch.topology.registry()),
                ),
                ("score", format_quantale_value(decision.selected_value)),
                (
                    "action",
                    format!(
                        "{:?}",
                        action_label(decision.first_hop, epoch.topology.registry())
                    ),
                ),
                ("halted", decision.halted.to_string()),
                ("blocked", decision.blocked.to_string()),
            ],
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
                current_payload = runtime_context.reset_payload();
                current_payload_origin = None;
                if let Some(dur) = sleep_dur {
                    std::thread::sleep(dur);
                }
                continue;
            }
            console::info("frontier", "halted", &[]);
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
                    &runtime_context,
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
            console::warn(
                "runtime_check",
                "unsafe_decision_skipped",
                &[("first_hop", decision.first_hop.to_string())],
            );
            consecutive_blocks += 1;
            continue;
        }

        consecutive_blocks = 0;

        let Some(active_node) = Node::decode(decision.first_hop, epoch.topology.registry()) else {
            console::error(
                "frontier",
                "invalid_first_hop",
                &[("first_hop", decision.first_hop.to_string())],
            );
            break;
        };
        let Some(active_node_name) = active_node.name(epoch.topology.registry()) else {
            console::error(
                "frontier",
                "invalid_first_hop",
                &[("first_hop", decision.first_hop.to_string())],
            );
            break;
        };

        // Invariants 18 + 19: validate decision report before running executor.
        for v in runtime_check::check_decision_with_policy(
            &decision,
            active_node_name,
            &runtime_invariants,
        ) {
            console::warn("runtime_check", "violation", &[("detail", v.to_string())]);
        }

        console::info(
            "frontier",
            "advanced",
            &[("node", active_node_name.to_string())],
        );

        if let Err(violation) = epoch.contracts.validate(
            active_node_name,
            &current_payload,
            current_payload_origin.as_deref(),
            ContractContext::Frontier,
        ) {
            console::warn(
                "contract",
                "frontier_skipped",
                &[
                    ("node", active_node_name.to_string()),
                    ("reason", violation.reason.clone()),
                ],
            );
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
                fatal!("tlog", "append_exploration_receipt_failed", error);
            }
            if let Err(error) = epoch.world.drain_lattice_queue() {
                fatal!("tensor", "drain_lattice_queue_failed", error);
            }
            if let Err(error) = tlog.log_step(&process_receipt, &decision) {
                fatal!("tlog", "log_step_failed", error);
            }
            consecutive_blocks += 1;
            maybe_hard_reset_after_blocks(
                &mut consecutive_blocks,
                &mut epoch.world,
                &epoch.accumulated_edges,
                &mut current_payload,
                std::time::Duration::from_millis(config.hard_reset_sleep_ms),
                projection_bias,
                &runtime_context,
            );
            if consecutive_blocks == 0 {
                current_payload_origin = None;
            }
            continue;
        }

        let execution = execute_active_node_blocking(
            &epoch,
            &config,
            &decision,
            active_node_name,
            &current_payload,
        );
        let process_receipt = &execution.receipt;
        let outcome = ExecutionOutcome::from(process_receipt);
        apply_hot_dispatch_if_needed(&mut epoch.world, &epoch.executor, &execution);

        console::info(
            "operator",
            "receipt",
            &[
                ("node", active_node_name.to_string()),
                ("exit", process_receipt.exit_code.to_string()),
                ("outcome", format!("{outcome:?}")),
                (
                    "stdout_len",
                    process_receipt.stdout_payload.len().to_string(),
                ),
            ],
        );
        if !process_receipt.stderr_payload.is_empty() {
            console::warn(
                "operator",
                "stderr",
                &[
                    ("node", active_node_name.to_string()),
                    ("stderr", process_receipt.stderr_payload.trim().to_string()),
                ],
            );
        }
        // Invariant 24: declared block nodes must result in blocked or halted state.
        if runtime_invariants.is_block_node(active_node_name)
            && outcome == ExecutionOutcome::Success
        {
            for v in runtime_check::check_decision_with_policy(
                &decision,
                active_node_name,
                &runtime_invariants,
            ) {
                if v.kind == runtime_check::RuntimeViolationKind::BlockNodeNotBlocked {
                    console::warn("runtime_check", "violation", &[("detail", v.to_string())]);
                }
            }
        }

        queue_execution_lattice_updates(&mut epoch.world, &decision, &execution, outcome);
        record_learning_edges(&mut epoch.learning_buffer, &epoch.topology, &decision, &execution, &learning_policy);
        update_execution_receipt_priors(
            &mut epoch.exploration_engine,
            &decision,
            &execution,
            process_receipt,
        );
        if let Err(error) = tlog.append_exploration_receipt(&json!({
            "node": active_node_name,
            "exit_code": process_receipt.exit_code,
            "prior": epoch.exploration_engine.receipt_prior_for(decision.first_hop),
            "fusion": execution.fusion.as_ref().map(FusionLogicalAdvance::receipt_json),
        })) {
            fatal!("tlog", "append_exploration_receipt_failed", error);
        }

        if process_receipt.exit_code == 0 && !process_receipt.stdout_payload.is_empty() {
            if epoch.executor.output_mode(active_node_name) == Some("tensor_plan") {
                match compile_tensor_plan(&process_receipt.stdout_payload) {
                    Ok(plan_edges) => {
                        let plan_edges =
                            filter_static_topology_edges(plan_edges, epoch.topology.tensor_edges());
                        if !plan_edges.is_empty() {
                            console::info(
                                "tensor",
                                "llm_plan_embedded",
                                &[("edges", plan_edges.len().to_string())],
                            );
                            if let Err(error) = epoch.world.embed_tensor_edges(&plan_edges) {
                                fatal!("tensor", "embed_tensor_edges_failed", error);
                            }
                            epoch.accumulated_edges.extend(plan_edges.clone());
                            if let Err(error) =
                                tlog.append_tensor_edges("plan:tensor_llm", &plan_edges)
                            {
                                fatal!("tlog", "append_tensor_edges_failed", error);
                            }
                        }
                        // Tensor plan edges are now in the GPU world — do not
                        // recycle them as context for the next operator call.
                        // Passing a JSON edge array as context creates nested
                        // JSON on each iteration that call_llm.py must unwrap.
                    }
                    Err(reason) => {
                        console::warn("tensor", "llm_plan_invalid", &[("reason", reason)]);
                        epoch.world.queue_lattice_update(
                            decision.selected_src,
                            decision.first_hop,
                            ExecutionOutcome::Failure,
                        );
                    }
                }
            } else {
                current_payload = json!({ "context": process_receipt.stdout_payload });
                current_payload_origin =
                    Some(execution.output_origin(active_node_name).to_string());
            }
        }

        if let Err(error) = epoch.world.drain_lattice_queue() {
            fatal!("tensor", "drain_lattice_queue_failed", error);
        }

        if let Err(error) = epoch.world.decay(config.decay_normal) {
            fatal!("tensor", "decay_failed", error);
        }

        if let Err(error) = tlog.log_step(process_receipt, &decision) {
            fatal!("tlog", "log_step_failed", error);
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
                &runtime_context,
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

    if let Err(error) = epoch.learning_buffer.flush() {
        console::warn("learning", "shutdown_flush_failed", &[("error", error)]);
    }

    if let Err(error) = tlog.flush() {
        fatal!("tlog", "flush_failed", error);
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
