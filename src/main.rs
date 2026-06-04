use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;

use serde_json::{Value, json};

use quantale_semiring_v2::{
    CompiledCkaPattern, ContractContext, ContractViolation, DecisionReport, ExecutionOutcome,
    ExplorationConfig, ExplorationEngine, GraphTopology, LAYER_CONFIDENCE, LearningPolicy, Node,
    NodeContracts, ProcessReceipt, ProjectionBias, ReloadPolicy, RuntimeContext, SystemConfig,
    TensorQuantaleWorld, TlogWriter, TopologyInvariants, TopologyRuntime, UniversalExecutor,
    ViolationKind, action_label, check, check_with_operators, compile_pattern, compile_tensor_plan,
    console, dispatch_decision_batch_blocking, format_quantale_value, format_violations,
    load_default_patterns, load_learned_tensor_edges, project_ready_batch_plan, runtime_check,
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

struct ActiveExecution {
    receipt: ProcessReceipt,
    fusion: Option<FusionLogicalAdvance>,
    /// Set when the execution used the GPU hot path: caller must call
    /// `world.gpu_dispatch_region(region_id, src, dst)` followed by
    /// `world.drain_device_receipts()` to fold the receipt into the tensor.
    hot_dispatch: Option<HotDispatchInfo>,
}

struct HotDispatchInfo {
    region_id: u32,
    src: i32,
    dst: i32,
}

struct FusionLogicalAdvance {
    entry: String,
    exit: String,
    region: String,
    members: Vec<String>,
    edges: Vec<(i32, i32)>,
}

macro_rules! fatal {
    ($scope:expr, $message:expr, $error:expr) => {{
        console::error($scope, $message, &[("error", $error.to_string())]);
        std::process::exit(1);
    }};
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("mutations") {
        std::process::exit(run_mutations_cli(&args[2..]));
    }

    if args.get(1).map(String::as_str) == Some("topology")
        && args.get(2).map(String::as_str) == Some("build-overlay")
    {
        match build_overlay_assets(".") {
            Ok(()) => {
                console::info(
                    "topology",
                    "overlay_written",
                    &[
                        ("topology", "assets/topology.generated.json".to_string()),
                        ("operators", "assets/operators.generated.json".to_string()),
                    ],
                );
                std::process::exit(0);
            }
            Err(error) => {
                fatal!("topology", "build_overlay_failed", error);
            }
        }
    }

    // --check-topology: validate the generated runtime topology and exit.
    if args.iter().any(|a| a == "--check-topology") {
        let topology = match GraphTopology::default_asset() {
            Ok(t) => t,
            Err(error) => {
                fatal!("topology", "parse_failed", error);
            }
        };
        let inv = TopologyInvariants::default_asset();
        let violations = check(&topology, &inv);
        let (warnings, fatal): (Vec<_>, Vec<_>) = violations
            .into_iter()
            .partition(|v| v.kind == ViolationKind::ConsumedBlockPoint);
        for v in &warnings {
            console::warn("topology", "violation", &[("detail", v.to_string())]);
        }
        if fatal.is_empty() {
            console::info(
                "topology",
                "ok",
                &[
                    ("nodes", topology.nodes.len().to_string()),
                    ("transitions", topology.transitions.len().to_string()),
                    ("warnings", warnings.len().to_string()),
                ],
            );
            std::process::exit(0);
        }
        console::error(
            "topology",
            "violations",
            &[
                ("count", fatal.len().to_string()),
                ("detail", format_violations(&fatal)),
            ],
        );
        std::process::exit(1);
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

    'ticks: loop {
        if config.max_ticks > 0 && tick >= config.max_ticks {
            break;
        }
        tick += 1;
        if let Some(next_fingerprint) = changed_asset_fingerprint(&epoch.fingerprint) {
            match build_runtime_epoch(epoch.id + 1, &mut config, &learning_policy, &mut tlog) {
                Ok(next_epoch) => {
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
            apply_hot_dispatch_if_needed(&mut epoch.world, &execution);
            queue_execution_lattice_updates(&mut epoch.world, &decision, &execution, outcome);
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

        match project_ready_batch_plan(
            &mut epoch.world,
            &epoch.compiled_patterns,
            projection_bias,
            &config.operator_registry,
        ) {
            Ok(Some(batch_plan)) => {
                if let Err(error) = tlog.append_batch_plan("scheduler:cka_parallel", &batch_plan) {
                    fatal!("tlog", "append_batch_plan_failed", error);
                }

                let mut batch_failure_count = 0usize;
                let mut plan_stdout = Vec::new();
                for batch in &batch_plan.batches {
                    if let Err(error) = epoch.world.commit_decision_batch(&batch.decisions) {
                        fatal!("tensor", "commit_decision_batch_failed", error);
                    }

                    for decision in &batch.decisions {
                        if let Err(error) = tlog.append_decision(decision) {
                            fatal!("tlog", "append_decision_failed", error);
                        }
                        console::info(
                            "batch",
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
                            ],
                        );
                    }

                    let scheduled_receipts = dispatch_decision_batch_blocking(
                        &epoch.executor,
                        &config.fusion_dispatch,
                        batch,
                        &current_payload,
                    );
                    let mut batch_stdout = Vec::new();

                    for scheduled in scheduled_receipts {
                        let active_node_name = scheduled.receipt.node_name.clone();
                        let outcome = ExecutionOutcome::from(&scheduled.receipt);
                        console::info(
                            "batch",
                            "operator_receipt",
                            &[
                                ("operator", active_node_name.clone()),
                                ("exit", scheduled.receipt.exit_code.to_string()),
                                ("outcome", format!("{outcome:?}")),
                                (
                                    "stdout_len",
                                    scheduled.receipt.stdout_payload.len().to_string(),
                                ),
                            ],
                        );
                        if !scheduled.receipt.stderr_payload.is_empty() {
                            console::warn(
                                "batch",
                                "operator_stderr",
                                &[
                                    ("operator", active_node_name.clone()),
                                    (
                                        "stderr",
                                        scheduled.receipt.stderr_payload.trim().to_string(),
                                    ),
                                ],
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
                            fatal!("tlog", "append_exploration_receipt_failed", error);
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
                                            console::info(
                                                "tensor",
                                                "batch_plan_embedded",
                                                &[("edges", plan_edges.len().to_string())],
                                            );
                                            if let Err(error) =
                                                epoch.world.embed_tensor_edges(&plan_edges)
                                            {
                                                fatal!(
                                                    "tensor",
                                                    "embed_tensor_edges_failed",
                                                    error
                                                );
                                            }
                                            epoch.accumulated_edges.extend(plan_edges.clone());
                                            if let Err(error) = tlog.append_tensor_edges(
                                                "plan:tensor_batch",
                                                &plan_edges,
                                            ) {
                                                fatal!("tlog", "append_tensor_edges_failed", error);
                                            }
                                        }
                                    }
                                    Err(reason) => {
                                        console::warn(
                                            "tensor",
                                            "batch_plan_invalid",
                                            &[("reason", reason)],
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
                            fatal!("tlog", "log_step_failed", error);
                        }
                        if scheduled.receipt.exit_code != 0 {
                            batch_failure_count += 1;
                        }
                    }

                    if let Err(error) = epoch.world.drain_lattice_queue() {
                        fatal!("tensor", "drain_lattice_queue_failed", error);
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
                        &runtime_context,
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
                    fatal!("tensor", "decay_failed", error);
                }
                let _ = epoch
                    .world
                    .reconstruct_projected_tensor_path(LAYER_CONFIDENCE);
                continue;
            }
            Ok(None) => {}
            Err(error) => {
                fatal!("batch", "project_ready_plan_failed", error);
            }
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
        apply_hot_dispatch_if_needed(&mut epoch.world, &execution);

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

    if let Err(error) = tlog.flush() {
        fatal!("tlog", "flush_failed", error);
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

/// If the execution used the GPU hot path, write a device receipt and drain it
/// into the quantale tensor.  No-ops for control/IO executions.
fn apply_hot_dispatch_if_needed(world: &mut TensorQuantaleWorld, execution: &ActiveExecution) {
    #[cfg(feature = "cuda")]
    if let Some(ref info) = execution.hot_dispatch {
        let result = world
            .gpu_dispatch_region(info.region_id as i32, info.src, info.dst)
            .and_then(|_| world.drain_device_receipts());
        if let Err(err) = result {
            console::warn(
                "gpu_dispatch",
                "drain_failed",
                &[("error", err.to_string())],
            );
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (world, execution);
}

fn is_hot_dispatch(node_name: &str, config: &SystemConfig, executor: &UniversalExecutor) -> bool {
    config.hot_region_registry.is_hot(node_name) || executor.is_hot_node(node_name)
}

fn execute_active_node_blocking(
    epoch: &RuntimeEpoch,
    config: &SystemConfig,
    decision: &DecisionReport,
    active_node_name: &str,
    current_payload: &Value,
) -> ActiveExecution {
    // ── Hot GPU path ──────────────────────────────────────────────────────────
    // Hot nodes (jit_cuda or registered GPU regions) bypass process-spawning.
    // The JIT kernel runs on-device; the caller is responsible for calling
    // world.gpu_dispatch_region() and drain_device_receipts() using the
    // returned `hot_dispatch` info so the tensor update stays GPU-side.
    if is_hot_dispatch(active_node_name, config, &epoch.executor) {
        if let Some(region_id) = config.hot_region_registry.region_id_for(active_node_name) {
            let jit_receipt = epoch
                .executor
                .execute_abstract_node_blocking(active_node_name, current_payload);
            return ActiveExecution {
                receipt: jit_receipt,
                fusion: None,
                hot_dispatch: Some(HotDispatchInfo {
                    region_id,
                    src: decision.selected_src,
                    dst: decision.first_hop,
                }),
            };
        }
    }

    // ── Fusion region path ────────────────────────────────────────────────────
    if let Some(entry) = config.fusion_dispatch.get_by_entry(active_node_name) {
        console::info(
            "fusion",
            "dispatch",
            &[
                ("entry", active_node_name.to_string()),
                ("region", entry.region.clone()),
                ("nodes", entry.nodes.join(" -> ")),
            ],
        );
        let receipt = epoch
            .executor
            .execute_fusion_entry_blocking(entry, current_payload);
        let fusion = build_fusion_logical_advance(entry, decision, epoch.topology.registry());
        return ActiveExecution { receipt, fusion, hot_dispatch: None };
    }

    let receipt = epoch
        .executor
        .execute_abstract_node_blocking(active_node_name, current_payload);
    ActiveExecution {
        receipt,
        fusion: None,
        hot_dispatch: None,
    }
}

fn build_fusion_logical_advance(
    entry: &quantale_semiring_v2::FusionEntry,
    decision: &DecisionReport,
    registry: &quantale_semiring_v2::NodeRegistry,
) -> Option<FusionLogicalAdvance> {
    let mut member_ids = Vec::with_capacity(entry.nodes.len());
    for member in &entry.nodes {
        let id = registry.id_of(member)? as i32;
        member_ids.push(id);
    }

    let first = *member_ids.first()?;
    if first != decision.first_hop {
        console::warn(
            "fusion",
            "logical_advance_skipped",
            &[
                ("region", entry.region.clone()),
                (
                    "reason",
                    "entry does not match selected first_hop".to_string(),
                ),
                ("entry_id", first.to_string()),
                ("first_hop", decision.first_hop.to_string()),
            ],
        );
        return None;
    }

    let mut edges = Vec::with_capacity(member_ids.len());
    edges.push((decision.selected_src, first));
    for pair in member_ids.windows(2) {
        edges.push((pair[0], pair[1]));
    }

    Some(FusionLogicalAdvance {
        entry: entry.nodes.first().cloned().unwrap_or_default(),
        exit: entry.nodes.last().cloned().unwrap_or_default(),
        region: entry.region.clone(),
        members: entry.nodes.clone(),
        edges,
    })
}

fn queue_execution_lattice_updates(
    world: &mut TensorQuantaleWorld,
    decision: &DecisionReport,
    execution: &ActiveExecution,
    outcome: ExecutionOutcome,
) {
    if outcome == ExecutionOutcome::Success {
        if let Some(fusion) = &execution.fusion {
            for (src, dst) in &fusion.edges {
                world.queue_lattice_update(*src, *dst, outcome);
            }
            return;
        }
    }

    world.queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
}

fn update_execution_receipt_priors(
    exploration_engine: &mut ExplorationEngine,
    decision: &DecisionReport,
    execution: &ActiveExecution,
    receipt: &ProcessReceipt,
) {
    exploration_engine.update_receipt_prior(decision.first_hop, receipt);
    if receipt.exit_code != 0 {
        return;
    }
    if let Some(fusion) = &execution.fusion {
        for (_, dst) in &fusion.edges {
            if *dst != decision.first_hop {
                exploration_engine.update_receipt_prior(*dst, receipt);
            }
        }
    }
}

impl FusionLogicalAdvance {
    fn receipt_json(&self) -> Value {
        json!({
            "kind": "fusion_region_executed",
            "entry": self.entry,
            "exit": self.exit,
            "members": self.members,
            "member_receipts": self.members.iter().map(|member| {
                json!({
                    "node": member,
                    "exit_code": 0,
                    "outcome": "success",
                    "logical_backend": "fused_region",
                })
            }).collect::<Vec<_>>(),
            "physical_backend": "cuda_jit",
            "logical_advance": "region_atomic",
            "region": self.region,
            "edges": self.edges.iter().map(|(src, dst)| {
                json!({ "src": src, "dst": dst })
            }).collect::<Vec<_>>(),
        })
    }
}

impl ActiveExecution {
    fn output_origin<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.fusion
            .as_ref()
            .filter(|_| self.receipt.exit_code == 0)
            .map(|fusion| fusion.exit.as_str())
            .unwrap_or(fallback)
    }
}

fn build_runtime_epoch(
    id: usize,
    config: &mut SystemConfig,
    learning_policy: &LearningPolicy,
    tlog: &mut TlogWriter,
) -> Result<RuntimeEpoch, String> {
    config.reload_default_operator_registry()?;
    config.reload_hot_region_registry();

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

    let executor = UniversalExecutor::from_config(config);
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

    // Phase 7: log fusion dispatch regions and dry-run CUDA C synthesis.
    if config.fusion_dispatch.is_empty() {
        console::info("fusion", "no_regions", &[("epoch", id.to_string())]);
    } else {
        console::info(
            "fusion",
            "regions_loaded",
            &[
                ("epoch", id.to_string()),
                ("count", config.fusion_dispatch.len().to_string()),
            ],
        );
        for entry in &config.fusion_dispatch.entries {
            console::info(
                "fusion",
                "region",
                &[
                    ("region", entry.region.clone()),
                    ("chain_len", entry.metadata.chain_len.to_string()),
                    ("inputs", entry.chain.inputs.len().to_string()),
                    ("outputs", entry.chain.outputs.len().to_string()),
                    (
                        "estimated_savings",
                        format!("{:.1}", entry.metadata.estimated_savings),
                    ),
                ],
            );
        }
        for kernel in config
            .fusion_dispatch
            .synthesize_all(&config.operator_registry)
        {
            console::info(
                "fusion",
                "kernel_synthesized",
                &[
                    ("region", kernel.region.clone()),
                    ("lines", kernel.source.lines().count().to_string()),
                ],
            );
        }
    }

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
    ReloadPolicy::default_asset()
        .map(|policy| policy.watched_asset_paths)
        .unwrap_or_else(|error| {
            console::warn(
                "reload",
                "policy_unavailable",
                &[("error", error.to_string())],
            );
            Vec::new()
        })
}

fn maybe_hard_reset_after_blocks(
    consecutive_blocks: &mut usize,
    world: &mut TensorQuantaleWorld,
    accumulated_edges: &[quantale_semiring_v2::TensorEdge],
    current_payload: &mut Value,
    hard_reset_sleep: std::time::Duration,
    projection_bias: ProjectionBias,
    runtime_context: &RuntimeContext,
) {
    if *consecutive_blocks == 0 {
        return;
    }

    // Hard reset: re-embed the full accumulated edge set (static topology +
    // all LLM-proposed edges collected since startup) so learned dev-chain
    // weights survive cycle restarts.  reset() zeros the tensor to BOTTOM and
    // re-initialises the witness via embed, which is required for close() to
    // build a correct transitive closure.
    console::warn(
        "runtime",
        "hard_reset",
        &[("consecutive_blocks", consecutive_blocks.to_string())],
    );
    if let Err(error) = world.reset() {
        console::warn(
            "runtime",
            "hard_reset_world_reset_failed",
            &[("error", error.to_string())],
        );
    }
    if let Err(error) = world.embed_tensor_edges(accumulated_edges) {
        console::warn(
            "runtime",
            "hard_reset_embed_failed",
            &[("error", error.to_string())],
        );
    }
    if let Err(error) = world.close() {
        console::warn(
            "runtime",
            "hard_reset_close_failed",
            &[("error", error.to_string())],
        );
    }
    *current_payload = runtime_context.reset_payload();
    *consecutive_blocks = 0;
    std::thread::sleep(hard_reset_sleep);

    // Invariant 17: verify reset restored a valid frontier. Uses project
    // (read-only) so active[] is not advanced.
    if let Ok(post_reset) = world.project(projection_bias) {
        if post_reset.blocked != 0 {
            console::warn(
                "runtime",
                "hard_reset_frontier_invalid",
                &[("first_hop", post_reset.first_hop.to_string())],
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

fn run_mutations_cli(args: &[String]) -> i32 {
    let Some(command) = args.first().map(String::as_str) else {
        console::error(
            "mutations",
            "usage",
            &[(
                "command",
                "quantale_semiring_v2 mutations <list|apply> [mutation_id]".to_string(),
            )],
        );
        return 2;
    };

    let mut child_args = vec!["crates/operators_lib/apply_mutations.py".to_string()];
    match command {
        "list" => child_args.push("--list".to_string()),
        "apply" => {
            child_args.push("--apply".to_string());
            if let Some(mutation_id) = args.get(1) {
                child_args.push("--id".to_string());
                child_args.push(mutation_id.to_string());
            }
        }
        _ => {
            console::error(
                "mutations",
                "usage",
                &[(
                    "command",
                    "quantale_semiring_v2 mutations <list|apply> [mutation_id]".to_string(),
                )],
            );
            return 2;
        }
    }

    match Command::new("python3").args(&child_args).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(error) => {
            console::error(
                "mutations",
                "command_failed",
                &[("error", error.to_string())],
            );
            1
        }
    }
}
