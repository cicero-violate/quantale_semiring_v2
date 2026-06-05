#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use std::collections::BTreeSet;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use std::path::PathBuf;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use std::time::SystemTime;

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use serde_json::{json, Value};

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use quantale_semiring_v2::ReloadPolicy;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use quantale_semiring_v2::{
    action_label, compile_tensor_plan, format_quantale_value, runtime_check, ContractContext,
    ContractViolation, DecisionReport, ExecutionOutcome, ExplorationConfig, ExplorationEngine,
    Node, NodeContracts, ProcessReceipt, ProjectionBias, LAYER_CONFIDENCE,
};
#[cfg(all(feature = "cuda", not(feature = "legacy-cpu-orchestration")))]
use quantale_semiring_v2::{build_node_dispatch_kinds, DispatchKindSummary};
use quantale_semiring_v2::{
    check, check_with_operators, compile_and_emit_pattern_edges, compile_pattern, console,
    format_violations, load_compiled_pattern_edges, load_default_patterns,
    load_learned_tensor_edges, GraphTopology, LearningPolicy, RuntimeContext, SystemConfig,
    TensorQuantaleWorld, TlogWriter, TopologyInvariants, TopologyRuntime, UniversalExecutor,
    ViolationKind,
};

use topology_core::build_overlay_assets;

mod cli;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
mod runtime_dispatch;
mod runtime_epoch;
/// Legacy CPU par-dispatch compatibility module.
/// Available only with `--features legacy-cpu-orchestration`.
#[cfg(feature = "legacy-cpu-orchestration")]
mod runtime_parallel;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
mod runtime_reset;

#[cfg(feature = "legacy-cpu-orchestration")]
use runtime_dispatch::record_learning_edge_for_pair;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use runtime_dispatch::{
    apply_hot_dispatch_if_needed, execute_active_node_blocking, filter_static_topology_edges,
    node_name, queue_execution_lattice_updates, record_learning_edges,
    update_execution_receipt_priors, FusionLogicalAdvance,
};
use runtime_epoch::build_runtime_epoch;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use runtime_epoch::changed_asset_fingerprint;
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
use runtime_epoch::RuntimeEpoch;
#[cfg(feature = "legacy-cpu-orchestration")]
use runtime_parallel::{
    all_members_dispatched_on_device, device_dispatched_parallel_receipts,
    dispatch_gpu_parallel_group, route_parallel_receipt,
};
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
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
    let runtime_context = match RuntimeContext::default_asset() {
        Ok(context) => context,
        Err(error) => {
            fatal!("runtime", "load_runtime_context_failed", error);
        }
    };
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

    let current_payload = runtime_context.default_payload();

    #[cfg(all(feature = "cuda", not(feature = "legacy-cpu-orchestration")))]
    {
        let dispatch_kinds = build_node_dispatch_kinds(&epoch.topology.document, &config);
        let dispatch_summary = DispatchKindSummary::from_kinds(&dispatch_kinds);
        if let Err(error) = epoch.world.set_dispatch_kinds(&dispatch_kinds) {
            fatal!("gpu_native", "dispatch_kinds_upload_failed", error);
        }
        console::info(
            "gpu_native",
            "dispatch_kinds_uploaded",
            &[
                ("hf_device", dispatch_summary.hf_device.to_string()),
                (
                    "abstract_device",
                    dispatch_summary.abstract_device.to_string(),
                ),
                (
                    "external_process",
                    dispatch_summary.external_process.to_string(),
                ),
                ("external_io", dispatch_summary.external_io.to_string()),
                ("unsupported", dispatch_summary.unsupported.to_string()),
            ],
        );

        let registry = epoch.topology.registry();
        let node_name_table: Vec<String> = (0..registry.len())
            .map(|id| {
                registry
                    .name_of(id)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("node_{id}"))
            })
            .collect();
        let max_device_steps = if config.max_ticks == 0 {
            64
        } else {
            config.max_ticks.min(64) as u32
        };
        let max_total_steps = if config.max_ticks == 0 {
            0
        } else {
            config.max_ticks.min(u32::MAX as usize) as u32
        };

        match gpu_native_supervisor_loop(
            &mut epoch.world,
            max_device_steps,
            max_total_steps,
            |world| match quantale_semiring_v2::orch_service::service_external_commands(
                world,
                &epoch.executor,
                &node_name_table,
                &current_payload,
            ) {
                Ok(results) => {
                    if !results.is_empty() {
                        console::info(
                            "gpu_native",
                            "external_commands_serviced",
                            &[("count", results.len().to_string())],
                        );
                    }
                    if let Err(error) = world.drain_device_receipt_ext() {
                        console::warn(
                            "gpu_native",
                            "external_receipt_drain_failed",
                            &[("error", error.to_string())],
                        );
                    }
                }
                Err(error) => {
                    console::warn(
                        "gpu_native",
                        "external_command_service_failed",
                        &[("error", error.to_string())],
                    );
                }
            },
        ) {
            Ok(total_steps) => {
                console::info(
                    "gpu_native",
                    "supervisor_exit",
                    &[("total_steps", total_steps.to_string())],
                );
            }
            Err(error) => {
                fatal!("gpu_native", "supervisor_failed", error);
            }
        }

        if let Err(error) = epoch.learning_buffer.flush() {
            console::warn("learning", "shutdown_flush_failed", &[("error", error)]);
        }
        if let Err(error) = tlog.flush() {
            fatal!("tlog", "flush_failed", error);
        }
        return;
    }

    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    {
        let mut runtime_context = runtime_context;
        let mut runtime_invariants = match runtime_check::RuntimeInvariantPolicy::default_asset() {
            Ok(policy) => policy,
            Err(error) => {
                fatal!("runtime", "load_runtime_invariants_failed", error);
            }
        };
        let projection_bias = ProjectionBias::default();
        let sleep_dur = (config.tick_sleep_ms > 0)
            .then(|| std::time::Duration::from_millis(config.tick_sleep_ms));
        let mut current_payload = current_payload;
        let mut current_payload_origin: Option<String> = None;
        let mut tick: usize = 0;
        let mut consecutive_blocks: usize = 0;

        // Phase-0 orchestration tier counters.
        // These are only tracked when the legacy CPU par hot-path is active.
        #[cfg(feature = "legacy-cpu-orchestration")]
        let mut gpu_selected_groups: usize = 0;
        #[cfg(feature = "legacy-cpu-orchestration")]
        let mut gpu_device_only_groups: usize = 0;
        #[cfg(feature = "legacy-cpu-orchestration")]
        let mut host_fallback_groups: usize = 0;
        #[cfg(feature = "legacy-cpu-orchestration")]
        let mut device_ring_receipts: usize = 0;
        #[cfg(feature = "legacy-cpu-orchestration")]
        let mut cpu_queue_receipts: usize = 0;
        #[cfg(feature = "legacy-cpu-orchestration")]
        let external_io_commands: usize = 0;

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
                record_learning_edges(
                    &mut epoch.learning_buffer,
                    &epoch.topology,
                    &decision,
                    &execution,
                    &learning_policy,
                );
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
                                    if let Err(error) = epoch.world.embed_tensor_edges(&plan_edges)
                                    {
                                        fatal!("tensor", "embed_tensor_edges_failed", error);
                                    }
                                    epoch.accumulated_edges.extend(plan_edges.clone());
                                    if let Err(error) = tlog
                                        .append_tensor_edges("exploration:plan_tensor", &plan_edges)
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

            // ── Legacy CPU parallel tier (disabled by default) ───────────────────
            // This block is only active with `--features legacy-cpu-orchestration`.
            // The GPU-native orchestration path (`gpu_native_supervisor_loop`) is
            // the default; this block is kept for compatibility and debugging.
            #[cfg(feature = "legacy-cpu-orchestration")]
            let parallel_committed = 'par_tier: {
                let Some(data) = &epoch.par_group_data else {
                    break 'par_tier false;
                };
                let step = match epoch.world.par_group_step(data, projection_bias) {
                    Err(error) => {
                        console::warn(
                            "parallel",
                            "gpu_step_error",
                            &[("error", error.to_string())],
                        );
                        break 'par_tier false;
                    }
                    Ok(None) => break 'par_tier false,
                    Ok(Some(result)) => result,
                };
                let (
                    group_idx,
                    par_decisions,
                    par_member_region_ids,
                    par_dispatched_on_device,
                    par_dispatch_descriptors,
                ) = step;

                let Some(par_plan) = epoch.par_group_host_plans.get(group_idx) else {
                    console::warn(
                        "parallel",
                        "missing_host_plan",
                        &[("group_idx", group_idx.to_string())],
                    );
                    break 'par_tier false;
                };
                let par_names = &par_plan.node_names;

                gpu_selected_groups += 1;

                // Members with dispatched_on_device == 1 ran their operator in-kernel (H_f path).
                // Fully device-dispatched groups bypass fusion lookup and the host
                // scheduler entirely; the CPU only logs and drains the device ring.
                let all_on_device = all_members_dispatched_on_device(
                    par_names,
                    &par_dispatched_on_device,
                    &par_dispatch_descriptors,
                );
                if all_on_device {
                    gpu_device_only_groups += 1;
                } else {
                    host_fallback_groups += 1;
                }
                let par_receipts = if all_on_device {
                    device_dispatched_parallel_receipts(par_names)
                } else {
                    let fusion_entries: Vec<Option<&quantale_semiring_v2::FusionEntry>> =
                        par_plan.fusion_entries.iter().map(Option::as_ref).collect();
                    dispatch_gpu_parallel_group(
                        &epoch.executor,
                        &fusion_entries,
                        par_names,
                        &current_payload,
                        &par_dispatched_on_device,
                        &par_dispatch_descriptors,
                    )
                };

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
                // Tracks which receipt sinks need draining after the par group.
                let mut any_device_ring = false;
                let mut any_lattice_queue = false;
                for (
                    ((((decision, receipt), par_node_name), &kernel_region_id), &on_device),
                    descriptor,
                ) in par_decisions
                    .iter()
                    .zip(par_receipts.iter())
                    .zip(par_names.iter())
                    .zip(par_member_region_ids.iter())
                    .zip(par_dispatched_on_device.iter())
                    .zip(par_dispatch_descriptors.iter())
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

                    let receipt_route = route_parallel_receipt(
                        &mut epoch.world,
                        par_node_name,
                        decision,
                        kernel_region_id,
                        on_device,
                        descriptor,
                        par_outcome,
                        receipt,
                    );
                    any_device_ring |= receipt_route.via_device_ring;
                    any_lattice_queue |= receipt_route.queued_lattice;
                    if receipt_route.via_device_ring {
                        device_ring_receipts += 1;
                    } else if receipt_route.queued_lattice {
                        cpu_queue_receipts += 1;
                    }
                    epoch
                        .exploration_engine
                        .update_receipt_prior(decision.first_hop, receipt);
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
                                                fatal!(
                                                    "tensor",
                                                    "embed_tensor_edges_failed",
                                                    error
                                                );
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
                                        any_lattice_queue = true;
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
                // Drain device ring for hot-region receipts first, then CPU queue for the rest.
                if any_device_ring {
                    if let Err(error) = epoch.world.drain_device_receipts() {
                        console::warn(
                            "parallel",
                            "drain_device_receipts_failed",
                            &[("error", error.to_string())],
                        );
                    }
                }
                if any_lattice_queue {
                    if let Err(error) = epoch.world.drain_lattice_queue() {
                        fatal!("tensor", "drain_lattice_queue_failed", error);
                    }
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
            #[cfg(feature = "legacy-cpu-orchestration")]
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

            let Some(active_node) = Node::decode(decision.first_hop, epoch.topology.registry())
            else {
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
            record_learning_edges(
                &mut epoch.learning_buffer,
                &epoch.topology,
                &decision,
                &execution,
                &learning_policy,
            );
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

        #[cfg(feature = "legacy-cpu-orchestration")]
        console::info(
            "orch_counters",
            "shutdown",
            &[
                ("gpu_selected_groups", gpu_selected_groups.to_string()),
                ("gpu_device_only_groups", gpu_device_only_groups.to_string()),
                ("host_fallback_groups", host_fallback_groups.to_string()),
                ("device_ring_receipts", device_ring_receipts.to_string()),
                ("cpu_queue_receipts", cpu_queue_receipts.to_string()),
                ("external_io_commands", external_io_commands.to_string()),
            ],
        );

        if let Err(error) = epoch.learning_buffer.flush() {
            console::warn("learning", "shutdown_flush_failed", &[("error", error)]);
        }

        if let Err(error) = tlog.flush() {
            fatal!("tlog", "flush_failed", error);
        }
    }
}

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
fn contract_violation_receipt(node_name: &str, violation: &ContractViolation) -> ProcessReceipt {
    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 125,
        stdout_payload: String::new(),
        stderr_payload: format!("contract violation: {}", violation.reason),
    }
}

/// Phase-7: GPU-native supervisor loop.
///
/// Runs the GPU orchestration engine in multi-step bursts.  The CPU is not in
/// the per-step hot path: it only services external commands when the GPU
/// yields `ORCH_WAIT_EXTERNAL`, persists learned deltas on each burst
/// boundary, and handles terminal conditions.
///
/// `max_device_steps` caps each GPU burst so the CPU can drain the learned-
/// delta ring and service external commands between bursts.
/// `service_fn` is called with the world when `ORCH_WAIT_EXTERNAL` is
/// returned; it should drain and service the device command ring.
///
/// Returns the device-reported step count before halting or an error.
#[cfg(feature = "cuda")]
pub fn gpu_native_supervisor_loop(
    world: &mut TensorQuantaleWorld,
    max_device_steps: u32,
    max_total_steps: u32,
    mut service_fn: impl FnMut(&mut TensorQuantaleWorld),
) -> Result<u32, quantale_semiring_v2::CudaError> {
    use quantale_semiring_v2::OrchStepStatus;

    let mut total_steps = 0u32;

    loop {
        if max_total_steps > 0 && total_steps >= max_total_steps {
            break;
        }

        let previous_steps = total_steps;
        let status = world.orchestrate_until_wait_or_halt(max_device_steps)?;
        let state = world.orch_state_snapshot()?;
        total_steps = u32::try_from(state.step.max(0)).unwrap_or(u32::MAX);
        console::info(
            "gpu_native",
            "burst_complete",
            &[
                ("status", format!("{status:?}")),
                ("step", state.step.to_string()),
                ("halted", state.halted.to_string()),
                ("blocked", state.blocked.to_string()),
                ("selected_group", state.selected_group.to_string()),
                ("selected_node", state.selected_node.to_string()),
                ("selected_src", state.selected_src.to_string()),
                ("selected_dst", state.selected_dst.to_string()),
                (
                    "pending_external_count",
                    state.pending_external_count.to_string(),
                ),
                (
                    "pending_receipt_count",
                    state.pending_receipt_count.to_string(),
                ),
                ("failure_count", state.failure_count.to_string()),
                ("consecutive_blocks", state.consecutive_blocks.to_string()),
                (
                    "hard_reset_requested",
                    state.hard_reset_requested.to_string(),
                ),
            ],
        );

        match status {
            OrchStepStatus::Continue => {
                // Burst exhausted or blocked — persist learned deltas and yield.
                let _ = world.learned_delta_apply();
                let _ = world.export_receipt_priors();
                if state.blocked != 0 {
                    console::warn(
                        "gpu_native",
                        "blocked",
                        &[
                            ("step", total_steps.to_string()),
                            ("selected_node", state.selected_node.to_string()),
                            ("selected_src", state.selected_src.to_string()),
                            ("selected_dst", state.selected_dst.to_string()),
                        ],
                    );
                    break;
                }
                if total_steps == previous_steps {
                    console::warn(
                        "gpu_native",
                        "no_progress",
                        &[("step", total_steps.to_string())],
                    );
                    break;
                }
                continue;
            }
            OrchStepStatus::WaitExternal => {
                // Delegate external command servicing to the caller-provided fn.
                service_fn(world);
            }
            OrchStepStatus::Halted => {
                console::info("gpu_native", "halted", &[("step", total_steps.to_string())]);
                break;
            }
            OrchStepStatus::Error => {
                console::warn(
                    "gpu_native",
                    "scheduler_error",
                    &[("step", total_steps.to_string())],
                );
                break;
            }
        }
    }

    Ok(total_steps)
}
