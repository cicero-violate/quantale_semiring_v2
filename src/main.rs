use quantale_semiring_v2::{
    DispatchKindSummary, GraphTopology, LearningPolicy, RuntimeContext, SystemConfig,
    TensorQuantaleWorld, TlogWriter, TopologyInvariants, TopologyRuntime, UniversalExecutor,
    ViolationKind, build_node_dispatch_kinds, build_node_reentrant_mask, check,
    check_with_operators, compile_and_emit_pattern_edges, compile_pattern, console,
    format_violations, load_compiled_pattern_edges, load_default_patterns,
    load_learned_tensor_edges,
};

use topology_core::build_overlay_assets;

mod cli;
mod runtime_epoch;

use runtime_epoch::build_runtime_epoch;

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

    let dispatch_kinds = build_node_dispatch_kinds(&epoch.topology.document, &config);
    let reentrant_mask = build_node_reentrant_mask(&epoch.topology.document);
    let dispatch_summary = DispatchKindSummary::from_kinds(&dispatch_kinds);
    if let Err(error) = epoch.world.set_dispatch_kinds(&dispatch_kinds) {
        fatal!("gpu_native", "dispatch_kinds_upload_failed", error);
    }
    if let Err(error) = epoch.world.set_reentrant_mask(&reentrant_mask) {
        fatal!("gpu_native", "reentrant_mask_upload_failed", error);
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
