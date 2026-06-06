use quantale_semiring_v2::console;
#[cfg(feature = "cuda")]
use quantale_semiring_v2::{
    DispatchKindSummary, LearningPolicy, RuntimeContext, SystemConfig, TlogWriter,
    build_node_dispatch_kinds, build_node_reentrant_mask,
};

mod app;

use app::{CliCommand, handle};
#[cfg(feature = "cuda")]
use app::{build_runtime_epoch, gpu_native_supervisor_loop};

#[cfg(feature = "cuda")]
macro_rules! fatal {
    ($scope:expr, $message:expr, $error:expr) => {{
        console::error($scope, $message, &[("error", $error.to_string())]);
        std::process::exit(1);
    }};
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if let CliCommand::Exit(code) = handle(&args) {
        std::process::exit(code);
    }
    run_runtime();
}

#[cfg(not(feature = "cuda"))]
fn run_runtime() {
    console::error(
        "runtime",
        "cuda_feature_required",
        &[(
            "hint",
            "run with --features cuda or use the default feature set".to_string(),
        )],
    );
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn run_runtime() {
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
