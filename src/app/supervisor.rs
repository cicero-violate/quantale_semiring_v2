use quantale_semiring_v2::{CudaError, OrchStepStatus, TensorQuantaleWorld, console};

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
) -> Result<u32, CudaError> {
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
