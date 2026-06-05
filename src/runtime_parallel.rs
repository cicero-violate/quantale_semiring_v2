use quantale_semiring_v2::{
    DecisionReport, ExecutionOutcome, FusionEntry, PAR_DISPATCH_FUSION_ENTRY,
    PAR_DISPATCH_HF_DEVICE, ParDispatchDescriptor, ProcessReceipt, TensorQuantaleWorld,
    UniversalExecutor, console,
};
use serde_json::{Value, json};

/// Dispatch operators for a GPU-committed par group concurrently.
///
/// Called after `TensorQuantaleWorld::par_group_step` commits the group.
/// Routes each member through fusion dispatch first, then falls back to
/// `execute_abstract_node_blocking`. Members already dispatched by the
/// par-group kernel receive a synthetic success receipt and are not executed
/// again on the host.
pub(super) fn dispatch_gpu_parallel_group(
    executor: &UniversalExecutor,
    fusion_entries: &[Option<&FusionEntry>],
    node_names: &[String],
    current_payload: &Value,
    dispatch_descriptors: &[ParDispatchDescriptor],
) -> Vec<ProcessReceipt> {
    if all_members_dispatched_on_device(node_names, dispatch_descriptors) {
        return node_names
            .iter()
            .map(|name| device_dispatched_receipt(name))
            .collect();
    }

    let mut receipts: Vec<Option<ProcessReceipt>> = vec![None; node_names.len()];
    let mut fusion_jobs = Vec::new();
    let mut host_jobs = Vec::new();

    for (idx, name) in node_names.iter().enumerate() {
        if dispatch_descriptors
            .get(idx)
            .map(|descriptor| descriptor.dispatch_kind == PAR_DISPATCH_HF_DEVICE)
            .unwrap_or(false)
        {
            receipts[idx] = Some(device_dispatched_receipt(name));
            continue;
        }

        let entry = fusion_entries.get(idx).copied().flatten();
        let dispatch_kind = dispatch_descriptors
            .get(idx)
            .map(|descriptor| descriptor.dispatch_kind)
            .unwrap_or_default();
        match dispatch_kind {
            PAR_DISPATCH_FUSION_ENTRY => match entry {
                Some(entry) => fusion_jobs.push((idx, entry)),
                None => receipts[idx] = Some(missing_fusion_receipt(name)),
            },
            _ => host_jobs.push((idx, name.as_str())),
        }
    }

    if host_jobs.is_empty() {
        for (idx, receipt) in
            executor.execute_fusion_entries_batch_blocking(&fusion_jobs, current_payload)
        {
            receipts[idx] = Some(receipt);
        }
        return collect_parallel_receipts(receipts);
    }

    std::thread::scope(|scope| {
        let host_handles: Vec<_> = host_jobs
            .into_iter()
            .map(|(idx, name)| {
                (
                    idx,
                    scope.spawn(move || {
                        executor.execute_abstract_node_blocking(name, current_payload)
                    }),
                )
            })
            .collect();

        let fusion_handle = if fusion_jobs.is_empty() {
            None
        } else {
            Some(scope.spawn(move || {
                executor.execute_fusion_entries_batch_blocking(&fusion_jobs, current_payload)
            }))
        };

        if let Some(handle) = fusion_handle {
            for (idx, receipt) in handle
                .join()
                .expect("parallel fusion batch worker panicked")
            {
                receipts[idx] = Some(receipt);
            }
        }

        for (idx, handle) in host_handles {
            receipts[idx] = Some(handle.join().expect("parallel dispatch worker panicked"));
        }
    });

    collect_parallel_receipts(receipts)
}

fn collect_parallel_receipts(receipts: Vec<Option<ProcessReceipt>>) -> Vec<ProcessReceipt> {
    receipts
        .into_iter()
        .map(|receipt| receipt.expect("parallel receipt missing"))
        .collect()
}

fn all_members_dispatched_on_device(
    node_names: &[String],
    dispatch_descriptors: &[ParDispatchDescriptor],
) -> bool {
    !node_names.is_empty()
        && dispatch_descriptors.len() >= node_names.len()
        && dispatch_descriptors
            .iter()
            .take(node_names.len())
            .all(|descriptor| descriptor.dispatch_kind == PAR_DISPATCH_HF_DEVICE)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ParallelReceiptRoute {
    pub via_device_ring: bool,
    pub queued_lattice: bool,
}

impl ParallelReceiptRoute {
    fn device_ring() -> Self {
        Self {
            via_device_ring: true,
            queued_lattice: false,
        }
    }

    fn lattice_queue() -> Self {
        Self {
            via_device_ring: false,
            queued_lattice: true,
        }
    }
}

/// Route one committed par-member receipt through the same tensor-update path
/// used before the par receipt router was extracted from `main.rs`.
pub(super) fn route_parallel_receipt(
    world: &mut TensorQuantaleWorld,
    node_name: &str,
    decision: &DecisionReport,
    kernel_region_id: i32,
    descriptor: &ParDispatchDescriptor,
    outcome: ExecutionOutcome,
    receipt: &ProcessReceipt,
) -> ParallelReceiptRoute {
    // H_f path (on_device == 1): the par kernel already wrote the receipt to the
    // device ring. No separate gpu_dispatch_region call is needed.
    if descriptor.dispatch_kind == PAR_DISPATCH_HF_DEVICE {
        console::info(
            "parallel",
            "hf_dispatch_receipt",
            &[
                ("node", node_name.to_string()),
                ("region_id", kernel_region_id.to_string()),
            ],
        );
        return ParallelReceiptRoute::device_ring();
    }

    // Successful fusion batches are host-launched today, but their tensor update
    // is already routed through the device ring.
    if descriptor.dispatch_kind == PAR_DISPATCH_FUSION_ENTRY
        && receipt.exit_code == 0
        && receipt
            .stdout_payload
            .contains("\"kernel\":\"jit_fused_batch\"")
    {
        return match world.push_device_receipt(
            -1,
            decision.selected_src,
            decision.first_hop,
            outcome.code(),
        ) {
            Ok(()) => {
                console::info(
                    "parallel",
                    "fusion_batch_device_receipt",
                    &[("node", node_name.to_string())],
                );
                ParallelReceiptRoute::device_ring()
            }
            Err(error) => {
                console::warn(
                    "parallel",
                    "fusion_batch_device_fallback",
                    &[
                        ("node", node_name.to_string()),
                        ("error", error.to_string()),
                    ],
                );
                world.queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
                ParallelReceiptRoute::lattice_queue()
            }
        };
    }

    // Hot-region CPU path: the CPU ran the operator, but the tensor receipt can
    // still be folded on the GPU via the hot-region mailbox.
    if kernel_region_id >= 0 {
        return match world.gpu_dispatch_region(
            kernel_region_id,
            decision.selected_src,
            decision.first_hop,
            outcome.code(),
        ) {
            Ok(()) => {
                console::info(
                    "parallel",
                    "device_ring_receipt",
                    &[
                        ("node", node_name.to_string()),
                        ("region_id", kernel_region_id.to_string()),
                    ],
                );
                ParallelReceiptRoute::device_ring()
            }
            Err(error) => {
                console::warn(
                    "parallel",
                    "device_ring_fallback",
                    &[
                        ("node", node_name.to_string()),
                        ("error", error.to_string()),
                    ],
                );
                world.queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
                ParallelReceiptRoute::lattice_queue()
            }
        };
    }

    // Default path: CPU queue -> drain_lattice_queue.
    world.queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
    ParallelReceiptRoute::lattice_queue()
}

fn missing_fusion_receipt(node_name: &str) -> ProcessReceipt {
    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 1,
        stdout_payload: String::new(),
        stderr_payload:
            "GPU par descriptor requested fusion dispatch, but no fusion entry was loaded"
                .to_string(),
    }
}

fn device_dispatched_receipt(node_name: &str) -> ProcessReceipt {
    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 0,
        stdout_payload: json!({
            "node": node_name,
            "dispatch": "device",
            "kernel": "tensor_quantale_par_group_step",
        })
        .to_string(),
        stderr_payload: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantale_semiring_v2::{
        PAR_DISPATCH_ABSTRACT_DEVICE, PAR_DISPATCH_FUSION_ENTRY, PAR_DISPATCH_HOST_FALLBACK,
    };
    use std::collections::HashMap;

    #[test]
    fn device_dispatched_members_do_not_run_host_executor() {
        let executor = UniversalExecutor::new(HashMap::from([(
            "Would::FailOnHost".to_string(),
            json!({
                "executable": "/definitely/not/a/real/binary"
            }),
        )]));
        let names = vec!["Would::FailOnHost".to_string()];
        let descriptors = [ParDispatchDescriptor {
            member_index: 0,
            node_id: 0,
            region_id: 3,
            dispatch_kind: PAR_DISPATCH_HF_DEVICE,
            src_node: 1,
            dst_node: 2,
        }];
        let receipts =
            dispatch_gpu_parallel_group(&executor, &[None], &names, &Value::Null, &descriptors);

        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].node_name, "Would::FailOnHost");
        assert_eq!(receipts[0].exit_code, 0);
        assert!(receipts[0].stderr_payload.is_empty());
        let stdout: Value = serde_json::from_str(&receipts[0].stdout_payload).unwrap();
        assert_eq!(stdout["dispatch"], "device");
    }

    #[test]
    fn all_device_dispatched_group_returns_before_host_scope_work() {
        let executor = UniversalExecutor::new(HashMap::from([
            (
                "Would::FailOnHostA".to_string(),
                json!({ "executable": "/definitely/not/a/real/binary" }),
            ),
            (
                "Would::FailOnHostB".to_string(),
                json!({ "executable": "/definitely/not/a/real/binary" }),
            ),
        ]));
        let names = vec![
            "Would::FailOnHostA".to_string(),
            "Would::FailOnHostB".to_string(),
        ];
        let descriptors = [
            ParDispatchDescriptor {
                member_index: 0,
                node_id: 0,
                region_id: 3,
                dispatch_kind: PAR_DISPATCH_HF_DEVICE,
                src_node: 1,
                dst_node: 2,
            },
            ParDispatchDescriptor {
                member_index: 1,
                node_id: 1,
                region_id: 4,
                dispatch_kind: PAR_DISPATCH_HF_DEVICE,
                src_node: 1,
                dst_node: 3,
            },
        ];

        let receipts = dispatch_gpu_parallel_group(
            &executor,
            &[None, None],
            &names,
            &Value::Null,
            &descriptors,
        );

        assert_eq!(receipts.len(), 2);
        assert!(receipts.iter().all(|receipt| receipt.exit_code == 0));
        assert!(
            receipts
                .iter()
                .all(|receipt| receipt.stderr_payload.is_empty())
        );
        assert!(receipts.iter().all(|receipt| {
            serde_json::from_str::<Value>(&receipt.stdout_payload)
                .map(|stdout| stdout["dispatch"] == "device")
                .unwrap_or(false)
        }));
    }

    #[test]
    fn all_device_fast_path_requires_all_members_to_be_hf_device() {
        let names = vec!["A".to_string(), "B".to_string()];
        let descriptors = [
            ParDispatchDescriptor {
                member_index: 0,
                node_id: 0,
                region_id: 3,
                dispatch_kind: PAR_DISPATCH_HF_DEVICE,
                src_node: 1,
                dst_node: 2,
            },
            ParDispatchDescriptor {
                member_index: 1,
                node_id: 1,
                region_id: -1,
                dispatch_kind: PAR_DISPATCH_HOST_FALLBACK,
                src_node: 1,
                dst_node: 3,
            },
        ];

        assert!(!all_members_dispatched_on_device(&names, &descriptors));
    }

    #[test]
    fn host_fallback_descriptors_run_host_executor() {
        let executor = UniversalExecutor::new(HashMap::from([(
            "Would::FailOnHost".to_string(),
            json!({
                "executable": "/definitely/not/a/real/binary"
            }),
        )]));
        let names = vec!["Would::FailOnHost".to_string()];
        let descriptors = [ParDispatchDescriptor {
            member_index: 0,
            node_id: 0,
            region_id: -1,
            dispatch_kind: PAR_DISPATCH_HOST_FALLBACK,
            src_node: 1,
            dst_node: 2,
        }];
        let receipts =
            dispatch_gpu_parallel_group(&executor, &[None], &names, &Value::Null, &descriptors);

        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].node_name, "Would::FailOnHost");
        assert_ne!(receipts[0].exit_code, 0);
        assert!(
            receipts[0]
                .stderr_payload
                .contains("Failed to spawn process")
        );
    }

    #[test]
    fn abstract_device_scaffold_still_uses_host_until_device_lowering_exists() {
        let executor = UniversalExecutor::new(HashMap::from([(
            "Abstract::NotYetLowered".to_string(),
            json!({
                "executable": "/definitely/not/a/real/binary"
            }),
        )]));
        let names = vec!["Abstract::NotYetLowered".to_string()];
        let descriptors = [ParDispatchDescriptor {
            member_index: 0,
            node_id: 0,
            region_id: -1,
            dispatch_kind: PAR_DISPATCH_ABSTRACT_DEVICE,
            src_node: 1,
            dst_node: 2,
        }];
        let receipts =
            dispatch_gpu_parallel_group(&executor, &[None], &names, &Value::Null, &descriptors);

        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].node_name, "Abstract::NotYetLowered");
        assert_ne!(receipts[0].exit_code, 0);
        assert!(
            receipts[0]
                .stderr_payload
                .contains("Failed to spawn process")
        );
    }

    #[test]
    fn fusion_only_group_does_not_need_host_fallback_scope() {
        let executor = UniversalExecutor::new(HashMap::new());
        let names = vec!["Fusion::Missing".to_string()];
        let descriptors = [ParDispatchDescriptor {
            member_index: 0,
            node_id: 0,
            region_id: -1,
            dispatch_kind: PAR_DISPATCH_FUSION_ENTRY,
            src_node: 1,
            dst_node: 2,
        }];

        let receipts =
            dispatch_gpu_parallel_group(&executor, &[None], &names, &Value::Null, &descriptors);

        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].exit_code, 1);
        assert!(
            receipts[0]
                .stderr_payload
                .contains("requested fusion dispatch")
        );
    }

    #[test]
    fn fusion_descriptor_requires_loaded_fusion_entry() {
        let executor = UniversalExecutor::new(HashMap::new());
        let names = vec!["Fusion::Entry".to_string()];
        let descriptors = [ParDispatchDescriptor {
            member_index: 0,
            node_id: 0,
            region_id: -1,
            dispatch_kind: PAR_DISPATCH_FUSION_ENTRY,
            src_node: 1,
            dst_node: 2,
        }];
        let receipts =
            dispatch_gpu_parallel_group(&executor, &[None], &names, &Value::Null, &descriptors);

        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].node_name, "Fusion::Entry");
        assert_eq!(receipts[0].exit_code, 1);
        assert!(
            receipts[0]
                .stderr_payload
                .contains("requested fusion dispatch")
        );
    }
}
