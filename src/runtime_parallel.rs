use quantale_semiring_v2::{
    FusionEntry, PAR_DISPATCH_FUSION_ENTRY, PAR_DISPATCH_HF_DEVICE, ParDispatchDescriptor,
    ProcessReceipt, UniversalExecutor,
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
    std::thread::scope(|scope| {
        let mut receipts: Vec<Option<ProcessReceipt>> = vec![None; node_names.len()];
        let mut handles = Vec::new();

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
            handles.push((
                idx,
                scope.spawn(move || match dispatch_kind {
                    PAR_DISPATCH_FUSION_ENTRY => match entry {
                        Some(entry) => {
                            executor.execute_fusion_entry_blocking(entry, current_payload)
                        }
                        None => missing_fusion_receipt(name),
                    },
                    _ => executor.execute_abstract_node_blocking(name.as_str(), current_payload),
                }),
            ));
        }

        for (idx, handle) in handles {
            receipts[idx] = Some(handle.join().expect("parallel dispatch worker panicked"));
        }

        receipts
            .into_iter()
            .map(|receipt| receipt.expect("parallel receipt missing"))
            .collect()
    })
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
    use quantale_semiring_v2::{PAR_DISPATCH_FUSION_ENTRY, PAR_DISPATCH_HOST_FALLBACK};
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
