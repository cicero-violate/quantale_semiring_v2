use quantale_semiring_v2::{FusionEntry, ProcessReceipt, UniversalExecutor};
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
    dispatched_on_device: &[i32],
) -> Vec<ProcessReceipt> {
    std::thread::scope(|scope| {
        let mut receipts: Vec<Option<ProcessReceipt>> = vec![None; node_names.len()];
        let mut handles = Vec::new();

        for (idx, name) in node_names.iter().enumerate() {
            if dispatched_on_device.get(idx).copied().unwrap_or(0) != 0 {
                receipts[idx] = Some(device_dispatched_receipt(name));
                continue;
            }

            let entry = fusion_entries.get(idx).copied().flatten();
            handles.push((
                idx,
                scope.spawn(move || {
                    if let Some(entry) = entry {
                        executor.execute_fusion_entry_blocking(entry, current_payload)
                    } else {
                        executor.execute_abstract_node_blocking(name.as_str(), current_payload)
                    }
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
