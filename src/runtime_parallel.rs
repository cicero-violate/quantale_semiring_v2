use serde_json::Value;
use quantale_semiring_v2::{FusionEntry, ProcessReceipt, UniversalExecutor};

/// Dispatch operators for a GPU-committed par group concurrently.
///
/// Called after `TensorQuantaleWorld::par_group_step` commits the group.
/// Routes each member through fusion dispatch first, then falls back to
/// `execute_abstract_node_blocking`.
pub(super) fn dispatch_gpu_parallel_group(
    executor: &UniversalExecutor,
    fusion_entries: &[Option<&FusionEntry>],
    node_names: &[String],
    current_payload: &Value,
) -> Vec<ProcessReceipt> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = node_names
            .iter()
            .zip(fusion_entries.iter())
            .map(|(name, entry)| {
                scope.spawn(move || {
                    if let Some(entry) = entry {
                        executor.execute_fusion_entry_blocking(entry, current_payload)
                    } else {
                        executor.execute_abstract_node_blocking(name.as_str(), current_payload)
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("parallel dispatch worker panicked"))
            .collect()
    })
}
