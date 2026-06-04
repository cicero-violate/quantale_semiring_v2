use serde_json::Value;
use quantale_semiring_v2::{
    CudaError, DecisionReport, NodeRegistry, OperatorRegistry, ProcessReceipt, ProjectionBias,
    TensorQuantaleWorld, UniversalExecutor, operator_effects_for_node, safe_parallel,
};

/// Result of a committed and dispatched parallel group.
pub(super) struct ParallelGroupOutcome {
    /// One decision per group member, in group order (from project_parallel_group).
    pub decisions: Vec<DecisionReport>,
    /// One receipt per group member, in group order (from concurrent dispatch).
    pub receipts: Vec<ProcessReceipt>,
    /// Resolved names for each member, in group order.
    pub node_names: Vec<String>,
}

/// Project, validate effect independence, commit atomically, and dispatch
/// concurrently for a single CKA `par` group.
///
/// Returns `Ok(None)` when the group is not ready (any member blocked/halted,
/// unknown node, or effects conflict).
/// Returns `Ok(Some(outcome))` after a successful GPU commit + concurrent
/// operator dispatch.
/// Returns `Err` on a tensor error (e.g. CUDA device fault).
pub(super) fn try_dispatch_parallel_group(
    world: &mut TensorQuantaleWorld,
    executor: &UniversalExecutor,
    operator_registry: &OperatorRegistry,
    group: &[i32],
    registry: &NodeRegistry,
    bias: ProjectionBias,
    current_payload: &Value,
) -> Result<Option<ParallelGroupOutcome>, CudaError> {
    // Read-only GPU projection — does not advance the frontier.
    let decisions = world.project_parallel_group(group, bias)?;

    // Skip if any member is blocked or halted.
    if decisions.iter().any(|d| d.blocked != 0 || d.halted != 0) {
        return Ok(None);
    }

    // Resolve node IDs to names in group order.
    let mut node_names: Vec<String> = Vec::with_capacity(group.len());
    for &id in group {
        let Some(name) = registry.name_of(id as usize) else {
            return Ok(None);
        };
        node_names.push(name.to_string());
    }

    // Validate pairwise effect independence.
    let mut effects = Vec::with_capacity(node_names.len());
    for name in &node_names {
        match operator_effects_for_node(name, operator_registry) {
            Ok(fx) => effects.push(fx),
            Err(_) => return Ok(None),
        }
    }
    for left in 0..effects.len() {
        for right in (left + 1)..effects.len() {
            if !safe_parallel(&effects[left], &effects[right]) {
                return Ok(None);
            }
        }
    }

    // Atomic GPU commit — advances consumed/active for all members at once.
    world.commit_decision_batch(&decisions)?;

    // Concurrent operator dispatch.
    let receipts: Vec<ProcessReceipt> = std::thread::scope(|scope| {
        let handles: Vec<_> = node_names
            .iter()
            .map(|name| {
                scope.spawn(move || {
                    executor.execute_abstract_node_blocking(name.as_str(), current_payload)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("parallel dispatch worker panicked"))
            .collect()
    });

    Ok(Some(ParallelGroupOutcome {
        decisions,
        receipts,
        node_names,
    }))
}
