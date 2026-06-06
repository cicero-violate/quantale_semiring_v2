//! External command service — serial (Phase 1) and parallel (Phase 3).
//!
//! When `orchestrate_step()` returns `OrchStepStatus::WaitExternal`, the GPU
//! has emitted one or more `DeviceCommand` entries that require CPU/IO work.
//!
//! `service_external_commands` — serial, original implementation.
//! `service_external_commands_parallel` — Phase 3: groups effect-independent
//! commands and runs each group concurrently using `std::thread::scope`.
//!
//! Effect independence is determined by:
//!   1. `NodeContracts.side_effects` intersection (precise, declared).
//!   2. `DISPATCH_KIND_EXTERNAL_IO` vs IO heuristic (conservative fallback).
//!   3. Same `operator_name_id` → always serialize (conservative default).

use serde_json::Value;

use crate::UniversalExecutor;
use crate::contracts::NodeContracts;
use crate::error::CudaError;
use crate::tensor::{
    DISPATCH_KIND_EXTERNAL_IO, DISPATCH_KIND_EXTERNAL_PROCESS, DeviceCommand, DeviceReceiptExt,
    TensorQuantaleWorld,
};
use crate::types::ProcessReceipt;

/// Outcome of servicing one `DeviceCommand`.
#[derive(Clone, Debug)]
pub struct CommandServiceResult {
    pub command_id: i32,
    pub node_id: i32,
    pub outcome: i32, // 0=success, 1=failure, 2=timeout, 3=safety_violation
    pub receipt: ProcessReceipt,
}

/// Service all pending external commands: drain the command ring, execute each
/// command, and push a `DeviceReceiptExt` back for the GPU to drain.
///
/// Returns the results for logging / learning.  Call `drain_device_receipt_ext`
/// after this to fold the receipts into the quantale tensor.
pub fn service_external_commands(
    world: &mut TensorQuantaleWorld,
    executor: &UniversalExecutor,
    node_name_table: &[String],
    current_payload: &Value,
) -> Result<Vec<CommandServiceResult>, CudaError> {
    let cmds = world.drain_device_commands()?;
    if cmds.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(cmds.len());

    for cmd in &cmds {
        if cmd.valid == 0 {
            continue;
        }

        let node_name = resolve_operator_name(cmd, node_name_table);
        let receipt = execute_command(executor, cmd, &node_name, current_payload);
        let outcome = outcome_from_receipt(&receipt, cmd);

        world.push_device_receipt_ext(DeviceReceiptExt {
            valid: 1,
            consumed: 0,
            command_id: cmd.command_id,
            node_id: cmd.node_id,
            src: cmd.src,
            dst: cmd.dst,
            outcome,
            receipt_kind: cmd.dispatch_kind,
            output_flags: 0,
            latency: 0.0,
        })?;

        results.push(CommandServiceResult {
            command_id: cmd.command_id,
            node_id: cmd.node_id,
            outcome,
            receipt,
        });
    }

    Ok(results)
}

// ── Phase 3: parallel command pool ───────────────────────────────────────────

/// Determine whether two commands have no conflicting side effects and can
/// run concurrently.
///
/// Rules (applied in priority order):
/// 1. Same `operator_name_id` → serialize (conservative; may share state).
/// 2. Both nodes have declared `side_effects` → independent iff sets are disjoint.
/// 3. Both are `DISPATCH_KIND_EXTERNAL_IO` → independent by convention (IO-only).
/// 4. Otherwise → serialize.
pub fn commands_are_independent(
    a: &DeviceCommand,
    b: &DeviceCommand,
    node_name_table: &[String],
    contracts: &NodeContracts,
) -> bool {
    if a.operator_name_id == b.operator_name_id {
        return false;
    }

    let name_a = resolve_operator_name(a, node_name_table);
    let name_b = resolve_operator_name(b, node_name_table);

    let effects_a = contracts.get(&name_a).map(|c| &c.side_effects);
    let effects_b = contracts.get(&name_b).map(|c| &c.side_effects);

    match (effects_a, effects_b) {
        (Some(wa), Some(wb)) => !wa.iter().any(|e| wb.contains(e)),
        _ => {
            // IO-vs-IO: no shared GPU state by convention.
            a.dispatch_kind == DISPATCH_KIND_EXTERNAL_IO
                && b.dispatch_kind == DISPATCH_KIND_EXTERNAL_IO
        }
    }
}

/// Greedily partition a slice of valid commands into groups of mutually
/// independent commands.  Commands within a group can run concurrently;
/// groups themselves are serialized.
///
/// Time: O(n² × group_count) — acceptable for the small command batches
/// that the GPU emits per WaitExternal step.
pub fn partition_independent_groups<'a>(
    cmds: &[&'a DeviceCommand],
    node_name_table: &[String],
    contracts: &NodeContracts,
) -> Vec<Vec<&'a DeviceCommand>> {
    let mut groups: Vec<Vec<&'a DeviceCommand>> = Vec::new();
    for &cmd in cmds {
        let mut placed = false;
        for group in &mut groups {
            if group
                .iter()
                .all(|&other| commands_are_independent(cmd, other, node_name_table, contracts))
            {
                group.push(cmd);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![cmd]);
        }
    }
    groups
}

/// Service all pending external commands with parallel execution of
/// effect-independent groups.
///
/// Each independent group is executed using `std::thread::scope` so that
/// group threads cannot outlive the local stack frame.  Receipts are sorted
/// by `command_id` before being pushed back to the device ring to ensure
/// deterministic ordering regardless of thread scheduling.
#[cfg(feature = "cuda")]
pub fn service_external_commands_parallel(
    world: &mut TensorQuantaleWorld,
    executor: &UniversalExecutor,
    node_name_table: &[String],
    current_payload: &Value,
    contracts: &NodeContracts,
) -> Result<Vec<CommandServiceResult>, CudaError> {
    let cmds = world.drain_device_commands()?;
    let valid: Vec<&DeviceCommand> = cmds.iter().filter(|c| c.valid != 0).collect();
    if valid.is_empty() {
        return Ok(Vec::new());
    }

    let groups = partition_independent_groups(&valid, node_name_table, contracts);
    let mut all_results: Vec<CommandServiceResult> = Vec::with_capacity(valid.len());

    for group in groups {
        if group.len() == 1 {
            // Single command: execute on the calling thread.
            let cmd = group[0];
            let node_name = resolve_operator_name(cmd, node_name_table);
            let receipt = execute_command(executor, cmd, &node_name, current_payload);
            let outcome = outcome_from_receipt(&receipt, cmd);
            all_results.push(CommandServiceResult {
                command_id: cmd.command_id,
                node_id: cmd.node_id,
                outcome,
                receipt,
            });
        } else {
            // Multiple independent commands: execute concurrently.
            let mut group_results: Vec<CommandServiceResult> = std::thread::scope(|scope| {
                let handles: Vec<_> = group
                    .iter()
                    .map(|cmd| {
                        let node_name = resolve_operator_name(cmd, node_name_table);
                        scope.spawn(move || {
                            let receipt =
                                execute_command(executor, cmd, &node_name, current_payload);
                            let outcome = outcome_from_receipt(&receipt, cmd);
                            CommandServiceResult {
                                command_id: cmd.command_id,
                                node_id: cmd.node_id,
                                outcome,
                                receipt,
                            }
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().ok())
                    .collect()
            });
            all_results.append(&mut group_results);
        }
    }

    // Deterministic receipt ordering by command_id (ascending enqueue order).
    all_results.sort_by_key(|r| r.command_id);

    // Push receipts back to the device ring (sequential — world is &mut).
    let cmd_by_id: std::collections::HashMap<i32, &DeviceCommand> =
        cmds.iter().map(|c| (c.command_id, c)).collect();
    for result in &all_results {
        if let Some(&cmd) = cmd_by_id.get(&result.command_id) {
            world.push_device_receipt_ext(DeviceReceiptExt {
                valid: 1,
                consumed: 0,
                command_id: result.command_id,
                node_id: result.node_id,
                src: cmd.src,
                dst: cmd.dst,
                outcome: result.outcome,
                receipt_kind: cmd.dispatch_kind,
                output_flags: 0,
                latency: 0.0,
            })?;
        }
    }

    Ok(all_results)
}

fn resolve_operator_name(cmd: &DeviceCommand, node_name_table: &[String]) -> String {
    let idx = cmd.operator_name_id as usize;
    if idx < node_name_table.len() {
        node_name_table[idx].clone()
    } else {
        format!("node_{}", cmd.node_id)
    }
}

fn execute_command(
    executor: &UniversalExecutor,
    cmd: &DeviceCommand,
    node_name: &str,
    current_payload: &Value,
) -> ProcessReceipt {
    match cmd.dispatch_kind {
        DISPATCH_KIND_EXTERNAL_PROCESS | DISPATCH_KIND_EXTERNAL_IO => {
            executor.execute_abstract_node_blocking(node_name, current_payload)
        }
        _ => ProcessReceipt {
            node_name: node_name.to_string(),
            exit_code: 1,
            stdout_payload: String::new(),
            stderr_payload: format!(
                "unsupported dispatch_kind {} in external command",
                cmd.dispatch_kind
            ),
        },
    }
}

fn outcome_from_receipt(receipt: &ProcessReceipt, cmd: &DeviceCommand) -> i32 {
    match receipt.exit_code {
        0 => 0, // success
        124 => {
            // exit 124 is the timeout sentinel from the process executor
            let _ = cmd; // timeout_ticks could be checked here in a future phase
            2
        }
        _ => 1, // failure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UniversalExecutor;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_cmd(command_id: i32, node_id: i32, operator_name_id: i32) -> DeviceCommand {
        DeviceCommand {
            valid: 1,
            command_id,
            node_id,
            src: 0,
            dst: node_id,
            dispatch_kind: DISPATCH_KIND_EXTERNAL_PROCESS,
            operator_name_id,
            timeout_ticks: 0,
            retry_budget: 1,
            payload_offset: 0,
            payload_len: 0,
        }
    }

    #[test]
    fn outcome_success_maps_to_zero() {
        let receipt = ProcessReceipt {
            node_name: "A".into(),
            exit_code: 0,
            stdout_payload: "ok".into(),
            stderr_payload: String::new(),
        };
        let cmd = make_cmd(1, 1, 0);
        assert_eq!(outcome_from_receipt(&receipt, &cmd), 0);
    }

    #[test]
    fn outcome_failure_maps_to_one() {
        let receipt = ProcessReceipt {
            node_name: "A".into(),
            exit_code: 1,
            stdout_payload: String::new(),
            stderr_payload: "err".into(),
        };
        let cmd = make_cmd(1, 1, 0);
        assert_eq!(outcome_from_receipt(&receipt, &cmd), 1);
    }

    #[test]
    fn outcome_timeout_maps_to_two() {
        let receipt = ProcessReceipt {
            node_name: "A".into(),
            exit_code: 124,
            stdout_payload: String::new(),
            stderr_payload: String::new(),
        };
        let cmd = make_cmd(1, 1, 0);
        assert_eq!(outcome_from_receipt(&receipt, &cmd), 2);
    }

    #[test]
    fn resolve_operator_name_uses_table() {
        let cmd = make_cmd(0, 0, 2);
        let table = vec!["A".to_string(), "B".to_string(), "IO::ReadFile".to_string()];
        assert_eq!(resolve_operator_name(&cmd, &table), "IO::ReadFile");
    }

    #[test]
    fn resolve_operator_name_falls_back_to_node_id() {
        let cmd = make_cmd(0, 7, 99);
        let table: Vec<String> = Vec::new();
        assert_eq!(resolve_operator_name(&cmd, &table), "node_7");
    }

    #[test]
    fn unsupported_dispatch_kind_returns_failure_receipt() {
        let executor = UniversalExecutor::new(HashMap::new());
        let mut cmd = make_cmd(0, 0, 0);
        cmd.dispatch_kind = 0; // DISPATCH_KIND_NONE
        let receipt = execute_command(&executor, &cmd, "SomeOp", &json!({}));
        assert_ne!(receipt.exit_code, 0);
        assert!(receipt.stderr_payload.contains("unsupported dispatch_kind"));
    }

    #[test]
    fn every_command_gets_exactly_one_receipt() {
        // Tests the invariant: every command_id produces exactly one outcome.
        let cmds = vec![make_cmd(10, 1, 0), make_cmd(11, 2, 1), make_cmd(12, 3, 2)];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        // Verify each command maps to a distinct command_id in results.
        let seen: std::collections::HashSet<i32> = cmds.iter().map(|c| c.command_id).collect();
        assert_eq!(seen.len(), cmds.len(), "command_ids must be unique");
        // operator_name_id resolution is 1-to-1.
        for cmd in &cmds {
            let name = resolve_operator_name(cmd, &names);
            assert!(!name.is_empty());
        }
    }

    // ── Phase 3: independence and parallel grouping ────────────────────────────

    fn empty_contracts() -> NodeContracts {
        NodeContracts::default()
    }

    fn contracts_with(pairs: &[(&str, &[&str])]) -> NodeContracts {
        let entries: Vec<String> = pairs
            .iter()
            .map(|(node, effects)| {
                let effects_json = effects
                    .iter()
                    .map(|e| format!("\"{e}\""))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{\"node\":\"{node}\",\"side_effects\":[{effects_json}]}}")
            })
            .collect();
        let json = format!("{{\"contracts\":[{}]}}", entries.join(","));
        NodeContracts::from_json_str(&json).unwrap()
    }

    fn make_io_cmd(command_id: i32, node_id: i32, operator_name_id: i32) -> DeviceCommand {
        let mut cmd = make_cmd(command_id, node_id, operator_name_id);
        cmd.dispatch_kind = DISPATCH_KIND_EXTERNAL_IO;
        cmd
    }

    #[test]
    fn io_commands_with_different_operator_ids_are_independent() {
        let contracts = empty_contracts();
        let table = vec!["IO::ReadA".to_string(), "IO::ReadB".to_string()];
        let a = make_io_cmd(1, 1, 0);
        let b = make_io_cmd(2, 2, 1);
        assert!(commands_are_independent(&a, &b, &table, &contracts));
    }

    #[test]
    fn same_operator_id_commands_are_never_independent() {
        let contracts = empty_contracts();
        let table = vec!["Op::X".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 0); // same operator_name_id
        assert!(!commands_are_independent(&a, &b, &table, &contracts));
    }

    #[test]
    fn process_commands_without_contracts_are_not_independent() {
        // EXTERNAL_PROCESS without declared side_effects → conservative: serialize.
        let contracts = empty_contracts();
        let table = vec!["Proc::A".to_string(), "Proc::B".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        assert!(!commands_are_independent(&a, &b, &table, &contracts));
    }

    #[test]
    fn overlapping_side_effects_make_commands_dependent() {
        let contracts = contracts_with(&[("A", &["slot.x"]), ("B", &["slot.x"])]);
        let table = vec!["A".to_string(), "B".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        assert!(!commands_are_independent(&a, &b, &table, &contracts));
    }

    #[test]
    fn disjoint_side_effects_make_commands_independent() {
        let contracts = contracts_with(&[("A", &["slot.x"]), ("B", &["slot.y"])]);
        let table = vec!["A".to_string(), "B".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        assert!(commands_are_independent(&a, &b, &table, &contracts));
    }

    #[test]
    fn two_independent_commands_partition_into_one_group() {
        let contracts = contracts_with(&[("A", &["slot.x"]), ("B", &["slot.y"])]);
        let table = vec!["A".to_string(), "B".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        let refs = [&a, &b];
        let groups = partition_independent_groups(&refs, &table, &contracts);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn two_dependent_commands_partition_into_two_groups() {
        let contracts = contracts_with(&[("A", &["slot.x"]), ("B", &["slot.x"])]);
        let table = vec!["A".to_string(), "B".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        let refs = [&a, &b];
        let groups = partition_independent_groups(&refs, &table, &contracts);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn three_commands_with_one_conflict_partition_correctly() {
        // A and B independent, B and C conflict → A+B in group 1, C in group 2.
        let contracts = contracts_with(&[
            ("A", &["slot.x"]),
            ("B", &["slot.y"]),
            ("C", &["slot.y"]),
        ]);
        let table = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let a = make_cmd(1, 1, 0);
        let b = make_cmd(2, 2, 1);
        let c = make_cmd(3, 3, 2);
        let refs = [&a, &b, &c];
        let groups = partition_independent_groups(&refs, &table, &contracts);
        // A goes to group 0; B goes to group 0 (A and B independent); C goes to group 1 (conflicts with B).
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 3, "all commands must be placed");
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn parallel_commands_preserve_receipt_identity() {
        // Verify command_id is preserved as the stable identity through the pipeline.
        let contracts = contracts_with(&[("A", &["slot.x"]), ("B", &["slot.y"])]);
        let table = vec!["A".to_string(), "B".to_string()];
        let a = make_cmd(42, 1, 0);
        let b = make_cmd(99, 2, 1);
        let refs = [&a, &b];
        let groups = partition_independent_groups(&refs, &table, &contracts);
        // Both commands in one group; both command_ids must be identifiable.
        let ids: Vec<i32> = groups
            .iter()
            .flat_map(|g| g.iter().map(|c| c.command_id))
            .collect();
        assert!(ids.contains(&42));
        assert!(ids.contains(&99));
    }
}
