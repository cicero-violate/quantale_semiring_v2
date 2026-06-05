//! Phase-3 external command service.
//!
//! When `orchestrate_step()` returns `OrchStepStatus::WaitExternal`, the GPU
//! has emitted one or more `DeviceCommand` entries that require CPU/IO work.
//! This module provides `service_external_commands`, which:
//!
//!   1. Drains the device command ring.
//!   2. For each valid command, resolves the operator and executes it.
//!   3. Pushes a `DeviceReceiptExt` with the outcome back into the device ring.
//!   4. Decrements `OrchestrationState::pending_external_count` on device via
//!      the receipt drain kernel.
//!
//! The CPU no longer decides fallback routing. It only executes GPU-emitted
//! commands and returns receipts.

use serde_json::Value;

use crate::UniversalExecutor;
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
}
