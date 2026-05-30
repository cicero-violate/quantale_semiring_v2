//! Hardcode-free universal execution pipeline for arbitrary OS processes.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

use crate::config::SystemConfig;
use crate::receipt::ProcessReceipt;

pub struct UniversalExecutor {
    /// Mapping of Node Names to their generic CLI schemas loaded from operators.json.
    pub operator_registry: HashMap<String, Value>,
}

impl UniversalExecutor {
    pub fn from_config(config: &SystemConfig) -> Self {
        Self {
            operator_registry: config.operator_registry.clone(),
        }
    }

    pub fn new(operator_registry: HashMap<String, Value>) -> Self {
        Self { operator_registry }
    }

    /// Spawns and executes any command contract defined in the operators configuration.
    pub async fn execute_abstract_node(
        &self,
        node_name: &str,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        self.execute_abstract_node_blocking(node_name, dynamic_payload)
    }

    /// Blocking implementation used by synchronous host loops and tests.
    pub fn execute_abstract_node_blocking(
        &self,
        node_name: &str,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        let op_config = match self.operator_registry.get(node_name) {
            Some(config) => config,
            None => {
                return ProcessReceipt {
                    node_name: node_name.to_string(),
                    exit_code: 127,
                    stdout_payload: String::new(),
                    stderr_payload: "Node operator contract missing from registry".to_string(),
                };
            }
        };

        let binary = op_config["executable"].as_str().unwrap_or("false");
        let empty_args = Vec::new();
        let static_args: Vec<&str> = op_config["static_args"]
            .as_array()
            .unwrap_or(&empty_args)
            .iter()
            .filter_map(Value::as_str)
            .collect();

        let mut command = Command::new(binary);
        command.args(&static_args);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return ProcessReceipt {
                    node_name: node_name.to_string(),
                    exit_code: 1,
                    stdout_payload: String::new(),
                    stderr_payload: format!("Failed to spawn process: {error}"),
                };
            }
        };

        let stdin_mode = op_config["input_mapping"]["stdin_mode"]
            .as_str()
            .unwrap_or("field");

        if stdin_mode == "json" {
            if let Some(mut stdin) = child.stdin.take() {
                let json_bytes = serde_json::to_vec(dynamic_payload).unwrap_or_default();
                if let Err(error) = stdin.write_all(&json_bytes) {
                    return ProcessReceipt {
                        node_name: node_name.to_string(),
                        exit_code: 1,
                        stdout_payload: String::new(),
                        stderr_payload: format!("Failed to write JSON stdin: {error}"),
                    };
                }
            }
        } else if let Some(stdin_field) = op_config["input_mapping"]["stdin_source"].as_str() {
            if let Some(mut stdin) = child.stdin.take() {
                let content = dynamic_payload[stdin_field].as_str().unwrap_or("");
                if let Err(error) = stdin.write_all(content.as_bytes()) {
                    return ProcessReceipt {
                        node_name: node_name.to_string(),
                        exit_code: 1,
                        stdout_payload: String::new(),
                        stderr_payload: format!("Failed to write process stdin: {error}"),
                    };
                }
            }
        }

        match child.wait_with_output() {
            Ok(output) => ProcessReceipt {
                node_name: node_name.to_string(),
                exit_code: output.status.code().unwrap_or(1),
                stdout_payload: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr_payload: String::from_utf8_lossy(&output.stderr).into_owned(),
            },
            Err(error) => ProcessReceipt {
                node_name: node_name.to_string(),
                exit_code: 1,
                stdout_payload: String::new(),
                stderr_payload: format!("Failed to wait for process: {error}"),
            },
        }
    }
}
