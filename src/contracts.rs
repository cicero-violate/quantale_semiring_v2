//! Declarative node execution contracts.
//!
//! Contracts are data-only guards loaded from `assets/node_contracts.json`.
//! They keep payload-dependent or side-effecting nodes from being executed by
//! exploration unless their declared preconditions are satisfied.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

const DEFAULT_NODE_CONTRACTS_JSON: &str = include_str!("../assets/node_contracts.json");

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct NodeContractRegistryFile {
    #[serde(default)]
    pub contracts: Vec<NodeContract>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct NodeContract {
    pub node: String,
    #[serde(default)]
    pub requires_payload: Vec<String>,
    #[serde(default)]
    pub allowed_after: Vec<String>,
    #[serde(default)]
    pub side_effects: Vec<String>,
    #[serde(default)]
    pub mutation_mode: Option<String>,
    #[serde(default)]
    pub exploration: ExplorationContract,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ExplorationContract {
    #[serde(default = "default_allow_direct")]
    pub allow_direct: bool,
}

impl Default for ExplorationContract {
    fn default() -> Self {
        Self {
            allow_direct: default_allow_direct(),
        }
    }
}

fn default_allow_direct() -> bool {
    true
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeContracts {
    contracts: HashMap<String, NodeContract>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractContext {
    Exploration,
    Frontier,
    Batch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractViolation {
    pub node: String,
    pub reason: String,
}

impl std::fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.node, self.reason)
    }
}

impl NodeContracts {
    pub fn default_asset() -> Self {
        fs::read_to_string("assets/node_contracts.json")
            .ok()
            .and_then(|input| Self::from_json_str(&input).ok())
            .unwrap_or_else(|| {
                Self::from_json_str(DEFAULT_NODE_CONTRACTS_JSON)
                    .expect("embedded node_contracts.json is valid")
            })
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let input = fs::read_to_string(path)
            .map_err(|error| format!("read node contracts '{}': {error}", path.display()))?;
        Self::from_json_str(&input)
    }

    pub fn from_json_str(input: &str) -> Result<Self, String> {
        let parsed: NodeContractRegistryFile = serde_json::from_str(input)
            .map_err(|error| format!("parse node contracts: {error}"))?;
        let mut contracts = HashMap::with_capacity(parsed.contracts.len());
        for contract in parsed.contracts {
            if contract.node.is_empty() {
                return Err("node contract missing node".to_string());
            }
            if contracts.insert(contract.node.clone(), contract).is_some() {
                return Err("duplicate node contract".to_string());
            }
        }
        Ok(Self { contracts })
    }

    pub fn get(&self, node: &str) -> Option<&NodeContract> {
        self.contracts.get(node)
    }

    pub fn validate(
        &self,
        node: &str,
        payload: &Value,
        payload_origin: Option<&str>,
        context: ContractContext,
    ) -> Result<(), ContractViolation> {
        let Some(contract) = self.get(node) else {
            return Ok(());
        };

        if context == ContractContext::Exploration && !contract.exploration.allow_direct {
            return Err(ContractViolation {
                node: node.to_string(),
                reason: "direct exploration is disabled by node contract".to_string(),
            });
        }

        if !contract.allowed_after.is_empty() {
            match payload_origin {
                Some(origin)
                    if contract
                        .allowed_after
                        .iter()
                        .any(|allowed| allowed == origin) => {}
                Some(origin) => {
                    return Err(ContractViolation {
                        node: node.to_string(),
                        reason: format!(
                            "payload origin '{origin}' is not in allowed_after {:?}",
                            contract.allowed_after
                        ),
                    });
                }
                None => {
                    return Err(ContractViolation {
                        node: node.to_string(),
                        reason: format!(
                            "missing payload origin; allowed_after {:?}",
                            contract.allowed_after
                        ),
                    });
                }
            }
        }

        let unwrapped = unwrap_context_payload(payload);
        for key in &contract.requires_payload {
            if !unwrapped.get(key).is_some_and(payload_value_present) {
                return Err(ContractViolation {
                    node: node.to_string(),
                    reason: format!("missing required payload key '{key}'"),
                });
            }
        }

        Ok(())
    }
}

fn payload_value_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
        Value::Bool(_) | Value::Number(_) => true,
    }
}

fn unwrap_context_payload(payload: &Value) -> Value {
    let mut current = payload.clone();
    for _ in 0..6 {
        if !current.get("context").is_some_and(Value::is_string) {
            return current;
        }
        let Some(context) = current.get("context").and_then(Value::as_str) else {
            return current;
        };
        let Ok(next) = serde_json::from_str::<Value>(context.trim()) else {
            return current;
        };
        if !next.is_object() {
            return current;
        }
        current = next;
    }
    current
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn contracts() -> NodeContracts {
        NodeContracts::from_json_str(
            r#"{
                "contracts": [{
                    "node": "Control::WriteOperator",
                    "requires_payload": ["filename", "source"],
                    "allowed_after": ["State::OperatorPlan"],
                    "exploration": {"allow_direct": false}
                }]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn validates_unwrapped_payload_and_origin() {
        let payload = json!({
            "context": "{\"filename\":\"x.py\",\"source\":\"print(1)\\n\"}"
        });
        assert!(
            contracts()
                .validate(
                    "Control::WriteOperator",
                    &payload,
                    Some("State::OperatorPlan"),
                    ContractContext::Frontier,
                )
                .is_ok()
        );
    }

    #[test]
    fn blocks_missing_required_payload() {
        let payload = json!({"context": "{\"filename\":\"x.py\"}"});
        let err = contracts()
            .validate(
                "Control::WriteOperator",
                &payload,
                Some("State::OperatorPlan"),
                ContractContext::Frontier,
            )
            .unwrap_err();
        assert!(err.reason.contains("source"));
    }

    #[test]
    fn blocks_disallowed_exploration() {
        let payload = json!({
            "context": "{\"filename\":\"x.py\",\"source\":\"print(1)\\n\"}"
        });
        let err = contracts()
            .validate(
                "Control::WriteOperator",
                &payload,
                Some("State::OperatorPlan"),
                ContractContext::Exploration,
            )
            .unwrap_err();
        assert!(err.reason.contains("direct exploration"));
    }

    #[test]
    fn blocks_wrong_origin() {
        let payload = json!({
            "context": "{\"filename\":\"x.py\",\"source\":\"print(1)\\n\"}"
        });
        let err = contracts()
            .validate(
                "Control::WriteOperator",
                &payload,
                Some("State::Plan"),
                ContractContext::Frontier,
            )
            .unwrap_err();
        assert!(err.reason.contains("allowed_after"));
    }
}
