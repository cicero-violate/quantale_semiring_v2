//! Runtime configuration for the CUDA quantale orchestrator.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::topology::GraphTopology;

pub const DEFAULT_OPERATORS_JSON: &str = include_str!("../assets/operators.json");
pub const DEFAULT_BLOCK_SIZE: usize = 512;

pub type OperatorRegistry = HashMap<String, Value>;

#[derive(Debug, Deserialize)]
struct OperatorRegistryFile {
    operators: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemConfig {
    pub matrix_dim: usize,
    pub matrix_len: usize,
    pub block_size: usize,
    pub tlog_path: PathBuf,
    pub learned_edges_path: PathBuf,
    pub operators_path: PathBuf,
    pub operator_registry: OperatorRegistry,
    pub ingress_capacity_hint: usize,
    pub max_ticks: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        let operators_path = PathBuf::from("assets/operators.json");
        let operator_registry = load_operator_registry(&operators_path)
            .or_else(|_| parse_operator_registry_str(DEFAULT_OPERATORS_JSON))
            .unwrap_or_default();
        let (matrix_dim, matrix_len) = GraphTopology::bundled_registry()
            .map(|registry| (registry.len(), registry.matrix_len()))
            .unwrap_or((0, 0));

        Self {
            matrix_dim,
            matrix_len,
            block_size: DEFAULT_BLOCK_SIZE,
            tlog_path: PathBuf::from("state/quantale.tlog"),
            learned_edges_path: PathBuf::from("state/learned_edges.jsonl"),
            operators_path,
            operator_registry,
            ingress_capacity_hint: 1024,
            max_ticks: 64,
        }
    }
}

impl SystemConfig {
    pub fn with_tlog_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tlog_path = path.into();
        self
    }

    pub fn with_operator_registry(mut self, operator_registry: OperatorRegistry) -> Self {
        self.operator_registry = operator_registry;
        self
    }

    pub fn with_operators_path(mut self, path: impl Into<PathBuf>) -> Result<Self, String> {
        self.operators_path = path.into();
        self.reload_operator_registry()?;
        Ok(self)
    }

    pub fn reload_operator_registry(&mut self) -> Result<(), String> {
        self.operator_registry = load_operator_registry(&self.operators_path)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let registry = GraphTopology::bundled_registry()
            .map_err(|error| format!("load bundled topology: {error}"))?;
        if self.matrix_dim != registry.len() {
            return Err(format!(
                "matrix_dim {} does not match bundled registry len {}",
                self.matrix_dim,
                registry.len()
            ));
        }
        if self.matrix_len != registry.matrix_len() {
            return Err(format!(
                "matrix_len {} does not match bundled registry matrix_len {}",
                self.matrix_len,
                registry.matrix_len()
            ));
        }
        if self.block_size == 0 {
            return Err("block_size must be nonzero".to_string());
        }
        if self.operator_registry.is_empty() {
            return Err("operator_registry must contain at least one operator".to_string());
        }
        Ok(())
    }
}

pub fn load_operator_registry(path: impl AsRef<Path>) -> Result<OperatorRegistry, String> {
    let input = fs::read_to_string(path.as_ref()).map_err(|error| {
        format!(
            "read operator registry '{}': {error}",
            path.as_ref().display()
        )
    })?;
    parse_operator_registry_str(&input)
}

pub fn parse_operator_registry_str(input: &str) -> Result<OperatorRegistry, String> {
    let parsed: OperatorRegistryFile =
        serde_json::from_str(input).map_err(|error| format!("parse operator registry: {error}"))?;
    let mut registry = OperatorRegistry::with_capacity(parsed.operators.len());

    for operator in parsed.operators {
        let node_name = operator
            .get("node_name")
            .and_then(Value::as_str)
            .ok_or_else(|| "operator missing string node_name".to_string())?
            .to_string();
        registry.insert(node_name, operator);
    }

    Ok(registry)
}
