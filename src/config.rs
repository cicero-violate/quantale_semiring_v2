//! Runtime configuration for the CUDA quantale orchestrator.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::fusion_dispatch::FusionDispatch;
use crate::topology::GraphTopology;

pub const DEFAULT_OPERATORS_JSON: &str = include_str!("../assets/operators.json");
pub const DEFAULT_RUNTIME_CONTEXT_JSON: &str = include_str!("../assets/runtime_context.json");
pub const DEFAULT_RELOAD_POLICY_JSON: &str = include_str!("../assets/reload_policy.json");
pub const DEFAULT_BLOCK_SIZE: usize = 512;

pub type OperatorRegistry = HashMap<String, Value>;

#[derive(Debug, Deserialize)]
struct OperatorRegistryFile {
    operators: Vec<Value>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct RuntimePolicyFile {
    ingress_capacity_hint: Option<usize>,
    max_ticks: Option<usize>,
    tick_sleep_ms: Option<u64>,
    decay_normal: Option<f32>,
    decay_blocked: Option<f32>,
    hard_reset_blocks: Option<usize>,
    hard_reset_sleep_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeContext {
    pub default_context: Value,
    pub start_node: String,
    pub reset_context: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ReloadPolicy {
    pub watched_asset_paths: Vec<PathBuf>,
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
    /// Loaded from `assets/topology.fusion.json`; JitChains built from
    /// `operator_registry`.  Empty when the file doesn't exist yet.
    pub fusion_dispatch: FusionDispatch,
    pub ingress_capacity_hint: usize,
    /// Maximum ticks before the loop exits. 0 means run forever.
    pub max_ticks: usize,
    /// Milliseconds to sleep after each tick. 0 means no sleep.
    pub tick_sleep_ms: u64,
    /// Decay factor applied each normal (unblocked) tick.
    pub decay_normal: f32,
    /// Decay factor applied when the current tick is blocked or halted.
    pub decay_blocked: f32,
    /// Consecutive blocked/failed steps before a hard reset fires.
    pub hard_reset_blocks: usize,
    /// Milliseconds to sleep during a hard reset.
    pub hard_reset_sleep_ms: u64,
}

impl RuntimeContext {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        let context: Self = serde_json::from_str(input)
            .map_err(|error| format!("parse runtime context: {error}"))?;
        context.validate()?;
        Ok(context)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let input = fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "read runtime context '{}': {error}",
                path.as_ref().display()
            )
        })?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, String> {
        Self::from_json_file("assets/runtime_context.json")
            .or_else(|_| Self::from_json_str(DEFAULT_RUNTIME_CONTEXT_JSON))
    }

    pub fn default_payload(&self) -> Value {
        json!({ "context": self.default_context.clone() })
    }

    pub fn reset_payload(&self) -> Value {
        json!({ "context": self.reset_context.clone() })
    }

    fn validate(&self) -> Result<(), String> {
        if invalid_context_value(&self.default_context) {
            return Err("runtime_context default_context must not be null".to_string());
        }
        if invalid_context_value(&self.reset_context) {
            return Err("runtime_context reset_context must not be null".to_string());
        }
        if self.start_node.trim().is_empty() {
            return Err("runtime_context start_node must be non-empty".to_string());
        }
        Ok(())
    }
}

impl ReloadPolicy {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        let policy: Self =
            serde_json::from_str(input).map_err(|error| format!("parse reload policy: {error}"))?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let input = fs::read_to_string(path.as_ref()).map_err(|error| {
            format!("read reload policy '{}': {error}", path.as_ref().display())
        })?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, String> {
        Self::from_json_file("assets/reload_policy.json")
            .or_else(|_| Self::from_json_str(DEFAULT_RELOAD_POLICY_JSON))
    }

    fn validate(&self) -> Result<(), String> {
        if self.watched_asset_paths.is_empty() {
            return Err("reload policy watched_asset_paths must not be empty".to_string());
        }
        if self
            .watched_asset_paths
            .iter()
            .any(|path| path.as_os_str().is_empty())
        {
            return Err(
                "reload policy watched_asset_paths must not contain empty paths".to_string(),
            );
        }
        Ok(())
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        let operators_path = default_operators_path();
        let operator_registry = load_operator_registry(&operators_path)
            .or_else(|_| parse_operator_registry_str(DEFAULT_OPERATORS_JSON))
            .unwrap_or_default();
        let (matrix_dim, matrix_len) = GraphTopology::bundled_registry()
            .map(|registry| (registry.len(), registry.matrix_len()))
            .unwrap_or((0, 0));
        let runtime_policy = load_runtime_policy("assets/runtime_policy.json");

        let fusion_dispatch =
            FusionDispatch::load("assets/topology.fusion.json", &operator_registry)
                .unwrap_or_default();

        Self {
            matrix_dim,
            matrix_len,
            block_size: DEFAULT_BLOCK_SIZE,
            tlog_path: PathBuf::from("state/quantale.tlog"),
            learned_edges_path: PathBuf::from("state/learned_edges.jsonl"),
            operators_path,
            operator_registry,
            fusion_dispatch,
            ingress_capacity_hint: runtime_policy.ingress_capacity_hint.unwrap_or(1024),
            max_ticks: max_ticks_from_env(runtime_policy.max_ticks.unwrap_or(64)),
            tick_sleep_ms: tick_sleep_ms_from_env(runtime_policy.tick_sleep_ms.unwrap_or(0)),
            decay_normal: runtime_policy.decay_normal.unwrap_or(0.995),
            decay_blocked: runtime_policy.decay_blocked.unwrap_or(0.97),
            hard_reset_blocks: runtime_policy.hard_reset_blocks.unwrap_or(3),
            hard_reset_sleep_ms: runtime_policy.hard_reset_sleep_ms.unwrap_or(200),
        }
    }
}

pub fn default_operators_path() -> PathBuf {
    let generated = PathBuf::from("assets/operators.generated.json");
    if generated.exists() {
        generated
    } else {
        PathBuf::from("assets/operators.json")
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
        self.reload_fusion_dispatch();
        Ok(())
    }

    pub fn reload_default_operator_registry(&mut self) -> Result<(), String> {
        self.operators_path = default_operators_path();
        self.reload_operator_registry()
    }

    /// Reload `topology.fusion.json` into `fusion_dispatch` using the current
    /// operator registry.  Silently resets to empty if the file is absent.
    pub fn reload_fusion_dispatch(&mut self) {
        self.fusion_dispatch =
            FusionDispatch::load("assets/topology.fusion.json", &self.operator_registry)
                .unwrap_or_default();
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
        if self.ingress_capacity_hint == 0 {
            return Err("ingress_capacity_hint must be nonzero".to_string());
        }
        if self.decay_normal <= 0.0 || self.decay_normal > 1.0 {
            return Err("decay_normal must be in (0, 1]".to_string());
        }
        if self.decay_blocked <= 0.0 || self.decay_blocked > 1.0 {
            return Err("decay_blocked must be in (0, 1]".to_string());
        }
        if self.hard_reset_blocks == 0 {
            return Err("hard_reset_blocks must be nonzero".to_string());
        }
        if self.operator_registry.is_empty() {
            return Err("operator_registry must contain at least one operator".to_string());
        }
        Ok(())
    }
}

fn load_runtime_policy(path: impl AsRef<Path>) -> RuntimePolicyFile {
    fs::read_to_string(path.as_ref())
        .ok()
        .and_then(|input| serde_json::from_str(&input).ok())
        .unwrap_or_default()
}

fn invalid_context_value(value: &Value) -> bool {
    value.is_null()
}

fn max_ticks_from_env(default: usize) -> usize {
    if std::env::var("QUANTALE_LOOP_FOREVER").as_deref() == Ok("1") {
        return 0;
    }
    std::env::var("QUANTALE_MAX_TICKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn tick_sleep_ms_from_env(default: u64) -> u64 {
    std::env::var("QUANTALE_TICK_SLEEP_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
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
