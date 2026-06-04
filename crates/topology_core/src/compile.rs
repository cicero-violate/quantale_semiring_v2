use std::fs;
use std::path::Path;

use crate::{CompiledTopology, CompiledTransition, GraphTopology, NodeRegistry, TopologyError};

pub const DEFAULT_TOPOLOGY_JSON: &str = include_str!("../../../assets/topology.generated.json");

impl GraphTopology {
    pub fn from_json_str(input: &str) -> Result<Self, TopologyError> {
        serde_json::from_str(input).map_err(|error| TopologyError::invalid_input(error.to_string()))
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, TopologyError> {
        let input = fs::read_to_string(path)
            .map_err(|error| TopologyError::invalid_input(error.to_string()))?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, TopologyError> {
        if Path::new("assets/topology.generated.json").exists() {
            return Self::from_json_file("assets/topology.generated.json");
        }
        // Fall back to the compile-time bundled generated topology only. The
        // hand-authored topology.json is a build input, not a runtime fallback.
        Self::from_json_str(DEFAULT_TOPOLOGY_JSON)
    }

    pub fn bundled_registry() -> Result<NodeRegistry, TopologyError> {
        Self::default_asset()?.compile().map(|ct| ct.registry)
    }

    pub fn compile(&self) -> Result<CompiledTopology, TopologyError> {
        let registry = self.build_registry()?;
        let mut seen = std::collections::BTreeSet::new();
        let mut transitions = Vec::with_capacity(self.transitions.len());

        for transition in &self.transitions {
            if !seen.insert((transition.from.as_str(), transition.to.as_str())) {
                return Err(TopologyError::invalid_input(format!(
                    "duplicate transition '{}' -> '{}'",
                    transition.from, transition.to
                )));
            }
            let src = registry.id_of(&transition.from).ok_or_else(|| {
                TopologyError::invalid_input(format!("source '{}' missing", transition.from))
            })?;
            let dst = registry.id_of(&transition.to).ok_or_else(|| {
                TopologyError::invalid_input(format!("destination '{}' missing", transition.to))
            })?;
            let default_weight = transition.default_weight;
            transitions.push(CompiledTransition {
                src,
                dst,
                confidence: transition.confidence.unwrap_or(default_weight),
                cost: transition.cost.unwrap_or(1.0 - default_weight),
                safety: transition.safety.unwrap_or(default_weight),
            });
        }

        Ok(CompiledTopology {
            matrix_name: self.matrix_name.clone(),
            node_count: registry.len(),
            matrix_len: registry.matrix_len(),
            registry,
            transitions,
            pages: self.pages.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn bundled_default_topology_is_generated_quantale_asset() {
        let value: Value = serde_json::from_str(DEFAULT_TOPOLOGY_JSON).unwrap();
        assert!(value.get("quantale").is_some());
        assert!(value.get("source_version").is_some());
        let transitions = value
            .get("transitions")
            .and_then(Value::as_array)
            .expect("generated topology transitions");
        assert!(!transitions.is_empty());
        assert!(
            transitions
                .iter()
                .all(|transition| transition.get("default_weight").is_none()),
            "generated runtime topology must not carry legacy default_weight"
        );
        assert!(
            transitions.iter().all(|transition| {
                transition.get("confidence").is_some()
                    && transition.get("cost").is_some()
                    && transition.get("safety").is_some()
            }),
            "generated runtime topology must carry explicit quantale triples"
        );
    }
}
