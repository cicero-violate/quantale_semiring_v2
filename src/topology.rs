//! Data-driven graph topology loading and compilation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CudaError;
use crate::tensor::TensorEdge;
use crate::types::QuantaleWeight;

pub const DEFAULT_TOPOLOGY_JSON: &str = include_str!("../assets/topology.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyNode {
    pub id: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub action: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyTransition {
    pub from: String,
    pub to: String,
    pub default_weight: QuantaleWeight,
    #[serde(default)]
    pub confidence: Option<QuantaleWeight>,
    #[serde(default)]
    pub cost: Option<f32>,
    #[serde(default)]
    pub safety: Option<QuantaleWeight>,
    #[serde(default)]
    pub policy_effect: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyPage {
    pub name: String,
    #[serde(default)]
    pub node_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphTopology {
    pub matrix_name: String,
    pub nodes: Vec<TopologyNode>,
    pub transitions: Vec<TopologyTransition>,
    #[serde(default)]
    pub pages: Vec<TopologyPage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledTopology {
    pub matrix_name: String,
    pub node_count: usize,
    pub matrix_len: usize,
    pub registry: NodeRegistry,
    pub tensor_edges: Vec<TensorEdge>,
    pub pages: Vec<TopologyPage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeRegistry {
    by_name: BTreeMap<String, usize>,
    by_id: BTreeMap<usize, String>,
    actions: BTreeMap<usize, String>,
}

impl NodeRegistry {
    pub fn id_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn name_of(&self, id: usize) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn matrix_len(&self) -> usize {
        self.len() * self.len()
    }

    pub fn contains_name(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub fn action_of(&self, id: usize) -> Option<&str> {
        self.actions.get(&id).map(String::as_str)
    }
}

impl GraphTopology {
    pub fn from_json_str(input: &str) -> Result<Self, CudaError> {
        serde_json::from_str(input).map_err(|error| CudaError::invalid_input(error.to_string()))
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let input = fs::read_to_string(path)
            .map_err(|error| CudaError::invalid_input(error.to_string()))?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, CudaError> {
        Self::from_json_str(DEFAULT_TOPOLOGY_JSON)
    }

    pub fn bundled_registry() -> Result<NodeRegistry, CudaError> {
        Self::default_asset()?.compile().map(|ct| ct.registry)
    }

    pub fn compile(&self) -> Result<CompiledTopology, CudaError> {
        let registry = self.build_registry()?;
        let mut tensor_edges = Vec::with_capacity(self.transitions.len());

        for transition in &self.transitions {
            let src = registry.id_of(&transition.from).ok_or_else(|| {
                CudaError::invalid_input(format!("source '{}' missing", transition.from))
            })?;
            let dst = registry.id_of(&transition.to).ok_or_else(|| {
                CudaError::invalid_input(format!("destination '{}' missing", transition.to))
            })?;
            let default_weight = transition.default_weight.raw();
            tensor_edges.push(TensorEdge::new(
                src as i32,
                dst as i32,
                transition
                    .confidence
                    .unwrap_or(transition.default_weight)
                    .raw(),
                transition.cost.unwrap_or(1.0 - default_weight),
                transition.safety.unwrap_or(transition.default_weight).raw(),
            ));
        }

        validate_pages(&registry, &self.pages)?;

        Ok(CompiledTopology {
            matrix_name: self.matrix_name.clone(),
            node_count: registry.len(),
            matrix_len: registry.matrix_len(),
            registry,
            tensor_edges,
            pages: self.pages.clone(),
        })
    }

    fn build_registry(&self) -> Result<NodeRegistry, CudaError> {
        if self.nodes.is_empty() {
            return Err(CudaError::invalid_input("no nodes"));
        }

        let mut by_name = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        let mut actions = BTreeMap::new();
        for node in &self.nodes {
            if by_name.insert(node.name.clone(), node.id).is_some()
                || by_id.insert(node.id, node.name.clone()).is_some()
            {
                return Err(CudaError::invalid_input("duplicate node item"));
            }
            if let Some(action) = &node.action {
                actions.insert(node.id, action.clone());
            }
        }

        for expected_id in 0..self.nodes.len() {
            if !by_id.contains_key(&expected_id) {
                return Err(CudaError::invalid_input(format!(
                    "topology node ids must be dense; missing id {expected_id}"
                )));
            }
        }

        for transition in &self.transitions {
            if !by_name.contains_key(&transition.from) {
                return Err(CudaError::invalid_input(format!(
                    "transition source '{}' missing",
                    transition.from
                )));
            }
            if !by_name.contains_key(&transition.to) {
                return Err(CudaError::invalid_input(format!(
                    "transition destination '{}' missing",
                    transition.to
                )));
            }
        }
        Ok(NodeRegistry {
            by_name,
            by_id,
            actions,
        })
    }
}

pub fn load_default_tensor_topology_edges() -> Result<Vec<TensorEdge>, CudaError> {
    Ok(GraphTopology::default_asset()?.compile()?.tensor_edges)
}

fn validate_pages(registry: &NodeRegistry, pages: &[TopologyPage]) -> Result<(), CudaError> {
    let mut names = BTreeSet::new();
    for page in pages {
        if !names.insert(page.name.clone()) {
            return Err(CudaError::invalid_input(format!(
                "duplicate page '{}'",
                page.name
            )));
        }
        for node in &page.node_names {
            if registry.id_of(node).is_none() {
                return Err(CudaError::invalid_input(format!(
                    "page '{}' references unknown node '{}'",
                    page.name, node
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_topology_fields_compile_to_tensor_edges() {
        let raw = r#"{
            "matrix_name": "test",
            "nodes": [
                {"id": 0, "name": "State::Goal", "type": "State"},
                {"id": 1, "name": "State::Input", "type": "State"}
            ],
            "transitions": [
                {
                    "from": "State::Goal",
                    "to": "State::Input",
                    "default_weight": 0.7,
                    "confidence": 0.9,
                    "cost": 2.5,
                    "safety": 0.8
                }
            ]
        }"#;
        let compiled = GraphTopology::from_json_str(raw)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(compiled.tensor_edges[0].confidence, 0.9);
        assert_eq!(compiled.tensor_edges[0].cost, 2.5);
        assert_eq!(compiled.tensor_edges[0].safety, 0.8);
    }

    #[test]
    fn tensor_topology_defaults_from_scalar_weight() {
        let raw = r#"{
            "matrix_name": "test",
            "nodes": [
                {"id": 0, "name": "State::Goal", "type": "State"},
                {"id": 1, "name": "State::Input", "type": "State"}
            ],
            "transitions": [
                {"from": "State::Goal", "to": "State::Input", "default_weight": 0.75}
            ]
        }"#;
        let compiled = GraphTopology::from_json_str(raw)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(compiled.tensor_edges[0].confidence, 0.75);
        assert!((compiled.tensor_edges[0].cost - 0.25).abs() < f32::EPSILON);
        assert_eq!(compiled.tensor_edges[0].safety, 0.75);
    }

    #[test]
    fn bundled_registry_round_trips_node_names_and_matrix_len() {
        let topology = GraphTopology::default_asset().unwrap();
        let compiled = topology.compile().unwrap();
        for node in &topology.nodes {
            let id = compiled.registry.id_of(&node.name).unwrap();
            assert_eq!(compiled.registry.name_of(id), Some(node.name.as_str()));
        }
        assert_eq!(
            compiled.registry.matrix_len(),
            compiled.registry.len() * compiled.registry.len()
        );
    }

    #[test]
    fn registry_reads_declared_actions() {
        let raw = r#"{
            "matrix_name": "test",
            "nodes": [
                {"id": 0, "name": "Control::Commit", "type": "Control", "action": "commit"}
            ],
            "transitions": []
        }"#;
        let compiled = GraphTopology::from_json_str(raw)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(compiled.registry.action_of(0), Some("commit"));
    }

    #[test]
    fn registry_rejects_sparse_node_ids() {
        let raw = r#"{
            "matrix_name": "test",
            "nodes": [
                {"id": 0, "name": "State::Goal", "type": "State"},
                {"id": 2, "name": "State::Input", "type": "State"}
            ],
            "transitions": []
        }"#;
        let err = GraphTopology::from_json_str(raw)
            .unwrap()
            .compile()
            .unwrap_err();
        assert!(err.message.contains("dense"));
    }
}
