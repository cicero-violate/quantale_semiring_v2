//! Data-driven graph topology loading and compilation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::algebra::{Q_BOTTOM, Q_UNIT};
use crate::edge::TransitionEdge;
use crate::error::CudaError;
use crate::node::{MATRIX_LEN, NODE_COUNT, Node};
use crate::types::QuantaleWeight;

pub const DEFAULT_TOPOLOGY_JSON: &str = include_str!("../assets/topology.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyNode {
    pub id: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyTransition {
    pub from: String,
    pub to: String,
    pub default_weight: QuantaleWeight,
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
    pub edges: Vec<TransitionEdge>,
    pub pages: Vec<TopologyPage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeRegistry {
    by_name: BTreeMap<String, usize>,
    by_id: BTreeMap<usize, String>,
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

    pub fn compile(&self) -> Result<CompiledTopology, CudaError> {
        let registry = self.build_registry()?;
        let mut edges = Vec::with_capacity(self.transitions.len());

        for transition in &self.transitions {
            let src = registry.id_of(&transition.from).ok_or_else(|| {
                CudaError::invalid_input(format!("source '{}' missing", transition.from))
            })?;
            let dst = registry.id_of(&transition.to).ok_or_else(|| {
                CudaError::invalid_input(format!("destination '{}' missing", transition.to))
            })?;
            edges.push(TransitionEdge::new(
                src as i32,
                dst as i32,
                transition.default_weight.raw(),
            ));
        }

        validate_pages(&registry, &self.pages)?;

        Ok(CompiledTopology {
            matrix_name: self.matrix_name.clone(),
            node_count: registry.len(),
            matrix_len: registry.len() * registry.len(),
            registry,
            edges,
            pages: self.pages.clone(),
        })
    }

    fn build_registry(&self) -> Result<NodeRegistry, CudaError> {
        if self.nodes.is_empty() {
            return Err(CudaError::invalid_input("no nodes"));
        }
        if self.nodes.len() > NODE_COUNT {
            return Err(CudaError::invalid_input("node count overflow"));
        }

        let mut by_name = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        for node in &self.nodes {
            if node.id >= NODE_COUNT || Node::decode_index(node.id).is_none() {
                return Err(CudaError::invalid_input("node ID invalid"));
            }
            if by_name.insert(node.name.clone(), node.id).is_some()
                || by_id.insert(node.id, node.name.clone()).is_some()
            {
                return Err(CudaError::invalid_input("duplicate node item"));
            }
        }
        Ok(NodeRegistry { by_name, by_id })
    }
}

impl CompiledTopology {
    pub fn dense_matrix(&self) -> Vec<f32> {
        let mut matrix = vec![Q_BOTTOM; MATRIX_LEN];
        for edge in &self.edges {
            let idx = (edge.src as usize) * NODE_COUNT + edge.dst as usize;
            matrix[idx] = matrix[idx].max(edge.value);
        }
        for node in 0..self.node_count {
            matrix[node * NODE_COUNT + node] = matrix[node * NODE_COUNT + node].max(Q_UNIT);
        }
        matrix
    }
}

pub fn load_default_topology_edges() -> Result<Vec<TransitionEdge>, CudaError> {
    Ok(GraphTopology::default_asset()?.compile()?.edges)
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
