//! Data-driven graph topology loading and compilation.
//!
//! This layer turns JSON topology assets into ordinary `TransitionEdge` values
//! for the existing CUDA matrix engine. It is intentionally bounded by the
//! current fixed CUDA node universe: oversized or non-canonical IDs are rejected
//! before upload.

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
            .map_err(|error| CudaError::invalid_input(format!("read topology: {error}")))?;
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
                CudaError::invalid_input(format!(
                    "transition source '{}' is not declared",
                    transition.from
                ))
            })?;
            let dst = registry.id_of(&transition.to).ok_or_else(|| {
                CudaError::invalid_input(format!(
                    "transition destination '{}' is not declared",
                    transition.to
                ))
            })?;
            edges.push(TransitionEdge::new(
                i32::try_from(src).map_err(|_| CudaError::invalid_input("source id overflow"))?,
                i32::try_from(dst)
                    .map_err(|_| CudaError::invalid_input("destination id overflow"))?,
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
            return Err(CudaError::invalid_input("topology has no nodes"));
        }
        if self.nodes.len() > NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "topology has {} nodes but current CUDA kernel supports {}",
                self.nodes.len(),
                NODE_COUNT
            )));
        }

        let mut by_name = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        for node in &self.nodes {
            if node.id >= NODE_COUNT {
                return Err(CudaError::invalid_input(format!(
                    "node '{}' id {} exceeds current CUDA universe {}",
                    node.name, node.id, NODE_COUNT
                )));
            }
            if Node::decode(node.id as i32).is_none() {
                return Err(CudaError::invalid_input(format!(
                    "node '{}' id {} is not decodable",
                    node.name, node.id
                )));
            }
            if by_name.insert(node.name.clone(), node.id).is_some() {
                return Err(CudaError::invalid_input(format!(
                    "duplicate node name '{}'",
                    node.name
                )));
            }
            if by_id.insert(node.id, node.name.clone()).is_some() {
                return Err(CudaError::invalid_input(format!(
                    "duplicate node id {}",
                    node.id
                )));
            }
        }

        Ok(NodeRegistry { by_name, by_id })
    }
}

impl CompiledTopology {
    pub fn dense_matrix(&self) -> Vec<f32> {
        let mut matrix = vec![Q_BOTTOM; MATRIX_LEN];
        for edge in &self.edges {
            let src = edge.src as usize;
            let dst = edge.dst as usize;
            matrix[src * NODE_COUNT + dst] = matrix[src * NODE_COUNT + dst].max(edge.value);
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
