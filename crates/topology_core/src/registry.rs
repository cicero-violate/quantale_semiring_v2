use std::collections::{BTreeMap, BTreeSet};

use crate::{GraphTopology, TopologyError, TopologyPage};

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
    pub(crate) fn build_registry(&self) -> Result<NodeRegistry, TopologyError> {
        if self.nodes.is_empty() {
            return Err(TopologyError::invalid_input("no nodes"));
        }

        let mut by_name = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        let mut actions = BTreeMap::new();
        for node in &self.nodes {
            if by_name.insert(node.name.clone(), node.id).is_some() {
                return Err(TopologyError::invalid_input(format!(
                    "duplicate node name '{}'",
                    node.name
                )));
            }
            if by_id.insert(node.id, node.name.clone()).is_some() {
                return Err(TopologyError::invalid_input(format!(
                    "duplicate node id {}",
                    node.id
                )));
            }
            if let Some(action) = &node.action {
                actions.insert(node.id, action.clone());
            }
        }

        for expected_id in 0..self.nodes.len() {
            if !by_id.contains_key(&expected_id) {
                return Err(TopologyError::invalid_input(format!(
                    "topology node ids must be dense; missing id {expected_id}"
                )));
            }
        }

        for transition in &self.transitions {
            if !by_name.contains_key(&transition.from) {
                return Err(TopologyError::invalid_input(format!(
                    "transition source '{}' missing",
                    transition.from
                )));
            }
            if !by_name.contains_key(&transition.to) {
                return Err(TopologyError::invalid_input(format!(
                    "transition destination '{}' missing",
                    transition.to
                )));
            }
        }

        validate_pages(
            &NodeRegistry {
                by_name: by_name.clone(),
                by_id: by_id.clone(),
                actions: actions.clone(),
            },
            &self.pages,
        )?;

        Ok(NodeRegistry {
            by_name,
            by_id,
            actions,
        })
    }
}

fn validate_pages(registry: &NodeRegistry, pages: &[TopologyPage]) -> Result<(), TopologyError> {
    let mut names = BTreeSet::new();
    for page in pages {
        if !names.insert(page.name.clone()) {
            return Err(TopologyError::invalid_input(format!(
                "duplicate page '{}'",
                page.name
            )));
        }
        for node in &page.node_names {
            if registry.id_of(node).is_none() {
                return Err(TopologyError::invalid_input(format!(
                    "page '{}' references unknown node '{}'",
                    page.name, node
                )));
            }
        }
    }
    Ok(())
}
