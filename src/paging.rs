use std::collections::BTreeMap;

use crate::error::CudaError;
use crate::topology::{CompiledTopology, TopologyPage};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixPagePlan {
    pub page_name: String,
    pub node_ids: Vec<usize>,
    pub matrix_len: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatrixPageRegistry {
    pages: BTreeMap<String, MatrixPagePlan>,
}

impl MatrixPageRegistry {
    pub fn from_compiled(topology: &CompiledTopology) -> Result<Self, CudaError> {
        let mut pages = BTreeMap::new();
        for page in &topology.pages {
            let plan = build_page_plan(topology, page)?;
            if pages.insert(plan.page_name.clone(), plan).is_some() {
                return Err(CudaError::invalid_input("duplicate matrix page"));
            }
        }
        Ok(Self { pages })
    }

    pub fn get(&self, page_name: &str) -> Option<&MatrixPagePlan> {
        self.pages.get(page_name)
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

fn build_page_plan(
    topology: &CompiledTopology,
    page: &TopologyPage,
) -> Result<MatrixPagePlan, CudaError> {
    let mut node_ids = Vec::with_capacity(page.node_names.len());
    for node_name in &page.node_names {
        let id = topology.registry.id_of(node_name).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "page '{}' references unknown node '{}'",
                page.name, node_name
            ))
        })?;
        node_ids.push(id);
    }
    node_ids.sort_unstable();
    node_ids.dedup();
    Ok(MatrixPagePlan {
        page_name: page.name.clone(),
        matrix_len: node_ids.len() * node_ids.len(),
        node_ids,
    })
}
