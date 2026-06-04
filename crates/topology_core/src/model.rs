use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyNode {
    pub id: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub action: Option<String>,
}

impl TopologyNode {
    /// True when this node is a GPU-resident compute region.
    ///
    /// GPU regions are dispatched through the device dispatch table rather than
    /// the CPU operator executor. They do not require an entry in
    /// `operators.generated.json`.
    pub fn is_gpu_region(&self) -> bool {
        self.node_type == "gpu_region"
    }

    /// True when this node belongs to the CPU control/IO graph.
    pub fn is_control_io(&self) -> bool {
        let t = self.node_type.as_str();
        t == "State" || t == "Control" || t == "Event"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyTransition {
    pub from: String,
    pub to: String,
    /// Legacy scalar weight.  New compiled transitions omit this field; the
    /// explicit `confidence`/`cost`/`safety` triple is the source of truth.
    /// Kept for backward compat with hand-authored topology.json transitions.
    #[serde(default)]
    pub default_weight: f32,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub cost: Option<f32>,
    #[serde(default)]
    pub safety: Option<f32>,
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
    /// CKA `par` groups compiled from source topology programs.
    /// Each group is a list of node names that can execute in parallel when
    /// effect-independent.  Emitted by `build_overlay_assets` into
    /// `topology.generated.json`; absent in other topology files.
    #[serde(default)]
    pub parallel_groups: Vec<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledTopology {
    pub matrix_name: String,
    pub node_count: usize,
    pub matrix_len: usize,
    pub registry: crate::NodeRegistry,
    pub transitions: Vec<CompiledTransition>,
    pub pages: Vec<TopologyPage>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledTransition {
    pub src: usize,
    pub dst: usize,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
}
