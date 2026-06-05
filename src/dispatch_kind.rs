//! Node-level dispatch-kind compiler for the GPU-native scheduler.
//!
//! The CUDA scheduler consumes a `DISPATCH_KIND_*` table indexed by topology
//! node id.  This module is the canonical bridge from topology/operator
//! metadata into that device table.

use serde_json::Value;

use crate::config::SystemConfig;
use crate::tensor::{
    DISPATCH_KIND_ABSTRACT_DEVICE, DISPATCH_KIND_EXTERNAL_IO, DISPATCH_KIND_EXTERNAL_PROCESS,
    DISPATCH_KIND_HF_DEVICE, DISPATCH_KIND_UNSUPPORTED, TENSOR_NODE_COUNT,
};
use crate::topology::{GraphTopology, TopologyNode};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchKindSummary {
    pub hf_device: usize,
    pub abstract_device: usize,
    pub external_process: usize,
    pub external_io: usize,
    pub unsupported: usize,
}

impl DispatchKindSummary {
    pub fn from_kinds(kinds: &[i32]) -> Self {
        let mut summary = Self::default();
        for &kind in kinds {
            match kind {
                DISPATCH_KIND_HF_DEVICE => summary.hf_device += 1,
                DISPATCH_KIND_ABSTRACT_DEVICE => summary.abstract_device += 1,
                DISPATCH_KIND_EXTERNAL_PROCESS => summary.external_process += 1,
                DISPATCH_KIND_EXTERNAL_IO => summary.external_io += 1,
                DISPATCH_KIND_UNSUPPORTED => summary.unsupported += 1,
                _ => summary.unsupported += 1,
            }
        }
        summary
    }
}

pub fn build_node_dispatch_kinds(topology: &GraphTopology, config: &SystemConfig) -> Vec<i32> {
    let mut kinds = vec![DISPATCH_KIND_UNSUPPORTED; TENSOR_NODE_COUNT];

    for node in &topology.nodes {
        if node.id >= TENSOR_NODE_COUNT {
            continue;
        }
        kinds[node.id] = classify_node_dispatch_kind(node, config);
    }

    kinds
}

fn classify_node_dispatch_kind(node: &TopologyNode, config: &SystemConfig) -> i32 {
    let name = node.name.as_str();

    if name == "Control::Halt" || node.is_gpu_region() || config.hot_region_registry.is_hot(name) {
        return DISPATCH_KIND_HF_DEVICE;
    }

    if config.fusion_dispatch.is_fusion_entry(name) {
        return DISPATCH_KIND_HF_DEVICE;
    }

    if config.abstract_device_coverage.is_covered(name) {
        return DISPATCH_KIND_ABSTRACT_DEVICE;
    }

    let Some(operator) = config.operator_registry.get(name) else {
        return if is_device_control_node(node) {
            DISPATCH_KIND_HF_DEVICE
        } else {
            DISPATCH_KIND_UNSUPPORTED
        };
    };

    if operator_is_external_io(name, operator) {
        return DISPATCH_KIND_EXTERNAL_IO;
    }

    if operator_is_external_process(operator) {
        return DISPATCH_KIND_EXTERNAL_PROCESS;
    }

    if is_device_control_node(node) {
        return DISPATCH_KIND_HF_DEVICE;
    }

    DISPATCH_KIND_UNSUPPORTED
}

fn is_device_control_node(node: &TopologyNode) -> bool {
    matches!(
        node.node_type.as_str(),
        "Control" | "Event" | "policy_node" | "event_node"
    )
}

fn operator_is_external_process(operator: &Value) -> bool {
    let executable = operator
        .get("executable")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        executable.as_str(),
        "python" | "python3" | "cargo" | "patch" | "node" | "npm" | "bash" | "sh"
    )
}

fn operator_is_external_io(node_name: &str, operator: &Value) -> bool {
    let mut haystack = String::from(node_name);
    if let Some(executable) = operator.get("executable").and_then(Value::as_str) {
        haystack.push(' ');
        haystack.push_str(executable);
    }
    if let Some(args) = operator.get("static_args").and_then(Value::as_array) {
        for arg in args.iter().filter_map(Value::as_str) {
            haystack.push(' ');
            haystack.push_str(arg);
        }
    }
    let haystack = haystack.to_ascii_lowercase();
    haystack.contains("market_feed")
        || haystack.contains("external_io")
        || haystack.contains("network")
        || haystack.contains("websocket")
        || haystack.contains("fetch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SystemConfig;
    use crate::topology::TopologyRuntime;

    fn default_table() -> (TopologyRuntime, Vec<i32>) {
        let config = SystemConfig::default();
        let topology = TopologyRuntime::load_checked_default().unwrap();
        let kinds = build_node_dispatch_kinds(&topology.document, &config);
        (topology, kinds)
    }

    #[test]
    fn dispatch_kind_table_has_tensor_capacity() {
        let (_, kinds) = default_table();
        assert_eq!(kinds.len(), TENSOR_NODE_COUNT);
    }

    #[test]
    fn default_dispatch_table_marks_market_feed_as_external_io() {
        let (topology, kinds) = default_table();
        let id = topology.registry().id_of("State::MarketFeed").unwrap();
        assert_eq!(kinds[id], DISPATCH_KIND_EXTERNAL_IO);
    }

    #[test]
    fn default_dispatch_table_marks_llm_process_nodes_as_external_process() {
        let (topology, kinds) = default_table();
        let analysis_id = topology.registry().id_of("State::AnalysisPlan").unwrap();
        let trade_id = topology.registry().id_of("State::TradePlan").unwrap();
        assert_eq!(kinds[analysis_id], DISPATCH_KIND_EXTERNAL_PROCESS);
        assert_eq!(kinds[trade_id], DISPATCH_KIND_EXTERNAL_PROCESS);
    }

    #[test]
    fn default_dispatch_table_keeps_halt_device_handled() {
        let (topology, kinds) = default_table();
        let id = topology.registry().id_of("Control::Halt").unwrap();
        assert_eq!(kinds[id], DISPATCH_KIND_HF_DEVICE);
    }

    #[test]
    fn default_dispatch_summary_exposes_external_work() {
        let (_, kinds) = default_table();
        let summary = DispatchKindSummary::from_kinds(&kinds);
        assert!(summary.external_io > 0);
        assert!(summary.external_process > 0);
    }
}
