//! Learned tensor-edge checkpoint ingress.
//!
//! JSONL state is evidence/checkpoint storage only. The runtime representation
//! of learned path weights is still the GPU tensor after these sparse edges are
//! embedded.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::tensor::TensorEdge;
use crate::topology::NodeRegistry;

const DEFAULT_LEARNING_POLICY_JSON: &str = include_str!("../assets/learning_policy.json");

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LearningPolicy {
    pub learned_edge_cost_floor: f32,
    pub confidence_clamp: [f32; 2],
    pub safety_clamp: [f32; 2],
}

impl Default for LearningPolicy {
    fn default() -> Self {
        serde_json::from_str(DEFAULT_LEARNING_POLICY_JSON)
            .expect("embedded learning_policy.json is valid")
    }
}

impl LearningPolicy {
    pub fn default_asset() -> Self {
        fs::read_to_string("assets/learning_policy.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
}

#[derive(Debug, Deserialize)]
struct NamedTensorEdge {
    from: String,
    to: String,
    confidence: f32,
    cost: f32,
    safety: f32,
}

pub fn load_learned_tensor_edges(
    path: impl AsRef<Path>,
    registry: &NodeRegistry,
    allowed_edges: &[TensorEdge],
    policy: &LearningPolicy,
) -> Result<Vec<TensorEdge>, String> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }

    let input = fs::read_to_string(path)
        .map_err(|error| format!("read learned edges '{}': {error}", path.display()))?;
    let allowed: BTreeSet<(i32, i32)> = allowed_edges
        .iter()
        .map(|edge| (edge.src, edge.dst))
        .collect();

    // Use a BTreeMap keyed by (src, dst) so that duplicate records for the
    // same endpoint pair are collapsed: the last record in the file wins
    // (latest-wins merge).  This prevents learned_edges.jsonl from growing
    // unboundedly and embedding the same edge dozens of times.
    let mut deduped: BTreeMap<(i32, i32), TensorEdge> = BTreeMap::new();

    for (index, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("parse learned edge {}: {error}", index + 1))?;
        let edge_value = value.get("edge").unwrap_or(&value).clone();
        let edge: NamedTensorEdge = serde_json::from_value(edge_value)
            .map_err(|error| format!("decode learned edge {}: {error}", index + 1))?;

        let src = registry.id_of(&edge.from).ok_or_else(|| {
            format!(
                "learned edge {} has unknown source '{}'",
                index + 1,
                edge.from
            )
        })?;
        let dst = registry.id_of(&edge.to).ok_or_else(|| {
            format!(
                "learned edge {} has unknown destination '{}'",
                index + 1,
                edge.to
            )
        })?;
        if !edge.cost.is_finite() || edge.cost < 0.0 {
            return Err(format!("learned edge {} has invalid cost", index + 1));
        }
        let src = src as i32;
        let dst = dst as i32;
        if !allowed.contains(&(src, dst)) {
            continue;
        }
        deduped.insert(
            (src, dst),
            TensorEdge::new(
                src,
                dst,
                edge.confidence.clamp(policy.confidence_clamp[0], policy.confidence_clamp[1]),
                // Clamp to floor so learned edges never produce zero-cost cycles
                edge.cost.max(policy.learned_edge_cost_floor),
                edge.safety.clamp(policy.safety_clamp[0], policy.safety_clamp[1]),
            ),
        );
    }
    Ok(deduped.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::GraphTopology;

    #[test]
    fn load_learned_edges_dedupes_latest_and_clamps_cost_floor() {
        let topology = GraphTopology::from_json_str(
            r#"{
                "matrix_name":"test",
                "nodes":[
                    {"id":0,"name":"State::A","type":"State"},
                    {"id":1,"name":"State::B","type":"State"}
                ],
                "transitions":[
                    {"from":"State::A","to":"State::B","default_weight":0.9,"cost":0.5}
                ],
                "pages":[]
            }"#,
        )
        .unwrap()
        .compile()
        .unwrap();
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        let path = std::env::temp_dir().join(format!(
            "learned_edges_test_{}_{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &path,
            concat!(
                r#"{"from":"State::A","to":"State::B","confidence":0.2,"cost":0.7,"safety":0.4}"#,
                "\n",
                r#"{"from":"State::A","to":"State::B","confidence":0.8,"cost":0.0,"safety":0.9}"#,
                "\n"
            ),
        )
        .unwrap();

        let policy = LearningPolicy::default();
        let edges = load_learned_tensor_edges(
            &path,
            &topology.registry,
            &static_edges,
            &policy,
        )
        .unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].confidence, 0.8);
        assert_eq!(edges[0].cost, policy.learned_edge_cost_floor);
        assert_eq!(edges[0].safety, 0.9);
    }
}
