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

#[derive(Debug, Deserialize)]
struct NamedTensorEdge {
    from: String,
    to: String,
    confidence: f32,
    cost: f32,
    safety: f32,
}

/// Minimum cost for any learned edge.  Prevents zero-cost learned cycles from
/// dominating semiring path search.  Only topology axioms may have cost = 0.
const LEARNED_EDGE_COST_FLOOR: f32 = 0.001;

pub fn load_learned_tensor_edges(
    path: impl AsRef<Path>,
    registry: &NodeRegistry,
    allowed_edges: &[TensorEdge],
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
                edge.confidence.clamp(0.0, 1.0),
                // Clamp to floor so learned edges never produce zero-cost cycles
                edge.cost.max(LEARNED_EDGE_COST_FLOOR),
                edge.safety.clamp(0.0, 1.0),
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

        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &topology.tensor_edges).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].confidence, 0.8);
        assert_eq!(edges[0].cost, LEARNED_EDGE_COST_FLOOR);
        assert_eq!(edges[0].safety, 0.9);
    }
}
