//! Learned tensor-edge checkpoint ingress.
//!
//! JSONL state is evidence/checkpoint storage only. The runtime representation
//! of learned path weights is still the GPU tensor after these sparse edges are
//! embedded.

use std::collections::BTreeSet;
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
    let mut edges = Vec::new();
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
        edges.push(TensorEdge::new(
            src,
            dst,
            edge.confidence.clamp(0.0, 1.0),
            edge.cost,
            edge.safety.clamp(0.0, 1.0),
        ));
    }
    Ok(edges)
}
