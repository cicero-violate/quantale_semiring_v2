//! Tensor LLM plan compilation from structured JSON edge arrays.
//!
//! The LLM operator emits data-only tensor edge proposals. This module validates
//! node names against `assets/topology.json` before any edge reaches VRAM.

use serde::Deserialize;

use crate::tensor::TensorEdge;
use crate::topology::GraphTopology;

#[derive(Deserialize)]
struct TensorPlanEdge {
    from: String,
    to: String,
    confidence: f32,
    cost: f32,
    safety: f32,
}

/// Parse and validate an LLM-generated tensor edge array against the live topology.
///
/// Expected format:
/// ```json
/// [
///   {
///     "from": "State::Plan",
///     "to": "State::Optimize",
///     "confidence": 0.95,
///     "cost": 2.0,
///     "safety": 0.90
///   }
/// ]
/// ```
///
/// Tensor plans require explicit layer values. They do not accept the legacy
/// scalar `weight` field because the tensor engine needs confidence, cost, and
/// safety as independent algebraic quantities.
pub fn compile_tensor_plan(raw: &str) -> Result<Vec<TensorEdge>, String> {
    let payload = extract_json_array(raw.trim());
    if payload.is_empty() {
        return Ok(Vec::new());
    }

    let plan_edges: Vec<TensorPlanEdge> =
        serde_json::from_str(payload).map_err(|e| format!("tensor plan JSON parse: {e}"))?;

    if plan_edges.is_empty() {
        return Ok(Vec::new());
    }

    let registry = GraphTopology::default_asset()
        .map_err(|e| format!("topology load: {e}"))?
        .compile()
        .map_err(|e| format!("topology compile: {e}"))?
        .registry;

    let mut edges = Vec::with_capacity(plan_edges.len());
    for pe in &plan_edges {
        let src = registry
            .id_of(&pe.from)
            .ok_or_else(|| format!("unknown node '{}' in tensor LLM plan", pe.from))?;
        let dst = registry
            .id_of(&pe.to)
            .ok_or_else(|| format!("unknown node '{}' in tensor LLM plan", pe.to))?;
        if !pe.cost.is_finite() || pe.cost < 0.0 {
            return Err(format!(
                "invalid nonnegative finite cost for edge '{}'=>'{}'",
                pe.from, pe.to
            ));
        }
        edges.push(TensorEdge::new(
            src as i32,
            dst as i32,
            pe.confidence.clamp(0.0, 1.0),
            pe.cost,
            pe.safety.clamp(0.0, 1.0),
        ));
    }

    Ok(edges)
}

/// Extract the outermost JSON array from raw LLM output, stripping markdown
/// code fences and any prose preamble the model may have prepended.
fn extract_json_array(raw: &str) -> &str {
    let stripped = raw
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    match (stripped.find('['), stripped.rfind(']')) {
        (Some(start), Some(end)) if end >= start => &stripped[start..=end],
        _ => stripped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_tensor_plan_returns_tensor_edges() {
        let raw = r#"[
            {"from": "State::Plan", "to": "State::Optimize", "confidence": 0.95, "cost": 2.0, "safety": 0.90},
            {"from": "State::Optimize", "to": "State::Execute", "confidence": 0.90, "cost": 1.0, "safety": 0.85}
        ]"#;
        let edges = compile_tensor_plan(raw).unwrap();
        assert_eq!(edges.len(), 2);
        assert!((edges[0].confidence - 0.95).abs() < f32::EPSILON);
        assert!((edges[0].cost - 2.0).abs() < f32::EPSILON);
        assert!((edges[0].safety - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn compile_empty_tensor_plan_returns_empty() {
        assert!(compile_tensor_plan("").unwrap().is_empty());
        assert!(compile_tensor_plan("  ").unwrap().is_empty());
        assert!(compile_tensor_plan("[]").unwrap().is_empty());
    }

    #[test]
    fn compile_tensor_plan_rejects_legacy_scalar_weight() {
        let raw = r#"[{"from":"State::Plan","to":"State::Execute","weight":0.9}]"#;
        assert!(compile_tensor_plan(raw).is_err());
    }

    #[test]
    fn compile_tensor_plan_rejects_bad_cost() {
        let raw = r#"[{"from":"State::Plan","to":"State::Execute","confidence":0.9,"cost":-1.0,"safety":0.8}]"#;
        assert!(compile_tensor_plan(raw).is_err());
    }

    #[test]
    fn compile_tensor_plan_strips_markdown_fences() {
        let raw = "```json
[{\"from\":\"State::Plan\",\"to\":\"State::Execute\",\"confidence\":0.8,\"cost\":1.0,\"safety\":0.7}]
```";
        let edges = compile_tensor_plan(raw).unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn compile_tensor_plan_rejects_unknown_node() {
        let raw = r#"[{"from":"State::Missing","to":"State::Execute","confidence":0.9,"cost":1.0,"safety":0.8}]"#;
        assert!(compile_tensor_plan(raw).is_err());
    }
}
