//! LLM plan compilation: structured JSON edge arrays → quantale matrix weights.
//!
//! The LLM operator (call_llm.py) emits a flat JSON array of directed edge
//! proposals. This module validates each node name against the compiled topology
//! and converts the surviving edges into `TransitionEdge` values ready for
//! `CudaWorld::load_edges`. Unknown node names are rejected with an error so
//! hallucinated nodes never reach VRAM.

use serde::Deserialize;

use crate::edge::TransitionEdge;
use crate::topology::GraphTopology;

#[derive(Deserialize)]
struct PlanEdge {
    from: String,
    to: String,
    weight: f32,
}

/// Parse and validate an LLM-generated edge array against the live topology.
///
/// Expected format:
/// ```json
/// [
///   { "from": "State::Plan", "to": "State::Optimize", "weight": 0.95 },
///   { "from": "State::Optimize", "to": "State::Execute", "weight": 0.90 }
/// ]
/// ```
///
/// Returns `Ok(vec![])` for empty input. Returns `Err` if the JSON is
/// malformed or any node name is not declared in `assets/topology.json`.
pub fn compile_llm_plan(raw: &str) -> Result<Vec<TransitionEdge>, String> {
    let payload = extract_json_array(raw.trim());
    if payload.is_empty() {
        return Ok(Vec::new());
    }

    let plan_edges: Vec<PlanEdge> = serde_json::from_str(payload)
        .map_err(|e| format!("plan JSON parse: {e}"))?;

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
            .ok_or_else(|| format!("unknown node '{}' in LLM plan", pe.from))?;
        let dst = registry
            .id_of(&pe.to)
            .ok_or_else(|| format!("unknown node '{}' in LLM plan", pe.to))?;
        edges.push(TransitionEdge::new(src as i32, dst as i32, pe.weight.clamp(0.0, 1.0)));
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
    fn compile_valid_plan_returns_edges() {
        let raw = r#"[
            {"from": "State::Plan", "to": "State::Optimize", "weight": 0.95},
            {"from": "State::Optimize", "to": "State::Execute", "weight": 0.90}
        ]"#;
        let edges = compile_llm_plan(raw).unwrap();
        assert_eq!(edges.len(), 2);
        assert!((edges[0].value - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn compile_empty_input_returns_empty() {
        assert!(compile_llm_plan("").unwrap().is_empty());
        assert!(compile_llm_plan("  ").unwrap().is_empty());
        assert!(compile_llm_plan("[]").unwrap().is_empty());
    }

    #[test]
    fn compile_unknown_node_returns_err() {
        let raw = r#"[{"from": "State::SkyNet", "to": "State::Plan", "weight": 1.0}]"#;
        assert!(compile_llm_plan(raw).is_err());
    }

    #[test]
    fn compile_strips_markdown_fences() {
        let raw = "```json\n[{\"from\": \"State::Plan\", \"to\": \"State::Optimize\", \"weight\": 0.8}]\n```";
        let edges = compile_llm_plan(raw).unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn compile_clamps_weight_above_one() {
        let raw = r#"[{"from": "State::Plan", "to": "State::Optimize", "weight": 99.0}]"#;
        let edges = compile_llm_plan(raw).unwrap();
        assert!((edges[0].value - 1.0).abs() < f32::EPSILON);
    }
}
