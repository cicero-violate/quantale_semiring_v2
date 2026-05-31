//! Compact workflow text compiler into data topology.

use std::collections::BTreeMap;

use crate::error::CudaError;
use crate::topology::{GraphTopology, TopologyNode, TopologyTransition};
use crate::types::QuantaleWeight;

pub fn compile_workflow_dsl(matrix_name: &str, input: &str) -> Result<GraphTopology, CudaError> {
    let mut node_ids = BTreeMap::<String, usize>::new();
    let mut transitions = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let transition = parse_line(line, line_index + 1)?;
        let next_id = node_ids.len();
        node_ids.entry(transition.from.clone()).or_insert(next_id);
        let next_id = node_ids.len();
        node_ids.entry(transition.to.clone()).or_insert(next_id);
        transitions.push(transition);
    }
    let nodes = node_ids
        .into_iter()
        .map(|(name, id)| TopologyNode {
            id,
            node_type: infer_node_type(&name).to_string(),
            name,
        })
        .collect();
    Ok(GraphTopology {
        matrix_name: matrix_name.to_string(),
        nodes,
        transitions,
        pages: Vec::new(),
    })
}

fn parse_line(line: &str, line_number: usize) -> Result<TopologyTransition, CudaError> {
    let (_, after_colon) = line
        .split_once(':')
        .ok_or_else(|| CudaError::invalid_input(format!("line {line_number}: missing ':'")))?;
    let (left, action) = after_colon
        .split_once("=>")
        .ok_or_else(|| CudaError::invalid_input(format!("line {line_number}: missing action")))?;
    let (from, rest) = left.split_once("->").ok_or_else(|| {
        CudaError::invalid_input(format!("line {line_number}: missing transition"))
    })?;
    let open = rest
        .find('[')
        .ok_or_else(|| CudaError::invalid_input(format!("line {line_number}: missing weight")))?;
    let close = rest[open + 1..].find(']').ok_or_else(|| {
        CudaError::invalid_input(format!("line {line_number}: unterminated weight"))
    })? + open
        + 1;
    let weight = rest[open + 1..close]
        .trim()
        .parse::<f32>()
        .map_err(|error| {
            CudaError::invalid_input(format!("line {line_number}: bad weight: {error}"))
        })?;
    Ok(TopologyTransition {
        from: from.trim().to_string(),
        to: rest[..open].trim().to_string(),
        default_weight: QuantaleWeight::new(weight),
        confidence: None,
        cost: None,
        safety: None,
        policy_effect: Some(action.trim().to_string()),
    })
}

fn infer_node_type(name: &str) -> &'static str {
    if name.starts_with("State::") {
        "State"
    } else if name.starts_with("Control::") {
        "Control"
    } else if name.starts_with("Event::") {
        "Event"
    } else {
        "External"
    }
}
