use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::programs::compile_source_programs;
use crate::{TopologyNode, TopologyTransition};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TopologyOverlay {
    #[serde(default)]
    pub nodes: Vec<TopologyNode>,
    #[serde(default)]
    pub transitions: Vec<TopologyTransition>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OperatorOverlay {
    #[serde(default)]
    pub operators: Vec<Value>,
}

pub fn build_overlay_assets(root: impl AsRef<Path>) -> Result<(), String> {
    let root = root.as_ref();
    let mut topology = read_json(root.join("assets/topology.json"))?;
    let mut operators = read_json(root.join("assets/operators.json"))?;

    let mut nodes = take_array(&mut topology, "nodes")?;
    let mut transitions = take_array(&mut topology, "transitions")?;
    let mut operator_contracts = take_array(&mut operators, "operators")?;

    for overlay in read_overlay_dir(root.join("overlays/topology"))? {
        let mut overlay = overlay;
        nodes.extend(take_array_default(&mut overlay, "nodes")?);
        transitions.extend(take_array_default(&mut overlay, "transitions")?);
    }

    for overlay in read_overlay_dir(root.join("overlays/operators"))? {
        let mut overlay = overlay;
        operator_contracts.extend(take_array_default(&mut overlay, "operators")?);
    }

    reject_duplicate_node_names(&nodes)?;
    assign_dense_ids(&mut nodes)?;

    // ── Source topology programs ──────────────────────────────────────────────
    // If assets/topology.source.json exists, compile its programs into
    // additional flat transitions and parallel group metadata.
    // Transitions that already exist in the flat baseline are skipped.
    let parallel_groups =
        extend_from_source_topology(root, &nodes, &mut transitions)?;

    reject_duplicate_transitions(&transitions)?;
    reject_unknown_transition_endpoints(&nodes, &transitions)?;
    reject_duplicate_operator_contracts(&operator_contracts)?;
    reject_operator_nodes_without_contracts(&nodes, &operator_contracts)?;

    topology["nodes"] = Value::Array(nodes);
    topology["transitions"] = Value::Array(transitions);
    if !parallel_groups.is_empty() {
        topology["parallel_groups"] = Value::Array(
            parallel_groups
                .into_iter()
                .map(|group| {
                    Value::Array(group.into_iter().map(Value::String).collect())
                })
                .collect(),
        );
    }
    operators["operators"] = Value::Array(operator_contracts);

    write_json(root.join("assets/topology.generated.json"), &topology)?;
    write_json(root.join("assets/operators.generated.json"), &operators)?;
    Ok(())
}

// ── Source topology integration ───────────────────────────────────────────────

/// Read `assets/topology.source.json`, compile its programs into flat
/// transitions, and append any genuinely-new edges to `transitions`.
/// Returns the collected parallel group node-name lists.
fn extend_from_source_topology(
    root: &Path,
    nodes: &[Value],
    transitions: &mut Vec<Value>,
) -> Result<Vec<Vec<String>>, String> {
    let source_path = root.join("assets/topology.source.json");
    if !source_path.exists() {
        return Ok(Vec::new());
    }

    let source = read_json(source_path)?;

    let existing: BTreeSet<(String, String)> = transitions
        .iter()
        .filter_map(|t| {
            let from = t.get("from")?.as_str()?.to_string();
            let to = t.get("to")?.as_str()?.to_string();
            Some((from, to))
        })
        .collect();

    let known: BTreeSet<String> = nodes
        .iter()
        .filter_map(|n| n.get("name")?.as_str().map(str::to_string))
        .collect();

    let (new_transitions, parallel_groups) =
        compile_source_programs(&source, &existing, &known)?;

    transitions.extend(new_transitions);
    Ok(parallel_groups)
}

fn read_json(path: PathBuf) -> Result<Value, String> {
    let input =
        fs::read_to_string(&path).map_err(|error| format!("read '{}': {error}", path.display()))?;
    serde_json::from_str(&input).map_err(|error| format!("parse '{}': {error}", path.display()))
}

fn write_json(path: PathBuf, value: &Value) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|error| format!("serialize '{}': {error}", path.display()))?;
    let output = format!("{output}\n");
    if fs::read_to_string(&path).ok().as_deref() == Some(output.as_str()) {
        return Ok(());
    }
    fs::write(&path, output).map_err(|error| format!("write '{}': {error}", path.display()))
}

fn read_overlay_dir(path: PathBuf) -> Result<Vec<Value>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut files = fs::read_dir(&path)
        .map_err(|error| format!("read overlay dir '{}': {error}", path.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read overlay dir '{}': {error}", path.display()))?;
    files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"));
    files.sort();
    files.into_iter().map(read_json).collect()
}

fn take_array(value: &mut Value, field: &str) -> Result<Vec<Value>, String> {
    value
        .get_mut(field)
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .ok_or_else(|| format!("missing array field '{field}'"))
}

fn take_array_default(value: &mut Value, field: &str) -> Result<Vec<Value>, String> {
    match value.get_mut(field) {
        None => Ok(Vec::new()),
        Some(array) => array
            .as_array_mut()
            .map(std::mem::take)
            .ok_or_else(|| format!("field '{field}' must be an array")),
    }
}

fn reject_duplicate_node_names(nodes: &[Value]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for node in nodes {
        let name = string_field(node, "name", "node")?;
        if !seen.insert(name.to_string()) {
            return Err(format!("duplicate node name: {name}"));
        }
    }
    Ok(())
}

fn assign_dense_ids(nodes: &mut [Value]) -> Result<(), String> {
    for (idx, node) in nodes.iter_mut().enumerate() {
        let Some(object) = node.as_object_mut() else {
            return Err("node must be an object".to_string());
        };
        object.insert("id".to_string(), Value::from(idx));
    }
    Ok(())
}

fn reject_duplicate_transitions(transitions: &[Value]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for transition in transitions {
        let src = string_field(transition, "from", "transition")?;
        let dst = string_field(transition, "to", "transition")?;
        if !seen.insert((src.to_string(), dst.to_string())) {
            return Err(format!("duplicate transition: {src} -> {dst}"));
        }
    }
    Ok(())
}

fn reject_unknown_transition_endpoints(
    nodes: &[Value],
    transitions: &[Value],
) -> Result<(), String> {
    let names = nodes
        .iter()
        .map(|node| string_field(node, "name", "node").map(str::to_string))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for transition in transitions {
        let src = string_field(transition, "from", "transition")?;
        let dst = string_field(transition, "to", "transition")?;
        if !names.contains(src) {
            return Err(format!("unknown transition source: {src}"));
        }
        if !names.contains(dst) {
            return Err(format!("unknown transition destination: {dst}"));
        }
    }
    Ok(())
}

fn reject_duplicate_operator_contracts(operators: &[Value]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for operator in operators {
        let name = string_field(operator, "node_name", "operator")?;
        if !seen.insert(name.to_string()) {
            return Err(format!("duplicate operator contract: {name}"));
        }
    }
    Ok(())
}

fn reject_operator_nodes_without_contracts(
    nodes: &[Value],
    operators: &[Value],
) -> Result<(), String> {
    let contracted = operators
        .iter()
        .map(|operator| string_field(operator, "node_name", "operator").map(str::to_string))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for node in nodes {
        let name = string_field(node, "name", "node")?;
        let node_type = node.get("type").and_then(Value::as_str);
        let action = node.get("action").and_then(Value::as_str);
        let is_operator_node =
            node_type == Some("Execution") || action.is_some_and(|action| action != "halt");
        if is_operator_node && !contracted.contains(name) {
            return Err(format!("operator node without contract: {name}"));
        }
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{context} missing non-empty string field '{field}'"))
}
