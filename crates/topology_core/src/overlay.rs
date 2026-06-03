use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::fusion::{FusionRegion, partition_fusible_regions};
use crate::programs::{
    build_effects_map, compile_source_programs, emit_patterns_compat,
    validate_boundary_governance, validate_kernel_slot_purity,
    validate_known_backends, validate_quantale_layers,
    validate_source_node_effects, validate_unique_source_node_names,
};
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
        extend_from_source_topology(root, &nodes, &mut transitions, &operator_contracts)?;

    reject_duplicate_transitions(&transitions)?;
    reject_unknown_transition_endpoints(&nodes, &transitions)?;
    reject_duplicate_operator_contracts(&operator_contracts)?;
    reject_operator_nodes_without_contracts(&nodes, &operator_contracts)?;

    // Phase 6: partition fusible regions from the complete merged transition set
    // before transitions are moved into the topology Value.
    let fusion_regions: Vec<FusionRegion> = {
        let source_path = root.join("assets/topology.source.json");
        if source_path.exists() {
            let source = read_json(source_path)?;
            partition_fusible_regions(&source, &transitions)
        } else {
            vec![]
        }
    };

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

    // Phase 6: emit topology.fusion.json when fusible regions were found.
    if !fusion_regions.is_empty() {
        let fusion_json = serde_json::json!({
            "regions": fusion_regions.iter().map(FusionRegion::to_json).collect::<Vec<_>>()
        });
        write_json(root.join("assets/topology.fusion.json"), &fusion_json)?;
    }

    Ok(())
}

// ── Source topology integration ───────────────────────────────────────────────

/// Read `assets/topology.source.json`, compile its programs into flat
/// transitions, and append any genuinely-new edges to `transitions`.
/// Also emits `assets/patterns.source.json` (patterns.json-compatible output
/// generated from source programs — Phase 3 proof that patterns.json can be
/// replaced by topology.source.json).
/// Returns the collected parallel group node-name lists.
fn extend_from_source_topology(
    root: &Path,
    nodes: &[Value],
    transitions: &mut Vec<Value>,
    operator_contracts: &[Value],
) -> Result<Vec<Vec<String>>, String> {
    let source_path = root.join("assets/topology.source.json");
    if !source_path.exists() {
        return Ok(Vec::new());
    }

    let source = read_json(source_path)?;

    // Phase 6: validate source node uniqueness and known backends.
    let name_violations = validate_unique_source_node_names(&source);
    if !name_violations.is_empty() {
        return Err(format!(
            "topology.source.json node name violations ({} total):\n{}",
            name_violations.len(),
            name_violations.join("\n")
        ));
    }
    let backend_violations = validate_known_backends(&source);
    if !backend_violations.is_empty() {
        return Err(format!(
            "topology.source.json backend violations ({} total):\n{}",
            backend_violations.len(),
            backend_violations.join("\n")
        ));
    }

    // Phase 2: validate declared node effects against slots/resources.
    let violations = validate_source_node_effects(&source);
    if !violations.is_empty() {
        return Err(format!(
            "topology.source.json slot/resource violations ({} total):\n{}",
            violations.len(),
            violations.join("\n")
        ));
    }

    // Phase 4: validate quantale layer declarations and program weights.
    let q_violations = validate_quantale_layers(&source);
    if !q_violations.is_empty() {
        return Err(format!(
            "topology.source.json quantale violations ({} total):\n{}",
            q_violations.len(),
            q_violations.join("\n")
        ));
    }

    // Phase 5: validate boundary governance and kernel slot purity.
    let gov_violations = validate_boundary_governance(&source);
    if !gov_violations.is_empty() {
        return Err(format!(
            "topology.source.json boundary governance violations ({} total):\n{}",
            gov_violations.len(),
            gov_violations.join("\n")
        ));
    }
    let purity_violations = validate_kernel_slot_purity(&source);
    if !purity_violations.is_empty() {
        return Err(format!(
            "topology.source.json kernel purity violations ({} total):\n{}",
            purity_violations.len(),
            purity_violations.join("\n")
        ));
    }

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

    // Phase 3: build the effects map and pass it for par independence checking.
    let source_nodes = source
        .get("nodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let effects_map = build_effects_map(operator_contracts, source_nodes);

    let (new_transitions, parallel_groups) =
        compile_source_programs(&source, &existing, &known, Some(&effects_map))?;

    transitions.extend(new_transitions);

    // Phase 3: emit patterns.source.json (patterns.json-compatible, generated
    // from topology.source.json programs).  This proves patterns.json can be
    // derived from the source topology rather than hand-authored.
    let patterns_compat = emit_patterns_compat(&source);
    write_json(root.join("assets/patterns.source.json"), &patterns_compat)?;

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
