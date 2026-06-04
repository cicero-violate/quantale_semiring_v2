use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::fusion::{FusionRegion, partition_fusible_regions};
use crate::programs::{
    build_effects_map, compile_source_programs, emit_patterns_compat, validate_boundary_governance,
    validate_kernel_slot_purity, validate_known_backends, validate_quantale_layers,
    validate_source_node_effects, validate_unique_source_node_names,
};
pub fn build_overlay_assets(root: impl AsRef<Path>) -> Result<(), String> {
    let root = root.as_ref();
    let source = read_json(root.join("assets/topology.source.json"))?;
    let mut operators = read_json(root.join("assets/operators.json"))?;

    validate_source_topology(&source)?;

    let mut topology = runtime_topology_from_source(&source)?;
    let mut nodes = take_array(&mut topology, "nodes")?;
    let mut transitions = take_array_default(&mut topology, "transitions")?;
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
    // Compile topology.source.json programs into the complete runtime transition
    // set.  Legacy assets/topology.json is not used as an orchestration baseline.
    let parallel_groups =
        extend_from_source_topology(root, &source, &nodes, &mut transitions, &operator_contracts)?;

    reject_duplicate_transitions(&transitions)?;
    reject_unknown_transition_endpoints(&nodes, &transitions)?;
    reject_duplicate_operator_contracts(&operator_contracts)?;
    reject_operator_nodes_without_contracts(&nodes, &operator_contracts)?;

    // Phase 6: partition fusible regions from the complete merged transition set
    // before transitions are moved into the topology Value.
    let fusion_regions: Vec<FusionRegion> = { partition_fusible_regions(&source, &transitions) };

    // Phase 7: emit split topology views while nodes/transitions are still owned.
    // G_s → { G_c (topology.control.json), G_h (topology.hot.json), R_h (regions.hot.json) }
    emit_split_topologies(root, &source, &nodes, &transitions)?;

    topology["nodes"] = Value::Array(nodes);
    topology["transitions"] = Value::Array(transitions);
    if !parallel_groups.is_empty() {
        topology["parallel_groups"] = Value::Array(
            parallel_groups
                .into_iter()
                .map(|group| Value::Array(group.into_iter().map(Value::String).collect()))
                .collect(),
        );
    }
    operators["operators"] = Value::Array(operator_contracts);

    write_json(root.join("assets/topology.generated.json"), &topology)?;
    write_json(root.join("assets/operators.generated.json"), &operators)?;

    // Phase 6: emit topology.fusion.json when fusible regions were found.
    // The file is absent if no linear-chain jit_cuda subgraph exists in the
    // compiled transition set (e.g. shortcut edges break the linear check).
    if !fusion_regions.is_empty() {
        let fusion_json = serde_json::json!({
            "regions": fusion_regions.iter().map(FusionRegion::to_json).collect::<Vec<_>>()
        });
        write_json(root.join("assets/topology.fusion.json"), &fusion_json)?;
    }

    Ok(())
}

// ── Split topology emission (G_s → G_c, G_h, R_h) ────────────────────────────

/// True when a source node is a GPU kernel region: kind=kernel AND runtime.backend=cuda.
fn is_hot_source_node(node: &Value) -> bool {
    node.get("kind").and_then(Value::as_str) == Some("kernel")
        && node
            .get("runtime")
            .and_then(|r| r.get("backend"))
            .and_then(Value::as_str)
            == Some("cuda")
}

/// Top-level driver: emit control, hot, and regions files from source declarations.
fn emit_split_topologies(
    root: &Path,
    source: &Value,
    nodes: &[Value],       // generated nodes (dense IDs already assigned)
    transitions: &[Value], // full compiled transition set
) -> Result<(), String> {
    let source_nodes = source
        .get("nodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    // hot_names_ordered preserves source declaration order (stable region_id assignment).
    // hot_set is the fast-lookup companion.
    let hot_names_ordered: Vec<String> = source_nodes
        .iter()
        .filter(|n| is_hot_source_node(n))
        .filter_map(|n| n.get("name").and_then(Value::as_str).map(str::to_string))
        .collect();
    let hot_set: BTreeSet<String> = hot_names_ordered.iter().cloned().collect();

    let node_count = nodes.len();

    emit_control_topology(root, &hot_set, nodes, transitions)?;
    emit_hot_topology(
        root,
        source,
        &hot_names_ordered,
        &hot_set,
        nodes,
        node_count,
    )?;
    emit_hot_regions(root, source_nodes, &hot_set)?;

    Ok(())
}

/// Emit `topology.control.json`: all non-hot nodes with their compiled transitions.
fn emit_control_topology(
    root: &Path,
    hot_set: &BTreeSet<String>,
    nodes: &[Value],
    transitions: &[Value],
) -> Result<(), String> {
    let ctrl_nodes: Vec<Value> = nodes
        .iter()
        .filter(|n| {
            n.get("name")
                .and_then(Value::as_str)
                .map(|name| !hot_set.contains(name))
                .unwrap_or(true)
        })
        .cloned()
        .collect();

    let ctrl_set: BTreeSet<&str> = ctrl_nodes
        .iter()
        .filter_map(|n| n.get("name").and_then(Value::as_str))
        .collect();

    // Keep only transitions where both endpoints are control nodes.
    let ctrl_transitions: Vec<Value> = transitions
        .iter()
        .filter(|t| {
            let from = t.get("from").and_then(Value::as_str).unwrap_or("");
            let to = t.get("to").and_then(Value::as_str).unwrap_or("");
            ctrl_set.contains(from) && ctrl_set.contains(to)
        })
        .cloned()
        .collect();

    let page_names: Vec<Value> = ctrl_nodes
        .iter()
        .filter_map(|n| n.get("name").and_then(Value::as_str))
        .map(|s| Value::String(s.to_string()))
        .collect();

    let doc = serde_json::json!({
        "matrix_name": "quantale_control",
        "nodes": ctrl_nodes,
        "transitions": ctrl_transitions,
        "pages": [{ "name": "control", "node_names": page_names }]
    });
    write_json(root.join("assets/topology.control.json"), &doc)
}

/// Emit `topology.hot.json`: hot kernel nodes + `Region::CommitReceipt` synthetic terminal.
///
/// Transitions come from source-level hot-to-hot edges; every leaf (no outgoing
/// hot edge) gets an additional edge to `Region::CommitReceipt`.
fn emit_hot_topology(
    root: &Path,
    source: &Value,
    hot_names_ordered: &[String],
    hot_set: &BTreeSet<String>,
    nodes: &[Value],
    node_count: usize,
) -> Result<(), String> {
    // Hot nodes with their generated-topology IDs, in generated-topology order.
    let mut hot_nodes: Vec<Value> = nodes
        .iter()
        .filter(|n| {
            n.get("name")
                .and_then(Value::as_str)
                .map(|name| hot_set.contains(name))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    // Synthetic terminal node: ID = node_count (out of tensor range, routing only).
    hot_nodes.push(serde_json::json!({
        "id":   node_count,
        "name": "Region::CommitReceipt",
        "type": "gpu_region"
    }));

    // Source-level transitions where both endpoints are hot.
    let source_transitions = source
        .get("transitions")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    let mut hot_transitions: Vec<Value> = source_transitions
        .iter()
        .filter(|t| {
            let from = t.get("from").and_then(Value::as_str).unwrap_or("");
            let to = t.get("to").and_then(Value::as_str).unwrap_or("");
            hot_set.contains(from) && hot_set.contains(to)
        })
        .map(strip_default_weight)
        .collect();

    // Leaf nodes = hot nodes with no outgoing hot-to-hot source transition.
    // Collect into owned strings before mutating hot_transitions.
    let has_outgoing: BTreeSet<String> = hot_transitions
        .iter()
        .filter_map(|t| t.get("from").and_then(Value::as_str).map(str::to_string))
        .collect();

    // Add CommitReceipt edges for every leaf, in stable source order.
    let leaf_edges: Vec<Value> = hot_names_ordered
        .iter()
        .filter(|name| !has_outgoing.contains(*name))
        .map(|name| {
            serde_json::json!({
                "from":       name,
                "to":         "Region::CommitReceipt",
                "confidence": 1.0,
                "cost":       0.0,
                "safety":     1.0
            })
        })
        .collect();
    hot_transitions.extend(leaf_edges);

    let page_names: Vec<Value> = hot_nodes
        .iter()
        .filter_map(|n| n.get("name").and_then(Value::as_str))
        .map(|s| Value::String(s.to_string()))
        .collect();

    let doc = serde_json::json!({
        "matrix_name": "quantale_hot",
        "nodes": hot_nodes,
        "transitions": hot_transitions,
        "pages": [{ "name": "hot", "node_names": page_names }]
    });
    write_json(root.join("assets/topology.hot.json"), &doc)
}

/// Emit `regions.hot.json`: slot declarations for each hot kernel node.
///
/// Slot names (reads/writes) are copied verbatim from the source node declarations.
/// `Region::CommitReceipt` is appended as the synthetic terminal with empty slots.
fn emit_hot_regions(
    root: &Path,
    source_nodes: &[Value],
    hot_set: &BTreeSet<String>,
) -> Result<(), String> {
    let mut regions: Vec<Value> = Vec::new();
    let mut region_id: u32 = 0;

    // Emit one entry per hot node, in source declaration order.
    for node in source_nodes {
        let name = match node.get("name").and_then(Value::as_str) {
            Some(n) => n,
            None => continue,
        };
        if !hot_set.contains(name) {
            continue;
        }
        let reads = node
            .get("reads")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![]));
        let writes = node
            .get("writes")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![]));

        regions.push(serde_json::json!({
            "region_id": region_id,
            "name":      name,
            "kind":      "gpu_region",
            "reads":     reads,
            "writes":    writes,
            "kernel":    "jit_fused",
            "pure":      true
        }));
        region_id += 1;
    }

    // Synthetic terminal: no reads, no writes.
    regions.push(serde_json::json!({
        "region_id": region_id,
        "name":      "Region::CommitReceipt",
        "kind":      "gpu_region",
        "reads":     [],
        "writes":    [],
        "kernel":    "terminal",
        "pure":      true
    }));

    write_json(
        root.join("assets/regions.hot.json"),
        &serde_json::json!({ "regions": regions }),
    )
}

// ── Source topology integration ───────────────────────────────────────────────

/// Read `assets/topology.source.json`, compile its programs into flat
/// transitions, and append any genuinely-new edges to `transitions`.
/// Also emits `assets/patterns.source.json` generated from source programs.
/// Returns the collected parallel group node-name lists.
fn extend_from_source_topology(
    root: &Path,
    source: &Value,
    nodes: &[Value],
    transitions: &mut Vec<Value>,
    operator_contracts: &[Value],
) -> Result<Vec<Vec<String>>, String> {
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
        compile_source_programs(source, &existing, &known, Some(&effects_map))?;

    transitions.extend(new_transitions);

    // Phase 3: emit patterns.source.json from topology.source.json programs.
    let patterns_compat = emit_patterns_compat(source);
    write_json(root.join("assets/patterns.source.json"), &patterns_compat)?;

    Ok(parallel_groups)
}

fn validate_source_topology(source: &Value) -> Result<(), String> {
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

    Ok(())
}

fn runtime_topology_from_source(source: &Value) -> Result<Value, String> {
    let mut object = Map::new();
    object.insert(
        "matrix_name".to_string(),
        source
            .get("matrix_name")
            .cloned()
            .unwrap_or_else(|| Value::String("quantale_semiring_v2".to_string())),
    );

    if let Some(version) = source.get("version") {
        object.insert("version".to_string(), version.clone());
        // source_version: stable fingerprint of which source revision was used
        // to generate this runtime artifact.  Format: "v{version}".
        object.insert(
            "source_version".to_string(),
            Value::String(format!("v{}", version)),
        );
    } else {
        // Fallback: node count fingerprint when source has no version field.
        let n = source
            .get("nodes")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        object.insert(
            "source_version".to_string(),
            Value::String(format!("v0.{n}n")),
        );
    }
    if let Some(slots) = source.get("slots") {
        object.insert("slots".to_string(), slots.clone());
    }
    if let Some(resources) = source.get("resources") {
        object.insert("resources".to_string(), resources.clone());
    }
    if let Some(quantale) = source.get("quantale") {
        object.insert("quantale".to_string(), quantale.clone());
    }

    let source_nodes = source
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "topology.source.json missing array field 'nodes'".to_string())?;
    let nodes = source_nodes
        .iter()
        .map(runtime_node_from_source)
        .collect::<Result<Vec<_>, _>>()?;
    object.insert("nodes".to_string(), Value::Array(nodes));

    if let Some(transitions) = source.get("transitions") {
        let transitions = transitions.as_array().ok_or_else(|| {
            "topology.source.json field 'transitions' must be an array".to_string()
        })?;
        object.insert(
            "transitions".to_string(),
            Value::Array(transitions.iter().map(strip_default_weight).collect()),
        );
    }
    if let Some(pages) = source.get("pages") {
        object.insert("pages".to_string(), pages.clone());
    } else {
        object.insert("pages".to_string(), default_pages(source_nodes));
    }

    Ok(Value::Object(object))
}

fn runtime_node_from_source(node: &Value) -> Result<Value, String> {
    let name = string_field(node, "name", "source node")?;
    let mut object = Map::new();
    object.insert("id".to_string(), Value::from(0));
    object.insert("name".to_string(), Value::String(name.to_string()));
    object.insert(
        "type".to_string(),
        Value::String(node_type_from_name(name).to_string()),
    );
    if let Some(action) = action_from_name(name) {
        object.insert("action".to_string(), Value::String(action.to_string()));
    }
    Ok(Value::Object(object))
}

fn strip_default_weight(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("default_weight");
    }
    value
}

fn node_type_from_name(name: &str) -> &str {
    name.split_once("::")
        .map(|(prefix, _)| prefix)
        .unwrap_or("State")
}

fn action_from_name(name: &str) -> Option<&'static str> {
    match name {
        "State::Execute" => Some("execute"),
        "Control::Retry" => Some("retry"),
        "Control::Repair" => Some("repair"),
        "Control::Commit" => Some("commit"),
        "Control::Rollback" => Some("rollback"),
        "Control::Halt" => Some("halt"),
        _ => None,
    }
}

fn default_pages(source_nodes: &[Value]) -> Value {
    let node_names = source_nodes
        .iter()
        .filter_map(|node| node.get("name").and_then(Value::as_str))
        .filter(|name| !name.starts_with("Analysis::") && !name.starts_with("Execution::"))
        .map(|name| Value::String(name.to_string()))
        .collect::<Vec<_>>();
    Value::Array(vec![serde_json::json!({
        "name": "main",
        "node_names": node_names
    })])
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn build_overlay_emits_all_generated_runtime_assets() {
        let root = std::env::temp_dir().join(format!(
            "quantale_overlay_test_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let assets = root.join("assets");
        fs::create_dir_all(&assets).unwrap();
        // Resolve asset paths relative to the workspace root via CARGO_MANIFEST_DIR.
        let workspace_assets =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        fs::copy(
            workspace_assets.join("topology.source.json"),
            assets.join("topology.source.json"),
        )
        .unwrap();
        fs::copy(
            workspace_assets.join("operators.json"),
            assets.join("operators.json"),
        )
        .unwrap();

        build_overlay_assets(&root).unwrap();

        for asset in [
            "topology.generated.json",
            "operators.generated.json",
            "patterns.source.json",
            "topology.control.json",
            "topology.hot.json",
            "regions.hot.json",
            "topology.fusion.json",
        ] {
            assert!(
                assets.join(asset).exists(),
                "missing generated asset {asset}"
            );
        }

        // Generated topology invariants.
        let topology = read_json(assets.join("topology.generated.json")).unwrap();
        assert!(topology.get("quantale").is_some());
        assert!(topology.get("source_version").is_some());
        let gen_transitions = topology
            .get("transitions")
            .and_then(Value::as_array)
            .unwrap();
        assert!(
            gen_transitions
                .iter()
                .all(|t| t.get("default_weight").is_none()),
            "generated topology must not carry legacy default_weight"
        );

        // Split-topology invariants.
        let ctrl = read_json(assets.join("topology.control.json")).unwrap();
        let hot = read_json(assets.join("topology.hot.json")).unwrap();
        let regions = read_json(assets.join("regions.hot.json")).unwrap();
        let fusion = read_json(assets.join("topology.fusion.json")).unwrap();

        let ctrl_names: std::collections::HashSet<&str> = ctrl
            .get("nodes")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|n| n.get("name").and_then(Value::as_str))
            .collect();
        let hot_names: std::collections::HashSet<&str> = hot
            .get("nodes")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|n| n.get("name").and_then(Value::as_str))
            .filter(|&n| n != "Region::CommitReceipt")
            .collect();

        // Invariant: control ∩ hot = ∅
        let overlap: Vec<&&str> = ctrl_names.intersection(&hot_names).collect();
        assert!(
            overlap.is_empty(),
            "control ∩ hot must be empty; found {overlap:?}"
        );

        // Invariant: hot topology has at least one transition.
        let hot_transitions = hot.get("transitions").and_then(Value::as_array).unwrap();
        assert!(
            !hot_transitions.is_empty(),
            "hot topology must have at least one transition"
        );

        // Invariant: every hot node (except CommitReceipt) has a region entry.
        let region_names: std::collections::HashSet<&str> = regions
            .get("regions")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|r| r.get("name").and_then(Value::as_str))
            .filter(|&n| n != "Region::CommitReceipt")
            .collect();
        for name in &hot_names {
            assert!(
                region_names.contains(name),
                "hot node '{name}' missing from regions.hot.json"
            );
        }

        let fusion_regions = fusion
            .get("regions")
            .and_then(Value::as_array)
            .expect("fusion regions");
        assert!(
            fusion_regions.iter().any(|region| {
                region
                    .get("nodes")
                    .and_then(Value::as_array)
                    .is_some_and(|nodes| {
                        nodes
                            == &[
                                Value::String("Analysis::Return1".to_string()),
                                Value::String("Analysis::Volatility".to_string()),
                                Value::String("Analysis::SignalScore".to_string()),
                            ]
                    })
            }),
            "generated fusion asset must retain the analysis linear chain"
        );

        // Invariant: all hot transitions reference known nodes in hot topology.
        for t in hot_transitions {
            let from = t.get("from").and_then(Value::as_str).unwrap();
            let to = t.get("to").and_then(Value::as_str).unwrap();
            let all_hot: std::collections::HashSet<&str> = hot
                .get("nodes")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .filter_map(|n| n.get("name").and_then(Value::as_str))
                .collect();
            assert!(
                all_hot.contains(from),
                "transition 'from' '{from}' not in hot nodes"
            );
            assert!(
                all_hot.contains(to),
                "transition 'to' '{to}' not in hot nodes"
            );
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn build_overlay_is_deterministic() {
        // Running build_overlay_assets twice on the same source must produce
        // byte-identical output for every generated artifact.
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let workspace_assets =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");

        let make_root = |suffix: &str| {
            let r = std::env::temp_dir().join(format!(
                "quantale_det_test_{}_{}_{suffix}",
                std::process::id(),
                ts
            ));
            let a = r.join("assets");
            fs::create_dir_all(&a).unwrap();
            fs::copy(
                workspace_assets.join("topology.source.json"),
                a.join("topology.source.json"),
            )
            .unwrap();
            fs::copy(
                workspace_assets.join("operators.json"),
                a.join("operators.json"),
            )
            .unwrap();
            r
        };

        let root1 = make_root("a");
        let root2 = make_root("b");
        build_overlay_assets(&root1).expect("first build_overlay run");
        build_overlay_assets(&root2).expect("second build_overlay run");

        let artifacts = [
            "topology.generated.json",
            "operators.generated.json",
            "topology.control.json",
            "topology.hot.json",
            "regions.hot.json",
            "topology.fusion.json",
        ];

        for artifact in artifacts {
            let content1 =
                fs::read_to_string(root1.join("assets").join(artifact)).unwrap_or_default();
            let content2 =
                fs::read_to_string(root2.join("assets").join(artifact)).unwrap_or_default();
            assert_eq!(
                content1, content2,
                "build_overlay_assets must be deterministic for '{artifact}'"
            );
        }

        fs::remove_dir_all(root1).unwrap();
        fs::remove_dir_all(root2).unwrap();
    }
}
