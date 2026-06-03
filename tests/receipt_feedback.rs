use quantale_semiring_v2::{LearningPolicy, TopologyRuntime, load_learned_tensor_edges};
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const SRC_NODE: &str = "Control::Block";
const DST_NODE: &str = "Control::Repair";

fn unique_temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "quantale_receipt_feedback_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn first_jsonl_value(path: &PathBuf) -> Result<Value, String> {
    let body = fs::read_to_string(path).map_err(|error| format!("read learned jsonl: {error}"))?;
    let line = body
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "learned edge file was empty".to_string())?;
    serde_json::from_str(line).map_err(|error| format!("parse learned jsonl line: {error}"))
}

#[test]
fn successful_receipt_increases_static_edge_as_tensor_feedback() -> Result<(), String> {
    let runtime = TopologyRuntime::load_checked_default().map_err(|error| error.to_string())?;
    let src_id = runtime
        .registry()
        .id_of(SRC_NODE)
        .ok_or_else(|| format!("missing node {SRC_NODE}"))?;
    let dst_id = runtime
        .registry()
        .id_of(DST_NODE)
        .ok_or_else(|| format!("missing node {DST_NODE}"))?;
    let static_edge = runtime
        .tensor_edges()
        .iter()
        .find(|edge| edge.src == src_id as i32 && edge.dst == dst_id as i32)
        .ok_or_else(|| format!("missing static edge {SRC_NODE} -> {DST_NODE}"))?;

    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).map_err(|error| format!("create temp dir: {error}"))?;
    let tlog_path = temp_dir.join("quantale.tlog");
    let learned_path = temp_dir.join("learned_edges.jsonl");
    let memory_path = temp_dir.join("memory.jsonl");

    let decision = json!({
        "kind": "Decision",
        "payload": {
            "selected_src": src_id,
            "first_hop": dst_id,
            "selected_value": static_edge.confidence
        }
    });
    let receipt = json!({
        "kind": "AgentStep",
        "payload": {
            "selected_src": src_id,
            "first_hop": dst_id,
            "selected_value": static_edge.confidence,
            "exit_code": 0
        }
    });
    fs::write(&tlog_path, format!("{decision}\n{receipt}\n"))
        .map_err(|error| format!("write tlog: {error}"))?;

    let mut child = Command::new("python3")
        .arg("crates/operators_lib/learn.py")
        .env("QUANTALE_LEARN_LOG", &tlog_path)
        .env("QUANTALE_LEARNED_EDGES", &learned_path)
        .env("QUANTALE_MEMORY_FILE", &memory_path)
        .env("QUANTALE_TOPOLOGY", "assets/topology.generated.json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("spawn learn.py: {error}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| "learn.py stdin unavailable".to_string())?
        .write_all(b"{}")
        .map_err(|error| format!("write learn.py stdin: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("wait learn.py: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "learn.py failed: status={:?} stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse learn.py stdout: {error}"))?;
    assert_eq!(stdout.as_array().map(Vec::len), Some(1));

    let learned_record = first_jsonl_value(&learned_path)?;
    let learned_edge = learned_record
        .get("edge")
        .ok_or_else(|| format!("missing edge in learned record: {learned_record}"))?;
    assert_eq!(
        learned_edge.get("from").and_then(Value::as_str),
        Some(SRC_NODE)
    );
    assert_eq!(
        learned_edge.get("to").and_then(Value::as_str),
        Some(DST_NODE)
    );
    let learned_confidence = learned_edge
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing learned confidence: {learned_edge}"))?
        as f32;
    assert!(
        learned_confidence > static_edge.confidence,
        "expected receipt to increase confidence above static {}, got {}",
        static_edge.confidence,
        learned_confidence
    );

    let embedded = load_learned_tensor_edges(
        &learned_path,
        runtime.registry(),
        runtime.tensor_edges(),
        &LearningPolicy::default_asset(),
    )?;
    let tensor_feedback = embedded
        .iter()
        .find(|edge| edge.src == src_id as i32 && edge.dst == dst_id as i32)
        .ok_or_else(|| "learned edge did not embed into tensor edge set".to_string())?;
    assert!(
        tensor_feedback.confidence > static_edge.confidence,
        "expected embedded tensor feedback to increase confidence above static {}, got {}",
        static_edge.confidence,
        tensor_feedback.confidence
    );

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}
