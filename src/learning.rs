//! Learned tensor-edge checkpoint ingress and egress.
//!
//! JSONL state is evidence/checkpoint storage only. The runtime representation
//! of learned path weights is still the GPU tensor after these sparse edges are
//! embedded.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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
    /// How much above the static base confidence a learned edge may go.
    /// Cap = min(base_confidence + max_confidence_above_base, confidence_clamp[1]).
    /// Prevents repeated successful traversals from saturating edges at 1.0,
    /// which would crowd out unvisited paths like the dev chain.
    #[serde(default = "LearningPolicy::default_max_confidence_above_base")]
    pub max_confidence_above_base: f32,
}

impl LearningPolicy {
    fn default_max_confidence_above_base() -> f32 {
        0.15
    }
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
    // Base confidence from static topology, keyed by (src, dst).
    let base_confidence: BTreeMap<(i32, i32), f32> = allowed_edges
        .iter()
        .map(|e| ((e.src, e.dst), e.confidence))
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
        let base_conf = base_confidence.get(&(src, dst)).copied().unwrap_or(0.0);
        let conf_cap =
            (base_conf + policy.max_confidence_above_base).min(policy.confidence_clamp[1]);
        deduped.insert(
            (src, dst),
            TensorEdge::new(
                src,
                dst,
                edge.confidence.clamp(policy.confidence_clamp[0], conf_cap),
                edge.cost.max(policy.learned_edge_cost_floor),
                edge.safety
                    .clamp(policy.safety_clamp[0], policy.safety_clamp[1]),
            ),
        );
    }
    Ok(deduped.into_values().collect())
}

// ── Egress ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct NamedLearningEdge {
    from: String,
    to: String,
    confidence: f32,
    cost: f32,
    safety: f32,
}

/// Buffers learned tensor-edge deltas and flushes them to the JSONL checkpoint
/// file in batches.  The file is created on first flush if absent.
///
/// Records are only written for successful (exit_code == 0) topology edges.
/// The load path (`load_learned_tensor_edges`) deduplicates by latest-wins, so
/// repeated flushes of the same edge are safe.
pub struct LearningBuffer {
    path: PathBuf,
    pending: Vec<NamedLearningEdge>,
    flush_threshold: usize,
}

impl LearningBuffer {
    pub fn new(path: impl Into<PathBuf>, flush_threshold: usize) -> Self {
        Self {
            path: path.into(),
            pending: Vec::new(),
            flush_threshold,
        }
    }

    /// Queue one learned edge.  Auto-flushes when the pending count reaches
    /// `flush_threshold`.  Flush errors are logged to stderr but do not panic.
    pub fn record(&mut self, from: &str, to: &str, confidence: f32, cost: f32, safety: f32) {
        self.pending.push(NamedLearningEdge {
            from: from.to_string(),
            to: to.to_string(),
            confidence,
            cost,
            safety,
        });
        if self.pending.len() >= self.flush_threshold {
            if let Err(error) = self.flush() {
                eprintln!("[learning] flush failed: {error}");
            }
        }
    }

    /// Write all pending records to the JSONL file and clear the buffer.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!("create learned-edges dir '{}': {e}", parent.display())
                })?;
            }
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| {
                format!("open learned-edges '{}': {e}", self.path.display())
            })?;
        let mut writer = std::io::BufWriter::new(file);
        for edge in &self.pending {
            let line = serde_json::json!({
                "from":       edge.from,
                "to":         edge.to,
                "confidence": edge.confidence,
                "cost":       edge.cost,
                "safety":     edge.safety,
            });
            let mut bytes = serde_json::to_vec(&line)
                .map_err(|e| format!("serialize learned edge: {e}"))?;
            bytes.push(b'\n');
            writer.write_all(&bytes)
                .map_err(|e| format!("write learned-edges '{}': {e}", self.path.display()))?;
        }
        writer.flush()
            .map_err(|e| format!("flush learned-edges '{}': {e}", self.path.display()))?;
        self.pending.clear();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::GraphTopology;

    fn make_temp_jsonl(content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "learned_edges_test_{}_{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, content).unwrap();
        path
    }

    fn compile_two_node(base_weight: f32) -> crate::topology::CompiledTopology {
        GraphTopology::from_json_str(&format!(
            r#"{{
                "matrix_name":"test",
                "nodes":[
                    {{"id":0,"name":"State::A","type":"State"}},
                    {{"id":1,"name":"State::B","type":"State"}}
                ],
                "transitions":[
                    {{"from":"State::A","to":"State::B","default_weight":{base_weight},"cost":0.5}}
                ],
                "pages":[]
            }}"#,
        ))
        .unwrap()
        .compile()
        .unwrap()
    }

    #[test]
    fn load_learned_edges_dedupes_latest_and_clamps_cost_floor() {
        let topology = compile_two_node(0.9);
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        let path = make_temp_jsonl(concat!(
            r#"{"from":"State::A","to":"State::B","confidence":0.2,"cost":0.7,"safety":0.4}"#,
            "\n",
            r#"{"from":"State::A","to":"State::B","confidence":0.8,"cost":0.0,"safety":0.9}"#,
            "\n"
        ));
        let policy = LearningPolicy::default();
        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &static_edges, &policy).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        // base=0.9, delta=0.15 → cap = min(1.05, 0.85) = 0.85; learned=0.8 < 0.85 → 0.8
        assert_eq!(edges[0].confidence, 0.8);
        assert_eq!(edges[0].cost, policy.learned_edge_cost_floor);
        assert_eq!(edges[0].safety, 0.9);
    }

    #[test]
    fn load_learned_edges_caps_confidence_above_base_plus_delta() {
        // base=0.6, delta=0.15 → cap = min(0.75, 0.85) = 0.75
        // learned confidence 1.0 should be clamped to 0.75
        let topology = compile_two_node(0.6);
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        let path = make_temp_jsonl(
            r#"{"from":"State::A","to":"State::B","confidence":1.0,"cost":0.1,"safety":0.9}"#,
        );
        let policy = LearningPolicy::default();
        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &static_edges, &policy).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        assert!(
            (edges[0].confidence - 0.75).abs() < 1e-5,
            "expected 0.75, got {}",
            edges[0].confidence
        );
    }

    #[test]
    fn load_learned_edges_caps_at_ceiling_when_base_is_high() {
        // base=0.95, delta=0.15 → base+delta=1.10, ceiling=0.85 → cap=0.85
        // learned confidence 1.0 should be clamped to 0.85
        let topology = compile_two_node(0.95);
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();
        let path = make_temp_jsonl(
            r#"{"from":"State::A","to":"State::B","confidence":1.0,"cost":0.05,"safety":0.95}"#,
        );
        let policy = LearningPolicy::default();
        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &static_edges, &policy).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        assert!(
            (edges[0].confidence - 0.85).abs() < 1e-5,
            "expected 0.85, got {}",
            edges[0].confidence
        );
    }

    fn temp_path(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "learning_buf_test_{}_{}_{suffix}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn learning_buffer_flush_writes_readable_jsonl() {
        let path = temp_path("flush");
        let topology = compile_two_node(0.8);
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();

        let mut buf = LearningBuffer::new(&path, 100);
        buf.record("State::A", "State::B", 0.85, 0.5, 0.9);
        assert_eq!(buf.len(), 1);
        buf.flush().unwrap();
        assert!(buf.is_empty());

        let policy = LearningPolicy::default();
        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &static_edges, &policy).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(edges.len(), 1);
        // base=0.8, delta=0.15 → cap=min(0.95,0.85)=0.85; recorded=0.85 ≤ cap → 0.85
        assert!((edges[0].confidence - 0.85).abs() < 1e-5);
        assert_eq!(edges[0].cost, 0.5);
        assert_eq!(edges[0].safety, 0.9);
    }

    #[test]
    fn learning_buffer_auto_flushes_at_threshold() {
        let path = temp_path("autof");
        let topology = compile_two_node(0.8);
        let static_edges: Vec<TensorEdge> = topology
            .transitions
            .iter()
            .copied()
            .map(TensorEdge::from)
            .collect();

        let mut buf = LearningBuffer::new(&path, 2);
        buf.record("State::A", "State::B", 0.82, 0.5, 0.9);
        assert_eq!(buf.len(), 1);
        // Second record triggers auto-flush (threshold=2).
        buf.record("State::A", "State::B", 0.84, 0.5, 0.9);
        assert!(buf.is_empty(), "auto-flush must clear pending");

        let policy = LearningPolicy::default();
        let edges =
            load_learned_tensor_edges(&path, &topology.registry, &static_edges, &policy).unwrap();
        let _ = std::fs::remove_file(&path);

        // latest-wins dedup: second record (0.84) survives
        assert_eq!(edges.len(), 1);
        assert!((edges[0].confidence - 0.84).abs() < 1e-5);
    }

    #[test]
    fn learning_buffer_empty_flush_is_noop() {
        let path = temp_path("noop");
        let mut buf = LearningBuffer::new(&path, 10);
        buf.flush().unwrap();
        assert!(!path.exists(), "empty flush must not create file");
    }
}
