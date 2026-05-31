//! Policy and receipt evidence compiled into lattice-edge deltas.
//!
//! Rule routing is owned by `assets/rule_delta.json`; this module evaluates the
//! current policy/receipt state against that data contract and compiles matching
//! node-name edges into matrix deltas.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::edge::LatticeEdge;
use crate::tensor::TensorEdge;
use crate::topology::GraphTopology;
use crate::types::QuantaleWeight;

const DEFAULT_RULE_DELTA_JSON: &str = include_str!("../assets/rule_delta.json");

pub struct ExecutionGatePolicy {
    pub evidence_accepted: bool,
    pub receipt_hash_nonzero: bool,
    pub task_blocked: bool,
    pub retry_allowed: bool,
    pub halt_allowed: bool,
    pub repair_allowed: bool,
    pub commit_allowed: bool,
}

impl Default for ExecutionGatePolicy {
    fn default() -> Self {
        Self {
            evidence_accepted: false,
            receipt_hash_nonzero: false,
            task_blocked: false,
            retry_allowed: true,
            halt_allowed: true,
            repair_allowed: true,
            commit_allowed: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProcessReceipt {
    pub node_name: String,
    pub exit_code: i32,
    pub stdout_payload: String,
    pub stderr_payload: String,
}

impl ProcessReceipt {
    /// Maps generic Unix exit codes directly to pure Quantale Weights.
    pub fn calculate_algebraic_feedback(&self) -> QuantaleWeight {
        match self.exit_code {
            0 => QuantaleWeight::one(),
            1 | 127 => QuantaleWeight::zero(),
            _ => QuantaleWeight(0.1f32),
        }
    }

    /// Collapses a generic process receipt into the existing matrix receipt edge model.
    pub fn to_execution_receipt(&self) -> ExecutionReceipt {
        let feedback = self.calculate_algebraic_feedback().raw();
        if self.exit_code == 0 {
            ExecutionReceipt::accepted(feedback, feedback, feedback)
        } else {
            ExecutionReceipt::rejected(feedback, feedback, feedback)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub accepted: bool,
    pub metrics: HashMap<String, f32>,
}

impl ExecutionReceipt {
    pub fn accepted(receipt_confidence: f32, hash_score: f32, validation_score: f32) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert("receipt_confidence".to_string(), receipt_confidence);
        metrics.insert("hash_score".to_string(), hash_score);
        metrics.insert("validation_score".to_string(), validation_score);
        metrics.insert("hash_nonzero".to_string(), 1.0);
        Self {
            accepted: true,
            metrics,
        }
    }

    pub fn accepted_without_hash(receipt_confidence: f32, rejection_score: f32) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert("receipt_confidence".to_string(), receipt_confidence);
        metrics.insert("rejection_score".to_string(), rejection_score);
        metrics.insert("rollback_score".to_string(), rejection_score);
        metrics.insert("repair_score".to_string(), rejection_score);
        metrics.insert("hash_nonzero".to_string(), 0.0);
        Self {
            accepted: true,
            metrics,
        }
    }

    pub fn rejected(rejection_score: f32, rollback_score: f32, repair_score: f32) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert("rejection_score".to_string(), rejection_score);
        metrics.insert("rollback_score".to_string(), rollback_score);
        metrics.insert("repair_score".to_string(), repair_score);
        metrics.insert("hash_nonzero".to_string(), 0.0);
        Self {
            accepted: false,
            metrics,
        }
    }

    pub fn hash_nonzero(&self) -> bool {
        self.metrics
            .get("hash_nonzero")
            .is_some_and(|value| *value > 0.5)
    }

    fn evaluate(&self, predicate: &str) -> bool {
        match predicate {
            "always" => true,
            "accepted" => self.accepted,
            "not_accepted" => !self.accepted,
            "hash_nonzero" => self.hash_nonzero(),
            "not_hash_nonzero" => !self.hash_nonzero(),
            _ => false,
        }
    }

    fn weight_field(&self, field: &str) -> Option<f32> {
        self.metrics.get(field).copied()
    }
}

#[derive(Debug, Deserialize)]
struct RuleDeltaAsset {
    policy: PolicyRulesFile,
    receipt: ReceiptRulesFile,
}

#[derive(Debug, Deserialize)]
struct PolicyRulesFile {
    rules: Vec<PolicyRule>,
}

#[derive(Debug, Deserialize)]
struct PolicyRule {
    when: PolicyCondition,
    edges: Vec<PolicyEdge>,
}

#[derive(Debug, Deserialize)]
struct PolicyCondition {
    #[serde(default)]
    all: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PolicyEdge {
    from: String,
    to: String,
    weight: f32,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    cost: Option<f32>,
    #[serde(default)]
    safety: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct ReceiptRulesFile {
    rules: Vec<ReceiptRule>,
}

#[derive(Debug, Deserialize)]
struct ReceiptRule {
    when: ReceiptCondition,
    edges: Vec<ReceiptEdge>,
}

#[derive(Debug, Deserialize)]
struct ReceiptCondition {
    #[serde(default)]
    all: Vec<String>,
    #[serde(default)]
    any: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ReceiptEdge {
    from: String,
    to: String,
    weight: ReceiptWeight,
    #[serde(default)]
    confidence: Option<ReceiptWeight>,
    #[serde(default)]
    cost: Option<ReceiptWeight>,
    #[serde(default)]
    safety: Option<ReceiptWeight>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReceiptWeight {
    Literal(f32),
    Field(String),
}

/// Compile execution-side policy conditions into ordinary matrix edges.
///
/// The returned edges are intended to be joined into the CUDA-resident
/// transition matrix with the same max-times semantics as all other movement:
/// `M := M ∨ M_policy`.
///
/// Policy is represented as ordinary matrix structure; projection reads the
/// closed matrix rather than a side-channel projection mask.
pub fn build_policy_edges(policy: ExecutionGatePolicy) -> Vec<LatticeEdge> {
    build_policy_edges_from_json(policy, DEFAULT_RULE_DELTA_JSON)
        .expect("bundled assets/rule_delta.json policy rules must compile")
}

/// Compile a runtime execution receipt into ordinary matrix edges.
///
/// These edges are joined into the same CUDA-resident transition matrix as the
/// static graph and policy graph. This lets concrete receipt evidence alter the
/// reachable path weights without introducing a separate CPU planner.
pub fn build_receipt_edges(receipt: ExecutionReceipt) -> Vec<LatticeEdge> {
    build_receipt_edges_from_json(receipt, DEFAULT_RULE_DELTA_JSON)
        .expect("bundled assets/rule_delta.json receipt rules must compile")
}

/// Compile execution-side policy conditions into tensor matrix edges.
///
/// If a rule edge omits tensor-specific fields, the scalar weight is converted
/// as confidence=weight, cost=1-weight, safety=weight.
pub fn build_tensor_policy_edges(policy: ExecutionGatePolicy) -> Vec<TensorEdge> {
    build_tensor_policy_edges_from_json(policy, DEFAULT_RULE_DELTA_JSON)
        .expect("bundled assets/rule_delta.json tensor policy rules must compile")
}

/// Compile a runtime execution receipt into tensor matrix edges.
///
/// If a rule edge omits tensor-specific fields, the resolved scalar weight is
/// converted as confidence=weight, cost=1-weight, safety=weight.
pub fn build_tensor_receipt_edges(receipt: ExecutionReceipt) -> Vec<TensorEdge> {
    build_tensor_receipt_edges_from_json(receipt, DEFAULT_RULE_DELTA_JSON)
        .expect("bundled assets/rule_delta.json tensor receipt rules must compile")
}

fn build_policy_edges_from_json(
    policy: ExecutionGatePolicy,
    input: &str,
) -> Result<Vec<LatticeEdge>, String> {
    let asset: RuleDeltaAsset =
        serde_json::from_str(input).map_err(|error| format!("parse rule delta asset: {error}"))?;
    let topology = compiled_topology()?;

    let mut edges = Vec::new();
    for rule in asset.policy.rules {
        if !rule.when.matches(&policy) {
            continue;
        }
        for edge in rule.edges {
            let src = topology
                .registry
                .id_of(&edge.from)
                .ok_or_else(|| format!("policy edge source '{}' is not declared", edge.from))?;
            let dst = topology
                .registry
                .id_of(&edge.to)
                .ok_or_else(|| format!("policy edge destination '{}' is not declared", edge.to))?;
            edges.push(LatticeEdge::new(src as i32, dst as i32, edge.weight));
        }
    }
    Ok(edges)
}

fn build_receipt_edges_from_json(
    receipt: ExecutionReceipt,
    input: &str,
) -> Result<Vec<LatticeEdge>, String> {
    let asset: RuleDeltaAsset =
        serde_json::from_str(input).map_err(|error| format!("parse rule delta asset: {error}"))?;
    let topology = compiled_topology()?;

    let mut edges = Vec::new();
    for rule in asset.receipt.rules {
        if !rule.when.matches(&receipt) {
            continue;
        }
        for edge in rule.edges {
            let src = topology
                .registry
                .id_of(&edge.from)
                .ok_or_else(|| format!("receipt edge source '{}' is not declared", edge.from))?;
            let dst = topology
                .registry
                .id_of(&edge.to)
                .ok_or_else(|| format!("receipt edge destination '{}' is not declared", edge.to))?;
            edges.push(LatticeEdge::new(
                src as i32,
                dst as i32,
                edge.weight.resolve(&receipt)?,
            ));
        }
    }
    Ok(edges)
}

fn build_tensor_policy_edges_from_json(
    policy: ExecutionGatePolicy,
    input: &str,
) -> Result<Vec<TensorEdge>, String> {
    let asset: RuleDeltaAsset =
        serde_json::from_str(input).map_err(|error| format!("parse rule delta asset: {error}"))?;
    let topology = compiled_topology()?;

    let mut edges = Vec::new();
    for rule in asset.policy.rules {
        if !rule.when.matches(&policy) {
            continue;
        }
        for edge in rule.edges {
            let src = topology
                .registry
                .id_of(&edge.from)
                .ok_or_else(|| format!("policy edge source '{}' is not declared", edge.from))?;
            let dst = topology
                .registry
                .id_of(&edge.to)
                .ok_or_else(|| format!("policy edge destination '{}' is not declared", edge.to))?;
            let weight = edge.weight.clamp(0.0, 1.0);
            edges.push(TensorEdge::new(
                src as i32,
                dst as i32,
                edge.confidence.unwrap_or(weight).clamp(0.0, 1.0),
                edge.cost.unwrap_or(1.0 - weight).max(0.0),
                edge.safety.unwrap_or(weight).clamp(0.0, 1.0),
            ));
        }
    }
    Ok(edges)
}

fn build_tensor_receipt_edges_from_json(
    receipt: ExecutionReceipt,
    input: &str,
) -> Result<Vec<TensorEdge>, String> {
    let asset: RuleDeltaAsset =
        serde_json::from_str(input).map_err(|error| format!("parse rule delta asset: {error}"))?;
    let topology = compiled_topology()?;

    let mut edges = Vec::new();
    for rule in asset.receipt.rules {
        if !rule.when.matches(&receipt) {
            continue;
        }
        for edge in rule.edges {
            let src = topology
                .registry
                .id_of(&edge.from)
                .ok_or_else(|| format!("receipt edge source '{}' is not declared", edge.from))?;
            let dst = topology
                .registry
                .id_of(&edge.to)
                .ok_or_else(|| format!("receipt edge destination '{}' is not declared", edge.to))?;
            let weight = edge.weight.resolve(&receipt)?.clamp(0.0, 1.0);
            let confidence = edge
                .confidence
                .as_ref()
                .map(|value| value.resolve(&receipt))
                .transpose()?
                .unwrap_or(weight)
                .clamp(0.0, 1.0);
            let cost = edge
                .cost
                .as_ref()
                .map(|value| value.resolve(&receipt))
                .transpose()?
                .unwrap_or(1.0 - weight)
                .max(0.0);
            let safety = edge
                .safety
                .as_ref()
                .map(|value| value.resolve(&receipt))
                .transpose()?
                .unwrap_or(weight)
                .clamp(0.0, 1.0);
            edges.push(TensorEdge::new(
                src as i32, dst as i32, confidence, cost, safety,
            ));
        }
    }
    Ok(edges)
}

fn compiled_topology() -> Result<crate::topology::CompiledTopology, String> {
    GraphTopology::default_asset()
        .map_err(|error| format!("load topology registry: {error}"))?
        .compile()
        .map_err(|error| format!("compile topology registry: {error}"))
}

impl PolicyCondition {
    fn matches(&self, policy: &ExecutionGatePolicy) -> bool {
        self.all
            .iter()
            .all(|predicate| policy.evaluate(predicate.as_str()))
    }
}

impl ReceiptCondition {
    fn matches(&self, receipt: &ExecutionReceipt) -> bool {
        self.all
            .iter()
            .all(|predicate| receipt.evaluate(predicate.as_str()))
            && (self.any.is_empty()
                || self
                    .any
                    .iter()
                    .any(|predicate| receipt.evaluate(predicate.as_str())))
    }
}

impl ReceiptWeight {
    fn resolve(&self, receipt: &ExecutionReceipt) -> Result<f32, String> {
        match self {
            Self::Literal(value) => Ok(*value),
            Self::Field(field) => receipt
                .weight_field(field)
                .ok_or_else(|| format!("unknown receipt weight field '{field}'")),
        }
    }
}

impl ExecutionGatePolicy {
    fn evaluate(&self, predicate: &str) -> bool {
        match predicate {
            "evidence_accepted" => self.evidence_accepted,
            "not_evidence_accepted" => !self.evidence_accepted,
            "receipt_hash_nonzero" => self.receipt_hash_nonzero,
            "not_receipt_hash_nonzero" => !self.receipt_hash_nonzero,
            "task_blocked" => self.task_blocked,
            "not_task_blocked" => !self.task_blocked,
            "retry_allowed" => self.retry_allowed,
            "not_retry_allowed" => !self.retry_allowed,
            "halt_allowed" => self.halt_allowed,
            "not_halt_allowed" => !self.halt_allowed,
            "repair_allowed" => self.repair_allowed,
            "not_repair_allowed" => !self.repair_allowed,
            "commit_allowed" => self.commit_allowed,
            "not_commit_allowed" => !self.commit_allowed,
            "commit_ready" => {
                self.commit_allowed && self.evidence_accepted && self.receipt_hash_nonzero
            }
            "not_commit_ready" => {
                !(self.commit_allowed && self.evidence_accepted && self.receipt_hash_nonzero)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn tensor_policy_edges_default_from_scalar_weight() {
        let policy = ExecutionGatePolicy {
            task_blocked: true,
            ..ExecutionGatePolicy::default()
        };
        let edges = build_tensor_policy_edges(policy);
        assert!(edges.iter().any(|edge| edge.confidence > 0.9));
        assert!(edges.iter().all(|edge| edge.cost >= 0.0));
        assert!(
            edges
                .iter()
                .all(|edge| edge.safety >= 0.0 && edge.safety <= 1.0)
        );
    }

    #[test]
    fn tensor_receipt_edges_resolve_fields() {
        let edges = build_tensor_receipt_edges(ExecutionReceipt::accepted(0.8, 0.7, 0.6));
        assert!(
            edges
                .iter()
                .any(|edge| (edge.confidence - 0.8).abs() < f32::EPSILON)
        );
        assert!(
            edges
                .iter()
                .any(|edge| (edge.confidence - 0.7).abs() < f32::EPSILON)
        );
        assert!(
            edges
                .iter()
                .any(|edge| (edge.confidence - 0.6).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn tensor_receipt_edges_accept_explicit_tensor_fields() {
        let raw = r#"{
            "policy": {"rules": []},
            "receipt": {
                "rules": [
                    {
                        "when": {"all": ["accepted"]},
                        "edges": [
                            {
                                "from": "Control::GateReceipt",
                                "to": "Event::ReceiptAccepted",
                                "weight": "receipt_confidence",
                                "confidence": "receipt_confidence",
                                "cost": 3.5,
                                "safety": "validation_score"
                            }
                        ]
                    }
                ]
            }
        }"#;
        let edges =
            build_tensor_receipt_edges_from_json(ExecutionReceipt::accepted(0.81, 0.82, 0.83), raw)
                .unwrap();
        assert_eq!(edges.len(), 1);
        assert!((edges[0].confidence - 0.81).abs() < f32::EPSILON);
        assert!((edges[0].cost - 3.5).abs() < f32::EPSILON);
        assert!((edges[0].safety - 0.83).abs() < f32::EPSILON);
    }
}
