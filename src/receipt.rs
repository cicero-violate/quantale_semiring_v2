//! Runtime receipt evidence compiled into matrix-edge deltas.
//!
//! Receipt routing is owned by `assets/receipt.json`; this module maps raw OS
//! process status into scalar receipt evidence and compiles matching receipt
//! rules into ordinary matrix-edge deltas.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::edge::TransitionEdge;
use crate::topology::GraphTopology;
use crate::types::QuantaleWeight;

const DEFAULT_RECEIPT_JSON: &str = include_str!("../assets/receipt.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProcessReceipt {
    pub node_name: String,
    pub exit_code: i32,
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
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReceiptWeight {
    Literal(f32),
    Field(String),
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

/// Compile a runtime execution receipt into ordinary matrix edges.
///
/// These edges are joined into the same CUDA-resident transition matrix as the
/// static graph and policy graph. This lets concrete receipt evidence alter the
/// reachable path weights without introducing a separate CPU planner.
pub fn build_receipt_edges(receipt: ExecutionReceipt) -> Vec<TransitionEdge> {
    build_receipt_edges_from_json(receipt, DEFAULT_RECEIPT_JSON)
        .expect("bundled assets/receipt.json must compile")
}

fn build_receipt_edges_from_json(
    receipt: ExecutionReceipt,
    input: &str,
) -> Result<Vec<TransitionEdge>, String> {
    let rules: ReceiptRulesFile =
        serde_json::from_str(input).map_err(|error| format!("parse receipt rules: {error}"))?;
    let topology = GraphTopology::default_asset()
        .map_err(|error| format!("load topology registry: {error}"))?
        .compile()
        .map_err(|error| format!("compile topology registry: {error}"))?;

    let mut edges = Vec::new();
    for rule in rules.rules {
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
            edges.push(TransitionEdge::new(
                src as i32,
                dst as i32,
                edge.weight.resolve(&receipt)?,
            ));
        }
    }
    Ok(edges)
}
