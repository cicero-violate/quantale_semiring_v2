//! Execution policy compiled into matrix-edge deltas.
//!
//! Policy routing is owned by `assets/policy.json`; this module evaluates the
//! current boolean policy state against that data contract and compiles matching
//! node-name edges into matrix deltas.

use serde::Deserialize;

use crate::edge::TransitionEdge;
use crate::topology::GraphTopology;

const DEFAULT_POLICY_JSON: &str = include_str!("../assets/policy.json");

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
}

/// Compile execution-side policy conditions into ordinary matrix edges.
///
/// The returned edges are intended to be joined into the CUDA-resident
/// transition matrix with the same max-times semantics as all other movement:
/// `M := M ∨ M_policy`.
///
/// Policy is represented as ordinary matrix structure; projection reads the
/// closed matrix rather than a side-channel projection mask.
pub fn build_policy_edges(policy: ExecutionGatePolicy) -> Vec<TransitionEdge> {
    build_policy_edges_from_json(policy, DEFAULT_POLICY_JSON)
        .expect("bundled assets/policy.json must compile")
}

fn build_policy_edges_from_json(
    policy: ExecutionGatePolicy,
    input: &str,
) -> Result<Vec<TransitionEdge>, String> {
    let rules: PolicyRulesFile =
        serde_json::from_str(input).map_err(|error| format!("parse policy rules: {error}"))?;
    let topology = GraphTopology::default_asset()
        .map_err(|error| format!("load topology registry: {error}"))?
        .compile()
        .map_err(|error| format!("compile topology registry: {error}"))?;

    let mut edges = Vec::new();
    for rule in rules.rules {
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
            edges.push(TransitionEdge::new(src as i32, dst as i32, edge.weight));
        }
    }
    Ok(edges)
}

impl PolicyCondition {
    fn matches(&self, policy: &ExecutionGatePolicy) -> bool {
        self.all
            .iter()
            .all(|predicate| policy.evaluate(predicate.as_str()))
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
