//! Runtime decision invariant checker (Phase 6 — invariants 18–20, 24).
//!
//! Call `decision_is_safe()` before every `execute_abstract_node` to enforce
//! invariant 20 (no frontier advance on a bottom score with blocked=0).
//! Call `check_decision()` to collect diagnostic violations for logging.
//!
//! Invariants enforced
//! -------------------
//!  18. score=⊥  ⟹  blocked=1  ∧  first_hop < 0   (ScoreBottomNotBlocked)
//!  19. s≠⊥, blocked=0  ⟹  first_hop = selected_dst  (ProjectionFirstHopMismatch)
//!  20. s=⊥ with blocked=0  →  skip executor call    (enforced by decision_is_safe)
//!  24. Declared block-node pattern executing ⟹ blocked=1 ∨ halted=1 (BlockNodeNotBlocked)

use crate::graph::DecisionReport;
use crate::tensor::TENSOR_NODE_COUNT;
use crate::types::BOTTOM;
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub const DEFAULT_RUNTIME_INVARIANTS_JSON: &str =
    include_str!("../assets/runtime_invariants.json");

/// A runtime tensor invariant violation.
#[derive(Clone, Debug)]
pub struct RuntimeViolation {
    pub kind: RuntimeViolationKind,
    pub detail: String,
}

impl std::fmt::Display for RuntimeViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.kind, self.detail)
    }
}

/// Describes which runtime invariant was violated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeViolationKind {
    /// Invariant 18: score=⊥ but blocked=0 or first_hop ≥ 0.
    ScoreBottomNotBlocked,
    /// Invariant 19: projection dst and first_hop disagree when score > ⊥.
    ProjectionFirstHopMismatch,
    /// Invariant 24: Control::Block node produced blocked=0 and halted=0.
    BlockNodeNotBlocked,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct RuntimeInvariantPolicy {
    pub block_node_patterns: Vec<String>,
}

impl RuntimeInvariantPolicy {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        let policy: Self = serde_json::from_str(input)
            .map_err(|error| format!("parse runtime invariants: {error}"))?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let input = fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "read runtime invariants '{}': {error}",
                path.as_ref().display()
            )
        })?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, String> {
        Self::from_json_file("assets/runtime_invariants.json")
            .or_else(|_| Self::from_json_str(DEFAULT_RUNTIME_INVARIANTS_JSON))
    }

    pub fn is_block_node(&self, node_name: &str) -> bool {
        self.block_node_patterns
            .iter()
            .any(|pattern| node_name.contains(pattern))
    }

    fn validate(&self) -> Result<(), String> {
        if self.block_node_patterns.is_empty() {
            return Err(
                "runtime invariants block_node_patterns must not be empty".to_string(),
            );
        }
        if self
            .block_node_patterns
            .iter()
            .any(|pattern| pattern.trim().is_empty())
        {
            return Err(
                "runtime invariants block_node_patterns must not contain empty patterns"
                    .to_string(),
            );
        }
        Ok(())
    }
}

impl std::fmt::Display for RuntimeViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScoreBottomNotBlocked => write!(f, "score_bottom_not_blocked"),
            Self::ProjectionFirstHopMismatch => write!(f, "projection_first_hop_mismatch"),
            Self::BlockNodeNotBlocked => write!(f, "block_node_not_blocked"),
        }
    }
}

/// Returns `false` when executing `report` would violate invariant 20
/// (score=⊥ with blocked=0, or first_hop outside valid range).
///
/// Always call this before `execute_abstract_node`.  When it returns false,
/// skip the executor call and increment `consecutive_blocks`.
pub fn decision_is_safe(report: &DecisionReport) -> bool {
    if report.blocked != 0 || report.halted != 0 {
        return true;
    }
    if report.selected_value <= BOTTOM {
        return false;
    }
    (0..TENSOR_NODE_COUNT as i32).contains(&report.first_hop)
}

/// Validate a `DecisionReport` against invariants 18, 19, and 24.
///
/// `executing_node_name` is the name of the node about to be (or just)
/// executed — required for the invariant 24 block-node check.
///
/// Returns all violations found; caller decides whether to abort or log.
pub fn check_decision(report: &DecisionReport, executing_node_name: &str) -> Vec<RuntimeViolation> {
    let fallback_policy = RuntimeInvariantPolicy::default_asset().unwrap_or_else(|_| {
        RuntimeInvariantPolicy {
            block_node_patterns: Vec::new(),
        }
    });
    check_decision_with_policy(report, executing_node_name, &fallback_policy)
}

pub fn check_decision_with_policy(
    report: &DecisionReport,
    executing_node_name: &str,
    policy: &RuntimeInvariantPolicy,
) -> Vec<RuntimeViolation> {
    let mut violations = Vec::new();

    // Invariant 18: s_t = ⊥  ⟹  blocked = 1  ∧  first_hop < 0
    if report.selected_value <= BOTTOM {
        if !(report.blocked != 0 && report.first_hop < 0) {
            violations.push(RuntimeViolation {
                kind: RuntimeViolationKind::ScoreBottomNotBlocked,
                detail: format!(
                    "score=⊥ but blocked={} and first_hop={}; \
                     kernel must not advance frontier on a bottom score",
                    report.blocked, report.first_hop
                ),
            });
        }
    }

    // Invariant 19: P_t = (u→v)  ∧  s_t ≠ ⊥  ⟹  h_t = v
    if report.selected_value > BOTTOM
        && report.blocked == 0
        && report.first_hop != report.selected_dst
    {
        violations.push(RuntimeViolation {
            kind: RuntimeViolationKind::ProjectionFirstHopMismatch,
            detail: format!(
                "projection dst={} but first_hop={}; \
                 decision report may be stale or two kernels raced",
                report.selected_dst, report.first_hop
            ),
        });
    }

    // Invariant 24: declared block node executing ⟹ blocked = 1 ∨ halted = 1
    if policy.is_block_node(executing_node_name) && report.blocked == 0 && report.halted == 0 {
        violations.push(RuntimeViolation {
            kind: RuntimeViolationKind::BlockNodeNotBlocked,
            detail: format!(
                "'{}' produced blocked=0 halted=0; \
                 a block node must set blocked or transition to halt",
                executing_node_name
            ),
        });
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(
        selected_value: f32,
        blocked: i32,
        halted: i32,
        first_hop: i32,
        selected_dst: i32,
    ) -> DecisionReport {
        DecisionReport {
            step: 0,
            selected_src: 0,
            selected_dst,
            first_hop,
            selected_value,
            halted,
            blocked,
        }
    }

    #[test]
    fn decision_is_safe_rejects_bottom_score_with_blocked_zero() {
        let r = report(BOTTOM, 0, 0, 3, 3);
        assert!(!decision_is_safe(&r));
    }

    #[test]
    fn decision_is_safe_accepts_bottom_score_when_blocked() {
        let r = report(BOTTOM, 1, 0, -1, -1);
        assert!(decision_is_safe(&r));
    }

    #[test]
    fn decision_is_safe_rejects_out_of_range_first_hop() {
        let r = report(0.9, 0, 0, -1, -1);
        assert!(!decision_is_safe(&r));
    }

    #[test]
    fn decision_is_safe_accepts_valid_report() {
        let r = report(0.9, 0, 0, 5, 5);
        assert!(decision_is_safe(&r));
    }

    #[test]
    fn check_decision_inv18_fires_when_bottom_and_not_blocked() {
        let r = report(BOTTOM, 0, 0, 3, 3);
        let vs = check_decision(&r, "State::Validate");
        assert!(
            vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::ScoreBottomNotBlocked)
        );
    }

    #[test]
    fn check_decision_inv18_fires_when_bottom_blocked_but_valid_first_hop() {
        // blocked=1 but first_hop >= 0 — both conditions must hold
        let r = report(BOTTOM, 1, 0, 3, 3);
        let vs = check_decision(&r, "State::Validate");
        assert!(
            vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::ScoreBottomNotBlocked)
        );
    }

    #[test]
    fn check_decision_inv18_passes_when_bottom_blocked_and_negative_hop() {
        let r = report(BOTTOM, 1, 0, -1, -1);
        let vs = check_decision(&r, "State::Validate");
        assert!(
            !vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::ScoreBottomNotBlocked)
        );
    }

    #[test]
    fn check_decision_inv19_fires_on_hop_dst_mismatch() {
        let r = report(0.8, 0, 0, 2, 5);
        let vs = check_decision(&r, "State::Validate");
        assert!(
            vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::ProjectionFirstHopMismatch)
        );
    }

    #[test]
    fn check_decision_inv19_passes_when_hop_matches_dst() {
        let r = report(0.8, 0, 0, 5, 5);
        let vs = check_decision(&r, "State::Validate");
        assert!(
            !vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::ProjectionFirstHopMismatch)
        );
    }

    #[test]
    fn check_decision_inv24_fires_for_block_node_not_blocked() {
        let r = report(BOTTOM, 0, 0, 3, 3);
        let vs = check_decision(&r, "Control::Block");
        assert!(
            vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::BlockNodeNotBlocked)
        );
    }

    #[test]
    fn check_decision_inv24_passes_when_block_node_sets_blocked() {
        let r = report(BOTTOM, 1, 0, -1, -1);
        let vs = check_decision(&r, "Control::Block");
        assert!(
            !vs.iter()
                .any(|v| v.kind == RuntimeViolationKind::BlockNodeNotBlocked)
        );
    }
}
