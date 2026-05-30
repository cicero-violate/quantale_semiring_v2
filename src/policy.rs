//! Execution policy compiled into matrix-edge deltas.

use crate::edge::{TransitionEdge, edge};
use crate::node::{ControlNode, EventNode, Node, StateNode};

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

const POLICY_ALLOW_WEIGHT: f32 = 0.99;
const POLICY_BLOCK_WEIGHT: f32 = 0.99;
const POLICY_REPAIR_WEIGHT: f32 = 0.97;
const POLICY_RETRY_WEIGHT: f32 = 0.90;

/// Compile execution-side policy conditions into ordinary matrix edges.
///
/// The returned edges are intended to be joined into the CUDA-resident
/// transition matrix with the same max-times semantics as all other movement:
/// `M := M ∨ M_policy`.
///
/// Policy is represented as ordinary matrix structure; projection reads the
/// closed matrix rather than a side-channel projection mask.
pub fn build_policy_edges(policy: ExecutionGatePolicy) -> Vec<TransitionEdge> {
    let mut edges = Vec::with_capacity(10);

    if policy.task_blocked {
        edges.push(edge(
            Node::control(ControlNode::GateExecution),
            Node::control(ControlNode::Block),
            POLICY_BLOCK_WEIGHT,
        ));
        edges.push(edge(
            Node::control(ControlNode::ChooseBest),
            Node::control(ControlNode::Block),
            POLICY_BLOCK_WEIGHT,
        ));
    } else {
        edges.push(edge(
            Node::control(ControlNode::GateExecution),
            Node::event(EventNode::ExecuteStarted),
            POLICY_ALLOW_WEIGHT,
        ));
    }

    if policy.evidence_accepted {
        edges.push(edge(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptAccepted),
            POLICY_ALLOW_WEIGHT,
        ));
    } else {
        edges.push(edge(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected),
            POLICY_BLOCK_WEIGHT,
        ));
        edges.push(edge(
            Node::event(EventNode::ReceiptRejected),
            Node::control(ControlNode::Rollback),
            POLICY_REPAIR_WEIGHT,
        ));
    }

    if policy.receipt_hash_nonzero {
        edges.push(edge(
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::HashNonzero),
            POLICY_ALLOW_WEIGHT,
        ));
        edges.push(edge(
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate),
            POLICY_ALLOW_WEIGHT,
        ));
    } else if policy.evidence_accepted {
        edges.push(edge(
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::ReceiptRejected),
            POLICY_BLOCK_WEIGHT,
        ));
        edges.push(edge(
            Node::event(EventNode::ReceiptRejected),
            Node::control(ControlNode::Rollback),
            POLICY_REPAIR_WEIGHT,
        ));
    }

    if policy.commit_allowed && policy.evidence_accepted && policy.receipt_hash_nonzero {
        edges.push(edge(
            Node::state(StateNode::Validate),
            Node::control(ControlNode::Commit),
            POLICY_ALLOW_WEIGHT,
        ));
        edges.push(edge(
            Node::control(ControlNode::Commit),
            Node::control(ControlNode::GateMemory),
            POLICY_ALLOW_WEIGHT,
        ));
    } else {
        edges.push(edge(
            Node::state(StateNode::Validate),
            Node::control(ControlNode::Block),
            POLICY_BLOCK_WEIGHT,
        ));
    }

    if policy.repair_allowed {
        edges.push(edge(
            Node::control(ControlNode::Rollback),
            Node::control(ControlNode::Repair),
            POLICY_REPAIR_WEIGHT,
        ));
    } else {
        edges.push(edge(
            Node::control(ControlNode::Rollback),
            Node::control(ControlNode::Block),
            POLICY_BLOCK_WEIGHT,
        ));
    }

    if policy.retry_allowed {
        edges.push(edge(
            Node::control(ControlNode::Block),
            Node::control(ControlNode::Retry),
            POLICY_RETRY_WEIGHT,
        ));
    } else if policy.halt_allowed {
        edges.push(edge(
            Node::control(ControlNode::Block),
            Node::control(ControlNode::Halt),
            POLICY_BLOCK_WEIGHT,
        ));
    }

    if policy.halt_allowed {
        edges.push(edge(
            Node::event(EventNode::LearnUpdated),
            Node::control(ControlNode::Halt),
            POLICY_ALLOW_WEIGHT,
        ));
    }

    edges
}
