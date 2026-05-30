//! Runtime receipt evidence compiled into matrix-edge deltas.

use crate::algebra::Q_BOTTOM;
use crate::edge::{TransitionEdge, edge};
use crate::node::{ControlNode, EventNode, Node, StateNode};
use crate::types::QuantaleWeight;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutionReceipt {
    pub accepted: bool,
    pub receipt_confidence: f32,
    pub hash_nonzero: bool,
    pub hash_score: f32,
    pub validation_score: f32,
    pub rejection_score: f32,
    pub rollback_score: f32,
    pub repair_score: f32,
}

impl ExecutionReceipt {
    pub const fn accepted(receipt_confidence: f32, hash_score: f32, validation_score: f32) -> Self {
        Self {
            accepted: true,
            receipt_confidence,
            hash_nonzero: true,
            hash_score,
            validation_score,
            rejection_score: Q_BOTTOM,
            rollback_score: Q_BOTTOM,
            repair_score: Q_BOTTOM,
        }
    }

    pub const fn accepted_without_hash(receipt_confidence: f32, rejection_score: f32) -> Self {
        Self {
            accepted: true,
            receipt_confidence,
            hash_nonzero: false,
            hash_score: Q_BOTTOM,
            validation_score: Q_BOTTOM,
            rejection_score,
            rollback_score: rejection_score,
            repair_score: rejection_score,
        }
    }

    pub const fn rejected(rejection_score: f32, rollback_score: f32, repair_score: f32) -> Self {
        Self {
            accepted: false,
            receipt_confidence: Q_BOTTOM,
            hash_nonzero: false,
            hash_score: Q_BOTTOM,
            validation_score: Q_BOTTOM,
            rejection_score,
            rollback_score,
            repair_score,
        }
    }
}

const RECEIPT_GATE_WEIGHT: f32 = 0.97;

/// Compile a runtime execution receipt into ordinary matrix edges.
///
/// These edges are joined into the same CUDA-resident transition matrix as the
/// static graph and policy graph. This lets concrete receipt evidence alter the
/// reachable path weights without introducing a separate CPU planner.
pub fn build_receipt_edges(receipt: ExecutionReceipt) -> Vec<TransitionEdge> {
    let mut edges = Vec::with_capacity(6);

    edges.push(edge(
        Node::event(EventNode::ReceiptAttached),
        Node::control(ControlNode::GateReceipt),
        RECEIPT_GATE_WEIGHT,
    ));

    if receipt.accepted && receipt.hash_nonzero {
        edges.push(edge(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptAccepted),
            receipt.receipt_confidence,
        ));
        edges.push(edge(
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::HashNonzero),
            receipt.hash_score,
        ));
        edges.push(edge(
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate),
            receipt.validation_score,
        ));
    } else {
        edges.push(edge(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected),
            receipt.rejection_score,
        ));
        edges.push(edge(
            Node::event(EventNode::ReceiptRejected),
            Node::control(ControlNode::Rollback),
            receipt.rollback_score,
        ));
        edges.push(edge(
            Node::control(ControlNode::Rollback),
            Node::control(ControlNode::Repair),
            receipt.repair_score,
        ));
    }

    edges
}
