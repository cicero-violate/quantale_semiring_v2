//! Closure reports and decision projection types.

use cudarc::driver::DeviceRepr;
use serde::Serialize;

use crate::algebra::{BOTTOM, Q_UNIT};
use crate::node::{ControlNode, EventNode, Node, StateNode};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct QuantaleCudaReport {
    pub step: i32,
    pub best_src: i32,
    pub best_dst: i32,
    pub best_value: f32,
    pub event_count: i32,
    pub goal_to_execute: f32,
    pub goal_to_learn: f32,
}

unsafe impl DeviceRepr for QuantaleCudaReport {}

/// Compact report for π(A*): the gated executable projection of least fixed point.
#[repr(C)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct DecisionReport {
    pub step: i32,
    pub selected_src: i32,
    pub selected_dst: i32,
    pub first_hop: i32,
    pub selected_value: f32,
    pub halted: i32,
    pub blocked: i32,
}

unsafe impl DeviceRepr for DecisionReport {}

/// Alias used when the algebraic role matters: this is not closure A*,
/// but the decision projection π(A*) plus the W witness first hop.
pub type DecisionProjection = DecisionReport;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantaleAction {
    RunExecutor,
    Commit,
    Retry,
    Repair,
    Rollback,
    Stop,
    ContinueTo(Node),
    Blocked,
    Unknown,
}

impl DecisionReport {
    pub fn selected_node(&self) -> Option<Node> {
        Node::decode(self.selected_dst)
    }

    pub fn first_hop_node(&self) -> Option<Node> {
        Node::decode(self.first_hop)
    }

    pub fn selected_action(&self) -> QuantaleAction {
        if self.blocked != 0 {
            return QuantaleAction::Blocked;
        }
        let Some(node) = self.selected_node() else {
            return QuantaleAction::Unknown;
        };
        if node == Node::state(StateNode::Execute) || node == Node::event(EventNode::ExecuteStarted)
        {
            QuantaleAction::RunExecutor
        } else if node == Node::control(ControlNode::Commit) {
            QuantaleAction::Commit
        } else if node == Node::control(ControlNode::Retry) {
            QuantaleAction::Retry
        } else if node == Node::control(ControlNode::Repair) {
            QuantaleAction::Repair
        } else if node == Node::control(ControlNode::Rollback) {
            QuantaleAction::Rollback
        } else if node == Node::control(ControlNode::Halt) {
            QuantaleAction::Stop
        } else {
            QuantaleAction::ContinueTo(node)
        }
    }
}

pub fn format_quantale_value(value: f32) -> String {
    if value <= BOTTOM {
        "⊥".to_string()
    } else if (value - Q_UNIT).abs() <= f32::EPSILON {
        "e".to_string()
    } else {
        format!("{value:.4}")
    }
}
