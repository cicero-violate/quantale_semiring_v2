//! Closure reports and decision projection types.

use cudarc::driver::DeviceRepr;

use crate::algebra::{Q_BOTTOM, Q_UNIT};
use crate::node::{ControlNode, EventNode, Node, StateNode};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

/// Compact report for π(A*): the gated executable projection of quantale closure.
#[repr(C)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
        match self.selected_node() {
            Some(Node::State(StateNode::Execute))
            | Some(Node::Event(EventNode::ExecuteStarted)) => QuantaleAction::RunExecutor,
            Some(Node::Control(ControlNode::Commit)) => QuantaleAction::Commit,
            Some(Node::Control(ControlNode::Retry)) => QuantaleAction::Retry,
            Some(Node::Control(ControlNode::Repair)) => QuantaleAction::Repair,
            Some(Node::Control(ControlNode::Rollback)) => QuantaleAction::Rollback,
            Some(Node::Control(ControlNode::Halt)) => QuantaleAction::Stop,
            Some(node) => QuantaleAction::ContinueTo(node),
            None => QuantaleAction::Unknown,
        }
    }
}

pub fn format_quantale_value(value: f32) -> String {
    if value <= Q_BOTTOM {
        "⊥".to_string()
    } else if (value - Q_UNIT).abs() <= f32::EPSILON {
        "e".to_string()
    } else {
        format!("{value:.4}")
    }
}
