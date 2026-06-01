//! Closure reports and decision projection types.

use cudarc::driver::DeviceRepr;
use serde::Serialize;

use crate::algebra::{BOTTOM, Q_UNIT};
use crate::node::Node;
use crate::topology::NodeRegistry;

/// Compact report for π(A*): the gated executable projection of least fixed point.
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

impl DecisionReport {
    pub fn selected_node(&self, registry: &NodeRegistry) -> Option<Node> {
        Node::decode(self.selected_dst, registry)
    }

    pub fn first_hop_node(&self, registry: &NodeRegistry) -> Option<Node> {
        Node::decode(self.first_hop, registry)
    }
}

pub fn action_label(node_id: i32, registry: &NodeRegistry) -> &str {
    if node_id < 0 {
        return "blocked";
    }
    registry.action_of(node_id as usize).unwrap_or("unknown")
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
