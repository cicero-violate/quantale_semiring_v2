//! Matrix edges and scalar evaluation.

use cudarc::driver::DeviceRepr;

use crate::node::Node;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionEdge {
    pub src: i32,
    pub dst: i32,
    pub value: f32,
}

unsafe impl DeviceRepr for TransitionEdge {}

impl TransitionEdge {
    pub const fn new(src: i32, dst: i32, value: f32) -> Self {
        Self { src, dst, value }
    }

    pub const fn from_nodes(src: Node, dst: Node, value: f32) -> Self {
        Self {
            src: src.encode(),
            dst: dst.encode(),
            value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eval {
    pub confidence: f32,
    pub utility: f32,
    pub risk: f32,
    pub cost: f32,
}

impl Eval {
    pub const fn new(confidence: f32, utility: f32, risk: f32, cost: f32) -> Self {
        Self {
            confidence,
            utility,
            risk,
            cost,
        }
    }

    pub fn weight(self) -> f32 {
        fn clamp(value: f32) -> f32 {
            if value.is_nan() || value <= 0.0 {
                0.0
            } else if value >= 1.0 {
                1.0
            } else {
                value
            }
        }

        clamp(self.confidence)
            * clamp(self.utility)
            * (1.0 - clamp(self.risk))
            * (1.0 - clamp(self.cost))
    }
}

pub const fn edge(src: Node, dst: Node, value: f32) -> TransitionEdge {
    TransitionEdge::from_nodes(src, dst, value)
}

pub fn edge_eval(src: Node, dst: Node, eval: Eval) -> TransitionEdge {
    TransitionEdge::from_nodes(src, dst, eval.weight())
}
