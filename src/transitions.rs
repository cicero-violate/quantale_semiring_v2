//! Static transition graph.

use crate::algebra::{Q_BOTTOM, Q_UNIT};
use crate::edge::{Eval, TransitionEdge, edge_eval};
use crate::node::{ControlNode, EventNode, Node, StateNode};

pub fn default_transition_edges() -> Vec<TransitionEdge> {
    vec![
        edge_eval(
            Node::state(StateNode::Goal),
            Node::control(ControlNode::GateInput),
            Eval::new(0.99, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateInput),
            Node::event(EventNode::FactArrived),
            Eval::new(0.98, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::FactArrived),
            Node::state(StateNode::Input),
            Eval::new(0.97, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Input),
            Node::event(EventNode::InputAccepted),
            Eval::new(0.96, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::InputAccepted),
            Node::state(StateNode::Parse),
            Eval::new(0.95, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Parse),
            Node::event(EventNode::ParseOk),
            Eval::new(0.92, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Parse),
            Node::event(EventNode::ParseErr),
            Eval::new(0.10, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ParseOk),
            Node::state(StateNode::Map),
            Eval::new(0.90, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ParseErr),
            Node::control(ControlNode::Repair),
            Eval::new(0.60, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::Repair),
            Node::state(StateNode::Parse),
            Eval::new(0.55, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Map),
            Node::event(EventNode::MapReady),
            Eval::new(0.93, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::MapReady),
            Node::state(StateNode::Search),
            Eval::new(0.92, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Search),
            Node::event(EventNode::CandidateFound),
            Eval::new(0.91, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::CandidateFound),
            Node::state(StateNode::Score),
            Eval::new(0.92, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Score),
            Node::event(EventNode::ScoreReady),
            Eval::new(0.94, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ScoreReady),
            Node::state(StateNode::Select),
            Eval::new(0.95, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Select),
            Node::event(EventNode::TopKSelected),
            Eval::new(0.94, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::TopKSelected),
            Node::state(StateNode::Plan),
            Eval::new(0.93, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Plan),
            Node::event(EventNode::PlanReady),
            Eval::new(0.92, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::PlanReady),
            Node::control(ControlNode::ChooseBest),
            Eval::new(0.96, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::ChooseBest),
            Node::state(StateNode::Execute),
            Eval::new(0.88, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::ChooseBest),
            Node::state(StateNode::Optimize),
            Eval::new(0.70, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Optimize),
            Node::event(EventNode::OptimizeReady),
            Eval::new(0.75, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::OptimizeReady),
            Node::state(StateNode::Execute),
            Eval::new(0.82, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Execute),
            Node::control(ControlNode::GateExecution),
            Eval::new(0.97, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateExecution),
            Node::event(EventNode::ExecuteStarted),
            Eval::new(0.96, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateExecution),
            Node::control(ControlNode::Block),
            Eval::new(0.05, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ExecuteStarted),
            Node::event(EventNode::ExecuteFinished),
            Eval::new(0.90, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ExecuteFinished),
            Node::event(EventNode::ReceiptAttached),
            Eval::new(0.85, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ReceiptAttached),
            Node::control(ControlNode::GateReceipt),
            Eval::new(0.97, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptAccepted),
            Eval::new(0.90, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateReceipt),
            Node::event(EventNode::ReceiptRejected),
            Eval::new(0.15, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ReceiptAccepted),
            Node::event(EventNode::HashNonzero),
            Eval::new(0.95, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::HashNonzero),
            Node::state(StateNode::Validate),
            Eval::new(0.96, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::ReceiptRejected),
            Node::control(ControlNode::Rollback),
            Eval::new(0.80, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::Rollback),
            Node::control(ControlNode::Repair),
            Eval::new(0.60, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Validate),
            Node::control(ControlNode::Commit),
            Eval::new(0.94, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Validate),
            Node::control(ControlNode::Block),
            Eval::new(0.05, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::Commit),
            Node::control(ControlNode::GateMemory),
            Eval::new(0.98, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
    ]
}

pub fn persistence_transition_edges() -> Vec<TransitionEdge> {
    vec![
        edge_eval(
            Node::control(ControlNode::GateMemory),
            Node::state(StateNode::Memory),
            Eval::new(0.95, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Memory),
            Node::event(EventNode::MemoryWritten),
            Eval::new(0.92, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::MemoryWritten),
            Node::control(ControlNode::GateLearn),
            Eval::new(0.90, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::control(ControlNode::GateLearn),
            Node::state(StateNode::Learn),
            Eval::new(0.88, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::state(StateNode::Learn),
            Node::event(EventNode::LearnUpdated),
            Eval::new(0.91, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
        edge_eval(
            Node::event(EventNode::LearnUpdated),
            Node::control(ControlNode::Halt),
            Eval::new(0.99, Q_UNIT, Q_BOTTOM, Q_BOTTOM),
        ),
    ]
}

pub fn full_transition_edges() -> Vec<TransitionEdge> {
    let mut edges = Vec::with_capacity(45);
    edges.extend(default_transition_edges());
    edges.extend(persistence_transition_edges());
    edges
}

/// Load transition edges from the bundled JSON topology asset.
///
/// This is the data-driven equivalent of `full_transition_edges()`. The current
/// CUDA kernels still require the canonical 44-node universe, so the topology
/// compiler validates node IDs before returning edges.
pub fn data_driven_transition_edges() -> Result<Vec<TransitionEdge>, crate::CudaError> {
    crate::topology::load_default_topology_edges()
}
