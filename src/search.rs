//! Runtime search evidence compiled into matrix-edge deltas.
//!
//! This module does not implement graph search, retrieval, or storage. External
//! candidate generators pass concrete candidates here; this module scores and
//! selects those candidates, then emits ordinary `TransitionEdge` deltas for the
//! quantale matrix.

use crate::algebra::Q_BOTTOM;
use crate::edge::{Eval, TransitionEdge, edge};
use crate::node::{EventNode, Node, StateNode};

#[derive(Clone, Debug, PartialEq)]
pub struct DomainCandidate {
    pub id: String,
    pub label: String,
    pub confidence: f32,
    pub utility: f32,
    pub risk: f32,
    pub cost: f32,
}

impl DomainCandidate {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        confidence: f32,
        utility: f32,
        risk: f32,
        cost: f32,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            confidence,
            utility,
            risk,
            cost,
        }
    }

    pub fn eval(&self) -> Eval {
        Eval::new(self.confidence, self.utility, self.risk, self.cost)
    }

    pub fn score(&self) -> f32 {
        self.eval().weight()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredCandidate {
    pub candidate: DomainCandidate,
    pub score: f32,
}

impl ScoredCandidate {
    pub fn from_candidate(candidate: DomainCandidate) -> Self {
        let score = candidate.score();
        Self { candidate, score }
    }
}

pub fn score_candidates(
    candidates: impl IntoIterator<Item = DomainCandidate>,
) -> Vec<ScoredCandidate> {
    candidates
        .into_iter()
        .map(ScoredCandidate::from_candidate)
        .filter(|candidate| candidate.score > Q_BOTTOM)
        .collect()
}

pub fn select_top_k(
    candidates: impl IntoIterator<Item = ScoredCandidate>,
    k: usize,
) -> Vec<ScoredCandidate> {
    if k == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<ScoredCandidate> = candidates.into_iter().collect();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.candidate.id.cmp(&right.candidate.id))
    });
    candidates.truncate(k);
    candidates
}

/// Compile selected search candidates into ordinary matrix edges.
///
/// The returned edges are joined into the CUDA-resident transition matrix with
/// the same max-times semantics as static transitions, policy edges, and
/// receipt edges: `M := M ∨ M_search`.
pub fn build_search_edges(top_k: &[ScoredCandidate]) -> Vec<TransitionEdge> {
    let mut edges = Vec::with_capacity(3);

    if top_k.is_empty() {
        return edges;
    }

    let best_score = top_k
        .iter()
        .map(|candidate| candidate.score)
        .fold(Q_BOTTOM, f32::max);
    let mean_score =
        top_k.iter().map(|candidate| candidate.score).sum::<f32>() / top_k.len() as f32;

    edges.push(edge(
        Node::state(StateNode::Search),
        Node::event(EventNode::CandidateFound),
        best_score,
    ));
    edges.push(edge(
        Node::state(StateNode::Score),
        Node::event(EventNode::ScoreReady),
        best_score,
    ));
    edges.push(edge(
        Node::state(StateNode::Select),
        Node::event(EventNode::TopKSelected),
        mean_score,
    ));

    edges
}

pub fn build_search_delta_edges(
    candidates: impl IntoIterator<Item = DomainCandidate>,
    k: usize,
) -> (Vec<ScoredCandidate>, Vec<TransitionEdge>) {
    let top_k = select_top_k(score_candidates(candidates), k);
    let edges = build_search_edges(&top_k);
    (top_k, edges)
}

/// Compatibility name for the external candidate-generation boundary.
///
/// This deliberately does not search the quantale graph. It accepts candidates
/// produced by an external source and compiles the selected evidence into matrix
/// deltas for `M := M ∨ ΔM`.
pub fn build_candidate_edges(
    candidates: impl IntoIterator<Item = DomainCandidate>,
    k: usize,
) -> (Vec<ScoredCandidate>, Vec<TransitionEdge>) {
    build_search_delta_edges(candidates, k)
}
