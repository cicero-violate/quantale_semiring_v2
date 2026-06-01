//! Static topology graph checker.
//!
//! Validates a `GraphTopology` against invariants before any tensor operations
//! run.  Call `check()` after parsing topology.json and before creating
//! `TensorQuantaleWorld`.  Returns every violation found; the caller decides
//! whether to abort.
//!
//! Invariants enforced by `check()`
//! ---------------------------------
//! Phase 1 — identity and weight (invariants 1–7, 13)
//!   1.  Unique node names           (DuplicateNodeName)
//!   1b. Unique node ids             (DuplicateNodeId)
//!   2.  Stable index mapping        (IndexMappingBroken)
//!   3.  Start node validity         (InvalidStartNode)
//!   4.  Halt nodes exist + outdeg=0 (NoHaltNode / HaltNodeHasSuccessors)
//!   5.  Edge uniqueness             (DuplicateEdge)
//!   6.  Weight domain validity      (WeightOutOfDomain)
//!   7.  Zero-confidence edge warn   (ZeroConfidenceEdge)
//!  13.  Deterministic tie-break     (IndeterminateOrdering)
//!
//! Structural checks (original five):
//!   endpoint validity, dead-end, reachability, path-to-halt
//!
//! Phase 3 — dominator and cycle checks (invariants 9–12)
//!   9.  Gate dominance         (DominanceViolation)
//!  10.  Receipt cutset         (ReceiptCutsetViolation)
//!  11.  SCC progress / exit    (UnsafeSCC)
//!  12.  No zero-cost cycle     (ZeroCostCycle)
//!
//! Phase 2 — operator binding (invariant 8) is in `check_with_operators()`.
//! Phase 4 — semiring law tests are in `tests/semiring_laws.rs`.
//! Phase 5 — frontier one-hot and reset assertions are in tensor.rs / main.rs.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::config::OperatorRegistry;
use crate::topology::{GraphTopology, TopologyTransition};

// ── Gate dominance configuration ──────────────────────────────────────────────

/// (gate, protected) pairs: every path from start to `protected` must pass
/// through `gate`.
///
/// Note: ("Control::Commit", "State::Memory") is intentionally absent because
/// the market trading path (PaperTradeFilled/PaperTradeRejected → Memory) is a
/// legitimate bypass of the commit gate.  The receipt-chain dominance pairs
/// below still protect the primary sequential path.
const REQUIRED_DOMINATORS: &[(&str, &str)] = &[
    ("State::Validate", "Control::Commit"),
    ("Control::GateReceipt", "Event::ReceiptAccepted"),
    ("Event::ReceiptAccepted", "Event::HashNonzero"),
    ("Event::HashNonzero", "State::Validate"),
];

/// Every path from `Event::ExecuteFinished` to `Control::Commit` must contain
/// all of these nodes.
const RECEIPT_CUTSET: &[&str] = &[
    "Event::ReceiptAttached",
    "Control::GateReceipt",
    "Event::ReceiptAccepted",
    "Event::HashNonzero",
    "State::Validate",
];

// ── ViolationKind ─────────────────────────────────────────────────────────────

/// Describes why a topology check failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    // ── original structural checks ───────────────────────────────────────────
    /// A transition references a name not in nodes[].
    UnknownEndpoint,
    /// A reachable non-halt node has outdeg = 0.
    DeadEnd,
    /// A pages-listed node is not reachable from the start node.
    Unreachable,
    /// A reachable node has no path to any halt node.
    CannotReachHalt,

    // ── Phase 1: identity and weight ─────────────────────────────────────────
    /// Two nodes share the same name.
    DuplicateNodeName,
    /// Two nodes share the same id.
    DuplicateNodeId,
    /// Round-trip name→id→name or id→name→id is inconsistent.
    IndexMappingBroken,
    /// nodes[0] does not have id = 0 (silent re-ordering).
    InvalidStartNode,
    /// No node has action = "halt".
    NoHaltNode,
    /// A halt node has outgoing transitions.
    HaltNodeHasSuccessors,
    /// Two transitions share the same (from, to) pair.
    DuplicateEdge,
    /// An effective edge weight is outside its valid domain.
    WeightOutOfDomain,
    /// A declared edge has effective confidence = 0.0.
    ZeroConfidenceEdge,
    /// Two outgoing edges from the same source have identical
    /// (default_weight, safety, cost, to) — tie-break is non-deterministic.
    IndeterminateOrdering,

    // ── Phase 2: operator binding ─────────────────────────────────────────────
    /// A State::* / Control::* node has no entry in operators.json.
    MissingOperator,
    /// A State::* / Control::* node's operator entry has no `action` or
    /// `output_mode` field — executor will report action="unknown".
    UnknownActionSemantics,

    // ── Phase 3: dominator and cycle checks ───────────────────────────────────
    /// A required gate does not dominate its protected node.
    DominanceViolation,
    /// A receipt-chain cutset member does not dominate Control::Commit on
    /// paths from Event::ExecuteFinished.
    ReceiptCutsetViolation,
    /// A non-trivial non-halt SCC has no exit edge.
    UnsafeSCC,
    /// A reachable SCC contains a cycle where every edge has cost = 0.
    ZeroCostCycle,
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownEndpoint => write!(f, "unknown_endpoint"),
            Self::DeadEnd => write!(f, "dead_end"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::CannotReachHalt => write!(f, "cannot_reach_halt"),
            Self::DuplicateNodeName => write!(f, "duplicate_node_name"),
            Self::DuplicateNodeId => write!(f, "duplicate_node_id"),
            Self::IndexMappingBroken => write!(f, "index_mapping_broken"),
            Self::InvalidStartNode => write!(f, "invalid_start_node"),
            Self::NoHaltNode => write!(f, "no_halt_node"),
            Self::HaltNodeHasSuccessors => write!(f, "halt_node_has_successors"),
            Self::DuplicateEdge => write!(f, "duplicate_edge"),
            Self::WeightOutOfDomain => write!(f, "weight_out_of_domain"),
            Self::ZeroConfidenceEdge => write!(f, "zero_confidence_edge"),
            Self::IndeterminateOrdering => write!(f, "indeterminate_ordering"),
            Self::MissingOperator => write!(f, "missing_operator"),
            Self::UnknownActionSemantics => write!(f, "unknown_action_semantics"),
            Self::DominanceViolation => write!(f, "dominance_violation"),
            Self::ReceiptCutsetViolation => write!(f, "receipt_cutset_violation"),
            Self::UnsafeSCC => write!(f, "unsafe_scc"),
            Self::ZeroCostCycle => write!(f, "zero_cost_cycle"),
        }
    }
}

// ── TopologyViolation ─────────────────────────────────────────────────────────

/// A single topology invariant violation.
#[derive(Clone, Debug)]
pub struct TopologyViolation {
    pub kind: ViolationKind,
    pub node: String,
    pub detail: String,
}

impl std::fmt::Display for TopologyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.kind, self.node, self.detail)
    }
}

// ── Public entry points ───────────────────────────────────────────────────────

/// Run all checks (phases 1 and 3) plus the original structural checks on
/// `topology`.  Returns every violation found; the caller decides whether to
/// abort.  All passes run so the caller sees every problem at once.
pub fn check(topology: &GraphTopology) -> Vec<TopologyViolation> {
    let mut violations = Vec::new();
    check_phase1(&mut violations, topology);
    check_structural(&mut violations, topology);
    check_phase3(&mut violations, topology);
    violations
}

/// Run operator-binding checks (phase 2) in addition to all other checks.
pub fn check_with_operators(
    topology: &GraphTopology,
    operator_registry: &OperatorRegistry,
) -> Vec<TopologyViolation> {
    let mut violations = check(topology);
    check_phase2(&mut violations, topology, operator_registry);
    violations
}

/// Format a violation list for stderr output.
pub fn format_violations(violations: &[TopologyViolation]) -> String {
    violations
        .iter()
        .map(|v| format!("[topology] {v}"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Phase 1: identity and weight ─────────────────────────────────────────────

fn check_phase1(violations: &mut Vec<TopologyViolation>, topology: &GraphTopology) {
    // ── 1. Unique node names ──────────────────────────────────────────────────
    let mut seen_names: HashSet<&str> = HashSet::new();
    for node in &topology.nodes {
        if !seen_names.insert(node.name.as_str()) {
            violations.push(TopologyViolation {
                kind: ViolationKind::DuplicateNodeName,
                node: node.name.clone(),
                detail: format!("'{}' appears more than once in nodes[]", node.name),
            });
        }
    }

    // ── 1b. Unique node ids ───────────────────────────────────────────────────
    let mut seen_ids: HashSet<usize> = HashSet::new();
    for node in &topology.nodes {
        if !seen_ids.insert(node.id) {
            violations.push(TopologyViolation {
                kind: ViolationKind::DuplicateNodeId,
                node: node.name.clone(),
                detail: format!("id {} used by more than one node", node.id),
            });
        }
    }

    // ── 2. Stable index mapping ───────────────────────────────────────────────
    // Build maps from the first occurrence of each name / id (duplicates already
    // reported above; skip them here to avoid noise).
    let name_to_id: HashMap<&str, usize> = topology
        .nodes
        .iter()
        .map(|n| (n.name.as_str(), n.id))
        .collect();
    let id_to_name: HashMap<usize, &str> =
        topology.nodes.iter().map(|n| (n.id, n.name.as_str())).collect();
    for node in &topology.nodes {
        // name → id → name must round-trip
        if let Some(&mapped_id) = name_to_id.get(node.name.as_str()) {
            if let Some(&back_name) = id_to_name.get(&mapped_id) {
                if back_name != node.name.as_str() {
                    violations.push(TopologyViolation {
                        kind: ViolationKind::IndexMappingBroken,
                        node: node.name.clone(),
                        detail: format!(
                            "round-trip name→id→name broken: '{}' → {} → '{}'",
                            node.name, mapped_id, back_name
                        ),
                    });
                }
            }
        }
        // id → name → id must round-trip
        if let Some(&mapped_name) = id_to_name.get(&node.id) {
            if let Some(&back_id) = name_to_id.get(mapped_name) {
                if back_id != node.id {
                    violations.push(TopologyViolation {
                        kind: ViolationKind::IndexMappingBroken,
                        node: node.name.clone(),
                        detail: format!(
                            "round-trip id→name→id broken: {} → '{}' → {}",
                            node.id, mapped_name, back_id
                        ),
                    });
                }
            }
        }
    }

    // ── 3. Start node validity ────────────────────────────────────────────────
    match topology.nodes.first() {
        None => {
            violations.push(TopologyViolation {
                kind: ViolationKind::InvalidStartNode,
                node: "(none)".to_string(),
                detail: "nodes[] is empty; no start node".to_string(),
            });
        }
        Some(first) if first.id != 0 => {
            violations.push(TopologyViolation {
                kind: ViolationKind::InvalidStartNode,
                node: first.name.clone(),
                detail: format!(
                    "nodes[0] has id={} but start convention requires id=0",
                    first.id
                ),
            });
        }
        _ => {}
    }

    // ── 4. Halt node existence and outdeg = 0 ─────────────────────────────────
    let halt_names: HashSet<&str> = topology
        .nodes
        .iter()
        .filter(|n| n.action.as_deref() == Some("halt"))
        .map(|n| n.name.as_str())
        .collect();
    if halt_names.is_empty() {
        violations.push(TopologyViolation {
            kind: ViolationKind::NoHaltNode,
            node: "(topology)".to_string(),
            detail: "no node declares action=\"halt\"; path-to-halt check is vacuous".to_string(),
        });
    }
    for t in &topology.transitions {
        if halt_names.contains(t.from.as_str()) {
            violations.push(TopologyViolation {
                kind: ViolationKind::HaltNodeHasSuccessors,
                node: t.from.clone(),
                detail: format!(
                    "halt node '{}' has outgoing transition to '{}'; execution cannot stop safely",
                    t.from, t.to
                ),
            });
        }
    }

    // ── 5. Edge uniqueness ────────────────────────────────────────────────────
    let mut seen_edges: HashSet<(&str, &str)> = HashSet::new();
    for t in &topology.transitions {
        if !seen_edges.insert((t.from.as_str(), t.to.as_str())) {
            violations.push(TopologyViolation {
                kind: ViolationKind::DuplicateEdge,
                node: t.from.clone(),
                detail: format!(
                    "duplicate transition '{}' → '{}'; causes non-deterministic weight \
                     overwrite in embed_tensor_edges",
                    t.from, t.to
                ),
            });
        }
    }

    // ── 6 + 7. Weight domain validity and zero-confidence warning ─────────────
    for t in &topology.transitions {
        check_edge_weights(violations, t);
    }

    // ── 13. Deterministic tie-break ───────────────────────────────────────────
    // Group outgoing edges by source; flag pairs with identical
    // (default_weight bits, safety bits, effective_cost bits, to_name).
    let mut by_src: HashMap<&str, Vec<&TopologyTransition>> = HashMap::new();
    for t in &topology.transitions {
        by_src.entry(t.from.as_str()).or_default().push(t);
    }
    for (&src, edges) in &by_src {
        let mut seen_tuples: HashSet<(u32, u32, u32, &str)> = HashSet::new();
        for t in edges {
            let dw = t.default_weight.raw().to_bits();
            let safety = t
                .safety
                .map(|s| s.raw())
                .unwrap_or_else(|| t.default_weight.raw())
                .to_bits();
            let cost = t
                .cost
                .unwrap_or_else(|| 1.0 - t.default_weight.raw())
                .to_bits();
            let key = (dw, safety, cost, t.to.as_str());
            if !seen_tuples.insert(key) {
                violations.push(TopologyViolation {
                    kind: ViolationKind::IndeterminateOrdering,
                    node: src.to_string(),
                    detail: format!(
                        "'{}' → '{}': two edges share identical (default_weight, safety, \
                         cost) — next-hop selection is non-deterministic",
                        src, t.to
                    ),
                });
            }
        }
    }
}

fn check_edge_weights(violations: &mut Vec<TopologyViolation>, t: &TopologyTransition) {
    let eff_confidence = t.confidence.map(|c| c.raw()).unwrap_or_else(|| t.default_weight.raw());
    let eff_safety = t.safety.map(|s| s.raw()).unwrap_or_else(|| t.default_weight.raw());
    let eff_cost = t.cost.unwrap_or_else(|| 1.0 - t.default_weight.raw());
    let label = format!("'{}' → '{}'", t.from, t.to);

    if !valid_unit(eff_confidence) {
        violations.push(TopologyViolation {
            kind: ViolationKind::WeightOutOfDomain,
            node: t.from.clone(),
            detail: format!(
                "{label}: effective confidence={eff_confidence} not in [0,1] or non-finite"
            ),
        });
    }
    if !valid_unit(eff_safety) {
        violations.push(TopologyViolation {
            kind: ViolationKind::WeightOutOfDomain,
            node: t.from.clone(),
            detail: format!(
                "{label}: effective safety={eff_safety} not in [0,1] or non-finite"
            ),
        });
    }
    if !valid_cost(eff_cost) {
        violations.push(TopologyViolation {
            kind: ViolationKind::WeightOutOfDomain,
            node: t.from.clone(),
            detail: format!("{label}: effective cost={eff_cost} is negative or non-finite"),
        });
    }
    if eff_confidence == 0.0 {
        violations.push(TopologyViolation {
            kind: ViolationKind::ZeroConfidenceEdge,
            node: t.from.clone(),
            detail: format!(
                "{label}: effective confidence=0.0; only missing transitions should \
                 produce ⊥ rows (W[i,j]=⊥ ⟺ (i,j)∉E)"
            ),
        });
    }
}

#[inline]
fn valid_unit(x: f32) -> bool {
    x.is_finite() && x >= 0.0 && x <= 1.0
}

#[inline]
fn valid_cost(x: f32) -> bool {
    x.is_finite() && x >= 0.0
}

// ── Original structural checks ────────────────────────────────────────────────

fn check_structural(violations: &mut Vec<TopologyViolation>, topology: &GraphTopology) {
    let node_ids: HashMap<&str, usize> = topology
        .nodes
        .iter()
        .map(|n| (n.name.as_str(), n.id))
        .collect();
    let halt_ids: HashSet<usize> = topology
        .nodes
        .iter()
        .filter(|n| n.action.as_deref() == Some("halt"))
        .map(|n| n.id)
        .collect();
    let id_to_name: HashMap<usize, &str> =
        topology.nodes.iter().map(|n| (n.id, n.name.as_str())).collect();

    let mut forward: HashMap<usize, Vec<usize>> =
        topology.nodes.iter().map(|n| (n.id, Vec::new())).collect();
    let mut reverse: HashMap<usize, Vec<usize>> =
        topology.nodes.iter().map(|n| (n.id, Vec::new())).collect();

    // Pass 1: endpoint validity
    for t in &topology.transitions {
        let src = node_ids.get(t.from.as_str()).copied();
        let dst = node_ids.get(t.to.as_str()).copied();
        if src.is_none() {
            violations.push(TopologyViolation {
                kind: ViolationKind::UnknownEndpoint,
                node: t.from.clone(),
                detail: format!(
                    "'{}' used as transition source but not declared in nodes",
                    t.from
                ),
            });
        }
        if dst.is_none() {
            violations.push(TopologyViolation {
                kind: ViolationKind::UnknownEndpoint,
                node: t.to.clone(),
                detail: format!(
                    "'{}' used as transition destination but not declared in nodes",
                    t.to
                ),
            });
        }
        if let (Some(s), Some(d)) = (src, dst) {
            forward.entry(s).or_default().push(d);
            reverse.entry(d).or_default().push(s);
        }
    }

    let reachable = bfs(&forward, 0);

    // Pass 2: dead-end
    for &nid in &reachable {
        if halt_ids.contains(&nid) {
            continue;
        }
        if forward.get(&nid).map(|v| v.is_empty()).unwrap_or(true) {
            let name = id_to_name.get(&nid).copied().unwrap_or("?");
            violations.push(TopologyViolation {
                kind: ViolationKind::DeadEnd,
                node: name.to_string(),
                detail: format!(
                    "'{}' is reachable and not a halt node but has no outgoing \
                     transitions (outdeg=0); execution produces Unknown(-1) blocked",
                    name
                ),
            });
        }
    }

    // Pass 3: reachability of page-listed nodes
    let page_names: HashSet<&str> = topology
        .pages
        .iter()
        .flat_map(|p| p.node_names.iter().map(String::as_str))
        .collect();
    for name in page_names {
        let Some(&nid) = node_ids.get(name) else {
            continue;
        };
        if !reachable.contains(&nid) {
            violations.push(TopologyViolation {
                kind: ViolationKind::Unreachable,
                node: name.to_string(),
                detail: format!(
                    "'{}' is listed in pages but unreachable from start (id=0)",
                    name
                ),
            });
        }
    }

    // Pass 4: path-to-halt
    let can_reach_halt = bfs_multi(&reverse, &halt_ids);
    for &nid in &reachable {
        if !can_reach_halt.contains(&nid) {
            let name = id_to_name.get(&nid).copied().unwrap_or("?");
            violations.push(TopologyViolation {
                kind: ViolationKind::CannotReachHalt,
                node: name.to_string(),
                detail: format!(
                    "'{}' is reachable but has no path to any halt node",
                    name
                ),
            });
        }
    }
}

// ── Phase 2: operator binding ─────────────────────────────────────────────────

fn check_phase2(
    violations: &mut Vec<TopologyViolation>,
    topology: &GraphTopology,
    operator_registry: &OperatorRegistry,
) {
    for node in &topology.nodes {
        // Only check State and Control nodes; skip Event nodes and halt nodes
        let is_state_or_control =
            node.node_type == "State" || node.node_type == "Control";
        if !is_state_or_control {
            continue;
        }
        if node.action.as_deref() == Some("halt") {
            continue;
        }
        match operator_registry.get(&node.name) {
            None => {
                violations.push(TopologyViolation {
                    kind: ViolationKind::MissingOperator,
                    node: node.name.clone(),
                    detail: format!(
                        "'{}' (type={}) has no operator binding in operators.json; \
                         executor will return exit 127",
                        node.name, node.node_type
                    ),
                });
            }
            // Invariant 25: operator entry must have a non-empty action or output_mode
            Some(op) => {
                let has_action = op
                    .get("action")
                    .and_then(|v| v.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                let has_output_mode = op
                    .get("output_mode")
                    .and_then(|v| v.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                if !has_action && !has_output_mode {
                    violations.push(TopologyViolation {
                        kind: ViolationKind::UnknownActionSemantics,
                        node: node.name.clone(),
                        detail: format!(
                            "'{}' operator entry has neither 'action' nor 'output_mode'; \
                             executor will report action=\"unknown\"",
                            node.name
                        ),
                    });
                }
            }
        }
    }
}

// ── Phase 3: dominator and cycle checks ──────────────────────────────────────

fn check_phase3(violations: &mut Vec<TopologyViolation>, topology: &GraphTopology) {
    if topology.nodes.is_empty() {
        return;
    }

    let node_ids: HashMap<&str, usize> = topology
        .nodes
        .iter()
        .map(|n| (n.name.as_str(), n.id))
        .collect();
    let halt_ids: HashSet<usize> = topology
        .nodes
        .iter()
        .filter(|n| n.action.as_deref() == Some("halt"))
        .map(|n| n.id)
        .collect();
    let id_to_name: HashMap<usize, &str> =
        topology.nodes.iter().map(|n| (n.id, n.name.as_str())).collect();

    // Build adjacency (skip unknown endpoints — already reported in structural pass)
    let mut forward: HashMap<usize, Vec<usize>> =
        topology.nodes.iter().map(|n| (n.id, Vec::new())).collect();
    let mut reverse: HashMap<usize, Vec<usize>> =
        topology.nodes.iter().map(|n| (n.id, Vec::new())).collect();
    for t in &topology.transitions {
        let (Some(&s), Some(&d)) = (
            node_ids.get(t.from.as_str()),
            node_ids.get(t.to.as_str()),
        ) else {
            continue;
        };
        forward.entry(s).or_default().push(d);
        reverse.entry(d).or_default().push(s);
    }

    let reachable = bfs(&forward, 0);
    let all_ids: Vec<usize> = topology.nodes.iter().map(|n| n.id).collect();

    // ── Invariants 9 + 10: dominator checks ───────────────────────────────────
    let dom = compute_dominators(&reverse, &all_ids, 0);

    // Invariant 9: required gate dominance pairs
    for &(gate_name, protected_name) in REQUIRED_DOMINATORS {
        let (Some(&gate_id), Some(&protected_id)) = (
            node_ids.get(gate_name),
            node_ids.get(protected_name),
        ) else {
            continue;
        };
        if !reachable.contains(&protected_id) {
            continue;
        }
        if let Some(protected_dom) = dom.get(&protected_id) {
            if !protected_dom.contains(&gate_id) {
                violations.push(TopologyViolation {
                    kind: ViolationKind::DominanceViolation,
                    node: protected_name.to_string(),
                    detail: format!(
                        "gate '{}' does not dominate '{}'; a bypass path reaches \
                         '{}' without passing through '{}'",
                        gate_name, protected_name, protected_name, gate_name
                    ),
                });
            }
        }
    }

    // Invariant 10: receipt cutset check
    if let (Some(&exec_fin_id), Some(&commit_id)) = (
        node_ids.get("Event::ExecuteFinished"),
        node_ids.get("Control::Commit"),
    ) {
        check_receipt_cutset(
            violations,
            &node_ids,
            &forward,
            &reverse,
            exec_fin_id,
            commit_id,
        );
    }

    // ── Invariants 11 + 12: SCC checks ────────────────────────────────────────
    let sccs = kosaraju_sccs(&forward, &reverse, &all_ids);
    let mut node_to_scc: HashMap<usize, usize> = HashMap::new();
    for (idx, scc) in sccs.iter().enumerate() {
        for &nid in scc {
            node_to_scc.insert(nid, idx);
        }
    }

    // Effective cost per edge: (src, dst) → cost
    let edge_costs: HashMap<(usize, usize), f32> = topology
        .transitions
        .iter()
        .filter_map(|t| {
            let s = *node_ids.get(t.from.as_str())?;
            let d = *node_ids.get(t.to.as_str())?;
            let cost = t.cost.unwrap_or_else(|| 1.0 - t.default_weight.raw());
            Some(((s, d), cost))
        })
        .collect();

    for (scc_idx, scc) in sccs.iter().enumerate() {
        // Non-trivial: size > 1, or size 1 with a self-loop
        let scc_set: HashSet<usize> = scc.iter().copied().collect();
        let self_loop = scc.len() == 1 && {
            let nid = scc[0];
            forward.get(&nid).map(|v| v.contains(&nid)).unwrap_or(false)
        };
        if scc.len() <= 1 && !self_loop {
            continue;
        }

        // Only reachable SCCs
        if !scc.iter().any(|nid| reachable.contains(nid)) {
            continue;
        }
        // Halt-containing SCCs are allowed
        if scc.iter().any(|nid| halt_ids.contains(nid)) {
            continue;
        }

        // Invariant 11: must have at least one exit edge
        let has_exit = scc.iter().any(|&src| {
            forward
                .get(&src)
                .into_iter()
                .flatten()
                .any(|&dst| node_to_scc.get(&dst) != Some(&scc_idx))
        });
        if !has_exit {
            let names: Vec<&str> = scc
                .iter()
                .filter_map(|nid| id_to_name.get(nid).copied())
                .collect();
            violations.push(TopologyViolation {
                kind: ViolationKind::UnsafeSCC,
                node: names.first().copied().unwrap_or("?").to_string(),
                detail: format!(
                    "SCC {:?} has no exit edge; planner loops indefinitely without \
                     reaching halt",
                    names
                ),
            });
        }

        // Invariant 12: no zero-cost infinite cycle
        let internal_edges: Vec<(usize, usize)> = scc
            .iter()
            .flat_map(|&src| {
                forward
                    .get(&src)
                    .into_iter()
                    .flatten()
                    .filter(|&&dst| scc_set.contains(&dst))
                    .map(move |&dst| (src, dst))
            })
            .collect();
        if !internal_edges.is_empty() {
            let all_zero = internal_edges
                .iter()
                .all(|key| edge_costs.get(key).copied().unwrap_or(0.0) == 0.0);
            if all_zero {
                let names: Vec<&str> = scc
                    .iter()
                    .filter_map(|nid| id_to_name.get(nid).copied())
                    .collect();
                violations.push(TopologyViolation {
                    kind: ViolationKind::ZeroCostCycle,
                    node: names.first().copied().unwrap_or("?").to_string(),
                    detail: format!(
                        "SCC {:?}: every internal edge has cost=0; semiring closure \
                         prefers this infinite loop over productive paths",
                        names
                    ),
                });
            }
        }
    }
}

/// Verify that every member of `RECEIPT_CUTSET` dominates `commit_id` when
/// the graph is restricted to the subgraph between `exec_fin_id` and
/// `commit_id`.
fn check_receipt_cutset(
    violations: &mut Vec<TopologyViolation>,
    node_ids: &HashMap<&str, usize>,
    forward: &HashMap<usize, Vec<usize>>,
    reverse: &HashMap<usize, Vec<usize>>,
    exec_fin_id: usize,
    commit_id: usize,
) {
    // Restrict to nodes reachable from exec_fin AND that can reach commit
    let from_exec = bfs(forward, exec_fin_id);
    if !from_exec.contains(&commit_id) {
        return; // no path exec_finished → commit — skip
    }
    let to_commit = bfs(reverse, commit_id);
    let sub_nodes: HashSet<usize> = from_exec.intersection(&to_commit).copied().collect();

    // Build restricted adjacency
    let mut sub_fwd: HashMap<usize, Vec<usize>> =
        sub_nodes.iter().map(|&n| (n, Vec::new())).collect();
    let mut sub_rev: HashMap<usize, Vec<usize>> =
        sub_nodes.iter().map(|&n| (n, Vec::new())).collect();
    for (&src, dsts) in forward {
        if !sub_nodes.contains(&src) {
            continue;
        }
        for &dst in dsts {
            if sub_nodes.contains(&dst) {
                sub_fwd.entry(src).or_default().push(dst);
                sub_rev.entry(dst).or_default().push(src);
            }
        }
    }

    // Dominator tree within the subgraph, starting from exec_fin_id
    let sub_all: Vec<usize> = sub_nodes.iter().copied().collect();
    let sub_dom = compute_dominators(&sub_rev, &sub_all, exec_fin_id);
    let commit_dom = match sub_dom.get(&commit_id) {
        Some(d) => d,
        None => return,
    };

    for &member in RECEIPT_CUTSET {
        let Some(&member_id) = node_ids.get(member) else {
            continue;
        };
        if !sub_nodes.contains(&member_id) {
            violations.push(TopologyViolation {
                kind: ViolationKind::ReceiptCutsetViolation,
                node: member.to_string(),
                detail: format!(
                    "cutset member '{}' is not on any path from \
                     'Event::ExecuteFinished' to 'Control::Commit'; \
                     receipt validation can be bypassed",
                    member
                ),
            });
            continue;
        }
        if !commit_dom.contains(&member_id) {
            violations.push(TopologyViolation {
                kind: ViolationKind::ReceiptCutsetViolation,
                node: member.to_string(),
                detail: format!(
                    "cutset member '{}' does not dominate 'Control::Commit' \
                     on paths from 'Event::ExecuteFinished'",
                    member
                ),
            });
        }
    }
}

// ── Graph algorithms ──────────────────────────────────────────────────────────

fn bfs(adj: &HashMap<usize, Vec<usize>>, start: usize) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        for &next in adj.get(&node).into_iter().flatten() {
            if visited.insert(next) {
                queue.push_back(next);
            }
        }
    }
    visited
}

fn bfs_multi(adj: &HashMap<usize, Vec<usize>>, sources: &HashSet<usize>) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for &s in sources {
        if visited.insert(s) {
            queue.push_back(s);
        }
    }
    while let Some(node) = queue.pop_front() {
        for &next in adj.get(&node).into_iter().flatten() {
            if visited.insert(next) {
                queue.push_back(next);
            }
        }
    }
    visited
}

/// Iterative dataflow dominator computation (O(n²) fixpoint).
///
/// `reverse` is the reverse adjacency.  `start` is the entry node.
/// Returns dom(v) = set of nodes that every path from `start` to `v` passes
/// through.
fn compute_dominators(
    reverse: &HashMap<usize, Vec<usize>>,
    all_nodes: &[usize],
    start: usize,
) -> HashMap<usize, HashSet<usize>> {
    let all_set: HashSet<usize> = all_nodes.iter().copied().collect();
    let mut dom: HashMap<usize, HashSet<usize>> = HashMap::new();
    dom.insert(start, std::iter::once(start).collect());
    for &nid in all_nodes {
        if nid != start {
            dom.insert(nid, all_set.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &v in all_nodes {
            if v == start {
                continue;
            }
            let preds: Vec<usize> = reverse
                .get(&v)
                .into_iter()
                .flatten()
                .copied()
                .collect();
            if preds.is_empty() {
                continue;
            }
            let new_dom = {
                let mut it = preds.iter().filter_map(|p| dom.get(p));
                let first = match it.next() {
                    Some(s) => s.clone(),
                    None => continue,
                };
                let mut inter: HashSet<usize> = it.fold(first, |acc, pd| {
                    acc.intersection(pd).copied().collect()
                });
                inter.insert(v);
                inter
            };
            if &new_dom != dom.get(&v).unwrap() {
                dom.insert(v, new_dom);
                changed = true;
            }
        }
    }
    dom
}

/// Kosaraju's SCC algorithm.  Returns a list of SCCs (each is a `Vec` of node
/// IDs).
fn kosaraju_sccs(
    forward: &HashMap<usize, Vec<usize>>,
    reverse: &HashMap<usize, Vec<usize>>,
    all_nodes: &[usize],
) -> Vec<Vec<usize>> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut finish: Vec<usize> = Vec::new();
    for &start in all_nodes {
        if !visited.contains(&start) {
            dfs_finish(forward, start, &mut visited, &mut finish);
        }
    }

    let mut visited2: HashSet<usize> = HashSet::new();
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    for &start in finish.iter().rev() {
        if !visited2.contains(&start) {
            let mut scc = Vec::new();
            dfs_collect(reverse, start, &mut visited2, &mut scc);
            sccs.push(scc);
        }
    }
    sccs
}

/// Iterative post-order DFS on `adj`.
fn dfs_finish(
    adj: &HashMap<usize, Vec<usize>>,
    start: usize,
    visited: &mut HashSet<usize>,
    finish: &mut Vec<usize>,
) {
    if visited.contains(&start) {
        return;
    }
    let empty: Vec<usize> = Vec::new();
    visited.insert(start);
    // Stack entries: (node, current_neighbor_index)
    let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
    while let Some((node, idx)) = stack.last_mut() {
        let node = *node;
        let neighbors = adj.get(&node).unwrap_or(&empty);
        if *idx < neighbors.len() {
            let next = neighbors[*idx];
            *idx += 1;
            if visited.insert(next) {
                stack.push((next, 0));
            }
        } else {
            stack.pop();
            finish.push(node);
        }
    }
}

/// Iterative DFS that collects all reachable nodes into `scc`.
fn dfs_collect(
    adj: &HashMap<usize, Vec<usize>>,
    start: usize,
    visited: &mut HashSet<usize>,
    scc: &mut Vec<usize>,
) {
    if !visited.insert(start) {
        return;
    }
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        scc.push(node);
        for &next in adj.get(&node).into_iter().flatten() {
            if visited.insert(next) {
                stack.push(next);
            }
        }
    }
}
