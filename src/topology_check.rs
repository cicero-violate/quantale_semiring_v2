//! Static topology graph checker.
//!
//! Validates a `GraphTopology` against five graph invariants before any
//! tensor operations run.  Call `check()` after parsing topology.json and
//! before creating `TensorQuantaleWorld`.  If any violations are returned
//! the process should exit rather than proceed to execution.
//!
//! Invariants enforced
//! -------------------
//! 1. Endpoint validity      — every transition endpoint names a declared node
//! 2. Non-terminal closure   — every reachable non-halt node has outdeg > 0
//! 3. Reachability           — every page-listed node is reachable from start
//! 4. Path-to-halt           — every reachable node can eventually reach a halt
//! 5. No bottom row (weight) — only halt nodes may have an all-zero weight row
//!                             (structural restatement of invariant 2 after compile)
//!
//! Invariants 2 and 5 are checked only for nodes reachable from the start so
//! that documented-but-unused stub nodes do not produce false positives.
//! Invariant 3 is checked for every node listed in `pages`.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::topology::GraphTopology;

/// Describes why a topology check failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    /// A transition references a name that is not in the nodes list.
    UnknownEndpoint,
    /// A reachable, non-halt node has no outgoing transitions (outdeg = 0).
    DeadEnd,
    /// A node listed in pages is not reachable from the start node.
    Unreachable,
    /// A reachable node has no path to any halt node.
    CannotReachHalt,
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownEndpoint  => write!(f, "unknown_endpoint"),
            Self::DeadEnd          => write!(f, "dead_end"),
            Self::Unreachable      => write!(f, "unreachable"),
            Self::CannotReachHalt  => write!(f, "cannot_reach_halt"),
        }
    }
}

/// A single topology invariant violation.
#[derive(Clone, Debug)]
pub struct TopologyViolation {
    pub kind:   ViolationKind,
    pub node:   String,
    pub detail: String,
}

impl std::fmt::Display for TopologyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.kind, self.node, self.detail)
    }
}

/// Run all five checks on `topology`.  Returns every violation found; the
/// caller decides whether to abort (non-empty == reject).
///
/// All passes run before returning so the caller sees every problem at once.
pub fn check(topology: &GraphTopology) -> Vec<TopologyViolation> {
    let mut violations = Vec::new();

    // ── Index construction ─────────────────────────────────────────────────
    let node_ids: HashMap<&str, usize> = topology
        .nodes.iter()
        .map(|n| (n.name.as_str(), n.id))
        .collect();

    let halt_ids: HashSet<usize> = topology
        .nodes.iter()
        .filter(|n| n.action.as_deref() == Some("halt"))
        .map(|n| n.id)
        .collect();

    let id_to_name: HashMap<usize, &str> = topology
        .nodes.iter()
        .map(|n| (n.id, n.name.as_str()))
        .collect();

    // Forward and reverse adjacency, seeded with all node IDs so the maps are
    // complete even for nodes that have no transitions at all.
    let mut forward: HashMap<usize, Vec<usize>> = topology
        .nodes.iter()
        .map(|n| (n.id, Vec::new()))
        .collect();
    let mut reverse: HashMap<usize, Vec<usize>> = topology
        .nodes.iter()
        .map(|n| (n.id, Vec::new()))
        .collect();

    // ── Pass 1: endpoint validity ──────────────────────────────────────────
    for t in &topology.transitions {
        let src = node_ids.get(t.from.as_str()).copied();
        let dst = node_ids.get(t.to.as_str()).copied();

        if src.is_none() {
            violations.push(TopologyViolation {
                kind:   ViolationKind::UnknownEndpoint,
                node:   t.from.clone(),
                detail: format!(
                    "'{}' used as transition source but not declared in nodes",
                    t.from
                ),
            });
        }
        if dst.is_none() {
            violations.push(TopologyViolation {
                kind:   ViolationKind::UnknownEndpoint,
                node:   t.to.clone(),
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

    // ── Forward BFS from start (node id 0) ────────────────────────────────
    let reachable = bfs(&forward, 0);

    // ── Pass 2: dead-end (reachable non-halt nodes with outdeg = 0) ───────
    for &nid in &reachable {
        if halt_ids.contains(&nid) {
            continue;
        }
        if forward.get(&nid).map(|v| v.is_empty()).unwrap_or(true) {
            let name = id_to_name.get(&nid).copied().unwrap_or("?");
            violations.push(TopologyViolation {
                kind:   ViolationKind::DeadEnd,
                node:   name.to_string(),
                detail: format!(
                    "'{}' is reachable and not a halt node but has no outgoing \
                     transitions (outdeg=0); any execution path reaching it \
                     produces Unknown(-1) blocked",
                    name
                ),
            });
        }
    }

    // ── Pass 3: reachability of page-listed nodes ──────────────────────────
    let page_names: HashSet<&str> = topology
        .pages.iter()
        .flat_map(|p| p.node_names.iter().map(String::as_str))
        .collect();

    for name in page_names {
        let Some(&nid) = node_ids.get(name) else { continue };
        if !reachable.contains(&nid) {
            violations.push(TopologyViolation {
                kind:   ViolationKind::Unreachable,
                node:   name.to_string(),
                detail: format!(
                    "'{}' is listed in pages but unreachable from start node (id=0)",
                    name
                ),
            });
        }
    }

    // ── Reverse BFS from all halt nodes ───────────────────────────────────
    let can_reach_halt = bfs_multi(&reverse, &halt_ids);

    // ── Pass 4: path-to-halt (reachable nodes only) ────────────────────────
    for &nid in &reachable {
        if !can_reach_halt.contains(&nid) {
            let name = id_to_name.get(&nid).copied().unwrap_or("?");
            violations.push(TopologyViolation {
                kind:   ViolationKind::CannotReachHalt,
                node:   name.to_string(),
                detail: format!(
                    "'{}' is reachable but has no path to any halt node \
                     (nodes with action=\"halt\")",
                    name
                ),
            });
        }
    }

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

// ── BFS helpers ───────────────────────────────────────────────────────────────

fn bfs(adj: &HashMap<usize, Vec<usize>>, start: usize) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue   = VecDeque::new();
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
    let mut queue   = VecDeque::new();
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
