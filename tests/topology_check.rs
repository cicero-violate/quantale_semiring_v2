use quantale_semiring_v2::{
    GraphTopology, TENSOR_NODE_COUNT, TopologyInvariants, ViolationKind, check,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn parse(json: &str) -> GraphTopology {
    GraphTopology::from_json_str(json).expect("parse topology")
}

fn violations_of_kind(topo: &GraphTopology, kind: ViolationKind) -> Vec<String> {
    check(topo, &TopologyInvariants::default())
        .into_iter()
        .filter(|v| v.kind == kind)
        .map(|v| v.node)
        .collect()
}

fn has_violation(topo: &GraphTopology, kind: ViolationKind, node: &str) -> bool {
    check(topo, &TopologyInvariants::default())
        .iter()
        .any(|v| v.kind == kind && v.node == node)
}

// ── minimal valid topology used in multiple tests ────────────────────────────

fn minimal_valid() -> &'static str {
    r#"{
        "matrix_name": "test",
        "nodes": [
            {"id": 0, "name": "State::Start", "type": "State"},
            {"id": 1, "name": "State::Work",  "type": "State"},
            {"id": 2, "name": "Control::Halt","type": "Control","action":"halt"}
        ],
        "transitions": [
            {"from": "State::Start", "to": "State::Work",   "default_weight": 0.9},
            {"from": "State::Work",  "to": "Control::Halt", "default_weight": 0.8}
        ],
        "pages": [{"name": "main", "node_names": ["State::Start","State::Work","Control::Halt"]}]
    }"#
}

// ── 1. valid topology passes ──────────────────────────────────────────────────

#[test]
fn check_passes_on_valid_topology() {
    let topo = parse(minimal_valid());
    let vs = check(&topo, &TopologyInvariants::default());
    assert!(vs.is_empty(), "expected no violations, got: {vs:?}");
}

// ── 2. endpoint validity ──────────────────────────────────────────────────────

#[test]
fn check_rejects_unknown_source_endpoint() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Ghost","to":"Control::Halt","default_weight":0.9},
            {"from":"State::A","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::UnknownEndpoint,
        "State::Ghost"
    ));
}

#[test]
fn check_rejects_unknown_destination_endpoint() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::A","to":"State::Missing","default_weight":0.9},
            {"from":"State::A","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::UnknownEndpoint,
        "State::Missing"
    ));
}

// ── 3. non-terminal closure (dead-end) ───────────────────────────────────────

#[test]
fn check_rejects_dead_end_non_halt_node() {
    // State::Dead is reachable from start but has no outgoing transitions.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::Dead", "type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::Dead",  "default_weight":0.9},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    assert!(
        has_violation(&topo, ViolationKind::DeadEnd, "State::Dead"),
        "State::Dead should be flagged as a dead end"
    );
}

#[test]
fn halt_node_is_not_flagged_as_dead_end() {
    let topo = parse(minimal_valid());
    let dead_ends = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(
        dead_ends.is_empty(),
        "halt node must not be a dead-end violation"
    );
}

#[test]
fn unreachable_dead_end_is_not_flagged() {
    // State::Orphan has no outgoing edges AND is unreachable — should not be
    // reported as DeadEnd (it cannot cause a runtime failure).
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start", "type":"State"},
            {"id":1,"name":"State::Orphan","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    let dead_ends = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(
        !dead_ends.contains(&"State::Orphan".to_string()),
        "unreachable node must not be reported as a dead end"
    );
}

// ── 4. reachability ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_unreachable_page_node() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start",     "type":"State"},
            {"id":1,"name":"State::Unreachable","type":"State"},
            {"id":2,"name":"Control::Halt",    "type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start",     "to":"Control::Halt","default_weight":0.9},
            {"from":"State::Unreachable","to":"Control::Halt","default_weight":0.8}
        ],
        "pages":[{"name":"main","node_names":["State::Start","State::Unreachable","Control::Halt"]}]
    }"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::Unreachable,
        "State::Unreachable"
    ));
}

#[test]
fn node_not_in_pages_is_not_flagged_unreachable() {
    // An orphan node that is NOT in pages should not produce an Unreachable violation.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start", "type":"State"},
            {"id":1,"name":"State::Orphan","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9}
        ],
        "pages":[{"name":"main","node_names":["State::Start","Control::Halt"]}]
    }"#,
    );
    let unreachable = violations_of_kind(&topo, ViolationKind::Unreachable);
    assert!(!unreachable.contains(&"State::Orphan".to_string()));
}

// ── 5. path-to-halt ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_node_with_no_path_to_halt() {
    // State::Loop only connects back to itself — no path to Control::Halt.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::Loop", "type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::Loop", "default_weight":0.9},
            {"from":"State::Loop", "to":"State::Loop", "default_weight":0.7},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::CannotReachHalt,
        "State::Loop"
    ));
}

#[test]
fn indirect_path_to_halt_is_accepted() {
    // State::A -> State::B -> Control::Halt (two hops, should pass).
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"State::B","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::A","to":"State::B",      "default_weight":0.9},
            {"from":"State::B","to":"Control::Halt", "default_weight":0.8}
        ],"pages":[]}"#,
    );
    assert!(check(&topo, &TopologyInvariants::default()).is_empty());
}

// ── 6. all violations returned at once ───────────────────────────────────────

#[test]
fn check_reports_all_violations_not_just_first() {
    // Three independent dead-end nodes — all three must be reported.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::DeadA","type":"State"},
            {"id":2,"name":"State::DeadB","type":"State"},
            {"id":3,"name":"State::DeadC","type":"State"},
            {"id":4,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::DeadA","default_weight":0.9},
            {"from":"State::Start","to":"State::DeadB","default_weight":0.8},
            {"from":"State::Start","to":"State::DeadC","default_weight":0.7},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    let dead = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(dead.contains(&"State::DeadA".to_string()));
    assert!(dead.contains(&"State::DeadB".to_string()));
    assert!(dead.contains(&"State::DeadC".to_string()));
    assert_eq!(dead.len(), 3);
}

// ── 7. regression guard: current bundled topology must be clean ───────────────

// Known ConsumedBlockPoint nodes in the current topology.  These are tracked
// here so new violations are caught immediately; fix each node by adding a
// second outgoing edge before removing it from this list.
const KNOWN_CONSUMED_BLOCK_POINTS: &[&str] = &[
    "Analysis::Return1", // parallel_prepare par + market_analysis_cycle adds AnalysisPlan→Return1
    "Control::Block",
    "Control::BuildTopologyOverlay",
    "Control::Repair",
    "Event::AnalysisFinished",
    "Execution::VectorScale", // hot VectorAdd→VectorScale chain adds VectorAdd as an alternate predecessor
    "State::Input",
    "State::Score", // parallel_prepare adds Parse→Score (was only CandidateFound→Score)
    "State::Search", // parallel_prepare adds Map→Search (was only MapReady→Search)
];

#[test]
fn current_topology_passes_all_checks() {
    let topo = GraphTopology::default_asset().expect("generated runtime topology");
    let violations = check(&topo, &TopologyInvariants::default());

    let unexpected: Vec<_> = violations
        .iter()
        .filter(|v| {
            v.kind != ViolationKind::ConsumedBlockPoint
                || !KNOWN_CONSUMED_BLOCK_POINTS.contains(&v.node.as_str())
        })
        .collect();

    if !unexpected.is_empty() {
        for v in &unexpected {
            eprintln!("[topology_check] {v}");
        }
    }
    assert!(
        unexpected.is_empty(),
        "generated runtime topology has {} unexpected violation(s); fix before committing",
        unexpected.len()
    );

    // Also assert that no NEW ConsumedBlockPoint nodes appeared beyond the known list.
    let cbp_nodes: Vec<&str> = violations
        .iter()
        .filter(|v| v.kind == ViolationKind::ConsumedBlockPoint)
        .map(|v| v.node.as_str())
        .collect();
    for node in &cbp_nodes {
        assert!(
            KNOWN_CONSUMED_BLOCK_POINTS.contains(node),
            "new ConsumedBlockPoint '{}' — add a second outgoing edge or add to known list",
            node
        );
    }
}

#[test]
fn current_topology_fits_tensor_universe() {
    let topo = GraphTopology::default_asset().expect("generated runtime topology");
    let compiled = topo.compile().expect("compile topology");
    assert!(
        compiled.node_count <= TENSOR_NODE_COUNT,
        "topology has {} nodes but generated tensor capacity is {}; rebuild after updating topology assets",
        compiled.node_count,
        TENSOR_NODE_COUNT
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 — identity and weight tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── duplicate node name ───────────────────────────────────────────────────────

#[test]
fn check_rejects_duplicate_node_names() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"State::A","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::A","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::DuplicateNodeName,
        "State::A"
    ));
}

#[test]
fn check_passes_with_unique_node_names() {
    let topo = parse(minimal_valid());
    let dups = violations_of_kind(&topo, ViolationKind::DuplicateNodeName);
    assert!(dups.is_empty(), "no duplicate names expected: {dups:?}");
}

// ── duplicate node id ─────────────────────────────────────────────────────────

#[test]
fn check_rejects_duplicate_node_ids() {
    // Both nodes use id=1; serde happily parses this, check() must catch it.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A","type":"State"},
            {"id":1,"name":"State::B","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A","default_weight":0.9},
            {"from":"State::A","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#,
    );
    let dup_ids = violations_of_kind(&topo, ViolationKind::DuplicateNodeId);
    assert!(!dup_ids.is_empty(), "duplicate id=1 must be reported");
}

// ── start node validity ───────────────────────────────────────────────────────

#[test]
fn check_rejects_start_node_with_wrong_id() {
    // nodes[0] has id=1 instead of id=0
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":1,"name":"State::Start","type":"State"},
            {"id":0,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::InvalidStartNode,
        "State::Start"
    ));
}

#[test]
fn check_passes_when_start_node_has_id_zero() {
    let topo = parse(minimal_valid());
    let vs = violations_of_kind(&topo, ViolationKind::InvalidStartNode);
    assert!(vs.is_empty());
}

// ── halt node validity ────────────────────────────────────────────────────────

#[test]
fn check_rejects_topology_with_no_halt_node() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::End","type":"State"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::End","default_weight":0.9},
            {"from":"State::End","to":"State::Start","default_weight":0.8}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::NoHaltNode,
        "(topology)"
    ));
}

#[test]
fn check_rejects_halt_node_with_outgoing_edge() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9},
            {"from":"Control::Halt","to":"State::Start","default_weight":0.5}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::HaltNodeHasSuccessors,
        "Control::Halt"
    ));
}

#[test]
fn check_passes_when_halt_has_no_successors() {
    let topo = parse(minimal_valid());
    let vs = violations_of_kind(&topo, ViolationKind::HaltNodeHasSuccessors);
    assert!(vs.is_empty());
}

// ── edge uniqueness ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_duplicate_edge() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::DuplicateEdge,
        "State::Start"
    ));
}

#[test]
fn check_passes_with_unique_edges() {
    let topo = parse(minimal_valid());
    let vs = violations_of_kind(&topo, ViolationKind::DuplicateEdge);
    assert!(vs.is_empty());
}

// ── weight domain validity ────────────────────────────────────────────────────

#[test]
fn check_rejects_confidence_above_one() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9,
             "confidence":1.5}
        ],"pages":[]}"#,
    );
    assert!(
        has_violation(&topo, ViolationKind::WeightOutOfDomain, "State::Start"),
        "confidence=1.5 must be rejected"
    );
}

#[test]
fn check_rejects_negative_cost() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9,
             "cost":-0.1}
        ],"pages":[]}"#,
    );
    assert!(
        has_violation(&topo, ViolationKind::WeightOutOfDomain, "State::Start"),
        "cost=-0.1 must be rejected"
    );
}

#[test]
fn check_passes_with_valid_weights() {
    let topo = parse(minimal_valid());
    let vs = violations_of_kind(&topo, ViolationKind::WeightOutOfDomain);
    assert!(vs.is_empty());
}

// ── zero-confidence edge warning ──────────────────────────────────────────────

#[test]
fn check_warns_on_zero_confidence_edge() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.0,
             "confidence":0.0}
        ],"pages":[]}"#,
    );
    // DuplicateEdge is also expected here; just confirm ZeroConfidenceEdge fires
    assert!(has_violation(
        &topo,
        ViolationKind::ZeroConfidenceEdge,
        "State::Start"
    ));
}

// ── deterministic ordering ────────────────────────────────────────────────────

#[test]
fn check_rejects_indeterminate_ordering() {
    // Two edges from State::Start to the same destination with identical
    // (default_weight, safety, cost, to) tuple — the full sort key is
    // identical so tie-break is impossible.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A","default_weight":0.8,
             "safety":0.8,"cost":0.2},
            {"from":"State::Start","to":"State::A","default_weight":0.8,
             "safety":0.8,"cost":0.2},
            {"from":"State::A","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    assert!(has_violation(
        &topo,
        ViolationKind::IndeterminateOrdering,
        "State::Start"
    ));
}

#[test]
fn check_passes_with_distinct_edge_tuples() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::B",    "type":"State"},
            {"id":3,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A","default_weight":0.9},
            {"from":"State::Start","to":"State::B","default_weight":0.7},
            {"from":"State::A","to":"Control::Halt","default_weight":0.9},
            {"from":"State::B","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::IndeterminateOrdering);
    assert!(
        vs.is_empty(),
        "distinct weights must not trigger IndeterminateOrdering"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 3 — dominator and cycle tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── dominator: bypass edge ────────────────────────────────────────────────────

/// Topology where State::Validate → Control::Commit exists, but also a bypass
/// edge State::Start → Control::Commit that skips the gate.
fn bypass_topology() -> &'static str {
    r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start",   "type":"State"},
            {"id":1,"name":"State::Validate","type":"State"},
            {"id":2,"name":"Control::Commit","type":"Control"},
            {"id":3,"name":"State::Memory",  "type":"State"},
            {"id":4,"name":"Control::Halt",  "type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start",   "to":"State::Validate","default_weight":0.9},
            {"from":"State::Start",   "to":"Control::Commit","default_weight":0.5},
            {"from":"State::Validate","to":"Control::Commit","default_weight":0.8},
            {"from":"Control::Commit","to":"State::Memory",  "default_weight":0.8},
            {"from":"State::Memory",  "to":"Control::Halt",  "default_weight":0.9}
        ],"pages":[]}"#
}

#[test]
fn check_rejects_bypass_of_required_gate() {
    let topo = parse(bypass_topology());
    assert!(
        has_violation(&topo, ViolationKind::DominanceViolation, "Control::Commit"),
        "bypass edge State::Start → Control::Commit must trigger DominanceViolation"
    );
}

#[test]
fn check_passes_when_gate_properly_dominates() {
    // Remove the bypass edge — now every path to Commit goes through Validate
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start",   "type":"State"},
            {"id":1,"name":"State::Validate","type":"State"},
            {"id":2,"name":"Control::Commit","type":"Control"},
            {"id":3,"name":"State::Memory",  "type":"State"},
            {"id":4,"name":"Control::Halt",  "type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start",   "to":"State::Validate","default_weight":0.9},
            {"from":"State::Validate","to":"Control::Commit","default_weight":0.8},
            {"from":"Control::Commit","to":"State::Memory",  "default_weight":0.8},
            {"from":"State::Memory",  "to":"Control::Halt",  "default_weight":0.9}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::DominanceViolation);
    assert!(
        vs.is_empty(),
        "clean chain must not trigger DominanceViolation: {vs:?}"
    );
}

// ── SCC progress ──────────────────────────────────────────────────────────────

#[test]
fn check_rejects_scc_with_no_exit_edge() {
    // State::A ↔ State::B form a cycle with no exit
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::B",    "type":"State"},
            {"id":3,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A",    "default_weight":0.9},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5},
            {"from":"State::A",    "to":"State::B",    "default_weight":0.8},
            {"from":"State::B",    "to":"State::A",    "default_weight":0.8}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::UnsafeSCC);
    assert!(
        !vs.is_empty(),
        "trapped cycle must be reported as UnsafeSCC"
    );
}

#[test]
fn check_passes_scc_with_exit_edge() {
    // State::A ↔ State::B cycle but with an exit to Control::Halt
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::B",    "type":"State"},
            {"id":3,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A",    "default_weight":0.9},
            {"from":"State::A",    "to":"State::B",    "default_weight":0.8},
            {"from":"State::B",    "to":"State::A",    "default_weight":0.8},
            {"from":"State::A",    "to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::UnsafeSCC);
    assert!(vs.is_empty(), "SCC with exit must not be flagged: {vs:?}");
}

// ── zero-cost cycle ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_zero_cost_infinite_cycle() {
    // A ↔ B with cost=0 on both edges — planner prefers this forever
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::B",    "type":"State"},
            {"id":3,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A",    "default_weight":0.9},
            {"from":"State::A",    "to":"State::B",    "default_weight":0.8,"cost":0.0},
            {"from":"State::B",    "to":"State::A",    "default_weight":0.8,"cost":0.0},
            {"from":"State::A",    "to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::ZeroCostCycle);
    assert!(
        !vs.is_empty(),
        "zero-cost cycle must be reported as ZeroCostCycle"
    );
}

#[test]
fn check_passes_cycle_with_nonzero_cost() {
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::B",    "type":"State"},
            {"id":3,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A",    "default_weight":0.9},
            {"from":"State::A",    "to":"State::B",    "default_weight":0.8,"cost":1.0},
            {"from":"State::B",    "to":"State::A",    "default_weight":0.8,"cost":1.0},
            {"from":"State::A",    "to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::ZeroCostCycle);
    assert!(
        vs.is_empty(),
        "non-zero-cost cycle must not be flagged: {vs:?}"
    );
}

// ── 14. Consumed block point ──────────────────────────────────────────────────

#[test]
fn check_flags_consumed_block_point() {
    // State::Funnel has 1 outgoing edge (→State::B) but 2 incoming edges.
    // After the first traversal consumes State::Funnel→State::B, re-entry
    // from State::A produces Unknown(-1) blocked.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start", "type":"State"},
            {"id":1,"name":"State::A",     "type":"State"},
            {"id":2,"name":"State::Funnel","type":"State"},
            {"id":3,"name":"State::B",     "type":"State"},
            {"id":4,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start", "to":"State::A",     "default_weight":0.9},
            {"from":"State::Start", "to":"State::Funnel","default_weight":0.8},
            {"from":"State::A",     "to":"State::Funnel","default_weight":0.7},
            {"from":"State::Funnel","to":"State::B",     "default_weight":0.9},
            {"from":"State::B",     "to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#,
    );
    assert!(
        has_violation(&topo, ViolationKind::ConsumedBlockPoint, "State::Funnel"),
        "single-exit multi-entry node must be flagged as ConsumedBlockPoint"
    );
}

#[test]
fn check_passes_multi_exit_node() {
    // State::Hub has 2 outgoing edges — not a block point even with multiple
    // predecessors.
    let topo = parse(
        r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::A",    "type":"State"},
            {"id":2,"name":"State::Hub",  "type":"State"},
            {"id":3,"name":"State::B",    "type":"State"},
            {"id":4,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::A",      "default_weight":0.9},
            {"from":"State::Start","to":"State::Hub",    "default_weight":0.8},
            {"from":"State::A",    "to":"State::Hub",    "default_weight":0.7},
            {"from":"State::Hub",  "to":"State::B",      "default_weight":0.9},
            {"from":"State::Hub",  "to":"Control::Halt", "default_weight":0.5},
            {"from":"State::B",    "to":"Control::Halt", "default_weight":0.9}
        ],"pages":[]}"#,
    );
    let vs = violations_of_kind(&topo, ViolationKind::ConsumedBlockPoint);
    assert!(vs.is_empty(), "multi-exit node must not be flagged: {vs:?}");
}
