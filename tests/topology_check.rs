use quantale_semiring_v2::{GraphTopology, topology_check::{self, ViolationKind}};

// ── helpers ──────────────────────────────────────────────────────────────────

fn parse(json: &str) -> GraphTopology {
    GraphTopology::from_json_str(json).expect("parse topology")
}

fn violations_of_kind(topo: &GraphTopology, kind: ViolationKind) -> Vec<String> {
    topology_check::check(topo)
        .into_iter()
        .filter(|v| v.kind == kind)
        .map(|v| v.node)
        .collect()
}

fn has_violation(topo: &GraphTopology, kind: ViolationKind, node: &str) -> bool {
    topology_check::check(topo)
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
    let vs = topology_check::check(&topo);
    assert!(vs.is_empty(), "expected no violations, got: {vs:?}");
}

// ── 2. endpoint validity ──────────────────────────────────────────────────────

#[test]
fn check_rejects_unknown_source_endpoint() {
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Ghost","to":"Control::Halt","default_weight":0.9},
            {"from":"State::A","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#);
    assert!(has_violation(&topo, ViolationKind::UnknownEndpoint, "State::Ghost"));
}

#[test]
fn check_rejects_unknown_destination_endpoint() {
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::A","to":"State::Missing","default_weight":0.9},
            {"from":"State::A","to":"Control::Halt","default_weight":0.8}
        ],"pages":[]}"#);
    assert!(has_violation(&topo, ViolationKind::UnknownEndpoint, "State::Missing"));
}

// ── 3. non-terminal closure (dead-end) ───────────────────────────────────────

#[test]
fn check_rejects_dead_end_non_halt_node() {
    // State::Dead is reachable from start but has no outgoing transitions.
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::Dead", "type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::Dead",  "default_weight":0.9},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#);
    assert!(has_violation(&topo, ViolationKind::DeadEnd, "State::Dead"),
        "State::Dead should be flagged as a dead end");
}

#[test]
fn halt_node_is_not_flagged_as_dead_end() {
    let topo = parse(minimal_valid());
    let dead_ends = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(dead_ends.is_empty(), "halt node must not be a dead-end violation");
}

#[test]
fn unreachable_dead_end_is_not_flagged() {
    // State::Orphan has no outgoing edges AND is unreachable — should not be
    // reported as DeadEnd (it cannot cause a runtime failure).
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start", "type":"State"},
            {"id":1,"name":"State::Orphan","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9}
        ],"pages":[]}"#);
    let dead_ends = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(!dead_ends.contains(&"State::Orphan".to_string()),
        "unreachable node must not be reported as a dead end");
}

// ── 4. reachability ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_unreachable_page_node() {
    let topo = parse(r#"{
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
    }"#);
    assert!(has_violation(&topo, ViolationKind::Unreachable, "State::Unreachable"));
}

#[test]
fn node_not_in_pages_is_not_flagged_unreachable() {
    // An orphan node that is NOT in pages should not produce an Unreachable violation.
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start", "type":"State"},
            {"id":1,"name":"State::Orphan","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"Control::Halt","default_weight":0.9}
        ],
        "pages":[{"name":"main","node_names":["State::Start","Control::Halt"]}]
    }"#);
    let unreachable = violations_of_kind(&topo, ViolationKind::Unreachable);
    assert!(!unreachable.contains(&"State::Orphan".to_string()));
}

// ── 5. path-to-halt ───────────────────────────────────────────────────────────

#[test]
fn check_rejects_node_with_no_path_to_halt() {
    // State::Loop only connects back to itself — no path to Control::Halt.
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::Start","type":"State"},
            {"id":1,"name":"State::Loop", "type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::Start","to":"State::Loop", "default_weight":0.9},
            {"from":"State::Loop", "to":"State::Loop", "default_weight":0.7},
            {"from":"State::Start","to":"Control::Halt","default_weight":0.5}
        ],"pages":[]}"#);
    assert!(has_violation(&topo, ViolationKind::CannotReachHalt, "State::Loop"));
}

#[test]
fn indirect_path_to_halt_is_accepted() {
    // State::A -> State::B -> Control::Halt (two hops, should pass).
    let topo = parse(r#"{
        "matrix_name":"t","nodes":[
            {"id":0,"name":"State::A","type":"State"},
            {"id":1,"name":"State::B","type":"State"},
            {"id":2,"name":"Control::Halt","type":"Control","action":"halt"}
        ],
        "transitions":[
            {"from":"State::A","to":"State::B",      "default_weight":0.9},
            {"from":"State::B","to":"Control::Halt", "default_weight":0.8}
        ],"pages":[]}"#);
    assert!(topology_check::check(&topo).is_empty());
}

// ── 6. all violations returned at once ───────────────────────────────────────

#[test]
fn check_reports_all_violations_not_just_first() {
    // Three independent dead-end nodes — all three must be reported.
    let topo = parse(r#"{
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
        ],"pages":[]}"#);
    let dead = violations_of_kind(&topo, ViolationKind::DeadEnd);
    assert!(dead.contains(&"State::DeadA".to_string()));
    assert!(dead.contains(&"State::DeadB".to_string()));
    assert!(dead.contains(&"State::DeadC".to_string()));
    assert_eq!(dead.len(), 3);
}

// ── 7. regression guard: current bundled topology must be clean ───────────────

#[test]
fn current_topology_passes_all_checks() {
    let topo = GraphTopology::default_asset().expect("bundled topology");
    let violations = topology_check::check(&topo);
    if !violations.is_empty() {
        for v in &violations {
            eprintln!("[topology_check] {v}");
        }
    }
    assert!(
        violations.is_empty(),
        "bundled topology.json has {} violation(s); fix before committing",
        violations.len()
    );
}
