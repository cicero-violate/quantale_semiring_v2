//! Source topology program compiler.
//!
//! Reads `topology.source.json` `programs` and compiles CKA algebraic
//! expressions into flat transition objects (same format as
//! `topology.generated.json` transitions) and parallel group metadata.
//!
//! Called by `build_overlay_assets` to merge program-derived edges with the
//! existing flat transition baseline from `topology.json`.
//!
//! Algebraic forms supported:
//!   seq(A, B, C)      — sequential composition: emit edges A→B, B→C
//!   choice(A, B)      — quantale join: union endpoints, no cross-edges
//!   par(A, B)         — concurrent independent composition + parallel group
//!   star(body, n)     — bounded unroll: repeat body n times
//!   zero/blocked      — bottom: no endpoints emitted
//!   one/skip/identity — identity: no endpoints emitted

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

// ── Operator effects types ────────────────────────────────────────────────────

/// Declared side effects for a single node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeEffects {
    pub reads: BTreeSet<String>,
    pub writes: BTreeSet<String>,
    pub locks: BTreeSet<String>,
}

/// Map from node name → declared effects.  Used for par independence checks.
pub type EffectsMap = BTreeMap<String, NodeEffects>;

/// Build an `EffectsMap` from the `operators` array in `operators.json` plus
/// any node declarations in `topology.source.json`.
///
/// Source-topology node declarations take precedence over operators.json when
/// both exist for the same node name.
pub fn build_effects_map(operator_contracts: &[Value], source_nodes: &[Value]) -> EffectsMap {
    let mut map = EffectsMap::new();

    for op in operator_contracts {
        let Some(name) = op.get("node_name").and_then(Value::as_str) else {
            continue;
        };
        if let Some(effects) = op.get("effects") {
            map.insert(
                name.to_string(),
                NodeEffects {
                    reads: str_set(effects.get("reads")),
                    writes: str_set(effects.get("writes")),
                    locks: str_set(effects.get("locks")),
                },
            );
        }
    }

    for node in source_nodes {
        let Some(name) = node.get("name").and_then(Value::as_str) else {
            continue;
        };
        let reads = str_set(node.get("reads"));
        let writes = str_set(node.get("writes"));
        let locks = str_set(node.get("locks"));
        if !reads.is_empty() || !writes.is_empty() || !locks.is_empty() {
            map.insert(name.to_string(), NodeEffects { reads, writes, locks });
        }
    }

    map
}

fn str_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

fn safe_parallel(a: &NodeEffects, b: &NodeEffects) -> bool {
    a.writes.is_disjoint(&b.writes)
        && a.writes.is_disjoint(&b.reads)
        && a.reads.is_disjoint(&b.writes)
        && a.locks.is_disjoint(&b.locks)
}

// ── Patterns compat emitter ───────────────────────────────────────────────────

/// Emit all programs from `topology.source.json` in the `patterns.json` format.
///
/// This generates a `{ "patterns": [...] }` value that can replace
/// `assets/patterns.json`, proving that topology.source.json is the source of
/// truth for CKA patterns.  Programs with `"kind": "cka_pattern"` are included
/// as-are; others are included too (they are structurally valid CKA patterns
/// even if they are also compiled to flat transitions).
pub fn emit_patterns_compat(source: &Value) -> Value {
    let programs = match source.get("programs").and_then(Value::as_array) {
        Some(p) => p,
        None => return json!({ "patterns": [] }),
    };

    let patterns: Vec<Value> = programs
        .iter()
        .filter_map(|prog| {
            let name = prog.get("name")?.as_str()?;
            let expr = prog.get("expr")?;
            let (confidence, cost, safety) = extract_weight(prog).ok()?;
            Some(json!({
                "name": name,
                "expr": expr,
                "confidence": confidence,
                "cost": cost,
                "safety": safety
            }))
        })
        .collect();

    json!({ "patterns": patterns })
}

// ── Program compiler ──────────────────────────────────────────────────────────

/// Compile programs from a `topology.source.json` value into flat transition
/// objects and parallel group node-name lists.
///
/// Transitions whose `(from, to)` pair already exists in `existing_edges` are
/// silently skipped — Phase-1 preserves all flat transitions from
/// `topology.json` and programs only extend the edge set with new paths.
///
/// Programs with `"kind": "cka_pattern"` are skipped: they are runtime tensor-
/// weight patterns only and must not contribute structural topology edges (e.g.
/// bounded_learn_loop creates a loop-back that bypasses a dominance gate).
///
/// When `effects` is `Some`, par branches are validated for effect independence.
/// When `None`, the check is skipped (useful for tests and migration tooling).
///
/// Unknown nodes referenced in expressions produce an error.
pub fn compile_source_programs(
    source: &Value,
    existing_edges: &BTreeSet<(String, String)>,
    known_node_names: &BTreeSet<String>,
    effects: Option<&EffectsMap>,
) -> Result<(Vec<Value>, Vec<Vec<String>>), String> {
    let programs = match source.get("programs") {
        None => return Ok((Vec::new(), Vec::new())),
        Some(Value::Array(programs)) => programs,
        Some(_) => return Err("topology.source.json 'programs' must be an array".to_string()),
    };

    let mut transitions: Vec<Value> = Vec::new();
    let mut all_parallel_groups: Vec<Vec<String>> = Vec::new();
    // Track edges added so far to avoid intra-source duplicates.
    let mut seen: BTreeSet<(String, String)> = existing_edges.clone();

    for program in programs {
        // cka_pattern programs are runtime tensor patterns only — skip them
        // here so they don't generate flat topology transitions.
        if program.get("kind").and_then(Value::as_str) == Some("cka_pattern") {
            continue;
        }

        let name = string_field(program, "name", "program")?;
        let (confidence, cost, safety) = extract_weight(program)?;

        let expr_value = program
            .get("expr")
            .ok_or_else(|| format!("program '{name}' missing 'expr'"))?;
        let parsed = parse_expr(expr_value)
            .map_err(|e| format!("program '{name}' expr parse error: {e}"))?;

        let mut prog_transitions: Vec<Value> = Vec::new();
        let mut prog_parallel_groups: Vec<Vec<String>> = Vec::new();
        compile_expr(
            &parsed,
            name,
            confidence,
            cost,
            safety,
            &mut prog_transitions,
            &mut prog_parallel_groups,
            known_node_names,
            effects,
        )?;

        for t in prog_transitions {
            let from = t["from"].as_str().unwrap_or_default().to_string();
            let to = t["to"].as_str().unwrap_or_default().to_string();
            let key = (from, to);
            if seen.insert(key) {
                transitions.push(t);
            }
        }
        all_parallel_groups.extend(prog_parallel_groups);
    }

    Ok((transitions, all_parallel_groups))
}

// ── Weight extraction ─────────────────────────────────────────────────────────

fn extract_weight(program: &Value) -> Result<(f64, f64, f64), String> {
    // New-style: weight object { confidence, cost, safety }
    if let Some(w) = program.get("weight").filter(|v| v.is_object()) {
        let confidence = f64_field(w, "confidence", "weight")?;
        let cost = f64_field(w, "cost", "weight")?;
        let safety = f64_field(w, "safety", "weight")?;
        return Ok((confidence, cost, safety));
    }
    // Compat-style: top-level confidence/cost/safety (patterns.json schema)
    let confidence = f64_field(program, "confidence", "program")?;
    let cost = f64_field(program, "cost", "program")?;
    let safety = f64_field(program, "safety", "program")?;
    Ok((confidence, cost, safety))
}

// ── CKA expression AST ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Expr {
    Zero,
    One,
    Node(String),
    Seq(Vec<Expr>),
    Choice(Vec<Expr>),
    Star { body: Box<Expr>, max_unroll: usize },
    Par(Vec<Expr>),
}

// ── Expression parser ─────────────────────────────────────────────────────────

fn parse_expr(value: &Value) -> Result<Expr, String> {
    match value {
        Value::String(s) => match s.as_str() {
            "zero" | "blocked" | "impossible" => Ok(Expr::Zero),
            "one" | "identity" | "skip" => Ok(Expr::One),
            node => Ok(Expr::Node(node.to_string())),
        },
        Value::Object(obj) => {
            if obj.contains_key("zero") || obj.contains_key("blocked") {
                return Ok(Expr::Zero);
            }
            if obj.contains_key("one") || obj.contains_key("skip") {
                return Ok(Expr::One);
            }
            if let Some(items) = obj.get("seq") {
                let items = parse_expr_array(items, "seq")?;
                if items.is_empty() {
                    return Err("'seq' requires at least one item".to_string());
                }
                return Ok(Expr::Seq(items));
            }
            if let Some(items) = obj.get("choice") {
                let items = parse_expr_array(items, "choice")?;
                if items.is_empty() {
                    return Err("'choice' requires at least one item".to_string());
                }
                return Ok(Expr::Choice(items));
            }
            if let Some(items) = obj.get("par") {
                let items = parse_expr_array(items, "par")?;
                if items.len() < 2 {
                    return Err("'par' requires at least two branches".to_string());
                }
                return Ok(Expr::Par(items));
            }
            if let Some(star) = obj.get("star") {
                let body = star
                    .get("body")
                    .ok_or_else(|| "'star' requires 'body'".to_string())?;
                let max_unroll = star
                    .get("max_unroll")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "'star' requires integer 'max_unroll'".to_string())?
                    as usize;
                if max_unroll == 0 {
                    return Err("star 'max_unroll' must be > 0".to_string());
                }
                return Ok(Expr::Star {
                    body: Box::new(parse_expr(body)?),
                    max_unroll,
                });
            }
            Err("unknown CKA expression object — expected seq/choice/par/star/zero/one".to_string())
        }
        _ => Err("CKA expression must be a string or object".to_string()),
    }
}

fn parse_expr_array(value: &Value, field: &str) -> Result<Vec<Expr>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("'{field}' must be an array"))?
        .iter()
        .map(parse_expr)
        .collect()
}

// ── Compilation: endpoint sets ────────────────────────────────────────────────

/// Tracks the frontier entry-points (starts) and exit-points (ends) of a
/// compiled sub-expression.  Used to connect adjacent sub-expressions in seq.
#[derive(Clone, Default, Debug)]
struct Endpoints {
    starts: Vec<String>,
    ends: Vec<String>,
}

impl Endpoints {
    fn from_node(name: String) -> Self {
        Self {
            starts: vec![name.clone()],
            ends: vec![name],
        }
    }

    fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

fn compile_expr(
    expr: &Expr,
    program_name: &str,
    confidence: f64,
    cost: f64,
    safety: f64,
    transitions: &mut Vec<Value>,
    parallel_groups: &mut Vec<Vec<String>>,
    known: &BTreeSet<String>,
    effects: Option<&EffectsMap>,
) -> Result<Endpoints, String> {
    match expr {
        Expr::Zero | Expr::One => Ok(Endpoints::default()),
        Expr::Node(name) => {
            if !known.contains(name.as_str()) {
                return Err(format!(
                    "program '{program_name}': unknown node '{name}'"
                ));
            }
            Ok(Endpoints::from_node(name.clone()))
        }
        Expr::Seq(items) => compile_seq(
            items,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
            effects,
        ),
        Expr::Choice(items) => {
            // Quantale join: each branch is compiled independently; starts and
            // ends are unioned.  No cross-edges are emitted between branches.
            let mut aggregate = Endpoints::default();
            for item in items {
                let ep = compile_expr(
                    item,
                    program_name,
                    confidence,
                    cost,
                    safety,
                    transitions,
                    parallel_groups,
                    known,
                    effects,
                )?;
                aggregate.starts.extend(ep.starts);
                aggregate.ends.extend(ep.ends);
            }
            Ok(aggregate)
        }
        Expr::Star { body, max_unroll } => compile_star(
            body,
            *max_unroll,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
            effects,
        ),
        Expr::Par(branches) => compile_par(
            branches,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
            effects,
        ),
    }
}

fn compile_seq(
    items: &[Expr],
    program_name: &str,
    confidence: f64,
    cost: f64,
    safety: f64,
    transitions: &mut Vec<Value>,
    parallel_groups: &mut Vec<Vec<String>>,
    known: &BTreeSet<String>,
    effects: Option<&EffectsMap>,
) -> Result<Endpoints, String> {
    let mut iter = items.iter();
    let Some(first) = iter.next() else {
        return Ok(Endpoints::default());
    };
    let mut aggregate = compile_expr(
        first, program_name, confidence, cost, safety,
        transitions, parallel_groups, known, effects,
    )?;
    let mut prev_ends = aggregate.ends.clone();

    for item in iter {
        let ep = compile_expr(
            item, program_name, confidence, cost, safety,
            transitions, parallel_groups, known, effects,
        )?;
        for from in &prev_ends {
            for to in &ep.starts {
                transitions.push(make_transition(from, to, confidence, cost, safety, program_name));
            }
        }
        if aggregate.starts.is_empty() {
            aggregate.starts = ep.starts.clone();
        }
        if !ep.ends.is_empty() {
            prev_ends = ep.ends.clone();
            aggregate.ends = ep.ends;
        }
    }
    Ok(aggregate)
}

fn compile_star(
    body: &Expr,
    max_unroll: usize,
    program_name: &str,
    confidence: f64,
    cost: f64,
    safety: f64,
    transitions: &mut Vec<Value>,
    parallel_groups: &mut Vec<Vec<String>>,
    known: &BTreeSet<String>,
    effects: Option<&EffectsMap>,
) -> Result<Endpoints, String> {
    let mut first = Endpoints::default();
    let mut prev_ends: Vec<String> = Vec::new();

    for idx in 0..max_unroll {
        let ep = compile_expr(
            body, program_name, confidence, cost, safety,
            transitions, parallel_groups, known, effects,
        )?;
        if idx == 0 {
            first = ep.clone();
        } else {
            for from in &prev_ends {
                for to in &ep.starts {
                    transitions.push(make_transition(from, to, confidence, cost, safety, program_name));
                }
            }
        }
        prev_ends = ep.ends;
    }

    if first.is_empty() {
        Ok(Endpoints::default())
    } else {
        Ok(Endpoints { starts: first.starts, ends: prev_ends })
    }
}

fn compile_par(
    branches: &[Expr],
    program_name: &str,
    confidence: f64,
    cost: f64,
    safety: f64,
    transitions: &mut Vec<Value>,
    parallel_groups: &mut Vec<Vec<String>>,
    known: &BTreeSet<String>,
    effects: Option<&EffectsMap>,
) -> Result<Endpoints, String> {
    // Phase 3: validate effect independence when an effects map is provided.
    if let Some(emap) = effects {
        validate_par_independence(branches, program_name, emap)?;
    }

    let mut aggregate = Endpoints::default();
    let mut group: Vec<String> = Vec::new();

    for branch in branches {
        let ep = compile_expr(
            branch, program_name, confidence, cost, safety,
            transitions, parallel_groups, known, effects,
        )?;
        group.extend(ep.starts.iter().cloned());
        aggregate.starts.extend(ep.starts);
        aggregate.ends.extend(ep.ends);
    }
    if !group.is_empty() {
        parallel_groups.push(group);
    }
    Ok(aggregate)
}

// ── Par independence ──────────────────────────────────────────────────────────

fn validate_par_independence(
    branches: &[Expr],
    program_name: &str,
    effects: &EffectsMap,
) -> Result<(), String> {
    let branch_effects: Vec<NodeEffects> = branches
        .iter()
        .map(|b| collect_branch_effects(b, program_name, effects))
        .collect::<Result<_, _>>()?;

    for i in 0..branch_effects.len() {
        for j in (i + 1)..branch_effects.len() {
            if !safe_parallel(&branch_effects[i], &branch_effects[j]) {
                return Err(format!(
                    "program '{program_name}': par branches {i} and {j} are not effect-independent"
                ));
            }
        }
    }
    Ok(())
}

fn collect_branch_effects(
    expr: &Expr,
    program_name: &str,
    effects: &EffectsMap,
) -> Result<NodeEffects, String> {
    let mut combined = NodeEffects::default();
    accumulate_effects(expr, program_name, effects, &mut combined)?;
    Ok(combined)
}

fn accumulate_effects(
    expr: &Expr,
    program_name: &str,
    effects: &EffectsMap,
    out: &mut NodeEffects,
) -> Result<(), String> {
    match expr {
        Expr::Zero | Expr::One => Ok(()),
        Expr::Node(name) => {
            let node_effects = effects.get(name.as_str()).ok_or_else(|| {
                format!(
                    "program '{program_name}': operator effects missing for par node '{name}'"
                )
            })?;
            out.reads.extend(node_effects.reads.iter().cloned());
            out.writes.extend(node_effects.writes.iter().cloned());
            out.locks.extend(node_effects.locks.iter().cloned());
            Ok(())
        }
        Expr::Seq(items) | Expr::Choice(items) | Expr::Par(items) => {
            for item in items {
                accumulate_effects(item, program_name, effects, out)?;
            }
            Ok(())
        }
        Expr::Star { body, .. } => accumulate_effects(body, program_name, effects, out),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_transition(
    from: &str,
    to: &str,
    confidence: f64,
    cost: f64,
    safety: f64,
    policy_effect: &str,
) -> Value {
    json!({
        "from":           from,
        "to":             to,
        "default_weight": confidence,
        "confidence":     confidence,
        "cost":           cost,
        "safety":         safety,
        "policy_effect":  policy_effect
    })
}

fn string_field<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{context} missing non-empty string field '{field}'"))
}

fn f64_field(value: &Value, field: &str, context: &str) -> Result<f64, String> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{context} missing numeric field '{field}'"))
}

// ── Phase 2: slot/resource validation ────────────────────────────────────────

/// Validate node declarations in `topology.source.json` against the declared
/// `slots` and `resources`.
///
/// For every node in `nodes[]`:
///   - each `reads[]` and `writes[]` entry must be a key in `slots`
///   - each `locks[]` entry must be a key in `resources`
///
/// Returns a list of human-readable violation strings.  An empty list means
/// the source topology is consistent.  Nodes without a `reads`/`writes`/`locks`
/// field are silently skipped (not all nodes declare effects yet).
///
/// This is intentionally non-fatal from the validator's perspective; callers
/// decide whether to treat violations as errors.
pub fn validate_source_node_effects(source: &Value) -> Vec<String> {
    let slots = match object_keys(source, "slots") {
        Ok(s) => s,
        Err(_) => BTreeSet::new(),
    };
    let resources = match object_keys(source, "resources") {
        Ok(r) => r,
        Err(_) => BTreeSet::new(),
    };

    let nodes = match source.get("nodes") {
        None => return Vec::new(),
        Some(Value::Array(nodes)) => nodes,
        Some(_) => return vec!["topology.source.json 'nodes' must be an array".to_string()],
    };

    let mut violations: Vec<String> = Vec::new();

    for node in nodes {
        let name = match node.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            Some(n) => n,
            None => {
                violations.push("source node missing non-empty 'name' field".to_string());
                continue;
            }
        };

        for field in &["reads", "writes"] {
            if let Some(Value::Array(items)) = node.get(*field) {
                for item in items {
                    match item.as_str() {
                        None => violations.push(format!(
                            "node '{name}': '{field}' entries must be strings"
                        )),
                        Some(slot) if !slots.contains(slot) => violations.push(format!(
                            "node '{name}': undeclared {field} slot '{slot}'"
                        )),
                        _ => {}
                    }
                }
            }
        }

        if let Some(Value::Array(locks)) = node.get("locks") {
            for lock in locks {
                match lock.as_str() {
                    None => violations
                        .push(format!("node '{name}': 'locks' entries must be strings")),
                    Some(r) if !resources.contains(r) => violations.push(format!(
                        "node '{name}': undeclared lock resource '{r}'"
                    )),
                    _ => {}
                }
            }
        }
    }

    violations
}

fn object_keys(source: &Value, field: &str) -> Result<BTreeSet<String>, String> {
    match source.get(field) {
        None => Ok(BTreeSet::new()),
        Some(Value::Object(obj)) => Ok(obj.keys().cloned().collect()),
        Some(_) => Err(format!("'{field}' must be an object")),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn nodes() -> BTreeSet<String> {
        ["A", "B", "C", "D", "E"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn source(programs: serde_json::Value) -> Value {
        json!({ "programs": programs })
    }

    fn compile(
        programs: serde_json::Value,
        existing: &[(&str, &str)],
    ) -> (Vec<Value>, Vec<Vec<String>>) {
        let ex: BTreeSet<_> = existing
            .iter()
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();
        compile_source_programs(&source(programs), &ex, &nodes(), None).unwrap()
    }

    #[test]
    fn seq_emits_adjacent_edges() {
        let (ts, _) = compile(
            json!([{
                "name": "p",
                "expr": { "seq": ["A", "B", "C"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }]),
            &[],
        );
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0]["from"], "A");
        assert_eq!(ts[0]["to"], "B");
        assert_eq!(ts[1]["from"], "B");
        assert_eq!(ts[1]["to"], "C");
    }

    #[test]
    fn seq_deduplicates_against_existing() {
        let (ts, _) = compile(
            json!([{
                "name": "p",
                "expr": { "seq": ["A", "B", "C"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }]),
            &[("A", "B")],
        );
        // A→B already exists, only B→C is new
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0]["from"], "B");
        assert_eq!(ts[0]["to"], "C");
    }

    #[test]
    fn choice_unions_endpoints_without_cross_edges() {
        let (ts, _) = compile(
            json!([{
                "name": "p",
                "expr": { "choice": ["A", "B"] },
                "confidence": 0.8, "cost": 2.0, "safety": 0.85
            }]),
            &[],
        );
        // choice emits no cross-edges between alternatives
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn par_emits_parallel_group() {
        let (ts, groups) = compile(
            json!([{
                "name": "p",
                "expr": { "par": [
                    { "seq": ["A", "B"] },
                    { "seq": ["C", "D"] }
                ]},
                "confidence": 0.99, "cost": 0.01, "safety": 0.99
            }]),
            &[],
        );
        // Each branch seq emits one edge; par doesn't add extra edges
        assert_eq!(ts.len(), 2);
        assert_eq!(groups.len(), 1);
        // Parallel group contains the start nodes of both branches
        assert!(groups[0].contains(&"A".to_string()));
        assert!(groups[0].contains(&"C".to_string()));
    }

    #[test]
    fn star_bounded_unroll() {
        let (ts, _) = compile(
            json!([{
                "name": "p",
                "expr": { "star": { "body": { "seq": ["A", "B"] }, "max_unroll": 3 } },
                "confidence": 0.7, "cost": 1.0, "safety": 0.9
            }]),
            &[],
        );
        // seq(A,B) emits A→B once per unroll (3×), and loop-back B→A twice (unrolls 1,2).
        // After dedup: A→B (1 unique) + B→A (1 unique) = 2 transitions.
        assert_eq!(ts.len(), 2);
        let pairs: Vec<(&str, &str)> = ts
            .iter()
            .map(|t| (t["from"].as_str().unwrap(), t["to"].as_str().unwrap()))
            .collect();
        assert!(pairs.contains(&("A", "B")));
        assert!(pairs.contains(&("B", "A")));
    }

    #[test]
    fn weight_object_schema() {
        let (ts, _) = compile(
            json!([{
                "name": "p",
                "expr": { "seq": ["A", "B"] },
                "weight": { "confidence": 0.95, "cost": 1.0, "safety": 0.98 }
            }]),
            &[],
        );
        assert_eq!(ts.len(), 1);
        assert!((ts[0]["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn unknown_node_is_rejected() {
        let ex: BTreeSet<(String, String)> = BTreeSet::new();
        let result = compile_source_programs(
            &source(json!([{
                "name": "p",
                "expr": { "seq": ["A", "Unknown"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }])),
            &ex,
            &nodes(),
            None,
        );
        assert!(result.unwrap_err().contains("unknown node"));
    }

    #[test]
    fn cka_pattern_kind_is_skipped_for_flat_transitions() {
        let (ts, _) = compile(
            json!([{
                "name": "loop_pattern",
                "kind": "cka_pattern",
                "expr": { "seq": ["A", "B"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }]),
            &[],
        );
        // cka_pattern programs must NOT produce flat topology transitions
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn par_independence_check_with_effects_map() {
        let mut emap = EffectsMap::new();
        emap.insert(
            "A".to_string(),
            NodeEffects { reads: BTreeSet::new(), writes: ["slot_a".to_string()].into(), locks: BTreeSet::new() },
        );
        emap.insert(
            "C".to_string(),
            NodeEffects { reads: BTreeSet::new(), writes: ["slot_c".to_string()].into(), locks: BTreeSet::new() },
        );
        let ex: BTreeSet<(String, String)> = BTreeSet::new();
        let result = compile_source_programs(
            &source(json!([{
                "name": "p",
                "expr": { "par": ["A", "C"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }])),
            &ex,
            &nodes(),
            Some(&emap),
        );
        // A and C write to disjoint slots — should pass
        assert!(result.is_ok());
    }

    #[test]
    fn par_with_conflicting_effects_is_rejected() {
        let mut emap = EffectsMap::new();
        emap.insert(
            "A".to_string(),
            NodeEffects { reads: BTreeSet::new(), writes: ["shared".to_string()].into(), locks: BTreeSet::new() },
        );
        emap.insert(
            "C".to_string(),
            NodeEffects { reads: ["shared".to_string()].into(), writes: BTreeSet::new(), locks: BTreeSet::new() },
        );
        let ex: BTreeSet<(String, String)> = BTreeSet::new();
        let result = compile_source_programs(
            &source(json!([{
                "name": "p",
                "expr": { "par": ["A", "C"] },
                "confidence": 0.9, "cost": 1.0, "safety": 0.9
            }])),
            &ex,
            &nodes(),
            Some(&emap),
        );
        assert!(result.unwrap_err().contains("not effect-independent"));
    }

    #[test]
    fn zero_and_one_emit_nothing() {
        let (ts, _) = compile(
            json!([
                { "name": "z", "expr": "zero", "confidence": 0.0, "cost": 0.0, "safety": 0.0 },
                { "name": "o", "expr": "one",  "confidence": 1.0, "cost": 0.0, "safety": 1.0 }
            ]),
            &[],
        );
        assert_eq!(ts.len(), 0);
    }

    // ── validate_source_node_effects ─────────────────────────────────────────

    fn source_with_nodes(slots: serde_json::Value, resources: serde_json::Value, nodes: serde_json::Value) -> Value {
        json!({ "slots": slots, "resources": resources, "nodes": nodes })
    }

    #[test]
    fn valid_node_declarations_produce_no_violations() {
        let src = source_with_nodes(
            json!({ "a": { "type": "json", "kind": "state" }, "b": { "type": "json", "kind": "state" } }),
            json!({ "lk": { "kind": "lock", "capacity": 1 } }),
            json!([{ "name": "Node::X", "reads": ["a"], "writes": ["b"], "locks": ["lk"] }]),
        );
        assert!(validate_source_node_effects(&src).is_empty());
    }

    #[test]
    fn undeclared_read_slot_is_flagged() {
        let src = source_with_nodes(
            json!({}),
            json!({}),
            json!([{ "name": "Node::X", "reads": ["missing.slot"], "writes": [], "locks": [] }]),
        );
        let v = validate_source_node_effects(&src);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("undeclared reads slot 'missing.slot'"), "{:?}", v);
    }

    #[test]
    fn undeclared_write_slot_is_flagged() {
        let src = source_with_nodes(
            json!({}),
            json!({}),
            json!([{ "name": "Node::X", "reads": [], "writes": ["missing.out"], "locks": [] }]),
        );
        let v = validate_source_node_effects(&src);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("undeclared writes slot"), "{:?}", v);
    }

    #[test]
    fn undeclared_lock_resource_is_flagged() {
        let src = source_with_nodes(
            json!({}),
            json!({}),
            json!([{ "name": "Node::X", "reads": [], "writes": [], "locks": ["no_such_lock"] }]),
        );
        let v = validate_source_node_effects(&src);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("undeclared lock resource"), "{:?}", v);
    }

    #[test]
    fn node_without_effects_fields_skipped() {
        // Node has no reads/writes/locks — should be fine (partial migration)
        let src = source_with_nodes(
            json!({}),
            json!({}),
            json!([{ "name": "Node::X", "kind": "event_node" }]),
        );
        assert!(validate_source_node_effects(&src).is_empty());
    }

    #[test]
    fn no_nodes_section_is_ok() {
        let src = json!({
            "slots": { "s": { "type": "json", "kind": "state" } },
            "resources": {}
        });
        assert!(validate_source_node_effects(&src).is_empty());
    }

    #[test]
    fn multiple_violations_all_reported() {
        let src = source_with_nodes(
            json!({}),
            json!({}),
            json!([
                { "name": "Node::A", "reads": ["x"], "writes": ["y"], "locks": [] },
                { "name": "Node::B", "reads": [],    "writes": [],    "locks": ["z"] }
            ]),
        );
        let v = validate_source_node_effects(&src);
        assert_eq!(v.len(), 3, "{:?}", v); // x, y, z all undeclared
    }

    #[test]
    fn empty_programs_array_is_ok() {
        let (ts, groups) = compile(json!([]), &[]);
        assert!(ts.is_empty());
        assert!(groups.is_empty());
    }

    #[test]
    fn missing_programs_field_is_ok() {
        let src = json!({ "matrix_name": "test" });
        let (ts, groups) =
            compile_source_programs(&src, &BTreeSet::new(), &nodes(), None).unwrap();
        assert!(ts.is_empty());
        assert!(groups.is_empty());
    }
}
