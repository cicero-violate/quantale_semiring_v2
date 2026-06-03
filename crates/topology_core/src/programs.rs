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

use std::collections::BTreeSet;

use serde_json::{Value, json};

/// Compile programs from a `topology.source.json` value into flat transition
/// objects and parallel group node-name lists.
///
/// Transitions whose `(from, to)` pair already exists in `existing_edges` are
/// silently skipped — Phase-1 preserves all flat transitions from
/// `topology.json` and programs only extend the edge set with new paths.
///
/// Unknown nodes referenced in expressions produce an error.
pub fn compile_source_programs(
    source: &Value,
    existing_edges: &BTreeSet<(String, String)>,
    known_node_names: &BTreeSet<String>,
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
) -> Result<Endpoints, String> {
    let mut iter = items.iter();
    let Some(first) = iter.next() else {
        return Ok(Endpoints::default());
    };
    let mut aggregate = compile_expr(
        first,
        program_name,
        confidence,
        cost,
        safety,
        transitions,
        parallel_groups,
        known,
    )?;
    let mut prev_ends = aggregate.ends.clone();

    for item in iter {
        let ep = compile_expr(
            item,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
        )?;
        // Connect every prev-end to every current-start.
        for from in &prev_ends {
            for to in &ep.starts {
                transitions.push(make_transition(
                    from,
                    to,
                    confidence,
                    cost,
                    safety,
                    program_name,
                ));
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
) -> Result<Endpoints, String> {
    let mut first = Endpoints::default();
    let mut prev_ends: Vec<String> = Vec::new();

    for idx in 0..max_unroll {
        let ep = compile_expr(
            body,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
        )?;
        if idx == 0 {
            first = ep.clone();
        } else {
            for from in &prev_ends {
                for to in &ep.starts {
                    transitions.push(make_transition(
                        from,
                        to,
                        confidence,
                        cost,
                        safety,
                        program_name,
                    ));
                }
            }
        }
        prev_ends = ep.ends;
    }

    if first.is_empty() {
        Ok(Endpoints::default())
    } else {
        Ok(Endpoints {
            starts: first.starts,
            ends: prev_ends,
        })
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
) -> Result<Endpoints, String> {
    let mut aggregate = Endpoints::default();
    let mut group: Vec<String> = Vec::new();

    for branch in branches {
        let ep = compile_expr(
            branch,
            program_name,
            confidence,
            cost,
            safety,
            transitions,
            parallel_groups,
            known,
        )?;
        // Collect start nodes into the parallel group metadata.
        group.extend(ep.starts.iter().cloned());
        aggregate.starts.extend(ep.starts);
        aggregate.ends.extend(ep.ends);
    }
    if !group.is_empty() {
        parallel_groups.push(group);
    }
    Ok(aggregate)
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
        compile_source_programs(&source(programs), &ex, &nodes()).unwrap()
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
        );
        assert!(result.unwrap_err().contains("unknown node"));
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
            compile_source_programs(&src, &BTreeSet::new(), &nodes()).unwrap();
        assert!(ts.is_empty());
        assert!(groups.is_empty());
    }
}
