//! Phase-4 control-flow lowering.
//!
//! Converts CKA pattern expressions (from `patterns.source.json`) into flat
//! `ControlEdge` + `EffectTable` device tables for GPU-native `orchestrate_step` and
//! `check_effects_independent` without any CPU-side pattern interpretation at
//! runtime.

use serde_json::Value;

use crate::tensor::{
    CONTROL_OP_CHOICE, CONTROL_OP_HALT_OP, CONTROL_OP_PAR, CONTROL_OP_SEQ, CONTROL_OP_STAR_BOUNDED,
    ControlEdge, EffectTable,
};
use crate::topology::NodeRegistry;

#[derive(Debug)]
pub enum LoweringError {
    UnknownNode(String),
    UnsupportedExpr(String),
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownNode(n) => write!(f, "unknown node '{n}' in pattern expression"),
            Self::UnsupportedExpr(e) => write!(f, "unsupported pattern expression: {e}"),
        }
    }
}

/// Lowered pattern control table, ready for upload to device via
/// `TensorQuantaleWorld::load_control_table`.
#[derive(Clone, Debug, Default)]
pub struct PatternControlTable {
    pub edges: Vec<ControlEdge>,
    /// Per-node effect entries (one per node, indexed by node id).
    /// Length matches `NodeRegistry::len()`.
    pub effect_table: Vec<EffectTable>,
}

impl PatternControlTable {
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

// Internal result of lowering one expression sub-tree.
struct LoweredExpr {
    entries: Vec<i32>, // node ids that begin this expression
    exits: Vec<i32>,   // node ids that complete this expression
}

/// Lower all patterns from the source JSON into a `PatternControlTable`.
///
/// `patterns_json` is the root object with a `"patterns"` array (the content
/// of `assets/patterns.source.json`).  Each pattern's `"expr"` field is
/// lowered recursively into `ControlEdge` entries.
///
/// The resulting `effect_table` is default-initialised (all zeros = no
/// declared effects); callers that know node resource usage may overwrite
/// specific entries after calling this function.
pub fn lower_patterns_from_json(
    patterns_json: &Value,
    registry: &NodeRegistry,
) -> Result<PatternControlTable, LoweringError> {
    let mut table = PatternControlTable {
        edges: Vec::new(),
        effect_table: vec![EffectTable::default(); registry.len()],
    };

    let patterns = match patterns_json.get("patterns").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return Ok(table),
    };

    for pattern in patterns {
        if let Some(expr) = pattern.get("expr") {
            lower_expr(expr, registry, &mut table.edges)?;
        }
    }

    // Remove duplicate (op, lhs, rhs) triples that arise from shared sub-expressions.
    table
        .edges
        .dedup_by(|a, b| a.op == b.op && a.lhs == b.lhs && a.rhs == b.rhs);
    Ok(table)
}

fn lower_expr(
    expr: &Value,
    registry: &NodeRegistry,
    out: &mut Vec<ControlEdge>,
) -> Result<LoweredExpr, LoweringError> {
    // Leaf: string node name.
    if let Some(name) = expr.as_str() {
        return match name {
            "one" => Ok(LoweredExpr {
                entries: vec![],
                exits: vec![],
            }),
            "zero" => {
                out.push(ControlEdge {
                    op: CONTROL_OP_HALT_OP,
                    lhs: -1,
                    rhs: -1,
                    guard: 0,
                    order: 0,
                    bound: 0,
                });
                Ok(LoweredExpr {
                    entries: vec![],
                    exits: vec![],
                })
            }
            _ => {
                let id = registry
                    .id_of(name)
                    .ok_or_else(|| LoweringError::UnknownNode(name.to_string()))?
                    as i32;
                Ok(LoweredExpr {
                    entries: vec![id],
                    exits: vec![id],
                })
            }
        };
    }

    // Object: structural combinator.
    if let Some(obj) = expr.as_object() {
        if let Some(seq_arr) = obj.get("seq").and_then(|v| v.as_array()) {
            return lower_seq(seq_arr, registry, out);
        }
        if let Some(par_arr) = obj.get("par").and_then(|v| v.as_array()) {
            return lower_par(par_arr, registry, out);
        }
        if let Some(choice_arr) = obj.get("choice").and_then(|v| v.as_array()) {
            return lower_choice(choice_arr, registry, out);
        }
        if let Some(star_obj) = obj.get("star") {
            return lower_star(star_obj, registry, out);
        }
    }

    Err(LoweringError::UnsupportedExpr(format!("{expr}")))
}

fn lower_seq(
    elements: &[Value],
    registry: &NodeRegistry,
    out: &mut Vec<ControlEdge>,
) -> Result<LoweredExpr, LoweringError> {
    if elements.is_empty() {
        return Ok(LoweredExpr {
            entries: vec![],
            exits: vec![],
        });
    }

    let lowered: Vec<LoweredExpr> = elements
        .iter()
        .map(|e| lower_expr(e, registry, out))
        .collect::<Result<_, _>>()?;

    // SEQ edges: exits[i] → entries[i+1] at sequence position i.
    for i in 0..lowered.len().saturating_sub(1) {
        for &lhs in &lowered[i].exits {
            for &rhs in &lowered[i + 1].entries {
                out.push(ControlEdge {
                    op: CONTROL_OP_SEQ,
                    lhs,
                    rhs,
                    guard: 0,
                    order: i as i32,
                    bound: 0,
                });
            }
        }
    }

    let entries = lowered
        .first()
        .map(|l| l.entries.clone())
        .unwrap_or_default();
    let exits = lowered.last().map(|l| l.exits.clone()).unwrap_or_default();
    Ok(LoweredExpr { entries, exits })
}

fn lower_par(
    elements: &[Value],
    registry: &NodeRegistry,
    out: &mut Vec<ControlEdge>,
) -> Result<LoweredExpr, LoweringError> {
    let lowered: Vec<LoweredExpr> = elements
        .iter()
        .map(|e| lower_expr(e, registry, out))
        .collect::<Result<_, _>>()?;

    // PAR edges: symmetric pairs between entry nodes of all branches.
    for i in 0..lowered.len() {
        for j in (i + 1)..lowered.len() {
            for &lhs in &lowered[i].entries {
                for &rhs in &lowered[j].entries {
                    out.push(ControlEdge {
                        op: CONTROL_OP_PAR,
                        lhs,
                        rhs,
                        guard: 0,
                        order: 0,
                        bound: 0,
                    });
                    out.push(ControlEdge {
                        op: CONTROL_OP_PAR,
                        lhs: rhs,
                        rhs: lhs,
                        guard: 0,
                        order: 0,
                        bound: 0,
                    });
                }
            }
        }
    }

    let mut entries: Vec<i32> = lowered
        .iter()
        .flat_map(|l| l.entries.iter().copied())
        .collect();
    let mut exits: Vec<i32> = lowered
        .iter()
        .flat_map(|l| l.exits.iter().copied())
        .collect();
    entries.dedup();
    exits.dedup();
    Ok(LoweredExpr { entries, exits })
}

fn lower_choice(
    elements: &[Value],
    registry: &NodeRegistry,
    out: &mut Vec<ControlEdge>,
) -> Result<LoweredExpr, LoweringError> {
    let lowered: Vec<LoweredExpr> = elements
        .iter()
        .map(|e| lower_expr(e, registry, out))
        .collect::<Result<_, _>>()?;

    // CHOICE edges: between entry nodes of all branch pairs.
    for i in 0..lowered.len() {
        for j in (i + 1)..lowered.len() {
            for &lhs in &lowered[i].entries {
                for &rhs in &lowered[j].entries {
                    out.push(ControlEdge {
                        op: CONTROL_OP_CHOICE,
                        lhs,
                        rhs,
                        guard: 0,
                        order: 0,
                        bound: 0,
                    });
                }
            }
        }
    }

    let mut entries: Vec<i32> = lowered
        .iter()
        .flat_map(|l| l.entries.iter().copied())
        .collect();
    let mut exits: Vec<i32> = lowered
        .iter()
        .flat_map(|l| l.exits.iter().copied())
        .collect();
    entries.dedup();
    exits.dedup();
    Ok(LoweredExpr { entries, exits })
}

fn lower_star(
    star_obj: &Value,
    registry: &NodeRegistry,
    out: &mut Vec<ControlEdge>,
) -> Result<LoweredExpr, LoweringError> {
    let body = star_obj
        .get("body")
        .ok_or_else(|| LoweringError::UnsupportedExpr("star missing 'body'".into()))?;
    let bound = star_obj
        .get("max_unroll")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let lowered = lower_expr(body, registry, out)?;

    // STAR_BOUNDED back-edges: exit nodes loop back to entry nodes.
    for &lhs in &lowered.exits {
        for &rhs in &lowered.entries {
            out.push(ControlEdge {
                op: CONTROL_OP_STAR_BOUNDED,
                lhs,
                rhs,
                guard: 0,
                order: 0,
                bound,
            });
        }
    }

    Ok(LoweredExpr {
        entries: lowered.entries,
        exits: lowered.exits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_registry(names: &[&str]) -> NodeRegistry {
        use crate::topology::GraphTopology;
        let nodes: Vec<Value> = names
            .iter()
            .enumerate()
            .map(|(i, n)| json!({ "id": i, "name": n, "type": "State" }))
            .collect();
        let topo: GraphTopology = serde_json::from_value(json!({
            "matrix_name": "test",
            "nodes": nodes,
            "transitions": []
        }))
        .unwrap();
        topo.compile().unwrap().registry
    }

    fn effects_independent_cpu(ea: &EffectTable, eb: &EffectTable) -> bool {
        (ea.writes & (eb.reads | eb.writes)) == 0 && (eb.writes & (ea.reads | ea.writes)) == 0
    }

    #[test]
    fn seq_lowering_produces_ordered_edges() {
        let registry = make_registry(&["A", "B", "C"]);
        let patterns = json!({
            "patterns": [{
                "name": "test_seq", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": { "seq": ["A", "B", "C"] }
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();

        let a = registry.id_of("A").unwrap() as i32;
        let b = registry.id_of("B").unwrap() as i32;
        let c = registry.id_of("C").unwrap() as i32;

        let seq_edges: Vec<_> = table
            .edges
            .iter()
            .filter(|e| e.op == CONTROL_OP_SEQ)
            .collect();
        assert_eq!(seq_edges.len(), 2, "expected 2 SEQ edges for A→B→C");

        let ab = seq_edges.iter().find(|e| e.lhs == a && e.rhs == b).unwrap();
        assert_eq!(ab.order, 0);
        let bc = seq_edges.iter().find(|e| e.lhs == b && e.rhs == c).unwrap();
        assert_eq!(bc.order, 1);
    }

    #[test]
    fn par_lowering_produces_symmetric_par_edges() {
        let registry = make_registry(&["X", "Y"]);
        let patterns = json!({
            "patterns": [{
                "name": "test_par", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": { "par": ["X", "Y"] }
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();

        let x = registry.id_of("X").unwrap() as i32;
        let y = registry.id_of("Y").unwrap() as i32;
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_PAR && e.lhs == x && e.rhs == y)
        );
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_PAR && e.lhs == y && e.rhs == x)
        );
    }

    #[test]
    fn choice_lowering_produces_choice_edges() {
        let registry = make_registry(&["R", "T"]);
        let patterns = json!({
            "patterns": [{
                "name": "test_choice", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": { "choice": ["R", "T"] }
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();

        let r = registry.id_of("R").unwrap() as i32;
        let t = registry.id_of("T").unwrap() as i32;
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_CHOICE && e.lhs == r && e.rhs == t)
        );
    }

    #[test]
    fn star_bounded_lowering_has_bound_set() {
        let registry = make_registry(&["V", "M", "L"]);
        let patterns = json!({
            "patterns": [{
                "name": "test_star", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": { "star": { "body": { "seq": ["V", "M", "L"] }, "max_unroll": 3 } }
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();

        let v = registry.id_of("V").unwrap() as i32;
        let l = registry.id_of("L").unwrap() as i32;
        let star_edge = table
            .edges
            .iter()
            .find(|e| e.op == CONTROL_OP_STAR_BOUNDED && e.lhs == l && e.rhs == v);
        assert!(star_edge.is_some(), "expected STAR_BOUNDED back-edge L→V");
        assert_eq!(star_edge.unwrap().bound, 3);
    }

    #[test]
    fn nested_seq_par_produces_seq_and_par_edges() {
        // seq [A, par [B, C], D]
        let registry = make_registry(&["A", "B", "C", "D"]);
        let patterns = json!({
            "patterns": [{
                "name": "nested", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": { "seq": ["A", { "par": ["B", "C"] }, "D"] }
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();

        let a = registry.id_of("A").unwrap() as i32;
        let b = registry.id_of("B").unwrap() as i32;
        let c = registry.id_of("C").unwrap() as i32;
        let d = registry.id_of("D").unwrap() as i32;

        // A precedes both B and C (seq position 0)
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_SEQ && e.lhs == a && e.rhs == b)
        );
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_SEQ && e.lhs == a && e.rhs == c)
        );
        // B and C are par (symmetric)
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_PAR && e.lhs == b && e.rhs == c)
        );
        // Both B and C precede D (seq position 1)
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_SEQ && e.lhs == b && e.rhs == d)
        );
        assert!(
            table
                .edges
                .iter()
                .any(|e| e.op == CONTROL_OP_SEQ && e.lhs == c && e.rhs == d)
        );
    }

    #[test]
    fn effects_independent_no_conflict() {
        let ea = EffectTable {
            reads: 0b01,
            writes: 0b10,
            locks: 0,
            safety_class: 0,
        };
        let eb = EffectTable {
            reads: 0b100,
            writes: 0b1000,
            locks: 0,
            safety_class: 0,
        };
        // ea.writes=0b10 ∩ (eb.reads|eb.writes)=0b1100 → 0 → independent
        assert!(effects_independent_cpu(&ea, &eb));
    }

    #[test]
    fn effects_conflict_write_read() {
        let ea = EffectTable {
            reads: 0b01,
            writes: 0b10,
            locks: 0,
            safety_class: 0,
        };
        let eb = EffectTable {
            reads: 0b10,
            writes: 0b01,
            locks: 0,
            safety_class: 0,
        };
        // ea.writes=0b10 ∩ (eb.reads|eb.writes)=0b11 → 0b10 ≠ 0 → not independent
        assert!(!effects_independent_cpu(&ea, &eb));
    }

    #[test]
    fn one_pattern_produces_empty_edge_set() {
        let registry = make_registry(&["A"]);
        let patterns = json!({
            "patterns": [{
                "name": "identity", "confidence": 1.0, "cost": 0.0, "safety": 1.0,
                "expr": "one"
            }]
        });
        let table = lower_patterns_from_json(&patterns, &registry).unwrap();
        // "one" is the identity; no edges should be emitted.
        assert!(
            table
                .edges
                .iter()
                .filter(|e| e.op != CONTROL_OP_HALT_OP)
                .count()
                == 0,
            "identity pattern should produce no control edges"
        );
    }
}
