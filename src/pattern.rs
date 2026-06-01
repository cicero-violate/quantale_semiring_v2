//! Data-driven Concurrent Kleene Algebra pattern compiler.
//!
//! CKA suggests structural tensor-edge deltas. The tensor quantale still owns
//! scoring/routing, and receipt deltas remain the execution truth gate.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::config::{load_operator_registry, OperatorRegistry};
use crate::error::CudaError;
use crate::tensor::TensorEdge;
use crate::topology::{CompiledTopology, GraphTopology};

pub const DEFAULT_PATTERNS_JSON: &str = include_str!("../assets/patterns.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum CkaExpr {
    Zero,
    One,
    Node(String),
    Seq(Vec<CkaExpr>),
    Choice(Vec<CkaExpr>),
    Star {
        body: Box<CkaExpr>,
        max_unroll: usize,
    },
    Par(Vec<CkaExpr>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CkaPattern {
    pub name: String,
    pub expr: CkaExpr,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CkaPatternSet {
    pub patterns: Vec<CkaPattern>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompiledCkaPattern {
    pub name: String,
    pub edges: Vec<TensorEdge>,
    pub parallel_groups: Vec<Vec<i32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorEffects {
    pub reads: BTreeSet<String>,
    pub writes: BTreeSet<String>,
    pub locks: BTreeSet<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Endpoints {
    starts: Vec<String>,
    ends: Vec<String>,
}

impl Endpoints {
    fn from_node(node: String) -> Self {
        Self {
            starts: vec![node.clone()],
            ends: vec![node],
        }
    }

    fn is_empty(&self) -> bool {
        self.starts.is_empty() && self.ends.is_empty()
    }
}

impl<'de> Deserialize<'de> for CkaExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_cka_expr_value(value).map_err(serde::de::Error::custom)
    }
}

pub fn load_default_patterns() -> Result<CkaPatternSet, CudaError> {
    parse_patterns_str(DEFAULT_PATTERNS_JSON)
}

pub fn load_patterns(path: impl AsRef<Path>) -> Result<CkaPatternSet, CudaError> {
    let input = fs::read_to_string(path.as_ref())
        .map_err(|error| CudaError::invalid_input(format!("read patterns asset: {error}")))?;
    parse_patterns_str(&input)
}

pub fn parse_patterns_str(input: &str) -> Result<CkaPatternSet, CudaError> {
    serde_json::from_str(input).map_err(|error| CudaError::invalid_input(error.to_string()))
}

pub fn compile_patterns_to_tensor_edges(
    patterns: &CkaPatternSet,
) -> Result<Vec<TensorEdge>, CudaError> {
    let topology = GraphTopology::default_asset()?.compile()?;
    let operator_registry = load_operator_registry("assets/operators.json").unwrap_or_default();
    let mut edges = Vec::new();
    for pattern in &patterns.patterns {
        edges.extend(compile_pattern(pattern, &topology, &operator_registry)?.edges);
    }
    Ok(edges)
}

pub fn compile_pattern(
    pattern: &CkaPattern,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
) -> Result<CompiledCkaPattern, CudaError> {
    validate_cka_expr(&pattern.expr, topology, operator_registry)?;
    let mut edges = Vec::new();
    let mut parallel_groups = Vec::new();
    compile_expr(
        &pattern.expr,
        pattern,
        topology,
        operator_registry,
        &mut edges,
        &mut parallel_groups,
    )?;
    Ok(CompiledCkaPattern {
        name: pattern.name.clone(),
        edges,
        parallel_groups,
    })
}

pub fn validate_cka_expr(
    expr: &CkaExpr,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
) -> Result<(), CudaError> {
    match expr {
        CkaExpr::Zero | CkaExpr::One => Ok(()),
        CkaExpr::Node(node) => {
            if topology.registry.id_of(node).is_none() {
                return Err(CudaError::invalid_input(format!(
                    "unknown CKA node '{node}'"
                )));
            }
            Ok(())
        }
        CkaExpr::Seq(items) => {
            if items.is_empty() {
                return Err(CudaError::invalid_input("seq requires at least one item"));
            }
            for item in items {
                validate_cka_expr(item, topology, operator_registry)?;
            }
            Ok(())
        }
        CkaExpr::Choice(items) => {
            if items.is_empty() {
                return Err(CudaError::invalid_input(
                    "choice requires at least one item",
                ));
            }
            for item in items {
                validate_cka_expr(item, topology, operator_registry)?;
            }
            Ok(())
        }
        CkaExpr::Star { body, max_unroll } => {
            if *max_unroll == 0 {
                return Err(CudaError::invalid_input(
                    "star max_unroll must be greater than zero",
                ));
            }
            validate_cka_expr(body, topology, operator_registry)
        }
        CkaExpr::Par(branches) => {
            if branches.len() < 2 {
                return Err(CudaError::invalid_input(
                    "par requires at least two branches",
                ));
            }
            for branch in branches {
                validate_cka_expr(branch, topology, operator_registry)?;
            }
            validate_parallel_independence(branches, operator_registry)
        }
    }
}

pub fn safe_parallel(a: &OperatorEffects, b: &OperatorEffects) -> bool {
    a.writes.is_disjoint(&b.writes)
        && a.writes.is_disjoint(&b.reads)
        && a.reads.is_disjoint(&b.writes)
        && a.locks.is_disjoint(&b.locks)
}

fn compile_expr(
    expr: &CkaExpr,
    pattern: &CkaPattern,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
    edges: &mut Vec<TensorEdge>,
    parallel_groups: &mut Vec<Vec<i32>>,
) -> Result<Endpoints, CudaError> {
    match expr {
        CkaExpr::Zero | CkaExpr::One => Ok(Endpoints::default()),
        CkaExpr::Node(node) => Ok(Endpoints::from_node(node.clone())),
        CkaExpr::Seq(items) => compile_seq(
            items,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        ),
        CkaExpr::Choice(items) => {
            let mut endpoints = Endpoints::default();
            for item in items {
                let compiled = compile_expr(
                    item,
                    pattern,
                    topology,
                    operator_registry,
                    edges,
                    parallel_groups,
                )?;
                endpoints.starts.extend(compiled.starts);
                endpoints.ends.extend(compiled.ends);
            }
            Ok(endpoints)
        }
        CkaExpr::Star { body, max_unroll } => compile_star(
            body,
            *max_unroll,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        ),
        CkaExpr::Par(branches) => compile_par(
            branches,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        ),
    }
}

fn compile_seq(
    items: &[CkaExpr],
    pattern: &CkaPattern,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
    edges: &mut Vec<TensorEdge>,
    parallel_groups: &mut Vec<Vec<i32>>,
) -> Result<Endpoints, CudaError> {
    let mut iter = items.iter();
    let Some(first) = iter.next() else {
        return Ok(Endpoints::default());
    };
    let mut aggregate = compile_expr(
        first,
        pattern,
        topology,
        operator_registry,
        edges,
        parallel_groups,
    )?;
    let mut previous_ends = aggregate.ends.clone();

    for item in iter {
        let current = compile_expr(
            item,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        )?;
        if !previous_ends.is_empty() && !current.starts.is_empty() {
            for from in &previous_ends {
                for to in &current.starts {
                    push_edge(edges, topology, from, to, pattern)?;
                }
            }
        }
        if aggregate.starts.is_empty() {
            aggregate.starts = current.starts.clone();
        }
        if !current.ends.is_empty() {
            previous_ends = current.ends.clone();
            aggregate.ends = current.ends;
        }
    }
    Ok(aggregate)
}

fn compile_star(
    body: &CkaExpr,
    max_unroll: usize,
    pattern: &CkaPattern,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
    edges: &mut Vec<TensorEdge>,
    parallel_groups: &mut Vec<Vec<i32>>,
) -> Result<Endpoints, CudaError> {
    let mut first_iteration = Endpoints::default();
    let mut previous_ends: Vec<String> = Vec::new();

    for index in 0..max_unroll {
        let current = compile_expr(
            body,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        )?;
        if index == 0 {
            first_iteration = current.clone();
        } else {
            for from in &previous_ends {
                for to in &current.starts {
                    push_edge(edges, topology, from, to, pattern)?;
                }
            }
        }
        previous_ends = current.ends;
    }

    if first_iteration.is_empty() {
        Ok(Endpoints::default())
    } else {
        Ok(Endpoints {
            starts: first_iteration.starts,
            ends: previous_ends,
        })
    }
}

fn compile_par(
    branches: &[CkaExpr],
    pattern: &CkaPattern,
    topology: &CompiledTopology,
    operator_registry: &OperatorRegistry,
    edges: &mut Vec<TensorEdge>,
    parallel_groups: &mut Vec<Vec<i32>>,
) -> Result<Endpoints, CudaError> {
    validate_parallel_independence(branches, operator_registry)?;

    let mut endpoints = Endpoints::default();
    let mut group = Vec::new();
    for branch in branches {
        let compiled = compile_expr(
            branch,
            pattern,
            topology,
            operator_registry,
            edges,
            parallel_groups,
        )?;
        for start in &compiled.starts {
            let id = topology
                .registry
                .id_of(start)
                .ok_or_else(|| CudaError::invalid_input(format!("unknown CKA node '{start}'")))?;
            group.push(id as i32);
        }
        endpoints.starts.extend(compiled.starts);
        endpoints.ends.extend(compiled.ends);
    }
    parallel_groups.push(group);
    Ok(endpoints)
}

fn push_edge(
    edges: &mut Vec<TensorEdge>,
    topology: &CompiledTopology,
    from: &str,
    to: &str,
    pattern: &CkaPattern,
) -> Result<(), CudaError> {
    let src = topology
        .registry
        .id_of(from)
        .ok_or_else(|| CudaError::invalid_input(format!("unknown CKA source '{from}'")))?;
    let dst = topology
        .registry
        .id_of(to)
        .ok_or_else(|| CudaError::invalid_input(format!("unknown CKA destination '{to}'")))?;
    edges.push(TensorEdge::new(
        src as i32,
        dst as i32,
        pattern.confidence,
        pattern.cost,
        pattern.safety,
    ));
    Ok(())
}

fn validate_parallel_independence(
    branches: &[CkaExpr],
    operator_registry: &OperatorRegistry,
) -> Result<(), CudaError> {
    let effects = branches
        .iter()
        .map(|branch| branch_effects(branch, operator_registry))
        .collect::<Result<Vec<_>, _>>()?;

    for left in 0..effects.len() {
        for right in (left + 1)..effects.len() {
            if !safe_parallel(&effects[left], &effects[right]) {
                return Err(CudaError::invalid_input(format!(
                    "par branches {left} and {right} are not effect-independent"
                )));
            }
        }
    }
    Ok(())
}

fn branch_effects(
    expr: &CkaExpr,
    operator_registry: &OperatorRegistry,
) -> Result<OperatorEffects, CudaError> {
    let mut effects = OperatorEffects::default();
    collect_effects(expr, operator_registry, &mut effects)?;
    Ok(effects)
}

fn collect_effects(
    expr: &CkaExpr,
    operator_registry: &OperatorRegistry,
    out: &mut OperatorEffects,
) -> Result<(), CudaError> {
    match expr {
        CkaExpr::Zero | CkaExpr::One => Ok(()),
        CkaExpr::Node(node) => {
            let Some(operator) = operator_registry.get(node) else {
                return Err(CudaError::invalid_input(format!(
                    "operator effects missing for par node '{node}'"
                )));
            };
            let parsed = parse_operator_effects(node, operator)?;
            out.reads.extend(parsed.reads);
            out.writes.extend(parsed.writes);
            out.locks.extend(parsed.locks);
            Ok(())
        }
        CkaExpr::Seq(items) | CkaExpr::Choice(items) | CkaExpr::Par(items) => {
            for item in items {
                collect_effects(item, operator_registry, out)?;
            }
            Ok(())
        }
        CkaExpr::Star { body, .. } => collect_effects(body, operator_registry, out),
    }
}

fn parse_operator_effects(node: &str, operator: &Value) -> Result<OperatorEffects, CudaError> {
    let effects = operator.get("effects").ok_or_else(|| {
        CudaError::invalid_input(format!("operator effects missing for par node '{node}'"))
    })?;
    Ok(OperatorEffects {
        reads: string_set(effects.get("reads"), node, "reads")?,
        writes: string_set(effects.get("writes"), node, "writes")?,
        locks: string_set(effects.get("locks"), node, "locks")?,
    })
}

pub fn operator_effects_for_node(
    node: &str,
    operator_registry: &OperatorRegistry,
) -> Result<OperatorEffects, CudaError> {
    let operator = operator_registry.get(node).ok_or_else(|| {
        CudaError::invalid_input(format!("operator effects missing for par node '{node}'"))
    })?;
    parse_operator_effects(node, operator)
}

fn string_set(
    value: Option<&Value>,
    node: &str,
    field: &str,
) -> Result<BTreeSet<String>, CudaError> {
    let Some(value) = value else {
        return Ok(BTreeSet::new());
    };
    let Some(items) = value.as_array() else {
        return Err(CudaError::invalid_input(format!(
            "operator effects field '{field}' for '{node}' must be an array"
        )));
    };
    let mut set = BTreeSet::new();
    for item in items {
        let Some(item) = item.as_str() else {
            return Err(CudaError::invalid_input(format!(
                "operator effects field '{field}' for '{node}' must contain strings"
            )));
        };
        set.insert(item.to_string());
    }
    Ok(set)
}

fn parse_cka_expr_value(value: Value) -> Result<CkaExpr, String> {
    match value {
        Value::String(value) => match value.as_str() {
            "zero" | "blocked" | "impossible" => Ok(CkaExpr::Zero),
            "one" | "identity" | "skip" => Ok(CkaExpr::One),
            _ => Ok(CkaExpr::Node(value)),
        },
        Value::Object(mut object) => {
            if let Some(value) = object.remove("node") {
                return value
                    .as_str()
                    .map(|node| CkaExpr::Node(node.to_string()))
                    .ok_or_else(|| "node expression requires a string".to_string());
            }
            if object.remove("zero").is_some() || object.remove("blocked").is_some() {
                return Ok(CkaExpr::Zero);
            }
            if object.remove("one").is_some() || object.remove("skip").is_some() {
                return Ok(CkaExpr::One);
            }
            if let Some(value) = object.remove("seq") {
                return Ok(CkaExpr::Seq(parse_expr_array(value, "seq")?));
            }
            if let Some(value) = object.remove("choice") {
                return Ok(CkaExpr::Choice(parse_expr_array(value, "choice")?));
            }
            if let Some(value) = object.remove("par") {
                return Ok(CkaExpr::Par(parse_expr_array(value, "par")?));
            }
            if let Some(value) = object.remove("star") {
                let Value::Object(mut star) = value else {
                    return Err("star expression requires an object".to_string());
                };
                let body = star
                    .remove("body")
                    .ok_or_else(|| "star expression requires body".to_string())?;
                let max_unroll = star
                    .remove("max_unroll")
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| "star expression requires integer max_unroll".to_string())?;
                let max_unroll = usize::try_from(max_unroll)
                    .map_err(|_| "star max_unroll overflows usize".to_string())?;
                return Ok(CkaExpr::Star {
                    body: Box::new(parse_cka_expr_value(body)?),
                    max_unroll,
                });
            }
            Err("unknown CKA expression object".to_string())
        }
        _ => Err("CKA expression must be a string or object".to_string()),
    }
}

fn parse_expr_array(value: Value, field: &str) -> Result<Vec<CkaExpr>, String> {
    let Value::Array(items) = value else {
        return Err(format!("{field} expression requires an array"));
    };
    items.into_iter().map(parse_cka_expr_value).collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn topology() -> CompiledTopology {
        GraphTopology::default_asset().unwrap().compile().unwrap()
    }

    fn registry() -> OperatorRegistry {
        load_operator_registry("assets/operators.json").unwrap()
    }

    fn pattern(expr: CkaExpr) -> CkaPattern {
        CkaPattern {
            name: "test".to_string(),
            expr,
            confidence: 0.7,
            cost: 2.0,
            safety: 0.8,
        }
    }

    #[test]
    fn seq_compiles_adjacent_tensor_edges() {
        let compiled = compile_pattern(
            &pattern(CkaExpr::Seq(vec![
                CkaExpr::Node("State::Plan".to_string()),
                CkaExpr::Node("State::Optimize".to_string()),
                CkaExpr::Node("State::Execute".to_string()),
            ])),
            &topology(),
            &registry(),
        )
        .unwrap();
        assert_eq!(compiled.edges.len(), 2);
        assert_eq!(compiled.edges[0].confidence, 0.7);
        assert_eq!(compiled.edges[0].cost, 2.0);
        assert_eq!(compiled.edges[0].safety, 0.8);
    }

    #[test]
    fn choice_compiles_alternatives_without_false_sequencing() {
        let compiled = compile_pattern(
            &pattern(CkaExpr::Choice(vec![
                CkaExpr::Seq(vec![
                    CkaExpr::Node("Control::Repair".to_string()),
                    CkaExpr::Node("State::Optimize".to_string()),
                ]),
                CkaExpr::Seq(vec![
                    CkaExpr::Node("Control::Retry".to_string()),
                    CkaExpr::Node("State::Optimize".to_string()),
                ]),
            ])),
            &topology(),
            &registry(),
        )
        .unwrap();
        assert_eq!(compiled.edges.len(), 2);
        assert_ne!(compiled.edges[0].src, compiled.edges[1].src);
    }

    #[test]
    fn star_bounded_unroll_compiles_finite_edges_only() {
        let compiled = compile_pattern(
            &pattern(CkaExpr::Star {
                body: Box::new(CkaExpr::Seq(vec![
                    CkaExpr::Node("State::Validate".to_string()),
                    CkaExpr::Node("State::Memory".to_string()),
                    CkaExpr::Node("State::Learn".to_string()),
                ])),
                max_unroll: 3,
            }),
            &topology(),
            &registry(),
        )
        .unwrap();
        assert_eq!(compiled.edges.len(), 8);
    }

    #[test]
    fn par_requires_effect_independence() {
        let mut registry = registry();
        registry.insert(
            "State::Map".to_string(),
            json!({"effects": {"reads": [], "writes": ["shared"], "locks": []}}),
        );
        registry.insert(
            "State::Parse".to_string(),
            json!({"effects": {"reads": ["shared"], "writes": [], "locks": []}}),
        );
        let err = compile_pattern(
            &pattern(CkaExpr::Par(vec![
                CkaExpr::Node("State::Map".to_string()),
                CkaExpr::Node("State::Parse".to_string()),
            ])),
            &topology(),
            &registry,
        )
        .unwrap_err();
        assert!(err.message.contains("not effect-independent"));
    }

    #[test]
    fn par_compiles_when_effects_are_independent() {
        let compiled = compile_pattern(
            &pattern(CkaExpr::Par(vec![
                CkaExpr::Seq(vec![
                    CkaExpr::Node("State::Map".to_string()),
                    CkaExpr::Node("State::Search".to_string()),
                ]),
                CkaExpr::Seq(vec![
                    CkaExpr::Node("State::Parse".to_string()),
                    CkaExpr::Node("State::Score".to_string()),
                ]),
            ])),
            &topology(),
            &registry(),
        )
        .unwrap();
        assert_eq!(compiled.edges.len(), 2);
        assert_eq!(compiled.parallel_groups.len(), 1);
        assert_eq!(compiled.parallel_groups[0].len(), 2);
    }

    #[test]
    fn unknown_node_is_rejected() {
        let err = compile_pattern(
            &pattern(CkaExpr::Node("State::Missing".to_string())),
            &topology(),
            &registry(),
        )
        .unwrap_err();
        assert!(err.message.contains("unknown CKA node"));
    }

    #[test]
    fn bad_pattern_json_is_rejected() {
        let err = parse_patterns_str(r#"{"patterns":[{"name":"bad","expr":{"star":{}},"confidence":1.0,"cost":1.0,"safety":1.0}]}"#)
            .unwrap_err();
        assert!(err.message.contains("star expression requires body"));
    }

    #[test]
    fn bundled_patterns_compile_to_tensor_edge_values() {
        let patterns = load_default_patterns().unwrap();
        let edges = compile_patterns_to_tensor_edges(&patterns).unwrap();
        assert!(!edges.is_empty());
        assert!(edges.iter().all(|edge| edge.confidence > 0.0));
        assert!(edges.iter().all(|edge| edge.cost >= 0.0));
        assert!(edges.iter().all(|edge| edge.safety > 0.0));
    }
}
