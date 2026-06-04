//! Effect-aware fusible region partitioner.
//!
//! Finds maximal GPU-resident subgraphs where Fusable(F) holds:
//!
//!   B(F) = all nodes: runtime.backend = "cuda"
//!   K(F) = all nodes: kind = "kernel"
//!   S(F) = tensor slot layouts are static (assumed for all declared tensor slots)
//!   A(F) = seq or par composition inside — no choice boundary (Phase 6: linear chains only)
//!   R(F) = no boundary_node, governance gate, or lock conflict inside
//!
//! Emits FusionRegion descriptors with merged slot effects and quantale algebra
//! metadata. The caller writes these to `topology.fusion.json`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::programs::NodeEffects;

// ── Public types ──────────────────────────────────────────────────────────────

/// A maximal GPU-resident fusible kernel region.
pub struct FusionRegion {
    pub region: String,
    pub backend: String,
    pub fusion: String,
    pub nodes: Vec<String>,
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub locks: Vec<String>,
    pub compose: Vec<String>,
    pub join: Vec<String>,
}

impl FusionRegion {
    pub fn to_json(&self) -> Value {
        json!({
            "region":  self.region,
            "backend": self.backend,
            "fusion":  self.fusion,
            "nodes":   self.nodes,
            "reads":   self.reads,
            "writes":  self.writes,
            "locks":   self.locks,
            "quantale": {
                "compose": self.compose,
                "join":    self.join
            }
        })
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Partition the compiled transition graph into maximal fusible kernel regions.
///
/// `source` is `topology.source.json` (for node kind/backend declarations).
/// `transitions` is the full merged transition list from the compiled topology.
///
/// Returns one `FusionRegion` per maximal linear-chain subgraph satisfying
/// `Fusable(F)`. Singleton kernel nodes and non-linear subgraphs are skipped.
pub fn partition_fusible_regions(source: &Value, transitions: &[Value]) -> Vec<FusionRegion> {
    let fusible = fusible_node_info(source);
    if fusible.is_empty() {
        return vec![];
    }

    let nodes: BTreeSet<String> = fusible.keys().cloned().collect();
    let (adj_out, adj_in) = build_adj(&fusible, transitions);
    let components = connected_components(&nodes, &adj_out, &adj_in);
    let (compose, join) = quantale_algebra(source);

    let mut regions = Vec::new();

    for component in components {
        if component.len() < 2 {
            continue; // singletons don't gain from fusion
        }

        let comp_set: BTreeSet<&str> = component.iter().map(String::as_str).collect();

        // Linear chain check: each node has at most one internal in-edge and
        // one internal out-edge.
        let is_linear = component.iter().all(|n| {
            let in_count = adj_in.get(n).map_or(0, |ins| {
                ins.iter().filter(|i| comp_set.contains(i.as_str())).count()
            });
            let out_count = adj_out.get(n).map_or(0, |outs| {
                outs.iter()
                    .filter(|o| comp_set.contains(o.as_str()))
                    .count()
            });
            in_count <= 1 && out_count <= 1
        });

        if !is_linear {
            continue; // parallel/branching fusion is a Phase 7 extension
        }

        let chain = match topo_chain(&component, &adj_out, &adj_in) {
            Some(c) => c,
            None => continue,
        };

        // Merge slot effects across the chain.
        let all_reads: BTreeSet<String> = chain
            .iter()
            .flat_map(|n| fusible[n].reads.iter().cloned())
            .collect();
        let all_writes: BTreeSet<String> = chain
            .iter()
            .flat_map(|n| fusible[n].writes.iter().cloned())
            .collect();
        let all_locks: BTreeSet<String> = chain
            .iter()
            .flat_map(|n| fusible[n].locks.iter().cloned())
            .collect();

        // External reads  = slots consumed by the region but not produced inside.
        // External writes = slots produced by the region but not consumed inside.
        let ext_reads: Vec<String> = all_reads.difference(&all_writes).cloned().collect();
        let ext_writes: Vec<String> = all_writes.difference(&all_reads).cloned().collect();

        regions.push(FusionRegion {
            region: chain.join("__"),
            backend: "cuda_jit".to_string(),
            fusion: "linear_chain".to_string(),
            nodes: chain,
            reads: ext_reads,
            writes: ext_writes,
            locks: all_locks.into_iter().collect(),
            compose: compose.clone(),
            join: join.clone(),
        });
    }

    regions
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn str_set(v: Option<&Value>) -> BTreeSet<String> {
    v.and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Extract fusible nodes: kind=kernel AND runtime.backend=cuda.
fn fusible_node_info(source: &Value) -> BTreeMap<String, NodeEffects> {
    let mut result = BTreeMap::new();
    let nodes = match source.get("nodes").and_then(Value::as_array) {
        Some(n) => n,
        None => return result,
    };
    for node in nodes {
        let kind = node.get("kind").and_then(Value::as_str).unwrap_or("");
        let backend = node
            .get("runtime")
            .and_then(|r| r.get("backend"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if kind != "kernel" || backend != "cuda" {
            continue;
        }
        let name = match node.get("name").and_then(Value::as_str) {
            Some(n) => n.to_string(),
            None => continue,
        };
        result.insert(
            name,
            NodeEffects {
                reads: str_set(node.get("reads")),
                writes: str_set(node.get("writes")),
                locks: str_set(node.get("locks")),
            },
        );
    }
    result
}

/// Build directed adjacency lists (out and in) restricted to the fusible set.
fn build_adj(
    fusible: &BTreeMap<String, NodeEffects>,
    transitions: &[Value],
) -> (BTreeMap<String, Vec<String>>, BTreeMap<String, Vec<String>>) {
    let mut adj_out: BTreeMap<String, Vec<String>> =
        fusible.keys().map(|k| (k.clone(), vec![])).collect();
    let mut adj_in: BTreeMap<String, Vec<String>> =
        fusible.keys().map(|k| (k.clone(), vec![])).collect();

    for t in transitions {
        if is_fallback_transition(t) {
            continue;
        }
        let from = t.get("from").and_then(Value::as_str).unwrap_or("");
        let to = t.get("to").and_then(Value::as_str).unwrap_or("");
        if fusible.contains_key(from) && fusible.contains_key(to) {
            adj_out
                .entry(from.to_string())
                .or_default()
                .push(to.to_string());
            adj_in
                .entry(to.to_string())
                .or_default()
                .push(from.to_string());
        }
    }

    (adj_out, adj_in)
}

fn is_fallback_transition(t: &Value) -> bool {
    t.get("policy_effect")
        .and_then(Value::as_str)
        .map(|effect| effect.contains("Fallback"))
        .unwrap_or(false)
}

/// Find connected components via DFS on the undirected view of the graph.
fn connected_components(
    nodes: &BTreeSet<String>,
    adj_out: &BTreeMap<String, Vec<String>>,
    adj_in: &BTreeMap<String, Vec<String>>,
) -> Vec<Vec<String>> {
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut components: Vec<Vec<String>> = Vec::new();

    for start in nodes {
        if visited.contains(start) {
            continue;
        }
        let mut component: Vec<String> = Vec::new();
        let mut stack = vec![start.clone()];

        while let Some(n) = stack.pop() {
            if !visited.insert(n.clone()) {
                continue;
            }
            component.push(n.clone());
            if let Some(outs) = adj_out.get(&n) {
                for nb in outs {
                    if !visited.contains(nb) {
                        stack.push(nb.clone());
                    }
                }
            }
            if let Some(ins) = adj_in.get(&n) {
                for nb in ins {
                    if !visited.contains(nb) {
                        stack.push(nb.clone());
                    }
                }
            }
        }

        if !component.is_empty() {
            component.sort();
            components.push(component);
        }
    }

    components
}

/// Topologically order a linear-chain component.
///
/// Finds the unique source (in-degree 0 within the component) and follows the
/// single successor at each step. Returns `None` if the component is not a
/// simple path (multiple sources, branching, or cycle).
fn topo_chain(
    component: &[String],
    adj_out: &BTreeMap<String, Vec<String>>,
    adj_in: &BTreeMap<String, Vec<String>>,
) -> Option<Vec<String>> {
    let comp_set: BTreeSet<&str> = component.iter().map(String::as_str).collect();

    // Unique source: in-degree 0 within component.
    let mut sources = component.iter().filter(|n| {
        adj_in.get(*n).map_or(0, |ins| {
            ins.iter().filter(|i| comp_set.contains(i.as_str())).count()
        }) == 0
    });
    let source = sources.next()?;
    if sources.next().is_some() {
        return None; // multiple sources
    }

    let mut chain = Vec::with_capacity(component.len());
    let mut current = source.clone();

    loop {
        chain.push(current.clone());
        let next_in_comp: Vec<&String> = adj_out
            .get(&current)
            .map(|outs| {
                outs.iter()
                    .filter(|o| comp_set.contains(o.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        match next_in_comp.as_slice() {
            [] => break,
            [next] => current = (*next).clone(),
            _ => return None, // branching — not linear
        }

        if chain.len() > component.len() {
            return None; // cycle guard
        }
    }

    if chain.len() == component.len() {
        Some(chain)
    } else {
        None
    }
}

/// Extract (compose, join) layer sequences from the quantale declaration.
fn quantale_algebra(source: &Value) -> (Vec<String>, Vec<String>) {
    let layers = source
        .get("quantale")
        .and_then(|q| q.get("layers"))
        .and_then(Value::as_array);
    match layers {
        None => (vec![], vec![]),
        Some(ls) => {
            let compose = ls
                .iter()
                .filter_map(|l| l.get("compose").and_then(Value::as_str).map(str::to_string))
                .collect();
            let join = ls
                .iter()
                .filter_map(|l| l.get("join").and_then(Value::as_str).map(str::to_string))
                .collect();
            (compose, join)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn source_with_kernels(nodes: serde_json::Value) -> Value {
        json!({
            "quantale": {
                "layers": [
                    { "name": "confidence", "join": "max", "compose": "times", "bottom": 0.0, "unit": 1.0 },
                    { "name": "cost",       "join": "min", "compose": "plus",  "bottom": "inf", "unit": 0.0 },
                    { "name": "safety",     "join": "max", "compose": "min",   "bottom": 0.0, "unit": 1.0 }
                ]
            },
            "nodes": nodes
        })
    }

    fn cuda_kernel(name: &str, reads: &[&str], writes: &[&str]) -> Value {
        json!({
            "name": name,
            "kind": "kernel",
            "runtime": { "backend": "cuda" },
            "reads": reads,
            "writes": writes,
            "locks": []
        })
    }

    fn trans(from: &str, to: &str) -> Value {
        json!({ "from": from, "to": to, "confidence": 0.9, "cost": 1.0, "safety": 0.9 })
    }

    fn fallback_trans(from: &str, to: &str) -> Value {
        json!({
            "from": from,
            "to": to,
            "policy_effect": "DirectFallback",
            "confidence": 0.5,
            "cost": 2.0,
            "safety": 0.8
        })
    }

    #[test]
    fn empty_source_produces_no_regions() {
        let src = json!({ "nodes": [] });
        assert!(partition_fusible_regions(&src, &[]).is_empty());
    }

    #[test]
    fn single_kernel_produces_no_region() {
        let src = source_with_kernels(json!([cuda_kernel("K::A", &["slot.in"], &["slot.out"])]));
        assert!(partition_fusible_regions(&src, &[]).is_empty());
    }

    #[test]
    fn two_connected_kernels_form_linear_chain() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["y"], &["z"])
        ]));
        let ts = vec![trans("K::A", "K::B")];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].fusion, "linear_chain");
        assert_eq!(regions[0].nodes, vec!["K::A", "K::B"]);
        assert_eq!(regions[0].reads, vec!["x"]);
        assert_eq!(regions[0].writes, vec!["z"]);
    }

    #[test]
    fn three_node_analysis_chain() {
        let src = source_with_kernels(json!([
            cuda_kernel(
                "Analysis::Return1",
                &["market.price", "market.open"],
                &["analysis.return"]
            ),
            cuda_kernel(
                "Analysis::Volatility",
                &["market.price", "analysis.return"],
                &["analysis.volatility"]
            ),
            cuda_kernel(
                "Analysis::SignalScore",
                &["analysis.return", "analysis.volatility"],
                &["analysis.signal_score"]
            )
        ]));
        let ts = vec![
            trans("Analysis::Return1", "Analysis::Volatility"),
            trans("Analysis::Volatility", "Analysis::SignalScore"),
        ];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.fusion, "linear_chain");
        assert_eq!(
            r.nodes,
            vec![
                "Analysis::Return1",
                "Analysis::Volatility",
                "Analysis::SignalScore"
            ]
        );
        // External reads: market.price and market.open (not produced inside)
        let reads: BTreeSet<&str> = r.reads.iter().map(String::as_str).collect();
        assert!(reads.contains("market.price"), "reads={:?}", r.reads);
        assert!(reads.contains("market.open"), "reads={:?}", r.reads);
        assert!(
            !reads.contains("analysis.return"),
            "internal slot leaked into reads"
        );
        assert!(
            !reads.contains("analysis.volatility"),
            "internal slot leaked into reads"
        );
        // External write: only the final output
        assert_eq!(r.writes, vec!["analysis.signal_score"]);
        assert!(r.locks.is_empty());
    }

    #[test]
    fn disconnected_kernels_produce_no_multi_node_region() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["p"], &["q"])
        ]));
        // No edge between A and B.
        let regions = partition_fusible_regions(&src, &[]);
        assert!(
            regions.is_empty(),
            "singleton components must not form regions"
        );
    }

    #[test]
    fn non_cuda_node_is_not_fusible() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            {
                "name": "K::B",
                "kind": "kernel",
                "runtime": { "backend": "cpu" },
                "reads": ["y"], "writes": ["z"], "locks": []
            }
        ]));
        let ts = vec![trans("K::A", "K::B")];
        // K::B is not cuda, so not fusible; K::A is a singleton → no region.
        assert!(partition_fusible_regions(&src, &ts).is_empty());
    }

    #[test]
    fn non_kernel_node_is_not_fusible() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            {
                "name": "H::B",
                "kind": "host_node",
                "runtime": { "backend": "cuda" },
                "reads": ["y"], "writes": ["z"], "locks": []
            }
        ]));
        let ts = vec![trans("K::A", "H::B")];
        assert!(partition_fusible_regions(&src, &ts).is_empty());
    }

    #[test]
    fn branching_subgraph_is_not_linear_and_skipped() {
        // A → B and A → C: out-degree 2 for A, not linear.
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["y"], &["z"]),
            cuda_kernel("K::C", &["y"], &["w"])
        ]));
        let ts = vec![trans("K::A", "K::B"), trans("K::A", "K::C")];
        assert!(partition_fusible_regions(&src, &ts).is_empty());
    }

    #[test]
    fn fallback_edge_does_not_break_linear_chain() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["a"], &["b"]),
            cuda_kernel("K::B", &["b"], &["c"]),
            cuda_kernel("K::C", &["c"], &["d"])
        ]));
        let ts = vec![
            trans("K::A", "K::B"),
            fallback_trans("K::A", "K::C"),
            trans("K::B", "K::C"),
        ];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].nodes, vec!["K::A", "K::B", "K::C"]);
    }

    #[test]
    fn quantale_algebra_extracted_correctly() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["y"], &["z"])
        ]));
        let ts = vec![trans("K::A", "K::B")];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions[0].compose, vec!["times", "plus", "min"]);
        assert_eq!(regions[0].join, vec!["max", "min", "max"]);
    }

    #[test]
    fn region_name_is_nodes_joined_by_double_underscore() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["y"], &["z"])
        ]));
        let ts = vec![trans("K::A", "K::B")];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions[0].region, "K::A__K::B");
    }

    #[test]
    fn region_backend_is_cuda_jit() {
        let src = source_with_kernels(json!([
            cuda_kernel("K::A", &["x"], &["y"]),
            cuda_kernel("K::B", &["y"], &["z"])
        ]));
        let ts = vec![trans("K::A", "K::B")];
        let regions = partition_fusible_regions(&src, &ts);
        assert_eq!(regions[0].backend, "cuda_jit");
    }
}
