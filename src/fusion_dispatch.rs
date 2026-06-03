//! Bridges `topology.fusion.json` to the `jit_kernel_fusion` JIT pipeline.
//!
//! `FusionDispatch` loads the fusion regions emitted by `topology build-overlay`,
//! builds `JitChain`s from the operator registry, and provides O(1) lookup so
//! the runtime can route fused-kernel execution without re-reading the fusion
//! artifact on every tick.
//!
//! # Dispatch path
//!
//! ```text
//! topology.fusion.json
//!   └─ FusionDispatch::load(path, &operator_registry)
//!       └─ detect_jit_chains(region.nodes, registry) → JitChain
//!           └─ synthesize_kernel(&chain, registry)   → CUDA C source
//!               └─ JitCache::get_or_compile(device, &chain, registry) → CudaFunction
//! ```
//!
//! Synthesis (CUDA C → PTX) is deferred to `JitCache` and is guarded by
//! `#[cfg(feature = "cuda")]`.  Everything in this module works without a
//! CUDA device.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::config::OperatorRegistry;
use crate::jit_kernel_fusion::{
    JitChain, JitChainMetadata, chain_metadata, detect_jit_chains, synthesize_kernel,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// A single fusible region with its compiled JIT chain and slot metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct FusionEntry {
    pub region: String,
    pub nodes: Vec<String>,
    pub chain: JitChain,
    pub metadata: JitChainMetadata,
    /// External input slots (reads not produced within the region).
    pub reads: Vec<String>,
    /// External output slots (writes not consumed within the region).
    pub writes: Vec<String>,
}

/// A synthesized (but not yet compiled) CUDA C kernel source for one region.
pub struct SynthesizedKernel {
    pub region: String,
    pub source: String,
}

/// Fast-dispatch index: maps region node names to `FusionEntry`s.
///
/// Loaded from `assets/topology.fusion.json` at startup.  Provides O(1)
/// dispatch lookup by either the entry node (first node in the chain) or any
/// member node.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FusionDispatch {
    by_entry: HashMap<String, usize>,
    by_member: HashMap<String, usize>,
    pub entries: Vec<FusionEntry>,
}

// ── Load API ──────────────────────────────────────────────────────────────────

impl FusionDispatch {
    /// Load `topology.fusion.json` and build `JitChain`s from the operator
    /// registry.
    ///
    /// Returns `Ok(FusionDispatch::default())` if the file does not exist.
    /// Regions whose nodes are not all `executable=jit_cuda` are skipped with
    /// an `eprintln!` warning; I/O or JSON parse errors are returned as `Err`.
    pub fn load(
        fusion_path: impl AsRef<Path>,
        registry: &OperatorRegistry,
    ) -> Result<Self, String> {
        let path = fusion_path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw =
            fs::read_to_string(path).map_err(|e| format!("read '{}': {e}", path.display()))?;
        Self::from_json_str(&raw, registry)
    }

    /// Parse a JSON string directly (useful for testing without the filesystem).
    pub fn from_json_str(json: &str, registry: &OperatorRegistry) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("parse fusion json: {e}"))?;
        Self::from_value(value, registry)
    }

    fn from_value(json: Value, registry: &OperatorRegistry) -> Result<Self, String> {
        let regions = match json.get("regions").and_then(Value::as_array) {
            Some(r) => r,
            None => return Ok(Self::default()),
        };

        let mut dispatch = Self::default();

        for region_val in regions {
            let region_name = region_val
                .get("region")
                .and_then(Value::as_str)
                .unwrap_or("?");

            let nodes: Vec<String> = str_vec(region_val.get("nodes"));
            let reads: Vec<String> = str_vec(region_val.get("reads"));
            let writes: Vec<String> = str_vec(region_val.get("writes"));

            if nodes.len() < 2 {
                continue;
            }

            // Build JitChain via the existing chain detector — validates that
            // each node is jit_cuda and that data-flow links are consistent.
            let chains = match detect_jit_chains(&nodes, registry) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[fusion_dispatch] region '{region_name}': skipped — {e}");
                    continue;
                }
            };

            if chains.len() != 1 {
                eprintln!(
                    "[fusion_dispatch] region '{region_name}': expected 1 JitChain, \
                     got {} (nodes may not be fully jit_cuda or not data-linked) — skipped",
                    chains.len()
                );
                continue;
            }

            let chain = chains.into_iter().next().unwrap();
            let idx = dispatch.entries.len();
            let metadata = chain_metadata(&chain, idx as i32);

            if let Some(first) = nodes.first() {
                dispatch.by_entry.insert(first.clone(), idx);
            }
            for node in &nodes {
                dispatch.by_member.insert(node.clone(), idx);
            }

            dispatch.entries.push(FusionEntry {
                region: region_name.to_string(),
                nodes,
                chain,
                metadata,
                reads,
                writes,
            });
        }

        Ok(dispatch)
    }
}

// ── Dispatch API ──────────────────────────────────────────────────────────────

impl FusionDispatch {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Look up the fusion entry that **starts** at `node` (entry-point lookup).
    ///
    /// Use this to check if the runtime's next selected node is a fusion entry.
    pub fn get_by_entry(&self, node: &str) -> Option<&FusionEntry> {
        self.by_entry.get(node).map(|&i| &self.entries[i])
    }

    /// Look up the fusion entry that contains `node` anywhere in the chain.
    pub fn get_by_member(&self, node: &str) -> Option<&FusionEntry> {
        self.by_member.get(node).map(|&i| &self.entries[i])
    }

    /// Return true if `node` is the entry point of any loaded fusion region.
    pub fn is_fusion_entry(&self, node: &str) -> bool {
        self.by_entry.contains_key(node)
    }

    /// Return true if `node` is a member of any loaded fusion region.
    pub fn is_fusion_member(&self, node: &str) -> bool {
        self.by_member.contains_key(node)
    }
}

// ── Synthesis API ─────────────────────────────────────────────────────────────

impl FusionDispatch {
    /// Synthesize CUDA C kernel source for all loaded regions.
    ///
    /// Does not require a CUDA device.  Useful for startup logging and offline
    /// kernel inspection.  Regions that fail synthesis are skipped with a
    /// warning; successes are returned in load order.
    pub fn synthesize_all(&self, registry: &OperatorRegistry) -> Vec<SynthesizedKernel> {
        self.entries
            .iter()
            .filter_map(|e| match synthesize_kernel(&e.chain, registry) {
                Ok(source) => Some(SynthesizedKernel {
                    region: e.region.clone(),
                    source,
                }),
                Err(err) => {
                    eprintln!("[fusion_dispatch] synthesize '{}': {err}", e.region);
                    None
                }
            })
            .collect()
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn str_vec(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn analysis_registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        r.insert(
            "Analysis::Return1".to_string(),
            json!({
                "node_name": "Analysis::Return1",
                "executable": "jit_cuda",
                "jit_body":   "out[i] = (in0[i] - in1[i]) / (in1[i] + 1e-8f);",
                "effects": {
                    "reads":  ["market.price", "market.open"],
                    "writes": ["analysis.return"],
                    "locks":  []
                }
            }),
        );
        r.insert(
            "Analysis::Volatility".to_string(),
            json!({
                "node_name": "Analysis::Volatility",
                "executable": "jit_cuda",
                "jit_body":   "out[i] = fabsf(in0[i] - in1[i]) / (in1[i] + 1e-8f);",
                "effects": {
                    "reads":  ["market.price", "analysis.return"],
                    "writes": ["analysis.volatility"],
                    "locks":  []
                }
            }),
        );
        r.insert(
            "Analysis::SignalScore".to_string(),
            json!({
                "node_name": "Analysis::SignalScore",
                "executable": "jit_cuda",
                "jit_body":   "out[i] = in0[i] / (1.0f + fabsf(in1[i]));",
                "effects": {
                    "reads":  ["analysis.return", "analysis.volatility"],
                    "writes": ["analysis.signal_score"],
                    "locks":  []
                }
            }),
        );
        r
    }

    fn analysis_fusion_json() -> String {
        json!({
            "regions": [{
                "region":  "Analysis::Return1__Analysis::Volatility__Analysis::SignalScore",
                "backend": "cuda_jit",
                "fusion":  "linear_chain",
                "nodes":   ["Analysis::Return1", "Analysis::Volatility", "Analysis::SignalScore"],
                "reads":   ["market.open", "market.price"],
                "writes":  ["analysis.signal_score"],
                "locks":   [],
                "quantale": {
                    "compose": ["times", "plus", "min"],
                    "join":    ["max",   "min",  "max"]
                }
            }]
        })
        .to_string()
    }

    #[test]
    fn empty_json_produces_empty_dispatch() {
        let dispatch =
            FusionDispatch::from_json_str(r#"{"regions":[]}"#, &OperatorRegistry::new()).unwrap();
        assert!(dispatch.is_empty());
    }

    #[test]
    fn missing_regions_field_produces_empty_dispatch() {
        let dispatch = FusionDispatch::from_json_str("{}", &OperatorRegistry::new()).unwrap();
        assert!(dispatch.is_empty());
    }

    #[test]
    fn analysis_chain_loads_one_entry() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        assert_eq!(dispatch.len(), 1);
    }

    #[test]
    fn entry_node_lookup_finds_return1() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        let entry = dispatch
            .get_by_entry("Analysis::Return1")
            .expect("entry not found");
        assert_eq!(entry.nodes.len(), 3);
        assert_eq!(entry.nodes[0], "Analysis::Return1");
    }

    #[test]
    fn member_lookup_finds_any_chain_node() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        assert!(dispatch.get_by_member("Analysis::Return1").is_some());
        assert!(dispatch.get_by_member("Analysis::Volatility").is_some());
        assert!(dispatch.get_by_member("Analysis::SignalScore").is_some());
        assert!(dispatch.get_by_member("State::MarketFeed").is_none());
    }

    #[test]
    fn is_fusion_entry_and_member_predicates() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        assert!(dispatch.is_fusion_entry("Analysis::Return1"));
        assert!(!dispatch.is_fusion_entry("Analysis::Volatility")); // not the entry
        assert!(dispatch.is_fusion_member("Analysis::Volatility"));
        assert!(!dispatch.is_fusion_member("Control::Block"));
    }

    #[test]
    fn chain_has_correct_inputs_and_output() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        let entry = dispatch.get_by_entry("Analysis::Return1").unwrap();
        // market.price and market.open are external inputs (not produced inside).
        // analysis.return and analysis.volatility are internal (produced + consumed).
        assert!(
            entry.chain.inputs.contains(&"market.price".to_string()),
            "inputs={:?}",
            entry.chain.inputs
        );
        assert!(
            entry.chain.inputs.contains(&"market.open".to_string()),
            "inputs={:?}",
            entry.chain.inputs
        );
        assert!(
            !entry.chain.inputs.contains(&"analysis.return".to_string()),
            "internal slot must not be input"
        );
        assert_eq!(
            entry.chain.outputs,
            vec!["analysis.signal_score".to_string()]
        );
        assert!(
            entry
                .chain
                .internals
                .contains(&"analysis.return".to_string()),
            "internals={:?}",
            entry.chain.internals
        );
        assert!(
            entry
                .chain
                .internals
                .contains(&"analysis.volatility".to_string()),
            "internals={:?}",
            entry.chain.internals
        );
    }

    #[test]
    fn metadata_estimated_savings_is_positive() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        let meta = &dispatch.entries[0].metadata;
        assert!(
            meta.estimated_savings > 0.0,
            "savings={}",
            meta.estimated_savings
        );
        assert_eq!(meta.chain_len, 3);
    }

    #[test]
    fn synthesize_all_produces_cuda_source() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        let kernels = dispatch.synthesize_all(&reg);
        assert_eq!(kernels.len(), 1);
        let src = &kernels[0].source;
        assert!(src.contains("__global__ void"), "src={src}");
        assert!(src.contains("jit_fused"), "src={src}");
        assert!(src.contains("in0[i]"), "src={src}");
        assert!(src.contains("out0[i]"), "src={src}");
    }

    #[test]
    fn non_jit_cuda_region_is_skipped() {
        let reg = OperatorRegistry::new(); // empty — no jit_cuda entries
        let json = json!({
            "regions": [{
                "region": "X__Y",
                "nodes":  ["X", "Y"],
                "reads":  ["a"],
                "writes": ["b"],
                "locks":  []
            }]
        })
        .to_string();
        // Should not error — just skips and returns empty
        let dispatch = FusionDispatch::from_json_str(&json, &reg).unwrap();
        assert!(dispatch.is_empty());
    }

    #[test]
    fn singleton_region_is_skipped() {
        let reg = analysis_registry();
        let json = json!({
            "regions": [{
                "region": "Analysis::Return1",
                "nodes":  ["Analysis::Return1"],
                "reads":  ["market.price", "market.open"],
                "writes": ["analysis.return"],
                "locks":  []
            }]
        })
        .to_string();
        let dispatch = FusionDispatch::from_json_str(&json, &reg).unwrap();
        assert!(dispatch.is_empty());
    }

    #[test]
    fn slot_metadata_matches_fusion_json() {
        let reg = analysis_registry();
        let dispatch = FusionDispatch::from_json_str(&analysis_fusion_json(), &reg).unwrap();
        let entry = &dispatch.entries[0];
        let reads: std::collections::BTreeSet<&str> =
            entry.reads.iter().map(String::as_str).collect();
        assert!(reads.contains("market.open"), "reads={:?}", entry.reads);
        assert!(reads.contains("market.price"), "reads={:?}", entry.reads);
        assert_eq!(entry.writes, vec!["analysis.signal_score"]);
    }
}
