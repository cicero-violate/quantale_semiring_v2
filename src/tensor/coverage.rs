use std::collections::BTreeMap;
use std::path::Path;

use super::GPU_HOT_REGION_COUNT;
use crate::error::CudaError;

const REGION_VECTOR_ADD_SLOTS: &[&str] = &["math.a", "math.b", "math.add_out"];
const REGION_VECTOR_SCALE_SLOTS: &[&str] = &["math.add_out", "math.scale", "math.out"];
const REGION_FUSED_ADD_SCALE_SLOTS: &[&str] = &["math.a", "math.b", "math.scale", "math.out"];
const REGION_ANALYSIS_RETURN1_SLOTS: &[&str] = &["market.price", "market.open", "analysis.return"];
const REGION_ANALYSIS_VOLATILITY_SLOTS: &[&str] =
    &["market.price", "analysis.return", "analysis.volatility"];
const REGION_ANALYSIS_SIGNAL_SCORE_SLOTS: &[&str] = &[
    "analysis.return",
    "analysis.volatility",
    "analysis.signal_score",
];
const REGION_ANALYSIS_FUSED_SIGNAL_SCORE_SLOTS: &[&str] =
    &["market.price", "market.open", "analysis.signal_score"];
const REGION_COMMIT_RECEIPT_SLOTS: &[&str] = &[];

/// Default number of f32 elements allocated per slot when building epoch-start
/// par-group slot tables.  Slots are zero-initialised; real data flows in via
/// hot-region operators once the epoch starts running.
pub const DEFAULT_PAR_SLOT_ELEMENTS: usize = 256;

pub fn gpu_region_slots(region_id: i32) -> Option<&'static [&'static str]> {
    match region_id {
        0 => Some(REGION_VECTOR_ADD_SLOTS),
        1 => Some(REGION_VECTOR_SCALE_SLOTS),
        2 => Some(REGION_FUSED_ADD_SCALE_SLOTS),
        3 => Some(REGION_ANALYSIS_RETURN1_SLOTS),
        4 => Some(REGION_ANALYSIS_VOLATILITY_SLOTS),
        5 => Some(REGION_ANALYSIS_SIGNAL_SCORE_SLOTS),
        6 => Some(REGION_COMMIT_RECEIPT_SLOTS),
        7 => Some(REGION_ANALYSIS_FUSED_SIGNAL_SCORE_SLOTS),
        _ => None,
    }
}

/// Static fusion-region lowering table for fusion batches that already have an
/// in-kernel H_f handler in `cuda/quantale_world.cu`.
///
/// This table is intentionally keyed by the generated fusion region name rather
/// than by hot-region metadata.  It lets the par tier lower known fusion batches
/// directly into `tensor_quantale_par_group_step` even when a matching fused
/// entry is absent from `regions.hot.json`; metadata signature matching remains
/// a compatibility fallback in `runtime_epoch.rs`.
pub fn fusion_hf_region_id(region_name: &str) -> Option<i32> {
    match region_name {
        "Execution::VectorAdd__Execution::VectorScale" => Some(2),
        "Analysis::Return1__Analysis::Volatility__Analysis::SignalScore" => Some(7),
        _ => None,
    }
}

pub fn static_hf_symbol(region_id: i32) -> Option<&'static str> {
    match region_id {
        0 => Some("region_vector_add"),
        1 => Some("region_vector_scale"),
        2 => Some("region_fused_add_scale"),
        3 => Some("region_analysis_return1"),
        4 => Some("region_analysis_volatility"),
        5 => Some("region_analysis_signal_score"),
        6 => Some("region_commit_receipt"),
        7 => Some("region_analysis_fused_signal_score"),
        _ => None,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FusionHfCoverage {
    entries: BTreeMap<String, FusionHfCoverageEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionHfCoverageEntry {
    pub region: String,
    pub entry: String,
    pub nodes: Vec<String>,
    pub hf_region_id: Option<i32>,
    pub covered: bool,
    pub reason: String,
    pub symbol: String,
    pub slots: Vec<String>,
}

impl FusionHfCoverage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::from_static_table());
        }
        let content = std::fs::read_to_string(path).map_err(|e| {
            CudaError::invalid_input(format!(
                "read fusion H_f coverage '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_json_str(&content)
    }

    pub fn from_static_table() -> Self {
        let mut entries = BTreeMap::new();
        for (region, hf_region_id) in [
            ("Execution::VectorAdd__Execution::VectorScale", 2),
            (
                "Analysis::Return1__Analysis::Volatility__Analysis::SignalScore",
                7,
            ),
        ] {
            entries.insert(
                region.to_string(),
                FusionHfCoverageEntry {
                    region: region.to_string(),
                    entry: String::new(),
                    nodes: Vec::new(),
                    hf_region_id: Some(hf_region_id),
                    covered: true,
                    reason: "static_hf_handler".to_string(),
                    symbol: static_hf_symbol(hf_region_id)
                        .unwrap_or_default()
                        .to_string(),
                    slots: gpu_region_slots(hf_region_id)
                        .unwrap_or(&[])
                        .iter()
                        .map(|slot| (*slot).to_string())
                        .collect(),
                },
            );
        }
        Self { entries }
    }

    pub fn from_json_str(input: &str) -> Result<Self, CudaError> {
        let value: serde_json::Value = serde_json::from_str(input)
            .map_err(|e| CudaError::invalid_input(format!("parse fusion H_f coverage: {e}")))?;
        let regions = value
            .get("regions")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                CudaError::invalid_input("fusion H_f coverage missing 'regions' array")
            })?;
        let mut entries = BTreeMap::new();
        for (idx, item) in regions.iter().enumerate() {
            let region = item
                .get("region")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    CudaError::invalid_input(format!(
                        "fusion H_f coverage region {idx}: missing region"
                    ))
                })?
                .to_string();
            let entry = item
                .get("entry")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let nodes = item
                .get("nodes")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let hf_region_id = match item.get("hf_region_id") {
                Some(v) if v.is_null() => None,
                Some(v) => Some(v.as_i64().ok_or_else(|| {
                    CudaError::invalid_input(format!(
                        "fusion H_f coverage region '{region}': hf_region_id must be integer or null"
                    ))
                })? as i32),
                None => None,
            };
            let covered = item
                .get("covered")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(hf_region_id.is_some());
            let reason = item
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let symbol = item
                .get("symbol")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let slots = item
                .get("slots")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if covered && hf_region_id.is_none() {
                return Err(CudaError::invalid_input(format!(
                    "fusion H_f coverage region '{region}' is covered but has no hf_region_id"
                )));
            }
            if let Some(region_id) = hf_region_id {
                if region_id < GPU_HOT_REGION_COUNT {
                    if fusion_hf_region_id(&region) != Some(region_id) {
                        return Err(CudaError::invalid_input(format!(
                            "fusion H_f coverage region '{region}' maps to static handler {region_id}, but compiled static table has {:?}",
                            fusion_hf_region_id(&region)
                        )));
                    }
                } else if covered && symbol.is_empty() {
                    return Err(CudaError::invalid_input(format!(
                        "generated fusion H_f coverage region '{region}' is covered but has no symbol"
                    )));
                }
            }
            entries.insert(
                region.clone(),
                FusionHfCoverageEntry {
                    region,
                    entry,
                    nodes,
                    hf_region_id,
                    covered,
                    reason,
                    symbol,
                    slots,
                },
            );
        }
        Ok(Self { entries })
    }

    pub fn region_id(&self, region_name: &str) -> Option<i32> {
        self.entries
            .get(region_name)
            .filter(|entry| entry.covered)
            .and_then(|entry| entry.hf_region_id)
    }

    pub fn slots_for_region_id(&self, region_id: i32) -> Option<&[String]> {
        self.entries
            .values()
            .find(|entry| entry.covered && entry.hf_region_id == Some(region_id))
            .map(|entry| entry.slots.as_slice())
    }

    pub fn has_handler_for_region_id(&self, region_id: i32) -> bool {
        if region_id >= 0 && region_id < GPU_HOT_REGION_COUNT {
            return gpu_region_slots(region_id).is_some();
        }
        self.entries.values().any(|entry| {
            entry.covered
                && entry.hf_region_id == Some(region_id)
                && !entry.symbol.is_empty()
                && !entry.slots.is_empty()
        })
    }

    pub fn region_count(&self) -> i32 {
        self.entries
            .values()
            .filter_map(|entry| entry.hf_region_id)
            .max()
            .map(|max_id| (max_id + 1).max(GPU_HOT_REGION_COUNT))
            .unwrap_or(GPU_HOT_REGION_COUNT)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AbstractDeviceCoverage {
    nodes: BTreeMap<String, AbstractDeviceCoverageEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractDeviceCoverageEntry {
    pub node: String,
    pub covered: bool,
    pub reason: String,
}

impl AbstractDeviceCoverage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path).map_err(|e| {
            CudaError::invalid_input(format!(
                "read abstract-device coverage '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(input: &str) -> Result<Self, CudaError> {
        let value: serde_json::Value = serde_json::from_str(input).map_err(|e| {
            CudaError::invalid_input(format!("parse abstract-device coverage: {e}"))
        })?;
        let nodes = value
            .get("nodes")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                CudaError::invalid_input("abstract-device coverage missing 'nodes' array")
            })?;
        let mut entries = BTreeMap::new();
        for (idx, item) in nodes.iter().enumerate() {
            let node = item
                .get("node")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    CudaError::invalid_input(format!(
                        "abstract-device coverage node {idx}: missing node"
                    ))
                })?
                .to_string();
            let covered = item
                .get("covered")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let reason = item
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            entries.insert(
                node.clone(),
                AbstractDeviceCoverageEntry {
                    node,
                    covered,
                    reason,
                },
            );
        }
        Ok(Self { nodes: entries })
    }

    pub fn is_covered(&self, node_name: &str) -> bool {
        self.nodes
            .get(node_name)
            .map(|entry| entry.covered)
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
