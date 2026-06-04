//! Hot-region metadata registry.
//!
//! Loaded from `assets/regions.hot.json` at startup. Maps node names to GPU
//! region descriptors so the main loop can route hot nodes away from the
//! CPU operator path without looking up the operator registry.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

// ── Region entry ──────────────────────────────────────────────────────────────

/// Metadata for a single GPU-resident compute region.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct HotRegionEntry {
    pub region_id: u32,
    pub name: String,
    /// Always "gpu_region" for hot regions.
    pub kind: String,
    /// External device-slot names that this region reads.
    pub reads: Vec<String>,
    /// Device-slot names this region writes.
    pub writes: Vec<String>,
    /// Kernel family: "jit_fused", "static", etc.
    pub kernel: String,
    /// True when the region has no side-effects beyond its declared writes.
    #[serde(default)]
    pub pure: bool,
}

// ── Registry ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HotRegionRegistry {
    by_name: HashMap<String, usize>,
    pub entries: Vec<HotRegionEntry>,
}

#[derive(Deserialize)]
struct RegionsFile {
    regions: Vec<HotRegionEntry>,
}

impl HotRegionRegistry {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let raw = fs::read_to_string(path.as_ref()).map_err(|e| {
            format!("read regions file '{}': {e}", path.as_ref().display())
        })?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let file: RegionsFile =
            serde_json::from_str(json).map_err(|e| format!("parse regions.hot.json: {e}"))?;
        let mut registry = Self::default();
        for entry in file.regions {
            let idx = registry.entries.len();
            registry.by_name.insert(entry.name.clone(), idx);
            registry.entries.push(entry);
        }
        Ok(registry)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&HotRegionEntry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// True if `node_name` is a registered hot GPU region.
    pub fn is_hot(&self, node_name: &str) -> bool {
        self.by_name.contains_key(node_name)
    }

    pub fn region_id_for(&self, node_name: &str) -> Option<u32> {
        self.by_name
            .get(node_name)
            .map(|&i| self.entries[i].region_id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
