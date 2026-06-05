use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
pub(super) struct AssetFingerprint {
    entries: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
}

pub(super) struct RuntimeEpoch {
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    pub(super) id: usize,
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    pub(super) fingerprint: AssetFingerprint,
    pub(super) topology: TopologyRuntime,
    pub(super) executor: UniversalExecutor,
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    pub(super) contracts: NodeContracts,
    /// Accumulates static topology edges plus every LLM-proposed edge so that
    /// hard reset re-embeds the full learned set, not just the static baseline.
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    pub(super) accumulated_edges: Vec<quantale_semiring_v2::TensorEdge>,
    pub(super) world: TensorQuantaleWorld,
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    pub(super) exploration_engine: ExplorationEngine,
    /// Buffers successful execution edge deltas for persistence to
    /// `state/learned_edges.jsonl`.  Flushed on epoch reload and shutdown.
    pub(super) learning_buffer: quantale_semiring_v2::LearningBuffer,
    /// GPU-resident par group tuple table for the GPU-native parallel
    /// dispatch tier. Eligibility is computed on-device from per-member
    /// `is_gpu_dispatchable` flags. `None` when no par groups exist or when
    /// the world fails to upload (e.g. no CUDA device).
    /// Only active with `--features legacy-cpu-orchestration`.
    #[cfg(feature = "legacy-cpu-orchestration")]
    pub(super) par_group_data: Option<quantale_semiring_v2::ParGroupGpuData>,
    /// CPU-side metadata for each compiled par group, pre-resolved once per
    /// epoch so the par hot path does not repeatedly consult the topology
    /// registry or fusion-dispatch index after the GPU selects a group.
    /// Only active with `--features legacy-cpu-orchestration`.
    #[cfg(feature = "legacy-cpu-orchestration")]
    pub(super) par_group_host_plans: Vec<ParGroupHostPlan>,
}

#[cfg(feature = "legacy-cpu-orchestration")]
pub(super) struct ParGroupHostPlan {
    pub(super) node_names: Vec<String>,
    pub(super) fusion_entries: Vec<Option<quantale_semiring_v2::FusionEntry>>,
}

pub(super) fn build_runtime_epoch(
    id: usize,
    config: &mut SystemConfig,
    learning_policy: &LearningPolicy,
    tlog: &mut TlogWriter,
) -> Result<RuntimeEpoch, String> {
    // Const: flush learned edges every 10 successful executions.
    const LEARNING_FLUSH_THRESHOLD: usize = 10;
    config.reload_default_operator_registry()?;
    config.reload_hot_region_registry()?;

    let topology = TopologyRuntime::load_checked_default().map_err(|error| error.to_string())?;
    let invariants = TopologyInvariants::default_asset();
    let operator_violations =
        check_with_operators(&topology.document, &invariants, &config.operator_registry);
    let blocking_operator_violations: Vec<_> = operator_violations
        .into_iter()
        .filter(|violation| violation.kind == ViolationKind::MissingOperator)
        .collect();
    if !blocking_operator_violations.is_empty() {
        return Err(format!(
            "{}\n{} operator topology violation(s) found",
            format_violations(&blocking_operator_violations),
            blocking_operator_violations.len()
        ));
    }

    let executor = UniversalExecutor::from_config(config);
    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    let contracts = NodeContracts::default_asset();

    let mut tensor_edges = topology.tensor_edges().to_vec();
    tlog.append_tensor_edges(&format!("topology:tensor:epoch:{id}"), &tensor_edges)
        .map_err(|error| error.to_string())?;

    let learned_edges = load_learned_tensor_edges(
        &config.learned_edges_path,
        topology.registry(),
        topology.tensor_edges(),
        learning_policy,
    )
    .map_err(|error| error.to_string())?;
    if !learned_edges.is_empty() {
        tlog.append_tensor_edges(&format!("state:learned:epoch:{id}"), &learned_edges)
            .map_err(|error| error.to_string())?;
        tensor_edges.extend(learned_edges);
    }

    // Prefer pre-compiled pattern edges from build-overlay; fall back to
    // runtime CKA compilation when the file is absent (e.g. first run before
    // build-overlay has been executed).
    let pattern_edges =
        match load_compiled_pattern_edges("assets/patterns.compiled.json", topology.registry())
            .map_err(|e| e.to_string())?
        {
            Some(edges) => edges,
            None => {
                let patterns = load_default_patterns().map_err(|error| error.to_string())?;
                let mut edges = Vec::new();
                for pattern in &patterns.patterns {
                    let compiled =
                        compile_pattern(pattern, &topology.compiled, &config.operator_registry)
                            .map_err(|error| error.to_string())?;
                    edges.extend(compiled.edges);
                }
                edges
            }
        };
    tlog.append_tensor_edges(&format!("pattern:cka:epoch:{id}"), &pattern_edges)
        .map_err(|error| error.to_string())?;
    tensor_edges.extend(pattern_edges);

    let world =
        TensorQuantaleWorld::from_tensor_edges(&tensor_edges).map_err(|error| error.to_string())?;

    #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
    let exploration_engine = {
        let exploration_config =
            ExplorationConfig::default_asset().map_err(|error| error.to_string())?;
        ExplorationEngine::new(
            exploration_config,
            &topology.document,
            config.operator_registry.clone(),
        )
        .map_err(|error| error.to_string())?
    };

    if config.fusion_dispatch.is_empty() {
        console::info("fusion", "no_regions", &[("epoch", id.to_string())]);
    } else {
        console::info(
            "fusion",
            "regions_loaded",
            &[
                ("epoch", id.to_string()),
                ("count", config.fusion_dispatch.len().to_string()),
            ],
        );
        for entry in &config.fusion_dispatch.entries {
            console::info(
                "fusion",
                "region",
                &[
                    ("region", entry.region.clone()),
                    ("chain_len", entry.metadata.chain_len.to_string()),
                    ("inputs", entry.chain.inputs.len().to_string()),
                    ("outputs", entry.chain.outputs.len().to_string()),
                    (
                        "estimated_savings",
                        format!("{:.1}", entry.metadata.estimated_savings),
                    ),
                ],
            );
        }
        for kernel in config
            .fusion_dispatch
            .synthesize_all(&config.operator_registry)
        {
            console::info(
                "fusion",
                "kernel_synthesized",
                &[
                    ("region", kernel.region.clone()),
                    ("lines", kernel.source.lines().count().to_string()),
                ],
            );
        }
    }

    // Build GPU-resident par group data.
    // Only compiled when `--features legacy-cpu-orchestration` is active;
    // the default path uses the GPU-native orchestration loop instead.
    #[cfg(feature = "legacy-cpu-orchestration")]
    let (par_group_data, par_group_host_plans) = {
        let par_region_ids: Vec<Vec<i32>> = topology
            .parallel_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&id| {
                        topology
                            .registry()
                            .name_of(id as usize)
                            .map(|name| {
                                par_member_region_id(
                                    name,
                                    &config.fusion_dispatch,
                                    &config.hot_region_registry,
                                    &config.fusion_hf_coverage,
                                )
                            })
                            .unwrap_or(-1)
                    })
                    .collect()
            })
            .collect();
        let par_dispatch_kinds: Vec<Vec<i32>> = topology
            .parallel_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&id| {
                        topology
                            .registry()
                            .name_of(id as usize)
                            .map(|name| {
                                classify_par_dispatch_kind(
                                    name,
                                    &config.fusion_dispatch,
                                    &config.hot_region_registry,
                                    &config.fusion_hf_coverage,
                                    &config.abstract_device_coverage,
                                )
                            })
                            .unwrap_or(quantale_semiring_v2::PAR_DISPATCH_HOST_FALLBACK)
                    })
                    .collect()
            })
            .collect();
        let par_is_dispatchable: Vec<Vec<bool>> = par_dispatch_kinds
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&kind| par_dispatch_kind_is_gpu_native(kind))
                    .collect()
            })
            .collect();
        let par_group_host_plans: Vec<ParGroupHostPlan> = topology
            .parallel_groups
            .iter()
            .map(|group| {
                let node_names: Vec<String> = group
                    .iter()
                    .filter_map(|&id| topology.registry().name_of(id as usize).map(str::to_string))
                    .collect();
                let fusion_entries = node_names
                    .iter()
                    .map(|name| config.fusion_dispatch.get_by_entry(name).cloned())
                    .collect();
                ParGroupHostPlan {
                    node_names,
                    fusion_entries,
                }
            })
            .collect();
        // Build DeviceSlotRegistry for par-group H_f hot-region dispatch.
        let par_slot_registry =
            build_par_slot_registry(&world, &par_region_ids, &config.fusion_hf_coverage);
        let par_group_data = world
            .make_par_group_data(
                &topology.parallel_groups,
                &par_region_ids,
                &par_is_dispatchable,
                &par_dispatch_kinds,
                par_slot_registry.as_ref(),
                Some(&config.fusion_hf_coverage),
            )
            .ok();
        (par_group_data, par_group_host_plans)
    };

    Ok(RuntimeEpoch {
        #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
        id,
        #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
        fingerprint: current_asset_fingerprint(),
        topology,
        executor,
        #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
        contracts,
        #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
        accumulated_edges: tensor_edges,
        world,
        #[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
        exploration_engine,
        learning_buffer: quantale_semiring_v2::LearningBuffer::new(
            &config.learned_edges_path,
            LEARNING_FLUSH_THRESHOLD,
        ),
        #[cfg(feature = "legacy-cpu-orchestration")]
        par_group_data,
        #[cfg(feature = "legacy-cpu-orchestration")]
        par_group_host_plans,
    })
}

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
pub(super) fn changed_asset_fingerprint(previous: &AssetFingerprint) -> Option<AssetFingerprint> {
    let current = current_asset_fingerprint();
    (current != *previous).then_some(current)
}

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
fn current_asset_fingerprint() -> AssetFingerprint {
    AssetFingerprint {
        entries: watched_asset_paths()
            .into_iter()
            .map(|path| {
                let metadata = std::fs::metadata(&path).ok();
                let stamp = metadata.and_then(|metadata| {
                    metadata
                        .modified()
                        .ok()
                        .map(|modified| (modified, metadata.len()))
                });
                (path, stamp)
            })
            .collect(),
    }
}

#[cfg(any(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
fn watched_asset_paths() -> Vec<PathBuf> {
    ReloadPolicy::default_asset()
        .map(|policy| policy.watched_asset_paths)
        .unwrap_or_else(|error| {
            console::warn(
                "reload",
                "policy_unavailable",
                &[("error", error.to_string())],
            );
            Vec::new()
        })
}

/// Build a `DeviceSlotRegistry` with zero-initialised device buffers for every
/// unique slot name referenced by hot-region par-group members.
///
/// These persistent buffers are the `float**` tables the Phase 2 H_f dispatch
/// path writes into and reads from.  Returned as `None` when no CUDA device is
/// available or when no hot-region slots are referenced.

#[cfg(feature = "legacy-cpu-orchestration")]
fn par_member_region_id(
    node_name: &str,
    fusion_dispatch: &quantale_semiring_v2::FusionDispatch,
    hot_region_registry: &quantale_semiring_v2::HotRegionRegistry,
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> i32 {
    if fusion_dispatch.is_fusion_entry(node_name) {
        return lowered_fusion_hot_region_id(
            node_name,
            fusion_dispatch,
            hot_region_registry,
            fusion_hf_coverage,
        )
        .map(|region_id| region_id as i32)
        .unwrap_or(-1);
    }

    hot_region_registry
        .region_id_for(node_name)
        .map(|region_id| region_id as i32)
        .unwrap_or(-1)
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn lowered_fusion_hot_region_id(
    node_name: &str,
    fusion_dispatch: &quantale_semiring_v2::FusionDispatch,
    hot_region_registry: &quantale_semiring_v2::HotRegionRegistry,
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> Option<u32> {
    let entry = fusion_dispatch.get_by_entry(node_name)?;

    if let Some(region_id) = fusion_hf_coverage.region_id(&entry.region) {
        if has_in_kernel_hf_handler(region_id, fusion_hf_coverage) {
            return Some(region_id as u32);
        }
    }

    let fusion_reads = string_set(&entry.reads);
    let fusion_writes = string_set(&entry.writes);

    hot_region_registry
        .entries
        .iter()
        .find(|hot| {
            hot.kind == "gpu_region"
                && hot.kernel == "jit_fused"
                && hot.pure
                && !entry.nodes.iter().any(|member| member == &hot.name)
                && string_set(&hot.reads) == fusion_reads
                && string_set(&hot.writes) == fusion_writes
                && has_in_kernel_hf_handler(hot.region_id as i32, fusion_hf_coverage)
        })
        .map(|hot| hot.region_id)
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn string_set(values: &[String]) -> BTreeSet<&str> {
    values.iter().map(String::as_str).collect()
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn classify_par_dispatch_kind(
    node_name: &str,
    fusion_dispatch: &quantale_semiring_v2::FusionDispatch,
    hot_region_registry: &quantale_semiring_v2::HotRegionRegistry,
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
    abstract_device_coverage: &quantale_semiring_v2::AbstractDeviceCoverage,
) -> i32 {
    if par_member_hf_region_id(
        node_name,
        fusion_dispatch,
        hot_region_registry,
        fusion_hf_coverage,
    )
    .is_some()
    {
        return quantale_semiring_v2::PAR_DISPATCH_HF_DEVICE;
    }

    if abstract_device_coverage.is_covered(node_name) {
        return quantale_semiring_v2::PAR_DISPATCH_ABSTRACT_DEVICE;
    }

    if fusion_dispatch.is_fusion_entry(node_name) {
        quantale_semiring_v2::PAR_DISPATCH_FUSION_ENTRY
    } else {
        quantale_semiring_v2::PAR_DISPATCH_HOST_FALLBACK
    }
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn par_dispatch_kind_is_gpu_native(kind: i32) -> bool {
    kind == quantale_semiring_v2::PAR_DISPATCH_HF_DEVICE
        || kind == quantale_semiring_v2::PAR_DISPATCH_ABSTRACT_DEVICE
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn par_member_hf_region_id(
    node_name: &str,
    fusion_dispatch: &quantale_semiring_v2::FusionDispatch,
    hot_region_registry: &quantale_semiring_v2::HotRegionRegistry,
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> Option<i32> {
    let region_id = par_member_region_id(
        node_name,
        fusion_dispatch,
        hot_region_registry,
        fusion_hf_coverage,
    );
    if region_id >= 0 && has_in_kernel_hf_handler(region_id, fusion_hf_coverage) {
        Some(region_id)
    } else {
        None
    }
}

#[cfg(feature = "legacy-cpu-orchestration")]
fn has_in_kernel_hf_handler(
    region_id: i32,
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> bool {
    region_id >= 0 && fusion_hf_coverage.has_handler_for_region_id(region_id)
}

#[cfg(all(feature = "cuda", feature = "legacy-cpu-orchestration"))]
fn build_par_slot_registry(
    world: &TensorQuantaleWorld,
    par_region_ids: &[Vec<i32>],
    fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> Option<quantale_semiring_v2::DeviceSlotRegistry> {
    use quantale_semiring_v2::{gpu_region_slots, DeviceSlotRegistry, DEFAULT_PAR_SLOT_ELEMENTS};

    // Collect unique slot names from all hot-region par members.
    let mut names: BTreeSet<String> = BTreeSet::new();
    for rids in par_region_ids {
        for &rid in rids {
            if rid < 0 {
                continue;
            }
            if let Some(slots) = gpu_region_slots(rid) {
                for &s in slots {
                    names.insert(s.to_string());
                }
            } else if let Some(slots) = fusion_hf_coverage.slots_for_region_id(rid) {
                for s in slots {
                    names.insert(s.clone());
                }
            }
        }
    }
    if names.is_empty() {
        return None;
    }

    let dev = world.device().clone();
    let mut registry = DeviceSlotRegistry::new();
    for name in names {
        match dev.alloc_zeros::<f32>(DEFAULT_PAR_SLOT_ELEMENTS) {
            Ok(buf) => {
                registry.insert(name, buf);
            }
            Err(_) => {
                return None;
            }
        }
    }
    Some(registry)
}

#[cfg(all(not(feature = "cuda"), feature = "legacy-cpu-orchestration"))]
fn build_par_slot_registry(
    _world: &TensorQuantaleWorld,
    _par_region_ids: &[Vec<i32>],
    _fusion_hf_coverage: &quantale_semiring_v2::FusionHfCoverage,
) -> Option<quantale_semiring_v2::DeviceSlotRegistry> {
    None
}

#[cfg(all(test, feature = "legacy-cpu-orchestration"))]
mod tests {
    use super::*;
    use quantale_semiring_v2::{
        AbstractDeviceCoverage, FusionDispatch, FusionHfCoverage, HotRegionRegistry,
        PAR_DISPATCH_ABSTRACT_DEVICE, PAR_DISPATCH_FUSION_ENTRY, PAR_DISPATCH_HF_DEVICE,
        PAR_DISPATCH_HOST_FALLBACK,
    };
    use serde_json::json;
    use std::collections::HashMap;

    fn fusion_dispatch() -> FusionDispatch {
        let registry = HashMap::from([
            (
                "Execution::VectorAdd".to_string(),
                json!({
                    "node_name": "Execution::VectorAdd",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = in0[i] + in1[i];",
                    "effects": {
                        "reads": ["math.a", "math.b"],
                        "writes": ["math.add_out"],
                        "locks": []
                    }
                }),
            ),
            (
                "Execution::VectorScale".to_string(),
                json!({
                    "node_name": "Execution::VectorScale",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = in0[i] * in1[i];",
                    "effects": {
                        "reads": ["math.add_out", "math.scale"],
                        "writes": ["math.out"],
                        "locks": []
                    }
                }),
            ),
            (
                "Analysis::Return1".to_string(),
                json!({
                    "node_name": "Analysis::Return1",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = (in0[i] - in1[i]) / (in1[i] + 1e-8f);",
                    "effects": {
                        "reads": ["market.price", "market.open"],
                        "writes": ["analysis.return"],
                        "locks": []
                    }
                }),
            ),
            (
                "Analysis::Volatility".to_string(),
                json!({
                    "node_name": "Analysis::Volatility",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = fabsf(in0[i] - in1[i]) / (in1[i] + 1e-8f);",
                    "effects": {
                        "reads": ["market.price", "analysis.return"],
                        "writes": ["analysis.volatility"],
                        "locks": []
                    }
                }),
            ),
            (
                "Analysis::SignalScore".to_string(),
                json!({
                    "node_name": "Analysis::SignalScore",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = in0[i] / (1.0f + fabsf(in1[i]));",
                    "effects": {
                        "reads": ["analysis.return", "analysis.volatility"],
                        "writes": ["analysis.signal_score"],
                        "locks": []
                    }
                }),
            ),
        ]);
        FusionDispatch::from_json_str(
            &json!({
                "regions": [
                    {
                        "region": "Execution::VectorAdd__Execution::VectorScale",
                        "nodes": ["Execution::VectorAdd", "Execution::VectorScale"],
                        "reads": ["math.a", "math.b", "math.scale"],
                        "writes": ["math.out"]
                    },
                    {
                        "region": "Analysis::Return1__Analysis::Volatility__Analysis::SignalScore",
                        "nodes": ["Analysis::Return1", "Analysis::Volatility", "Analysis::SignalScore"],
                        "reads": ["market.open", "market.price"],
                        "writes": ["analysis.signal_score"]
                    }
                ]
            })
            .to_string(),
            &registry,
        )
        .unwrap()
    }

    fn fusion_hf_coverage() -> FusionHfCoverage {
        FusionHfCoverage::from_static_table()
    }

    fn empty_fusion_hf_coverage() -> FusionHfCoverage {
        FusionHfCoverage::default()
    }

    fn empty_abstract_device_coverage() -> AbstractDeviceCoverage {
        AbstractDeviceCoverage::default()
    }

    fn abstract_device_coverage() -> AbstractDeviceCoverage {
        AbstractDeviceCoverage::from_json_str(
            r#"{
                "schema":"abstract_device_coverage.v1",
                "nodes":[{"node":"Control::Allow", "covered":true, "reason":"noop_marker_device_receipt"}]
            }"#,
        )
        .unwrap()
    }

    fn unsupported_fusion_dispatch() -> FusionDispatch {
        let registry = HashMap::from([
            (
                "Unsupported::A".to_string(),
                json!({
                    "node_name": "Unsupported::A",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = in0[i] + 1.0f;",
                    "effects": {
                        "reads": ["unsupported.in"],
                        "writes": ["unsupported.mid"],
                        "locks": []
                    }
                }),
            ),
            (
                "Unsupported::B".to_string(),
                json!({
                    "node_name": "Unsupported::B",
                    "executable": "jit_cuda",
                    "jit_body": "out[i] = in0[i] * 2.0f;",
                    "effects": {
                        "reads": ["unsupported.mid"],
                        "writes": ["unsupported.out"],
                        "locks": []
                    }
                }),
            ),
        ]);
        FusionDispatch::from_json_str(
            &json!({
                "regions": [
                    {
                        "region": "Unsupported::A__Unsupported::B",
                        "nodes": ["Unsupported::A", "Unsupported::B"],
                        "reads": ["unsupported.in"],
                        "writes": ["unsupported.out"]
                    }
                ]
            })
            .to_string(),
            &registry,
        )
        .unwrap()
    }

    fn hot_region_registry() -> HotRegionRegistry {
        HotRegionRegistry::from_json_str(
            &json!({
                "regions": [
                    {
                        "region_id": 0,
                        "name": "Execution::VectorAdd",
                        "kind": "gpu_region",
                        "reads": ["math.a", "math.b"],
                        "writes": ["math.add_out"],
                        "kernel": "jit_fused",
                        "pure": true
                    },
                    {
                        "region_id": 1,
                        "name": "Execution::VectorScale",
                        "kind": "gpu_region",
                        "reads": ["math.add_out", "math.scale"],
                        "writes": ["math.out"],
                        "kernel": "jit_fused",
                        "pure": true
                    },
                    {
                        "region_id": 2,
                        "name": "Execution::FusedVectorAddScale",
                        "kind": "gpu_region",
                        "reads": ["math.a", "math.b", "math.scale"],
                        "writes": ["math.out"],
                        "kernel": "jit_fused",
                        "pure": true
                    },
                    {
                        "region_id": 3,
                        "name": "Analysis::Return1",
                        "kind": "gpu_region",
                        "reads": ["market.price", "market.open"],
                        "writes": ["analysis.return"],
                        "kernel": "jit_fused",
                        "pure": true
                    },
                    {
                        "region_id": 7,
                        "name": "Analysis::FusedReturnVolatilitySignalScore",
                        "kind": "gpu_region",
                        "reads": ["market.open", "market.price"],
                        "writes": ["analysis.signal_score"],
                        "kernel": "jit_fused",
                        "pure": true
                    }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn fusion_entry_with_exact_fused_hot_region_lowers_to_that_region() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            par_member_region_id("Execution::VectorAdd", &fusion, &hot, &fusion_hf_coverage()),
            2
        );
    }

    #[test]
    fn analysis_fusion_entry_with_exact_fused_hot_region_lowers_to_region() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            par_member_region_id("Analysis::Return1", &fusion, &hot, &fusion_hf_coverage()),
            7
        );
    }

    #[test]
    fn lowerable_fusion_entry_classifies_as_hf_device_dispatch() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            classify_par_dispatch_kind(
                "Analysis::Return1",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &empty_abstract_device_coverage()
            ),
            PAR_DISPATCH_HF_DEVICE
        );
    }

    #[test]
    fn explicitly_supported_fusion_entry_still_lowers_without_hot_metadata() {
        let fusion = fusion_dispatch();
        let mut hot = hot_region_registry();
        hot.entries
            .retain(|entry| entry.name != "Analysis::FusedReturnVolatilitySignalScore");

        assert_eq!(
            par_member_region_id("Analysis::Return1", &fusion, &hot, &fusion_hf_coverage()),
            7
        );
        assert_eq!(
            classify_par_dispatch_kind(
                "Analysis::Return1",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &empty_abstract_device_coverage()
            ),
            PAR_DISPATCH_HF_DEVICE
        );
    }

    #[test]
    fn signature_matching_still_lowers_when_manifest_has_no_entry() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            par_member_region_id(
                "Execution::VectorAdd",
                &fusion,
                &hot,
                &empty_fusion_hf_coverage()
            ),
            2
        );
    }

    #[test]
    fn supported_hot_member_classifies_as_hf_device_dispatch() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            classify_par_dispatch_kind(
                "Execution::VectorScale",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &empty_abstract_device_coverage()
            ),
            PAR_DISPATCH_HF_DEVICE
        );
    }

    #[test]
    fn ordinary_non_fusion_member_keeps_host_fallback_dispatch_kind() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            classify_par_dispatch_kind(
                "State::Parse",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &empty_abstract_device_coverage()
            ),
            PAR_DISPATCH_HOST_FALLBACK
        );
    }

    #[test]
    fn lowerable_hf_dispatch_kind_is_gpu_native() {
        assert!(par_dispatch_kind_is_gpu_native(PAR_DISPATCH_HF_DEVICE));
    }

    #[test]
    fn host_bound_fusion_dispatch_kind_is_not_gpu_native() {
        assert!(!par_dispatch_kind_is_gpu_native(PAR_DISPATCH_FUSION_ENTRY));
    }

    #[test]
    fn host_fallback_dispatch_kind_is_not_gpu_native() {
        assert!(!par_dispatch_kind_is_gpu_native(PAR_DISPATCH_HOST_FALLBACK));
    }

    #[test]
    fn covered_abstract_marker_classifies_as_abstract_device() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            classify_par_dispatch_kind(
                "Control::Allow",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &abstract_device_coverage()
            ),
            PAR_DISPATCH_ABSTRACT_DEVICE
        );
    }

    #[test]
    fn abstract_device_dispatch_kind_is_reserved_gpu_native_scaffold() {
        assert!(par_dispatch_kind_is_gpu_native(
            PAR_DISPATCH_ABSTRACT_DEVICE
        ));
    }

    #[test]
    fn unsupported_fusion_entry_does_not_inherit_first_hot_member_region() {
        let fusion = unsupported_fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            par_member_region_id("Unsupported::A", &fusion, &hot, &fusion_hf_coverage()),
            -1
        );
        assert_eq!(
            classify_par_dispatch_kind(
                "Unsupported::A",
                &fusion,
                &hot,
                &fusion_hf_coverage(),
                &empty_abstract_device_coverage()
            ),
            PAR_DISPATCH_FUSION_ENTRY
        );
    }

    #[test]
    fn non_fusion_hot_member_keeps_own_hot_region() {
        let fusion = fusion_dispatch();
        let hot = hot_region_registry();

        assert_eq!(
            par_member_region_id(
                "Execution::VectorScale",
                &fusion,
                &hot,
                &fusion_hf_coverage()
            ),
            1
        );
    }
}
