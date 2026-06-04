use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AssetFingerprint {
    entries: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
}

pub(super) struct RuntimeEpoch {
    pub(super) id: usize,
    pub(super) fingerprint: AssetFingerprint,
    pub(super) topology: TopologyRuntime,
    pub(super) executor: UniversalExecutor,
    pub(super) contracts: NodeContracts,
    /// Accumulates static topology edges plus every LLM-proposed edge so that
    /// hard reset re-embeds the full learned set, not just the static baseline.
    pub(super) accumulated_edges: Vec<quantale_semiring_v2::TensorEdge>,
    pub(super) world: TensorQuantaleWorld,
    pub(super) exploration_engine: ExplorationEngine,
    /// Buffers successful execution edge deltas for persistence to
    /// `state/learned_edges.jsonl`.  Flushed on epoch reload and shutdown.
    pub(super) learning_buffer: quantale_semiring_v2::LearningBuffer,
    /// GPU-resident par group table and eligibility mask for the GPU-native
    /// parallel dispatch tier.  `None` when no par groups exist or when the
    /// world fails to upload (e.g. no CUDA device).
    pub(super) par_group_data: Option<quantale_semiring_v2::ParGroupGpuData>,
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
    let pattern_edges = match load_compiled_pattern_edges(
        "assets/patterns.compiled.json",
        topology.registry(),
    )
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

    let exploration_config =
        ExplorationConfig::default_asset().map_err(|error| error.to_string())?;
    let exploration_engine = ExplorationEngine::new(
        exploration_config,
        &topology.document,
        config.operator_registry.clone(),
    )
    .map_err(|error| error.to_string())?;

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

    // Build GPU-resident par group data.  A group is eligible only when every
    // member operator is GPU-executable (jit_cuda / fusion entry / hot region).
    let par_eligible: Vec<bool> = topology
        .parallel_groups
        .iter()
        .map(|group| {
            group.iter().all(|&id| {
                let Some(name) = topology.registry().name_of(id as usize) else {
                    return false;
                };
                executor.is_hot_node(name) || config.fusion_dispatch.is_fusion_entry(name)
            })
        })
        .collect();
    let par_group_data = world
        .make_par_group_data(&topology.parallel_groups, &par_eligible)
        .ok();

    Ok(RuntimeEpoch {
        id,
        fingerprint: current_asset_fingerprint(),
        topology,
        executor,
        contracts,
        accumulated_edges: tensor_edges,
        world,
        exploration_engine,
        learning_buffer: quantale_semiring_v2::LearningBuffer::new(
            &config.learned_edges_path,
            LEARNING_FLUSH_THRESHOLD,
        ),
        par_group_data,
    })
}

pub(super) fn changed_asset_fingerprint(previous: &AssetFingerprint) -> Option<AssetFingerprint> {
    let current = current_asset_fingerprint();
    (current != *previous).then_some(current)
}

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
