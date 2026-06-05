use quantale_semiring_v2::{
    LearningBuffer, LearningPolicy, SystemConfig, TensorQuantaleWorld, TlogWriter,
    TopologyInvariants, TopologyRuntime, UniversalExecutor, ViolationKind, check_with_operators,
    compile_pattern, console, format_violations, load_compiled_pattern_edges,
    load_default_patterns, load_learned_tensor_edges,
};

pub(crate) struct RuntimeEpoch {
    pub(crate) topology: TopologyRuntime,
    pub(crate) executor: UniversalExecutor,
    pub(crate) world: TensorQuantaleWorld,
    pub(crate) learning_buffer: LearningBuffer,
}

pub(crate) fn build_runtime_epoch(
    id: usize,
    config: &mut SystemConfig,
    learning_policy: &LearningPolicy,
    tlog: &mut TlogWriter,
) -> Result<RuntimeEpoch, String> {
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
            "{}
{} operator topology violation(s) found",
            format_violations(&blocking_operator_violations),
            blocking_operator_violations.len()
        ));
    }

    let executor = UniversalExecutor::from_config(config);

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

    Ok(RuntimeEpoch {
        topology,
        executor,
        world,
        learning_buffer: LearningBuffer::new(&config.learned_edges_path, LEARNING_FLUSH_THRESHOLD),
    })
}
