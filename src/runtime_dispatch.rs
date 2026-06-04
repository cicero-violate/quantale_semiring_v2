use super::*;

pub(super) struct ActiveExecution {
    pub(super) receipt: ProcessReceipt,
    pub(super) fusion: Option<FusionLogicalAdvance>,
    /// Set when the execution used the GPU hot path: caller must call
    /// `world.gpu_dispatch_region(region_id, src, dst)` followed by
    /// `world.drain_device_receipts()` to fold the receipt into the tensor.
    #[cfg(feature = "cuda")]
    hot_dispatch: Option<HotDispatchInfo>,
}

#[cfg(feature = "cuda")]
struct HotDispatchInfo {
    region_id: u32,
    src: i32,
    dst: i32,
}

pub(super) struct FusionLogicalAdvance {
    entry: String,
    exit: String,
    region: String,
    members: Vec<String>,
    edges: Vec<(i32, i32)>,
}

pub(super) fn node_name(node_id: i32, registry: &quantale_semiring_v2::NodeRegistry) -> String {
    Node::decode(node_id, registry)
        .and_then(|node| node.name(registry))
        .map(str::to_string)
        .unwrap_or_else(|| format!("Unknown({node_id})"))
}

pub(super) fn filter_static_topology_edges(
    edges: Vec<quantale_semiring_v2::TensorEdge>,
    topology_edges: &[quantale_semiring_v2::TensorEdge],
) -> Vec<quantale_semiring_v2::TensorEdge> {
    let allowed: BTreeSet<(i32, i32)> = topology_edges
        .iter()
        .map(|edge| (edge.src, edge.dst))
        .collect();
    edges
        .into_iter()
        .filter(|edge| allowed.contains(&(edge.src, edge.dst)))
        .collect()
}

/// If the execution used the GPU hot path, write a device receipt and drain it
/// into the quantale tensor.  No-ops for control/IO executions.
pub(super) fn apply_hot_dispatch_if_needed(
    world: &mut TensorQuantaleWorld,
    executor: &UniversalExecutor,
    execution: &ActiveExecution,
) {
    #[cfg(feature = "cuda")]
    if let Some(ref info) = execution.hot_dispatch {
        let outcome = match execution.receipt.exit_code {
            0 => 0,   // success
            124 => 2, // timeout
            _ => 1,   // failure
        };
        let result = executor
            .dispatch_hot_region_with_slots(
                world,
                info.region_id as i32,
                info.src,
                info.dst,
                outcome,
            )
            .and_then(|_| world.drain_device_receipts());
        if let Err(err) = result {
            console::warn(
                "gpu_dispatch",
                "drain_failed",
                &[("error", err.to_string())],
            );
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (world, executor, execution);
}

#[cfg(feature = "cuda")]
fn is_hot_dispatch(node_name: &str, config: &SystemConfig, executor: &UniversalExecutor) -> bool {
    config
        .split_topology
        .as_ref()
        .map(|split| split.hot.contains(node_name))
        .unwrap_or_else(|| {
            config.hot_region_registry.is_hot(node_name) || executor.is_hot_node(node_name)
        })
}

pub(super) fn execute_active_node_blocking(
    epoch: &RuntimeEpoch,
    config: &SystemConfig,
    decision: &DecisionReport,
    active_node_name: &str,
    current_payload: &Value,
) -> ActiveExecution {
    if let Some(entry) = config.fusion_dispatch.get_by_entry(active_node_name) {
        console::info(
            "fusion",
            "dispatch",
            &[
                ("entry", active_node_name.to_string()),
                ("region", entry.region.clone()),
                ("nodes", entry.nodes.join(" -> ")),
            ],
        );
        let receipt = epoch
            .executor
            .execute_fusion_entry_blocking(entry, current_payload);
        let fusion = build_fusion_logical_advance(entry, decision, epoch.topology.registry());
        return ActiveExecution {
            receipt,
            fusion,
            #[cfg(feature = "cuda")]
            hot_dispatch: None,
        };
    }

    #[cfg(feature = "cuda")]
    if is_hot_dispatch(active_node_name, config, &epoch.executor) {
        if let Some(region_id) = config.hot_region_registry.region_id_for(active_node_name) {
            let jit_receipt = epoch
                .executor
                .execute_abstract_node_blocking(active_node_name, current_payload);
            return ActiveExecution {
                receipt: jit_receipt,
                fusion: None,
                hot_dispatch: Some(HotDispatchInfo {
                    region_id,
                    src: decision.selected_src,
                    dst: decision.first_hop,
                }),
            };
        }
    }

    let receipt = epoch
        .executor
        .execute_abstract_node_blocking(active_node_name, current_payload);
    ActiveExecution {
        receipt,
        fusion: None,
        #[cfg(feature = "cuda")]
        hot_dispatch: None,
    }
}

fn build_fusion_logical_advance(
    entry: &quantale_semiring_v2::FusionEntry,
    decision: &DecisionReport,
    registry: &quantale_semiring_v2::NodeRegistry,
) -> Option<FusionLogicalAdvance> {
    let mut member_ids = Vec::with_capacity(entry.nodes.len());
    for member in &entry.nodes {
        let id = registry.id_of(member)? as i32;
        member_ids.push(id);
    }

    let first = *member_ids.first()?;
    if first != decision.first_hop {
        console::warn(
            "fusion",
            "logical_advance_skipped",
            &[
                ("region", entry.region.clone()),
                (
                    "reason",
                    "entry does not match selected first_hop".to_string(),
                ),
                ("entry_id", first.to_string()),
                ("first_hop", decision.first_hop.to_string()),
            ],
        );
        return None;
    }

    let mut edges = Vec::with_capacity(member_ids.len());
    edges.push((decision.selected_src, first));
    for pair in member_ids.windows(2) {
        edges.push((pair[0], pair[1]));
    }

    Some(FusionLogicalAdvance {
        entry: entry.nodes.first().cloned().unwrap_or_default(),
        exit: entry.nodes.last().cloned().unwrap_or_default(),
        region: entry.region.clone(),
        members: entry.nodes.clone(),
        edges,
    })
}

pub(super) fn queue_execution_lattice_updates(
    world: &mut TensorQuantaleWorld,
    decision: &DecisionReport,
    execution: &ActiveExecution,
    outcome: ExecutionOutcome,
) {
    if outcome == ExecutionOutcome::Success {
        if let Some(fusion) = &execution.fusion {
            for (src, dst) in &fusion.edges {
                world.queue_lattice_update(*src, *dst, outcome);
            }
            return;
        }
    }

    world.queue_lattice_update(decision.selected_src, decision.first_hop, outcome);
}

pub(super) fn update_execution_receipt_priors(
    exploration_engine: &mut ExplorationEngine,
    decision: &DecisionReport,
    execution: &ActiveExecution,
    receipt: &ProcessReceipt,
) {
    exploration_engine.update_receipt_prior(decision.first_hop, receipt);
    if receipt.exit_code != 0 {
        return;
    }
    if let Some(fusion) = &execution.fusion {
        for (_, dst) in &fusion.edges {
            if *dst != decision.first_hop {
                exploration_engine.update_receipt_prior(*dst, receipt);
            }
        }
    }
}

impl FusionLogicalAdvance {
    pub(super) fn receipt_json(&self) -> Value {
        json!({
            "kind": "fusion_region_executed",
            "entry": self.entry,
            "exit": self.exit,
            "members": self.members,
            "member_receipts": self.members.iter().map(|member| {
                json!({
                    "node": member,
                    "exit_code": 0,
                    "outcome": "success",
                    "logical_backend": "fused_region",
                })
            }).collect::<Vec<_>>(),
            "physical_backend": "cuda_jit",
            "logical_advance": "region_atomic",
            "region": self.region,
            "edges": self.edges.iter().map(|(src, dst)| {
                json!({ "src": src, "dst": dst })
            }).collect::<Vec<_>>(),
        })
    }
}

/// Record edge deltas for a successful execution into the learning buffer.
/// Only topology edges (those present in the static edge set) are persisted;
/// ephemeral CKA-pattern-only edges are skipped.
///
/// Takes topology and buffer as separate references so the caller can split-borrow
/// the enclosing RuntimeEpoch without a mutable-immutable conflict.
pub(super) fn record_learning_edges(
    learning_buffer: &mut quantale_semiring_v2::LearningBuffer,
    topology: &TopologyRuntime,
    decision: &DecisionReport,
    execution: &ActiveExecution,
    learning_policy: &LearningPolicy,
) {
    if execution.receipt.exit_code != 0 {
        return;
    }
    let registry = topology.registry();
    let tensor_edges = topology.tensor_edges();
    let boost = learning_policy.max_confidence_above_base * 0.5;

    let pairs: Vec<(i32, i32)> = match &execution.fusion {
        Some(fusion) => fusion.edges.clone(),
        None => vec![(decision.selected_src, decision.first_hop)],
    };

    for (src, dst) in pairs {
        let Some(src_name) = registry.name_of(src as usize) else {
            continue;
        };
        let Some(dst_name) = registry.name_of(dst as usize) else {
            continue;
        };
        let Some(base) = tensor_edges.iter().find(|e| e.src == src && e.dst == dst) else {
            continue;
        };
        let confidence = (base.confidence + boost).min(1.0);
        learning_buffer.record(src_name, dst_name, confidence, base.cost, base.safety);
    }
}

impl ActiveExecution {
    pub(super) fn output_origin<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.fusion
            .as_ref()
            .filter(|_| self.receipt.exit_code == 0)
            .map(|fusion| fusion.exit.as_str())
            .unwrap_or(fallback)
    }
}
