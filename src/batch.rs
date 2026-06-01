//! Safe decision-batch preparation and dispatch for CKA parallel groups.
//!
//! CUDA projects and commits effect-safe CKA `par` groups. Host workers then
//! execute the corresponding operators concurrently.

use serde::Serialize;
use serde_json::Value;

use crate::config::OperatorRegistry;
use crate::egress::UniversalExecutor;
use crate::error::CudaError;
use crate::node::Node;
use crate::pattern::{operator_effects_for_node, safe_parallel, CompiledCkaPattern};
use crate::projection::DecisionReport;
use crate::receipt::ProcessReceipt;
use crate::tensor::{ProjectionBias, TensorQuantaleWorld};
use crate::topology::{GraphTopology, NodeRegistry};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionBatch {
    pub step: i32,
    pub decisions: Vec<DecisionReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchPlan {
    pub pattern_name: String,
    pub batches: Vec<DecisionBatch>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScheduledReceipt {
    pub decision: DecisionReport,
    pub receipt: ProcessReceipt,
}

impl BatchPlan {
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

pub fn project_ready_batch_plan(
    world: &mut TensorQuantaleWorld,
    compiled_patterns: &[CompiledCkaPattern],
    bias: ProjectionBias,
    operator_registry: &OperatorRegistry,
) -> Result<Option<BatchPlan>, CudaError> {
    let node_registry = GraphTopology::bundled_registry()?;
    for compiled in compiled_patterns {
        for group in &compiled.parallel_groups {
            if group.len() < 2 {
                continue;
            }
            validate_parallel_group_effects(group, operator_registry, &node_registry)?;
            let decisions = world.project_parallel_group(group, bias)?;
            if decisions
                .iter()
                .all(|decision| decision.blocked == 0 && decision.halted == 0)
            {
                let plan = prepare_parallel_batch_plan(
                    compiled,
                    &decisions,
                    operator_registry,
                    &node_registry,
                )?;
                if !plan.is_empty() {
                    return Ok(Some(plan));
                }
            }
        }
    }
    Ok(None)
}

pub fn dispatch_decision_batch_blocking(
    executor: &UniversalExecutor,
    batch: &DecisionBatch,
    dynamic_payload: &Value,
) -> Vec<ScheduledReceipt> {
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(batch.decisions.len());
        for decision in &batch.decisions {
            handles.push(scope.spawn(move || {
                let node_name =
                    decision_node_name(decision, executor.node_registry()).unwrap_or("Unknown");
                let receipt = executor.execute_abstract_node_blocking(node_name, dynamic_payload);
                ScheduledReceipt {
                    decision: *decision,
                    receipt,
                }
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().expect("parallel scheduler worker panicked"))
            .collect()
    })
}

fn decision_node_name<'a>(
    decision: &DecisionReport,
    registry: &'a NodeRegistry,
) -> Option<&'a str> {
    Node::decode(decision.first_hop, registry).and_then(|node| node.name(registry))
}

pub fn prepare_parallel_batch_plan(
    compiled: &CompiledCkaPattern,
    decisions: &[DecisionReport],
    operator_registry: &OperatorRegistry,
    node_registry: &NodeRegistry,
) -> Result<BatchPlan, CudaError> {
    let mut batches = Vec::new();

    for group in &compiled.parallel_groups {
        if group.len() < 2 {
            continue;
        }
        validate_parallel_group_effects(group, operator_registry, node_registry)?;

        let mut batch_decisions = Vec::with_capacity(group.len());
        for node_id in group {
            let name = node_name(*node_id, node_registry)?;
            let decision = decisions
                .iter()
                .find(|decision| decision_matches_node(decision, *node_id))
                .copied()
                .ok_or_else(|| {
                    CudaError::invalid_input(format!(
                        "parallel group node '{name}' has no runnable decision"
                    ))
                })?;
            batch_decisions.push(decision);
        }

        let step = batch_decisions
            .iter()
            .map(|decision| decision.step)
            .min()
            .unwrap_or_default();
        batches.push(DecisionBatch {
            step,
            decisions: batch_decisions,
        });
    }

    Ok(BatchPlan {
        pattern_name: compiled.name.clone(),
        batches,
    })
}

pub fn validate_parallel_group_effects(
    group: &[i32],
    operator_registry: &OperatorRegistry,
    node_registry: &NodeRegistry,
) -> Result<(), CudaError> {
    let mut effects = Vec::with_capacity(group.len());
    for node_id in group {
        let name = node_name(*node_id, node_registry)?;
        effects.push(operator_effects_for_node(name, operator_registry)?);
    }

    for left in 0..effects.len() {
        for right in (left + 1)..effects.len() {
            if !safe_parallel(&effects[left], &effects[right]) {
                return Err(CudaError::invalid_input(format!(
                    "parallel group nodes '{}' and '{}' are not effect-independent",
                    node_name(group[left], node_registry)?,
                    node_name(group[right], node_registry)?
                )));
            }
        }
    }
    Ok(())
}

fn decision_matches_node(decision: &DecisionReport, node_id: i32) -> bool {
    decision.blocked == 0
        && decision.halted == 0
        && (decision.first_hop == node_id || decision.selected_dst == node_id)
}

fn node_name(node_id: i32, registry: &NodeRegistry) -> Result<&str, CudaError> {
    Node::decode(node_id, registry)
        .and_then(|node| node.name(registry))
        .ok_or_else(|| CudaError::invalid_input(format!("invalid parallel group node {node_id}")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::config::load_operator_registry;
    use crate::tensor::TensorEdge;
    use crate::topology::GraphTopology;

    fn test_registry() -> crate::topology::NodeRegistry {
        GraphTopology::default_asset()
            .unwrap()
            .compile()
            .unwrap()
            .registry
    }

    fn node_id(registry: &crate::topology::NodeRegistry, name: &str) -> i32 {
        registry.id_of(name).unwrap() as i32
    }

    fn compiled_group(nodes: Vec<i32>) -> CompiledCkaPattern {
        CompiledCkaPattern {
            name: "parallel_prepare".to_string(),
            edges: vec![TensorEdge::new(nodes[0], nodes[1], 0.75, 1.0, 0.8)],
            parallel_groups: vec![nodes],
        }
    }

    fn decision(step: i32, node_raw: i32, goal_raw: i32) -> DecisionReport {
        DecisionReport {
            step,
            selected_src: goal_raw,
            selected_dst: node_raw,
            first_hop: node_raw,
            selected_value: 1.0,
            halted: 0,
            blocked: 0,
        }
    }

    #[test]
    fn batch_plan_collects_runnable_parallel_decisions() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let compiled = compiled_group(vec![map, parse]);
        let op_registry = load_operator_registry("assets/operators.json").unwrap();

        let plan = prepare_parallel_batch_plan(
            &compiled,
            &[decision(7, map, goal), decision(7, parse, goal)],
            &op_registry,
            &node_reg,
        )
        .unwrap();

        assert_eq!(plan.pattern_name, "parallel_prepare");
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].step, 7);
        assert_eq!(plan.batches[0].decisions.len(), 2);
    }

    #[test]
    fn batch_plan_rejects_missing_member_decision() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let compiled = compiled_group(vec![map, parse]);
        let op_registry = load_operator_registry("assets/operators.json").unwrap();

        let err = prepare_parallel_batch_plan(
            &compiled,
            &[decision(7, map, goal)],
            &op_registry,
            &node_reg,
        )
        .unwrap_err();
        assert!(err.message.contains("has no runnable decision"));
    }

    #[test]
    fn batch_plan_rejects_blocked_member_decision() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let compiled = compiled_group(vec![map, parse]);
        let op_registry = load_operator_registry("assets/operators.json").unwrap();
        let mut blocked = decision(7, parse, goal);
        blocked.blocked = 1;

        let err = prepare_parallel_batch_plan(
            &compiled,
            &[decision(7, map, goal), blocked],
            &op_registry,
            &node_reg,
        )
        .unwrap_err();
        assert!(err.message.contains("has no runnable decision"));
    }

    #[test]
    fn batch_plan_revalidates_effect_conflicts() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let compiled = compiled_group(vec![map, parse]);
        let mut op_registry = load_operator_registry("assets/operators.json").unwrap();
        op_registry.insert(
            "State::Map".to_string(),
            json!({"effects": {"reads": [], "writes": ["shared"], "locks": []}}),
        );
        op_registry.insert(
            "State::Parse".to_string(),
            json!({"effects": {"reads": ["shared"], "writes": [], "locks": []}}),
        );

        let err = prepare_parallel_batch_plan(
            &compiled,
            &[decision(7, map, goal), decision(7, parse, goal)],
            &op_registry,
            &node_reg,
        )
        .unwrap_err();
        assert!(err.message.contains("not effect-independent"));
    }

    #[test]
    fn dispatcher_executes_batch_workers_concurrently() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let op_registry = load_operator_registry("assets/operators.json").unwrap();
        let executor = UniversalExecutor::new(op_registry);
        let batch = DecisionBatch {
            step: 9,
            decisions: vec![decision(9, map, goal), decision(9, parse, goal)],
        };

        let receipts =
            dispatch_decision_batch_blocking(&executor, &batch, &json!({"context": "x"}));
        assert_eq!(receipts.len(), 2);
        assert!(receipts.iter().all(|s| s.receipt.exit_code == 0));
        assert!(receipts.iter().any(|s| s.receipt.node_name == "State::Map"));
        assert!(receipts
            .iter()
            .any(|s| s.receipt.node_name == "State::Parse"));
    }

    #[test]
    fn dispatcher_routes_cuda_ptx_to_egress_not_process() {
        let node_reg = test_registry();
        let map = node_id(&node_reg, "State::Map");
        let parse = node_id(&node_reg, "State::Parse");
        let goal = node_id(&node_reg, "State::Goal");
        let mut op_registry = load_operator_registry("assets/operators.json").unwrap();
        op_registry.insert(
            "State::Map".to_string(),
            json!({
                "node_name": "State::Map",
                "executable": "cuda_ptx",
                "static_args": [],
                "input_mapping": {
                    "module": "cuda/trading_execution_kernels.ptx",
                    "module_name": "quantale_trading_execution_kernels",
                    "kernel": "fused_alpha_and_risk_kernel",
                    "plane": "execution",
                    "scheduler_contract": "atomic_operator_fixed_budget"
                },
                "effects": {"reads": ["market.feed"], "writes": ["execution.gpu.results"], "locks": []}
            }),
        );
        op_registry.insert(
            "State::Parse".to_string(),
            json!({
                "node_name": "State::Parse",
                "executable": "cuda_ptx",
                "static_args": [],
                "input_mapping": {
                    "module": "cuda/trading_execution_kernels.ptx",
                    "module_name": "quantale_trading_execution_kernels",
                    "kernel": "fused_orderbook_and_alpha_kernel",
                    "plane": "execution",
                    "scheduler_contract": "atomic_operator_fixed_budget"
                },
                "effects": {"reads": ["market.feed"], "writes": ["execution.gpu.results"], "locks": []}
            }),
        );
        let executor = UniversalExecutor::new(op_registry);
        let batch = DecisionBatch {
            step: 9,
            decisions: vec![decision(9, map, goal), decision(9, parse, goal)],
        };

        let receipts =
            dispatch_decision_batch_blocking(&executor, &batch, &json!({"context": "x"}));
        assert_eq!(receipts.len(), 2);

        // In all cases the dispatch must not attempt to spawn a "cuda_ptx" binary process.
        for r in &receipts {
            assert!(!r.receipt.stderr_payload.contains("Failed to spawn process"));
        }

        // Without the cuda feature: egress returns an explicit capability error.
        #[cfg(not(feature = "cuda"))]
        for r in &receipts {
            assert_eq!(r.receipt.exit_code, 1);
            assert!(r
                .receipt
                .stderr_payload
                .contains("requires the cuda feature"));
        }
    }
}
