use quantale_semiring_v2::{
    COST_INFINITY, ControlNode, ExplorationCandidate, ExplorationConfig, ExplorationDecision,
    ExplorationEngine, GraphTopology, LAYER_CONFIDENCE, LAYER_COST, LAYER_SAFETY, NODE_COUNT, Node,
    ProcessReceipt, ProjectionBias, StateNode, TensorEdge, TensorQuantaleWorld,
    load_operator_registry, tensor_idx,
};

fn host_tensor(edges: &[TensorEdge]) -> Vec<f32> {
    let mut tensor = vec![0.0; 3 * NODE_COUNT * NODE_COUNT];
    for i in 0..NODE_COUNT as i32 {
        tensor[tensor_idx(LAYER_CONFIDENCE, i, i)] = 1.0;
        tensor[tensor_idx(LAYER_COST, i, i)] = 0.0;
        tensor[tensor_idx(LAYER_SAFETY, i, i)] = 1.0;
    }
    for i in 0..NODE_COUNT as i32 {
        for j in 0..NODE_COUNT as i32 {
            if i != j {
                tensor[tensor_idx(LAYER_COST, i, j)] = COST_INFINITY;
            }
        }
    }
    for edge in edges {
        tensor[tensor_idx(LAYER_CONFIDENCE, edge.src, edge.dst)] = edge.confidence;
        tensor[tensor_idx(LAYER_COST, edge.src, edge.dst)] = edge.cost;
        tensor[tensor_idx(LAYER_SAFETY, edge.src, edge.dst)] = edge.safety;
    }
    for k in 0..NODE_COUNT as i32 {
        for i in 0..NODE_COUNT as i32 {
            for j in 0..NODE_COUNT as i32 {
                let c = tensor[tensor_idx(LAYER_CONFIDENCE, i, k)]
                    * tensor[tensor_idx(LAYER_CONFIDENCE, k, j)];
                let cidx = tensor_idx(LAYER_CONFIDENCE, i, j);
                tensor[cidx] = tensor[cidx].max(c);

                let a = tensor[tensor_idx(LAYER_COST, i, k)];
                let b = tensor[tensor_idx(LAYER_COST, k, j)];
                let cost = if a >= COST_INFINITY || b >= COST_INFINITY {
                    COST_INFINITY
                } else {
                    a + b
                };
                let eidx = tensor_idx(LAYER_COST, i, j);
                tensor[eidx] = tensor[eidx].min(cost);

                let safety = tensor[tensor_idx(LAYER_SAFETY, i, k)]
                    .min(tensor[tensor_idx(LAYER_SAFETY, k, j)]);
                let sidx = tensor_idx(LAYER_SAFETY, i, j);
                tensor[sidx] = tensor[sidx].max(safety);
            }
        }
    }
    tensor
}

fn config() -> ExplorationConfig {
    ExplorationConfig::default_asset().expect("default exploration config")
}

fn topology() -> GraphTopology {
    GraphTopology::default_asset().expect("default topology")
}

fn operators() -> quantale_semiring_v2::OperatorRegistry {
    load_operator_registry("assets/operators.json").expect("operator registry")
}

#[test]
fn exploration_config_loads_from_json() {
    let config = config();
    assert_eq!(config.beam_width, 8);
    assert_eq!(config.max_depth, 4);
    assert_eq!(config.strategies.len(), 3);
    assert_eq!(config.strategies[0].start, "State::Plan");
}

#[test]
fn exploration_rejects_unknown_strategy_node() {
    let raw = r#"{
        "engine": {"beam_width": 1, "max_depth": 1, "max_batches": 1, "novelty_weight": 0.0, "receipt_weight": 0.0, "entropy_penalty": 0.0},
        "strategies": [{"name": "bad", "start": "State::Nope", "bias": {"confidence": 1.0, "cost": 1.0, "safety": 1.0}}]
    }"#;
    let config = ExplorationConfig::from_json_str(raw).expect("parse config");
    let err = ExplorationEngine::new(config, &topology(), operators())
        .expect_err("unknown node rejected");
    assert!(err.message.contains("unknown exploration strategy node"));
}

#[test]
fn exploration_seeds_tokens() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let start = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let validate = Node::state(StateNode::Validate).encode();
    let repair = Node::control(ControlNode::Repair).encode();
    let tensor = host_tensor(&[
        TensorEdge::new(start, plan, 0.9, 2.0, 0.8),
        TensorEdge::new(start, validate, 0.7, 1.0, 0.95),
        TensorEdge::new(start, repair, 0.6, 1.5, 0.9),
    ]);
    let tokens = engine.seed_tokens(&tensor).expect("seed");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].node, plan);
    assert_eq!(tokens[1].node, validate);
    assert_eq!(tokens[2].node, repair);
}

#[test]
fn exploration_expands_bounded_depth_only() {
    let mut config = config();
    config.max_depth = 1;
    let mut engine = ExplorationEngine::new(config, &topology(), operators()).expect("engine");
    let tensor = host_tensor(&topology().compile().unwrap().tensor_edges);
    let candidates = engine.host_expand_exploration(&tensor).expect("expand");
    assert!(engine.tokens().iter().all(|token| token.depth <= 1));
    assert!(!candidates.is_empty());
}

#[test]
fn exploration_selects_topk() {
    let mut config = config();
    config.beam_width = 2;
    let engine = ExplorationEngine::new(config, &topology(), operators()).expect("engine");
    let selected = engine.host_select_topk(vec![
        ExplorationCandidate {
            token_id: 1,
            first_hop: 1,
            terminal_node: 1,
            value: 1.0,
        },
        ExplorationCandidate {
            token_id: 2,
            first_hop: 2,
            terminal_node: 2,
            value: 3.0,
        },
        ExplorationCandidate {
            token_id: 3,
            first_hop: 3,
            terminal_node: 3,
            value: 2.0,
        },
    ]);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].terminal_node, 2);
    assert_eq!(selected[1].terminal_node, 3);
}

#[test]
fn exploration_backtracks_winning_path() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let tensor = host_tensor(&topology().compile().unwrap().tensor_edges);
    let candidates = engine.host_expand_exploration(&tensor).expect("expand");
    let candidate = candidates[0];
    let path = engine.reconstruct_exploration_path(candidate);
    assert!(!path.is_empty());
    assert_eq!(path.last().unwrap().encode(), candidate.terminal_node);
}

#[test]
fn exploration_respects_effect_safety() {
    let engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let execute = Node::state(StateNode::Execute).encode();
    let err = engine
        .validate_candidate_effect(&ExplorationCandidate {
            token_id: 0,
            first_hop: execute,
            terminal_node: execute,
            value: 1.0,
        })
        .expect_err("execute lock rejected");
    assert!(err.message.contains("exclusive unsafe lock"));
}

#[test]
fn exploration_falls_back_to_cka_batch() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let tensor = vec![0.0; 3 * NODE_COUNT * NODE_COUNT];
    let decision = engine.propose(&tensor, true, true).expect("proposal");
    assert_eq!(decision, ExplorationDecision::UseCkaBatch);
}

#[test]
fn exploration_falls_back_to_single_frontier() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let tensor = vec![0.0; 3 * NODE_COUNT * NODE_COUNT];
    let decision = engine.propose(&tensor, false, true).expect("proposal");
    assert_eq!(decision, ExplorationDecision::SingleFrontier);
}

#[test]
fn receipts_update_exploration_prior() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let validate = Node::state(StateNode::Validate).encode();
    let ok = ProcessReceipt {
        node_name: "State::Validate".to_string(),
        exit_code: 0,
        stdout_payload: String::new(),
        stderr_payload: String::new(),
    };
    engine.update_receipt_prior(validate, &ok);
    let after_success = engine.receipt_prior_for(validate);
    assert!(after_success > 0.0);
    let fail = ProcessReceipt { exit_code: 1, ..ok };
    engine.update_receipt_prior(validate, &fail);
    assert!(engine.receipt_prior_for(validate) < after_success);
}

#[test]
fn tensor_world_exploration_api_seeds_when_cuda_available() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let mut world =
        TensorQuantaleWorld::from_tensor_edges(&[TensorEdge::new(goal, plan, 0.9, 1.0, 0.9)])
            .unwrap();
    world.close().unwrap();
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    world
        .seed_exploration(&mut engine)
        .expect("seed exploration");
    assert_eq!(engine.tokens().len(), 1);
}

#[test]
fn exploration_config_supports_projection_bias_deserialize() {
    let config = config();
    assert_eq!(
        config.strategies[0].bias,
        ProjectionBias {
            confidence: 0.70,
            cost: 1.50,
            safety: 0.80
        }
    );
}

#[test]
fn gpu_exploration_expands_tokens_bounded() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let optimize = Node::state(StateNode::Optimize).encode();
    let validate = Node::state(StateNode::Validate).encode();
    let repair = Node::control(ControlNode::Repair).encode();
    let mut world = TensorQuantaleWorld::from_tensor_edges(&[
        TensorEdge::new(goal, plan, 0.9, 1.0, 0.9),
        TensorEdge::new(goal, validate, 0.8, 1.0, 0.95),
        TensorEdge::new(goal, repair, 0.7, 1.0, 0.8),
        TensorEdge::new(plan, optimize, 0.9, 1.0, 0.9),
    ])
    .unwrap();
    world.close().unwrap();
    let mut config = config();
    config.max_depth = 1;
    let mut engine = ExplorationEngine::new(config, &topology(), operators()).expect("engine");
    let selected = world.expand_exploration(&mut engine).expect("gpu expand");
    assert!(!selected.is_empty());
    assert!(engine.tokens().iter().all(|token| token.depth <= 1));
    assert!(engine.tokens().iter().any(|token| token.node == optimize));
}

#[test]
fn gpu_exploration_selects_topk_candidates() {
    let mut world =
        TensorQuantaleWorld::from_tensor_edges(&topology().compile().unwrap().tensor_edges)
            .unwrap();
    world.close().unwrap();
    let mut config = config();
    config.beam_width = 2;
    config.max_depth = 2;
    let mut engine = ExplorationEngine::new(config, &topology(), operators()).expect("engine");
    let selected = world.expand_exploration(&mut engine).expect("gpu expand");
    assert_eq!(selected.len(), 2);
    assert!(selected[0].value >= selected[1].value);
}

#[test]
fn gpu_exploration_commit_advances_frontier() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let validate = Node::state(StateNode::Validate).encode();
    let repair = Node::control(ControlNode::Repair).encode();
    let optimize = Node::state(StateNode::Optimize).encode();
    let mut world = TensorQuantaleWorld::from_tensor_edges(&[
        TensorEdge::new(goal, plan, 0.9, 1.0, 0.9),
        TensorEdge::new(goal, validate, 0.8, 1.0, 0.95),
        TensorEdge::new(goal, repair, 0.7, 1.0, 0.8),
        TensorEdge::new(plan, optimize, 0.9, 1.0, 0.9),
        TensorEdge::new(validate, optimize, 0.9, 1.0, 0.9),
        TensorEdge::new(repair, optimize, 0.9, 1.0, 0.9),
    ])
    .unwrap();
    world.close().unwrap();
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let selected = world.expand_exploration(&mut engine).expect("gpu expand");
    let decision = world
        .commit_exploration_candidate(&selected[0])
        .expect("gpu commit");
    assert_eq!(decision.blocked, 0);
    assert_eq!(decision.first_hop, selected[0].first_hop);

    let next = world.frontier_step(ProjectionBias::default()).unwrap();
    assert_eq!(next.selected_src, selected[0].first_hop);
}

#[test]
fn exploration_anti_repeat_skips_committed_terminal() {
    let mut engine = ExplorationEngine::new(config(), &topology(), operators()).expect("engine");
    let first = ExplorationCandidate {
        token_id: 0,
        first_hop: 7,
        terminal_node: 21,
        value: 9.0,
    };
    let second = ExplorationCandidate {
        token_id: 1,
        first_hop: 10,
        terminal_node: 10,
        value: 1.0,
    };
    engine.load_gpu_state(Vec::new(), vec![first, second]);
    assert_eq!(engine.best_commit_candidate(), Some(first));
    engine.mark_candidate_committed(&first);
    assert!(!engine.candidate_allowed_by_repeat_policy(&first));
    assert_eq!(engine.best_commit_candidate(), Some(second));
}

#[test]
fn operator_registry_covers_all_symbolic_nodes() {
    let registry = operators();
    for id in 0..NODE_COUNT as i32 {
        let name = quantale_semiring_v2::node_name(id);
        assert!(
            registry.contains_key(&name),
            "missing operator contract for {name}"
        );
    }
}
