use quantale_semiring_v2::{
    COST_INFINITY, ControlNode, ExecutionOutcome, LAYER_CONFIDENCE, LAYER_COST, LAYER_SAFETY,
    NODE_COUNT, Node, ProjectionBias, StateNode, TensorEdge, TensorQuantaleWorld, tensor_idx,
};

fn host_tensor_closure(edges: &[TensorEdge]) -> Vec<f32> {
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
        let c = tensor_idx(LAYER_CONFIDENCE, edge.src, edge.dst);
        let e = tensor_idx(LAYER_COST, edge.src, edge.dst);
        let s = tensor_idx(LAYER_SAFETY, edge.src, edge.dst);
        tensor[c] = tensor[c].max(edge.confidence);
        tensor[e] = tensor[e].min(edge.cost);
        tensor[s] = tensor[s].max(edge.safety);
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
                let e = if a >= COST_INFINITY || b >= COST_INFINITY {
                    COST_INFINITY
                } else {
                    a + b
                };
                let eidx = tensor_idx(LAYER_COST, i, j);
                tensor[eidx] = tensor[eidx].min(e);

                let s = tensor[tensor_idx(LAYER_SAFETY, i, k)]
                    .min(tensor[tensor_idx(LAYER_SAFETY, k, j)]);
                let sidx = tensor_idx(LAYER_SAFETY, i, j);
                tensor[sidx] = tensor[sidx].max(s);
            }
        }
    }
    tensor
}

#[test]
fn host_tensor_layers_close_with_distinct_semirings() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let execute = Node::state(StateNode::Execute).encode();
    let direct = TensorEdge::new(goal, execute, 0.6, 10.0, 0.9);
    let indirect_a = TensorEdge::new(goal, plan, 0.9, 2.0, 0.7);
    let indirect_b = TensorEdge::new(plan, execute, 0.8, 3.0, 0.8);
    let tensor = host_tensor_closure(&[direct, indirect_a, indirect_b]);

    assert!((tensor[tensor_idx(LAYER_CONFIDENCE, goal, execute)] - 0.72).abs() < 1e-6);
    assert!((tensor[tensor_idx(LAYER_COST, goal, execute)] - 5.0).abs() < 1e-6);
    assert!((tensor[tensor_idx(LAYER_SAFETY, goal, execute)] - 0.9).abs() < 1e-6);
}

#[test]
fn gpu_tensor_closure_matches_layer_semantics() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let execute = Node::state(StateNode::Execute).encode();
    let edges = [
        TensorEdge::new(goal, execute, 0.6, 10.0, 0.9),
        TensorEdge::new(goal, plan, 0.9, 2.0, 0.7),
        TensorEdge::new(plan, execute, 0.8, 3.0, 0.8),
    ];
    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    world.close().unwrap();
    world.synchronize().unwrap();
    let tensor = world.tensor().unwrap();

    assert!((tensor[tensor_idx(LAYER_CONFIDENCE, goal, execute)] - 0.72).abs() < 1e-5);
    assert!((tensor[tensor_idx(LAYER_COST, goal, execute)] - 5.0).abs() < 1e-5);
    assert!((tensor[tensor_idx(LAYER_SAFETY, goal, execute)] - 0.9).abs() < 1e-5);
}

#[test]
fn gpu_tensor_projection_uses_blended_score() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let execute = Node::state(StateNode::Execute).encode();
    let repair = Node::control(ControlNode::Repair).encode();
    let edges = [
        TensorEdge::new(goal, plan, 0.95, 10.0, 0.95),
        TensorEdge::new(goal, repair, 0.70, 1.0, 0.70),
        TensorEdge::new(repair, execute, 0.70, 1.0, 0.70),
    ];
    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    world.close().unwrap();
    let decision = world
        .project(ProjectionBias {
            confidence: 0.5,
            cost: 3.0,
            safety: 0.5,
        })
        .unwrap();

    assert_eq!(decision.blocked, 0);
    assert_eq!(decision.first_hop, repair);
}

#[test]
fn gpu_tensor_update_and_decay_mutate_layers() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let mut world =
        TensorQuantaleWorld::from_tensor_edges(&[TensorEdge::new(goal, plan, 0.8, 2.0, 0.9)])
            .unwrap();
    world
        .update_lattice_edge(goal, plan, ExecutionOutcome::SafetyViolation)
        .unwrap();
    world.decay(0.9).unwrap();
    world.synchronize().unwrap();
    let tensor = world.tensor().unwrap();

    assert!(tensor[tensor_idx(LAYER_CONFIDENCE, goal, plan)] < 0.8);
    assert!(tensor[tensor_idx(LAYER_COST, goal, plan)] > 2.0);
    assert_eq!(tensor[tensor_idx(LAYER_SAFETY, goal, plan)], 0.0);
}

#[test]
fn gpu_tensor_frontier_step_advances_active_state() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let execute = Node::state(StateNode::Execute).encode();
    let edges = [
        TensorEdge::new(goal, plan, 0.95, 1.0, 0.95),
        TensorEdge::new(plan, execute, 0.90, 1.0, 0.90),
    ];
    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    world.close().unwrap();
    let first = world.frontier_step(ProjectionBias::default()).unwrap();
    assert_eq!(first.blocked, 0);
    assert_eq!(first.first_hop, plan);

    let second = world.frontier_step(ProjectionBias::default()).unwrap();
    assert_eq!(second.blocked, 0);
    assert_eq!(second.selected_src, plan);
    assert_eq!(second.first_hop, execute);
}

#[test]
fn gpu_tensor_tick_closes_and_advances_frontier() {
    let goal = Node::state(StateNode::Goal).encode();
    let plan = Node::state(StateNode::Plan).encode();
    let execute = Node::state(StateNode::Execute).encode();
    let edges = [
        TensorEdge::new(goal, plan, 0.95, 1.0, 0.95),
        TensorEdge::new(plan, execute, 0.90, 1.0, 0.90),
    ];
    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    let decision = world.tick(ProjectionBias::default()).unwrap();
    assert_eq!(decision.blocked, 0);
    assert_eq!(decision.first_hop, plan);

    let tensor = world.tensor().unwrap();
    assert!((tensor[tensor_idx(LAYER_CONFIDENCE, goal, execute)] - 0.855).abs() < 1e-5);
}

#[test]
fn gpu_tensor_witness_reconstructs_distinct_layer_paths() {
    let goal = Node::state(StateNode::Goal);
    let plan = Node::state(StateNode::Plan);
    let repair = Node::control(ControlNode::Repair);
    let validate = Node::state(StateNode::Validate);
    let execute = Node::state(StateNode::Execute);

    let edges = [
        // Best confidence path: Goal -> Plan -> Execute = 0.95 * 0.95 = 0.9025
        TensorEdge::new(goal.encode(), plan.encode(), 0.95, 10.0, 0.40),
        TensorEdge::new(plan.encode(), execute.encode(), 0.95, 10.0, 0.40),
        // Cheapest path: Goal -> Repair -> Execute = 1.0 + 1.0 = 2.0
        TensorEdge::new(goal.encode(), repair.encode(), 0.70, 1.0, 0.50),
        TensorEdge::new(repair.encode(), execute.encode(), 0.70, 1.0, 0.50),
        // Safest path: Goal -> Validate -> Execute = min(0.99, 0.99) = 0.99
        TensorEdge::new(goal.encode(), validate.encode(), 0.60, 5.0, 0.99),
        TensorEdge::new(validate.encode(), execute.encode(), 0.60, 5.0, 0.99),
    ];

    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    world.close().unwrap();
    world.synchronize().unwrap();

    let confidence_path = world
        .reconstruct_tensor_path(LAYER_CONFIDENCE, goal, execute)
        .unwrap();
    let cost_path = world
        .reconstruct_tensor_path(LAYER_COST, goal, execute)
        .unwrap();
    let safety_path = world
        .reconstruct_tensor_path(LAYER_SAFETY, goal, execute)
        .unwrap();

    assert_eq!(confidence_path, vec![goal, plan, execute]);
    assert_eq!(cost_path, vec![goal, repair, execute]);
    assert_eq!(safety_path, vec![goal, validate, execute]);

    assert_ne!(confidence_path, cost_path);
    assert_ne!(confidence_path, safety_path);
    assert_ne!(cost_path, safety_path);
}

#[test]
fn gpu_tensor_projects_and_commits_parallel_group() {
    let goal = Node::state(StateNode::Goal).encode();
    let map = Node::state(StateNode::Map).encode();
    let search = Node::state(StateNode::Search).encode();
    let parse = Node::state(StateNode::Parse).encode();
    let score = Node::state(StateNode::Score).encode();
    let edges = [
        TensorEdge::new(goal, map, 0.95, 1.0, 0.95),
        TensorEdge::new(goal, parse, 0.94, 1.0, 0.94),
        TensorEdge::new(map, search, 0.93, 1.0, 0.93),
        TensorEdge::new(parse, score, 0.92, 1.0, 0.92),
    ];
    let mut world = TensorQuantaleWorld::from_tensor_edges(&edges).unwrap();
    world.close().unwrap();

    let decisions = world
        .project_parallel_group(&[map, parse], ProjectionBias::default())
        .unwrap();
    assert_eq!(decisions.len(), 2);
    assert!(decisions.iter().all(|decision| decision.blocked == 0));
    assert_eq!(decisions[0].first_hop, map);
    assert_eq!(decisions[1].first_hop, parse);

    world.commit_decision_batch(&decisions).unwrap();
    let next = world.project(ProjectionBias::default()).unwrap();
    assert!(next.selected_src == map || next.selected_src == parse);
}
