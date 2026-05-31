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
