//! Behavioral and regression tests for the upgrade.v2 work.
//!
//! Covers:
//!  - TypedIR: non-lowerable ops return Err
//!  - Hot region slot validation against declared operators
//!  - Hot topology executable invariants
//!  - Split topology loads through SystemConfig
//!  - HostStagingBuffer and AsyncUploadQueue staging behaviour
//!  - Ring-buffer push/pop (CUDA-gated)

use quantale_semiring_v2::{
    AsyncUploadQueue, HostStagingBuffer, HotRegionRegistry, TypedIrOp,
    ir_op_to_jit_body, load_operator_registry,
};
#[cfg(not(feature = "cuda"))]
use quantale_semiring_v2::DeviceSlotRegistry;

// ── TypedIR rejection tests ───────────────────────────────────────────────────

#[test]
fn ir_reduce_rejects_scalar_lowering() {
    let op = TypedIrOp::Reduce {
        input: "x".into(),
        output: "y".into(),
        init: 0.0,
        body: "acc + in0[i]".into(),
    };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "Reduce must return Err; scalar element body cannot express parallel reduction"
    );
}

#[test]
fn ir_topk_rejects_scalar_lowering() {
    let op = TypedIrOp::TopK { input: "x".into(), output: "y".into(), k: 5 };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "TopK must return Err; scalar element body cannot select top-k"
    );
}

#[test]
fn ir_matmul_rejects_scalar_lowering() {
    let op = TypedIrOp::MatMul { a: "a".into(), b: "b".into(), output: "c".into() };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "MatMul must return Err; scalar element body cannot express GEMM"
    );
}

#[test]
fn ir_join_rejects_scalar_lowering() {
    let op = TypedIrOp::Join {
        left: "l".into(),
        right: "r".into(),
        output: "o".into(),
        key: "k".into(),
    };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "Join must return Err; scalar element body cannot hash-join"
    );
}

#[test]
fn ir_sort_rejects_scalar_lowering() {
    let op = TypedIrOp::Sort {
        input: "x".into(),
        output: "y".into(),
        key: "k".into(),
        ascending: true,
    };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "Sort must return Err; scalar element body cannot sort"
    );
}

#[test]
fn ir_graph_traverse_rejects_scalar_lowering() {
    let op = TypedIrOp::GraphTraverse {
        nodes: "n".into(),
        edges: "e".into(),
        output: "o".into(),
        max_depth: 3,
    };
    assert!(
        ir_op_to_jit_body(&op).is_err(),
        "GraphTraverse must return Err; scalar element body cannot do BFS"
    );
}

// Verify the ops that should succeed still do.

#[test]
fn ir_map_lowers_to_element_body() {
    let op = TypedIrOp::Map {
        input: "x".into(),
        output: "y".into(),
        body: "in0[i] * 2.0f".into(),
    };
    let body = ir_op_to_jit_body(&op).expect("Map should lower");
    assert!(body.contains("out[i]"));
    assert!(body.contains("in0[i]"));
}

#[test]
fn ir_filter_lowers_to_ternary() {
    let op = TypedIrOp::Filter {
        input: "x".into(),
        output: "y".into(),
        predicate: "in0[i] > 0.0f".into(),
    };
    let body = ir_op_to_jit_body(&op).expect("Filter should lower");
    assert!(body.contains("?"));
}

#[test]
fn ir_verify_lowers_to_flag_expression() {
    let op = TypedIrOp::Verify { input: "x".into(), predicate: "in0[i] >= 0.0f".into() };
    let body = ir_op_to_jit_body(&op).expect("Verify should lower");
    assert!(body.contains("1.0f"));
    assert!(body.contains("0.0f"));
}

#[test]
fn ir_embed_lowers_to_row_read() {
    let op = TypedIrOp::Embed { input: "x".into(), output: "y".into(), dim: 64 };
    let body = ir_op_to_jit_body(&op).expect("Embed should lower");
    assert!(body.contains("64"), "body should reference dim");
}

// ── Hot region slot validation ────────────────────────────────────────────────

#[test]
fn hot_region_slots_all_declared_in_operator_registry() {
    let registry = load_operator_registry("assets/operators.generated.json")
        .expect("load operators.generated.json");
    let hot = HotRegionRegistry::load("assets/regions.hot.json")
        .expect("load regions.hot.json");

    let declared: std::collections::HashSet<String> = registry
        .values()
        .flat_map(|op| {
            let reads = op["effects"]["reads"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string));
            let writes = op["effects"]["writes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string));
            reads.chain(writes)
        })
        .collect();

    let violations = hot.validate_slots(&declared);
    assert!(
        violations.is_empty(),
        "undeclared hot region slots: {violations:?}"
    );
}

// ── Hot topology executable invariants ───────────────────────────────────────

#[test]
fn hot_topology_has_nodes_and_transitions() {
    let raw = std::fs::read_to_string("assets/topology.hot.json")
        .expect("read topology.hot.json");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse topology.hot.json");

    let nodes = doc["nodes"].as_array().expect("nodes array");
    let transitions = doc["transitions"].as_array().expect("transitions array");

    assert!(!nodes.is_empty(), "hot topology must have at least one node");
    assert!(
        !transitions.is_empty(),
        "hot topology must have at least one transition"
    );
}

#[test]
fn hot_topology_transition_endpoints_exist_in_nodes() {
    let raw = std::fs::read_to_string("assets/topology.hot.json")
        .expect("read topology.hot.json");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse");

    let node_names: std::collections::HashSet<&str> = doc["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();

    for t in doc["transitions"].as_array().unwrap() {
        let from = t["from"].as_str().unwrap();
        let to   = t["to"].as_str().unwrap();
        assert!(node_names.contains(from), "transition 'from' endpoint '{from}' not in nodes");
        assert!(node_names.contains(to),   "transition 'to' endpoint '{to}' not in nodes");
    }
}

#[test]
fn hot_topology_has_no_control_io_nodes() {
    let raw = std::fs::read_to_string("assets/topology.hot.json")
        .expect("read topology.hot.json");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    for node in doc["nodes"].as_array().unwrap() {
        let t = node["type"].as_str().unwrap_or("");
        assert!(
            t != "State" && t != "Control" && t != "Event",
            "hot topology must not contain control/IO node '{}' (type='{t}')",
            node["name"].as_str().unwrap_or("?")
        );
    }
}

// ── Split topology loads through SystemConfig ─────────────────────────────────

#[test]
fn system_config_loads_split_topology() {
    // SystemConfig::default() loads the split topology from the asset files.
    // If the split topology files are present and valid, it should be Some(_).
    let config = quantale_semiring_v2::SystemConfig::default();
    assert!(
        config.split_topology.is_some(),
        "SplitTopologyRuntime must load successfully from assets"
    );
}

#[test]
fn split_topology_control_and_hot_are_disjoint() {
    let config = quantale_semiring_v2::SystemConfig::default();
    let split = config.split_topology.as_ref().expect("split topology present");
    let overlap: Vec<&str> = split
        .control
        .node_names
        .iter()
        .filter(|n| split.hot.node_names.contains(*n))
        .map(String::as_str)
        .collect();
    assert!(
        overlap.is_empty(),
        "control and hot topologies must be disjoint: {overlap:?}"
    );
}

#[test]
fn split_topology_hot_has_at_least_one_transition() {
    let config = quantale_semiring_v2::SystemConfig::default();
    let split = config.split_topology.as_ref().expect("split topology present");
    assert!(
        !split.hot.transitions.is_empty(),
        "hot topology must have at least one transition"
    );
}

// ── HostStagingBuffer behaviour ───────────────────────────────────────────────

#[test]
fn host_staging_buffer_preserves_data() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = HostStagingBuffer::from_slice(&data);
    assert_eq!(buf.len(), 4);
    assert!(!buf.is_empty());
    assert_eq!(buf.data, data);
}

#[test]
fn host_staging_buffer_empty() {
    let buf = HostStagingBuffer::from_slice(&[]);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

// ── AsyncUploadQueue staging behaviour (non-CUDA) ────────────────────────────

#[test]
fn upload_queue_stage_accumulates_slots() {
    let mut q = AsyncUploadQueue::new();
    assert_eq!(q.pending(), 0);

    q.stage("math.a", &[1.0, 2.0]).unwrap();
    q.stage("math.b", &[3.0, 4.0]).unwrap();
    assert_eq!(q.pending(), 2);
}

#[test]
fn upload_queue_stage_same_slot_twice_accumulates() {
    let mut q = AsyncUploadQueue::new();
    q.stage("x", &[1.0]).unwrap();
    q.stage("x", &[2.0]).unwrap();
    assert_eq!(q.pending(), 2, "staging the same slot twice should accumulate both entries");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn upload_queue_flush_clears_staged_no_cuda() {
    let mut q = AsyncUploadQueue::new();
    q.stage("x", &[1.0, 2.0]).unwrap();
    let mut reg = DeviceSlotRegistry::new();
    // Non-CUDA flush just clears the queue without uploading.
    q.flush(&mut reg).unwrap();
    assert_eq!(q.pending(), 0, "flush must clear the staged queue");
}

// ── Ring buffer CUDA tests ────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod ring_buffer_cuda {
    use quantale_semiring_v2::{DeviceRingBuffer, TensorQuantaleWorld};

    const MODULE: &str = "quantale_semiring_v2_tensor";

    /// Returns (world, dev) or skips the test.
    macro_rules! world_or_skip {
        () => {{
            match TensorQuantaleWorld::empty() {
                Ok(w) => {
                    let dev = w.device().clone();
                    (w, dev)
                }
                Err(e) => {
                    eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                    return;
                }
            }
        }};
    }

    #[test]
    fn ring_push_then_pop_returns_same_values() {
        let (world, dev) = world_or_skip!();
        let capacity = 16;
        let mut ring = DeviceRingBuffer::new(&dev, capacity).expect("ring alloc");
        let src_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let src_buf  = dev.htod_copy(src_data.clone()).expect("htod src");

        ring.push(&dev, MODULE, &src_buf).expect("push");
        let dst = ring.pop(&dev, MODULE, src_data.len()).expect("pop");
        let result = dev.dtoh_sync_copy(&dst).expect("dtoh");

        assert_eq!(result, src_data, "push then pop must return identical values");
        drop(world);
    }

    #[test]
    fn ring_push_pop_fifo_order() {
        let (world, dev) = world_or_skip!();
        let capacity = 32;
        let mut ring = DeviceRingBuffer::new(&dev, capacity).expect("ring alloc");

        let a: Vec<f32> = vec![10.0, 20.0];
        let b: Vec<f32> = vec![30.0, 40.0];
        ring.push(&dev, MODULE, &dev.htod_copy(a.clone()).unwrap()).expect("push a");
        ring.push(&dev, MODULE, &dev.htod_copy(b.clone()).unwrap()).expect("push b");

        let got_a = dev.dtoh_sync_copy(&ring.pop(&dev, MODULE, 2).expect("pop a")).unwrap();
        let got_b = dev.dtoh_sync_copy(&ring.pop(&dev, MODULE, 2).expect("pop b")).unwrap();
        assert_eq!(got_a, a, "FIFO: first push must be first pop");
        assert_eq!(got_b, b, "FIFO: second push must be second pop");
        drop(world);
    }

    #[test]
    fn ring_wraparound_works() {
        let (world, dev) = world_or_skip!();
        // capacity=4, push 3, pop 3, push 3 (wraps around), pop 3
        let capacity = 4;
        let mut ring = DeviceRingBuffer::new(&dev, capacity).expect("ring alloc");

        let first  = vec![1.0f32, 2.0, 3.0];
        let second = vec![7.0f32, 8.0, 9.0];
        ring.push(&dev, MODULE, &dev.htod_copy(first.clone()).unwrap()).unwrap();
        ring.pop(&dev, MODULE, 3).unwrap();
        ring.push(&dev, MODULE, &dev.htod_copy(second.clone()).unwrap()).unwrap();
        let got = dev.dtoh_sync_copy(&ring.pop(&dev, MODULE, 3).unwrap()).unwrap();
        assert_eq!(got, second, "values written after wraparound must be readable");
        drop(world);
    }
}
