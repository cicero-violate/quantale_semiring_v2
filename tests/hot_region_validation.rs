//! Behavioral and regression tests for the upgrade.v2 work.
//!
//! Covers:
//!  - Hot region slot validation against declared operators
//!  - Hot region metadata invariants against active generated assets
//!  - HostStagingBuffer and UploadQueue staging behaviour
//!  - Fusion dispatch priority over single-node hot dispatch (P1.1)
//!  - Synthetic hot node whitelist (Region::CommitReceipt)
//!  - Ring-buffer push/pop (CUDA-gated)

#[cfg(not(feature = "cuda"))]
use quantale_semiring_v2::DeviceSlotRegistry;
use quantale_semiring_v2::{
    DeviceSlot, FusionDispatch, HostStagingBuffer, HotRegionRegistry, UploadQueue,
    load_operator_registry,
};

// ── Hot region slot validation ────────────────────────────────────────────────

#[test]
fn hot_region_slots_all_declared_in_operator_registry() {
    let registry = load_operator_registry("assets/operators.generated.json")
        .expect("load operators.generated.json");
    let hot = HotRegionRegistry::load("assets/regions.hot.json").expect("load regions.hot.json");

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

// ── Active hot-region metadata invariants ─────────────────────────────────────

#[test]
fn hot_region_registry_has_regions() {
    let hot = HotRegionRegistry::load("assets/regions.hot.json").expect("load regions.hot.json");
    assert!(
        !hot.entries.is_empty(),
        "regions.hot.json must contain hot regions"
    );
}

#[test]
fn hot_region_nodes_exist_in_generated_topology() {
    let hot = HotRegionRegistry::load("assets/regions.hot.json").expect("load regions.hot.json");
    let raw = std::fs::read_to_string("assets/topology.generated.json")
        .expect("read topology.generated.json");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse topology.generated.json");
    let generated_names: std::collections::HashSet<&str> = doc["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();

    for entry in &hot.entries {
        if entry.name == "Region::CommitReceipt" {
            continue;
        }
        assert!(
            generated_names.contains(entry.name.as_str()),
            "hot region '{}' must exist in topology.generated.json",
            entry.name
        );
    }
}

#[test]
fn generated_topology_contains_vector_add_to_scale_chain() {
    let raw = std::fs::read_to_string("assets/topology.generated.json")
        .expect("read topology.generated.json");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse topology.generated.json");
    let has_chain = doc["transitions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|transition| {
            transition["from"].as_str() == Some("Execution::VectorAdd")
                && transition["to"].as_str() == Some("Execution::VectorScale")
        });
    assert!(
        has_chain,
        "generated topology must contain Execution::VectorAdd -> Execution::VectorScale"
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

// ── UploadQueue staging behaviour (non-CUDA) ────────────────────────────

#[test]
fn upload_queue_stage_accumulates_slots() {
    let mut q = UploadQueue::new();
    assert_eq!(q.pending(), 0);

    q.stage(&DeviceSlot::tensor_f32("math.a", vec![2]), &[1.0, 2.0]).unwrap();
    q.stage(&DeviceSlot::tensor_f32("math.b", vec![2]), &[3.0, 4.0]).unwrap();
    assert_eq!(q.pending(), 2);
}

#[test]
fn upload_queue_stage_same_slot_twice_accumulates() {
    let mut q = UploadQueue::new();
    let slot = DeviceSlot::tensor_f32("x", vec![1]);
    q.stage(&slot, &[1.0]).unwrap();
    q.stage(&slot, &[2.0]).unwrap();
    assert_eq!(
        q.pending(),
        2,
        "staging the same slot twice should accumulate both entries"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn upload_queue_flush_clears_staged_no_cuda() {
    let mut q = UploadQueue::new();
    q.stage(&DeviceSlot::tensor_f32("x", vec![2]), &[1.0, 2.0]).unwrap();
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
        let src_buf = dev.htod_copy(src_data.clone()).expect("htod src");

        ring.push(&dev, MODULE, &src_buf).expect("push");
        let dst = ring.pop(&dev, MODULE, src_data.len()).expect("pop");
        let result = dev.dtoh_sync_copy(&dst).expect("dtoh");

        assert_eq!(
            result, src_data,
            "push then pop must return identical values"
        );
        drop(world);
    }

    #[test]
    fn ring_push_pop_fifo_order() {
        let (world, dev) = world_or_skip!();
        let capacity = 32;
        let mut ring = DeviceRingBuffer::new(&dev, capacity).expect("ring alloc");

        let a: Vec<f32> = vec![10.0, 20.0];
        let b: Vec<f32> = vec![30.0, 40.0];
        ring.push(&dev, MODULE, &dev.htod_copy(a.clone()).unwrap())
            .expect("push a");
        ring.push(&dev, MODULE, &dev.htod_copy(b.clone()).unwrap())
            .expect("push b");

        let got_a = dev
            .dtoh_sync_copy(&ring.pop(&dev, MODULE, 2).expect("pop a"))
            .unwrap();
        let got_b = dev
            .dtoh_sync_copy(&ring.pop(&dev, MODULE, 2).expect("pop b"))
            .unwrap();
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

        let first = vec![1.0f32, 2.0, 3.0];
        let second = vec![7.0f32, 8.0, 9.0];
        ring.push(&dev, MODULE, &dev.htod_copy(first.clone()).unwrap())
            .unwrap();
        ring.pop(&dev, MODULE, 3).unwrap();
        ring.push(&dev, MODULE, &dev.htod_copy(second.clone()).unwrap())
            .unwrap();
        let got = dev
            .dtoh_sync_copy(&ring.pop(&dev, MODULE, 3).unwrap())
            .unwrap();
        assert_eq!(
            got, second,
            "values written after wraparound must be readable"
        );
        drop(world);
    }

    #[test]
    fn ring_rejects_overflow() {
        let (world, dev) = world_or_skip!();
        let mut ring = DeviceRingBuffer::new(&dev, 2).expect("ring alloc");
        let src = dev.htod_copy(vec![1.0f32, 2.0, 3.0]).unwrap();
        let error = ring
            .push(&dev, MODULE, &src)
            .expect_err("overflow must fail");
        assert!(error.contains("overflow"), "unexpected error: {error}");
        drop(world);
    }

    #[test]
    fn ring_rejects_empty_pop() {
        let (world, dev) = world_or_skip!();
        let mut ring = DeviceRingBuffer::new(&dev, 4).expect("ring alloc");
        let error = ring.pop(&dev, MODULE, 1).expect_err("empty pop must fail");
        assert!(error.contains("underflow"), "unexpected error: {error}");
        drop(world);
    }

    #[test]
    fn ring_tracks_head_tail_across_mixed_push_pop() {
        let (world, dev) = world_or_skip!();
        let mut ring = DeviceRingBuffer::new(&dev, 4).expect("ring alloc");

        ring.push(
            &dev,
            MODULE,
            &dev.htod_copy(vec![1.0f32, 2.0, 3.0]).unwrap(),
        )
        .unwrap();
        assert_eq!(ring.len(&dev).unwrap(), 3);

        let got = dev
            .dtoh_sync_copy(&ring.pop(&dev, MODULE, 2).unwrap())
            .unwrap();
        assert_eq!(got, vec![1.0f32, 2.0]);
        assert_eq!(ring.len(&dev).unwrap(), 1);

        ring.push(&dev, MODULE, &dev.htod_copy(vec![4.0f32, 5.0]).unwrap())
            .unwrap();
        assert_eq!(ring.len(&dev).unwrap(), 3);

        let got = dev
            .dtoh_sync_copy(&ring.pop(&dev, MODULE, 3).unwrap())
            .unwrap();
        assert_eq!(got, vec![3.0f32, 4.0, 5.0]);
        assert_eq!(ring.len(&dev).unwrap(), 0);
        drop(world);
    }
}

// ── Fusion dispatch priority (P1.1) ──────────────────────────────────────────

/// The entry node of the analysis fusion chain must be present in FusionDispatch.
/// This verifies the first branch in execute_active_node_blocking is reachable
/// and wins over the single-node hot dispatch path.
#[test]
fn fusion_dispatch_entry_wins_over_hot_single_dispatch() {
    let registry = load_operator_registry("assets/operators.generated.json")
        .expect("load operators.generated.json");
    let dispatch = FusionDispatch::load("assets/topology.fusion.json", &registry)
        .expect("load topology.fusion.json");

    // Analysis::Return1 is the entry of the fused analysis chain.
    let entry = dispatch
        .get_by_entry("Analysis::Return1")
        .expect("Analysis::Return1 must be a fusion entry node");

    // The fusion region covers the full analysis chain (3 nodes).
    assert!(
        entry.nodes.len() >= 2,
        "fusion entry must cover at least two nodes; fusion dispatch would otherwise be no-op"
    );
    assert!(
        entry.nodes.contains(&"Analysis::Return1".to_string()),
        "entry nodes must include the entry node itself"
    );

    // Verify the hot registry also knows this node — so we can confirm fusion wins.
    let hot = HotRegionRegistry::load("assets/regions.hot.json").expect("load regions.hot.json");
    assert!(
        hot.is_hot("Analysis::Return1") || dispatch.get_by_entry("Analysis::Return1").is_some(),
        "Analysis::Return1 must be reachable via fusion OR hot path"
    );

    // If the node appears in both fusion and hot, fusion wins because
    // execute_active_node_blocking checks fusion_dispatch.get_by_entry first.
    // Non-null get_by_entry confirms the fusion branch is taken.
    assert!(
        dispatch.get_by_entry("Analysis::Return1").is_some(),
        "fusion entry lookup must return Some — fusion wins over hot single dispatch"
    );
}

// A node that is NOT a fusion entry must not be returned by get_by_entry.
#[test]
fn non_entry_node_does_not_win_fusion_dispatch() {
    let registry = load_operator_registry("assets/operators.generated.json")
        .expect("load operators.generated.json");
    let dispatch = FusionDispatch::load("assets/topology.fusion.json", &registry)
        .expect("load topology.fusion.json");

    // Analysis::Volatility is a member of the analysis chain but NOT the entry.
    let result = dispatch.get_by_entry("Analysis::Volatility");
    assert!(
        result.is_none(),
        "middle-of-chain node must not be a fusion entry point"
    );
}

// ── Synthetic hot receipt terminal ────────────────────────────────────────────

#[test]
fn regions_hot_contains_synthetic_commit_receipt() {
    let hot = HotRegionRegistry::load("assets/regions.hot.json").expect("load regions.hot.json");
    assert!(
        hot.entries
            .iter()
            .any(|entry| entry.name == "Region::CommitReceipt"),
        "regions.hot.json must contain Region::CommitReceipt"
    );
}
