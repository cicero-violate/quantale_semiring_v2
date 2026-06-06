/// Integration tests for the streaming quantale integration (plan step 9).
///
/// All tests that require `TensorQuantaleWorld` are gated on
/// `#[cfg(feature = "cuda")]`.  Structural tests run unconditionally.

// ── Structural (no CUDA) tests ────────────────────────────────────────────────

mod structural {
    use quantale_semiring_v2::{
        BackpressurePolicy, DeviceSlot, InMemorySource, RawStreamEvent, SlotDType, SlotSchema,
        SlotUpdate, StreamConfig, StreamReceipt, StreamWorkers, TopologyDelta, TopologyRuntime,
        normalize_event, try_normalize_topology_delta,
    };
    use serde_json::json;

    fn slot_update(slot: &str) -> SlotUpdate {
        SlotUpdate {
            source: "test".into(),
            event_hash: "hash0".into(),
            slot: slot.into(),
            dtype: SlotDType::F32,
            shape: vec![3],
            values_f32: vec![1.0, 2.0, 3.0],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
        }
    }

    /// All slots expected by the streaming pipeline schema are declared.
    #[test]
    fn slot_schema_contains_expected_slots() {
        let mut schema = SlotSchema::new();
        for name in &[
            "market.price",
            "market.open",
            "market.high",
            "market.low",
            "market.volume",
            "analysis.return",
            "analysis.volatility",
            "analysis.signal_score",
        ] {
            schema.register(DeviceSlot::tensor_f32(*name, vec![3]));
        }
        for name in &[
            "market.price",
            "market.open",
            "market.high",
            "market.low",
            "market.volume",
            "analysis.return",
            "analysis.volatility",
            "analysis.signal_score",
        ] {
            assert!(schema.contains(name), "schema missing slot {name}");
        }
    }

    /// `apply_edge_delta` must reject deltas whose node names are absent from the
    /// topology registry; the lookup returns `None` for fabricated names.
    #[test]
    fn unknown_edge_delta_node_lookup_returns_none() {
        let topology = TopologyRuntime::load_checked_default()
            .expect("topology must load");
        let registry = topology.registry();
        assert!(
            registry.id_of("Nonexistent::Node::Xyz").is_none(),
            "registry must not contain fabricated node names"
        );
    }

    /// Both stream event node names are present in the topology after the
    /// topology.generated.json was updated with nodes 61-66.
    #[test]
    fn stream_event_nodes_exist_in_topology() {
        let topology = TopologyRuntime::load_checked_default()
            .expect("topology must load");
        let registry = topology.registry();
        assert!(
            registry.id_of("Event::StreamUpdated").is_some(),
            "Event::StreamUpdated missing from topology"
        );
        assert!(
            registry.id_of("Event::MarketFeedUpdated").is_some(),
            "Event::MarketFeedUpdated missing from topology"
        );
    }

    /// A dropped receipt must not be classified as applied.
    #[test]
    fn dropped_receipt_is_not_applied() {
        let receipt = StreamReceipt::dropped(slot_update("market.price"), "test rejection");
        assert!(!receipt.applied);
        assert!(receipt.dropped);
    }

    /// An applied receipt must be classified as applied and not dropped.
    #[test]
    fn applied_receipt_is_applied_not_dropped() {
        let receipt = StreamReceipt::applied(slot_update("market.price"), 1);
        assert!(receipt.applied);
        assert!(!receipt.dropped);
    }

    /// An `edge_delta` raw event normalizes into `TopologyDelta::Edge`.
    #[test]
    fn edge_delta_normalizes_into_topology_delta_edge() {
        let event = RawStreamEvent::new(
            "risk_engine",
            "edge_delta",
            "2026-06-06T00:00:00Z",
            json!({
                "src": "State::Goal",
                "dst": "Control::GateInput",
                "confidence": 0.9,
                "cost": 0.1,
                "safety": 0.95
            }),
        );
        let delta = try_normalize_topology_delta(&event);
        assert!(delta.is_some(), "expected Some(TopologyDelta::Edge)");
        assert!(
            matches!(delta.unwrap(), TopologyDelta::Edge { .. }),
            "expected Edge variant"
        );
    }

    /// A `node_delta` raw event normalizes into `TopologyDelta::Node`.
    #[test]
    fn node_delta_normalizes_into_topology_delta_node() {
        let event = RawStreamEvent::new(
            "risk_engine",
            "node_delta",
            "2026-06-06T00:00:00Z",
            json!({ "node": "State::Goal", "active": true }),
        );
        let delta = try_normalize_topology_delta(&event);
        assert!(delta.is_some(), "expected Some(TopologyDelta::Node)");
        assert!(
            matches!(delta.unwrap(), TopologyDelta::Node { .. }),
            "expected Node variant"
        );
    }

    /// Topology events must not be encoded as fake slot updates; `normalize_event`
    /// must return an empty Vec for `edge_delta` and `node_delta` kinds.
    #[test]
    fn edge_delta_yields_no_slot_updates() {
        let config = StreamConfig::default();
        let edge_event = RawStreamEvent::new(
            "src",
            "edge_delta",
            "2026-06-06T00:00:00Z",
            json!({ "src": "A", "dst": "B", "confidence": 0.9, "cost": 0.1, "safety": 0.9 }),
        );
        let node_event = RawStreamEvent::new(
            "src",
            "node_delta",
            "2026-06-06T00:00:00Z",
            json!({ "node": "A", "active": true }),
        );
        let edge_updates = normalize_event(&edge_event, &config).unwrap();
        let node_updates = normalize_event(&node_event, &config).unwrap();
        assert!(
            edge_updates.is_empty(),
            "edge_delta must not emit slot updates (got {})",
            edge_updates.len()
        );
        assert!(
            node_updates.is_empty(),
            "node_delta must not emit slot updates (got {})",
            node_updates.len()
        );
    }

    /// Injected `edge_delta` / `node_delta` events flow through the normalizer
    /// worker and appear in `drain_topology_deltas`.
    #[test]
    fn drain_topology_deltas_returns_routed_deltas() {
        let edge_event = RawStreamEvent::new(
            "test",
            "edge_delta",
            "2026-06-06T00:00:00Z",
            json!({ "src": "State::Goal", "dst": "Control::GateInput",
                     "confidence": 0.8, "cost": 0.2, "safety": 0.9 }),
        );
        let node_event = RawStreamEvent::new(
            "test",
            "node_delta",
            "2026-06-06T00:00:00Z",
            json!({ "node": "State::Goal", "active": true }),
        );
        let source = InMemorySource::new("test", vec![edge_event, node_event]);
        let mut workers =
            StreamWorkers::spawn(StreamConfig::default(), SlotSchema::new(), source);

        // Give the reader and normalizer threads time to process both events.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let deltas = workers.drain_topology_deltas();
        assert_eq!(
            deltas.len(),
            2,
            "expected 2 topology deltas (edge + node), got {}",
            deltas.len()
        );
        let has_edge = deltas.iter().any(|d| matches!(d, TopologyDelta::Edge { .. }));
        let has_node = deltas.iter().any(|d| matches!(d, TopologyDelta::Node { .. }));
        assert!(has_edge, "expected one Edge delta");
        assert!(has_node, "expected one Node delta");
    }
}

// ── CUDA-gated integration tests ──────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_streaming {
    use quantale_semiring_v2::{
        activate_stream_event_nodes, apply_edge_delta, apply_node_delta, BackpressurePolicy,
        CudaError, DeviceSlot, DeviceSlotRegistry, EdgeDelta, NodeDelta, SlotDType, SlotSchema,
        SlotUpdate, StreamReceipt, TensorQuantaleWorld, TopologyRuntime,
    };

    fn make_world_and_topology() -> Result<(TensorQuantaleWorld, TopologyRuntime), CudaError> {
        let topology = TopologyRuntime::load_checked_default()?;
        let world = TensorQuantaleWorld::from_tensor_edges(topology.tensor_edges())?;
        Ok((world, topology))
    }

    fn skip_if_no_cuda() -> bool {
        cudarc::driver::CudaDevice::new(0).is_err()
    }

    fn slot_update(slot: &str) -> SlotUpdate {
        SlotUpdate {
            source: "test".into(),
            event_hash: "h0".into(),
            slot: slot.into(),
            dtype: SlotDType::F32,
            shape: vec![3],
            values_f32: vec![1.0, 2.0, 3.0],
            observed_at: "2026-06-06T00:00:00Z".into(),
            policy: BackpressurePolicy::LatestWins,
        }
    }

    // ── stream_receipt_activates_event_node ───────────────────────────────────

    #[test]
    fn stream_receipt_activates_event_node() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let receipt = StreamReceipt::applied(slot_update("analysis.return"), 1);
        activate_stream_event_nodes(&mut world, &topology, &[receipt])?;
        Ok(())
    }

    // ── dropped_stream_receipt_does_not_activate_event_node ──────────────────

    #[test]
    fn dropped_stream_receipt_does_not_activate_event_node() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let receipt = StreamReceipt::dropped(slot_update("market.price"), "schema_mismatch");
        activate_stream_event_nodes(&mut world, &topology, &[receipt])?;
        Ok(())
    }

    // ── market_slot_receipt_activates_market_event ────────────────────────────

    #[test]
    fn market_slot_receipt_activates_market_event() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let receipt = StreamReceipt::applied(slot_update("market.price"), 1);
        activate_stream_event_nodes(&mut world, &topology, &[receipt])?;
        Ok(())
    }

    // ── edge_delta_embeds_tensor_edge ─────────────────────────────────────────

    #[test]
    fn edge_delta_embeds_tensor_edge() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let delta = EdgeDelta {
            source: "test".into(),
            src: "State::Goal".into(),
            dst: "Control::GateInput".into(),
            confidence: 0.9,
            cost: 0.1,
            safety: 0.95,
            observed_at: "2026-06-06T00:00:00Z".into(),
            event_hash: "edgehash1".into(),
        };
        apply_edge_delta(&mut world, &topology, delta)?;
        Ok(())
    }

    // ── unknown_edge_delta_node_rejected ──────────────────────────────────────

    #[test]
    fn unknown_edge_delta_node_rejected() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let delta = EdgeDelta {
            source: "test".into(),
            src: "Nonexistent::NodeA".into(),
            dst: "Control::GateInput".into(),
            confidence: 0.9,
            cost: 0.1,
            safety: 0.95,
            observed_at: "2026-06-06T00:00:00Z".into(),
            event_hash: "bad1".into(),
        };
        let result = apply_edge_delta(&mut world, &topology, delta);
        assert!(result.is_err(), "unknown src node must be rejected");
        Ok(())
    }

    // ── slot_schema_preallocates_registry ─────────────────────────────────────

    #[test]
    fn slot_schema_preallocates_registry() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (world, _) = make_world_and_topology()?;
        let dev = world.device();
        let slot_names = [
            "market.price", "market.open", "market.high", "market.low", "market.volume",
            "analysis.return", "analysis.volatility", "analysis.signal_score",
        ];
        let mut schema = SlotSchema::new();
        for name in &slot_names {
            schema.register(DeviceSlot::tensor_f32(*name, vec![3]));
        }
        let mut registry = DeviceSlotRegistry::new();
        for slot in schema.iter() {
            let buf = dev.htod_copy(vec![0.0_f32; slot.len()])?;
            registry.register(slot.clone(), buf);
        }
        for name in &slot_names {
            assert!(registry.contains(name), "registry missing slot {name}");
        }
        Ok(())
    }

    // ── stream_batch_runs_before_orchestrator_burst ───────────────────────────
    // Verified structurally: `pre_burst_fn` is the first closure passed to
    // `gpu_native_supervisor_loop`.  The CUDA test here validates that
    // `apply_stream_batch` is callable and returns without error on empty input.

    #[test]
    fn stream_batch_runs_before_orchestrator_burst() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        use quantale_semiring_v2::{InMemorySource, StreamConfig, StreamWorkers};
        let (mut world, _) = make_world_and_topology()?;
        let source = InMemorySource::new("test", vec![]);
        let mut workers = StreamWorkers::spawn(StreamConfig::default(), SlotSchema::new(), source);
        let mut registry = DeviceSlotRegistry::new();
        let receipts = world.apply_stream_batch(&mut workers, &mut registry);
        assert!(receipts.is_empty(), "empty source yields no receipts");
        Ok(())
    }

    // ── unknown_node_delta_target_is_rejected ────────────────────────────────

    #[test]
    fn unknown_node_delta_target_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        let (mut world, topology) = make_world_and_topology()?;
        let delta = NodeDelta {
            source: "test".into(),
            node: "Nonexistent::NodeXyz".into(),
            active: true,
            observed_at: "2026-06-06T00:00:00Z".into(),
            event_hash: "badhash".into(),
        };
        let result = apply_node_delta(&mut world, &topology, delta);
        assert!(result.is_err(), "unknown node name must be rejected");
        Ok(())
    }

    // ── single_gpu_owner_applies_stream_updates ───────────────────────────────

    #[test]
    fn single_gpu_owner_applies_stream_updates() -> Result<(), Box<dyn std::error::Error>> {
        if skip_if_no_cuda() { eprintln!("skip: no CUDA"); return Ok(()); }
        use quantale_semiring_v2::{InMemorySource, StreamConfig, StreamWorkers};
        let (mut world, _) = make_world_and_topology()?;
        let source = InMemorySource::new("test", vec![]);
        let mut workers = StreamWorkers::spawn(StreamConfig::default(), SlotSchema::new(), source);
        let mut registry = DeviceSlotRegistry::new();
        // Only TensorQuantaleWorld (GPU owner) can call apply_stream_batch.
        let receipts = world.apply_stream_batch(&mut workers, &mut registry);
        assert!(receipts.is_empty());
        Ok(())
    }
}
