#[cfg(feature = "cuda")]
mod cuda_smoke {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use quantale_semiring_v2::{
        DISPATCH_KIND_EXTERNAL_IO, DISPATCH_KIND_EXTERNAL_PROCESS, DISPATCH_KIND_HF_DEVICE,
        DeviceSlot, DeviceSlotRegistry, FusionHfCoverage, JitCache, JitChain, OperatorRegistry, OrchStepStatus,
        PAR_DISPATCH_ABSTRACT_DEVICE, PAR_DISPATCH_HF_DEVICE, ProjectionBias, SystemConfig,
        TENSOR_NODE_COUNT, TensorEdge, TensorQuantaleWorld, TopologyRuntime, UploadQueue,
        build_node_dispatch_kinds, load_operator_registry,
    };

    #[test]
    fn cuda_feature_runtime_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };

        let registry = load_operator_registry("assets/operators.generated.json")?;
        let chain = JitChain {
            operators: vec!["Analysis::Return1".to_string()],
            inputs: vec!["market.price".to_string(), "market.open".to_string()],
            outputs: vec!["analysis.return".to_string()],
            internals: vec![],
        };

        let func = JitCache::new()
            .get_or_compile(&device, &chain, &registry as &OperatorRegistry)
            .map_err(|e| format!("jit compile failed: {e}"))?;
        let price = device.htod_copy(vec![110.0f32, 120.0, 90.0, 105.0])?;
        let open = device.htod_copy(vec![100.0f32, 100.0, 100.0, 100.0])?;
        let mut out = device.htod_copy(vec![0.0f32; 4])?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, (&price, &open, &mut out, 4_i32))? };
        let actual = device.dtoh_sync_copy(&out)?;
        let expected = [0.1f32, 0.2, -0.1, 0.05];
        for (idx, (&a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 1e-5, "idx={idx} actual={a} expected={e}");
        }
        Ok(())
    }

    #[test]
    fn gpu_dispatch_region_uses_device_slot_registry() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        let device = world.device().clone();

        let mut slots = DeviceSlotRegistry::new();
        slots.insert("math.a", device.htod_copy(vec![1.0f32, 2.0, 3.0, 4.0])?);
        slots.insert("math.b", device.htod_copy(vec![10.0f32, 20.0, 30.0, 40.0])?);
        slots.insert("math.add_out", device.htod_copy(vec![0.0f32; 4])?);

        world.gpu_dispatch_region_with_slots(&slots, 0, 0, 1, 0)?;

        let actual = device.dtoh_sync_copy(slots.get("math.add_out").unwrap())?;
        assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    fn generated_fixture_hf_fragment() -> String {
        [
            "// region: Fixture::Add__Fixture::Scale",
            "// nodes: Fixture::Add, Fixture::Scale",
            "// reads: fixture.a, fixture.b, fixture.scale",
            "// writes: fixture.out",
            "// hf_case: 8 region_fusion_stub_fixture_add_fixture_scale",
            "__device__ void region_fusion_stub_fixture_add_fixture_scale(float** slot_ptrs, int n, DeviceReceipt* r) {",
            "    if (!slot_ptrs || n <= 0) return;",
            "    float* slot_0 = slot_ptrs[0];",
            "    float* slot_1 = slot_ptrs[1];",
            "    float* slot_2 = slot_ptrs[2];",
            "    float* slot_3 = slot_ptrs[3];",
            "    for (int i = threadIdx.x; i < n; i += blockDim.x) {",
            "        float reg_fixture_tmp = slot_0[i] + slot_1[i];",
            "        slot_3[i] = reg_fixture_tmp * slot_2[i];",
            "    }",
            "    r->outcome = 0;",
            "}",
            "",
        ]
        .join("\n")
    }

    fn generated_fixture_coverage() -> Result<FusionHfCoverage, Box<dyn std::error::Error>> {
        Ok(FusionHfCoverage::from_json_str(
            r#"{
                "schema":"fusion_hf_coverage.v1",
                "regions":[{
                    "region":"Fixture::Add__Fixture::Scale",
                    "entry":"Fixture::Add",
                    "nodes":["Fixture::Add","Fixture::Scale"],
                    "hf_region_id":8,
                    "covered":true,
                    "reason":"generated_hf_handler",
                    "symbol":"region_fusion_stub_fixture_add_fixture_scale",
                    "slots":["fixture.a","fixture.b","fixture.scale","fixture.out"]
                }]
            }"#,
        )?)
    }

    #[test]
    fn generated_hf_region_id_eight_runs_through_par_group_step()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty_with_generated_fusion_hf_fragments(
            &generated_fixture_hf_fragment(),
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld generated H_f compile: {e}");
                return Ok(());
            }
        };
        let device = world.device().clone();
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(0, 2, 0.8, 1.0, 0.8),
        ])?;

        let mut slots = DeviceSlotRegistry::new();
        slots.insert("fixture.a", device.htod_copy(vec![1.0f32, 2.0, 3.0, 4.0])?);
        slots.insert(
            "fixture.b",
            device.htod_copy(vec![10.0f32, 20.0, 30.0, 40.0])?,
        );
        slots.insert(
            "fixture.scale",
            device.htod_copy(vec![2.0f32, 3.0, 4.0, 5.0])?,
        );
        slots.insert("fixture.out", device.htod_copy(vec![0.0f32; 4])?);
        let coverage = generated_fixture_coverage()?;
        let data = world.make_par_group_data(
            &[vec![1, 2]],
            &[vec![8, 8]],
            &[vec![true, true]],
            &[vec![PAR_DISPATCH_HF_DEVICE, PAR_DISPATCH_HF_DEVICE]],
            Some(&slots),
            Some(&coverage),
        )?;
        let Some((_group, decisions, region_ids, dispatched, descriptors)) =
            world.par_group_step(&data, ProjectionBias::default())?
        else {
            panic!("expected generated H_f par group to be selected");
        };
        assert_eq!(decisions.len(), 2);
        assert_eq!(region_ids, vec![8, 8]);
        assert_eq!(dispatched, vec![1, 1]);
        assert!(
            descriptors
                .iter()
                .all(|d| d.dispatch_kind == PAR_DISPATCH_HF_DEVICE)
        );

        let actual = device.dtoh_sync_copy(slots.get("fixture.out").unwrap())?;
        assert_eq!(actual, vec![22.0, 66.0, 132.0, 220.0]);
        world.drain_device_receipts()?;
        Ok(())
    }

    #[test]
    fn abstract_device_par_member_writes_device_receipt_on_gpu()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(0, 2, 0.8, 1.0, 0.8),
        ])?;
        let data = world.make_par_group_data(
            &[vec![1, 2]],
            &[vec![-1, -1]],
            &[vec![true, true]],
            &[vec![
                PAR_DISPATCH_ABSTRACT_DEVICE,
                PAR_DISPATCH_ABSTRACT_DEVICE,
            ]],
            None,
            None,
        )?;
        let Some((_group, _decisions, region_ids, dispatched, descriptors)) =
            world.par_group_step(&data, ProjectionBias::default())?
        else {
            panic!("expected abstract-device par group to be selected");
        };
        assert_eq!(region_ids, vec![-1, -1]);
        assert_eq!(dispatched, vec![1, 1]);
        assert!(
            descriptors
                .iter()
                .all(|d| d.dispatch_kind == PAR_DISPATCH_ABSTRACT_DEVICE)
        );
        world.drain_device_receipts()?;
        Ok(())
    }

    // ── Phase-1 CUDA smoke tests ──────────────────────────────────────────────

    /// Phase-1: `orchestration_state_init` zeroes all state fields on the GPU.
    #[test]
    fn orchestration_state_init_sets_zero_state() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::OrchestrationState;
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        let state = world.orch_state_snapshot()?;
        assert_eq!(state.step, 0, "step should be 0 after init");
        assert_eq!(state.halted, 0);
        assert_eq!(state.blocked, 0);
        assert_eq!(
            state.selected_group, -1,
            "selected_group should be -1 (none)"
        );
        assert_eq!(state.selected_node, -1, "selected_node should be -1 (none)");
        assert_eq!(state.pending_external_count, 0);
        assert_eq!(state.pending_receipt_count, 0);
        assert_eq!(state.failure_count, 0);
        assert_eq!(state.rollback_requested, 0);
        assert_eq!(state.star_bound, 0, "Phase-4: star_bound should be 0");
        assert_eq!(
            state,
            OrchestrationState {
                step: 0,
                halted: 0,
                blocked: 0,
                current_frontier_epoch: 0,
                selected_group: -1,
                selected_node: -1,
                pending_external_count: 0,
                pending_receipt_count: 0,
                failure_count: 0,
                rollback_requested: 0,
                star_bound: 0,
                consecutive_blocks: 0,
                block_threshold: 0,
                hard_reset_requested: 0,
                rollback_available: 0,
                failure_action: 0,
                selected_src: -1,
                selected_dst: -1,
                selected_control_edge: -1,
                selected_control_op: -1,
                selected_control_lhs: -1,
                selected_control_rhs: -1,
                control_epoch: 0,
                star_counter_epoch: 0,
                last_block_reason: 0,
            }
        );
        Ok(())
    }

    /// Phase-1: device command ring FIFO semantics — push 3 commands, drain them,
    /// verify order and that the ring wraps correctly (overflow guard).
    #[test]
    fn command_ring_push_pop_fifo() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::DeviceCommand;
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        for i in 0..3_i32 {
            world.push_device_command(DeviceCommand {
                valid: 1,
                command_id: i,
                node_id: 10 + i,
                src: i,
                dst: i + 1,
                dispatch_kind: DISPATCH_KIND_EXTERNAL_PROCESS,
                operator_name_id: 10 + i,
                timeout_ticks: 0,
                retry_budget: 1,
                payload_offset: 0,
                payload_len: 0,
            })?;
        }
        let cmds = world.drain_device_commands()?;
        assert_eq!(cmds.len(), 3, "expected 3 commands in FIFO order");
        for (idx, cmd) in cmds.iter().enumerate() {
            assert_eq!(
                cmd.command_id, idx as i32,
                "FIFO order violated at slot {idx}"
            );
            assert_eq!(cmd.node_id, 10 + idx as i32);
        }
        // After drain the ring should be empty.
        let cmds2 = world.drain_device_commands()?;
        assert!(cmds2.is_empty(), "ring should be empty after full drain");
        Ok(())
    }

    /// Phase-1: extended receipt ring FIFO semantics — push 2 receipts, drain
    /// via `drain_device_receipt_ext`, verify tensor was updated.
    #[test]
    fn receipt_ext_ring_push_pop_fifo() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{DeviceReceiptExt, TensorEdge};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[TensorEdge::new(0, 1, 0.5, 5.0, 0.5)])?;
        assert_eq!(world.orch_state_snapshot()?.pending_receipt_count, 0);

        // Push a success receipt for edge 0→1.
        world.push_device_receipt_ext(DeviceReceiptExt {
            valid: 1,
            consumed: 0,
            command_id: 42,
            node_id: 1,
            src: 0,
            dst: 1,
            outcome: 0, // success
            receipt_kind: 0,
            output_flags: 0,
            latency: 0.0,
        })?;
        assert_eq!(world.orch_state_snapshot()?.pending_receipt_count, 1);
        // Push a failure receipt for edge 0→2 (won't affect tensor — no edge embedded).
        world.push_device_receipt_ext(DeviceReceiptExt {
            valid: 1,
            consumed: 0,
            command_id: 43,
            node_id: 2,
            src: 0,
            dst: 2,
            outcome: 1, // failure
            receipt_kind: 0,
            output_flags: 0,
            latency: 0.0,
        })?;
        assert_eq!(world.orch_state_snapshot()?.pending_receipt_count, 2);
        // Drain: success receipt should set confidence[0,1] = 1.0.
        world.drain_device_receipt_ext()?;
        assert_eq!(world.orch_state_snapshot()?.pending_receipt_count, 0);
        // Verify using the existing tensor snapshot helper.
        let tensor = world.tensor()?;
        // Confidence layer 0: entry [0,1] = tensor[0 * N*N + 0*N + 1].
        use quantale_semiring_v2::{MATRIX_LEN, TENSOR_NODE_COUNT};
        let conf_01 = tensor[0 * MATRIX_LEN + 0 * TENSOR_NODE_COUNT + 1];
        assert!(
            (conf_01 - 1.0f32).abs() < 1e-5,
            "success receipt should have set confidence[0,1] = 1.0, got {conf_01}"
        );
        Ok(())
    }

    // ── Phase-2 CUDA smoke tests ──────────────────────────────────────────────

    /// Phase-2: a pure GPU-native graph (two HF_DEVICE nodes) runs to ORCH_CONTINUE
    /// with the scheduler selecting the highest-score ready node without CPU involvement.
    #[test]
    fn scheduler_selects_first_ready_singleton() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        // Embed two edges: node 0→1 (high confidence) and node 0→2 (lower).
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(0, 2, 0.5, 1.0, 0.5),
        ])?;
        // All nodes are HF_DEVICE by default. Step once.
        let status = world.orchestrate_step()?;
        assert_ne!(
            status,
            OrchStepStatus::Error,
            "orchestrate_step returned Error"
        );
        // Should select and commit a step — state.step incremented.
        let state = world.orch_state_snapshot()?;
        assert_eq!(
            state.step, 1,
            "state.step should be 1 after one orchestrate_step"
        );
        assert_eq!(
            state.selected_node, 1,
            "scheduler should select node 1 (higher score)"
        );
        Ok(())
    }

    /// Phase-2: a node classified as EXTERNAL_PROCESS causes the scheduler to
    /// emit a DeviceCommand and return ORCH_WAIT_EXTERNAL.
    #[test]
    fn scheduler_emits_external_command_for_process_node() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[TensorEdge::new(0, 1, 0.8, 1.0, 0.8)])?;
        // Mark node 1 as external process.
        let mut kinds = vec![DISPATCH_KIND_HF_DEVICE; TENSOR_NODE_COUNT];
        kinds[1] = DISPATCH_KIND_EXTERNAL_PROCESS;
        world.set_dispatch_kinds(&kinds)?;

        let status = world.orchestrate_step()?;
        assert_eq!(
            status,
            OrchStepStatus::WaitExternal,
            "expected ORCH_WAIT_EXTERNAL for external node"
        );
        // Exactly one command should have been emitted.
        let cmds = world.drain_device_commands()?;
        assert_eq!(
            cmds.len(),
            1,
            "expected exactly 1 command for external node"
        );
        assert_eq!(cmds[0].node_id, 1);
        assert_eq!(cmds[0].dispatch_kind, DISPATCH_KIND_EXTERNAL_PROCESS);
        Ok(())
    }

    #[test]
    fn default_dispatch_table_reaches_market_feed_external_command()
    -> Result<(), Box<dyn std::error::Error>> {
        let topology = TopologyRuntime::load_checked_default()?;
        let goal = topology.registry().id_of("State::Goal").unwrap() as i32;
        let market = topology.registry().id_of("State::MarketFeed").unwrap() as i32;

        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[TensorEdge::new(goal, market, 0.97, 0.03, 0.97)])?;

        let config = SystemConfig::default();
        let kinds = build_node_dispatch_kinds(&topology.document, &config);
        assert_eq!(kinds[market as usize], DISPATCH_KIND_EXTERNAL_IO);
        world.set_dispatch_kinds(&kinds)?;

        let status = world.orchestrate_until_wait_or_halt(4)?;
        assert_eq!(status, OrchStepStatus::WaitExternal);

        let cmds = world.drain_device_commands()?;
        assert!(
            cmds.iter()
                .any(|cmd| cmd.node_id == market && cmd.dispatch_kind == DISPATCH_KIND_EXTERNAL_IO),
            "expected a market-feed external IO command, got {cmds:?}"
        );
        Ok(())
    }

    /// Phase-2: the GPU-native scheduler selects the deterministic highest-score singleton.
    #[test]
    fn gpu_scheduler_selects_expected_singleton() -> Result<(), Box<dyn std::error::Error>> {
        let mut gpu_world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        gpu_world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(0, 2, 0.7, 1.0, 0.7),
            TensorEdge::new(0, 3, 0.5, 1.0, 0.5),
        ])?;

        gpu_world.orchestrate_step()?;
        let gpu_state = gpu_world.orch_state_snapshot()?;

        assert_eq!(gpu_state.selected_node, 1);
        Ok(())
    }

    // ── Phase-5 CUDA smoke tests ──────────────────────────────────────────────

    /// Phase-5: failure_policy_classify_and_emit returns RETRY while budget > 0,
    /// decrementing the budget each call, until it reaches zero and returns BLOCK.
    #[test]
    fn failure_policy_retries_until_budget() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{FAILURE_ACTION_BLOCK, FAILURE_ACTION_RETRY};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        // Arm each node with 2 retries.
        world.failure_policy_init(2, -1)?;

        // First failure → RETRY (budget: 2 → 1).
        let action1 = world.failure_policy_classify_and_emit(1, 0, 0, 1, 100)?;
        assert_eq!(
            action1, FAILURE_ACTION_RETRY,
            "first failure should be RETRY"
        );

        // Second failure → RETRY (budget: 1 → 0).
        let action2 = world.failure_policy_classify_and_emit(1, 0, 0, 1, 101)?;
        assert_eq!(
            action2, FAILURE_ACTION_RETRY,
            "second failure should be RETRY"
        );

        // Third failure → BLOCK (budget: 0, repair_on_block=0).
        let action3 = world.failure_policy_classify_and_emit(1, 0, 0, 1, 102)?;
        assert_eq!(
            action3, FAILURE_ACTION_BLOCK,
            "budget exhausted: should be BLOCK"
        );
        Ok(())
    }

    /// Phase-5: after the retry budget is exhausted the action is BLOCK and
    /// OrchestrationState::consecutive_blocks is incremented.
    #[test]
    fn failure_policy_blocks_after_budget() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{FAILURE_ACTION_BLOCK, FAILURE_ACTION_RETRY};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        // Arm with 1 retry.
        world.failure_policy_init(1, -1)?;

        let a1 = world.failure_policy_classify_and_emit(1, 1, 0, 2, 10)?;
        assert_eq!(a1, FAILURE_ACTION_RETRY);

        let a2 = world.failure_policy_classify_and_emit(1, 1, 0, 2, 11)?;
        assert_eq!(a2, FAILURE_ACTION_BLOCK);

        // consecutive_blocks should have been incremented by the BLOCK action.
        let state = world.orch_state_snapshot()?;
        assert!(
            state.consecutive_blocks >= 1,
            "consecutive_blocks should be >= 1 after a BLOCK action, got {}",
            state.consecutive_blocks
        );
        Ok(())
    }

    /// Phase-5: set_rollback_marker snapshots state; apply_rollback restores it.
    #[test]
    fn failure_policy_rollback_marker_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        // Embed a single edge so the tensor has non-trivial consumed state.
        use quantale_semiring_v2::TensorEdge;
        world.embed_tensor_edges(&[TensorEdge::new(0, 1, 0.9, 1.0, 0.8)])?;
        world.close()?;

        // Take a rollback snapshot before any frontier advancement.
        world.set_rollback_marker()?;
        let state_after_snap = world.orch_state_snapshot()?;
        assert_eq!(
            state_after_snap.rollback_available, 1,
            "rollback_available should be 1"
        );

        // Advance one GPU-native scheduler step (marks consumed[0*N+1]).
        world.orchestrate_step()?;

        // Restore.
        world.apply_rollback()?;
        let state_after_restore = world.orch_state_snapshot()?;
        assert_eq!(
            state_after_restore.rollback_available, 0,
            "rollback_available cleared"
        );
        Ok(())
    }

    // ── Phase-7 smoke tests ───────────────────────────────────────────────────

    /// Phase-7: orchestrate_step now uses the fixed pointer-parameter kernel;
    /// it must not SIGABRT and must increment state.step.
    #[test]
    fn orchestrate_step_no_longer_sigabrt() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(0, 2, 0.5, 1.0, 0.5),
        ])?;
        world.close()?;

        let status = world.orchestrate_step()?;
        assert_ne!(
            status,
            OrchStepStatus::Error,
            "orchestrate_step must not return Error"
        );

        let state = world.orch_state_snapshot()?;
        assert_eq!(
            state.step, 1,
            "state.step should be 1 after one orchestrate_step"
        );
        assert_eq!(
            state.selected_node, 1,
            "should select node 1 (highest score 0.9)"
        );
        Ok(())
    }

    /// Phase-7: orchestrate_until_wait_or_halt respects max_steps and returns
    /// without SIGABRTing or running more iterations than requested.
    #[test]
    fn orchestrate_until_wait_or_halt_respects_max_steps() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        // No edges → immediately blocked on every step.
        let status = world.orchestrate_until_wait_or_halt(10)?;
        // With no edges the scheduler is blocked from the first step; it should
        // return Continue (blocked) after at most one iteration.
        assert_ne!(
            status,
            OrchStepStatus::Error,
            "should not return Error on empty graph"
        );
        Ok(())
    }

    /// Phase-7: GPU-native loop advances the graph across multiple steps without
    /// CPU action selection.  Embed a chain 0→1→2 and let the GPU run through it.
    #[test]
    fn gpu_native_loop_advances_graph_state() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 1.0, 0.9),
            TensorEdge::new(1, 2, 0.8, 1.0, 0.8),
        ])?;
        world.close()?;

        // Let the GPU run up to 5 steps.
        let status = world.orchestrate_until_wait_or_halt(5)?;
        assert_ne!(status, OrchStepStatus::Error, "should not return Error");

        // The scheduler should have made at least one step.
        let state = world.orch_state_snapshot()?;
        assert!(
            state.step >= 1,
            "GPU should have advanced at least one step, got {}",
            state.step
        );
        Ok(())
    }

    /// Phase-6: folding a success receipt increments receipt_priors[node_id].
    #[test]
    fn learned_delta_fold_success_updates_prior() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        world.learned_delta_init()?;

        // Fold a success receipt for node 5, edge 2→5.
        world.learned_delta_fold_receipt(2, 5, 5, 0)?;

        let priors = world.export_receipt_priors()?;
        assert!(
            priors[5] > 0.0,
            "receipt_priors[5] should be > 0 after success fold, got {}",
            priors[5]
        );
        // Other nodes should stay at 0.
        assert_eq!(priors[0], 0.0, "unaffected node 0 should remain 0");
        Ok(())
    }

    /// Phase-6: folding a failure receipt does NOT update receipt_priors (only success does).
    #[test]
    fn learned_delta_fold_failure_does_not_update_prior() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        world.learned_delta_init()?;

        // Fold a failure receipt; prior should remain 0.
        world.learned_delta_fold_receipt(0, 3, 3, 1)?;

        let priors = world.export_receipt_priors()?;
        assert_eq!(
            priors[3], 0.0,
            "receipt_priors[3] should stay 0 after failure fold, got {}",
            priors[3]
        );
        Ok(())
    }

    /// Phase-6: learned_delta_apply applies accumulated soft updates to the tensor.
    #[test]
    fn learned_delta_apply_updates_tensor() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{
            LAYER_CONFIDENCE, LAYER_SAFETY, MATRIX_LEN, TENSOR_NODE_COUNT, TensorEdge,
        };
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };

        // Embed edge 2→3 with moderate confidence and safety.
        world.embed_tensor_edges(&[TensorEdge::new(2, 3, 0.5, 2.0, 0.5)])?;
        world.close()?;

        // Record tensor values before applying deltas.
        let before = world.tensor()?;
        let c_before = before[LAYER_CONFIDENCE as usize * MATRIX_LEN + 2 * TENSOR_NODE_COUNT + 3];
        let s_before = before[LAYER_SAFETY as usize * MATRIX_LEN + 2 * TENSOR_NODE_COUNT + 3];

        // Fold a success receipt (confidence_delta=+0.1, safety_delta=+0.1).
        world.learned_delta_fold_receipt(2, 3, 3, 0)?;

        // Apply the delta ring to the tensor.
        world.learned_delta_apply()?;

        let after = world.tensor()?;
        let c_after = after[LAYER_CONFIDENCE as usize * MATRIX_LEN + 2 * TENSOR_NODE_COUNT + 3];
        let s_after = after[LAYER_SAFETY as usize * MATRIX_LEN + 2 * TENSOR_NODE_COUNT + 3];

        assert!(
            c_after > c_before,
            "confidence[2,3] should increase after success delta: before={c_before}, after={c_after}"
        );
        assert!(
            s_after > s_before,
            "safety[2,3] should increase after success delta: before={s_before}, after={s_after}"
        );
        Ok(())
    }

    #[test]
    fn upload_queue_flushes_pinned_host_data() -> Result<(), Box<dyn std::error::Error>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };

        let mut queue = UploadQueue::new();
        let mut slots = DeviceSlotRegistry::new();
        let slot = DeviceSlot::tensor_f32("math.a", vec![4]);
        queue.stage(&slot, &[1.0, 2.0, 3.0, 4.0])?;
        queue.flush(&mut slots, &device)?;
        assert_eq!(queue.pending(), 0);
        assert_eq!(queue.in_flight(), 1);
        queue.synchronize()?;
        assert_eq!(queue.in_flight(), 0);

        let actual = device.dtoh_sync_copy(slots.get("math.a").unwrap())?;
        assert_eq!(actual, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    // ── Phase-8 smoke tests ───────────────────────────────────────────────────

    /// Phase-8: push an event onto the trace ring, drain it, verify the
    /// drained event matches what was pushed.
    #[test]
    fn trace_ring_push_drain_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{ORCH_EVENT_STEP_COMMITTED, TensorQuantaleWorld};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };
        world.push_trace_event(ORCH_EVENT_STEP_COMMITTED, 0)?;
        let events = world.drain_trace_events()?;
        assert_eq!(events.len(), 1, "expected 1 drained event");
        assert_eq!(events[0].event_kind, ORCH_EVENT_STEP_COMMITTED);
        Ok(())
    }

    /// Phase-8: frontier_valid check returns true on a freshly initialised
    /// world (all active[] values are 0).
    #[test]
    fn invariant_frontier_valid_passes_on_init() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };
        assert!(
            world.check_frontier_valid()?,
            "frontier should be valid on init"
        );
        Ok(())
    }

    /// Phase-8: no-duplicate-receipts check returns true when receipt ring is empty.
    #[test]
    fn invariant_no_duplicate_receipts_passes_on_empty_ring()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };
        assert!(
            world.check_no_duplicate_receipts()?,
            "no duplicates on empty ring"
        );
        Ok(())
    }

    /// Phase-8: replay_snapshot followed by replay_restore is a no-op from the
    /// scheduler's perspective — the state, consumed[], and active[] all match.
    #[test]
    fn replay_snapshot_restore_is_identity() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };
        let before = world.orch_state_snapshot()?;
        world.replay_snapshot()?;
        world.replay_restore()?;
        let after = world.orch_state_snapshot()?;
        assert_eq!(
            before.step, after.step,
            "step must survive snapshot/restore"
        );
        assert_eq!(before.halted, after.halted);
        assert_eq!(before.blocked, after.blocked);
        Ok(())
    }

    // ── Phase-9 scheduler-integrated control-flow tests ──────────────────────
    // These tests exercise SEQ/PAR/CHOICE/STAR through orchestrate_step, making
    // tensor_quantale_orchestrate_step the sole runtime control-flow entrypoint.

    /// Phase-9: a SEQ control edge advances the active frontier through the
    /// GPU scheduler without any CPU control-flow decision.
    #[test]
    fn orchestrate_step_seq_advances_active_frontier() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{CONTROL_OP_SEQ, ControlEdge, OrchStepStatus};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        // Load a single SEQ edge: node 0 → node 1.
        world.load_control_table(
            vec![ControlEdge {
                op: CONTROL_OP_SEQ,
                lhs: 0,
                rhs: 1,
                guard: 0,
                order: 0,
                bound: 0,
            }],
            vec![],
        )?;
        world.embed_tensor_edges(&[TensorEdge::new(0, 1, 0.8, 1.0, 0.8)])?;

        let status = world.orchestrate_step()?;
        assert_ne!(status, OrchStepStatus::Error, "must not error");

        let state = world.orch_state_snapshot()?;
        assert_eq!(state.step, 1, "step should increment");
        assert_eq!(state.selected_node, 1, "SEQ should advance to rhs=1");
        assert_eq!(
            state.selected_control_op, CONTROL_OP_SEQ,
            "selected_control_op must be SEQ"
        );
        assert_eq!(state.selected_control_lhs, 0);
        assert_eq!(state.selected_control_rhs, 1);
        assert_eq!(state.control_epoch, 1, "control_epoch must increment");
        Ok(())
    }

    /// Phase-9: a CHOICE control edge selects the highest-scoring branch
    /// entirely on device.
    #[test]
    fn orchestrate_step_choice_selects_highest_score_branch()
    -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{CONTROL_OP_CHOICE, ControlEdge, OrchStepStatus};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        // Two CHOICE edges from node 0.
        world.load_control_table(
            vec![
                ControlEdge {
                    op: CONTROL_OP_CHOICE,
                    lhs: 0,
                    rhs: 1,
                    guard: 0,
                    order: 0,
                    bound: 0,
                },
                ControlEdge {
                    op: CONTROL_OP_CHOICE,
                    lhs: 0,
                    rhs: 2,
                    guard: 0,
                    order: 1,
                    bound: 0,
                },
            ],
            vec![],
        )?;
        // Node 1 scores significantly higher than node 2.
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.9, 0.1, 0.9),
            TensorEdge::new(0, 2, 0.3, 0.8, 0.3),
        ])?;

        let status = world.orchestrate_step()?;
        assert_ne!(status, OrchStepStatus::Error, "must not error");

        let state = world.orch_state_snapshot()?;
        assert_eq!(state.step, 1);
        assert_eq!(
            state.selected_control_op, CONTROL_OP_CHOICE,
            "selected_control_op must be CHOICE"
        );
        assert_eq!(
            state.selected_node, 1,
            "CHOICE must select the highest-scoring branch (node 1)"
        );
        assert_eq!(state.control_epoch, 1);
        Ok(())
    }

    /// Phase-9: a STAR_BOUNDED control edge increments the per-edge counter and
    /// commits the body step through the GPU scheduler.
    #[test]
    fn orchestrate_step_star_body_increments_counter() -> Result<(), Box<dyn std::error::Error>> {
        use quantale_semiring_v2::{CONTROL_OP_STAR_BOUNDED, ControlEdge, OrchStepStatus};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        // STAR_BOUNDED back-edge: lhs=0 → rhs=1 (body node), bound=3.
        world.load_control_table(
            vec![ControlEdge {
                op: CONTROL_OP_STAR_BOUNDED,
                lhs: 0,
                rhs: 1,
                guard: 0,
                order: 0,
                bound: 3,
            }],
            vec![],
        )?;
        world.embed_tensor_edges(&[TensorEdge::new(0, 1, 0.8, 1.0, 0.8)])?;

        let status = world.orchestrate_step()?;
        assert_ne!(status, OrchStepStatus::Error, "must not error");

        let state = world.orch_state_snapshot()?;
        assert_eq!(state.step, 1);
        assert_eq!(
            state.selected_control_op, CONTROL_OP_STAR_BOUNDED,
            "selected_control_op must be STAR_BOUNDED"
        );
        assert_eq!(state.selected_node, 1, "body node (rhs) must become active");
        assert_eq!(
            state.star_counter_epoch, 1,
            "star_counter_epoch must increment"
        );
        assert_eq!(state.star_bound, 3, "star_bound must reflect edge bound");
        assert_eq!(state.control_epoch, 1);
        Ok(())
    }

    /// Phase-9: PAR control edges with independent effects are committed
    /// together by the GPU scheduler in a single step.
    #[test]
    fn orchestrate_step_par_commits_independent_members() -> Result<(), Box<dyn std::error::Error>>
    {
        use quantale_semiring_v2::{CONTROL_OP_PAR, ControlEdge, EffectTable, OrchStepStatus};
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: {e}");
                return Ok(());
            }
        };
        // Two PAR edges from node 0 to nodes 1 and 2.
        world.load_control_table(
            vec![
                ControlEdge {
                    op: CONTROL_OP_PAR,
                    lhs: 0,
                    rhs: 1,
                    guard: 0,
                    order: 0,
                    bound: 0,
                },
                ControlEdge {
                    op: CONTROL_OP_PAR,
                    lhs: 0,
                    rhs: 2,
                    guard: 0,
                    order: 1,
                    bound: 0,
                },
            ],
            // Non-overlapping effect sets → nodes 1 and 2 are independent.
            vec![
                EffectTable {
                    reads: 0,
                    writes: 0,
                    locks: 0,
                    safety_class: 0,
                }, // node 0
                EffectTable {
                    reads: 0b0001,
                    writes: 0b0010,
                    locks: 0,
                    safety_class: 0,
                }, // node 1
                EffectTable {
                    reads: 0b0100,
                    writes: 0b1000,
                    locks: 0,
                    safety_class: 0,
                }, // node 2
            ],
        )?;
        world.embed_tensor_edges(&[
            TensorEdge::new(0, 1, 0.8, 1.0, 0.8),
            TensorEdge::new(0, 2, 0.7, 1.0, 0.7),
        ])?;

        let status = world.orchestrate_step()?;
        assert_ne!(status, OrchStepStatus::Error, "must not error");

        let state = world.orch_state_snapshot()?;
        assert_eq!(state.step, 1, "step must increment");
        assert_eq!(
            state.selected_control_op, CONTROL_OP_PAR,
            "selected_control_op must be PAR"
        );
        assert_eq!(state.control_epoch, 1, "control_epoch must increment");
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_smoke_skipped_without_cuda_feature() {}
