use super::*;

impl TensorQuantaleWorld {
    // ── Phase-1 orchestration state wrappers ─────────────────────────────────

    /// Launch `orchestration_state_init` to zero the device-resident state block.
    /// Called once at world construction.
    pub fn orch_state_init(&mut self) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCH_STATE_INIT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_STATE_INIT_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.orch_buffers.state,)) }
            .map_err(|error| CudaError::new(ORCH_STATE_INIT_KERNEL, error))
    }

    /// Copy the live device orchestration state into a snapshot and return it.
    pub fn orch_state_snapshot(&mut self) -> Result<OrchestrationState, CudaError> {
        let mut snapshot = self
            .dev
            .htod_copy(vec![OrchestrationState::default()])
            .map_err(|error| CudaError::new("htod_copy orch_snapshot", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCH_STATE_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_STATE_SNAPSHOT_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&self.orch_buffers.state, &mut snapshot)) }
            .map_err(|error| CudaError::new(ORCH_STATE_SNAPSHOT_KERNEL, error))?;
        let result = self
            .dev
            .dtoh_sync_copy(&snapshot)
            .map_err(|error| CudaError::new("dtoh_sync_copy orch_snapshot", error))?;
        Ok(result[0])
    }

    /// Push one `DeviceCommand` into the device command ring.
    /// Returns `Err` if the ring is full (capacity = `DEVICE_COMMAND_RING_SIZE`).
    pub fn push_device_command(&mut self, cmd: DeviceCommand) -> Result<(), CudaError> {
        let ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_CMD_RING_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_CMD_RING_PUSH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    ring_size,
                    cmd,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_CMD_RING_PUSH_KERNEL, error))
    }

    /// Drain the device command ring to the host and return all valid commands.
    pub fn drain_device_commands(&mut self) -> Result<Vec<DeviceCommand>, CudaError> {
        let head = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_head)
            .map_err(|error| CudaError::new("dtoh command_head", error))?[0];
        let tail = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_tail)
            .map_err(|error| CudaError::new("dtoh command_tail", error))?[0];
        if head == tail {
            return Ok(Vec::new());
        }
        let ring = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.command_ring)
            .map_err(|error| CudaError::new("dtoh command_ring", error))?;
        let ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let mut out = Vec::new();
        let mut h = head;
        while h != tail {
            let cmd = ring[(h % ring_size) as usize];
            if cmd.valid != 0 {
                out.push(cmd);
            }
            h += 1;
        }
        // Advance head to match tail on the device.
        self.orch_buffers.command_head = self
            .dev
            .htod_copy(vec![tail])
            .map_err(|error| CudaError::new("htod command_head advance", error))?;
        Ok(out)
    }

    /// Push one `DeviceReceiptExt` into the extended receipt ring.
    pub fn push_device_receipt_ext(&mut self, receipt: DeviceReceiptExt) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_RECEIPT_EXT_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_RECEIPT_EXT_PUSH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.orch_buffers.receipt_ext_ring,
                    &mut self.orch_buffers.receipt_ext_tail,
                    &self.orch_buffers.receipt_ext_head,
                    ring_size,
                    receipt,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_RECEIPT_EXT_PUSH_KERNEL, error))
    }

    /// Drain the extended receipt ring on-device, applying tensor updates.
    pub fn drain_device_receipt_ext(&mut self) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DEVICE_RECEIPT_EXT_DRAIN_KERNEL)
            .ok_or(CudaError::missing_function(DEVICE_RECEIPT_EXT_DRAIN_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.orch_buffers.receipt_ext_ring,
                    ring_size,
                    &mut self.orch_buffers.receipt_ext_head,
                    &self.orch_buffers.receipt_ext_tail,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|error| CudaError::new(DEVICE_RECEIPT_EXT_DRAIN_KERNEL, error))
    }

    // ── Phase-2 orchestration step wrappers ──────────────────────────────────

    /// Upload a dispatch-kind table to the device.
    /// `kinds` must have length `TENSOR_NODE_COUNT`; each entry is one of the
    /// `DISPATCH_KIND_*` constants.
    pub fn set_dispatch_kinds(&mut self, kinds: &[i32]) -> Result<(), CudaError> {
        if kinds.len() != TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "set_dispatch_kinds: expected {TENSOR_NODE_COUNT} entries, got {}",
                kinds.len()
            )));
        }
        self.orch_buffers.dispatch_kinds = self
            .dev
            .htod_copy(kinds.to_vec())
            .map_err(|error| CudaError::new("htod_copy dispatch_kinds update", error))?;
        Ok(())
    }

    /// Upload a node-level reentrant-consumption mask.
    /// `mask[id] != 0` means edges incident to that node are not one-shot.
    pub fn set_reentrant_mask(&mut self, mask: &[i32]) -> Result<(), CudaError> {
        if mask.len() != TENSOR_NODE_COUNT {
            return Err(CudaError::invalid_input(format!(
                "set_reentrant_mask: expected {TENSOR_NODE_COUNT} entries, got {}",
                mask.len()
            )));
        }
        self.orch_buffers.reentrant_mask = self
            .dev
            .htod_copy(mask.to_vec())
            .map_err(|error| CudaError::new("htod_copy reentrant_mask update", error))?;
        Ok(())
    }

    /// Launch one `tensor_quantale_orchestrate_step` and return the status.
    ///
    /// The kernel:
    ///   1. Drains the extended receipt ring.
    ///   2. Selects the next ready node (singleton path).
    ///   3. Commits consumed/active state for GPU-native nodes.
    ///   4. Emits a `DeviceCommand` for external-dispatch nodes.
    ///   5. Returns `OrchStepStatus` to the host.
    pub fn orchestrate_step(&mut self) -> Result<OrchStepStatus, CudaError> {
        use cudarc::driver::DevicePtr;

        let ctrl_edge_count = self.orch_buffers.control_edges.len() as i32;
        let effect_count = self.orch_buffers.effect_table.len() as i32;
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        let bundle = TensorWorldBundleHost {
            tensor_dev: *self.tensor.device_ptr() as u64,
            witness_dev: *self.witness.device_ptr() as u64,
            consumed_dev: *self.consumed.device_ptr() as u64,
            active_dev: *self.active.device_ptr() as u64,
            next_active_dev: *self.next_active.device_ptr() as u64,
            reentrant_mask_dev: *self.orch_buffers.reentrant_mask.device_ptr() as u64,
            bias_dev: *self.orch_buffers.default_bias.device_ptr() as u64,
            decision_dev: *self.decision.device_ptr() as u64,
            control_edges_dev: *self.orch_buffers.control_edges.device_ptr() as u64,
            control_edge_count: ctrl_edge_count,
            effects_dev: *self.orch_buffers.effect_table.device_ptr() as u64,
            effect_count,
            star_counters_dev: *self.orch_buffers.star_counters.device_ptr() as u64,
            star_counter_count,
        };

        let bundle_dev = self
            .dev
            .htod_copy(vec![bundle])
            .map_err(|error| CudaError::new("htod_copy TensorWorldBundle", error))?;

        let cmd_ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let ext_ring_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;

        let kernel = self
            .dev
            .get_func(MODULE_NAME, ORCHESTRATE_STEP_KERNEL)
            .ok_or(CudaError::missing_function(ORCHESTRATE_STEP_KERNEL))?;

        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &bundle_dev,
                    &mut self.orch_buffers.state,
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    cmd_ring_size,
                    &mut self.orch_buffers.receipt_ext_ring,
                    &mut self.orch_buffers.receipt_ext_head,
                    &self.orch_buffers.receipt_ext_tail,
                    ext_ring_size,
                    &self.orch_buffers.dispatch_kinds,
                    &mut self.orch_buffers.step_status,
                ),
            )
        }
        .map_err(|error| CudaError::new(ORCHESTRATE_STEP_KERNEL, error))?;

        let status_vec = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.step_status)
            .map_err(|error| CudaError::new("dtoh step_status", error))?;
        Ok(OrchStepStatus::from_code(status_vec[0]))
    }

    // ── Phase-7 supervisor loop ───────────────────────────────────────────────

    /// Run the GPU scheduler for up to `max_steps` iterations without CPU
    /// involvement in per-step decisions.
    ///
    /// Returns as soon as one of the following occurs:
    /// - `ORCH_WAIT_EXTERNAL`: GPU has emitted an external command and is
    ///   waiting for the CPU service to respond.
    /// - `ORCH_HALTED`: the graph reached its halt node or `state.halted` was
    ///   set by the failure policy.
    /// - `ORCH_ERROR`: internal kernel error (unrecoverable).
    /// - `state.blocked == 1`: no ready singleton found; returns `Continue`
    ///   so the outer supervisor loop can decide on repair or shutdown.
    ///
    /// Returns `Continue` after exhausting `max_steps` without hitting a stop
    /// condition, giving the CPU a chance to service external commands or
    /// apply learned deltas between bursts.
    ///
    /// This is the Phase-7 host-loop demotion entry point:
    /// ```text
    /// loop {
    ///     match world.orchestrate_until_wait_or_halt(max_steps)? {
    ///         Continue      => continue,
    ///         WaitExternal  => service_external_commands(&mut world)?,
    ///         Halted        => break,
    ///         Error         => { snapshot(); break; }
    ///     }
    /// }
    /// ```
    pub fn orchestrate_until_wait_or_halt(
        &mut self,
        max_steps: u32,
    ) -> Result<OrchStepStatus, CudaError> {
        for _ in 0..max_steps {
            let status = self.orchestrate_step()?;
            match status {
                OrchStepStatus::Continue => {
                    // Blocked check: if the scheduler found no ready node,
                    // yield back to the CPU rather than spin.
                    let state = self.orch_state_snapshot()?;
                    if state.blocked != 0 {
                        return Ok(OrchStepStatus::Continue);
                    }
                }
                OrchStepStatus::WaitExternal | OrchStepStatus::Halted | OrchStepStatus::Error => {
                    return Ok(status);
                }
            }
        }
        Ok(OrchStepStatus::Continue)
    }

    // ── Phase-4 control-flow methods ─────────────────────────────────────────

    /// Upload a lowered pattern control table to the device.
    ///
    /// If `edges` is empty the device control table retains its current content.
    /// If `effects` is empty the device effect table retains its current content.
    pub fn load_control_table(
        &mut self,
        edges: Vec<ControlEdge>,
        effects: Vec<EffectTable>,
    ) -> Result<(), CudaError> {
        if !edges.is_empty() {
            let edge_count = edges.len();
            self.orch_buffers.control_edges = self
                .dev
                .htod_copy(edges)
                .map_err(|e| CudaError::new("load_control_table edges", e))?;
            // Resize star_counters to match edge count (capped at MAX_CONTROL_EDGES).
            let counter_len = edge_count.min(MAX_CONTROL_EDGES);
            let init_count = counter_len as i32;
            self.orch_buffers.star_counters = self
                .dev
                .htod_copy(vec![0_i32; counter_len])
                .map_err(|e| CudaError::new("load_control_table star_counters", e))?;
            self.orch_buffers.replay_star_counters =
                self.dev
                    .htod_copy(vec![0_i32; counter_len])
                    .map_err(|e| CudaError::new("load_control_table replay_star_counters", e))?;
            // Zero-init via dedicated kernel for consistency.
            if let Some(f) = self.dev.get_func(MODULE_NAME, STAR_COUNTERS_INIT_KERNEL) {
                unsafe {
                    f.launch(
                        kernel_config(),
                        (&mut self.orch_buffers.star_counters, init_count),
                    )
                }
                .map_err(|e| CudaError::new(STAR_COUNTERS_INIT_KERNEL, e))?;
            }
        }
        if !effects.is_empty() {
            self.orch_buffers.effect_table = self
                .dev
                .htod_copy(effects)
                .map_err(|e| CudaError::new("load_control_table effects", e))?;
        }
        Ok(())
    }

    // ── Phase-5 failure policy wrappers ──────────────────────────────────────

    /// Initialise the per-node failure policy table on the GPU.
    ///
    /// Every node receives `default_budget` retries.  -1 = unlimited.
    /// `default_block_threshold` sets consecutive-block limit; -1 = disabled.
    /// Call this once after world construction to arm retry budgets.
    pub fn failure_policy_init(
        &mut self,
        default_budget: i32,
        default_block_threshold: i32,
    ) -> Result<(), CudaError> {
        let n_nodes = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_INIT_KERNEL)
            .ok_or(CudaError::missing_function(FAILURE_POLICY_INIT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (
                    &mut self.orch_buffers.failure_policies,
                    n_nodes,
                    default_budget,
                    default_block_threshold,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_INIT_KERNEL, e))
    }

    /// Classify a receipt failure, update the per-node retry budget, and choose
    /// the corrective action on-device.
    ///
    /// Returns the `FAILURE_ACTION_*` code written by the kernel.  When the
    /// action is `FAILURE_ACTION_EXTERNAL_REPAIR` a repair `DeviceCommand` is
    /// also pushed into the device command ring.
    pub fn failure_policy_classify_and_emit(
        &mut self,
        outcome: i32,
        node_id: i32,
        src: i32,
        dst: i32,
        command_id: i32,
    ) -> Result<i32, CudaError> {
        self.orch_buffers.failure_action_out = self
            .dev
            .htod_copy(vec![FAILURE_ACTION_BLOCK])
            .map_err(|e| CudaError::new("reset failure_action_out", e))?;
        let req = FailureClassifyRequest {
            outcome,
            node_id,
            src,
            dst,
            command_id,
        };
        let req_buf = self
            .dev
            .htod_copy(vec![req])
            .map_err(|e| CudaError::new("htod failure_classify_req", e))?;
        let n_policies = self.orch_buffers.failure_policies.len() as i32;
        let cmd_ring_size = DEVICE_COMMAND_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_CLASSIFY_KERNEL)
            .ok_or(CudaError::missing_function(FAILURE_POLICY_CLASSIFY_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &req_buf,
                    &mut self.orch_buffers.failure_policies,
                    n_policies,
                    &mut self.orch_buffers.state,
                    &mut self.orch_buffers.command_ring,
                    &mut self.orch_buffers.command_tail,
                    &self.orch_buffers.command_head,
                    cmd_ring_size,
                    &mut self.orch_buffers.failure_action_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_CLASSIFY_KERNEL, e))?;
        let result = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.failure_action_out)
            .map_err(|e| CudaError::new("dtoh failure_action_out", e))?;
        Ok(result[0])
    }

    /// Snapshot the current `consumed[]` and `active[]` arrays as a rollback marker.
    ///
    /// Sets `OrchestrationState::rollback_available = 1`.  The snapshot can be
    /// restored by calling `apply_rollback`.
    pub fn set_rollback_marker(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_SET_ROLLBACK_KERNEL)
            .ok_or(CudaError::missing_function(
                FAILURE_POLICY_SET_ROLLBACK_KERNEL,
            ))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.consumed,
                    &self.active,
                    &mut self.orch_buffers.rollback_consumed,
                    &mut self.orch_buffers.rollback_active,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_SET_ROLLBACK_KERNEL, e))
    }

    /// Restore `consumed[]` and `active[]` from the saved rollback marker.
    ///
    /// No-op if `OrchestrationState::rollback_available == 0`.
    /// Clears `rollback_available` and `consecutive_blocks` on success.
    pub fn apply_rollback(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, FAILURE_POLICY_APPLY_ROLLBACK_KERNEL)
            .ok_or(CudaError::missing_function(
                FAILURE_POLICY_APPLY_ROLLBACK_KERNEL,
            ))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &mut self.consumed,
                    &mut self.active,
                    &self.orch_buffers.rollback_consumed,
                    &self.orch_buffers.rollback_active,
                    &mut self.orch_buffers.state,
                ),
            )
        }
        .map_err(|e| CudaError::new(FAILURE_POLICY_APPLY_ROLLBACK_KERNEL, e))
    }

    // ── Phase-6 wrappers ──────────────────────────────────────────────────────

    /// Zero-initialise the per-node receipt prior table on device.
    pub fn learned_delta_init(&mut self) -> Result<(), CudaError> {
        let n = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_INIT_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_INIT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&mut self.orch_buffers.receipt_priors, n)) }
            .map_err(|e| CudaError::new(LEARNED_DELTA_INIT_KERNEL, e))
    }

    /// Fold one receipt into the per-node prior table and push a `LearnedDelta`
    /// entry to the on-device ring.  `outcome` uses the same codes as
    /// `DeviceReceiptExt::outcome` (0 = success, 1 = failure, 2 = timeout,
    /// 3 = safety violation).
    pub fn learned_delta_fold_receipt(
        &mut self,
        src: i32,
        dst: i32,
        node_id: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let n_nodes = TENSOR_NODE_COUNT as i32;
        let ring_size = LEARNED_DELTA_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_FOLD_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_FOLD_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    src,
                    dst,
                    node_id,
                    outcome,
                    &mut self.orch_buffers.receipt_priors,
                    n_nodes,
                    &mut self.orch_buffers.learned_delta_ring,
                    &mut self.orch_buffers.learned_delta_tail,
                    &self.orch_buffers.learned_delta_head,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(LEARNED_DELTA_FOLD_KERNEL, e))
    }

    /// Drain the on-device learned-delta ring and apply soft tensor updates.
    /// Each pending `LearnedDelta` entry increments or decrements the
    /// corresponding confidence/cost/safety cell in the live tensor.
    pub fn learned_delta_apply(&mut self) -> Result<(), CudaError> {
        let ring_size = LEARNED_DELTA_RING_SIZE as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, LEARNED_DELTA_APPLY_KERNEL)
            .ok_or(CudaError::missing_function(LEARNED_DELTA_APPLY_KERNEL))?;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &mut self.orch_buffers.learned_delta_ring,
                    &mut self.orch_buffers.learned_delta_head,
                    &self.orch_buffers.learned_delta_tail,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(LEARNED_DELTA_APPLY_KERNEL, e))
    }

    /// Copy the GPU-resident receipt prior table to host and return it.
    /// Also writes the snapshot into the on-device export buffer for
    /// subsequent CPU persistence without a second kernel launch.
    pub fn export_receipt_priors(&mut self) -> Result<Vec<f32>, CudaError> {
        let n = TENSOR_NODE_COUNT as i32;
        let f = self
            .dev
            .get_func(MODULE_NAME, RECEIPT_PRIOR_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(RECEIPT_PRIOR_SNAPSHOT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: ((TENSOR_NODE_COUNT as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (
                    &self.orch_buffers.receipt_priors,
                    &mut self.orch_buffers.receipt_prior_snapshot_buf,
                    n,
                ),
            )
        }
        .map_err(|e| CudaError::new(RECEIPT_PRIOR_SNAPSHOT_KERNEL, e))?;
        self.dev
            .dtoh_sync_copy(&self.orch_buffers.receipt_prior_snapshot_buf)
            .map_err(|e| CudaError::new("dtoh receipt_prior_snapshot_buf", e))
    }

    // ── Phase-8 wrappers ──────────────────────────────────────────────────────

    /// Push one event onto the device trace ring.
    /// Single-thread kernel (1 block × 1 thread).
    pub fn push_trace_event(&mut self, event_kind: i32, outcome: i32) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_TRACE_PUSH_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_TRACE_PUSH_KERNEL))?;
        let ring_size = ORCH_TRACE_RING_SIZE as i32;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &self.orch_buffers.state,
                    event_kind,
                    outcome,
                    &mut self.orch_buffers.trace_ring,
                    &mut self.orch_buffers.trace_tail,
                    &self.orch_buffers.trace_head,
                    ring_size,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_TRACE_PUSH_KERNEL, e))
    }

    /// Drain pending trace events to host.  Returns a `Vec` of drained events.
    /// Single-thread kernel writes entries to `trace_drain_buf`; host copies.
    pub fn drain_trace_events(&mut self) -> Result<Vec<OrchestrationEvent>, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_TRACE_DRAIN_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_TRACE_DRAIN_KERNEL))?;
        let ring_size = ORCH_TRACE_RING_SIZE as i32;
        let max_count = ORCH_TRACE_RING_SIZE as i32;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.orch_buffers.trace_ring,
                    &mut self.orch_buffers.trace_head,
                    &self.orch_buffers.trace_tail,
                    ring_size,
                    &mut self.orch_buffers.trace_drain_buf,
                    &mut self.orch_buffers.trace_drain_count,
                    max_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_TRACE_DRAIN_KERNEL, e))?;
        let count_vec = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.trace_drain_count)
            .map_err(|e| CudaError::new("dtoh trace_drain_count", e))?;
        let n = count_vec[0].max(0) as usize;
        let all = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.trace_drain_buf)
            .map_err(|e| CudaError::new("dtoh trace_drain_buf", e))?;
        Ok(all.into_iter().take(n).collect())
    }

    /// Run the no-duplicate-receipts invariant check.
    /// Returns `Ok(true)` when the invariant holds (no violation).
    pub fn check_no_duplicate_receipts(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.orch_buffers.receipt_ext_ring,
                    size,
                    &mut self.orch_buffers.orch_violation_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_DUPLICATE_RECEIPTS_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Run the frontier-valid invariant check.
    /// Returns `Ok(true)` when all `active[i]` ∈ {0, 1}.
    pub fn check_frontier_valid(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_FRONTIER_VALID_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_FRONTIER_VALID_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let n = TENSOR_NODE_COUNT as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (&self.active, n, &mut self.orch_buffers.orch_violation_out),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_FRONTIER_VALID_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Run the no-command-without-receipt invariant check.
    /// Returns `Ok(true)` when the invariant holds.
    pub fn check_no_command_without_receipt(&mut self) -> Result<bool, CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL)
            .ok_or(CudaError::missing_function(
                ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL,
            ))?;
        self.orch_buffers.orch_violation_out = self
            .dev
            .htod_copy(vec![0_i32])
            .map_err(|e| CudaError::new("htod reset orch_violation_out", e))?;
        let cmd_size = DEVICE_COMMAND_RING_SIZE as i32;
        let ext_size = DEVICE_RECEIPT_EXT_RING_SIZE as i32;
        unsafe {
            f.launch(
                kernel_config(),
                (
                    &self.orch_buffers.command_ring,
                    cmd_size,
                    &self.orch_buffers.receipt_ext_ring,
                    ext_size,
                    &mut self.orch_buffers.orch_violation_out,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_CHECK_NO_CMD_WITHOUT_RECEIPT_KERNEL, e))?;
        let v = self
            .dev
            .dtoh_sync_copy(&self.orch_buffers.orch_violation_out)
            .map_err(|e| CudaError::new("dtoh orch_violation_out", e))?;
        Ok(v[0] == 0)
    }

    /// Snapshot current orchestration state (+ consumed + active) into the
    /// replay buffers.  Block-parallel copy kernel (N threads).
    pub fn replay_snapshot(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_REPLAY_SNAPSHOT_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_REPLAY_SNAPSHOT_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (TENSOR_NODE_COUNT as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        unsafe {
            f.launch(
                cfg,
                (
                    &self.orch_buffers.state,
                    &mut self.orch_buffers.replay_state,
                    &self.consumed,
                    &mut self.orch_buffers.replay_consumed,
                    &self.active,
                    &mut self.orch_buffers.replay_active,
                    &self.orch_buffers.star_counters,
                    &mut self.orch_buffers.replay_star_counters,
                    star_counter_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_REPLAY_SNAPSHOT_KERNEL, e))
    }

    /// Restore orchestration state from the replay buffers back to live state.
    /// Block-parallel copy kernel (N threads).
    pub fn replay_restore(&mut self) -> Result<(), CudaError> {
        let f = self
            .dev
            .get_func(MODULE_NAME, ORCH_REPLAY_RESTORE_KERNEL)
            .ok_or(CudaError::missing_function(ORCH_REPLAY_RESTORE_KERNEL))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (TENSOR_NODE_COUNT as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let star_counter_count = self.orch_buffers.star_counters.len() as i32;
        unsafe {
            f.launch(
                cfg,
                (
                    &mut self.orch_buffers.state,
                    &self.orch_buffers.replay_state,
                    &mut self.consumed,
                    &self.orch_buffers.replay_consumed,
                    &mut self.active,
                    &self.orch_buffers.replay_active,
                    &mut self.orch_buffers.star_counters,
                    &self.orch_buffers.replay_star_counters,
                    star_counter_count,
                ),
            )
        }
        .map_err(|e| CudaError::new(ORCH_REPLAY_RESTORE_KERNEL, e))
    }
}
