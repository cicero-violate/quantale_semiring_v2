use super::*;

impl TensorQuantaleWorld {
    /// Build and upload par group GPU data using this world's CUDA device.
    ///
    /// `groups` are the compiled par groups (node ID lists).
    /// `region_ids[g][i]` is the hot-region id for member `i` (-1 if not hot).
    /// `is_gpu_dispatchable[g][i]` is `true` for GPU-native dispatch members.
    /// `dispatch_kinds[g][i]` is the initial descriptor kind the kernel should
    /// emit for each committed member; H_f dispatch may upgrade it in-kernel.
    /// Eligibility is encoded per-member in the table and validated by the kernel.
    ///
    /// `slot_registry` provides `float**` device slot tables for hot-region members
    /// so that Phase 2 of `par_group_step` can call region functions with real data.
    /// Pass `None` to use receipt-only mode for all members.
    pub fn make_par_group_data(
        &self,
        groups: &[Vec<i32>],
        region_ids: &[Vec<i32>],
        is_gpu_dispatchable: &[Vec<bool>],
        dispatch_kinds: &[Vec<i32>],
        slot_registry: Option<&DeviceSlotRegistry>,
        fusion_hf_coverage: Option<&FusionHfCoverage>,
    ) -> Result<ParGroupGpuData, CudaError> {
        ParGroupGpuData::build(
            &self.dev,
            groups,
            region_ids,
            is_gpu_dispatchable,
            dispatch_kinds,
            slot_registry,
            fusion_hf_coverage,
        )
    }

    /// GPU-native parallel group step: select the first eligible, all-ready CKA
    /// par group, commit it on-device, and return the committed decisions.
    ///
    /// Returns `Ok(None)` when no group is ready — tensor state is unchanged.
    /// Returns `Ok(Some((group_idx, decisions, region_ids, dispatched_on_device, descriptors)))`.
    ///
    /// `dispatched_on_device[i] == 1` means member `i` was dispatched in-kernel
    /// via the H_f path: the region function ran on-device and its DeviceReceipt
    /// was written to the ring.  The CPU must skip `execute_*_blocking` and
    /// `gpu_dispatch_region` for those members (call `drain_device_receipts` only).
    pub fn par_group_step(
        &mut self,
        data: &ParGroupGpuData,
        bias: ProjectionBias,
    ) -> Result<
        Option<(
            usize,
            Vec<DecisionReport>,
            Vec<i32>,
            Vec<i32>,
            Vec<ParDispatchDescriptor>,
        )>,
        CudaError,
    > {
        if data.num_groups == 0 {
            return Ok(None);
        }
        let bias_buf = self
            .dev
            .htod_copy(vec![bias])
            .map_err(|e| CudaError::new("htod par_group bias", e))?;
        let mut out_buf = self
            .dev
            .htod_copy(vec![ParGroupStepOutputRaw::default()])
            .map_err(|e| CudaError::new("htod par_group output", e))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PAR_GROUP_STEP_KERNEL)
            .ok_or(CudaError::missing_function(PAR_GROUP_STEP_KERNEL))?;
        use cudarc::driver::DevicePtr;
        let hf_params = ParGroupHfParamsHost {
            slot_table_ptrs_dev: *data.member_slot_table_ptrs.device_ptr(),
            element_counts_dev: *data.member_element_counts.device_ptr(),
            receipt_ring_dev: *self.device_receipt_buffer.ring.device_ptr(),
            ring_tail_dev: *self.device_receipt_buffer.tail.device_ptr(),
            ring_size: DEVICE_RECEIPT_RING_SIZE as i32,
            region_count: data.region_count,
        };
        let hf_buf = self
            .dev
            .htod_copy(vec![hf_params])
            .map_err(|e| CudaError::new("htod par_group hf_params", e))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &self.witness,
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &bias_buf,
                    &mut self.decision,
                    &data.group_offsets_buf,
                    &data.table_buf,
                    data.num_groups as i32,
                    &mut out_buf,
                    &hf_buf,
                ),
            )
        }
        .map_err(|e| CudaError::new(PAR_GROUP_STEP_KERNEL, e))?;
        let output = self
            .dev
            .dtoh_sync_copy(&out_buf)
            .map_err(|e| CudaError::new("dtoh par_group output", e))?;
        let raw = &output[0];
        if raw.selected_group_idx < 0 {
            return Ok(None);
        }
        let sz = (raw.group_size as usize).min(MAX_PAR_GROUP_SIZE);
        Ok(Some((
            raw.selected_group_idx as usize,
            raw.decisions[..sz].to_vec(),
            raw.region_ids[..sz].to_vec(),
            raw.dispatched_on_device[..sz].to_vec(),
            raw.dispatch_descriptors[..sz].to_vec(),
        )))
    }
}
