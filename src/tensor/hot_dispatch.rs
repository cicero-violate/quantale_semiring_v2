use super::*;

impl TensorQuantaleWorld {
    /// Drain all `DeviceReceipt`s in the device ring buffer on-device.
    ///
    /// The GPU reads the ring and applies confidence/cost/
    /// safety atomics directly.
    pub fn drain_device_receipts(&mut self) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DRAIN_DEVICE_RECEIPTS_KERNEL)
            .ok_or(CudaError::missing_function(DRAIN_DEVICE_RECEIPTS_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.tensor,
                    &self.device_receipt_buffer.ring,
                    ring_size,
                    &mut self.device_receipt_buffer.head,
                    &self.device_receipt_buffer.tail,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_drain_device_receipts", error))
    }

    /// Push a generic execution receipt into the device receipt ring.
    ///
    /// This is for GPU-dispatched work that is not a registered hot region, such
    /// as a batched fusion JIT kernel. Call `drain_device_receipts` afterwards
    /// to apply the tensor update on-device.
    pub fn push_device_receipt(
        &mut self,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, PUSH_DEVICE_RECEIPT_KERNEL)
            .ok_or(CudaError::missing_function(PUSH_DEVICE_RECEIPT_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_id,
                    src_node,
                    dst_node,
                    outcome,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_push_device_receipt", error))
    }

    pub fn gpu_dispatch_region(
        &mut self,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        use crate::tensor::GpuDispatchMailboxHost;
        let mailbox = GpuDispatchMailboxHost {
            pending_region_id: region_id,
            src_node,
            dst_node,
            outcome,
            dispatched: 0,
        };
        let mailbox_buf = self
            .dev
            .htod_copy(vec![mailbox])
            .map_err(|error| CudaError::new("htod_copy gpu dispatch mailbox", error))?;
        let region_count = GPU_HOT_REGION_COUNT;
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, GPU_DISPATCH_KERNEL)
            .ok_or(CudaError::missing_function(GPU_DISPATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mailbox_buf,
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_count,
                    0_u64,
                    0_i32,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_gpu_dispatch", error))
    }

    /// Dispatch a hot GPU region with real device-slot backing.
    ///
    /// `DeviceSlotRegistry` supplies the region's ordered `float**` slot table,
    /// so the CUDA dispatch switch calls the region function and writes output
    /// slots before appending the device receipt.
    pub fn gpu_dispatch_region_with_slots(
        &mut self,
        registry: &DeviceSlotRegistry,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), CudaError> {
        let slot_names = gpu_region_slots(region_id).ok_or_else(|| {
            CudaError::invalid_input(format!("unknown GPU hot region id {region_id}"))
        })?;
        let mailbox = GpuDispatchMailboxHost {
            pending_region_id: region_id,
            src_node,
            dst_node,
            outcome,
            dispatched: 0,
        };
        let mailbox_buf = self
            .dev
            .htod_copy(vec![mailbox])
            .map_err(|error| CudaError::new("htod_copy gpu dispatch mailbox", error))?;
        let (slot_ptrs, element_count) = registry
            .device_slot_ptr_table(&self.dev, slot_names)
            .map_err(CudaError::invalid_input)?;
        let region_count = GPU_HOT_REGION_COUNT;
        let ring_size = DEVICE_RECEIPT_RING_SIZE as i32;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, GPU_DISPATCH_KERNEL)
            .ok_or(CudaError::missing_function(GPU_DISPATCH_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mailbox_buf,
                    &mut self.device_receipt_buffer.ring,
                    &mut self.device_receipt_buffer.tail,
                    ring_size,
                    region_count,
                    &slot_ptrs,
                    element_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_gpu_dispatch", error))?;
        self.dev
            .synchronize()
            .map_err(|error| CudaError::new("synchronize gpu dispatch", error))
    }
}
