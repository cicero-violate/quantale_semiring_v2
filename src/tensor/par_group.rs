use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

use super::{FusionHfCoverage, GPU_HOT_REGION_COUNT, MAX_PAR_GROUP_SIZE, gpu_region_slots};
use crate::device_slots::DeviceSlotRegistry;
use crate::error::CudaError;
use crate::graph::DecisionReport;

/// GPU-resident data for the par-group-step kernel.
///
/// Built once at epoch start from the topology's compiled par groups and the
/// per-member dispatch metadata. Uploaded to the GPU device at construction.
pub struct ParGroupGpuData {
    pub(crate) table_buf: CudaSlice<i32>,
    /// Offset of each packed group record inside `table_buf`.
    ///
    /// `table_buf[group_offsets[g]]` is the group size and
    /// `table_buf[group_offsets[g] + 1]` is the first member tuple.  This lets
    /// the par-step kernel index groups directly instead of walking the
    /// variable-length table serially.
    pub(crate) group_offsets_buf: CudaSlice<i32>,
    pub num_groups: usize,
    /// Per-member slot table pointer array (shape: num_groups × MAX_PAR_GROUP_SIZE).
    /// Each entry is the device address of the `float**` pointer table for that
    /// member's hot region.  0 = no slot table (receipt-only / non-hot member).
    pub(crate) member_slot_table_ptrs: CudaSlice<u64>,
    /// Element count per member slot table (same shape as member_slot_table_ptrs).
    pub(crate) member_element_counts: CudaSlice<i32>,
    pub(crate) region_count: i32,
    /// Keeps the per-member `float**` device arrays alive for the epoch lifetime.
    #[allow(dead_code)]
    pub(crate) _slot_table_storage: Vec<CudaSlice<u64>>,
}

impl ParGroupGpuData {
    /// Build and upload par group data.
    ///
    /// `region_ids[g][i]` is the hot-region id for member `i` of group `g`, or
    /// `-1` when the member is not a hot-region operator.
    ///
    /// `is_gpu_dispatchable[g][i]` is `true` when the member has a GPU-native
    /// dispatch kind (H_f device or abstract-device receipt). The table is packed as
    /// `[g0_size, g0_n0, g0_r0, g0_e0, g0_k0, g0_n1, ...]`
    /// — `(node_id, region_id, is_gpu_dispatchable, dispatch_kind)` tuples.  The kernel computes
    /// eligibility on-device from the `is_gpu_dispatchable` flags rather than from
    /// a separate CPU-precomputed mask.
    ///
    /// `slot_registry` is used to build per-member `float**` slot pointer tables for
    /// hot-region members.  When a member's slots are registered, the kernel runs the
    /// `__device__` region function with real slot data (H_f path, D_h closed for that
    /// member).  When slots are absent the entry is 0 (receipt-only).
    pub fn build(
        dev: &Arc<CudaDevice>,
        groups: &[Vec<i32>],
        region_ids: &[Vec<i32>],
        is_gpu_dispatchable: &[Vec<bool>],
        dispatch_kinds: &[Vec<i32>],
        slot_registry: Option<&DeviceSlotRegistry>,
        fusion_hf_coverage: Option<&FusionHfCoverage>,
    ) -> Result<Self, CudaError> {
        use cudarc::driver::DevicePtr;

        assert_eq!(groups.len(), region_ids.len());
        assert_eq!(groups.len(), is_gpu_dispatchable.len());
        assert_eq!(groups.len(), dispatch_kinds.len());

        let num_groups = groups.len();
        let flat_size = (num_groups * MAX_PAR_GROUP_SIZE).max(1);

        // Allocate host-side flat arrays for per-member slot table pointers and
        // element counts.  Index: g * MAX_PAR_GROUP_SIZE + i.
        let mut slot_ptrs_host = vec![0u64; flat_size];
        let mut elem_counts_host = vec![0i32; flat_size];
        let mut slot_table_storage: Vec<CudaSlice<u64>> = Vec::new();

        if let Some(registry) = slot_registry {
            for (g, rids) in region_ids.iter().enumerate() {
                for (i, &rid) in rids.iter().enumerate() {
                    if rid < 0 {
                        continue;
                    }
                    let static_slots = gpu_region_slots(rid);
                    let generated_slots =
                        fusion_hf_coverage.and_then(|coverage| coverage.slots_for_region_id(rid));
                    let slot_refs: Vec<&str> = match (static_slots, generated_slots) {
                        (Some(slots), _) => slots.to_vec(),
                        (None, Some(slots)) => slots.iter().map(String::as_str).collect(),
                        (None, None) => continue,
                    };
                    if slot_refs.is_empty() {
                        continue;
                    }
                    match registry.device_slot_ptr_table(dev, &slot_refs) {
                        Ok((ptr_table, elem_count)) => {
                            let device_addr = *ptr_table.device_ptr();
                            slot_ptrs_host[g * MAX_PAR_GROUP_SIZE + i] = device_addr;
                            elem_counts_host[g * MAX_PAR_GROUP_SIZE + i] = elem_count;
                            // Transmute: CudaSlice<CUdeviceptr> is layout-equivalent to
                            // CudaSlice<u64> since CUdeviceptr = u64.
                            // Safety: cudarc guarantees CUdeviceptr = u64.
                            let raw: CudaSlice<u64> = unsafe { std::mem::transmute(ptr_table) };
                            slot_table_storage.push(raw);
                        }
                        Err(_) => { /* slots not registered; stays 0 (receipt-only) */ }
                    }
                }
            }
        }

        let member_slot_table_ptrs = dev
            .htod_copy(slot_ptrs_host)
            .map_err(|e| CudaError::new("htod par_group slot_table_ptrs", e))?;
        let member_element_counts = dev
            .htod_copy(elem_counts_host)
            .map_err(|e| CudaError::new("htod par_group element_counts", e))?;

        if groups.is_empty() {
            let table_buf = dev
                .htod_copy(vec![0_i32])
                .map_err(|e| CudaError::new("htod par_group table empty", e))?;
            let group_offsets_buf = dev
                .htod_copy(vec![0_i32])
                .map_err(|e| CudaError::new("htod par_group offsets empty", e))?;
            return Ok(Self {
                table_buf,
                group_offsets_buf,
                num_groups: 0,
                member_slot_table_ptrs,
                member_element_counts,
                region_count: fusion_hf_coverage
                    .map(FusionHfCoverage::region_count)
                    .unwrap_or(GPU_HOT_REGION_COUNT),
                _slot_table_storage: slot_table_storage,
            });
        }

        // Packed table: [g0_size, g0_n0, g0_r0, g0_e0, g0_k0, g0_n1, ...]
        let mut table: Vec<i32> = Vec::new();
        let mut group_offsets: Vec<i32> = Vec::with_capacity(num_groups);
        for (((group, rids), dispatchable), kinds) in groups
            .iter()
            .zip(region_ids.iter())
            .zip(is_gpu_dispatchable.iter())
            .zip(dispatch_kinds.iter())
        {
            group_offsets.push(table.len() as i32);
            table.push(group.len() as i32);
            for (((&node_id, &rid), &disp), &kind) in group
                .iter()
                .zip(rids.iter())
                .zip(dispatchable.iter())
                .zip(kinds.iter())
            {
                table.push(node_id);
                table.push(rid);
                table.push(disp as i32);
                table.push(kind);
            }
        }
        let table_buf = dev
            .htod_copy(table)
            .map_err(|e| CudaError::new("htod par_group table", e))?;
        let group_offsets_buf = dev
            .htod_copy(group_offsets)
            .map_err(|e| CudaError::new("htod par_group offsets", e))?;
        Ok(Self {
            table_buf,
            group_offsets_buf,
            num_groups,
            member_slot_table_ptrs,
            member_element_counts,
            region_count: fusion_hf_coverage
                .map(FusionHfCoverage::region_count)
                .unwrap_or(GPU_HOT_REGION_COUNT),
            _slot_table_storage: slot_table_storage,
        })
    }
}

/// C-compatible descriptor for one committed par-group member.
/// Must match the CUDA `ParDispatchDescriptor` definition exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParDispatchDescriptor {
    pub member_index: i32,
    pub node_id: i32,
    pub region_id: i32,
    pub dispatch_kind: i32,
    pub src_node: i32,
    pub dst_node: i32,
}

unsafe impl DeviceRepr for ParDispatchDescriptor {}

/// C-compatible output struct for `tensor_quantale_par_group_step`.
/// Must match the CUDA `ParGroupStepOutput` definition exactly.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct ParGroupStepOutputRaw {
    pub selected_group_idx: i32,
    pub group_size: i32,
    pub decisions: [DecisionReport; MAX_PAR_GROUP_SIZE],
    /// Hot-region id for each committed member; -1 when the member is not a hot region.
    pub region_ids: [i32; MAX_PAR_GROUP_SIZE],
    /// 1 when the member was dispatched in-kernel via the H_f path (Phase 2).
    /// CPU must skip execute_*_blocking and gpu_dispatch_region for those members.
    pub dispatched_on_device: [i32; MAX_PAR_GROUP_SIZE],
    /// Per-member dispatch descriptors emitted by the GPU. Non-H_f members keep
    /// explicit dispatch kinds so future tiers can consume descriptors without
    /// re-deriving member routing on the host.
    pub dispatch_descriptors: [ParDispatchDescriptor; MAX_PAR_GROUP_SIZE],
}

impl Default for ParGroupStepOutputRaw {
    fn default() -> Self {
        Self {
            selected_group_idx: -1,
            group_size: 0,
            decisions: [DecisionReport::default(); MAX_PAR_GROUP_SIZE],
            region_ids: [-1_i32; MAX_PAR_GROUP_SIZE],
            dispatched_on_device: [0_i32; MAX_PAR_GROUP_SIZE],
            dispatch_descriptors: [ParDispatchDescriptor::default(); MAX_PAR_GROUP_SIZE],
        }
    }
}

unsafe impl DeviceRepr for ParGroupStepOutputRaw {}

/// Host-side mirror of the CUDA `ParGroupHfParams` struct.
///
/// Uploaded as a single device word so the kernel parameter count stays within
/// the cudarc `LaunchAsync` arity limit.  All pointer fields are raw device
/// addresses (u64 = CUdeviceptr).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ParGroupHfParamsHost {
    pub slot_table_ptrs_dev: u64,
    pub element_counts_dev: u64,
    pub receipt_ring_dev: u64,
    pub ring_tail_dev: u64,
    pub ring_size: i32,
    pub region_count: i32,
}

unsafe impl DeviceRepr for ParGroupHfParamsHost {}
