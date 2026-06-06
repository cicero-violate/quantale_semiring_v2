//! Generalized GPU memory slot registry, buffer pool, and ring buffer.
//!
//! Replaces the thin `SlotBuffers = HashMap<String, CudaSlice<f32>>` with
//! typed descriptors so that every device allocation is annotated with its
//! `DataKind`, dtype string, and logical shape.

use crate::types::DataKind;

// ── Slot descriptor ───────────────────────────────────────────────────────────

/// Metadata describing a single named GPU memory region.
#[derive(Clone, Debug)]
pub struct DeviceSlot {
    pub name: String,
    pub kind: DataKind,
    /// Element dtype as a lowercase string: "f32", "f64", "i32", "u8", etc.
    pub dtype: String,
    /// Logical shape: e.g. [rows, cols] for a matrix or [len] for a vector.
    pub shape: Vec<usize>,
}

impl DeviceSlot {
    pub fn tensor_f32(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            kind: DataKind::Tensor,
            dtype: "f32".into(),
            shape,
        }
    }

    /// Total number of elements implied by `shape`.
    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty() || self.shape.iter().any(|&d| d == 0)
    }
}

// ── DeviceSlotRegistry ────────────────────────────────────────────────────────

/// Registry mapping slot names to typed descriptors and live device buffers.
///
/// `#[cfg(feature = "cuda")]` guards all buffer-holding state; the non-CUDA
/// variant compiles but holds no data.
#[cfg(feature = "cuda")]
pub struct DeviceSlotRegistry {
    slots: std::collections::HashMap<String, (DeviceSlot, cudarc::driver::CudaSlice<f32>)>,
}

#[cfg(feature = "cuda")]
impl Default for DeviceSlotRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl DeviceSlotRegistry {
    pub fn new() -> Self {
        Self {
            slots: std::collections::HashMap::new(),
        }
    }

    /// Register a named slot with its descriptor and pre-allocated buffer.
    pub fn register(&mut self, slot: DeviceSlot, buf: cudarc::driver::CudaSlice<f32>) {
        self.slots.insert(slot.name.clone(), (slot, buf));
    }

    /// Insert a raw f32 buffer under `name`, synthesising a Tensor descriptor.
    pub fn insert(&mut self, name: impl Into<String>, buf: cudarc::driver::CudaSlice<f32>) {
        use cudarc::driver::DeviceSlice;
        let name = name.into();
        let slot = DeviceSlot::tensor_f32(&name, vec![buf.len()]);
        self.slots.insert(name, (slot, buf));
    }

    pub fn get(&self, name: &str) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.slots.get(name).map(|(_, buf)| buf)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut cudarc::driver::CudaSlice<f32>> {
        self.slots.get_mut(name).map(|(_, buf)| buf)
    }

    pub fn slot_meta(&self, name: &str) -> Option<&DeviceSlot> {
        self.slots.get(name).map(|(slot, _)| slot)
    }

    pub fn remove(&mut self, name: &str) -> Option<cudarc::driver::CudaSlice<f32>> {
        self.slots.remove(name).map(|(_, buf)| buf)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.slots.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DeviceSlot, &cudarc::driver::CudaSlice<f32>)> {
        self.slots.values().map(|(slot, buf)| (slot, buf))
    }

    /// Build a device-resident `float**` pointer table for `slot_names`.
    ///
    /// The returned slice contains CUDA device addresses for the named f32
    /// slots in exactly the requested order. All slots must exist, be f32, and
    /// have the same element count because the hot-region kernels run one
    /// element loop length across every read/write slot.
    pub fn device_slot_ptr_table(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        slot_names: &[&str],
    ) -> Result<
        (
            cudarc::driver::CudaSlice<cudarc::driver::sys::CUdeviceptr>,
            i32,
        ),
        String,
    > {
        use cudarc::driver::{DevicePtr, DeviceSlice};

        let mut ptrs = Vec::with_capacity(slot_names.len());
        let mut element_count: Option<usize> = None;

        for &name in slot_names {
            let (slot, buf) = self
                .slots
                .get(name)
                .ok_or_else(|| format!("missing device slot '{name}'"))?;
            if slot.dtype != "f32" {
                return Err(format!(
                    "device slot '{name}' has dtype '{}', expected f32",
                    slot.dtype
                ));
            }
            let len = buf.len();
            if len == 0 {
                return Err(format!("device slot '{name}' is empty"));
            }
            match element_count {
                Some(expected) if expected != len => {
                    return Err(format!(
                        "device slot '{name}' has len {len}, expected {expected}"
                    ));
                }
                None => element_count = Some(len),
                _ => {}
            }
            ptrs.push(*buf.device_ptr());
        }

        let element_count = element_count.unwrap_or(0);
        if element_count > i32::MAX as usize {
            return Err(format!(
                "slot element count {element_count} exceeds i32::MAX"
            ));
        }
        let ptr_table = dev
            .htod_copy(ptrs)
            .map_err(|e| format!("DeviceSlotRegistry pointer table htod_copy: {e}"))?;
        Ok((ptr_table, element_count as i32))
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct DeviceSlotRegistry;

#[cfg(not(feature = "cuda"))]
impl DeviceSlotRegistry {
    pub fn new() -> Self {
        Self
    }

    pub fn device_slot_ptr_table(
        &self,
        _dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        _slot_names: &[&str],
    ) -> Result<
        (
            cudarc::driver::CudaSlice<cudarc::driver::sys::CUdeviceptr>,
            i32,
        ),
        String,
    > {
        Err("DeviceSlotRegistry pointer tables require the cuda feature".to_string())
    }
}

// ── DeviceBufferPool ──────────────────────────────────────────────────────────

/// Fixed-capacity reuse pool for f32 device buffers.
///
/// `acquire` returns a buffer of at least `len` elements: a recycled one if
/// the pool holds a suitably-sized free buffer, otherwise a fresh allocation.
/// `release` returns a buffer to the pool; entries beyond `capacity` are
/// dropped (the CUDA device frees the memory).
#[cfg(feature = "cuda")]
pub struct DeviceBufferPool {
    free: Vec<cudarc::driver::CudaSlice<f32>>,
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl DeviceBufferPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            free: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn acquire(
        &mut self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        use cudarc::driver::DeviceSlice;
        // Prefer a free buffer that is large enough to avoid re-allocation.
        if let Some(pos) = self.free.iter().position(|b| b.len() >= len) {
            return Ok(self.free.swap_remove(pos));
        }
        dev.htod_copy(vec![0.0f32; len])
            .map_err(|e| format!("DeviceBufferPool::acquire htod_copy: {e}"))
    }

    pub fn release(&mut self, buf: cudarc::driver::CudaSlice<f32>) {
        if self.free.len() < self.capacity {
            self.free.push(buf);
        }
        // else: buf is dropped, freeing device memory
    }
}

#[cfg(not(feature = "cuda"))]
pub struct DeviceBufferPool;

#[cfg(not(feature = "cuda"))]
impl DeviceBufferPool {
    pub fn new(_capacity: usize) -> Self {
        Self
    }
}

// ── DeviceRingBuffer ──────────────────────────────────────────────────────────

/// Circular FIFO of f32 values resident entirely on the device.
///
/// `head` and `tail` are single-element `CudaSlice<i32>` so that GPU kernels
/// can atomically advance them without a host round-trip.
#[cfg(feature = "cuda")]
pub struct DeviceRingBuffer {
    pub data: cudarc::driver::CudaSlice<f32>,
    pub head: cudarc::driver::CudaSlice<i32>,
    pub tail: cudarc::driver::CudaSlice<i32>,
    pub capacity: usize,
}

#[cfg(feature = "cuda")]
impl DeviceRingBuffer {
    pub fn new(
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        capacity: usize,
    ) -> Result<Self, String> {
        let data = dev
            .htod_copy(vec![0.0f32; capacity])
            .map_err(|e| format!("DeviceRingBuffer data alloc: {e}"))?;
        let head = dev
            .htod_copy(vec![0i32])
            .map_err(|e| format!("DeviceRingBuffer head alloc: {e}"))?;
        let tail = dev
            .htod_copy(vec![0i32])
            .map_err(|e| format!("DeviceRingBuffer tail alloc: {e}"))?;
        Ok(Self {
            data,
            head,
            tail,
            capacity,
        })
    }

    fn indices(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(i32, i32), String> {
        let head = dev
            .dtoh_sync_copy(&self.head)
            .map_err(|e| format!("DeviceRingBuffer head read: {e}"))?
            .first()
            .copied()
            .ok_or_else(|| "DeviceRingBuffer head missing".to_string())?;
        let tail = dev
            .dtoh_sync_copy(&self.tail)
            .map_err(|e| format!("DeviceRingBuffer tail read: {e}"))?
            .first()
            .copied()
            .ok_or_else(|| "DeviceRingBuffer tail missing".to_string())?;
        Ok((head, tail))
    }

    pub fn len(&self, dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<usize, String> {
        let (head, tail) = self.indices(dev)?;
        if tail < head {
            return Err(format!(
                "DeviceRingBuffer invalid indices: head={head} tail={tail}"
            ));
        }
        Ok((tail - head) as usize)
    }

    /// Push `src` into the ring using the GPU-side `device_ring_push` kernel.
    ///
    /// The kernel writes serially (single-thread) to avoid head/tail races.
    pub fn push(
        &mut self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        module: &str,
        src: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<(), String> {
        use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};
        let n = src.len() as i32;
        if n == 0 {
            return Ok(());
        }
        if n as usize > self.capacity {
            return Err(format!(
                "DeviceRingBuffer overflow: push len {n} exceeds capacity {}",
                self.capacity
            ));
        }
        let used = self.len(dev)?;
        let available = self.capacity.saturating_sub(used);
        if n as usize > available {
            return Err(format!(
                "DeviceRingBuffer overflow: push len {n} exceeds available {available}"
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let cap = self.capacity as i32;
        let kernel = dev
            .get_func(module, "device_ring_push")
            .ok_or_else(|| "device_ring_push not loaded".to_string())?;
        unsafe { kernel.launch(cfg, (&mut self.data, &mut self.tail, cap, src, n)) }
            .map_err(|e| format!("device_ring_push launch: {e}"))
    }

    /// Pop `n` elements from the ring using the GPU-side `device_ring_pop` kernel.
    pub fn pop(
        &mut self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        module: &str,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        use cudarc::driver::LaunchAsync;
        use cudarc::driver::LaunchConfig;
        let used = self.len(dev)?;
        if n > used {
            return Err(format!(
                "DeviceRingBuffer underflow: pop len {n} exceeds available {used}"
            ));
        }
        let dst = dev
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| format!("DeviceRingBuffer pop alloc: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let cap = self.capacity as i32;
        let ni32 = n as i32;
        let kernel = dev
            .get_func(module, "device_ring_pop")
            .ok_or_else(|| "device_ring_pop not loaded".to_string())?;
        let mut dst_mut = dst;
        unsafe { kernel.launch(cfg, (&self.data, &mut self.head, cap, &mut dst_mut, ni32)) }
            .map_err(|e| format!("device_ring_pop launch: {e}"))?;
        Ok(dst_mut)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct DeviceRingBuffer;

#[cfg(not(feature = "cuda"))]
impl DeviceRingBuffer {
    pub fn new(_capacity: usize) -> Result<Self, String> {
        Ok(Self)
    }
}

// ── Ingress pipeline ──────────────────────────────────────────────────────────

/// Host-side staging buffer for CPU→GPU data upload.
pub struct HostStagingBuffer {
    pub data: Vec<f32>,
}

impl HostStagingBuffer {
    pub fn from_slice(src: &[f32]) -> Self {
        Self { data: src.to_vec() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(feature = "cuda")]
pub struct PinnedHostBuffer {
    ptr: std::ptr::NonNull<f32>,
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for PinnedHostBuffer {}

#[cfg(feature = "cuda")]
impl PinnedHostBuffer {
    pub fn from_slice(src: &[f32]) -> Result<Self, String> {
        use cudarc::driver::sys;
        use std::ffi::c_void;

        let byte_len = std::mem::size_of_val(src);
        let alloc_len = byte_len.max(1);
        let mut raw: *mut c_void = std::ptr::null_mut();
        unsafe {
            sys::lib()
                .cuMemHostAlloc(&mut raw, alloc_len, sys::CU_MEMHOSTALLOC_PORTABLE)
                .result()
                .map_err(|e| format!("cuMemHostAlloc: {e:?}"))?;
        }
        let ptr = std::ptr::NonNull::new(raw.cast::<f32>())
            .ok_or_else(|| "cuMemHostAlloc returned null".to_string())?;
        if !src.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), src.len());
            }
        }
        Ok(Self {
            ptr,
            len: src.len(),
        })
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::lib().cuMemFreeHost(self.ptr.as_ptr().cast());
        }
    }
}

#[cfg(feature = "cuda")]
struct InFlightUpload {
    _slot: String,
}

/// Upload queue: accumulates CPU data and uploads it to the device slot registry
/// in one `flush` call.
///
/// Staging is `Vec`-based (synchronous copy on `stage`).  The H2D transfer in
/// `flush` is performed via `htod_copy`, which is synchronous.  The "upload
/// queue" abstraction separates the staging phase from the transfer phase so
/// callers can batch multiple slots before paying the H2D cost, but the flush
/// itself is not a CUDA async transfer (no stream; no `cudaMemcpyAsync`).
/// A future refactor can introduce a CUDA stream and pinned host buffers here
/// once the correctness baseline is established.
pub struct UploadQueue {
    staged: Vec<(DeviceSlot, HostStagingBuffer)>,
    #[cfg(feature = "cuda")]
    in_flight: Vec<InFlightUpload>,
    #[cfg(feature = "cuda")]
    in_flight_device: Option<std::sync::Arc<cudarc::driver::CudaDevice>>,
}

impl Default for UploadQueue {
    fn default() -> Self {
        Self {
            staged: Vec::new(),
            #[cfg(feature = "cuda")]
            in_flight: Vec::new(),
            #[cfg(feature = "cuda")]
            in_flight_device: None,
        }
    }
}

impl UploadQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage `data` for upload under `slot_meta`.  Preserves shape and dtype
    /// metadata so `flush` can call `registry.register` with full slot context.
    pub fn stage(&mut self, slot_meta: &DeviceSlot, data: &[f32]) -> Result<(), String> {
        self.staged
            .push((slot_meta.clone(), HostStagingBuffer::from_slice(data)));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    /// Upload all staged buffers to the device slot registry and clear the queue.
    ///
    /// Preserves full slot metadata (shape, dtype) by calling `registry.register`
    /// rather than the raw `insert` path which would synthesize a minimal descriptor.
    pub fn flush(
        &mut self,
        registry: &mut DeviceSlotRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), String> {
        self.synchronize()?;
        for (slot_meta, buf) in self.staged.drain(..) {
            let device_buf = dev
                .htod_copy(buf.data)
                .map_err(|e| format!("UploadQueue htod '{}': {e}", slot_meta.name))?;
            self.in_flight.push(InFlightUpload {
                _slot: slot_meta.name.clone(),
            });
            registry.register(slot_meta, device_buf);
        }
        self.in_flight_device = Some(dev.clone());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    /// Wait for all queued uploads to finish and clear in-flight markers.
    pub fn synchronize(&mut self) -> Result<(), String> {
        if let Some(dev) = &self.in_flight_device {
            dev.synchronize()
                .map_err(|e| format!("UploadQueue synchronize: {e}"))?;
        }
        self.in_flight.clear();
        self.in_flight_device = None;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn flush(&mut self, _registry: &mut DeviceSlotRegistry) -> Result<(), String> {
        self.staged.clear();
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.staged.len()
    }

    #[cfg(feature = "cuda")]
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(feature = "cuda")]
impl Drop for UploadQueue {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}
