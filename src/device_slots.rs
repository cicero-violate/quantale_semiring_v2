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
}

#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct DeviceSlotRegistry;

#[cfg(not(feature = "cuda"))]
impl DeviceSlotRegistry {
    pub fn new() -> Self {
        Self
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
        Ok(Self { data, head, tail, capacity })
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
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
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
        let dst = dev
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| format!("DeviceRingBuffer pop alloc: {e}"))?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
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

// ── Async ingress pipeline ────────────────────────────────────────────────────

/// Page-locked host buffer for async H2D DMA.
///
/// MVP uses a regular heap `Vec`; a production implementation would call
/// `cudaMallocHost` here for true pinned (page-locked) allocation that avoids
/// the DMA bounce buffer.
pub struct PinnedHostBuffer {
    pub data: Vec<f32>,
}

impl PinnedHostBuffer {
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

/// Async upload queue: stages CPU data for H2D transfer.
///
/// `stage` accumulates (slot_name, buffer) pairs.  `flush` uploads all staged
/// buffers to the `DeviceSlotRegistry` in one pass, using `htod_copy` which
/// internally issues an async DMA if the source is pinned.
pub struct AsyncUploadQueue {
    staged: Vec<(String, PinnedHostBuffer)>,
}

impl Default for AsyncUploadQueue {
    fn default() -> Self {
        Self { staged: Vec::new() }
    }
}

impl AsyncUploadQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage `data` for upload under `slot`.  The copy into `PinnedHostBuffer`
    /// is synchronous; the H2D transfer happens on `flush`.
    pub fn stage(&mut self, slot: &str, data: &[f32]) -> Result<(), String> {
        self.staged.push((slot.to_string(), PinnedHostBuffer::from_slice(data)));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    /// Upload all staged buffers to the device slot registry and clear the queue.
    pub fn flush(
        &mut self,
        registry: &mut DeviceSlotRegistry,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), String> {
        for (slot, buf) in self.staged.drain(..) {
            let device_buf = dev
                .htod_copy(buf.data)
                .map_err(|e| format!("AsyncUploadQueue flush htod_copy '{slot}': {e}"))?;
            registry.insert(slot, device_buf);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn flush(
        &mut self,
        _registry: &mut DeviceSlotRegistry,
    ) -> Result<(), String> {
        self.staged.clear();
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.staged.len()
    }
}
