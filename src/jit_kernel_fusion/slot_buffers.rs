#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct SlotBuffers {
    buffers: HashMap<String, CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl SlotBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains(&self, slot: &str) -> bool {
        self.buffers.contains_key(slot)
    }

    pub fn get(&self, slot: &str) -> Option<&CudaSlice<f32>> {
        self.buffers.get(slot)
    }

    pub fn insert(&mut self, slot: impl Into<String>, buffer: CudaSlice<f32>) {
        self.buffers.insert(slot.into(), buffer);
    }

    pub fn remove(&mut self, slot: &str) -> Option<CudaSlice<f32>> {
        self.buffers.remove(slot)
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct SlotBuffers;
