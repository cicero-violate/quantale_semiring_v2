// Slot buffer storage for JIT kernel execution.
//
// DeviceSlotRegistry is the canonical type; SlotBuffers is a module-local
// alias so that the JIT execution path in egress.rs compiles without changes.
pub use crate::device_slots::DeviceSlotRegistry as SlotBuffers;
