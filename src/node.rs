//! Compact numeric node IDs.
//!
//! Names, actions, and graph membership are owned by `topology::NodeRegistry`.

use crate::topology::NodeRegistry;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node(pub i32);

impl Node {
    pub fn encode(self) -> i32 {
        self.0
    }

    pub fn decode(id: i32, registry: &NodeRegistry) -> Option<Self> {
        if id >= 0 && registry.name_of(id as usize).is_some() {
            Some(Self(id))
        } else {
            None
        }
    }

    pub fn name(self, registry: &NodeRegistry) -> Option<&str> {
        registry.name_of(self.0 as usize)
    }
}
