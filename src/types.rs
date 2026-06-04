//! Runtime scalar constants, receipt types, and generalized data-kind taxonomy.

use serde::{Deserialize, Serialize};

pub const BOTTOM: f32 = 0.0;
pub const Q_UNIT: f32 = 1.0;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProcessReceipt {
    pub node_name: String,
    pub exit_code: i32,
    pub stdout_payload: String,
    pub stderr_payload: String,
}

/// Generalized GPU data classification for typed device slots.
///
/// Covers the full range of data kinds that the hot GPU fabric can hold
/// so that `DeviceSlotRegistry` is not tied to financial tensor shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataKind {
    Tensor,
    Table,
    Stream,
    Graph,
    Text,
    Embedding,
    Image,
    Audio,
    SparseMatrix,
    EventLog,
    KeyValue,
    TimeSeries,
}

impl Default for DataKind {
    fn default() -> Self {
        Self::Tensor
    }
}
