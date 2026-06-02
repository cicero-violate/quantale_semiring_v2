//! CUDA/runtime error type.

use std::fmt;

#[derive(Debug, Clone)]
pub struct CudaError {
    pub operation: &'static str,
    pub message: String,
}

impl CudaError {
    pub(crate) fn new(operation: &'static str, error: impl fmt::Debug) -> Self {
        Self {
            operation,
            message: format!("{error:?}"),
        }
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            operation: "input",
            message: message.into(),
        }
    }

    pub(crate) fn missing_function(name: &'static str) -> Self {
        Self {
            operation: "get_func",
            message: format!("missing CUDA function {name}"),
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA operation {} failed: {}",
            self.operation, self.message
        )
    }
}

impl std::error::Error for CudaError {}

impl From<topology_core::TopologyError> for CudaError {
    fn from(error: topology_core::TopologyError) -> Self {
        Self::invalid_input(error.message)
    }
}
