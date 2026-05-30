//! Runtime configuration for the CUDA quantale orchestrator.

use std::path::PathBuf;

use crate::node::{MATRIX_LEN, NODE_COUNT, THREAD_COUNT};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemConfig {
    pub matrix_dim: usize,
    pub matrix_len: usize,
    pub block_size: usize,
    pub tlog_path: PathBuf,
    pub ingress_capacity_hint: usize,
    pub max_ticks: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            matrix_dim: NODE_COUNT,
            matrix_len: MATRIX_LEN,
            block_size: THREAD_COUNT,
            tlog_path: PathBuf::from("quantale.tlog"),
            ingress_capacity_hint: 1024,
            max_ticks: 64,
        }
    }
}

impl SystemConfig {
    pub fn with_tlog_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tlog_path = path.into();
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.matrix_dim != NODE_COUNT {
            return Err(format!(
                "matrix_dim {} does not match NODE_COUNT {}",
                self.matrix_dim, NODE_COUNT
            ));
        }
        if self.matrix_len != MATRIX_LEN {
            return Err(format!(
                "matrix_len {} does not match MATRIX_LEN {}",
                self.matrix_len, MATRIX_LEN
            ));
        }
        if self.block_size == 0 {
            return Err("block_size must be nonzero".to_string());
        }
        Ok(())
    }
}
