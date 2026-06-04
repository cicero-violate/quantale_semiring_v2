//! Hardcode-free universal execution pipeline for arbitrary OS processes.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
#[cfg(feature = "cuda")]
use std::sync::Mutex;

pub use crate::device_slots::{AsyncUploadQueue, PinnedHostBuffer};

use serde_json::Value;

use crate::config::SystemConfig;
use crate::fusion_dispatch::FusionEntry;
use crate::hot_region::HotRegionRegistry;
#[cfg(feature = "cuda")]
use crate::jit_kernel_fusion::{JitCache, JitChain, SlotBuffers};
use crate::topology::{GraphTopology, NodeRegistry};
use crate::types::ProcessReceipt;

pub struct UniversalExecutor {
    /// Mapping of node names to their generated runtime operator contracts.
    pub operator_registry: HashMap<String, Value>,
    node_registry: NodeRegistry,
    #[allow(dead_code)]
    hot_region_registry: HotRegionRegistry,
    #[cfg(feature = "cuda")]
    jit_cache: Mutex<JitCache>,
    #[cfg(feature = "cuda")]
    slot_buffers: Mutex<SlotBuffers>,
}

impl UniversalExecutor {
    pub fn from_config(config: &SystemConfig) -> Self {
        Self {
            operator_registry: config.operator_registry.clone(),
            node_registry: GraphTopology::bundled_registry()
                .expect("topology.generated.json or bundled topology must compile"),
            hot_region_registry: config.hot_region_registry.clone(),
            #[cfg(feature = "cuda")]
            jit_cache: Mutex::new(JitCache::new()),
            #[cfg(feature = "cuda")]
            slot_buffers: Mutex::new(SlotBuffers::default()),
        }
    }

    pub fn from_registry(
        operator_registry: HashMap<String, Value>,
        node_registry: NodeRegistry,
    ) -> Self {
        Self {
            operator_registry,
            node_registry,
            hot_region_registry: HotRegionRegistry::default(),
            #[cfg(feature = "cuda")]
            jit_cache: Mutex::new(JitCache::new()),
            #[cfg(feature = "cuda")]
            slot_buffers: Mutex::new(SlotBuffers::default()),
        }
    }

    pub fn new(operator_registry: HashMap<String, Value>) -> Self {
        Self {
            operator_registry,
            node_registry: GraphTopology::bundled_registry()
                .expect("topology.generated.json or bundled topology must compile"),
            hot_region_registry: HotRegionRegistry::default(),
            #[cfg(feature = "cuda")]
            jit_cache: Mutex::new(JitCache::new()),
            #[cfg(feature = "cuda")]
            slot_buffers: Mutex::new(SlotBuffers::default()),
        }
    }

    pub fn node_registry(&self) -> &NodeRegistry {
        &self.node_registry
    }

    /// True if `node_name` should be dispatched via the GPU region path.
    ///
    /// Checks both the `HotRegionRegistry` (nodes with explicit GPU region
    /// metadata) and the operator registry (`executable = "jit_cuda"`).  The
    /// hot path bypasses process-spawning and routes directly through
    /// `TensorQuantaleWorld::gpu_dispatch_region`.
    pub fn is_hot_node(&self, node_name: &str) -> bool {
        if self.hot_region_registry.is_hot(node_name) {
            return true;
        }
        self.operator_registry
            .get(node_name)
            .and_then(|op| op["executable"].as_str())
            .map(|e| e == "jit_cuda")
            .unwrap_or(false)
    }

    /// Return the declared `output_mode` for a node operator, if any.
    ///
    /// Operators that emit a JSON tensor edge plan should declare
    /// `"output_mode": "tensor_plan"`. The main loop uses this to decide
    /// whether to run `compile_llm_tensor_plan` on the operator's stdout.
    pub fn output_mode<'a>(&'a self, node_name: &str) -> Option<&'a str> {
        self.operator_registry
            .get(node_name)
            .and_then(|op| op["output_mode"].as_str())
    }

    /// Blocking implementation used by synchronous host loops and tests.
    pub fn execute_abstract_node_blocking(
        &self,
        node_name: &str,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        let op_config = match self.operator_registry.get(node_name) {
            Some(config) => config,
            None => {
                return ProcessReceipt {
                    node_name: node_name.to_string(),
                    exit_code: 127,
                    stdout_payload: String::new(),
                    stderr_payload: "Node operator contract missing from registry".to_string(),
                };
            }
        };

        let binary = op_config["executable"].as_str().unwrap_or("false");
        if binary == "jit_cuda" {
            return self.execute_jit_cuda_blocking(node_name, dynamic_payload);
        }
        let empty_args = Vec::new();
        let static_args: Vec<&str> = op_config["static_args"]
            .as_array()
            .unwrap_or(&empty_args)
            .iter()
            .filter_map(Value::as_str)
            .collect();

        let mut command = Command::new(binary);
        command.args(&static_args);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return ProcessReceipt {
                    node_name: node_name.to_string(),
                    exit_code: 1,
                    stdout_payload: String::new(),
                    stderr_payload: format!("Failed to spawn process: {error}"),
                };
            }
        };

        let stdin_mode = op_config["input_mapping"]["stdin_mode"]
            .as_str()
            .unwrap_or("field");

        if stdin_mode == "json" {
            if let Some(mut stdin) = child.stdin.take() {
                let json_bytes = serde_json::to_vec(dynamic_payload).unwrap_or_default();
                if let Err(error) = stdin.write_all(&json_bytes) {
                    return ProcessReceipt {
                        node_name: node_name.to_string(),
                        exit_code: 1,
                        stdout_payload: String::new(),
                        stderr_payload: format!("Failed to write JSON stdin: {error}"),
                    };
                }
            }
        } else if let Some(stdin_field) = op_config["input_mapping"]["stdin_source"].as_str() {
            if let Some(mut stdin) = child.stdin.take() {
                let content = dynamic_payload[stdin_field].as_str().unwrap_or("");
                if let Err(error) = stdin.write_all(content.as_bytes()) {
                    return ProcessReceipt {
                        node_name: node_name.to_string(),
                        exit_code: 1,
                        stdout_payload: String::new(),
                        stderr_payload: format!("Failed to write process stdin: {error}"),
                    };
                }
            }
        }

        match child.wait_with_output() {
            Ok(output) => ProcessReceipt {
                node_name: node_name.to_string(),
                exit_code: output.status.code().unwrap_or(1),
                stdout_payload: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr_payload: String::from_utf8_lossy(&output.stderr).into_owned(),
            },
            Err(error) => ProcessReceipt {
                node_name: node_name.to_string(),
                exit_code: 1,
                stdout_payload: String::new(),
                stderr_payload: format!("Failed to wait for process: {error}"),
            },
        }
    }

    /// Execute a precomputed fusion region as one CUDA JIT chain.
    ///
    /// This is the normal runtime bridge from `FusionDispatch` into the same
    /// JIT cache and slot-buffer machinery used by individual `jit_cuda` nodes.
    pub fn execute_fusion_entry_blocking(
        &self,
        entry: &FusionEntry,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        self.execute_jit_chain_blocking(
            &format!("Fusion::{}", entry.region),
            &entry.chain,
            dynamic_payload,
        )
    }

    #[cfg(feature = "cuda")]
    fn execute_jit_chain_blocking(
        &self,
        node_name: &str,
        chain: &JitChain,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        execute_jit_chain_blocking(
            node_name,
            chain,
            dynamic_payload,
            &self.operator_registry,
            &self.jit_cache,
            &self.slot_buffers,
        )
    }

    #[cfg(not(feature = "cuda"))]
    fn execute_jit_chain_blocking(
        &self,
        node_name: &str,
        _chain: &crate::jit_kernel_fusion::JitChain,
        _dynamic_payload: &Value,
    ) -> ProcessReceipt {
        cuda_err_receipt(
            node_name,
            format!("jit_cuda fusion region '{node_name}' requires the cuda feature"),
        )
    }

    #[cfg(feature = "cuda")]
    fn execute_jit_cuda_blocking(
        &self,
        node_name: &str,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        use crate::jit_kernel_fusion::chain_for_single_operator;

        let chain = match chain_for_single_operator(node_name, &self.operator_registry) {
            Ok(chain) => chain,
            Err(error) => return cuda_err_receipt(node_name, error),
        };
        self.execute_jit_chain_blocking(node_name, &chain, dynamic_payload)
    }

    #[cfg(not(feature = "cuda"))]
    fn execute_jit_cuda_blocking(
        &self,
        node_name: &str,
        _dynamic_payload: &Value,
    ) -> ProcessReceipt {
        cuda_err_receipt(
            node_name,
            format!("jit_cuda operator '{node_name}' requires the cuda feature"),
        )
    }
}

#[cfg(feature = "cuda")]
fn execute_jit_chain_blocking(
    node_name: &str,
    chain: &JitChain,
    dynamic_payload: &Value,
    registry: &HashMap<String, Value>,
    cache: &Mutex<JitCache>,
    slot_buffers: &Mutex<SlotBuffers>,
) -> ProcessReceipt {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use std::sync::{Arc, OnceLock};

    static DEVICE: OnceLock<Result<Arc<CudaDevice>, String>> = OnceLock::new();

    let device = match DEVICE.get_or_init(|| CudaDevice::new(0).map_err(|e| e.to_string())) {
        Ok(d) => Arc::clone(d),
        Err(e) => return cuda_err_receipt(node_name, e),
    };

    if chain.outputs.len() != 1 {
        return cuda_err_receipt(
            node_name,
            format!(
                "jit_cuda executor requires exactly one output slot, got {}",
                chain.outputs.len()
            ),
        );
    }

    let n = payload_array_len_for_slots(dynamic_payload, &chain.inputs);
    let mut inputs = Vec::with_capacity(chain.inputs.len());
    {
        let mut buffers = slot_buffers
            .lock()
            .map_err(|_| cuda_err_receipt(node_name, "slot buffer lock poisoned"));
        let Ok(ref mut buffers) = buffers else {
            return buffers.err().unwrap();
        };
        for (idx, slot) in chain.inputs.iter().enumerate() {
            if let Some(buffer) = buffers.get(slot) {
                inputs.push(buffer.clone());
            } else {
                let host = payload_slot_floats(dynamic_payload, slot, idx, n);
                let dev = match device.htod_copy(host) {
                    Ok(dev) => dev,
                    Err(error) => {
                        return cuda_err_receipt(node_name, format!("htod_copy failed: {error}"));
                    }
                };
                buffers.insert(slot.clone(), dev.clone());
                inputs.push(dev);
            }
        }
    }

    let func = {
        let mut cache = cache
            .lock()
            .map_err(|_| cuda_err_receipt(node_name, "JIT cache lock poisoned"));
        let Ok(ref mut cache) = cache else {
            return cache.err().unwrap();
        };
        match cache.get_or_compile(&device, chain, registry) {
            Ok(func) => func,
            Err(error) => return cuda_err_receipt(node_name, error),
        }
    };

    let mut dev_out = match device.htod_copy(vec![0.0f32; n]) {
        Ok(out) => out,
        Err(error) => {
            return cuda_err_receipt(node_name, format!("htod_copy output failed: {error}"));
        }
    };

    let threads: u32 = 256;
    let blocks: u32 = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let launch_result = unsafe {
        match inputs.as_slice() {
            [a] => func.launch(cfg, (a, &mut dev_out, n as i32)),
            [a, b] => func.launch(cfg, (a, b, &mut dev_out, n as i32)),
            [a, b, c] => func.launch(cfg, (a, b, c, &mut dev_out, n as i32)),
            _ => {
                return cuda_err_receipt(
                    node_name,
                    format!(
                        "jit_cuda executor supports 1..=3 input slots, got {}",
                        inputs.len()
                    ),
                );
            }
        }
    };
    if let Err(error) = launch_result {
        return cuda_err_receipt(node_name, format!("kernel launch failed: {error}"));
    }

    let results = match device.dtoh_sync_copy(&dev_out) {
        Ok(v) => v,
        Err(e) => return cuda_err_receipt(node_name, format!("dtoh_sync_copy failed: {e}")),
    };

    let stdout = serde_json::json!({
        "node": node_name,
        "kernel": "jit_fused",
        "n": n,
        "chain": chain.operators,
        "outputs": chain.outputs,
        "results": &results[..results.len().min(8)],
    })
    .to_string();

    if let Ok(mut buffers) = slot_buffers.lock() {
        buffers.insert(chain.outputs[0].clone(), dev_out);
    }

    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 0,
        stdout_payload: stdout,
        stderr_payload: String::new(),
    }
}

fn cuda_err_receipt(node_name: &str, msg: impl std::fmt::Display) -> ProcessReceipt {
    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 1,
        stdout_payload: String::new(),
        stderr_payload: msg.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn payload_array_len_for_slots(value: &Value, slots: &[String]) -> usize {
    slots
        .iter()
        .enumerate()
        .filter_map(|(idx, slot)| payload_array_for_slot(value, slot, idx))
        .map(|a| a.len())
        .max()
        .unwrap_or(64)
        .max(1)
}

#[cfg(feature = "cuda")]
fn payload_slot_floats(value: &Value, slot: &str, idx: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    if let Some(arr) = payload_array_for_slot(value, slot, idx) {
        for (i, v) in arr.iter().enumerate().take(n) {
            out[i] = v.as_f64().unwrap_or(0.0) as f32;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn payload_array_for_slot<'a>(value: &'a Value, slot: &str, idx: usize) -> Option<&'a Vec<Value>> {
    let fallback_keys = ["a", "b", "c"];
    value
        .get(slot)
        .and_then(Value::as_array)
        .or_else(|| {
            slot.rsplit('.')
                .next()
                .and_then(|key| value.get(key))
                .and_then(Value::as_array)
        })
        .or_else(|| {
            fallback_keys
                .get(idx)
                .and_then(|key| value.get(*key))
                .and_then(Value::as_array)
        })
}
