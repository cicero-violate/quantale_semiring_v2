//! Hardcode-free universal execution pipeline for arbitrary OS processes.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

use crate::config::SystemConfig;
use crate::receipt::ProcessReceipt;
use crate::topology::{GraphTopology, NodeRegistry};

pub struct UniversalExecutor {
    /// Mapping of Node Names to their generic CLI schemas loaded from operators.json.
    pub operator_registry: HashMap<String, Value>,
    node_registry: NodeRegistry,
}

impl UniversalExecutor {
    pub fn from_config(config: &SystemConfig) -> Self {
        Self {
            operator_registry: config.operator_registry.clone(),
            node_registry: GraphTopology::bundled_registry()
                .expect("bundled assets/topology.json must compile"),
        }
    }

    pub fn new(operator_registry: HashMap<String, Value>) -> Self {
        Self {
            operator_registry,
            node_registry: GraphTopology::bundled_registry()
                .expect("bundled assets/topology.json must compile"),
        }
    }

    pub fn node_registry(&self) -> &NodeRegistry {
        &self.node_registry
    }

    /// Spawns and executes any command contract defined in the operators configuration.
    pub async fn execute_abstract_node(
        &self,
        node_name: &str,
        dynamic_payload: &Value,
    ) -> ProcessReceipt {
        self.execute_abstract_node_blocking(node_name, dynamic_payload)
    }

    /// Return the declared `output_mode` for a node operator, if any.
    ///
    /// Operators that emit a JSON tensor edge plan should declare `"output_mode": "tensor_plan"`
    /// in `operators.json`. The main loop uses this to decide whether to run
    /// `compile_llm_tensor_plan` on the operator's stdout.
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
        if binary == "cuda_ptx" {
            return execute_cuda_ptx_blocking(node_name, op_config, dynamic_payload);
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
}

#[cfg(feature = "cuda")]
fn execute_cuda_ptx_blocking(
    node_name: &str,
    op_config: &Value,
    dynamic_payload: &Value,
) -> ProcessReceipt {
    use cudarc::driver::{CudaDevice, LaunchConfig};
    use std::sync::{Arc, OnceLock};

    // PTX compiled from cuda/trading_execution_kernels.cu by build.rs at build time.
    const PTX_BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/trading_execution_kernels.ptx"));
    const MODULE_NAME: &str = "quantale_trading_execution_kernels";
    const KERNEL_NAMES: &[&str] = &[
        "fused_alpha_and_risk_kernel",
        "fused_orderbook_and_alpha_kernel",
        "fused_feed_alpha_and_risk_kernel",
    ];

    // Device and module are initialised once per process.
    static DEVICE: OnceLock<Result<Arc<CudaDevice>, String>> = OnceLock::new();
    static MODULE: OnceLock<Result<(), String>> = OnceLock::new();

    let kernel_name = match op_config["input_mapping"]["kernel"].as_str() {
        Some(k) if !k.is_empty() => k,
        _ => return cuda_err_receipt(node_name, "missing kernel name in input_mapping"),
    };

    let device =
        match DEVICE.get_or_init(|| CudaDevice::new(0).map(Arc::new).map_err(|e| e.to_string())) {
            Ok(d) => Arc::clone(d),
            Err(e) => return cuda_err_receipt(node_name, e),
        };

    if let Err(e) = MODULE.get_or_init(|| {
        let ptx_src = std::str::from_utf8(PTX_BYTES)
            .map_err(|e| format!("PTX bytes are not valid UTF-8: {e}"))?;
        let ptx = cudarc::driver::Ptx::from(ptx_src);
        device
            .load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)
            .map_err(|e| e.to_string())
    }) {
        return cuda_err_receipt(node_name, e);
    }

    let func = match device.get_func(MODULE_NAME, kernel_name) {
        Some(f) => f,
        None => {
            return cuda_err_receipt(
                node_name,
                format!("kernel '{kernel_name}' not found in module '{MODULE_NAME}'"),
            )
        }
    };

    // Marshal JSON payload arrays → device buffers.
    // Fields "a", "b", "c" are float arrays; missing fields default to zeros.
    let n = payload_array_len(dynamic_payload);
    let a = payload_floats(dynamic_payload, "a", n);
    let b = payload_floats(dynamic_payload, "b", n);
    let c = payload_floats(dynamic_payload, "c", n);

    let (dev_a, dev_b, dev_c, mut dev_out) = match (
        device.htod_copy(a),
        device.htod_copy(b),
        device.htod_copy(c),
        device.htod_copy(vec![0.0f32; n]),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(o)) => (a, b, c, o),
        _ => return cuda_err_receipt(node_name, "htod_copy failed"),
    };

    let threads: u32 = 256;
    let blocks: u32 = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    if let Err(e) = unsafe { func.launch(cfg, (&dev_a, &dev_b, &dev_c, &mut dev_out, n as i32)) } {
        return cuda_err_receipt(node_name, format!("kernel launch failed: {e}"));
    }

    let results = match device.dtoh_sync_copy(&dev_out) {
        Ok(v) => v,
        Err(e) => return cuda_err_receipt(node_name, format!("dtoh_sync_copy failed: {e}")),
    };

    let stdout = serde_json::json!({
        "node": node_name,
        "kernel": kernel_name,
        "n": n,
        "results": &results[..results.len().min(8)],
    })
    .to_string();

    ProcessReceipt {
        node_name: node_name.to_string(),
        exit_code: 0,
        stdout_payload: stdout,
        stderr_payload: String::new(),
    }
}

#[cfg(not(feature = "cuda"))]
fn execute_cuda_ptx_blocking(
    node_name: &str,
    _op_config: &Value,
    _dynamic_payload: &Value,
) -> ProcessReceipt {
    cuda_err_receipt(
        node_name,
        format!("cuda_ptx operator '{node_name}' requires the cuda feature"),
    )
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
fn payload_array_len(value: &Value) -> usize {
    ["a", "b", "c"]
        .iter()
        .filter_map(|k| value[k].as_array())
        .map(|a| a.len())
        .max()
        .unwrap_or(64)
        .max(1)
}

#[cfg(feature = "cuda")]
fn payload_floats(value: &Value, key: &str, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    if let Some(arr) = value[key].as_array() {
        for (i, v) in arr.iter().enumerate().take(n) {
            out[i] = v.as_f64().unwrap_or(0.0) as f32;
        }
    }
    out
}
