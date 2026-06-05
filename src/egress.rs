//! Hardcode-free universal execution pipeline for arbitrary OS processes.

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchAsync;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
pub use crate::device_slots::PinnedHostBuffer;
pub use crate::device_slots::{HostStagingBuffer, UploadQueue};

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

    #[cfg(feature = "cuda")]
    pub fn dispatch_hot_region_with_slots(
        &self,
        world: &mut crate::TensorQuantaleWorld,
        region_id: i32,
        src_node: i32,
        dst_node: i32,
        outcome: i32,
    ) -> Result<(), crate::CudaError> {
        let buffers = self
            .slot_buffers
            .lock()
            .map_err(|_| crate::CudaError::invalid_input("slot buffer lock poisoned"))?;
        world.gpu_dispatch_region_with_slots(&buffers, region_id, src_node, dst_node, outcome)
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

    /// Execute descriptor-backed fusion entries through one batch dispatch
    /// boundary.
    ///
    /// The current JIT cache compiles and launches one synthesized chain per
    /// fusion entry. Keeping this as a batch-shaped API lets the par dispatcher
    /// stop treating fusion descriptors like generic host fallbacks, while the
    /// later multi-chain CUDA ABI can replace this implementation without
    /// changing the par dispatch surface.
    pub fn execute_fusion_entries_batch_blocking(
        &self,
        entries: &[(usize, &FusionEntry)],
        dynamic_payload: &Value,
    ) -> Vec<(usize, ProcessReceipt)> {
        #[cfg(feature = "cuda")]
        {
            return execute_jit_fusion_batch_blocking(
                entries,
                dynamic_payload,
                &self.operator_registry,
                &self.jit_cache,
                &self.slot_buffers,
            );
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = dynamic_payload;
            entries
                .iter()
                .map(|&(idx, entry)| {
                    (
                        idx,
                        cuda_err_receipt(
                            &format!("Fusion::{}", entry.region),
                            "jit_cuda fusion batch requires the cuda feature",
                        ),
                    )
                })
                .collect()
        }
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
fn execute_jit_fusion_batch_blocking(
    entries: &[(usize, &FusionEntry)],
    dynamic_payload: &Value,
    registry: &HashMap<String, Value>,
    cache: &Mutex<JitCache>,
    slot_buffers: &Mutex<SlotBuffers>,
) -> Vec<(usize, ProcessReceipt)> {
    use cudarc::driver::{CudaDevice, LaunchConfig};
    use std::sync::{Arc, OnceLock};

    if entries.is_empty() {
        return Vec::new();
    }

    let batch_name = format!(
        "FusionBatch::{}",
        entries
            .iter()
            .map(|(_, entry)| entry.region.as_str())
            .collect::<Vec<_>>()
            .join("__")
    );

    if entries.len() > 3 {
        return entries
            .iter()
            .map(|&(idx, entry)| {
                (
                    idx,
                    cuda_err_receipt(
                        &format!("Fusion::{}", entry.region),
                        format!(
                            "jit_cuda fusion batch launch supports at most 3 chains, got {}",
                            entries.len()
                        ),
                    ),
                )
            })
            .collect();
    }

    static DEVICE: OnceLock<Result<Arc<CudaDevice>, String>> = OnceLock::new();
    let device = match DEVICE.get_or_init(|| CudaDevice::new(0).map_err(|e| e.to_string())) {
        Ok(d) => Arc::clone(d),
        Err(e) => {
            return entries
                .iter()
                .map(|&(idx, entry)| {
                    (
                        idx,
                        cuda_err_receipt(&format!("Fusion::{}", entry.region), e),
                    )
                })
                .collect();
        }
    };

    let chains: Vec<JitChain> = entries
        .iter()
        .map(|(_, entry)| entry.chain.clone())
        .collect();
    if let Some((idx, entry)) = entries
        .iter()
        .find(|(_, entry)| entry.chain.outputs.len() != 1 || entry.chain.inputs.len() > 3)
    {
        return vec![(
            *idx,
            cuda_err_receipt(
                &format!("Fusion::{}", entry.region),
                format!(
                    "jit_cuda fusion batch supports 1 output and 1..=3 inputs per chain, got {} outputs and {} inputs",
                    entry.chain.outputs.len(),
                    entry.chain.inputs.len()
                ),
            ),
        )];
    }
    let total_inputs: usize = entries
        .iter()
        .map(|(_, entry)| entry.chain.inputs.len())
        .sum();
    if total_inputs > 8 {
        return entries
            .iter()
            .map(|&(idx, entry)| {
                (
                    idx,
                    cuda_err_receipt(
                        &format!("Fusion::{}", entry.region),
                        format!(
                            "jit_cuda fusion batch launch supports at most 8 total inputs for 3-chain batches, got {total_inputs}"
                        ),
                    ),
                )
            })
            .collect();
    }

    let mut all_input_slots = Vec::new();
    for chain in &chains {
        all_input_slots.extend(chain.inputs.iter().cloned());
    }
    let n = payload_array_len_for_slots(dynamic_payload, &all_input_slots);

    let mut inputs_by_chain = Vec::with_capacity(chains.len());
    {
        let mut buffers = slot_buffers
            .lock()
            .map_err(|_| cuda_err_receipt(&batch_name, "slot buffer lock poisoned"));
        let Ok(ref mut buffers) = buffers else {
            let receipt = buffers.err().unwrap();
            return entries
                .iter()
                .map(|&(idx, _)| (idx, receipt.clone()))
                .collect();
        };

        for chain in &chains {
            let mut inputs = Vec::with_capacity(chain.inputs.len());
            for (idx, slot) in chain.inputs.iter().enumerate() {
                if let Some(buffer) = buffers.get(slot) {
                    inputs.push(buffer.clone());
                } else {
                    let host = payload_slot_floats(dynamic_payload, slot, idx, n);
                    let dev = match device.htod_copy(host) {
                        Ok(dev) => dev,
                        Err(error) => {
                            return entries
                                .iter()
                                .map(|&(idx, entry)| {
                                    (
                                        idx,
                                        cuda_err_receipt(
                                            &format!("Fusion::{}", entry.region),
                                            format!("htod_copy failed: {error}"),
                                        ),
                                    )
                                })
                                .collect();
                        }
                    };
                    buffers.insert(slot.clone(), dev.clone());
                    inputs.push(dev);
                }
            }
            inputs_by_chain.push(inputs);
        }
    }

    let func = {
        let mut cache = cache
            .lock()
            .map_err(|_| cuda_err_receipt(&batch_name, "JIT cache lock poisoned"));
        let Ok(ref mut cache) = cache else {
            let receipt = cache.err().unwrap();
            return entries
                .iter()
                .map(|&(idx, _)| (idx, receipt.clone()))
                .collect();
        };
        match cache.get_or_compile_batch(&device, &chains, registry) {
            Ok(func) => func,
            Err(error) => {
                return entries
                    .iter()
                    .map(|&(idx, entry)| {
                        (
                            idx,
                            cuda_err_receipt(&format!("Fusion::{}", entry.region), &error),
                        )
                    })
                    .collect();
            }
        }
    };

    let mut outputs = Vec::with_capacity(chains.len());
    for _ in &chains {
        match device.htod_copy(vec![0.0f32; n]) {
            Ok(out) => outputs.push(out),
            Err(error) => {
                return entries
                    .iter()
                    .map(|&(idx, entry)| {
                        (
                            idx,
                            cuda_err_receipt(
                                &format!("Fusion::{}", entry.region),
                                format!("htod_copy output failed: {error}"),
                            ),
                        )
                    })
                    .collect();
            }
        }
    }

    let threads: u32 = 256;
    let blocks: u32 = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    if let Err(error) = launch_jit_batch(func, cfg, &inputs_by_chain, &mut outputs, n) {
        return entries
            .iter()
            .map(|&(idx, entry)| {
                (
                    idx,
                    cuda_err_receipt(
                        &format!("Fusion::{}", entry.region),
                        format!("batch kernel launch failed: {error}"),
                    ),
                )
            })
            .collect();
    }

    let mut receipts = Vec::with_capacity(entries.len());
    for ((idx, entry), out) in entries.iter().zip(outputs.iter()) {
        let results = match device.dtoh_sync_copy(out) {
            Ok(v) => v,
            Err(error) => {
                receipts.push((
                    *idx,
                    cuda_err_receipt(
                        &format!("Fusion::{}", entry.region),
                        format!("dtoh_sync_copy failed: {error}"),
                    ),
                ));
                continue;
            }
        };
        let stdout = serde_json::json!({
            "node": format!("Fusion::{}", entry.region),
            "kernel": "jit_fused_batch",
            "n": n,
            "chain": entry.chain.operators,
            "outputs": entry.chain.outputs,
            "results": &results[..results.len().min(8)],
        })
        .to_string();
        receipts.push((
            *idx,
            ProcessReceipt {
                node_name: format!("Fusion::{}", entry.region),
                exit_code: 0,
                stdout_payload: stdout,
                stderr_payload: String::new(),
            },
        ));
    }

    if let Ok(mut buffers) = slot_buffers.lock() {
        for (entry, out) in entries.iter().map(|(_, entry)| *entry).zip(outputs) {
            buffers.insert(entry.chain.outputs[0].clone(), out);
        }
    }

    receipts
}

#[cfg(feature = "cuda")]
fn launch_jit_batch(
    func: cudarc::driver::CudaFunction,
    cfg: cudarc::driver::LaunchConfig,
    inputs_by_chain: &[Vec<cudarc::driver::CudaSlice<f32>>],
    outputs: &mut [cudarc::driver::CudaSlice<f32>],
    n: usize,
) -> Result<(), String> {
    let launch_result = unsafe {
        match inputs_by_chain {
            [c0] => {
                let out0 = &mut outputs[0];
                match c0.as_slice() {
                    [a] => func.launch(cfg, (a, out0, n as i32)),
                    [a, b] => func.launch(cfg, (a, b, out0, n as i32)),
                    [a, b, c] => func.launch(cfg, (a, b, c, out0, n as i32)),
                    _ => return Err(format!("unsupported chain input count {}", c0.len())),
                }
            }
            [c0, c1] => {
                let (left, right) = outputs.split_at_mut(1);
                let out0 = &mut left[0];
                let out1 = &mut right[0];
                match (c0.as_slice(), c1.as_slice()) {
                    ([a0], [a1]) => func.launch(cfg, (a0, out0, a1, out1, n as i32)),
                    ([a0], [a1, b1]) => func.launch(cfg, (a0, out0, a1, b1, out1, n as i32)),
                    ([a0], [a1, b1, c1]) => {
                        func.launch(cfg, (a0, out0, a1, b1, c1, out1, n as i32))
                    }
                    ([a0, b0], [a1]) => func.launch(cfg, (a0, b0, out0, a1, out1, n as i32)),
                    ([a0, b0], [a1, b1]) => {
                        func.launch(cfg, (a0, b0, out0, a1, b1, out1, n as i32))
                    }
                    ([a0, b0], [a1, b1, c1]) => {
                        func.launch(cfg, (a0, b0, out0, a1, b1, c1, out1, n as i32))
                    }
                    ([a0, b0, c0], [a1]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, out1, n as i32))
                    }
                    ([a0, b0, c0], [a1, b1]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, b1, out1, n as i32))
                    }
                    ([a0, b0, c0], [a1, b1, c1]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, b1, c1, out1, n as i32))
                    }
                    _ => {
                        return Err(format!(
                            "unsupported chain input counts {} and {}",
                            c0.len(),
                            c1.len()
                        ));
                    }
                }
            }
            [c0, c1, c2] => {
                let (first, rest) = outputs.split_at_mut(1);
                let (second, third) = rest.split_at_mut(1);
                let out0 = &mut first[0];
                let out1 = &mut second[0];
                let out2 = &mut third[0];
                match (c0.as_slice(), c1.as_slice(), c2.as_slice()) {
                    ([a0], [a1], [a2]) => {
                        func.launch(cfg, (a0, out0, a1, out1, a2, out2, n as i32))
                    }
                    ([a0], [a1], [a2, b2]) => {
                        func.launch(cfg, (a0, out0, a1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0], [a1], [a2, b2, c2]) => {
                        func.launch(cfg, (a0, out0, a1, out1, a2, b2, c2, out2, n as i32))
                    }
                    ([a0], [a1, b1], [a2]) => {
                        func.launch(cfg, (a0, out0, a1, b1, out1, a2, out2, n as i32))
                    }
                    ([a0], [a1, b1], [a2, b2]) => {
                        func.launch(cfg, (a0, out0, a1, b1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0], [a1, b1], [a2, b2, c2]) => {
                        func.launch(cfg, (a0, out0, a1, b1, out1, a2, b2, c2, out2, n as i32))
                    }
                    ([a0], [a1, b1, c1], [a2]) => {
                        func.launch(cfg, (a0, out0, a1, b1, c1, out1, a2, out2, n as i32))
                    }
                    ([a0], [a1, b1, c1], [a2, b2]) => {
                        func.launch(cfg, (a0, out0, a1, b1, c1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0], [a1, b1, c1], [a2, b2, c2]) => func.launch(
                        cfg,
                        (a0, out0, a1, b1, c1, out1, a2, b2, c2, out2, n as i32),
                    ),
                    ([a0, b0], [a1], [a2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, out1, a2, out2, n as i32))
                    }
                    ([a0, b0], [a1], [a2, b2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0, b0], [a1], [a2, b2, c2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, out1, a2, b2, c2, out2, n as i32))
                    }
                    ([a0, b0], [a1, b1], [a2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, b1, out1, a2, out2, n as i32))
                    }
                    ([a0, b0], [a1, b1], [a2, b2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, b1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0, b0], [a1, b1], [a2, b2, c2]) => func.launch(
                        cfg,
                        (a0, b0, out0, a1, b1, out1, a2, b2, c2, out2, n as i32),
                    ),
                    ([a0, b0], [a1, b1, c1], [a2]) => {
                        func.launch(cfg, (a0, b0, out0, a1, b1, c1, out1, a2, out2, n as i32))
                    }
                    ([a0, b0], [a1, b1, c1], [a2, b2]) => func.launch(
                        cfg,
                        (a0, b0, out0, a1, b1, c1, out1, a2, b2, out2, n as i32),
                    ),
                    ([a0, b0], [a1, b1, c1], [a2, b2, c2]) => func.launch(
                        cfg,
                        (a0, b0, out0, a1, b1, c1, out1, a2, b2, c2, out2, n as i32),
                    ),
                    ([a0, b0, c0], [a1], [a2]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, out1, a2, out2, n as i32))
                    }
                    ([a0, b0, c0], [a1], [a2, b2]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, out1, a2, b2, out2, n as i32))
                    }
                    ([a0, b0, c0], [a1], [a2, b2, c2]) => func.launch(
                        cfg,
                        (a0, b0, c0, out0, a1, out1, a2, b2, c2, out2, n as i32),
                    ),
                    ([a0, b0, c0], [a1, b1], [a2]) => {
                        func.launch(cfg, (a0, b0, c0, out0, a1, b1, out1, a2, out2, n as i32))
                    }
                    ([a0, b0, c0], [a1, b1], [a2, b2]) => func.launch(
                        cfg,
                        (a0, b0, c0, out0, a1, b1, out1, a2, b2, out2, n as i32),
                    ),
                    ([a0, b0, c0], [a1, b1], [a2, b2, c2]) => func.launch(
                        cfg,
                        (a0, b0, c0, out0, a1, b1, out1, a2, b2, c2, out2, n as i32),
                    ),
                    ([a0, b0, c0], [a1, b1, c1], [a2]) => func.launch(
                        cfg,
                        (a0, b0, c0, out0, a1, b1, c1, out1, a2, out2, n as i32),
                    ),
                    ([a0, b0, c0], [a1, b1, c1], [a2, b2]) => func.launch(
                        cfg,
                        (a0, b0, c0, out0, a1, b1, c1, out1, a2, b2, out2, n as i32),
                    ),
                    ([_, _, _], [_, _, _], [_, _, _]) => {
                        return Err("unsupported chain input counts 3, 3, and 3".to_string());
                    }
                    _ => {
                        return Err(format!(
                            "unsupported chain input counts {}, {}, and {}",
                            c0.len(),
                            c1.len(),
                            c2.len()
                        ));
                    }
                }
            }
            _ => {
                return Err(format!(
                    "unsupported batch chain count {}",
                    inputs_by_chain.len()
                ));
            }
        }
    };
    launch_result.map_err(|error| format!("{error}"))
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

    {
        let mut buffers = slot_buffers
            .lock()
            .map_err(|_| cuda_err_receipt(node_name, "slot buffer lock poisoned"));
        let Ok(ref mut buffers) = buffers else {
            return buffers.err().unwrap();
        };
        buffers.insert(chain.outputs[0].clone(), dev_out.clone());
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
