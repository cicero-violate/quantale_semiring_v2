#[cfg(feature = "cuda")]
use {
    super::chain::JitChain,
    super::synth::{
        JIT_BATCH_KERNEL_NAME, JIT_KERNEL_NAME, synthesize_batch_kernel, synthesize_kernel,
    },
    cudarc::driver::{CudaDevice, CudaFunction},
    cudarc::nvrtc::compile_ptx,
    serde_json::Value,
    std::collections::HashMap,
    std::sync::Arc,
};

#[derive(Default)]
pub struct JitCache {
    #[cfg(feature = "cuda")]
    modules: HashMap<Vec<String>, String>,
    #[cfg(feature = "cuda")]
    batch_modules: HashMap<Vec<Vec<String>>, String>,
    #[cfg(feature = "cuda")]
    compile_count: usize,
}

impl JitCache {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "cuda")]
impl JitCache {
    pub fn get_or_compile(
        &mut self,
        device: &Arc<CudaDevice>,
        chain: &JitChain,
        registry: &HashMap<String, Value>,
    ) -> Result<CudaFunction, String> {
        if let Some(module_name) = self.modules.get(&chain.operators) {
            return device
                .get_func(module_name, JIT_KERNEL_NAME)
                .ok_or_else(|| format!("cached JIT function missing from module '{module_name}'"));
        }

        let source = synthesize_kernel(chain, registry)?;
        let ptx = compile_ptx(source).map_err(|error| format!("{error:?}"))?;
        let module_name = format!("jit_kernel_fusion_{}", self.modules.len());
        device
            .load_ptx(ptx, &module_name, &[JIT_KERNEL_NAME])
            .map_err(|error| format!("{error:?}"))?;
        self.modules
            .insert(chain.operators.clone(), module_name.clone());
        self.compile_count += 1;
        device
            .get_func(&module_name, JIT_KERNEL_NAME)
            .ok_or_else(|| format!("JIT function missing from module '{module_name}'"))
    }

    pub fn get_or_compile_batch(
        &mut self,
        device: &Arc<CudaDevice>,
        chains: &[JitChain],
        registry: &HashMap<String, Value>,
    ) -> Result<CudaFunction, String> {
        let key: Vec<Vec<String>> = chains.iter().map(|chain| chain.operators.clone()).collect();
        if let Some(module_name) = self.batch_modules.get(&key) {
            return device
                .get_func(module_name, JIT_BATCH_KERNEL_NAME)
                .ok_or_else(|| {
                    format!("cached JIT batch function missing from module '{module_name}'")
                });
        }

        let source = synthesize_batch_kernel(chains, registry)?;
        let ptx = compile_ptx(source).map_err(|error| format!("{error:?}"))?;
        let module_name = format!("jit_kernel_fusion_batch_{}", self.batch_modules.len());
        device
            .load_ptx(ptx, &module_name, &[JIT_BATCH_KERNEL_NAME])
            .map_err(|error| format!("{error:?}"))?;
        self.batch_modules.insert(key, module_name.clone());
        self.compile_count += 1;
        device
            .get_func(&module_name, JIT_BATCH_KERNEL_NAME)
            .ok_or_else(|| format!("JIT batch function missing from module '{module_name}'"))
    }
}
