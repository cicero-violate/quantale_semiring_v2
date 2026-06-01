use std::collections::HashMap;

use super::chain::JitChain;

#[cfg(feature = "cuda")]
use {
    super::synth::{JIT_KERNEL_NAME, synthesize_kernel},
    cudarc::driver::{CudaDevice, CudaFunction},
    cudarc::nvrtc::compile_ptx,
    serde_json::Value,
    std::sync::Arc,
};

#[derive(Default)]
pub struct JitCache {
    modules: HashMap<Vec<String>, String>,
    compile_count: usize,
}

impl JitCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile_count(&self) -> usize {
        self.compile_count
    }

    pub fn contains(&self, chain: &JitChain) -> bool {
        self.modules.contains_key(&chain.operators)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_operator_sequence() {
        let chain = JitChain {
            operators: vec!["a".to_string(), "b".to_string()],
            inputs: vec![],
            outputs: vec!["o".to_string()],
            internals: vec![],
        };
        let mut cache = JitCache::new();
        assert!(!cache.contains(&chain));
        cache
            .modules
            .insert(chain.operators.clone(), "module".to_string());
        assert!(cache.contains(&chain));
    }
}
