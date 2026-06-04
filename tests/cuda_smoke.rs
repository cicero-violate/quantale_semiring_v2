#[cfg(feature = "cuda")]
mod cuda_smoke {
    use quantale_semiring_v2::{JitCache, JitChain, OperatorRegistry, load_operator_registry};
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

    #[test]
    fn cuda_feature_runtime_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };

        let registry = load_operator_registry("assets/operators.generated.json")?;
        let chain = JitChain {
            operators: vec!["Analysis::Return1".to_string()],
            inputs: vec!["market.price".to_string(), "market.open".to_string()],
            outputs: vec!["analysis.return".to_string()],
            internals: vec![],
        };

        let func = JitCache::new()
            .get_or_compile(&device, &chain, &registry as &OperatorRegistry)
            .map_err(|e| format!("jit compile failed: {e}"))?;
        let price = device.htod_copy(vec![110.0f32, 120.0, 90.0, 105.0])?;
        let open  = device.htod_copy(vec![100.0f32, 100.0, 100.0, 100.0])?;
        let mut out = device.htod_copy(vec![0.0f32; 4])?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, (&price, &open, &mut out, 4_i32))? };
        let actual = device.dtoh_sync_copy(&out)?;
        let expected = [0.1f32, 0.2, -0.1, 0.05];
        for (idx, (&a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-5,
                "idx={idx} actual={a} expected={e}"
            );
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_smoke_skipped_without_cuda_feature() {}
