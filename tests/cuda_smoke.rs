use quantale_semiring_v2::{JitCache, JitChain, OperatorRegistry, load_operator_registry};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

#[test]
fn cuda_feature_runtime_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("skip: cuda feature disabled");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        let device = match CudaDevice::new(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skip: no cuda device: {error}");
                return Ok(());
            }
        };

        let registry = load_operator_registry("assets/operators.generated.json")
            .or_else(|_| load_operator_registry("assets/operators.json"))?;
        let chain = JitChain {
            operators: vec!["Analysis::Return1".to_string()],
            inputs: vec!["market.price".to_string(), "market.open".to_string()],
            outputs: vec!["analysis.return".to_string()],
            internals: vec![],
        };

        let func = JitCache::new()
            .get_or_compile(&device, &chain, &registry as &OperatorRegistry)
            .map_err(|error| format!("jit compile failed: {error}"))?;
        let price = device.htod_copy(vec![110.0f32, 120.0, 90.0, 105.0])?;
        let open = device.htod_copy(vec![100.0f32, 100.0, 100.0, 100.0])?;
        let mut out = device.htod_copy(vec![0.0f32; 4])?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, (&price, &open, &mut out, 4_i32))? };
        let actual = device.dtoh_sync_copy(&out)?;
        let expected = [0.1f32, 0.2, -0.1, 0.05];
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    Ok(())
}
