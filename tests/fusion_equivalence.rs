#[cfg(not(feature = "cuda"))]
#[test]
fn fusion_equivalence_skipped_without_cuda_feature() {}

#[cfg(feature = "cuda")]
mod cuda_fusion {
    use quantale_semiring_v2::{
        FusionDispatch, JitCache, OperatorRegistry, UniversalExecutor, load_operator_registry,
    };
    use serde_json::{Value, json};
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

    const ANALYSIS_ENTRY: &str = "Analysis::Return1";
    const TOLERANCE: f32 = 1e-5;

    fn analysis_payload() -> Value {
        json!({
            "market.price": [110.0, 120.0, 90.0, 105.0, 101.0, 99.0, 130.0, 80.0],
            "market.open":  [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        })
    }

    fn receipt_results(receipt_stdout: &str) -> Result<Vec<f32>, String> {
        let value: Value = serde_json::from_str(receipt_stdout)
            .map_err(|e| format!("parse receipt stdout: {e}: {receipt_stdout}"))?;
        value
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("receipt missing results array: {value}"))?
            .iter()
            .map(|item| {
                item.as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| format!("non-float result: {item}"))
            })
            .collect()
    }

    fn canonical_results(registry: &OperatorRegistry) -> Result<Vec<f32>, String> {
        let executor = UniversalExecutor::new(registry.clone());
        let payload = analysis_payload();
        for node in ["Analysis::Return1", "Analysis::Volatility"] {
            let receipt = executor.execute_abstract_node_blocking(node, &payload);
            if receipt.exit_code != 0 {
                return Err(format!("{node} failed: {}", receipt.stderr_payload));
            }
        }
        let receipt = executor.execute_abstract_node_blocking("Analysis::SignalScore", &payload);
        if receipt.exit_code != 0 {
            return Err(format!(
                "Analysis::SignalScore failed: {}",
                receipt.stderr_payload
            ));
        }
        receipt_results(&receipt.stdout_payload)
    }

    fn fused_results(registry: &OperatorRegistry) -> Result<Option<Vec<f32>>, String> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(None);
            }
        };
        let dispatch = FusionDispatch::load("assets/topology.fusion.json", registry)?;
        let entry = dispatch
            .get_by_entry(ANALYSIS_ENTRY)
            .ok_or_else(|| format!("missing fusion entry for {ANALYSIS_ENTRY}"))?;
        assert_eq!(entry.writes, vec!["analysis.signal_score".to_string()]);

        let mut cache = JitCache::new();
        let func = cache
            .get_or_compile(&device, &entry.chain, registry)
            .map_err(|e| format!("jit compile failed: {e}"))?;

        let payload = analysis_payload();
        let mut inputs = Vec::new();
        let mut n = None;
        for slot in &entry.chain.inputs {
            let arr = payload
                .get(slot)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("payload missing input slot {slot}"))?;
            n = Some(n.unwrap_or(arr.len()).max(arr.len()));
            let host: Vec<f32> = arr
                .iter()
                .map(|item| item.as_f64().unwrap_or(0.0) as f32)
                .collect();
            inputs.push(device.htod_copy(host).map_err(|e| e.to_string())?);
        }
        let n = n.unwrap_or(0);
        let mut out = device
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| e.to_string())?;
        let cfg = LaunchConfig {
            grid_dim: (((n as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            match inputs.as_slice() {
                [a, b] => func
                    .launch(cfg, (a, b, &mut out, n as i32))
                    .map_err(|e| e.to_string())?,
                [a, b, c] => func
                    .launch(cfg, (a, b, c, &mut out, n as i32))
                    .map_err(|e| e.to_string())?,
                _ => return Err(format!("unexpected input count {}", inputs.len())),
            }
        }
        let actual = device
            .dtoh_sync_copy(&out)
            .map_err(|e| e.to_string())?;
        Ok(Some(actual))
    }

    #[test]
    fn canonical_and_fused_market_analysis_are_equivalent() -> Result<(), String> {
        let registry = load_operator_registry("assets/operators.generated.json")?;
        let Some(fused) = fused_results(&registry)? else {
            return Ok(());
        };
        let canonical = canonical_results(&registry)?;
        assert_eq!(canonical.len(), fused.len());
        for (idx, (c, f)) in canonical.iter().zip(fused.iter()).enumerate() {
            assert!(
                (c - f).abs() <= TOLERANCE,
                "idx={idx} canonical={c} fused={f}"
            );
        }
        Ok(())
    }
}
