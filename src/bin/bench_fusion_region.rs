#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use quantale_semiring_v2::{
    FusionDispatch, JitCache, OperatorRegistry, UniversalExecutor, load_operator_registry,
};
#[cfg(feature = "cuda")]
use serde_json::{Value, json};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda")]
const ANALYSIS_ENTRY: &str = "Analysis::Return1";
#[cfg(feature = "cuda")]
const ANALYSIS_CHAIN: [&str; 3] = [
    "Analysis::Return1",
    "Analysis::Volatility",
    "Analysis::SignalScore",
];

#[cfg(feature = "cuda")]
fn parse_iterations() -> usize {
    env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1000)
}

#[cfg(feature = "cuda")]
fn analysis_payload() -> Value {
    json!({
        "market.price": [110.0, 120.0, 90.0, 105.0, 101.0, 99.0, 130.0, 80.0],
        "market.open":  [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    })
}

#[cfg(feature = "cuda")]
fn receipt_results(receipt_stdout: &str) -> Result<Vec<f32>, String> {
    let value: Value = serde_json::from_str(receipt_stdout)
        .map_err(|error| format!("parse receipt stdout: {error}: {receipt_stdout}"))?;
    value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("receipt missing results array: {value}"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| format!("non-float result: {item}"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn canonical_once(
    executor: &UniversalExecutor,
    payload: &Value,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    for node in ANALYSIS_CHAIN {
        let receipt = executor.execute_abstract_node_blocking(node, payload);
        if receipt.exit_code != 0 {
            return Err(format!("{node} failed: {}", receipt.stderr_payload).into());
        }
        if node == "Analysis::SignalScore" {
            return Ok(receipt_results(&receipt.stdout_payload)?);
        }
    }
    Err("analysis chain was empty".into())
}

#[cfg(feature = "cuda")]
struct FusedRegion {
    device: std::sync::Arc<CudaDevice>,
    func: cudarc::driver::CudaFunction,
    inputs: Vec<cudarc::driver::CudaSlice<f32>>,
    out: cudarc::driver::CudaSlice<f32>,
    n: usize,
}

#[cfg(feature = "cuda")]
impl FusedRegion {
    fn new(
        registry: &OperatorRegistry,
        payload: &Value,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        let dispatch = FusionDispatch::load("assets/topology.fusion.json", registry)?;
        let entry = dispatch
            .get_by_entry(ANALYSIS_ENTRY)
            .ok_or_else(|| format!("missing fusion entry for {ANALYSIS_ENTRY}"))?;
        let mut cache = JitCache::new();
        let func = cache.get_or_compile(&device, &entry.chain, registry)?;

        let mut inputs = Vec::new();
        let mut n = None;
        for slot in &entry.chain.inputs {
            let arr = payload
                .get(slot)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("payload missing input slot {slot}"))?;
            n = Some(n.unwrap_or(arr.len()).max(arr.len()));
            let host = arr
                .iter()
                .map(|item| item.as_f64().unwrap_or(0.0) as f32)
                .collect::<Vec<_>>();
            inputs.push(device.htod_copy(host)?);
        }
        let n = n.unwrap_or(0).max(1);
        let out = device.htod_copy(vec![0.0f32; n])?;
        Ok(Self {
            device,
            func,
            inputs,
            out,
            n,
        })
    }

    fn run(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = LaunchConfig {
            grid_dim: (((self.n as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            match self.inputs.as_slice() {
                [a, b] => self
                    .func
                    .clone()
                    .launch(cfg, (a, b, &mut self.out, self.n as i32))?,
                [a, b, c] => self
                    .func
                    .clone()
                    .launch(cfg, (a, b, c, &mut self.out, self.n as i32))?,
                _ => return Err(format!("unexpected input count {}", self.inputs.len()).into()),
            }
        }
        Ok(self.device.dtoh_sync_copy(&self.out)?)
    }
}

#[cfg(feature = "cuda")]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(feature = "cuda")]
fn avg_us(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1_000_000.0 / iterations as f64
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("{{\"status\":\"skipped\",\"reason\":\"cuda feature disabled\"}}");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = parse_iterations();
    let payload = analysis_payload();
    let registry = load_operator_registry("assets/operators.generated.json")?;

    let setup_start = Instant::now();
    let executor = UniversalExecutor::new(registry.clone());
    let mut fused = match FusedRegion::new(&registry, &payload) {
        Ok(fused) => fused,
        Err(error) => {
            println!(
                "{}",
                json!({
                    "status": "skipped",
                    "reason": error.to_string(),
                    "region": ANALYSIS_CHAIN
                })
            );
            return Ok(());
        }
    };
    let setup = setup_start.elapsed();

    let canonical_warmup = canonical_once(&executor, &payload)?;
    let fused_warmup = fused.run()?;
    if canonical_warmup.len() != fused_warmup.len()
        || canonical_warmup
            .iter()
            .zip(fused_warmup.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5)
    {
        return Err(format!(
            "warmup mismatch canonical={canonical_warmup:?} fused={fused_warmup:?}"
        )
        .into());
    }

    let canonical_start = Instant::now();
    for _ in 0..iterations {
        let _ = canonical_once(&executor, &payload)?;
    }
    let canonical = canonical_start.elapsed();

    let fused_start = Instant::now();
    for _ in 0..iterations {
        let _ = fused.run()?;
    }
    let fused_elapsed = fused_start.elapsed();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "region": ANALYSIS_CHAIN,
            "iterations": iterations,
            "input_len": fused.n,
            "setup_ms": duration_ms(setup),
            "canonical_total_ms": duration_ms(canonical),
            "canonical_avg_us": avg_us(canonical, iterations),
            "fused_total_ms": duration_ms(fused_elapsed),
            "fused_avg_us": avg_us(fused_elapsed, iterations),
            "speedup": canonical.as_secs_f64() / fused_elapsed.as_secs_f64(),
            "result_sample": fused_warmup,
            "note": "timings include synchronized result copy; canonical path uses three jit_cuda dispatches through UniversalExecutor"
        }))?
    );
    Ok(())
}
