use std::env;
use std::time::{Duration, Instant};

use quantale_semiring_v2::{CudaWorld, full_transition_edges};

#[derive(Clone, Copy, Debug)]
struct Sample {
    iterations: usize,
    total: Duration,
}

impl Sample {
    fn avg_us(self) -> f64 {
        self.total.as_secs_f64() * 1_000_000.0 / self.iterations as f64
    }

    fn total_ms(self) -> f64 {
        self.total.as_secs_f64() * 1_000.0
    }
}

fn parse_iterations() -> usize {
    env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(100)
}

fn bench<F>(iterations: usize, mut f: F) -> Result<Sample, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<Duration, Box<dyn std::error::Error>>,
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        total += f()?;
    }
    Ok(Sample { iterations, total })
}

fn timed<F>(mut f: F) -> Result<Duration, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = Instant::now();
    f()?;
    Ok(start.elapsed())
}

fn print_sample(name: &str, sample: Sample) {
    println!(
        "{name:<24} iterations={:<6} total_ms={:>10.3} avg_us={:>10.3}",
        sample.iterations,
        sample.total_ms(),
        sample.avg_us()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = parse_iterations();
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let edges = full_transition_edges();

    println!("quantale_semiring_v2 benchmark");
    println!("profile={profile}");
    println!("iterations={iterations}");
    println!("edge_count={}", edges.len());

    let mut warmup = CudaWorld::from_edges(&edges)?;
    warmup.step()?;
    warmup.frontier_step()?;
    warmup.synchronize()?;

    let mut closure_world = CudaWorld::from_edges(&edges)?;
    let closure = bench(iterations, || {
        closure_world.reset()?;
        closure_world.load_edges(&edges)?;
        closure_world.synchronize()?;
        timed(|| {
            closure_world.closure_assign()?;
            closure_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("closure", closure);

    let mut projection_world = CudaWorld::from_edges(&edges)?;
    projection_world.closure_assign()?;
    projection_world.synchronize()?;
    let projection = bench(iterations, || {
        timed(|| {
            projection_world.project_decision_path()?;
            projection_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("projection", projection);

    let mut frontier_world = CudaWorld::from_edges(&edges)?;
    frontier_world.step()?;
    frontier_world.synchronize()?;
    let frontier_step = bench(iterations, || {
        timed(|| {
            frontier_world.frontier_step()?;
            frontier_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("frontier_step", frontier_step);

    let mut tick_world = CudaWorld::from_edges(&edges)?;
    let end_to_end_tick = bench(iterations, || {
        timed(|| {
            tick_world.step()?;
            tick_world.frontier_step()?;
            tick_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("end_to_end_tick", end_to_end_tick);

    println!("compare=debug: cargo run --bin bench_quantale -- <N>");
    println!("compare=release: cargo run --release --bin bench_quantale -- <N>");
    println!("note=timings are synchronized CUDA wall-clock durations, not speedup claims");
    Ok(())
}
