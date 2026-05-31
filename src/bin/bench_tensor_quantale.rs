use std::env;
use std::time::{Duration, Instant};

use quantale_semiring_v2::{
    ProjectionBias, TensorQuantaleWorld, default_tensor_edges_from_scalar, full_transition_edges,
};

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

fn timed<F>(mut f: F) -> Result<Duration, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = Instant::now();
    f()?;
    Ok(start.elapsed())
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
    let scalar_edges = full_transition_edges();
    let tensor_edges = default_tensor_edges_from_scalar(&scalar_edges);
    let bias = ProjectionBias::default();

    println!("quantale_semiring_v2 tensor benchmark");
    println!("profile={profile}");
    println!("iterations={iterations}");
    println!("edge_count={}", tensor_edges.len());
    println!("layers=3 confidence=max-times cost=min-plus safety=max-min");

    let mut warmup = TensorQuantaleWorld::from_tensor_edges(&tensor_edges)?;
    warmup.close()?;
    warmup.project(bias)?;
    warmup.synchronize()?;

    let closure = bench(iterations, || {
        let mut world = TensorQuantaleWorld::from_tensor_edges(&tensor_edges)?;
        world.synchronize()?;
        timed(|| {
            world.close()?;
            world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("tensor_closure", closure);

    let mut projection_world = TensorQuantaleWorld::from_tensor_edges(&tensor_edges)?;
    projection_world.close()?;
    projection_world.synchronize()?;
    let projection = bench(iterations, || {
        timed(|| {
            projection_world.project(bias)?;
            projection_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("tensor_projection", projection);

    let update = bench(iterations, || {
        timed(|| {
            projection_world.decay(0.99)?;
            projection_world.synchronize()?;
            Ok(())
        })
    })?;
    print_sample("tensor_decay", update);

    println!("compare=debug: cargo run --bin bench_tensor_quantale -- <N>");
    println!("compare=release: cargo run --release --bin bench_tensor_quantale -- <N>");
    println!("note=timings are synchronized CUDA wall-clock durations, not speedup claims");
    Ok(())
}
