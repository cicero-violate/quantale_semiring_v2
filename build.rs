use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=assets/operators.json");
    println!("cargo:rerun-if-changed=assets/topology.source.json");
    println!("cargo:rerun-if-changed=assets/topology.generated.json");
    println!("cargo:rerun-if-changed=assets/regions.hot.json");
    println!("cargo:rerun-if-changed=assets/topology.hot.json");
    println!("cargo:rerun-if-changed=assets/topology.control.json");
    println!("cargo:rerun-if-changed=assets/fusion_hf.generated.json");
    println!("cargo:rerun-if-changed=assets/fusion_hf.stubs.cu");
    println!("cargo:rerun-if-changed=assets/abstract_device.generated.json");
    println!("cargo:rerun-if-changed=cuda/quantale_world.cu");
    println!("cargo:rerun-if-changed=cuda/quantale/00_prelude.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/01_state_and_rings.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/02_control_flow_abi.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/03_tensor_core.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/04_exploration.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/05_jit_and_receipts.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/06_scheduler.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/07_failure_policy.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/08_learning.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/09_trace_replay.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/10_device_float_ring.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/11_hot_regions_dispatch.cuh");
    println!("cargo:rerun-if-changed=cuda/quantale/12_par_group.cuh");

    let topology_path = if Path::new("assets/topology.generated.json").exists() {
        "assets/topology.generated.json"
    } else {
        "assets/topology.source.json"
    };
    let topology = fs::read_to_string(topology_path).expect("read topology asset");
    let node_count = derive_node_count(&topology)
        .unwrap_or_else(|| panic!("derive node count from {topology_path}"));

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    fs::write(
        Path::new(&out_dir).join("topology_constants.rs"),
        format!(
            "pub const TENSOR_NODE_COUNT: usize = {node_count};\n\
             pub const CUDA_TENSOR_NODE_COUNT_DEFINE: &str = \"#define N {node_count}\\n\";\n"
        ),
    )
    .expect("write generated topology constants");
}

fn derive_node_count(input: &str) -> Option<usize> {
    let mut max_id = None;
    let mut search = input;
    while let Some(pos) = search.find("\"id\"") {
        search = &search[pos + 4..];
        let colon = search.find(':')?;
        search = &search[colon + 1..];
        let trimmed = search.trim_start();
        let digits: String = trimmed
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            continue;
        }
        let id = digits.parse::<usize>().ok()?;
        max_id = Some(max_id.map_or(id, |current: usize| current.max(id)));
    }
    max_id.map(|id| id + 1)
}
