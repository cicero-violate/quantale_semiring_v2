use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/trading_execution_kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");
    println!("cargo:rerun-if-env-changed=CXX");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by Cargo"));
    let ptx_out = out_dir.join("trading_execution_kernels.ptx");
    let nvcc = find_nvcc();

    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "cuda/trading_execution_kernels.cu",
        "-ptx",
        "-o",
        ptx_out.to_str().expect("OUT_DIR path is utf-8"),
        "-std=c++17",
        "--use_fast_math",
        "-Xcompiler",
        "-fPIC",
    ]);
    if let Ok(host) = env::var("CUDAHOSTCXX").or_else(|_| env::var("CXX")) {
        cmd.args(["-ccbin", &host]);
    } else if Path::new("/usr/bin/g++-12").exists() {
        cmd.args(["-ccbin", "/usr/bin/g++-12"]);
    } else if Path::new("/usr/bin/g++-11").exists() {
        cmd.args(["-ccbin", "/usr/bin/g++-11"]);
    }

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("nvcc failed to start for trading_execution_kernels.cu: {e}"));
    if !status.success() {
        panic!("nvcc failed to compile cuda/trading_execution_kernels.cu");
    }

    println!("cargo:warning=nvcc compiled {}", ptx_out.display());
}

fn find_nvcc() -> PathBuf {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let candidate = PathBuf::from(&cuda_home).join("bin/nvcc");
        if candidate.exists() {
            return candidate;
        }
    }
    // Try common install paths before falling back to PATH
    for candidate in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"] {
        if Path::new(candidate).exists() {
            return PathBuf::from(candidate);
        }
    }
    PathBuf::from("nvcc")
}
