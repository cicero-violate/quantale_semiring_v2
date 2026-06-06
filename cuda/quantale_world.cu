// Assembled CUDA quantale world translation unit.
//
// Rust/NVRTC assembly in src/tensor.rs concatenates the same fragments with
// include_str! so generated H_f markers can be replaced before compilation.
// Keep this file as a human-readable nvcc-compatible shell.

#include "quantale/00_prelude.cuh"
#include "quantale/01_state_and_rings.cuh"
#include "quantale/02_control_flow_abi.cuh"
#include "quantale/03_tensor_core.cuh"
#include "quantale/04_exploration.cuh"
#include "quantale/05_jit_and_receipts.cuh"
#include "quantale/06_scheduler.cuh"
#include "quantale/07_failure_policy.cuh"
#include "quantale/08_learning.cuh"
#include "quantale/09_trace_replay.cuh"
#include "quantale/10_device_float_ring.cuh"
#include "quantale/11_hot_regions_dispatch.cuh"
#include "quantale/12_par_group.cuh"
