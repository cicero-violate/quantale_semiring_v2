pub const TENSOR_LAYER_COUNT: usize = 3;
include!(concat!(env!("OUT_DIR"), "/topology_constants.rs"));
pub const MATRIX_LEN: usize = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT;
pub const TENSOR_LEN: usize = TENSOR_LAYER_COUNT * MATRIX_LEN;
pub const COST_INFINITY: f32 = 1.0e20;

pub const LAYER_CONFIDENCE: i32 = 0;
pub const LAYER_COST: i32 = 1;
pub const LAYER_SAFETY: i32 = 2;

// Phase-2 orchestration step status codes (mirror ORCH_* defines in .cu).
pub const ORCH_CONTINUE: i32 = 0;
pub const ORCH_WAIT_EXTERNAL: i32 = 1;
pub const ORCH_HALTED: i32 = 2;
pub const ORCH_ERROR: i32 = 3;

// Phase-2 dispatch kind codes (mirror DISPATCH_KIND_* defines in .cu).
pub const DISPATCH_KIND_NONE: i32 = 0;
pub const DISPATCH_KIND_HF_DEVICE: i32 = 1;
pub const DISPATCH_KIND_HOST_FALLBACK: i32 = 2;
pub const DISPATCH_KIND_FUSION_ENTRY: i32 = 3;
pub const DISPATCH_KIND_ABSTRACT_DEVICE: i32 = 4;
pub const DISPATCH_KIND_EXTERNAL_PROCESS: i32 = 5;
pub const DISPATCH_KIND_EXTERNAL_IO: i32 = 6;
pub const DISPATCH_KIND_UNSUPPORTED: i32 = 7;

// Phase-4 control-flow operation codes (mirror CONTROL_OP_* in .cu).
pub const CONTROL_OP_SEQ: i32 = 0;
pub const CONTROL_OP_PAR: i32 = 1;
pub const CONTROL_OP_CHOICE: i32 = 2;
pub const CONTROL_OP_STAR_BOUNDED: i32 = 3;
pub const CONTROL_OP_GATE: i32 = 4;
pub const CONTROL_OP_HALT_OP: i32 = 5;

// GPU-native scheduler decision kind codes (mirror CONTROL_* in .cu).
pub const CONTROL_NONE: i32 = 0;
pub const CONTROL_SEQ_READY: i32 = 1;
pub const CONTROL_PAR_READY: i32 = 2;
pub const CONTROL_CHOICE_READY: i32 = 3;
pub const CONTROL_STAR_BODY_READY: i32 = 4;
pub const CONTROL_STAR_EXIT_READY: i32 = 5;
pub const CONTROL_HALT_READY: i32 = 6;
pub const CONTROL_BLOCKED: i32 = 7;

// Block-reason codes set on OrchestrationState (mirror ORCH_BLOCK_REASON_* in .cu).
pub const ORCH_BLOCK_REASON_NONE: i32 = 0;
pub const ORCH_BLOCK_REASON_NO_READY_NODE: i32 = 1;
pub const ORCH_BLOCK_REASON_STAR_EXHAUSTED: i32 = 2;
pub const ORCH_BLOCK_REASON_UNSUPPORTED: i32 = 3;
pub const ORCH_BLOCK_REASON_ALL_CONSUMED: i32 = 4;

/// Maximum number of control edges in a loaded control table.
pub const MAX_CONTROL_EDGES: usize = 1024;

/// Kernel name for zeroing the per-edge star counter buffer.
pub const STAR_COUNTERS_INIT_KERNEL: &str = "star_counters_init";

// Phase-5 failure classification codes (mirror FAILURE_CLASS_* in .cu).
pub const FAILURE_CLASS_SPAWN_FAILURE: i32 = 0;
pub const FAILURE_CLASS_TIMEOUT: i32 = 1;
pub const FAILURE_CLASS_SAFETY: i32 = 2;
pub const FAILURE_CLASS_CONTRACT: i32 = 3;
pub const FAILURE_CLASS_GPU_ERROR: i32 = 4;
pub const FAILURE_CLASS_UNKNOWN: i32 = 5;

// Phase-5 failure action codes (mirror FAILURE_ACTION_* in .cu).
pub const FAILURE_ACTION_RETRY: i32 = 0;
pub const FAILURE_ACTION_BLOCK: i32 = 1;
pub const FAILURE_ACTION_ROLLBACK: i32 = 2;
pub const FAILURE_ACTION_HALT: i32 = 3;
pub const FAILURE_ACTION_EXTERNAL_REPAIR: i32 = 4;

// Phase-5 repair dispatch kind (extends DISPATCH_KIND_* in .cu).
pub const DISPATCH_KIND_REPAIR: i32 = 8;

// Phase-6 learned-delta ring capacity (mirrors LEARNED_DELTA_RING_SIZE in .cu).
pub const LEARNED_DELTA_RING_SIZE: usize = 256;

// Phase-8 trace ring capacity and event kind codes (mirror ORCH_TRACE_RING_SIZE
// and ORCH_EVENT_* defines in .cu).
pub const ORCH_TRACE_RING_SIZE: usize = 512;
pub const ORCH_EVENT_STEP_COMMITTED: i32 = 0;
pub const ORCH_EVENT_WAIT_EXTERNAL: i32 = 1;
pub const ORCH_EVENT_HALTED: i32 = 2;
pub const ORCH_EVENT_BLOCKED: i32 = 3;
pub const ORCH_EVENT_RECEIPT_DRAINED: i32 = 4;

pub const MAX_PAR_GROUP_SIZE: usize = 8;

pub const DEVICE_RECEIPT_RING_SIZE: usize = 256;
pub const DEVICE_COMMAND_RING_SIZE: usize = 64;
pub const DEVICE_RECEIPT_EXT_RING_SIZE: usize = 256;
pub const GPU_HOT_REGION_COUNT: i32 = 8;
pub const PAR_DISPATCH_NONE: i32 = 0;
pub const PAR_DISPATCH_HF_DEVICE: i32 = 1;
pub const PAR_DISPATCH_HOST_FALLBACK: i32 = 2;
pub const PAR_DISPATCH_FUSION_ENTRY: i32 = 3;
pub const PAR_DISPATCH_ABSTRACT_DEVICE: i32 = 4;

pub const EXPLORATION_MAX_TOKENS: usize = TENSOR_NODE_COUNT * TENSOR_NODE_COUNT;
pub const EXPLORATION_MAX_SELECTED: usize = TENSOR_NODE_COUNT;
