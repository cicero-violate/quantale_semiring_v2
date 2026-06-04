//! Typed intermediate representation for data transformations.
//!
//! A sequence of `TypedIrOp`s describes a data pipeline that `FusionDispatch`
//! can compile into a fused JIT CUDA region via `ir_op_to_jit_body`.
//!
//! Compiler path:
//!   Input → TypedIR → FusionPlan → GPURegion

use crate::types::DataKind;

// ── TypedIrOp ─────────────────────────────────────────────────────────────────

/// A single typed data-transformation operation.
///
/// Each variant names its input and output device-slot(s). The IR layer is
/// intentionally scalar-body-oriented so that `ir_op_to_jit_body` can
/// synthesise a CUDA C expression directly from the op variant.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TypedIrOp {
    /// Element-wise transformation: `out[i] = body(in[i])`.
    Map {
        input: String,
        output: String,
        /// CUDA C expression using `in0[i]` as the input variable.
        body: String,
    },
    /// Element-wise predicate filter: passes values where `predicate` is true.
    Filter {
        input: String,
        output: String,
        /// CUDA C boolean expression using `in0[i]`.
        predicate: String,
    },
    /// Reduction with a scalar accumulator and element body.
    Reduce {
        input: String,
        output: String,
        init: f32,
        /// CUDA C expression that updates `acc` given `in0[i]`.
        body: String,
    },
    /// Key-based join of two slots.
    Join {
        left: String,
        right: String,
        output: String,
        /// Slot name providing join keys.
        key: String,
    },
    /// In-place sort by a named key slot.
    Sort {
        input: String,
        output: String,
        key: String,
        ascending: bool,
    },
    /// Sliding-window aggregation.
    Window {
        input: String,
        output: String,
        size: usize,
        /// CUDA C expression using `in0[i..i+size]`.
        body: String,
    },
    /// Select the top-k elements from a slot.
    TopK {
        input: String,
        output: String,
        k: usize,
    },
    /// Row-major matrix multiplication A × B → output.
    MatMul {
        a: String,
        b: String,
        output: String,
    },
    /// BFS/DFS over a CSR graph.
    GraphTraverse {
        nodes: String,
        edges: String,
        output: String,
        max_depth: u32,
    },
    /// Project `input` into an embedding space of dimension `dim`.
    Embed {
        input: String,
        output: String,
        dim: usize,
    },
    /// Assert a predicate holds for every element; writes 1/0 flags.
    Verify {
        input: String,
        /// CUDA C boolean expression using `in0[i]`.
        predicate: String,
    },
}

impl TypedIrOp {
    /// Device-slot names read by this op.
    pub fn reads(&self) -> Vec<&str> {
        match self {
            Self::Map { input, .. } | Self::Filter { input, .. } => vec![input.as_str()],
            Self::Reduce { input, .. } => vec![input.as_str()],
            Self::Join { left, right, key, .. } => {
                vec![left.as_str(), right.as_str(), key.as_str()]
            }
            Self::Sort { input, key, .. } => vec![input.as_str(), key.as_str()],
            Self::Window { input, .. } => vec![input.as_str()],
            Self::TopK { input, .. } => vec![input.as_str()],
            Self::MatMul { a, b, .. } => vec![a.as_str(), b.as_str()],
            Self::GraphTraverse { nodes, edges, .. } => vec![nodes.as_str(), edges.as_str()],
            Self::Embed { input, .. } => vec![input.as_str()],
            Self::Verify { input, .. } => vec![input.as_str()],
        }
    }

    /// Device-slot name written by this op, if any.
    pub fn writes(&self) -> Option<&str> {
        match self {
            Self::Map { output, .. }
            | Self::Filter { output, .. }
            | Self::Reduce { output, .. }
            | Self::Join { output, .. }
            | Self::Sort { output, .. }
            | Self::Window { output, .. }
            | Self::TopK { output, .. }
            | Self::MatMul { output, .. }
            | Self::GraphTraverse { output, .. }
            | Self::Embed { output, .. } => Some(output.as_str()),
            Self::Verify { .. } => None,
        }
    }

    /// `DataKind` of the output slot produced by this op.
    pub fn output_kind(&self) -> DataKind {
        match self {
            Self::Embed { .. } => DataKind::Embedding,
            Self::GraphTraverse { .. } => DataKind::Graph,
            Self::Verify { .. } => DataKind::Tensor,
            _ => DataKind::Tensor,
        }
    }
}

// ── IrPipeline ────────────────────────────────────────────────────────────────

/// A named linear sequence of `TypedIrOp`s forming a data pipeline.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IrPipeline {
    pub name: String,
    pub ops: Vec<TypedIrOp>,
}

impl IrPipeline {
    /// Slot names read by the pipeline that are not produced by an earlier op.
    pub fn external_reads(&self) -> Vec<String> {
        let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut external = Vec::new();
        for op in &self.ops {
            for read in op.reads() {
                if !produced.contains(read) {
                    let r = read.to_string();
                    if !external.contains(&r) {
                        external.push(r);
                    }
                }
            }
            if let Some(w) = op.writes() {
                produced.insert(w.to_string());
            }
        }
        external
    }

    /// Slot names written by the pipeline that are not consumed by a later op.
    pub fn external_writes(&self) -> Vec<String> {
        let all_reads: std::collections::HashSet<String> = self
            .ops
            .iter()
            .flat_map(|op| op.reads().into_iter().map(str::to_string))
            .collect();
        self.ops
            .iter()
            .filter_map(|op| op.writes())
            .map(str::to_string)
            .filter(|w| !all_reads.contains(w))
            .collect()
    }
}

// ── IR → JIT body synthesis ───────────────────────────────────────────────────

/// Synthesise a CUDA C `jit_body` expression from a single `TypedIrOp`.
///
/// The returned string is suitable for the `"jit_body"` field of a synthetic
/// operator registry entry consumed by `detect_jit_chains`.  Input slots are
/// bound in argument order: `in0[i]`, `in1[i]`, `in2[i]`.
pub fn ir_op_to_jit_body(op: &TypedIrOp) -> Result<String, String> {
    match op {
        TypedIrOp::Map { body, .. } => Ok(format!("out[i] = {body};")),
        TypedIrOp::Filter { predicate, .. } => {
            Ok(format!("out[i] = ({predicate}) ? in0[i] : 0.0f;"))
        }
        TypedIrOp::Reduce { .. } => Err(
            "Reduce cannot be lowered to a scalar element body: parallel reduction \
             requires a multi-pass warp-shuffle kernel (device helper \
             `parallel_reduce` in quantale_world.cu)"
                .into(),
        ),
        TypedIrOp::Window { body, size, .. } => {
            Ok(format!(
                "float acc = 0.0f; \
                 for (int w = 0; w < {size} && (i + w) < n; ++w) {{ acc += in0[i + w]; }} \
                 out[i] = {body};"
            ))
        }
        TypedIrOp::TopK { .. } => Err(
            "TopK cannot be lowered to a scalar element body: requires bitonic sort \
             over the full input (device helper `topk_bitonic` in quantale_world.cu)"
                .into(),
        ),
        TypedIrOp::MatMul { .. } => Err(
            "MatMul cannot be lowered to a scalar element body: requires tiled \
             shared-memory GEMM or a cuBLAS binding"
                .into(),
        ),
        TypedIrOp::Embed { dim, .. } => {
            Ok(format!("out[i] = in0[i / {dim}];"))
        }
        TypedIrOp::Verify { predicate, .. } => {
            Ok(format!("out[i] = ({predicate}) ? 1.0f : 0.0f;"))
        }
        TypedIrOp::Join { .. } => Err(
            "Join cannot be lowered to a scalar element body: requires a device \
             hash table for key-based joining"
                .into(),
        ),
        TypedIrOp::Sort { .. } => Err(
            "Sort cannot be lowered to a scalar element body: requires bitonic sort \
             or thrust::sort over the full input"
                .into(),
        ),
        TypedIrOp::GraphTraverse { .. } => Err(
            "GraphTraverse cannot be lowered to a scalar element body: requires a \
             BFS frontier kernel over a CSR adjacency structure"
                .into(),
        ),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_body_wraps_expression() {
        let op = TypedIrOp::Map {
            input: "a".into(),
            output: "b".into(),
            body: "in0[i] * 2.0f".into(),
        };
        let body = ir_op_to_jit_body(&op).unwrap();
        assert!(body.contains("out[i]"), "body={body}");
        assert!(body.contains("in0[i] * 2.0f"), "body={body}");
    }

    #[test]
    fn filter_uses_ternary() {
        let op = TypedIrOp::Filter {
            input: "x".into(),
            output: "y".into(),
            predicate: "in0[i] > 0.0f".into(),
        };
        let body = ir_op_to_jit_body(&op).unwrap();
        assert!(body.contains("?"), "body={body}");
    }

    #[test]
    fn pipeline_external_reads_excludes_internal() {
        let pipeline = IrPipeline {
            name: "test".into(),
            ops: vec![
                TypedIrOp::Map {
                    input: "a".into(),
                    output: "b".into(),
                    body: "in0[i]".into(),
                },
                TypedIrOp::Map {
                    input: "b".into(),
                    output: "c".into(),
                    body: "in0[i]".into(),
                },
            ],
        };
        let reads = pipeline.external_reads();
        assert_eq!(reads, vec!["a"]);
        let writes = pipeline.external_writes();
        assert_eq!(writes, vec!["c"]);
    }
}
