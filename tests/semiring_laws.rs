//! Unit tests for the three-layer quantale semiring algebraic laws (Phase 4).
//!
//! Each test verifies a law on the CPU tensor representation.  These are
//! pure-Rust tests — no CUDA device required.
//!
//! Laws verified
//! -------------
//! Identity (invariant 14):  I ⊗ W = W,  W ⊗ I = W
//! Bottom (invariant 15):    ⊥ ⊗ x = ⊥,  x ⊗ ⊥ = ⊥,  x ⊕ ⊥ = x

use quantale_semiring_v2::{
    COST_INFINITY, LAYER_CONFIDENCE, LAYER_COST, LAYER_SAFETY, TENSOR_LEN, TENSOR_NODE_COUNT,
    TensorEdge, tensor_idx,
};

// ── helpers ───────────────────────────────────────────────────────────────────

const N: i32 = TENSOR_NODE_COUNT as i32;

/// Build the identity matrix tensor (no edges).
fn identity_tensor() -> Vec<f32> {
    let mut t = vec![0.0f32; TENSOR_LEN];
    for i in 0..N {
        t[tensor_idx(LAYER_CONFIDENCE, i, i)] = 1.0;
        t[tensor_idx(LAYER_COST, i, i)] = 0.0;
        t[tensor_idx(LAYER_SAFETY, i, i)] = 1.0;
    }
    // Off-diagonal cost entries are ∞ in the identity
    for i in 0..N {
        for j in 0..N {
            if i != j {
                t[tensor_idx(LAYER_COST, i, j)] = COST_INFINITY;
            }
        }
    }
    t
}

/// Embed edges into a tensor starting from `base`.
fn embed_edges(base: &mut [f32], edges: &[TensorEdge]) {
    for e in edges {
        let ci = tensor_idx(LAYER_CONFIDENCE, e.src, e.dst);
        let ei = tensor_idx(LAYER_COST, e.src, e.dst);
        let si = tensor_idx(LAYER_SAFETY, e.src, e.dst);
        base[ci] = base[ci].max(e.confidence);
        base[ei] = base[ei].min(e.cost);
        base[si] = base[si].max(e.safety);
    }
}

/// CPU max-times / min-plus / max-min matrix multiply for the three layers.
fn tensor_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; TENSOR_LEN];
    for i in 0..N {
        for j in 0..N {
            // confidence: max-times  (join = max, compose = multiply)
            let mut conf = 0.0f32;
            for k in 0..N {
                conf = conf.max(
                    a[tensor_idx(LAYER_CONFIDENCE, i, k)] * b[tensor_idx(LAYER_CONFIDENCE, k, j)],
                );
            }
            c[tensor_idx(LAYER_CONFIDENCE, i, j)] = conf;

            // cost: min-plus  (join = min, compose = add)
            let mut cost = COST_INFINITY;
            for k in 0..N {
                let ak = a[tensor_idx(LAYER_COST, i, k)];
                let bk = b[tensor_idx(LAYER_COST, k, j)];
                let sum = if ak >= COST_INFINITY || bk >= COST_INFINITY {
                    COST_INFINITY
                } else {
                    ak + bk
                };
                cost = cost.min(sum);
            }
            c[tensor_idx(LAYER_COST, i, j)] = cost;

            // safety: max-min  (join = max, compose = min)
            let mut safety = 0.0f32;
            for k in 0..N {
                safety = safety
                    .max(a[tensor_idx(LAYER_SAFETY, i, k)].min(b[tensor_idx(LAYER_SAFETY, k, j)]));
            }
            c[tensor_idx(LAYER_SAFETY, i, j)] = safety;
        }
    }
    c
}

fn tensors_approx_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// A small deterministic set of test edges.
fn test_edges() -> Vec<TensorEdge> {
    vec![
        TensorEdge::new(0, 1, 0.9, 0.5, 0.8),
        TensorEdge::new(1, 2, 0.8, 1.0, 0.7),
        TensorEdge::new(2, 3, 0.7, 1.5, 0.9),
        TensorEdge::new(3, 0, 0.6, 2.0, 0.6),
    ]
}

fn world_tensor(edges: &[TensorEdge]) -> Vec<f32> {
    let mut t = identity_tensor();
    embed_edges(&mut t, edges);
    t
}

// ── Invariant 14: identity laws ───────────────────────────────────────────────

#[test]
fn semiring_identity_left() {
    let id = identity_tensor();
    let w = world_tensor(&test_edges());
    let result = tensor_mul(&id, &w);
    assert!(
        tensors_approx_equal(&result, &w),
        "I ⊗ W ≠ W: identity left law violated"
    );
}

#[test]
fn semiring_identity_right() {
    let id = identity_tensor();
    let w = world_tensor(&test_edges());
    let result = tensor_mul(&w, &id);
    assert!(
        tensors_approx_equal(&result, &w),
        "W ⊗ I ≠ W: identity right law violated"
    );
}

#[test]
fn semiring_identity_is_idempotent() {
    let id = identity_tensor();
    let result = tensor_mul(&id, &id);
    assert!(
        tensors_approx_equal(&result, &id),
        "I ⊗ I ≠ I: identity matrix is not idempotent"
    );
}

// ── Invariant 15: bottom absorption and join ──────────────────────────────────

/// Build an all-⊥ (bottom) tensor.
fn bottom_tensor() -> Vec<f32> {
    // Confidence layer: all 0.0 (bottom for max-times)
    // Cost layer: all COST_INFINITY (bottom for min-plus)
    // Safety layer: all 0.0 (bottom for max-min)
    let mut t = vec![0.0f32; TENSOR_LEN];
    for i in 0..N {
        for j in 0..N {
            t[tensor_idx(LAYER_COST, i, j)] = COST_INFINITY;
        }
    }
    t
}

#[test]
fn semiring_bottom_absorb_left() {
    let bot = bottom_tensor();
    let w = world_tensor(&test_edges());
    let result = tensor_mul(&bot, &w);
    // ⊥ ⊗ W should equal ⊥: all confidence entries = 0, all cost entries = ∞
    for i in 0..N {
        for j in 0..N {
            let conf = result[tensor_idx(LAYER_CONFIDENCE, i, j)];
            let cost = result[tensor_idx(LAYER_COST, i, j)];
            let safety = result[tensor_idx(LAYER_SAFETY, i, j)];
            assert!(
                conf == 0.0,
                "⊥⊗W: confidence[{i},{j}]={conf} ≠ 0 (⊥ absorb left broken)"
            );
            assert!(
                cost >= COST_INFINITY,
                "⊥⊗W: cost[{i},{j}]={cost} < ∞ (⊥ absorb left broken)"
            );
            assert!(
                safety == 0.0,
                "⊥⊗W: safety[{i},{j}]={safety} ≠ 0 (⊥ absorb left broken)"
            );
        }
    }
}

#[test]
fn semiring_bottom_absorb_right() {
    let bot = bottom_tensor();
    let w = world_tensor(&test_edges());
    let result = tensor_mul(&w, &bot);
    for i in 0..N {
        for j in 0..N {
            let conf = result[tensor_idx(LAYER_CONFIDENCE, i, j)];
            let cost = result[tensor_idx(LAYER_COST, i, j)];
            let safety = result[tensor_idx(LAYER_SAFETY, i, j)];
            assert!(
                conf == 0.0,
                "W⊗⊥: confidence[{i},{j}]={conf} ≠ 0 (⊥ absorb right broken)"
            );
            assert!(
                cost >= COST_INFINITY,
                "W⊗⊥: cost[{i},{j}]={cost} < ∞ (⊥ absorb right broken)"
            );
            assert!(
                safety == 0.0,
                "W⊗⊥: safety[{i},{j}]={safety} ≠ 0 (⊥ absorb right broken)"
            );
        }
    }
}

#[test]
fn semiring_bottom_join_identity() {
    // x ⊕ ⊥ = x  (join = pointwise max/min appropriate per layer)
    let w = world_tensor(&test_edges());
    // Compute W ⊕ ⊥ = pointwise join
    let bot = bottom_tensor();
    let mut result = vec![0.0f32; TENSOR_LEN];
    for i in 0..N {
        for j in 0..N {
            // confidence: max
            result[tensor_idx(LAYER_CONFIDENCE, i, j)] =
                w[tensor_idx(LAYER_CONFIDENCE, i, j)].max(bot[tensor_idx(LAYER_CONFIDENCE, i, j)]);
            // cost: min (⊕ for min-plus is min)
            result[tensor_idx(LAYER_COST, i, j)] =
                w[tensor_idx(LAYER_COST, i, j)].min(bot[tensor_idx(LAYER_COST, i, j)]);
            // safety: max
            result[tensor_idx(LAYER_SAFETY, i, j)] =
                w[tensor_idx(LAYER_SAFETY, i, j)].max(bot[tensor_idx(LAYER_SAFETY, i, j)]);
        }
    }
    assert!(
        tensors_approx_equal(&result, &w),
        "W ⊕ ⊥ ≠ W: bottom join identity law violated"
    );
}
