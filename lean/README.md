# Lean / cLean proof boundary

This directory is the proof boundary for `quantale_semiring_v2`.

Current file:

```text
QuantaleSemiringV2/Spec.lean
```

The implementation target is the current unified node universe:

```text
Node = StateNode ⊔ ControlNode ⊔ EventNode
NODE_COUNT = 44
Scalar compatibility Q = ([0,1], max, ×, 0, 1)
Tensor runtime T ∈ R^(3 × 44 × 44)
```

The intended Lean/cLean bridge is:

```text
quantale_supremum_assign          ↔ scalar matrixJoin
quantale_tensor_assign            ↔ scalar matrixMul over max-times Q
quantale_least_fixed_point        ↔ scalar closureSpec

tensor_quantale_closure           ↔ IsTensorClosure
tensor_quantale_project           ↔ BlendedProjectionSpec
tensor_quantale_frontier_step     ↔ TensorFrontierSpec
tensor_quantale_tick              ↔ IsTensorClosure + TensorFrontierSpec
build_tensor_policy_edges         ↔ tensor policyEdgeSpec
build_tensor_receipt_edges        ↔ tensor receiptEdgeSpec
witness[3 × N × N]                ↔ per-layer path witness tensor W_L
```

Rust/CPU is not modeled as owning quantale state. CPU appears only as lattice-edge ingress, compact-report egress, path decoding from `witness_matrix`, and external side-effect execution.

`cLean` integration should attach the CUDA kernels in `../cuda/quantale_world.cu` to the contract above. Only update Lean artifacts when the actual Lean/cLean toolchain is available. Do not add placeholder proofs or fake scaffolding. No local `lean`/`lake`/`cLean` binary is installed in this workspace, so this proof artifact has not been typechecked here.


## Tensor layers

```text
Layer 0: confidence/correctness  max-times  join=max  compose=×
Layer 1: compute/time cost       min-plus   join=min  compose=+
Layer 2: security/safety         max-min    join=max  compose=min
```

`Spec.lean` now names the tensor closure and projection boundary. The file remains a proof-boundary artifact; it has not been typechecked in this workspace because no local Lean/lake/cLean binary is installed.
