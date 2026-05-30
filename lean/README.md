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
Q = ([0,1], max, ×, 0, 1)
```

The intended Lean/cLean bridge is:

```text
quantale_join_assign        ↔ matrixJoin
quantale_mul_assign         ↔ matrixMul over max-times Q
quantale_closure_assign     ↔ closureSpec
quantale_step               ↔ closure + active frontier update
quantale_decide_path        ↔ projectionSpec + first-hop witness
build_policy_edges          ↔ policyEdgeSpec
build_receipt_edges         ↔ receiptEdgeSpec
next_hop[N × N]             ↔ path witness matrix W
```

Rust/CPU is not modeled as owning quantale state. CPU appears only as transition-edge ingress, compact-report egress, path decoding from `next_hop`, and external side-effect execution.

`cLean` integration should attach the CUDA kernels in `../cuda/quantale_world.cu` to the contract above. Only update Lean artifacts when the actual Lean/cLean toolchain is available. Do not add placeholder proofs or fake scaffolding. No local `lean`/`lake`/`cLean` binary is installed in this workspace, so this proof artifact has not been typechecked here.
