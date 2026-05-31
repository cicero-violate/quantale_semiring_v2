# Usage

## Requirements

- Rust
- CUDA with NVRTC support
- Compatible NVIDIA GPU

## Build

```bash
cargo check
cargo test
```

## Run benchmark

```bash
cargo run --bin bench_quantale -- 100
cargo run --release --bin bench_quantale -- 100
```

The benchmark measures:

```text
closure
projection
frontier_step
end_to_end_tick
```

## Run orchestrator

```bash
cargo run
```

The runtime:

1. Creates a CUDA world.
2. Loads topology edges.
3. Seeds ingress candidates.
4. Computes closure and projection.
5. Executes the selected operator.
6. Converts process results into receipt edges.
7. Reinjects feedback into the graph.
8. Logs activity into quantale.tlog.

## Operators

Operators are configured in:

```text
assets/operators.json
```

Supported input modes:

```text
stdin_mode = json
stdin_source = field name
```

## Search evidence

External candidates are transformed into graph updates through:

```text
DomainCandidate
→ score_candidates
→ select_top_k
→ build_search_edges
→ M := M ∨ ΔM
```

The system does not implement retrieval or database search internally.

## Transaction log

Runtime records are written as JSONL to:

```text
quantale.tlog
```

Record types:

```text
Decision
CudaReport
Receipt
LatticeEdges
AgentStep
```

## Formal model

Lean specification:

```text
lean/QuantaleSemiringV2/Spec.lean
```
