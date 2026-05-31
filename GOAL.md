# Goal

Build a CUDA-resident tensor quantale orchestrator where workflow routing is algebraic rather than imperative.

## Canonical model

```text
T ∈ R^(3 × N × N)
```

Layers:

```text
confidence/correctness: max-times
compute/time cost:      min-plus
security/safety:        max-min
```

## Non-negotiables

- Tensor state remains GPU-resident.
- Tensor edges carry confidence, cost, and safety directly.
- No scalar sidecar metadata model.
- No CPU planner.
- No hidden if/else routing tree.
- Runtime feedback updates tensor cells directly.

## Current milestone

The tensor engine, tensor compilation, tensor rule deltas, tensor frontier step, tensor tick, and tensor runtime loop are implemented.
