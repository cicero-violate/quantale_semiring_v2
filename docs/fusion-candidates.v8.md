# Fusion Candidates v8

## Current Enabled Region

```text
Analysis::Return1
  -> Analysis::Volatility
  -> Analysis::SignalScore
```

Evidence:

```text
assets/topology.fusion.json
tests/fusion_equivalence.rs
src/bin/bench_fusion_region.rs
state/profiles/repeated_compute_cached.json
```

## Decision

No additional fusion regions are enabled in Plan v8.

The only repeated CUDA-safe tensor sequence with current evidence is the
existing market-analysis region.  Boundary nodes, governance gates, host I/O,
locks, mutation operators, and Python operators remain hard barriers.

## Next Gate

Add a new region only after a real trace profile shows the same ordered
CUDA-safe kernel sequence recurring in at least two loop epochs and the
benchmark shows material launch or compute overhead.
