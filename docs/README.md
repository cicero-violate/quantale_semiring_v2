# Quantale Semiring v2 — Algorithm Canon

This documentation set organizes the AI algorithms previously listed into a practical implementation canon for the neurosymbolic quantale/semiring system.

The ranking is not a general-purpose AI ranking. It is optimized for the current project constraints:

- Rust-first implementation
- Topology graph / influence graph architecture
- Tensor-valued edge semantics
- Small local data
- GTX 1050-class GPU or similarly constrained compute
- Need for explainable macro/world-structure mapping
- No large training budget
- Preference for executable graph logic over black-box model training

## Documents

| File                           | Purpose                                                                  |
|--------------------------------+--------------------------------------------------------------------------|
| `01_leverage_ranking.md`       | Full ranked list of algorithm families by leverage for this project.     |
| `02_phase_roadmap.md`          | Phased implementation roadmap from graph core to RL.                     |
| `03_algorithm_catalog.md`      | Catalog of the algorithm families and individual algorithms.             |
| `04_math_formulas.md`          | Core formulas for leverage scoring, influence graphs, and edge learning. |
| `05_implementation_targets.md` | Concrete build targets mapped to algorithms.                             |
| `06_low_leverage_defer.md`     | Algorithms to defer and why.                                             |
| `ongoing_projects.md`          | Active project plans, including internet-as-data-model retrieval.        |
| `algorithm_reference.csv`      | Machine-readable ranked reference table.                                 |

## Canonical Priority

```text
1. Graph search
2. Dynamic programming
3. Constraint solving
4. Linear / convex optimization
5. Classical machine learning
6. Embeddings
7. Reinforcement learning
8. Deep vision
9. GANs
```

## Core Principle

```text
Highest leverage = maximum world-structure insight per unit of code, data, compute, and maintenance.
```

For this project, the highest-value path is to build an executable world model first:

```text
raw data -> graph nodes -> weighted delayed edges -> constraints -> simulation -> decisions
```
