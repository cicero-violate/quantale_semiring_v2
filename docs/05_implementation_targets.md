# 05 — Implementation Targets

This document maps algorithm families to concrete implementation targets in the quantale/semiring system.

## Target Stack

```text
Data ingestion
  -> entity resolution
  -> graph node creation
  -> edge scoring
  -> constraint validation
  -> propagation/simulation
  -> explanation receipts
```

## Build Target Table

| Layer                  | Algorithm                     | Build target                                   |
|------------------------+-------------------------------+------------------------------------------------|
| Topology traversal     | BFS / DFS / A* / Beam Search  | Find possible execution paths.                 |
| Influence propagation  | Bellman-Ford / Floyd-Warshall | Compute macro impact paths.                    |
| Safety/invariants      | Constraint Propagation        | Block invalid transitions.                     |
| Plan repair            | Backtracking Search           | Find alternative valid graph paths.            |
| Edge learning          | Ridge / Lasso / SGD           | Learn/update influence weights.                |
| State compression      | PCA                           | Reduce noisy macro variables.                  |
| Regime detection       | K-Means / DBSCAN / GMM        | Detect market/world states.                    |
| Shock detection        | Z-Score / Isolation Forest    | Detect unusual events.                         |
| Pattern mining         | Apriori / FP-Growth           | Find repeated event chains.                    |
| Semantic bridge        | BERT/FastText embeddings      | Map text/news/docs into graph nodes.           |
| Hidden-state inference | HMM / Viterbi                 | Infer latent regimes or workflows.             |
| Tabular scoring        | Decision Tree / Random Forest | Predict and explain edge/state outcomes.       |
| Uncertainty            | Bayesian Regression / BBN     | Estimate confidence in edges and predictions.  |
| Later policy           | MDP / Q-Learning / PPO        | Learn action selection after simulator exists. |

## Minimal Rust Module Layout

```text
src/
  graph/
    traversal.rs          # BFS, DFS, A*, beam search
    shortest_path.rs      # Bellman-Ford, Floyd-Warshall
    constraints.rs        # constraint propagation, backtracking
  influence/
    edge.rs               # edge schema and tensor layers
    scoring.rs            # lag corr, mutual info, rule/event scores
    propagation.rs        # X[t+1] update logic
  learning/
    linear.rs             # ridge/lasso/elastic-net baseline
    pca.rs                # dimensionality reduction
    clustering.rs         # k-means/dbscan/gmm wrappers
    anomaly.rs            # z-score/isolation forest
  semantic/
    embeddings.rs         # pretrained embeddings bridge
    topics.rs             # lda/lsa/nmf topic bridge
  simulation/
    state.rs              # world state representation
    rollout.rs            # consequence simulation
    receipts.rs           # explanation traces
```

## Minimal Data Structures

### Node

```rust
pub struct NodeId(pub u64);

pub struct MacroNode {
    pub id: NodeId,
    pub name: String,
    pub kind: String,
    pub value: f64,
    pub timestamp_ms: i64,
}
```

### Edge

```rust
pub struct InfluenceEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub sign: f64,
    pub weight: f64,
    pub delay_steps: usize,
    pub confidence: f64,
    pub friction: f64,
    pub risk: f64,
}
```

### Tensor Edge View

```rust
pub struct EdgeTensor {
    pub layers: usize,
    pub nodes: usize,
    pub data: Vec<f32>, // shape: [layers, nodes, nodes]
}
```

## First Implementation Order

1. Node/edge schema
2. BFS/DFS traversal
3. Weighted propagation
4. Constraint propagation
5. Bellman-Ford paths
6. Floyd-Warshall all-pairs impact
7. Ridge/SGD edge update
8. PCA compression
9. Z-score anomaly detection
10. Receipts/explanation traces

## Definition of Done for Phase 1

Phase 1 is done when the system can:

- Load nodes and edges.
- Traverse the graph.
- Score possible paths.
- Reject invalid transitions.
- Propagate one state update.
- Produce a receipt explaining why a node changed.
- Save/load graph state.
