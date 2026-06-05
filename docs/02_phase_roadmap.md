# 02 — Phase Roadmap

## Phase 1 — Build the World Graph Core

```text
Graph search + dynamic programming + constraints + linear models
```

Goal:

```text
Create an executable topology/influence graph before training heavy models.
```

| Priority | Algorithm | Direct use |
|---:|---|---|
| 1 | BFS / DFS | Traverse topology graph. |
| 2 | Uniform Cost / Dijkstra-like routing | Find lowest-cost execution path. |
| 3 | A* / Beam Search | Guided planning and graph search. |
| 4 | Bellman-Ford | Weighted influence propagation with negative/friction edges. |
| 5 | Floyd-Warshall | All-pairs influence map. |
| 6 | Viterbi | Best hidden state path. |
| 7 | Constraint Propagation | Validate legal/safe execution. |
| 8 | Backtracking Search | Repair invalid plans. |
| 9 | Ridge / Lasso / Elastic Net | Learn edge weights. |
| 10 | PCA | Reduce macro variables. |

## Phase 2 — Detect Structure in Data

```text
Clustering + anomaly detection + association rules + time series
```

Goal:

```text
Detect regimes, shocks, recurring patterns, and lagged influence.
```

| Priority | Algorithm | Direct use |
|---:|---|---|
| 11 | K-Means / K-Means++ | Cluster similar states. |
| 12 | DBSCAN | Detect irregular clusters and noise. |
| 13 | Gaussian Mixture Models | Soft regimes. |
| 14 | Z-Score | Cheap anomaly detection. |
| 15 | Isolation Forest | Shock detection. |
| 16 | Apriori / FP-Growth | Repeated causal/event patterns. |
| 17 | VAR | Multi-variable macro influence. |
| 18 | ARIMAX / SARIMAX | Time series with external variables. |
| 19 | DTW | Compare event sequences. |
| 20 | Levenshtein / LCS | Compare traces/plans. |

## Phase 3 — Add Prediction + Scoring

```text
Trees + boosting + Bayesian uncertainty
```

Goal:

```text
Add explainable prediction and probabilistic confidence to graph edges/states.
```

| Priority | Algorithm | Direct use |
|---:|---|---|
| 21 | Decision Tree | Explainable decision rule. |
| 22 | Random Forest | Robust tabular predictor. |
| 23 | Extra Trees | Fast ensemble baseline. |
| 24 | XGBoost / LightGBM / CatBoost | Strong structured-data prediction. |
| 25 | Bayesian Regression | Uncertainty on influence. |
| 26 | Bayesian Belief Network | Probabilistic causal graph. |
| 27 | Gaussian Process | Expensive uncertainty model. |

## Phase 4 — Add Semantic Translation

```text
Embeddings + attention + LLM translation
```

Goal:

```text
Convert documents, news, logs, and text into graph nodes and edges.
```

| Priority | Algorithm | Direct use |
|---:|---|---|
| 28 | FastText / Word2Vec | Cheap semantic lookup. |
| 29 | BERT embeddings | Text -> vector -> graph node. |
| 30 | LSA / NMF / LDA | Topic extraction from documents/news. |
| 31 | Self-Attention | Conceptual routing model. |
| 32 | Transformer | Use pretrained only. |
| 33 | GPT / T5 / BART / RoBERTa | Use as external parser, not training target. |

## Phase 5 — Add Reinforcement Learning Later

```text
MDP + Q-Learning + PPO
```

Goal:

```text
Learn action selection only after the world graph can simulate consequences.
```

| Priority | Algorithm | Direct use |
|---:|---|---|
| 34 | MDP | Formal state/action/reward model. |
| 35 | Bellman Equation | Value propagation. |
| 36 | Q-Learning | Small discrete policy learning. |
| 37 | DQN | Only after stable simulator. |
| 38 | Actor-Critic | Later continuous control. |
| 39 | PPO | Later, once simulator and reward are reliable. |
| 40 | REINFORCE | Mostly educational baseline. |

## Phase Dependency Graph

```text
Phase 1 graph core
  -> Phase 2 structure detection
    -> Phase 3 prediction/scoring
      -> Phase 4 semantic translation
        -> Phase 5 reinforcement learning
```

RL comes last because it needs a working simulator, state model, reward model, and action space.
