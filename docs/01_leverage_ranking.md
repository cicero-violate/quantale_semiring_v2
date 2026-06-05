# 01 — Leverage Ranking

## Variables

```math
 a = \text{algorithm}
```

```math
 L(a)=\text{leverage score}
```

```math
 I=\text{impact on system}
```

```math
 F=\text{fit with Rust + topology graph + tensor quantale}
```

```math
 D=\text{data efficiency}
```

```math
 G=\text{GPU/parallel fit}
```

```math
 C=\text{implementation cost}
```

```math
 M=\text{maintenance cost}
```

```math
 B=\text{big-training-budget penalty}
```

## Leverage Formula

```math
L(a)=\frac{I \cdot F \cdot D \cdot G}{C \cdot M \cdot B}
```

One-line meaning:

```text
Highest leverage = high system impact, low data cost, easy integration, and reusable graph/tensor execution.
```

## Current Resource Assumption

```math
R_{you}=\{\text{Rust},\text{topology graph},\text{tensor edges},\text{GTX 1050-class GPU},\text{small data},\text{LLM-assisted coding},\text{macro influence goal}\}
```

```text
You need algorithms that compile into graph/tensor logic before heavy model training.
```

## Ranked Families

| Tier | Rank | Algorithm family                                                                               |            Leverage | Use first?      | Reason                                                                                                           |
|------+------+------------------------------------------------------------------------------------------------+---------------------+-----------------+------------------------------------------------------------------------------------------------------------------|
| S    |    1 | Graph / search algorithms: BFS, DFS, Uniform Cost, Bidirectional Search, A*, IDA*, Beam Search |                10.0 | Yes             | Directly maps to topology traversal, planning, dependency search, and execution routing.                         |
| S    |    2 | Dynamic programming: Bellman-Ford, Floyd-Warshall, Viterbi, DTW, LCS, Levenshtein              |                10.0 | Yes             | Best fit for state transitions, shortest paths, sequence alignment, trace comparison, and macro propagation.     |
| S    |    3 | Constraint algorithms: Constraint Propagation, Backtracking Search                             |                10.0 | Yes             | Perfect for safety gates, invariant checks, execution permissions, resource limits, and valid state transitions. |
| S    |    4 | Linear optimization: Simplex, Dual Simplex, Conjugate Gradient                                 |                 9.5 | Yes             | Maps cleanly to resource allocation, scheduling, capital flow, and macro constraints.                            |
| S    |    5 | Local optimization: Hill Climbing, Simulated Annealing, Tabu Search                            |                 9.5 | Yes             | Cheap, easy, and useful for graph weights, paths, schedules, and strategy improvement.                           |
| S    |    6 | Gradient Descent / SGD                                                                         |                 9.0 | Yes             | Core update rule for scores, weights, learned edge strength, and a future autodiff layer.                        |
| S    |    7 | PCA / SVD-style reduction                                                                      |                 9.0 | Yes             | Compresses macro variables, detects dominant factors, and simplifies noisy data.                                 |
| S    |    8 | Clustering: K-Means, K-Means++, DBSCAN, GMM                                                    |                 8.8 | Yes             | Groups similar states, companies, sectors, events, or market regimes.                                            |
| S    |    9 | Anomaly detection: Z-Score, Isolation Forest, LOF                                              |                 8.7 | Yes             | Finds shocks, unusual data, broken edges, regime changes, and stress signals.                                    |
| S    |   10 | Association rules: Apriori, FP-Growth, ECLAT                                                   |                 8.5 | Yes             | Detects repeated co-occurrence patterns such as “when X happens, Y often follows.”                               |
| A    |   11 | Tree models: Decision Trees, Random Forest, Extra Trees                                        |                 8.3 | Yes             | Explainable, low-maintenance, and good with small/medium structured data.                                        |
| A    |   12 | Boosted trees: XGBoost, LightGBM, CatBoost, GBM, AdaBoost                                      |                 8.2 | Later           | Strong for tabular prediction; less native to Rust/GPU graph path unless wrapped.                                |
| A    |   13 | Linear / regularized regression: OLS, Ridge, Lasso, Elastic Net, Quantile Regression           |                 8.0 | Yes             | Excellent baseline for influence strength, lagged effects, and macro variable modeling.                          |
| A    |   14 | Time-series models: AR, ARIMA, ARIMAX, SARIMA, SARIMAX, VAR, Exponential Smoothing             |                 8.0 | Yes             | Directly useful for macro structure and lagged influence.                                                        |
| A    |   15 | Bayesian models: Bayesian Regression, Bayesian Belief Networks, Gaussian Process               |                 7.8 | Later           | Useful for uncertainty, but can become expensive or complex.                                                     |
| A    |   16 | HMM / Viterbi                                                                                  |                 7.8 | Yes             | Good for hidden regimes: market state, workflow state, failure state.                                            |
| A    |   17 | MDP / Bellman Equation / Q-Learning                                                            |                 7.5 | Later           | Useful once environment, state, action, and reward are formalized.                                               |
| B    |   18 | Embeddings: Word2Vec, GloVe, FastText, BERT embeddings                                         |                 7.2 | Yes, pretrained | Good translation layer from text/data into symbolic graph. Do not train from scratch yet.                        |
| B    |   19 | Attention / Transformer concepts                                                               |                 7.0 | Conceptually    | Useful as a routing pattern; training transformers is not first priority.                                        |
| B    |   20 | MCTS / Minimax / Alpha-Beta / Expectimax                                                       |                 6.8 | Later           | Useful for planning/search, but less important than basic graph + DP first.                                      |
| B    |   21 | MLP / Perceptron                                                                               |                 6.5 | Later           | Simple neural scoring layer; useful but not first priority.                                                      |
| B    |   22 | RNN / LSTM / GRU / ESN                                                                         |                 6.2 | Later           | Useful for sequences, but classical time-series + graph propagation gives faster leverage now.                   |
| C    |   23 | Autoencoders / VAE                                                                             |                 5.8 | Later           | Useful for compression/anomaly detection, but heavier than PCA/Isolation Forest.                                 |
| C    |   24 | CNNs / vision models: LeNet, AlexNet, VGG, ResNet, MobileNet, EfficientNet                     |                 5.0 | Not now         | High leverage only if the input is images/video.                                                                 |
| C    |   25 | Object detection / segmentation: YOLO, R-CNN, U-Net, Mask R-CNN, DeepLab                       |                 4.5 | Not now         | Powerful, but wrong domain unless processing visual streams.                                                     |
| D    |   26 | GANs / StyleGAN / CycleGAN / Pix2Pix                                                           |                 3.5 | No              | Creative generation is low leverage for the current execution/influence-map system.                              |
| D    |   27 | Large text generation models: GPT, T5, BART, RoBERTa, Transformer-XL                           | 3.0 train / 8.0 use | Use externally  | Training is unrealistic; using pretrained/API models as parsers/translators is useful.                           |
| D    |   28 | PPO / Actor-Critic / REINFORCE / DQN                                                           | 3.0 now / 8.0 later | Not first       | Needs environment, reward, simulator, and many episodes.                                                         |

## Final Ranking

```text
Graph Search
> Dynamic Programming
> Constraint Solving
> Linear / Convex Optimization
> Classical ML
> Embeddings
> RL
> Deep Vision
> GANs
```
