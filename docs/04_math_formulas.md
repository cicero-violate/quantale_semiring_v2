# 04 — Math Formulas

## 1. Leverage Score

```math
L(a)=\frac{I \cdot F \cdot D \cdot G}{C \cdot M \cdot B}
```

Where:

| Symbol | Meaning                                             |
|--------+-----------------------------------------------------|
| `a`    | Algorithm                                           |
| `L(a)` | Leverage score                                      |
| `I`    | Impact on the system                                |
| `F`    | Fit with Rust/topology/tensor quantale architecture |
| `D`    | Data efficiency                                     |
| `G`    | GPU/parallel fit                                    |
| `C`    | Implementation cost                                 |
| `M`    | Maintenance cost                                    |
| `B`    | Big-training-budget penalty                         |

## 2. Dynamic Weighted Influence Graph

```math
G_t=(V_t,E_t,W_t,\Tau_t,S_t)
```

Where:

| Symbol  | Meaning                                    |
|---------+--------------------------------------------|
| `V_t`   | Variables/nodes at time `t`                |
| `E_t`   | Directed edges                             |
| `W_t`   | Edge weights                               |
| `Tau_t` | Time delays/lags                           |
| `S_t`   | Edge signs: positive or negative influence |

## 3. Node Update Formula

```math
x_j(t+1)=f_j\left(\sum_i w_{ij}\cdot s_{ij}\cdot x_i(t-\tau_{ij})+u_j(t)\right)+\epsilon_j(t)
```

Meaning:

```text
Each variable updates from delayed weighted influence from other variables, plus external shocks and noise.
```

Where:

| Symbol         | Meaning                                         |
|----------------+-------------------------------------------------|
| `x_i(t)`       | Value of source variable `i` at time `t`        |
| `x_j(t+1)`     | Next value of target variable `j`               |
| `w_ij`         | Influence strength from `i` to `j`              |
| `s_ij`         | Sign of influence: `+1` or `-1`                 |
| `tau_ij`       | Time delay from source to target                |
| `u_j(t)`       | External shock/input                            |
| `epsilon_j(t)` | Noise/error                                     |
| `f_j`          | Optional nonlinear activation/response function |

## 4. Matrix Form

```math
X_{t+1}=f(WX_t+U_t+\epsilon_t)
```

Meaning:

```text
The whole world state updates by multiplying the current state by an influence matrix.
```

## 5. Matrix Form With Delays

```math
X_{t+1}=f\left(\sum_{\tau=0}^{k}W_{\tau}X_{t-\tau}+U_t\right)
```

Meaning:

```text
Different causes act at different time delays.
```

Example:

```text
Oil(t) -> Transport(t+7) -> Food(t+21) -> Stress(t+30) -> Demand(t+45)
```

## 6. Edge-Scoring Formula

```math
w_{ij}=\alpha C_{ij}+\beta G_{ij}+\gamma M_{ij}+\delta R_{ij}+\eta E_{ij}
```

Where:

| Symbol | Meaning                        |
|--------+--------------------------------|
| `C_ij` | Lagged correlation score       |
| `G_ij` | Granger-style prediction score |
| `M_ij` | Mutual information score       |
| `R_ij` | Domain-rule score              |
| `E_ij` | Event-evidence score           |

Meaning:

```text
An influence edge is strong when statistics, timing, domain logic, and event evidence agree.
```

## 7. Tensor Edge Representation

```math
T \in \mathbb{R}^{L \times N \times N}
```

Where:

| Symbol       | Meaning                                            |
|--------------+----------------------------------------------------|
| `N`          | Number of macro variables/nodes                    |
| `L`          | Number of semantic edge layers                     |
| `T[:, i, j]` | All known influence data from node `i` to node `j` |

Example layers:

```text
T_0 = confidence
T_1 = cost/friction
T_2 = safety/risk
T_3 = delay
T_4 = capital flow
```

## 8. Simplest System Formula

```text
Next State = Current State x Influence Graph + Shock
```

More explicitly:

```math
X_{t+1}=G(X_t,D_t,U_t)
```

Where:

| Symbol | Meaning                    |
|--------+----------------------------|
| `X_t`  | Current world/system state |
| `D_t`  | Observed data stream       |
| `U_t`  | External shocks            |
| `G`    | Executable influence graph |
