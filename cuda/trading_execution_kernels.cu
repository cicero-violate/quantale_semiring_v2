// Legacy trading execution kernels.
//
// Math operators moved to runtime JIT synthesis from generated operator assets.
// This file is no longer compiled by build.rs.

extern "C" {

// Fuses DynamicAlphaSignalEvaluator + PortfolioRiskConstraintFilter.
// market_feed:      [n] raw price/volume features
// portfolio_state:  [n] current position weights
// trading_signals:  [n] raw alpha signal scores
// results:          [n] risk-adjusted execution decisions (output)
__global__ void fused_alpha_and_risk_kernel(
    const float* __restrict__ market_feed,
    const float* __restrict__ portfolio_state,
    const float* __restrict__ trading_signals,
    float* __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float alpha  = tanhf(market_feed[idx]) * trading_signals[idx];
    float risk   = 1.0f - fabsf(portfolio_state[idx]);
    results[idx] = fmaxf(-1.0f, fminf(1.0f, alpha * risk));
}

// Fuses OrderbookImbalanceWeaver + AlphaSignalEvaluator.
// orderbook:      [n] bid/ask imbalance features
// alpha_signals:  [n] upstream alpha scores
// results:        [n] imbalance-weighted alpha (output)
__global__ void fused_orderbook_and_alpha_kernel(
    const float* __restrict__ orderbook,
    const float* __restrict__ alpha_signals,
    const float* __restrict__ unused,
    float*       __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    (void)unused;

    float imb    = tanhf(orderbook[idx]);
    results[idx] = alpha_signals[idx] * (1.0f + 0.5f * imb)
        / fmaxf(1.0f, fabsf(alpha_signals[idx]));
}

// Fuses feed normalisation + alpha evaluation + risk filter in one pass.
// feed:            [n] raw market feed values
// alpha_signals:   [n] pre-computed alpha scores
// portfolio_state: [n] current position weights
// results:         [n] final execution decisions (output)
__global__ void fused_feed_alpha_and_risk_kernel(
    const float* __restrict__ feed,
    const float* __restrict__ alpha_signals,
    const float* __restrict__ portfolio_state,
    float*       __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float norm  = tanhf(feed[idx]);
    float alpha = alpha_signals[idx] * norm;
    float risk  = 1.0f - fabsf(portfolio_state[idx]);
    results[idx] = fmaxf(-1.0f, fminf(1.0f, alpha * risk));
}

} // extern "C"
