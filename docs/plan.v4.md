# Plan: Data-Driven Market Feed, Paper Trades, and Continuous Agent Loop

## Goal

Connect the quantale agent to live market data, let the LLM propose structured
paper-trade decisions, and support continuous execution without hardcoding
markets, symbols, trade policy, or operator routing in Rust.

This is a simulation-only trading path. No real broker, exchange, wallet, or
order-routing API is allowed in this phase.

The runtime contract stays:

- JSON assets define operators, effects, topology, prompts, schemas, symbols,
  market-feed providers, and paper-trade limits
- Python asset operators perform external I/O and emit data-only JSON
- Rust dispatch remains generic over `assets/operators.json`
- Tensor state stores transition weights and learned routing preferences
- JSONL state stores observations, paper orders, fills, positions, and receipts
- The LLM can choose from declared topology nodes and schemas, but cannot invent
  executable behavior

---

## Current Baseline

Already present:

- `State::Plan` calls `assets/call_llm.py` and expects a tensor-edge JSON array
- `call_llm.py` loads topology/operators/templates from assets and prompts the LLM
  using data from `topology.json` and `operators.json`
- `assets/operators.json` defines generic process operators and `jit_cuda`
  execution operators
- `src/egress.rs` dispatches operators from JSON, not from Rust operator names
- `src/main.rs` runs for `config.max_ticks`, currently fixed at 64
- State operators can read/write JSONL files under `state/`

Missing:

- No market feed operator
- No market universe/config asset
- No market analysis asset or operator catalog
- No paper-trade operator
- No LLM template for choosing analysis chains
- No LLM template for trade decisions
- No schema validation for trade decisions
- No continuous-loop configuration
- No loop pacing/backoff
- No explicit guardrail that dummy trades cannot become real trades

---

## New Assets

### 1. `assets/market_feed.json`

Declare market-data providers and symbols. Rust must not know any symbols or
provider endpoints.

Shape:

```json
{
  "active_provider": "stooq_daily_csv",
  "symbols": ["SPY", "QQQ", "AAPL"],
  "providers": {
    "stooq_daily_csv": {
      "kind": "http_csv",
      "url_template": "https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv",
      "timeout_seconds": 10,
      "cache_seconds": 30
    }
  }
}
```

Rules:

- Provider selection is data-only via `active_provider`
- Symbol list is data-only via `symbols`
- The operator returns observations for all configured symbols
- If network fails, operator returns a structured failure receipt, not prose
- No API keys are stored in repo assets; optional auth comes from environment

### 2. `assets/trading_policy.json`

Declare paper-trading limits and validation rules.

Shape:

```json
{
  "mode": "paper",
  "base_currency": "USD",
  "starting_cash": 100000.0,
  "max_notional_per_order": 1000.0,
  "max_position_notional": 5000.0,
  "allowed_sides": ["buy", "sell", "hold"],
  "allowed_order_types": ["market"],
  "allow_fractional": true,
  "state_files": {
    "orders": "state/paper_orders.jsonl",
    "fills": "state/paper_fills.jsonl",
    "positions": "state/paper_positions.json"
  }
}
```

Rules:

- Only `mode: "paper"` is supported
- Any other mode is a hard operator failure
- Order sizing and symbols are validated from this file and `market_feed.json`
- The paper broker applies deterministic simulated fills from the latest observed
  market price

### 3. `assets/trade_decision_schema.json`

Define the only shape the LLM may emit for paper-trade decisions.

Shape:

```json
{
  "type": "object",
  "required": ["orders", "reason"],
  "properties": {
    "orders": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["symbol", "side", "quantity", "order_type"],
        "properties": {
          "symbol": {"type": "string"},
          "side": {"type": "string"},
          "quantity": {"type": "number"},
          "order_type": {"type": "string"}
        }
      }
    },
    "reason": {"type": "string"}
  }
}
```

No third-party JSON-schema dependency is required initially. The paper-trade
operator can validate the subset it needs directly from the asset.

### 4. `assets/market_analysis.json`

Declare JIT-capable market analysis operators and their input/output slots.
Rust must not know indicator names, window sizes, thresholds, or market symbols.

Shape:

```json
{
  "default_chain": [
    "Analysis::Return1",
    "Analysis::Volatility",
    "Analysis::SignalScore"
  ],
  "features": {
    "price": "market.price",
    "volume": "market.volume"
  },
  "analysis_outputs": {
    "score": "analysis.signal_score",
    "risk": "analysis.volatility"
  }
}
```

Rules:

- The LLM may choose an analysis chain only from topology-visible operators
- Every chosen analysis operator must be declared in `assets/operators.json`
- `jit_cuda` analysis operators declare math in `jit_body`
- Slot flow comes entirely from `effects.reads` and `effects.writes`
- Indicator names are documentation only; slots are the executable contract

### 5. `assets/analysis_decision_schema.json`

Define the shape the LLM may emit when selecting a market analysis chain.

Shape:

```json
{
  "type": "object",
  "required": ["analysis_chain", "reason"],
  "properties": {
    "analysis_chain": {
      "type": "array",
      "items": {"type": "string"}
    },
    "reason": {"type": "string"}
  }
}
```

The selected chain is data, not code. A validator maps operator names to
topology nodes and `operators.json` entries before anything reaches execution.

---

## New Python Operators

### 1. `assets/market_feed.py`

Responsibilities:

- Read `assets/market_feed.json`
- Fetch real market observations from the configured provider
- Normalize observations into one JSON object
- Write the latest observation to `state/market_feed.jsonl`
- Print a compact JSON payload to stdout

Output shape:

```json
{
  "market_feed": {
    "provider": "stooq_daily_csv",
    "observed_at": "2026-06-01T17:20:00Z",
    "symbols": [
      {
        "symbol": "SPY",
        "price": 123.45,
        "open": 120.0,
        "high": 124.0,
        "low": 119.5,
        "volume": 1000000
      }
    ]
  }
}
```

Failure shape:

```json
{
  "market_feed": {
    "error": "provider_timeout",
    "provider": "stooq_daily_csv"
  }
}
```

### 2. `assets/paper_trade.py`

Responsibilities:

- Read `assets/trading_policy.json`
- Read latest market prices from the incoming payload or `state/market_feed.jsonl`
- Read LLM trade decision JSON from the incoming payload
- Validate symbols, side, quantity, order type, and notional limits
- Simulate fills only; never call a broker
- Update `state/paper_orders.jsonl`, `state/paper_fills.jsonl`, and
  `state/paper_positions.json`
- Print a structured paper-trade receipt

Output shape:

```json
{
  "paper_trade_receipt": {
    "accepted": true,
    "orders": [...],
    "fills": [...],
    "positions": {...},
    "cash": 99950.0
  }
}
```

Hard failures:

- `trading_policy.mode != "paper"`
- Unknown symbol
- Invalid side/order type
- Quantity <= 0 unless side is `hold`
- Order exceeds configured notional limits
- Missing market price

### 3. `assets/analysis_plan.py`

Responsibilities:

- Call `assets/call_llm.py --template analysis`
- Provide the latest market feed, `assets/market_analysis.json`, and topology
  JIT operator summaries to the LLM
- Validate the LLM's selected analysis chain against
  `assets/analysis_decision_schema.json`, `assets/topology.json`, and
  `assets/operators.json`
- Print a structured analysis-chain decision

Output shape:

```json
{
  "analysis_plan": {
    "analysis_chain": [
      "Analysis::Return1",
      "Analysis::Volatility",
      "Analysis::SignalScore"
    ],
    "reason": "Use return and volatility before score generation."
  }
}
```

### 4. `assets/analysis_result.py`

Responsibilities:

- Read the JIT analysis receipts from context
- Normalize outputs into `analysis.result`
- Persist compact observations to `state/analysis_results.jsonl`
- Print analysis output for `State::TradePlan`

This operator does no math. Math belongs in JIT kernels declared in
`assets/operators.json`.

---

## LLM Prompt Changes

Extend `assets/call_llm_templates.json` with a `trade` template.

### `analysis` template

Add an `analysis` template so the LLM can choose how to analyze market data.

The template must:

- Explain that output is analysis-chain decision JSON only
- Include latest market feed observations
- Include `assets/market_analysis.json`
- Include topology-visible `jit_cuda` analysis operators with reads/writes
- Require each selected operator to be adjacent by slot dependency
- Forbid inventing indicators, slots, kernels, symbols, or CUDA code
- Forbid trade orders in this template

Example output:

```json
{
  "analysis_chain": [
    "Analysis::Return1",
    "Analysis::Volatility",
    "Analysis::SignalScore"
  ],
  "reason": "The feed needs return and volatility features before scoring."
}
```

### `trade` template

The `trade` template must:

- Explain that output is paper-trade decision JSON only
- Include market observations from context
- Include JIT analysis results from context
- Include the configured symbol universe and paper limits
- Include the trade decision schema
- Forbid real trading, broker APIs, credentials, leverage, and hidden orders
- Allow `hold` decisions

Example output:

```json
{
  "orders": [
    {
      "symbol": "SPY",
      "side": "hold",
      "quantity": 0.0,
      "order_type": "market"
    }
  ],
  "reason": "No high-confidence edge in the latest feed."
}
```

`assets/call_llm.py` should load optional extra prompt context from:

- `assets/market_feed.json`
- `assets/market_analysis.json`
- `assets/analysis_decision_schema.json`
- `assets/trading_policy.json`
- `assets/trade_decision_schema.json`

The template loader remains data-driven: the Python script injects asset content,
not hardcoded symbols or limits.

The LLM's role split is:

- `analysis`: choose a JIT analysis chain from declared operators
- `trade`: use market feed plus analysis result to choose paper orders
- `plan`: update tensor edges/topology preferences

---

## Operator Registry Changes

Add entries to `assets/operators.json`:

### `State::MarketFeed`

```json
{
  "node_name": "State::MarketFeed",
  "executable": "python3",
  "static_args": ["assets/market_feed.py"],
  "input_mapping": {"stdin_mode": "json"},
  "effects": {
    "reads": ["market.config"],
    "writes": ["market.feed", "state/market_feed.jsonl"],
    "locks": []
  }
}
```

### `State::TradePlan`

```json
{
  "node_name": "State::TradePlan",
  "executable": "python3",
  "static_args": ["assets/call_llm.py", "--template", "trade"],
  "input_mapping": {"stdin_mode": "json"},
  "effects": {
    "reads": ["market.feed", "trading.policy"],
    "writes": ["trade.decision"],
    "locks": []
  }
}
```

### `State::AnalysisPlan`

```json
{
  "node_name": "State::AnalysisPlan",
  "executable": "python3",
  "static_args": ["assets/call_llm.py", "--template", "analysis"],
  "input_mapping": {"stdin_mode": "json"},
  "effects": {
    "reads": ["market.feed", "analysis.config"],
    "writes": ["analysis.plan"],
    "locks": []
  }
}
```

### JIT analysis operators

Declare analysis kernels as ordinary `jit_cuda` operators in
`assets/operators.json`.

Example shape:

```json
{
  "node_name": "Analysis::SignalScore",
  "executable": "jit_cuda",
  "static_args": [],
  "jit_body": "out[i] = in0[i] / (1.0f + fabsf(in1[i]));",
  "effects": {
    "reads": ["analysis.return", "analysis.volatility"],
    "writes": ["analysis.signal_score"],
    "locks": []
  }
}
```

Rules:

- These are examples to add as data, not Rust constants
- Chain fusion uses the existing `src/jit_kernel_fusion/` detector/synth/cache
- Adding a new indicator is an `operators.json` and optional topology edit
- No Rust code may name an analysis operator

### `State::PaperTrade`

```json
{
  "node_name": "State::PaperTrade",
  "executable": "python3",
  "static_args": ["assets/paper_trade.py"],
  "input_mapping": {"stdin_mode": "json"},
  "effects": {
    "reads": ["trade.decision", "market.feed", "trading.policy"],
    "writes": ["trade.receipt", "state/paper_orders.jsonl", "state/paper_positions.json"],
    "locks": ["paper_broker"]
  }
}
```

Names above are examples to add as data. Rust must not special-case them.

---

## Topology Changes

Add topology nodes for the new states and connect them through declared
transitions:

- `State::MarketFeed`
- `State::AnalysisPlan`
- `State::TradePlan`
- `State::PaperTrade`
- optional events:
  - `Event::MarketFeedUpdated`
  - `Event::AnalysisPlanReady`
  - `Event::AnalysisFinished`
  - `Event::TradeDecisionReady`
  - `Event::PaperTradeFilled`
  - `Event::PaperTradeRejected`
- analysis execution nodes declared from `assets/operators.json`, for example:
  - `Analysis::Return1`
  - `Analysis::Volatility`
  - `Analysis::SignalScore`

Initial transition intent:

```text
State::Input
  -> State::MarketFeed
  -> Event::MarketFeedUpdated
  -> State::AnalysisPlan
  -> Analysis::* jit_cuda chain
  -> Event::AnalysisFinished
  -> State::Plan
  -> State::TradePlan
  -> Event::TradeDecisionReady
  -> State::PaperTrade
  -> Event::PaperTradeFilled | Event::PaperTradeRejected
  -> State::Memory
  -> State::Learn
```

All edges must be declared in `assets/topology.json`; the LLM sees them through
the prompt and emits only tensor-edge weights.

The analysis chain is still data-driven:

- Topology declares which analysis nodes can be reached
- Operators declare executable JIT bodies and slot effects
- The LLM chooses a chain from those declared nodes
- The JIT chain detector fuses adjacent compatible operators from effects

---

## Runtime Loop Changes

### 1. Configurable tick count

Change `SystemConfig::default()` so `max_ticks` can be set by environment:

- `QUANTALE_MAX_TICKS=64` for bounded runs
- `QUANTALE_MAX_TICKS=0` or `QUANTALE_LOOP_FOREVER=1` for continuous mode

### 2. Loop pacing

Add environment-controlled sleep after each full tick:

- `QUANTALE_TICK_SLEEP_MS=1000`

No busy infinite loop.

### 3. Run script option

Update `run.sh`:

```bash
./run.sh
./run.sh --forever
./run.sh --ticks 10
```

Behavior:

- `--forever` exports `QUANTALE_LOOP_FOREVER=1`
- `--ticks N` exports `QUANTALE_MAX_TICKS=N`
- default remains bounded

Do not make the default command infinite.

---

## Payload Handling

The main loop currently wraps stdout into `{"context": stdout}` repeatedly.
For this phase:

- Keep generic payload passing
- Add compact context normalization for LLM prompts
- Preserve structured outputs from market feed and paper trade in context
- Avoid passing entire historical JSONL contents into LLM prompts

Required behavior:

- Market feed output can be consumed by `State::TradePlan`
- Market feed output can be consumed by `State::AnalysisPlan`
- Analysis plan output can route into topology-visible `jit_cuda` analysis nodes
- JIT analysis output can be consumed by `State::TradePlan`
- Trade decision output can be consumed by `State::PaperTrade`
- Paper-trade receipt can be consumed by memory/learn

No Rust field names for symbols, sides, prices, indicators, or analysis outputs
should be hardcoded beyond generic JSON payload movement.

---

## Safety Constraints

- No real broker integration
- No real order submission
- No credential loading for trading
- No live leverage, margin, shorting, options, or derivatives execution
- No hidden network calls from Rust
- All market network I/O happens in `assets/market_feed.py`
- All market analysis math happens in `jit_cuda` operators declared in
  `assets/operators.json`
- All paper trading happens in `assets/paper_trade.py`
- All limits are declared in `assets/trading_policy.json`
- Operator failures must be structured receipts, not panics

---

## Verification

Unit/smoke tests:

- `assets/market_feed.py` with fixture provider returns normalized market JSON
- `assets/call_llm.py --template analysis` renders prompt with market analysis
  config and topology-visible JIT operators
- Analysis-chain validation rejects unknown operators
- Analysis-chain validation rejects non-adjacent slot flow
- JIT analysis operator chain synthesizes from `jit_body` and effects
- `assets/paper_trade.py` accepts a valid `hold` decision
- `assets/paper_trade.py` rejects non-paper mode
- `assets/paper_trade.py` rejects unknown symbols
- `assets/paper_trade.py` rejects notional-limit violations
- `assets/call_llm.py --template trade` renders prompt with symbols, limits, and
  schema loaded from assets plus analysis results from context
- `cargo check`
- `cargo test --no-default-features`

Runtime smoke:

```bash
QUANTALE_MAX_TICKS=5 ./run.sh
```

Continuous-mode smoke:

```bash
timeout 30s ./run.sh --forever
```

Expected files:

- `state/market_feed.jsonl`
- `state/analysis_results.jsonl`
- `state/paper_orders.jsonl`
- `state/paper_fills.jsonl`
- `state/paper_positions.json`

Expected grep checks:

- No broker API endpoints in Rust source
- No symbols hardcoded in Rust source
- No indicator names hardcoded in Rust source
- No `live`, `real`, or `broker` execution mode accepted by paper-trade operator

---

## Acceptance Criteria

- Market symbols and provider are configurable from `assets/market_feed.json`
- Analysis operators and chains are configurable from `assets/operators.json`,
  `assets/topology.json`, and `assets/market_analysis.json`
- LLM can choose a valid JIT market-analysis chain
- JIT market-analysis kernels are synthesized from declared `jit_body` strings
- LLM trade prompt gets market feed, symbols, limits, and schema from assets
- LLM trade prompt includes analysis results from the JIT analysis stage
- LLM can emit a valid paper-trade decision
- Paper-trade operator simulates fills and updates state files
- Invalid trade decisions fail safely with structured error receipts
- Continuous mode is available but never the default
- Bounded runs still work unchanged
- `cargo check` passes
- `cargo test --no-default-features` passes

---

## Non-Goals

- Real brokerage integration
- Real exchange order routing
- Financial advice
- Portfolio optimization beyond simple paper-trade validation
- Intraday low-latency data guarantees
- Authentication or paid market-data providers
- Multi-account support
- Real P&L accounting beyond deterministic paper fills
