# Paid Problem: Market Watchlist Triage

## Workflow

Daily or on-demand market watchlist triage:

```text
ingest configured symbols
compute return, volatility, and signal score
propose paper-only actions
emit a receipt-backed rationale
update routing from observed decision quality
```

## Manual Problem

The recurring manual task is reviewing a watchlist and deciding which symbols
need closer attention.  The agent output shortens that review by ranking
symbols, producing a paper-only action, and preserving a receipt for later
quality review.

## Acceptance Metric

Track:

```text
symbols_triaged_per_run
minutes_saved_per_run
manual_checks_avoided
receipt_success_rate
```

## Safety Boundary

This remains simulation-only.  No broker credentials, real orders, exchange
transactions, wallet operations, or live trading actions are part of Plan v8.
