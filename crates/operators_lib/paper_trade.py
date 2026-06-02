#!/usr/bin/env python3
"""State::PaperTrade operator: validate and simulate paper-trade fills.

Reads trading_policy.json, extracts the LLM trade decision and market prices
from the context payload, validates against configured limits, simulates fills,
updates state files, and prints a structured receipt.

Hard failures (exit 0 with accepted=false):
  - trading_policy.mode != "paper"
  - unknown symbol
  - invalid side or order_type
  - quantity <= 0 for non-hold orders
  - order exceeds notional limits
  - missing market price for a non-hold order

Simulation logic: fill at the latest observed price from market feed.
"""

import datetime
import json
import os
import pathlib
import sys

ASSET_DIR = pathlib.Path(__file__).resolve().parent
STATE_DIR = pathlib.Path("state")
POLICY_PATH = ASSET_DIR / "trading_policy.json"
MARKET_FEED_CONFIG_PATH = ASSET_DIR / "market_feed.json"


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def read_stdin_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[paper_trade] stdin parse error: {exc}\n")
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_json_with_key(value, key: str):
    """Recursively search a nested payload for a dict containing the given key."""
    if isinstance(value, dict):
        if key in value:
            return value
        for v in value.values():
            result = extract_json_with_key(v, key)
            if result is not None:
                return result
    if isinstance(value, list):
        for item in value:
            result = extract_json_with_key(item, key)
            if result is not None:
                return result
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return extract_json_with_key(parsed, key)
            except json.JSONDecodeError:
                pass
    return None


def extract_market_prices(payload: dict) -> dict:
    """Return {symbol: price} from the latest market feed in payload or state file."""
    prices = {}

    feed_obj = extract_json_with_key(payload, "market_feed")
    if feed_obj is not None:
        feed = feed_obj.get("market_feed", {})
        for entry in feed.get("symbols", []):
            sym = entry.get("symbol")
            price = entry.get("price")
            if sym and price is not None:
                prices[sym] = float(price)

    if not prices:
        state_log = STATE_DIR / "market_feed.jsonl"
        if state_log.exists():
            last_line = ""
            try:
                with state_log.open() as fh:
                    for line in fh:
                        stripped = line.strip()
                        if stripped:
                            last_line = stripped
            except OSError:
                pass
            if last_line:
                try:
                    feed_obj = json.loads(last_line)
                    feed = feed_obj.get("market_feed", {})
                    for entry in feed.get("symbols", []):
                        sym = entry.get("symbol")
                        price = entry.get("price")
                        if sym and price is not None:
                            prices[sym] = float(price)
                except (json.JSONDecodeError, AttributeError):
                    pass

    return prices


def load_positions(path: pathlib.Path, starting_cash: float) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"cash": starting_cash, "holdings": {}}


def save_positions(path: pathlib.Path, positions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(positions, indent=2))


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def rejection(reason: str, orders: list) -> dict:
    return {
        "paper_trade_receipt": {
            "accepted": False,
            "rejection_reason": reason,
            "orders": orders,
            "fills": [],
        }
    }


def main() -> None:
    try:
        policy = load_policy()
    except Exception as exc:
        output = rejection(f"policy_load_failed: {exc}", [])
        print(json.dumps(output))
        sys.exit(0)

    mode = policy.get("mode", "")
    if mode != "paper":
        output = rejection(f"unsupported_mode:{mode}", [])
        print(json.dumps(output))
        sys.exit(0)

    allowed_sides = set(policy.get("allowed_sides", ["buy", "sell", "hold"]))
    allowed_order_types = set(policy.get("allowed_order_types", ["market"]))
    max_notional = float(policy.get("max_notional_per_order", 1000.0))
    max_position = float(policy.get("max_position_notional", 5000.0))
    starting_cash = float(policy.get("starting_cash", 100000.0))
    state_files = policy.get("state_files", {})
    orders_path = pathlib.Path(state_files.get("orders", "state/paper_orders.jsonl"))
    fills_path = pathlib.Path(state_files.get("fills", "state/paper_fills.jsonl"))
    positions_path = pathlib.Path(state_files.get("positions", "state/paper_positions.json"))

    try:
        mf_config = json.loads(MARKET_FEED_CONFIG_PATH.read_text())
        allowed_symbols = set(mf_config.get("symbols", []))
    except Exception:
        allowed_symbols = set()

    payload = read_stdin_payload()
    market_prices = extract_market_prices(payload)

    decision_obj = extract_json_with_key(payload, "orders")
    if decision_obj is None:
        output = rejection("no_trade_decision_in_context", [])
        print(json.dumps(output))
        sys.exit(0)

    orders = decision_obj.get("orders", [])
    reason = decision_obj.get("reason", "")
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    positions = load_positions(positions_path, starting_cash)
    cash = float(positions.get("cash", starting_cash))
    holdings = dict(positions.get("holdings", {}))

    accepted_orders = []
    fills = []
    rejections_list = []

    for order in orders:
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        quantity = float(order.get("quantity", 0.0))
        order_type = order.get("order_type", "")

        if allowed_symbols and symbol not in allowed_symbols:
            rejections_list.append({"symbol": symbol, "reason": "unknown_symbol"})
            continue

        if side not in allowed_sides:
            rejections_list.append({"symbol": symbol, "reason": f"invalid_side:{side}"})
            continue

        if order_type not in allowed_order_types:
            rejections_list.append({"symbol": symbol, "reason": f"invalid_order_type:{order_type}"})
            continue

        if side == "hold":
            accepted_orders.append(order)
            continue

        if quantity <= 0:
            rejections_list.append({"symbol": symbol, "reason": "quantity_not_positive"})
            continue

        price = market_prices.get(symbol)
        if price is None:
            rejections_list.append({"symbol": symbol, "reason": "missing_market_price"})
            continue

        notional = quantity * price
        if notional > max_notional:
            rejections_list.append({"symbol": symbol, "reason": f"exceeds_max_notional:{notional:.2f}>{max_notional}"})
            continue

        current_position_notional = holdings.get(symbol, 0.0) * price
        if side == "buy" and (current_position_notional + notional) > max_position:
            rejections_list.append({"symbol": symbol, "reason": f"exceeds_max_position_notional"})
            continue

        if side == "buy" and cash < notional:
            rejections_list.append({"symbol": symbol, "reason": "insufficient_cash"})
            continue

        fill = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "fill_price": price,
            "notional": notional,
            "ts": ts,
        }
        fills.append(fill)
        accepted_orders.append(order)

        if side == "buy":
            cash -= notional
            holdings[symbol] = holdings.get(symbol, 0.0) + quantity
        elif side == "sell":
            cash += notional
            holdings[symbol] = max(0.0, holdings.get(symbol, 0.0) - quantity)

    positions["cash"] = cash
    positions["holdings"] = holdings

    try:
        for order in accepted_orders:
            append_jsonl(orders_path, {"ts": ts, "order": order})
        for fill in fills:
            append_jsonl(fills_path, fill)
        save_positions(positions_path, positions)
    except OSError as exc:
        sys.stderr.write(f"[paper_trade] state write failed: {exc}\n")

    output = {
        "paper_trade_receipt": {
            "accepted": True,
            "orders": accepted_orders,
            "fills": fills,
            "rejections": rejections_list,
            "positions": {"cash": cash, "holdings": holdings},
            "cash": cash,
            "reason": reason,
            "ts": ts,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
