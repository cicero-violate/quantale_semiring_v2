#!/usr/bin/env python3
"""State::MarketFeed operator: fetch and normalize market data from configured provider.

Reads assets/market_feed.json, fetches prices for all configured symbols,
writes to state/market_feed.jsonl, and prints a compact JSON payload.

Supported provider kinds:
  http_csv           — stooq-style per-symbol CSV fetch
  coingecko_markets  — CoinGecko /coins/markets JSON (batch, no auth required)

Output shape (success):
  {"market_feed": {"provider": "...", "observed_at": "...", "symbols": [...]}}

Output shape (failure):
  {"market_feed": {"error": "provider_timeout", "provider": "..."}}
"""

import csv
import datetime
import io
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ASSET_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets"
STATE_DIR = pathlib.Path("state")
MARKET_FEED_CONFIG = ASSET_DIR / "market_feed.json"
MARKET_FEED_LOG = STATE_DIR / "market_feed.jsonl"
CACHE_FILE = STATE_DIR / "market_feed_cache.json"


def load_config() -> dict:
    return json.loads(MARKET_FEED_CONFIG.read_text())


def cache_valid(path: pathlib.Path, cache_seconds: int):
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > cache_seconds:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _http_get(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quantale-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ── CSV provider (stooq) ──────────────────────────────────────────────────────

def _safe_float(raw, default: float = 0.0) -> float:
    s = str(raw).strip() if raw is not None else ""
    if s in {"N/D", "N/A", "n/d", "n/a", "-", "", "null", "NULL"}:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def fetch_csv_symbol(symbol: str, provider: dict, timeout: int) -> dict:
    url = provider["url_template"].replace("{symbol}", symbol.lower())
    try:
        raw = _http_get(url, timeout)
    except urllib.error.URLError as exc:
        return {"symbol": symbol, "error": f"fetch_failed: {exc.reason}"}
    except Exception as exc:
        return {"symbol": symbol, "error": f"fetch_failed: {exc}"}

    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = [r for r in reader if any(v.strip() for v in r.values())]
    if not rows:
        return {"symbol": symbol, "error": "no_data_rows"}

    row = rows[-1]
    close  = _safe_float(row.get("Close") or row.get("close"))
    open_  = _safe_float(row.get("Open")  or row.get("open"))
    high   = _safe_float(row.get("High")  or row.get("high"))
    low_   = _safe_float(row.get("Low")   or row.get("low"))
    volume = int(_safe_float(row.get("Volume") or row.get("volume")))

    if close == 0.0:
        return {"symbol": symbol, "error": "no_data_nd"}

    return {"symbol": symbol, "price": close, "open": open_,
            "high": high, "low": low_, "volume": volume}


# ── CoinGecko /coins/markets provider ────────────────────────────────────────

def fetch_coingecko(symbols: list, provider: dict, timeout: int) -> list:
    symbol_map: dict = provider.get("symbol_map", {})
    id_to_symbol = {v: k for k, v in symbol_map.items()}
    coingecko_ids = ",".join(
        symbol_map[s] for s in symbols if s in symbol_map
    )
    if not coingecko_ids:
        return [{"symbol": s, "error": "no_coingecko_id"} for s in symbols]

    url = provider["url_template"].replace("{coingecko_ids}", coingecko_ids)
    try:
        raw = _http_get(url, timeout)
    except urllib.error.URLError as exc:
        return [{"symbol": s, "error": f"fetch_failed: {exc.reason}"} for s in symbols]
    except Exception as exc:
        return [{"symbol": s, "error": f"fetch_failed: {exc}"} for s in symbols]

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return [{"symbol": s, "error": f"json_parse: {exc}"} for s in symbols]

    results_by_id = {}
    for coin in data:
        cid = coin.get("id", "")
        sym = id_to_symbol.get(cid, cid.upper())
        results_by_id[sym] = {
            "symbol": sym,
            "price":  float(coin.get("current_price") or 0),
            "open":   float(coin.get("current_price", 0)) - float(coin.get("price_change_24h") or 0),
            "high":   float(coin.get("high_24h") or 0),
            "low":    float(coin.get("low_24h") or 0),
            "volume": int(float(coin.get("total_volume") or 0)),
        }

    out = []
    for s in symbols:
        if s in results_by_id:
            out.append(results_by_id[s])
        elif s not in symbol_map:
            out.append({"symbol": s, "error": "no_coingecko_id"})
        else:
            out.append({"symbol": s, "error": "not_in_response"})
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        config = load_config()
    except Exception as exc:
        print(json.dumps({"market_feed": {"error": f"config_load_failed: {exc}", "provider": "unknown"}}))
        sys.exit(0)

    provider_name = config.get("active_provider", "stooq_daily_csv")
    symbols = config.get("symbols", [])
    provider = config.get("providers", {}).get(provider_name)

    if provider is None:
        print(json.dumps({"market_feed": {"error": "provider_not_configured", "provider": provider_name}}))
        sys.exit(0)

    timeout = provider.get("timeout_seconds", 10)
    cache_seconds = provider.get("cache_seconds", 30)
    kind = provider.get("kind", "http_csv")

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cached = cache_valid(CACHE_FILE, cache_seconds)
    if cached is not None:
        print(json.dumps(cached))
        sys.exit(0)

    observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    if kind == "coingecko_markets":
        symbol_results = fetch_coingecko(symbols, provider, timeout)
    else:
        symbol_results = [fetch_csv_symbol(s, provider, timeout) for s in symbols]

    output = {
        "market_feed": {
            "provider": provider_name,
            "observed_at": observed_at,
            "symbols": symbol_results,
        }
    }

    try:
        with MARKET_FEED_LOG.open("a") as fh:
            fh.write(json.dumps(output, separators=(",", ":")) + "\n")
    except OSError as exc:
        sys.stderr.write(f"[market_feed] log write failed: {exc}\n")

    try:
        CACHE_FILE.write_text(json.dumps(output, separators=(",", ":")))
    except OSError as exc:
        sys.stderr.write(f"[market_feed] cache write failed: {exc}\n")

    print(json.dumps(output))


if __name__ == "__main__":
    main()
