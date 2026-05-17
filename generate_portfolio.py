"""Generate a synthetic financial portfolio CSV designed for pandas demos.

The schema is intentionally chosen so a downstream notebook can practice:
  - time-series resampling and rolling windows (timestamp, closing_price_usd)
  - groupby + pivot (ticker, transaction_type)
  - merges for FX normalization (currency, price_local)
  - conditional/signed aggregation and cumulative positions (transaction_type, quantity)
  - categorical/string ops with missing values (market_event)
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEED = 42
N_ROWS = 100
OUT_PATH = Path(__file__).parent / "portfolio.csv"

TICKERS: dict[str, dict[str, float]] = {
    "AAPL":  {"base_price": 185.0, "vol": 2.5,  "currency_weights": {"USD": 0.70, "EUR": 0.15, "GBP": 0.10, "CHF": 0.05}},
    "MSFT":  {"base_price": 420.0, "vol": 4.0,  "currency_weights": {"USD": 0.75, "EUR": 0.15, "GBP": 0.10}},
    "GOOGL": {"base_price": 175.0, "vol": 3.0,  "currency_weights": {"USD": 0.80, "EUR": 0.10, "JPY": 0.10}},
    "TSLA":  {"base_price": 240.0, "vol": 9.0,  "currency_weights": {"USD": 0.60, "EUR": 0.20, "GBP": 0.10, "JPY": 0.10}},
    "AMZN":  {"base_price": 195.0, "vol": 3.5,  "currency_weights": {"USD": 0.70, "EUR": 0.15, "CHF": 0.10, "JPY": 0.05}},
}

# Approximate FX: units of local currency per 1 USD.
FX_PER_USD = {"USD": 1.00, "EUR": 0.92, "GBP": 0.79, "CHF": 0.88, "JPY": 155.0}

TRANSACTION_TYPES = ["BUY", "SELL", "DIVIDEND"]
TRANSACTION_WEIGHTS = [0.55, 0.35, 0.10]

# Includes None to exercise missing-value handling.
MARKET_EVENTS = [None, "earnings_beat", "earnings_miss", "split", "dividend_announced", "guidance_raise"]
EVENT_WEIGHTS = [0.70, 0.08, 0.07, 0.03, 0.07, 0.05]


def generate() -> list[dict]:
    rng = np.random.default_rng(SEED)
    tickers = list(TICKERS.keys())

    start = datetime(2025, 1, 2, 9, 30, tzinfo=timezone.utc)
    # Spread 100 events across ~60 trading days with intraday jitter.
    minute_offsets = np.sort(rng.integers(0, 60 * 24 * 60, size=N_ROWS))

    rows: list[dict] = []
    # Ensure every ticker appears at least once by seeding the first 5 rows.
    forced_tickers = list(tickers)
    rng.shuffle(forced_tickers)

    for i in range(N_ROWS):
        ticker = forced_tickers[i] if i < len(forced_tickers) else rng.choice(tickers)
        cfg = TICKERS[ticker]

        ts = start + timedelta(minutes=int(minute_offsets[i]))

        # Random walk-ish closing price in USD around the base price.
        drift = rng.normal(0.0, cfg["vol"])
        trend = (i / N_ROWS - 0.5) * cfg["vol"] * 4  # small directional drift across the file
        closing_price_usd = round(max(cfg["base_price"] + drift + trend, 1.0), 2)

        currencies, weights = zip(*cfg["currency_weights"].items())
        currency = rng.choice(currencies, p=np.array(weights) / sum(weights))
        # Transaction price wiggles around the daily close, then convert to local currency.
        price_usd = closing_price_usd * (1.0 + rng.normal(0.0, 0.005))
        price_local = round(price_usd * FX_PER_USD[currency], 2 if currency != "JPY" else 0)

        transaction_type = rng.choice(TRANSACTION_TYPES, p=TRANSACTION_WEIGHTS)
        if transaction_type == "DIVIDEND":
            quantity = int(rng.integers(1, 50))
        else:
            quantity = int(rng.integers(5, 500))

        market_event = rng.choice(
            np.array(MARKET_EVENTS, dtype=object), p=EVENT_WEIGHTS
        )

        rows.append({
            "timestamp": ts.isoformat(),
            "ticker": ticker,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "price_local": price_local,
            "currency": currency,
            "closing_price_usd": closing_price_usd,
            "market_event": market_event if market_event is not None else "",
        })
    return rows


def main() -> None:
    rows = generate()
    fieldnames = [
        "timestamp",
        "ticker",
        "transaction_type",
        "quantity",
        "price_local",
        "currency",
        "closing_price_usd",
        "market_event",
    ]
    with OUT_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
