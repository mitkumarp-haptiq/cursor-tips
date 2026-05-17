"""Five short, vectorized pandas demos against ``portfolio.csv``.

Each ``demo_*`` function maps to one of the techniques the CSV was designed for:

1. FX-normalize then aggregate (merge + groupby).
2. Signed position tracking (map + groupby.cumsum).
3. Daily close panel (pivot_table + pct_change + rolling).
4. Event impact study (groupby.agg with ``dropna=False``).
5. Multi-index slicing (set_index + sorted ``.loc``).

Run with::

    python pandas_demos.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "portfolio.csv"

# Units of local currency per 1 USD. Mirrors generate_portfolio.FX_PER_USD so the
# inverse round-trips cleanly back to the original USD price.
FX_PER_USD: dict[str, float] = {
    "USD": 1.00,
    "EUR": 0.92,
    "GBP": 0.79,
    "CHF": 0.88,
    "JPY": 155.0,
}


def load_portfolio(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the CSV with proper dtypes (timestamp parsed, event kept nullable)."""
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        dtype={
            "ticker": "category",
            "transaction_type": "category",
            "currency": "category",
        },
    )
    # Treat empty market_event cells as missing rather than the empty string.
    df["market_event"] = df["market_event"].where(df["market_event"].notna() & (df["market_event"] != ""))
    return df


def demo_fx_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """1) FX-normalize then aggregate notional traded per ticker (USD)."""
    fx = pd.DataFrame(
        {"currency": list(FX_PER_USD.keys()), "fx_per_usd": list(FX_PER_USD.values())}
    )
    merged = df.merge(fx, on="currency", how="left", validate="many_to_one")
    merged = merged.assign(
        price_usd=lambda d: d["price_local"] / d["fx_per_usd"],
        notional_usd=lambda d: d["price_usd"] * d["quantity"],
    )
    return (
        merged.groupby("ticker", observed=True)["notional_usd"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
        .to_frame()
    )


def demo_signed_positions(df: pd.DataFrame) -> pd.DataFrame:
    """2) Running share counts per ticker via signed cumulative sum."""
    sign_map = {"BUY": 1, "SELL": -1, "DIVIDEND": 0}
    signed_qty = df["transaction_type"].map(sign_map).astype("int64") * df["quantity"]
    out = (
        df.assign(signed_qty=signed_qty)
        .sort_values("timestamp")
        .assign(
            running_position=lambda d: d.groupby("ticker", observed=True)["signed_qty"].cumsum()
        )
    )
    # Last running position per ticker = net shares currently held.
    return (
        out.groupby("ticker", observed=True)["running_position"]
        .last()
        .to_frame("net_shares")
    )


def demo_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """3) Daily close panel → daily returns → 7-day rolling mean (smoothed)."""
    daily_close = df.pivot_table(
        index=df["timestamp"].dt.date,
        columns="ticker",
        values="closing_price_usd",
        aggfunc="last",
        observed=True,
    )
    daily_close.index = pd.to_datetime(daily_close.index).rename("date")
    # Forward-fill so a missing trading day for one ticker does not blow up pct_change.
    smoothed = daily_close.sort_index().ffill().pct_change().rolling(7, min_periods=3).mean()
    return smoothed.tail(10).round(4)


def demo_event_impact(df: pd.DataFrame) -> pd.DataFrame:
    """4) Closing-price stats grouped by market event (NaN = no event)."""
    stats = (
        df.groupby("market_event", dropna=False, observed=True)["closing_price_usd"]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    # Rename the implicit NaN group so the output is self-documenting.
    stats.index = stats.index.fillna("<no_event>")
    return stats.sort_values("count", ascending=False)


def demo_multiindex_slice(df: pd.DataFrame, ticker: str = "TSLA") -> pd.DataFrame:
    """5) MultiIndex ``(ticker, timestamp)`` slicing with time-window ``.loc``."""
    indexed = df.set_index(["ticker", "timestamp"]).sort_index()
    # Cross-section by ticker, then slice the first 7 days of that ticker's
    # history with label-based .loc. Deriving the window from data keeps the
    # demo non-empty even if the CSV is regenerated with a different seed.
    sub = indexed.loc[ticker]
    start = sub.index.min().normalize()
    end = start + pd.Timedelta(days=7)
    window = sub.loc[start:end]
    return window[["transaction_type", "quantity", "price_local", "currency"]]


def _print_section(title: str, payload: object) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")
    print(payload)


def main() -> None:
    df = load_portfolio()
    _print_section("1) FX-normalized notional traded per ticker (USD)", demo_fx_normalize(df))
    _print_section("2) Net shares held per ticker (signed cumsum)", demo_signed_positions(df))
    _print_section("3) 7-day rolling mean of daily returns (tail 10)", demo_daily_panel(df))
    _print_section("4) Closing-price stats by market_event", demo_event_impact(df))
    _print_section("5) TSLA transactions, first 7 days of history", demo_multiindex_slice(df))


if __name__ == "__main__":
    main()
