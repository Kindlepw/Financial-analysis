"""
Valuation engine: combines data fetching and DCF modelling to produce a
structured comparison of intrinsic values vs. current market prices for all
S&P 500 companies.
"""

import logging
from typing import Optional

import pandas as pd

from .data_fetcher import fetch_sp500_data
from .dcf_model import calculate_intrinsic_value

logger = logging.getLogger(__name__)

# Thresholds for Buy / Hold / Sell signals
BUY_THRESHOLD: float = 0.15    # Intrinsic value > current price by ≥15 %
SELL_THRESHOLD: float = -0.15  # Intrinsic value < current price by ≥15 %


def generate_signal(upside_pct: float) -> str:
    """Convert a percentage upside / downside into a Valuation Signal.

    Args:
        upside_pct: (intrinsic_value - current_price) / current_price

    Returns:
        "Buy", "Hold", or "Sell".
    """
    if upside_pct >= BUY_THRESHOLD:
        return "Buy"
    if upside_pct <= SELL_THRESHOLD:
        return "Sell"
    return "Hold"


def build_valuation_row(valuation: dict) -> Optional[dict]:
    """Turn a raw valuation dictionary into a formatted results row.

    Args:
        valuation: Output of ``dcf_model.calculate_intrinsic_value``.

    Returns:
        Formatted row dict suitable for a results DataFrame, or None when
        the intrinsic value cannot be compared meaningfully to the price.
    """
    current_price = valuation["current_price"]
    intrinsic_value = valuation["intrinsic_value"]

    if current_price <= 0:
        return None

    upside_pct = (intrinsic_value - current_price) / current_price
    signal = generate_signal(upside_pct)

    return {
        "Ticker": valuation["ticker"],
        "Company": valuation["name"],
        "Sector": valuation["sector"],
        "Current Price ($)": round(current_price, 2),
        "Intrinsic Value ($)": round(intrinsic_value, 2),
        "% Upside/Downside": round(upside_pct * 100, 1),
        "Valuation Signal": signal,
        "WACC (%)": round(valuation.get("wacc", 0) * 100, 2),
        "FCF Growth Rate (%)": round(valuation.get("fcf_growth_rate", 0) * 100, 2),
    }


def run_valuation(
    tickers: Optional[list[str]] = None,
    output_csv: Optional[str] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Run the full S&P 500 DCF valuation pipeline.

    Steps:
        1. Fetch financial data for each company.
        2. Run the DCF model to estimate intrinsic value per share.
        3. Compare to the current market price and assign a signal.
        4. Return results sorted by upside potential (descending).

    Args:
        tickers: Optional explicit list of tickers to evaluate.  If None,
            the full S&P 500 list is fetched from Wikipedia.
        output_csv: If provided, the results DataFrame is saved to this path.
        delay: Seconds to wait between successive Yahoo Finance API calls.

    Returns:
        DataFrame with columns: Ticker, Company, Sector, Current Price ($),
        Intrinsic Value ($), % Upside/Downside, Valuation Signal, WACC (%),
        FCF Growth Rate (%).
    """
    # --- 1. Fetch data ---
    logger.info("Starting data fetch for %s…",
                f"{len(tickers)} tickers" if tickers else "S&P 500")
    company_data_list = fetch_sp500_data(tickers=tickers, delay=delay)

    if not company_data_list:
        logger.warning("No company data could be fetched.")
        return pd.DataFrame()

    # --- 2. Run DCF for each company ---
    rows = []
    for company_data in company_data_list:
        valuation = calculate_intrinsic_value(company_data)
        if valuation is None:
            continue
        row = build_valuation_row(valuation)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.warning("DCF model produced no valid valuations.")
        return pd.DataFrame()

    # --- 3. Build and sort the results DataFrame ---
    df = pd.DataFrame(rows)
    df.sort_values("% Upside/Downside", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Valuation complete: %d companies analysed, %d Buy, %d Hold, %d Sell.",
        len(df),
        (df["Valuation Signal"] == "Buy").sum(),
        (df["Valuation Signal"] == "Hold").sum(),
        (df["Valuation Signal"] == "Sell").sum(),
    )

    # --- 4. Optionally save to CSV ---
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info("Results saved to %s", output_csv)

    return df


def print_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print a formatted summary table of valuation results.

    Args:
        df: Results DataFrame from ``run_valuation``.
        top_n: Number of top-upside companies to highlight.
    """
    if df.empty:
        print("No valuation results to display.")
        return

    total = len(df)
    buy_count = (df["Valuation Signal"] == "Buy").sum()
    hold_count = (df["Valuation Signal"] == "Hold").sum()
    sell_count = (df["Valuation Signal"] == "Sell").sum()

    print("\n" + "=" * 80)
    print("S&P 500 DCF VALUATION SUMMARY")
    print("=" * 80)
    print(f"Total companies analysed : {total}")
    print(f"  Buy  (≥+15% upside)     : {buy_count}")
    print(f"  Hold (−15% to +15%)     : {hold_count}")
    print(f"  Sell (≥−15% downside)   : {sell_count}")
    print("=" * 80)

    display_cols = [
        "Ticker",
        "Company",
        "Current Price ($)",
        "Intrinsic Value ($)",
        "% Upside/Downside",
        "Valuation Signal",
    ]

    # Top undervalued
    top_buy = df[df["Valuation Signal"] == "Buy"].head(top_n)
    if not top_buy.empty:
        print(f"\nTop {min(top_n, len(top_buy))} Most Undervalued (BUY signal):")
        print(top_buy[display_cols].to_string(index=False))

    # Top overvalued
    top_sell = df[df["Valuation Signal"] == "Sell"].tail(top_n)
    if not top_sell.empty:
        print(f"\nTop {min(top_n, len(top_sell))} Most Overvalued (SELL signal):")
        print(top_sell[display_cols].to_string(index=False))

    print("\nFull results (all companies, sorted by upside):")
    print(df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")
