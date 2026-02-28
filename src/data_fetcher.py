"""
Data fetcher for S&P 500 companies.

Sources:
- S&P 500 tickers: Wikipedia (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- Financial data: Yahoo Finance via yfinance
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
REQUEST_DELAY_SECONDS = 0.5


def get_sp500_tickers() -> list[str]:
    """Fetch the current list of S&P 500 tickers from Wikipedia.

    Returns:
        List of ticker symbols for all S&P 500 companies.

    Raises:
        RuntimeError: If the Wikipedia page cannot be fetched or parsed.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; FinancialAnalysisBot/1.0)"
            )
        }
        response = requests.get(SP500_WIKIPEDIA_URL, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch S&P 500 tickers from Wikipedia: {exc}"
        ) from exc

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError(
            "Could not locate the S&P 500 constituents table on Wikipedia."
        )

    tickers = []
    for row in table.find_all("tr")[1:]:  # skip header
        cells = row.find_all("td")
        if cells:
            ticker = cells[0].get_text(strip=True).replace(".", "-")
            tickers.append(ticker)

    if not tickers:
        raise RuntimeError("No tickers were parsed from the Wikipedia table.")

    logger.info("Fetched %d S&P 500 tickers from Wikipedia.", len(tickers))
    return tickers


def fetch_company_data(ticker: str) -> Optional[dict]:
    """Fetch financial data for a single company from Yahoo Finance.

    Retrieves:
    - Historical free cash flows (operating cash flow minus capital expenditures)
    - Current stock price
    - Shares outstanding
    - Beta (for WACC)
    - Total debt and equity market cap (for WACC)
    - Interest expense (for cost of debt)
    - Company name and sector

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        Dictionary with financial data, or None if data is insufficient.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # --- Current price ---
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if not current_price or current_price <= 0:
            logger.debug("%s: no valid current price, skipping.", ticker)
            return None

        # --- Shares outstanding ---
        shares_outstanding = info.get("sharesOutstanding") or info.get(
            "impliedSharesOutstanding"
        )
        if not shares_outstanding or shares_outstanding <= 0:
            logger.debug("%s: no valid shares outstanding, skipping.", ticker)
            return None

        # --- Historical free cash flow ---
        cash_flow_df = stock.cashflow  # annual, columns are dates
        if cash_flow_df is None or cash_flow_df.empty:
            logger.debug("%s: no cash flow data, skipping.", ticker)
            return None

        historical_fcf = _extract_free_cash_flows(cash_flow_df)
        if len(historical_fcf) < 2:
            logger.debug(
                "%s: insufficient FCF history (%d years), skipping.",
                ticker,
                len(historical_fcf),
            )
            return None

        # --- WACC inputs ---
        beta = info.get("beta") or 1.0
        if beta <= 0:
            beta = 1.0

        market_cap = info.get("marketCap") or (current_price * shares_outstanding)
        total_debt = info.get("totalDebt") or 0.0

        # Interest expense for cost of debt calculation
        income_stmt = stock.income_stmt
        interest_expense = _extract_interest_expense(income_stmt)

        # Effective tax rate
        tax_rate = _estimate_tax_rate(income_stmt)

        # Revenue for exit multiple terminal value
        revenue = info.get("totalRevenue") or 0.0

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "current_price": float(current_price),
            "shares_outstanding": float(shares_outstanding),
            "historical_fcf": historical_fcf,
            "beta": float(beta),
            "market_cap": float(market_cap),
            "total_debt": float(total_debt),
            "interest_expense": float(interest_expense),
            "tax_rate": float(tax_rate),
            "revenue": float(revenue),
        }

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("%s: error fetching data – %s", ticker, exc)
        return None


def _extract_free_cash_flows(cash_flow_df: pd.DataFrame) -> list[float]:
    """Extract annual free cash flow values from a yfinance cash flow DataFrame.

    Free cash flow = Operating Cash Flow − Capital Expenditures.

    Args:
        cash_flow_df: DataFrame returned by yf.Ticker.cashflow (rows are line
            items, columns are annual dates, most recent first).

    Returns:
        List of annual FCF values, oldest first, with NaN years dropped.
    """
    # Row labels vary slightly across yfinance versions; try common variants
    ocf_labels = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash From Operating Activities",
    ]
    capex_labels = [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
        "Purchases Of Property Plant And Equipment",
    ]

    ocf_row = _find_row(cash_flow_df, ocf_labels)
    capex_row = _find_row(cash_flow_df, capex_labels)

    if ocf_row is None:
        return []

    # Reverse so that values are in chronological order (oldest → newest)
    ocf_values = ocf_row.dropna().iloc[::-1].tolist()

    if capex_row is not None:
        capex_values = capex_row.dropna().iloc[::-1].tolist()
        # CapEx is usually negative in yfinance; subtract to compute FCF
        fcf = [
            ocf - abs(capex)
            for ocf, capex in zip(ocf_values, capex_values)
        ]
    else:
        # Fall back to operating cash flow alone
        fcf = ocf_values

    return [float(v) for v in fcf if v is not None]


def _find_row(df: pd.DataFrame, labels: list[str]) -> Optional[pd.Series]:
    """Return the first row in *df* whose index matches one of *labels*."""
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def _extract_interest_expense(income_stmt: Optional[pd.DataFrame]) -> float:
    """Extract the most recent annual interest expense from an income statement."""
    if income_stmt is None or income_stmt.empty:
        return 0.0

    interest_labels = [
        "Interest Expense",
        "Interest Expense Non Operating",
        "Net Interest Income",
    ]
    row = _find_row(income_stmt, interest_labels)
    if row is None:
        return 0.0

    value = row.dropna().iloc[0] if not row.dropna().empty else 0.0
    return abs(float(value))


def _estimate_tax_rate(income_stmt: Optional[pd.DataFrame]) -> float:
    """Estimate effective tax rate from the income statement.

    Falls back to a standard corporate rate of 21 % if data is unavailable.
    """
    default_rate = 0.21
    if income_stmt is None or income_stmt.empty:
        return default_rate

    tax_labels = ["Tax Provision", "Income Tax Expense"]
    pretax_labels = ["Pretax Income", "Income Before Tax"]

    tax_row = _find_row(income_stmt, tax_labels)
    pretax_row = _find_row(income_stmt, pretax_labels)

    if tax_row is None or pretax_row is None:
        return default_rate

    tax = tax_row.dropna().iloc[0] if not tax_row.dropna().empty else None
    pretax = pretax_row.dropna().iloc[0] if not pretax_row.dropna().empty else None

    if tax is None or pretax is None or pretax <= 0:
        return default_rate

    rate = float(tax) / float(pretax)
    # Clamp to a sensible range
    return max(0.10, min(0.40, rate))


def fetch_sp500_data(
    tickers: Optional[list[str]] = None,
    delay: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Fetch financial data for all S&P 500 companies.

    Args:
        tickers: Optional explicit list of tickers.  If None, the list is
            fetched from Wikipedia.
        delay: Seconds to wait between API calls to avoid rate-limiting.

    Returns:
        List of company data dictionaries (only companies with sufficient data).
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        logger.info("Fetching %s (%d/%d)…", ticker, i, total)
        data = fetch_company_data(ticker)
        if data is not None:
            results.append(data)
        time.sleep(delay)

    logger.info(
        "Successfully fetched data for %d / %d companies.", len(results), total
    )
    return results
