"""
Discounted Cash Flow (DCF) model for company valuation.

Methodology:
- Free cash flow projection over a configurable horizon (default: 7 years)
  using historical compound annual growth rates (CAGR) and global growth caps.
- Terminal value calculated with both the Gordon Growth Model (perpetuity
  growth) and an exit EBITDA/FCF multiple; the average of the two is used.
- Weighted Average Cost of Capital (WACC) estimated from CAPM cost of equity
  and after-tax cost of debt, weighted by market-value capital structure.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# --- Macro assumptions (can be overridden) ---
RISK_FREE_RATE: float = 0.045        # 10-year US Treasury yield (~4.5 %)
EQUITY_RISK_PREMIUM: float = 0.055   # Historical US ERP
LONG_TERM_GROWTH_RATE: float = 0.025 # Terminal perpetuity growth rate (~GDP)
FCF_EXIT_MULTIPLE: float = 20.0      # FCF exit multiple for terminal value
PROJECTION_YEARS: int = 7            # Default projection horizon (configurable)

# Growth rate caps to keep projections realistic
MAX_INITIAL_GROWTH: float = 0.30     # +30 % upper bound
MIN_INITIAL_GROWTH: float = -0.20    # −20 % lower bound


def calculate_historical_growth_rate(historical_fcf: list[float]) -> float:
    """Compute the CAGR of free cash flow from historical data.

    Only uses periods where both the starting and ending FCF are positive
    (negative FCF CAGR is hard to interpret and often misleading).  Falls back
    to a simple average year-over-year growth rate when CAGR is not applicable.

    Args:
        historical_fcf: List of annual FCF values, oldest first.

    Returns:
        Estimated annual FCF growth rate, clamped to [MIN, MAX] bounds.
    """
    if len(historical_fcf) < 2:
        return 0.05  # default 5 %

    # Filter to a maximum of the last 5 years
    fcf = historical_fcf[-5:]

    # Attempt CAGR between first and last value if both positive
    first, last = fcf[0], fcf[-1]
    n = len(fcf) - 1
    if first > 0 and last > 0 and n > 0:
        cagr = (last / first) ** (1.0 / n) - 1.0
        return _clamp(cagr, MIN_INITIAL_GROWTH, MAX_INITIAL_GROWTH)

    # Fall back to average of year-over-year growth rates (skip zero-crossing)
    yoy_rates = []
    for prev, curr in zip(fcf[:-1], fcf[1:]):
        if prev > 0 and curr > 0:
            yoy_rates.append(curr / prev - 1.0)

    if yoy_rates:
        avg_rate = sum(yoy_rates) / len(yoy_rates)
        return _clamp(avg_rate, MIN_INITIAL_GROWTH, MAX_INITIAL_GROWTH)

    return 0.03  # conservative default when FCF history is mostly negative


def calculate_wacc(
    beta: float,
    market_cap: float,
    total_debt: float,
    interest_expense: float,
    tax_rate: float,
    risk_free_rate: float = RISK_FREE_RATE,
    equity_risk_premium: float = EQUITY_RISK_PREMIUM,
) -> float:
    """Estimate the Weighted Average Cost of Capital (WACC).

    WACC = (E/V) × Re + (D/V) × Rd × (1 − t)

    where:
        Re  = risk-free rate + beta × equity risk premium  (CAPM)
        Rd  = interest expense / total debt  (or a market proxy)
        E/V = equity weight in total capital
        D/V = debt weight in total capital
        t   = effective corporate tax rate

    Args:
        beta: Company beta (systematic risk relative to market).
        market_cap: Market capitalisation (equity value) in USD.
        total_debt: Book value of interest-bearing debt in USD.
        interest_expense: Annual interest paid in USD.
        tax_rate: Effective corporate tax rate (0–1).
        risk_free_rate: Current risk-free rate (default: 10-yr Treasury).
        equity_risk_premium: Expected market risk premium above risk-free rate.

    Returns:
        WACC as a decimal (e.g. 0.09 for 9 %).
    """
    # Cost of equity via CAPM
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Cost of debt
    if total_debt > 0 and interest_expense > 0:
        cost_of_debt = interest_expense / total_debt
        # Clamp to a reasonable range (1 %–20 %)
        cost_of_debt = _clamp(cost_of_debt, 0.01, 0.20)
    else:
        # Default to risk-free rate + 150 bps spread
        cost_of_debt = risk_free_rate + 0.015

    after_tax_cost_of_debt = cost_of_debt * (1.0 - tax_rate)

    total_capital = market_cap + total_debt
    if total_capital <= 0:
        return cost_of_equity  # edge case: no debt data

    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital

    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt

    # Clamp WACC to a plausible range (4 %–25 %)
    return _clamp(wacc, 0.04, 0.25)


def project_free_cash_flows(
    base_fcf: float,
    initial_growth_rate: float,
    years: int = PROJECTION_YEARS,
    long_term_growth_rate: float = LONG_TERM_GROWTH_RATE,
) -> list[float]:
    """Project annual free cash flows over the forecast horizon.

    Growth fades linearly from *initial_growth_rate* in year 1 toward
    *long_term_growth_rate* by the final projection year, producing a
    two-stage DCF profile that reflects the typical corporate life cycle.

    Args:
        base_fcf: Most recent annual free cash flow (Year 0).
        initial_growth_rate: Near-term FCF growth rate.
        years: Number of projection years.
        long_term_growth_rate: Terminal growth rate to fade toward.

    Returns:
        List of projected FCF values for years 1 through *years*.
    """
    projected = []
    current_fcf = base_fcf
    for t in range(1, years + 1):
        # Linear fade from initial to long-term growth rate
        fade_factor = (t - 1) / max(years - 1, 1)
        growth = initial_growth_rate + fade_factor * (
            long_term_growth_rate - initial_growth_rate
        )
        current_fcf = current_fcf * (1.0 + growth)
        projected.append(current_fcf)
    return projected


def calculate_terminal_value(
    terminal_fcf: float,
    wacc: float,
    long_term_growth_rate: float = LONG_TERM_GROWTH_RATE,
    exit_multiple: float = FCF_EXIT_MULTIPLE,
) -> float:
    """Estimate terminal value using perpetuity growth and exit multiple methods.

    The two approaches are averaged to reduce model dependence on any single
    assumption.

    Args:
        terminal_fcf: FCF in the final projection year.
        wacc: Weighted average cost of capital.
        long_term_growth_rate: Perpetuity growth rate for Gordon Growth Model.
        exit_multiple: FCF exit multiple for the second method.

    Returns:
        Blended terminal value (average of the two methods) in USD.
    """
    # Method 1: Gordon Growth Model (perpetuity growth)
    spread = wacc - long_term_growth_rate
    if spread <= 0.001:
        spread = 0.001  # prevent division by near-zero
    tv_perpetuity = terminal_fcf * (1.0 + long_term_growth_rate) / spread

    # Method 2: Exit multiple
    tv_exit_multiple = terminal_fcf * exit_multiple

    return (tv_perpetuity + tv_exit_multiple) / 2.0


def discount_cash_flows(
    cash_flows: list[float],
    terminal_value: float,
    wacc: float,
) -> float:
    """Discount projected cash flows and terminal value back to present value.

    Args:
        cash_flows: Projected annual FCF values for years 1 through N.
        terminal_value: Terminal value at end of year N.
        wacc: Discount rate.

    Returns:
        Sum of all present values (enterprise value proxy).
    """
    pv_sum = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        pv_sum += cf / (1.0 + wacc) ** t

    # Discount terminal value to present
    n = len(cash_flows)
    pv_sum += terminal_value / (1.0 + wacc) ** n
    return pv_sum


def calculate_intrinsic_value(company_data: dict) -> Optional[dict]:
    """Run the full DCF model for a single company.

    Steps:
        1. Estimate FCF growth rate from historical data.
        2. Compute WACC from capital structure and beta.
        3. Project free cash flows over the forecast horizon.
        4. Compute terminal value (blended perpetuity + exit multiple).
        5. Discount all cash flows to derive equity value per share.

    Args:
        company_data: Dictionary returned by ``data_fetcher.fetch_company_data``.

    Returns:
        Dictionary with valuation results, or None if the model cannot run.
    """
    ticker = company_data.get("ticker", "UNKNOWN")
    historical_fcf = company_data.get("historical_fcf", [])

    if not historical_fcf:
        logger.debug("%s: no FCF data available for DCF.", ticker)
        return None

    base_fcf = historical_fcf[-1]
    if base_fcf == 0:
        logger.debug("%s: most recent FCF is zero; cannot project.", ticker)
        return None

    # --- Step 1: Growth rate ---
    growth_rate = calculate_historical_growth_rate(historical_fcf)

    # For negative base FCF, use a conservative recovery assumption
    if base_fcf < 0:
        logger.debug(
            "%s: most recent FCF is negative (%.0f); applying recovery assumption.",
            ticker,
            base_fcf,
        )
        growth_rate = max(growth_rate, 0.05)

    # --- Step 2: WACC ---
    wacc = calculate_wacc(
        beta=company_data["beta"],
        market_cap=company_data["market_cap"],
        total_debt=company_data["total_debt"],
        interest_expense=company_data["interest_expense"],
        tax_rate=company_data["tax_rate"],
    )

    # --- Step 3: Project FCF ---
    projected_fcf = project_free_cash_flows(
        base_fcf=base_fcf,
        initial_growth_rate=growth_rate,
        years=PROJECTION_YEARS,
    )

    # --- Step 4: Terminal value ---
    terminal_value = calculate_terminal_value(
        terminal_fcf=projected_fcf[-1],
        wacc=wacc,
    )

    # --- Step 5: Present value (enterprise value proxy) ---
    equity_value = discount_cash_flows(projected_fcf, terminal_value, wacc)

    # Adjust for net debt to get equity value
    net_debt = company_data["total_debt"]
    equity_value_adjusted = equity_value - net_debt

    shares = company_data["shares_outstanding"]
    if shares <= 0:
        return None

    intrinsic_value_per_share = equity_value_adjusted / shares

    # Negative intrinsic value → company is deeply distressed; still report it
    return {
        "ticker": ticker,
        "name": company_data.get("name", ticker),
        "sector": company_data.get("sector", "Unknown"),
        "current_price": company_data["current_price"],
        "intrinsic_value": round(intrinsic_value_per_share, 2),
        "wacc": round(wacc, 4),
        "fcf_growth_rate": round(growth_rate, 4),
        "base_fcf": round(base_fcf, 0),
        "shares_outstanding": shares,
    }


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the interval [*low*, *high*]."""
    return max(low, min(high, value))
