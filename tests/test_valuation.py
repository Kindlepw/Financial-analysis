"""
Unit tests for the valuation engine (src/valuation.py).
"""

import pandas as pd
import pytest

from src.valuation import BUY_THRESHOLD, SELL_THRESHOLD, build_valuation_row, generate_signal


# ---------------------------------------------------------------------------
# generate_signal
# ---------------------------------------------------------------------------
class TestGenerateSignal:
    def test_buy_above_threshold(self):
        assert generate_signal(BUY_THRESHOLD) == "Buy"
        assert generate_signal(BUY_THRESHOLD + 0.10) == "Buy"

    def test_sell_below_threshold(self):
        assert generate_signal(SELL_THRESHOLD) == "Sell"
        assert generate_signal(SELL_THRESHOLD - 0.10) == "Sell"

    def test_hold_in_middle(self):
        assert generate_signal(0.0) == "Hold"
        assert generate_signal(0.05) == "Hold"
        assert generate_signal(-0.05) == "Hold"

    def test_boundary_just_below_buy(self):
        assert generate_signal(BUY_THRESHOLD - 0.001) == "Hold"

    def test_boundary_just_above_sell(self):
        assert generate_signal(SELL_THRESHOLD + 0.001) == "Hold"


# ---------------------------------------------------------------------------
# build_valuation_row
# ---------------------------------------------------------------------------
class TestBuildValuationRow:
    def _sample_valuation(self, **overrides):
        base = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "current_price": 180.0,
            "intrinsic_value": 210.0,
            "wacc": 0.09,
            "fcf_growth_rate": 0.10,
            "base_fcf": 90_000_000_000.0,
            "shares_outstanding": 15_500_000_000,
        }
        base.update(overrides)
        return base

    def test_buy_signal_when_undervalued(self):
        row = build_valuation_row(self._sample_valuation(
            current_price=100.0, intrinsic_value=200.0
        ))
        assert row["Valuation Signal"] == "Buy"

    def test_sell_signal_when_overvalued(self):
        row = build_valuation_row(self._sample_valuation(
            current_price=200.0, intrinsic_value=100.0
        ))
        assert row["Valuation Signal"] == "Sell"

    def test_hold_signal_when_fairly_valued(self):
        row = build_valuation_row(self._sample_valuation(
            current_price=100.0, intrinsic_value=105.0
        ))
        assert row["Valuation Signal"] == "Hold"

    def test_upside_pct_calculated_correctly(self):
        row = build_valuation_row(self._sample_valuation(
            current_price=100.0, intrinsic_value=150.0
        ))
        assert abs(row["% Upside/Downside"] - 50.0) < 0.01

    def test_zero_current_price_returns_none(self):
        row = build_valuation_row(self._sample_valuation(current_price=0.0))
        assert row is None

    def test_row_contains_required_columns(self):
        row = build_valuation_row(self._sample_valuation())
        required = {
            "Ticker", "Company", "Sector", "Current Price ($)",
            "Intrinsic Value ($)", "% Upside/Downside", "Valuation Signal",
        }
        assert required.issubset(row.keys())

    def test_ticker_matches(self):
        row = build_valuation_row(self._sample_valuation(ticker="MSFT"))
        assert row["Ticker"] == "MSFT"
