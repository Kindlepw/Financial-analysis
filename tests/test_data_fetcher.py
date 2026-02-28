"""
Unit tests for the data fetcher utilities (src/data_fetcher.py).

Network calls are mocked so these tests run without internet access.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetcher import (
    _estimate_tax_rate,
    _extract_free_cash_flows,
    _extract_interest_expense,
    _find_row,
)


# ---------------------------------------------------------------------------
# _find_row
# ---------------------------------------------------------------------------
class TestFindRow:
    def _make_df(self, index_labels):
        return pd.DataFrame(
            [[1] * 3] * len(index_labels),
            index=index_labels,
            columns=["2023", "2022", "2021"],
        )

    def test_finds_first_matching_label(self):
        df = self._make_df(["Operating Cash Flow", "Capital Expenditure"])
        row = _find_row(df, ["Operating Cash Flow", "Cash From Operations"])
        assert row is not None
        assert row.name == "Operating Cash Flow"

    def test_returns_none_when_no_match(self):
        df = self._make_df(["Revenue", "Net Income"])
        row = _find_row(df, ["Operating Cash Flow"])
        assert row is None

    def test_uses_second_label_when_first_missing(self):
        df = self._make_df(["Total Cash From Operating Activities"])
        row = _find_row(df, ["Operating Cash Flow",
                              "Total Cash From Operating Activities"])
        assert row is not None


# ---------------------------------------------------------------------------
# _extract_free_cash_flows
# ---------------------------------------------------------------------------
class TestExtractFreeCashFlows:
    def _make_cashflow_df(self, ocf_values, capex_values=None):
        """Build a DataFrame that mimics yfinance's cashflow structure.

        Columns are dates (newest first); rows are line items.
        """
        col_count = len(ocf_values)
        columns = [f"2023-{i:02d}-01" for i in range(col_count, 0, -1)]
        data = {"Operating Cash Flow": ocf_values[::-1]}  # reverse → newest first
        if capex_values is not None:
            data["Capital Expenditure"] = capex_values[::-1]
        return pd.DataFrame(data, index=columns).T

    def test_fcf_is_ocf_minus_capex(self):
        df = self._make_cashflow_df(
            ocf_values=[100, 110, 120],
            capex_values=[-20, -22, -24],  # negative as yfinance reports them
        )
        fcf = _extract_free_cash_flows(df)
        assert len(fcf) == 3
        # Oldest first: FCF = OCF - |capex|
        assert abs(fcf[0] - (100 - 20)) < 0.01
        assert abs(fcf[1] - (110 - 22)) < 0.01
        assert abs(fcf[2] - (120 - 24)) < 0.01

    def test_falls_back_to_ocf_when_no_capex(self):
        df = self._make_cashflow_df(ocf_values=[100, 110, 120])
        fcf = _extract_free_cash_flows(df)
        assert fcf == [100.0, 110.0, 120.0]

    def test_empty_dataframe_returns_empty_list(self):
        fcf = _extract_free_cash_flows(pd.DataFrame())
        assert fcf == []


# ---------------------------------------------------------------------------
# _extract_interest_expense
# ---------------------------------------------------------------------------
class TestExtractInterestExpense:
    def test_returns_absolute_value(self):
        df = pd.DataFrame(
            [[-500_000_000]],
            index=["Interest Expense"],
            columns=["2023"],
        )
        result = _extract_interest_expense(df)
        assert result == 500_000_000.0

    def test_none_input_returns_zero(self):
        assert _extract_interest_expense(None) == 0.0

    def test_empty_df_returns_zero(self):
        assert _extract_interest_expense(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------------------
# _estimate_tax_rate
# ---------------------------------------------------------------------------
class TestEstimateTaxRate:
    def test_standard_tax_rate(self):
        df = pd.DataFrame(
            [[21_000_000], [100_000_000]],
            index=["Tax Provision", "Pretax Income"],
            columns=["2023"],
        )
        rate = _estimate_tax_rate(df)
        assert abs(rate - 0.21) < 0.001

    def test_none_returns_default(self):
        assert _estimate_tax_rate(None) == 0.21

    def test_empty_df_returns_default(self):
        assert _estimate_tax_rate(pd.DataFrame()) == 0.21

    def test_clamped_to_max(self):
        df = pd.DataFrame(
            [[90_000_000], [100_000_000]],
            index=["Tax Provision", "Pretax Income"],
            columns=["2023"],
        )
        rate = _estimate_tax_rate(df)
        assert rate <= 0.40

    def test_clamped_to_min(self):
        df = pd.DataFrame(
            [[1_000_000], [100_000_000]],
            index=["Tax Provision", "Pretax Income"],
            columns=["2023"],
        )
        rate = _estimate_tax_rate(df)
        assert rate >= 0.10
