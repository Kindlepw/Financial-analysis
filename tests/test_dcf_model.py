"""
Unit tests for the DCF model (src/dcf_model.py).
"""

import pytest

from src.dcf_model import (
    MAX_INITIAL_GROWTH,
    MIN_INITIAL_GROWTH,
    PROJECTION_YEARS,
    _clamp,
    calculate_historical_growth_rate,
    calculate_intrinsic_value,
    calculate_terminal_value,
    calculate_wacc,
    discount_cash_flows,
    project_free_cash_flows,
)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_value_below_low(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_value_above_high(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_value_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_boundary_values(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# calculate_historical_growth_rate
# ---------------------------------------------------------------------------
class TestCalculateHistoricalGrowthRate:
    def test_single_value_returns_default(self):
        rate = calculate_historical_growth_rate([1_000_000])
        assert rate == 0.05

    def test_empty_list_returns_default(self):
        rate = calculate_historical_growth_rate([])
        assert rate == 0.05

    def test_positive_growth_cagr(self):
        # FCF doubles from first to last value over 4 periods (5 data points)
        fcf = [100, 120, 145, 175, 200]
        rate = calculate_historical_growth_rate(fcf)
        expected_cagr = (200 / 100) ** (1 / 4) - 1
        assert abs(rate - expected_cagr) < 0.001

    def test_growth_rate_clamped_to_max(self):
        # Very fast growth should be capped
        fcf = [1, 1_000_000]
        rate = calculate_historical_growth_rate(fcf)
        assert rate <= MAX_INITIAL_GROWTH

    def test_growth_rate_clamped_to_min(self):
        # Severe decline should be capped
        fcf = [1_000_000, 1]
        rate = calculate_historical_growth_rate(fcf)
        assert rate >= MIN_INITIAL_GROWTH

    def test_negative_fcf_falls_back_to_conservative(self):
        # All-negative FCF; function should return a non-explosive value
        fcf = [-100, -80, -60, -40, -20]
        rate = calculate_historical_growth_rate(fcf)
        assert MIN_INITIAL_GROWTH <= rate <= MAX_INITIAL_GROWTH


# ---------------------------------------------------------------------------
# calculate_wacc
# ---------------------------------------------------------------------------
class TestCalculateWacc:
    def test_reasonable_wacc_range(self):
        wacc = calculate_wacc(
            beta=1.0,
            market_cap=500e9,
            total_debt=100e9,
            interest_expense=3e9,
            tax_rate=0.21,
        )
        assert 0.04 <= wacc <= 0.25

    def test_zero_debt_returns_cost_of_equity(self):
        wacc = calculate_wacc(
            beta=1.0,
            market_cap=500e9,
            total_debt=0,
            interest_expense=0,
            tax_rate=0.21,
        )
        # With no debt, WACC ≈ cost of equity = Rf + beta * ERP
        from src.dcf_model import EQUITY_RISK_PREMIUM, RISK_FREE_RATE
        expected_re = RISK_FREE_RATE + 1.0 * EQUITY_RISK_PREMIUM
        assert abs(wacc - expected_re) < 0.001

    def test_high_beta_increases_wacc(self):
        wacc_low = calculate_wacc(1.0, 500e9, 50e9, 2e9, 0.21)
        wacc_high = calculate_wacc(2.0, 500e9, 50e9, 2e9, 0.21)
        assert wacc_high > wacc_low

    def test_wacc_clamped_to_max(self):
        # Extreme beta should be capped
        wacc = calculate_wacc(10.0, 1e9, 0, 0, 0.21)
        assert wacc <= 0.25


# ---------------------------------------------------------------------------
# project_free_cash_flows
# ---------------------------------------------------------------------------
class TestProjectFreeCashFlows:
    def test_returns_correct_number_of_years(self):
        projected = project_free_cash_flows(1_000_000, 0.10, years=7)
        assert len(projected) == 7

    def test_growth_moves_toward_long_term_rate(self):
        # With initial > long_term growth, later years should grow slower
        projected = project_free_cash_flows(1_000_000, 0.20, years=7,
                                            long_term_growth_rate=0.025)
        yoy_rates = [projected[i] / projected[i - 1] - 1
                     for i in range(1, len(projected))]
        # Rates should be decreasing
        assert all(yoy_rates[i] >= yoy_rates[i + 1] - 0.001
                   for i in range(len(yoy_rates) - 1))

    def test_negative_base_fcf_projects_forward(self):
        projected = project_free_cash_flows(-500_000, 0.05, years=5)
        assert len(projected) == 5
        # Positive growth on negative FCF makes it more negative (deeper loss)
        assert projected[-1] < projected[0]


# ---------------------------------------------------------------------------
# calculate_terminal_value
# ---------------------------------------------------------------------------
class TestCalculateTerminalValue:
    def test_positive_terminal_fcf(self):
        tv = calculate_terminal_value(1_000_000, wacc=0.09)
        assert tv > 0

    def test_wacc_equals_growth_rate_does_not_crash(self):
        # spread approaches zero; function should clamp and return a large value
        tv = calculate_terminal_value(1_000_000, wacc=0.025,
                                      long_term_growth_rate=0.025)
        assert tv > 0

    def test_blended_result_between_two_methods(self):
        from src.dcf_model import LONG_TERM_GROWTH_RATE, FCF_EXIT_MULTIPLE
        fcf = 1_000_000
        wacc = 0.09
        spread = wacc - LONG_TERM_GROWTH_RATE
        tv_perpetuity = fcf * (1 + LONG_TERM_GROWTH_RATE) / spread
        tv_multiple = fcf * FCF_EXIT_MULTIPLE
        expected = (tv_perpetuity + tv_multiple) / 2
        tv = calculate_terminal_value(fcf, wacc)
        assert abs(tv - expected) < 1.0  # floating point tolerance


# ---------------------------------------------------------------------------
# discount_cash_flows
# ---------------------------------------------------------------------------
class TestDiscountCashFlows:
    def test_discounted_pv_less_than_nominal_sum(self):
        cfs = [1_000_000] * 7
        tv = 10_000_000
        pv = discount_cash_flows(cfs, tv, wacc=0.09)
        nominal_sum = sum(cfs) + tv
        assert pv < nominal_sum

    def test_higher_wacc_lowers_pv(self):
        cfs = [1_000_000] * 7
        tv = 10_000_000
        pv_low = discount_cash_flows(cfs, tv, wacc=0.05)
        pv_high = discount_cash_flows(cfs, tv, wacc=0.15)
        assert pv_low > pv_high


# ---------------------------------------------------------------------------
# calculate_intrinsic_value (integration)
# ---------------------------------------------------------------------------
class TestCalculateIntrinsicValue:
    def _sample_data(self, **overrides):
        base = {
            "ticker": "TEST",
            "name": "Test Corp",
            "sector": "Technology",
            "current_price": 150.0,
            "shares_outstanding": 1_000_000_000,
            "historical_fcf": [5e9, 6e9, 7e9, 8e9, 9e9],
            "beta": 1.2,
            "market_cap": 150e9,
            "total_debt": 20e9,
            "interest_expense": 1e9,
            "tax_rate": 0.21,
            "revenue": 50e9,
        }
        base.update(overrides)
        return base

    def test_returns_dict_with_required_keys(self):
        result = calculate_intrinsic_value(self._sample_data())
        assert result is not None
        required_keys = {
            "ticker", "name", "sector", "current_price",
            "intrinsic_value", "wacc", "fcf_growth_rate",
        }
        assert required_keys.issubset(result.keys())

    def test_zero_fcf_returns_none(self):
        data = self._sample_data(historical_fcf=[5e9, 0])
        result = calculate_intrinsic_value(data)
        assert result is None

    def test_empty_fcf_returns_none(self):
        result = calculate_intrinsic_value(self._sample_data(historical_fcf=[]))
        assert result is None

    def test_intrinsic_value_is_numeric(self):
        result = calculate_intrinsic_value(self._sample_data())
        assert isinstance(result["intrinsic_value"], float)

    def test_negative_base_fcf_still_produces_result(self):
        data = self._sample_data(historical_fcf=[-1e9, -0.8e9, -0.5e9])
        result = calculate_intrinsic_value(data)
        assert result is not None
