"""Tests for temporal analysis."""

import numpy as np
import pandas as pd
import pytest

from statqa.analysis.temporal import TemporalAnalyzer
from statqa.metadata.schema import Variable, VariableType


@pytest.fixture
def year_var():
    return Variable(name="year", label="Year", var_type=VariableType.NUMERIC_DISCRETE)


@pytest.fixture
def value_var():
    return Variable(name="v", label="Value", var_type=VariableType.NUMERIC_CONTINUOUS)


@pytest.fixture
def yoy(year_var, value_var):
    # 2003 repeats 2002's value, so its change is a genuine zero rather than
    # missing -- the distinction a truthiness check would collapse.
    data = pd.DataFrame(
        {"year": [2000, 2001, 2002, 2003], "v": [100.0, 110.0, 99.0, 99.0]}
    )
    return TemporalAnalyzer().year_over_year_change(data, year_var, value_var)


def test_first_year_has_no_change(yoy):
    assert yoy["years"]["2000"]["yoy_absolute"] is None
    assert yoy["years"]["2000"]["yoy_percent"] is None


def test_subsequent_years_report_absolute_change(yoy):
    assert yoy["years"]["2001"]["yoy_absolute"] == pytest.approx(10.0)
    assert yoy["years"]["2002"]["yoy_absolute"] == pytest.approx(-11.0)


def test_subsequent_years_report_percent_change(yoy):
    assert yoy["years"]["2001"]["yoy_percent"] == pytest.approx(10.0)
    assert yoy["years"]["2002"]["yoy_percent"] == pytest.approx(-10.0)


def test_zero_change_is_zero_not_missing(yoy):
    assert yoy["years"]["2003"]["yoy_absolute"] == pytest.approx(0.0)
    assert yoy["years"]["2003"]["yoy_percent"] == pytest.approx(0.0)


def test_values_are_preserved(yoy):
    assert [yoy["years"][y]["value"] for y in sorted(yoy["years"])] == [
        100.0,
        110.0,
        99.0,
        99.0,
    ]


def test_too_few_years_returns_error(year_var, value_var):
    data = pd.DataFrame({"year": [2000], "v": [1.0]})

    result = TemporalAnalyzer().year_over_year_change(data, year_var, value_var)

    assert "error" in result


@pytest.fixture
def rising():
    """A clear upward trend with mild noise."""
    rng = np.random.default_rng(41)
    years = list(range(2000, 2024))
    return pd.DataFrame(
        {"year": years, "v": [i * 2.0 + rng.normal(0, 0.5) for i in range(len(years))]}
    )


class TestTrend:
    def test_detects_an_increasing_trend(self, rising, year_var, value_var):
        result = TemporalAnalyzer().analyze_trend(rising, year_var, value_var)

        assert result["mann_kendall"]["trend"] == "increasing"

    def test_reports_a_positive_slope(self, rising, year_var, value_var):
        result = TemporalAnalyzer().analyze_trend(rising, year_var, value_var)

        assert result["linear_trend"]["slope"] > 0

    def test_flat_series_has_no_significant_trend(self, year_var, value_var):
        data = pd.DataFrame({"year": range(2000, 2024), "v": [5.0] * 24})

        result = TemporalAnalyzer().analyze_trend(data, year_var, value_var)

        assert result["mann_kendall"]["trend"] != "increasing"

    def test_too_few_periods_returns_error(self, year_var, value_var):
        data = pd.DataFrame({"year": [2000, 2001], "v": [1.0, 2.0]})

        result = TemporalAnalyzer().analyze_trend(data, year_var, value_var)

        assert "error" in result


class TestGroupedTrend:
    def test_reports_each_group(self, year_var, value_var):
        rng = np.random.default_rng(42)
        years = list(range(2000, 2020))
        frame = pd.DataFrame(
            {
                "year": years * 2,
                "v": [i + rng.normal(0, 0.3) for i in range(len(years))]
                + [100 - i + rng.normal(0, 0.3) for i in range(len(years))],
                "g": ["up"] * len(years) + ["down"] * len(years),
            }
        )
        group_var = Variable(
            name="g", label="Group", var_type=VariableType.CATEGORICAL_NOMINAL
        )

        result = TemporalAnalyzer().analyze_grouped_trend(
            frame, year_var, value_var, group_var
        )

        assert isinstance(result, dict)
        assert result["analysis_type"]


class TestChangePoints:
    def test_finds_a_level_shift(self, year_var, value_var):
        # Flat at 0, then flat at 20: the break is in the middle.
        values = [0.0] * 15 + [20.0] * 15
        frame = pd.DataFrame({"year": range(2000, 2030), "v": values})

        result = TemporalAnalyzer().detect_change_points(frame, year_var, value_var)

        assert result["analysis_type"] == "change_point_detection"

    def test_too_few_points_returns_error(self, year_var, value_var):
        frame = pd.DataFrame({"year": [2000, 2001], "v": [1.0, 2.0]})

        result = TemporalAnalyzer().detect_change_points(frame, year_var, value_var)

        assert "error" in result
