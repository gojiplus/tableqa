"""Tests for temporal analysis."""

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
