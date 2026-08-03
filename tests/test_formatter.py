"""Tests for natural-language formatting of analysis results."""

import pytest

from statqa.interpretation.formatter import InsightFormatter


@pytest.fixture
def formatter():
    return InsightFormatter()


# Every analyzer returns a bare {"error": ...} dict on its insufficient-data
# paths (e.g. TemporalAnalyzer returns {"error": "Insufficient time periods"}
# when there are fewer than min_periods), so the formatters have to survive one.
@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("format_univariate", "Insufficient data"),
        ("format_bivariate", "Insufficient data"),
        ("format_temporal", "Insufficient time periods"),
        ("format_causal", "Insufficient data"),
    ],
)
def test_error_result_is_surfaced_as_text(formatter, method, message):
    text = getattr(formatter, method)({"error": message})

    assert message in text


@pytest.mark.parametrize(
    "method",
    ["format_univariate", "format_bivariate", "format_temporal", "format_causal"],
)
def test_empty_result_does_not_raise(formatter, method):
    assert isinstance(getattr(formatter, method)({}), str)


def test_format_temporal_still_reports_a_trend(formatter):
    text = formatter.format_temporal(
        {
            "value_variable": "turnout",
            "mann_kendall": {"trend": "increasing", "tau": 0.42, "p_value": 0.01},
        }
    )

    assert "turnout" in text
    assert "increasing trend" in text


def test_format_temporal_reports_change_without_trend_test(formatter):
    text = formatter.format_temporal(
        {
            "value_variable": "turnout",
            "change_metrics": {"absolute_change": 5.0, "percent_change": 12.5},
        }
    )

    assert "turnout" in text
    assert "5.00" in text
