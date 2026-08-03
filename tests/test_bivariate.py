"""Tests for two-variable analysis."""

import json

import numpy as np
import pandas as pd
import pytest

from statqa.analysis.bivariate import BivariateAnalyzer
from tests.factories import categorical_var as cat
from tests.factories import numeric_var as num


@pytest.fixture
def analyzer():
    return BivariateAnalyzer()


@pytest.fixture
def linear_data():
    """y is x plus small noise, so the correlation is strong and positive."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 200)
    return pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 1, 200)})


class TestNumericNumeric:
    def test_reports_correlation_type(self, analyzer, linear_data):
        result = analyzer.analyze(linear_data, num("x"), num("y"))

        assert result["analysis_type"] == "numeric_numeric"
        assert result["n"] == 200

    def test_detects_strong_positive_correlation(self, analyzer, linear_data):
        result = analyzer.analyze(linear_data, num("x"), num("y"))

        assert result["pearson"]["r"] == pytest.approx(1.0, abs=0.01)
        assert result["pearson"]["significant"] is True
        assert result["strength"] == "very strong"

    def test_detects_negative_correlation(self, analyzer):
        x = np.linspace(0, 100, 200)
        data = pd.DataFrame({"x": x, "y": -x})

        result = analyzer.analyze(data, num("x"), num("y"))

        assert result["pearson"]["r"] == pytest.approx(-1.0, abs=0.01)

    def test_includes_spearman_when_robust(self, analyzer, linear_data):
        result = analyzer.analyze(linear_data, num("x"), num("y"))

        assert result["spearman"]["rho"] == pytest.approx(1.0, abs=0.01)

    def test_omits_spearman_when_not_robust(self, linear_data):
        result = BivariateAnalyzer(use_robust=False).analyze(
            linear_data, num("x"), num("y")
        )

        assert "spearman" not in result

    def test_uncorrelated_data_is_not_significant(self, analyzer):
        rng = np.random.default_rng(7)
        data = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})

        result = analyzer.analyze(data, num("x"), num("y"))

        assert result["pearson"]["significant"] is False

    def test_too_few_rows_returns_none(self, analyzer):
        data = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})

        assert analyzer.analyze(data, num("x"), num("y")) is None

    def test_rows_with_missing_values_are_dropped(self, analyzer):
        rng = np.random.default_rng(1)
        x = list(rng.normal(size=50)) + [np.nan] * 5
        y = list(rng.normal(size=50)) + [1.0] * 5
        data = pd.DataFrame({"x": x, "y": y})

        result = analyzer.analyze(data, num("x"), num("y"))

        assert result["n"] == 50


class TestCategoricalCategorical:
    def test_reports_chi_square(self, analyzer):
        # Perfectly associated: g1 determines g2.
        data = pd.DataFrame({"g1": [1, 1, 2, 2] * 25, "g2": ["a", "a", "b", "b"] * 25})

        result = analyzer.analyze(data, cat("g1"), cat("g2"))

        assert result["analysis_type"] == "categorical_categorical"
        assert result["chi_square"]["significant"] is True
        assert result["chi_square"]["dof"] == 1

    def test_independent_categories_are_not_significant(self, analyzer):
        rng = np.random.default_rng(3)
        data = pd.DataFrame(
            {
                "g1": rng.choice([1, 2], size=400),
                "g2": rng.choice(["a", "b"], size=400),
            }
        )

        result = analyzer.analyze(data, cat("g1"), cat("g2"))

        assert result["chi_square"]["significant"] is False

    def test_too_few_rows_returns_none(self, analyzer):
        data = pd.DataFrame({"g1": [1, 2], "g2": ["a", "b"]})

        assert analyzer.analyze(data, cat("g1"), cat("g2")) is None


class TestCategoricalNumeric:
    def test_two_groups_use_t_test(self, analyzer):
        rng = np.random.default_rng(5)
        data = pd.DataFrame(
            {
                "g": [1] * 100 + [2] * 100,
                "v": list(rng.normal(0, 1, 100)) + list(rng.normal(5, 1, 100)),
            }
        )

        result = analyzer.analyze(data, cat("g"), num("v"))

        assert result["analysis_type"] == "categorical_numeric"
        assert result["t_test"]["significant"] is True
        assert "anova" not in result

    def test_three_groups_use_anova(self, analyzer):
        rng = np.random.default_rng(6)
        data = pd.DataFrame(
            {
                "g": [1] * 60 + [2] * 60 + [3] * 60,
                "v": (
                    list(rng.normal(0, 1, 60))
                    + list(rng.normal(4, 1, 60))
                    + list(rng.normal(8, 1, 60))
                ),
            }
        )

        result = analyzer.analyze(data, cat("g"), num("v"))

        assert result["anova"]["significant"] is True
        assert "t_test" not in result

    def test_argument_order_does_not_matter(self, analyzer):
        rng = np.random.default_rng(5)
        data = pd.DataFrame(
            {
                "g": [1] * 60 + [2] * 60,
                "v": list(rng.normal(0, 1, 60)) + list(rng.normal(3, 1, 60)),
            }
        )

        forward = analyzer.analyze(data, cat("g"), num("v"))
        reversed_ = analyzer.analyze(data, num("v"), cat("g"))

        assert forward["analysis_type"] == reversed_["analysis_type"]


class TestInterpretationHelpers:
    @pytest.mark.parametrize(
        ("r", "expected"),
        [
            (0.05, "negligible"),
            (0.2, "weak"),
            (0.4, "moderate"),
            (0.6, "strong"),
            (0.8, "very strong"),
        ],
    )
    def test_correlation_strength_labels(self, analyzer, r, expected):
        assert analyzer._interpret_correlation(r) == expected

    @pytest.mark.parametrize(
        ("d", "expected"),
        [(0.1, "negligible"), (0.3, "small"), (0.6, "medium"), (1.2, "large")],
    )
    def test_cohens_d_labels(self, analyzer, d, expected):
        assert analyzer._interpret_cohens_d(d) == expected


class TestBatchAnalyze:
    def test_analyzes_every_pair(self, analyzer, linear_data):
        variables = {"x": num("x"), "y": num("y")}

        results = analyzer.batch_analyze(linear_data, variables)

        assert len(results) == 1

    def test_max_pairs_caps_output(self, analyzer):
        rng = np.random.default_rng(2)
        data = pd.DataFrame({c: rng.normal(size=100) for c in "abcd"})
        variables = {c: num(c) for c in "abcd"}

        results = analyzer.batch_analyze(data, variables, max_pairs=2)

        assert len(results) == 2


class TestPerfectCorrelation:
    """A derived or unit-converted column correlates perfectly with its source."""

    @pytest.fixture
    def perfect(self, analyzer):
        x = np.linspace(0, 100, 50)
        return analyzer.analyze(pd.DataFrame({"x": x, "y": x * 3}), num("x"), num("y"))

    def test_correlation_is_still_reported(self, perfect):
        assert perfect["pearson"]["r"] == pytest.approx(1.0)

    def test_effect_size_is_omitted_rather_than_infinite(self, perfect):
        # Cohen's d diverges at |r| = 1; an inf here serializes as `Infinity`,
        # which is not valid JSON for any downstream consumer.
        assert "effect_size" not in perfect

    def test_result_is_valid_json(self, perfect):
        assert "Infinity" not in json.dumps(perfect)
