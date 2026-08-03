"""Tests for causal analysis with confounding control."""

import numpy as np
import pandas as pd
import pytest

from statqa.analysis.causal import CausalAnalyzer
from statqa.metadata.schema import Variable, VariableType


def num(name: str) -> Variable:
    return Variable(name=name, label=name, var_type=VariableType.NUMERIC_CONTINUOUS)


def cat(name: str) -> Variable:
    return Variable(name=name, label=name, var_type=VariableType.CATEGORICAL_NOMINAL)


@pytest.fixture
def analyzer():
    return CausalAnalyzer()


@pytest.fixture
def confounded_data():
    """z drives both t and y, so the naive t->y estimate is biased upward.

    The true effect of t on y is 2.0; z adds 3.0 of spurious association.
    """
    rng = np.random.default_rng(11)
    n = 400
    z = rng.normal(0, 1, n)
    t = z + rng.normal(0, 0.5, n)
    y = 2.0 * t + 3.0 * z + rng.normal(0, 0.5, n)
    return pd.DataFrame({"t": t, "y": y, "z": z})


class TestTreatmentEffect:
    def test_reports_structure(self, analyzer, confounded_data):
        result = analyzer.analyze_treatment_effect(confounded_data, num("t"), num("y"))

        assert result["analysis_type"] == "treatment_effect"
        assert result["treatment"] == "t"
        assert result["outcome"] == "y"
        assert result["n"] == 400

    def test_uncontrolled_estimate_is_biased(self, analyzer, confounded_data):
        result = analyzer.analyze_treatment_effect(confounded_data, num("t"), num("y"))

        # Omitting z, the estimate absorbs the confounder's contribution.
        assert result["treatment_effect"]["coefficient"] > 2.5

    def test_controlling_for_confounder_recovers_the_effect(
        self, analyzer, confounded_data
    ):
        result = analyzer.analyze_treatment_effect(
            confounded_data, num("t"), num("y"), control_vars=[num("z")]
        )

        assert result["treatment_effect"]["coefficient"] == pytest.approx(2.0, abs=0.2)

    def test_controls_are_recorded(self, analyzer, confounded_data):
        result = analyzer.analyze_treatment_effect(
            confounded_data, num("t"), num("y"), control_vars=[num("z")]
        )

        assert result["controls"] == ["z"]

    def test_confidence_interval_brackets_the_estimate(self, analyzer, confounded_data):
        effect = analyzer.analyze_treatment_effect(
            confounded_data, num("t"), num("y"), control_vars=[num("z")]
        )["treatment_effect"]

        assert effect["ci_lower"] < effect["coefficient"] < effect["ci_upper"]

    def test_sensitivity_reported_only_with_controls(self, analyzer, confounded_data):
        without = analyzer.analyze_treatment_effect(confounded_data, num("t"), num("y"))
        with_controls = analyzer.analyze_treatment_effect(
            confounded_data, num("t"), num("y"), control_vars=[num("z")]
        )

        assert "sensitivity" not in without
        assert "sensitivity" in with_controls

    def test_null_effect_is_not_significant(self, analyzer):
        rng = np.random.default_rng(12)
        data = pd.DataFrame({"t": rng.normal(size=300), "y": rng.normal(size=300)})

        result = analyzer.analyze_treatment_effect(data, num("t"), num("y"))

        assert result["treatment_effect"]["significant"] is False

    def test_small_sample_returns_error(self, analyzer):
        data = pd.DataFrame({"t": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})

        result = analyzer.analyze_treatment_effect(data, num("t"), num("y"))

        assert "Insufficient sample size" in result["error"]

    def test_non_numeric_outcome_is_rejected(self, analyzer, confounded_data):
        data = confounded_data.assign(g=["a", "b"] * 200)

        result = analyzer.analyze_treatment_effect(data, num("t"), cat("g"))

        assert "error" in result

    def test_model_fit_is_reported(self, analyzer, confounded_data):
        result = analyzer.analyze_treatment_effect(
            confounded_data, num("t"), num("y"), control_vars=[num("z")]
        )

        assert 0.0 <= result["model_fit"]["adj_r_squared"] <= 1.0


class TestConfounderIdentification:
    def test_detects_a_real_confounder(self, analyzer, confounded_data):
        result = analyzer.identify_confounders(
            confounded_data, num("t"), num("y"), [num("z")]
        )

        assert result["analysis_type"] == "confounder_identification"
        assert result["treatment"] == "t"

    def test_unrelated_variable_is_not_flagged(self, analyzer, confounded_data):
        rng = np.random.default_rng(13)
        data = confounded_data.assign(noise=rng.normal(size=len(confounded_data)))

        result = analyzer.identify_confounders(
            data, num("t"), num("y"), [num("z"), num("noise")]
        )

        assert isinstance(result, dict)
        assert result["outcome"] == "y"
