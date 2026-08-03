"""Tests for question templates."""

import pytest

from statqa.qa.templates import QuestionTemplate, QuestionType, infer_question_type

ANSWER = "The formatted answer."


class TestInferQuestionType:
    @pytest.mark.parametrize(
        ("insight", "expected"),
        [
            ({"analysis_type": "temporal_trend"}, QuestionType.TEMPORAL),
            ({"analysis_type": "year_over_year"}, QuestionType.TEMPORAL),
            ({"mann_kendall": {}}, QuestionType.TEMPORAL),
            ({"analysis_type": "treatment_effect"}, QuestionType.CAUSAL),
            ({"treatment_effect": {}}, QuestionType.CAUSAL),
            ({"analysis_type": "numeric_numeric"}, QuestionType.CORRELATIONAL),
            ({"pearson": {}}, QuestionType.CORRELATIONAL),
            ({"analysis_type": "categorical_numeric"}, QuestionType.COMPARATIVE),
            ({"group_stats": {}}, QuestionType.COMPARATIVE),
            ({"skewness": 0.1}, QuestionType.DISTRIBUTIONAL),
            ({"frequencies": {}}, QuestionType.DISTRIBUTIONAL),
        ],
    )
    def test_routing(self, insight, expected):
        assert infer_question_type(insight) == expected

    def test_unrecognised_insight_falls_back(self):
        assert isinstance(infer_question_type({}), QuestionType)


class TestDescriptive:
    def test_mean_produces_questions(self):
        template = QuestionTemplate(QuestionType.DESCRIPTIVE)

        pairs = template.generate({"label": "Age", "mean": 50.0}, ANSWER)

        assert pairs
        assert any("average" in p["question"].lower() for p in pairs)

    def test_mode_produces_questions(self):
        template = QuestionTemplate(QuestionType.DESCRIPTIVE)

        pairs = template.generate({"label": "Gender", "mode": "Male"}, ANSWER)

        assert any("common" in p["question"].lower() for p in pairs)

    def test_insight_without_either_produces_nothing(self):
        template = QuestionTemplate(QuestionType.DESCRIPTIVE)

        assert template.generate({"label": "Age"}, ANSWER) == []


class TestOtherTypes:
    def test_comparative(self):
        pairs = QuestionTemplate(QuestionType.COMPARATIVE).generate(
            {"var_categorical": "Gender", "var_numeric": "Income"}, ANSWER
        )

        assert pairs
        assert all(p["type"] == "comparative" for p in pairs)

    def test_temporal(self):
        pairs = QuestionTemplate(QuestionType.TEMPORAL).generate(
            {"value_variable": "Turnout", "time_variable": "year"}, ANSWER
        )

        assert pairs
        assert all(p["type"] == "temporal" for p in pairs)

    def test_causal_without_controls(self):
        pairs = QuestionTemplate(QuestionType.CAUSAL).generate(
            {"treatment": "Education", "outcome": "Income"}, ANSWER
        )

        assert len(pairs) == 2

    def test_causal_with_controls_adds_a_question(self):
        pairs = QuestionTemplate(QuestionType.CAUSAL).generate(
            {"treatment": "Education", "outcome": "Income", "controls": ["age"]},
            ANSWER,
        )

        assert len(pairs) == 3
        assert any("Controlling for" in p["question"] for p in pairs)

    def test_correlational(self):
        pairs = QuestionTemplate(QuestionType.CORRELATIONAL).generate(
            {"var1": "Age", "var2": "Income"}, ANSWER
        )

        assert pairs
        assert all(p["type"] == "correlational" for p in pairs)

    def test_distributional_needs_a_spread_or_shape_statistic(self):
        pairs = QuestionTemplate(QuestionType.DISTRIBUTIONAL).generate(
            {"label": "Age", "std": 11.2}, ANSWER
        )

        assert pairs
        assert all(p["type"] == "distributional" for p in pairs)

    def test_distributional_adds_a_frequency_question(self):
        pairs = QuestionTemplate(QuestionType.DISTRIBUTIONAL).generate(
            {"label": "Gender", "frequencies": {"Male": 10}}, ANSWER
        )

        assert any("frequency" in p["question"].lower() for p in pairs)

    def test_distributional_without_statistics_produces_nothing(self):
        assert (
            QuestionTemplate(QuestionType.DISTRIBUTIONAL).generate(
                {"label": "Age"}, ANSWER
            )
            == []
        )


class TestGenerateContract:
    @pytest.mark.parametrize("question_type", list(QuestionType))
    def test_answer_is_carried_through(self, question_type):
        insight = {
            "label": "Age",
            "mean": 1.0,
            "var_categorical": "g",
            "var_numeric": "v",
            "value_variable": "v",
            "treatment": "t",
            "outcome": "o",
            "var1": "a",
            "var2": "b",
        }

        pairs = QuestionTemplate(question_type).generate(insight, ANSWER)

        assert all(p["answer"] == ANSWER for p in pairs)
