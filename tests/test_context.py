"""Tests for LLM context building."""

import pytest

from statqa.interpretation.context import ContextBuilder
from statqa.metadata.schema import (
    Codebook,
    DataGeneratingProcess,
    Variable,
    VariableType,
)


@pytest.fixture
def builder():
    return ContextBuilder()


@pytest.fixture
def age():
    return Variable(
        name="age",
        label="Respondent Age",
        var_type=VariableType.NUMERIC_CONTINUOUS,
        description="Age in years",
        units="years",
        range_min=18,
        range_max=99,
        dgp=DataGeneratingProcess.SURVEY,
    )


@pytest.fixture
def gender():
    return Variable(
        name="gender",
        label="Gender",
        var_type=VariableType.CATEGORICAL_NOMINAL,
        valid_values={1: "Male", 2: "Female"},
        dgp=DataGeneratingProcess.SURVEY,
    )


@pytest.fixture
def codebook(age, gender):
    return Codebook(
        name="Study",
        description="A survey of adults",
        variables={"age": age, "gender": gender},
    )


class TestDatasetContext:
    def test_leads_with_the_dataset_description(self, builder, codebook):
        text = builder.build_dataset_context(codebook)

        assert "A survey of adults" in text

    def test_counts_variables_by_kind(self, builder, codebook):
        text = builder.build_dataset_context(codebook)

        assert "2 total" in text
        assert "1 numeric" in text
        assert "1 categorical" in text

    def test_reports_the_dominant_data_source(self, builder, codebook):
        text = builder.build_dataset_context(codebook)

        assert "survey" in text.lower()

    def test_empty_codebook_still_produces_text(self, builder):
        text = builder.build_dataset_context(Codebook(name="Empty"))

        assert isinstance(text, str)


class TestVariableContext:
    def test_includes_label_and_type(self, builder, age):
        text = builder.build_variable_context(age)

        assert "Respondent Age" in text
        assert "numeric_continuous" in text

    def test_includes_units_and_range_when_detailed(self, builder, age):
        text = builder.build_variable_context(age, detailed=True)

        assert "years" in text

    def test_brief_form_is_shorter(self, builder, age):
        assert len(builder.build_variable_context(age, detailed=False)) <= len(
            builder.build_variable_context(age, detailed=True)
        )

    def test_categorical_lists_value_labels(self, builder, gender):
        text = builder.build_variable_context(gender)

        assert "Male" in text


class TestAnalysisContext:
    def test_mentions_each_variable(self, builder, age, gender):
        text = builder.build_analysis_context([age, gender])

        assert "Respondent Age" in text
        assert "Gender" in text

    def test_accepts_a_codebook(self, builder, age, codebook):
        text = builder.build_analysis_context([age], codebook=codebook)

        assert isinstance(text, str)
        assert text


class TestCodebookSummary:
    def test_includes_variables(self, builder, codebook):
        text = builder.build_codebook_summary(codebook)

        assert "age" in text

    def test_max_variables_truncates(self, builder, codebook):
        text = builder.build_codebook_summary(codebook, max_variables=1)

        assert isinstance(text, str)


class TestInsightPrompt:
    @pytest.fixture
    def result(self):
        return {
            "analysis_type": "numeric_numeric",
            "var1": "age",
            "var2": "income",
            "pearson": {"r": 0.6, "p_value": 0.001, "significant": True},
            "n": 200,
        }

    @pytest.mark.parametrize("task", ["interpret", "enhance", "question"])
    def test_each_task_produces_a_prompt(self, builder, result, age, task):
        text = builder.build_insight_prompt(result, [age], task=task)

        assert isinstance(text, str)
        assert text.strip()

    def test_prompt_carries_the_numbers(self, builder, result, age):
        text = builder.build_insight_prompt(result, [age], task="interpret")

        assert "0.6" in text or "0.60" in text
