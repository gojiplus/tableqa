"""Tests for Q/A pair generation, provenance and export."""

import json

import pytest

from statqa.qa.generator import QAGenerator


@pytest.fixture
def generator():
    return QAGenerator()


@pytest.fixture
def univariate_insight():
    return {
        "analysis_type": "univariate",
        "variable": "age",
        "label": "Respondent Age",
        "type": "numeric_continuous",
        "mean": 50.2,
        "median": 49.8,
        "std": 11.2,
        "analyzer": "UnivariateAnalyzer",
        "computation_log": ["data['age'].mean()  # 50.2"],
    }


class TestGenerateQaPairs:
    def test_produces_pairs(self, generator, univariate_insight):
        pairs = generator.generate_qa_pairs(univariate_insight, "The mean age is 50.2.")

        assert pairs
        assert all(p["question"] and p["answer"] for p in pairs)

    def test_answer_is_the_formatted_insight(self, generator, univariate_insight):
        answer = "The mean age is 50.2."

        pairs = generator.generate_qa_pairs(univariate_insight, answer)

        assert all(p["answer"] == answer for p in pairs)

    def test_every_pair_carries_provenance(self, generator, univariate_insight):
        pairs = generator.generate_qa_pairs(univariate_insight, "answer")

        for pair in pairs:
            provenance = pair["provenance"]
            assert provenance["tool"] == "statqa"
            assert provenance["generation_method"] == "template"
            assert provenance["generated_at"]
            assert provenance["tool_version"]

    def test_provenance_records_the_analyzer(self, generator, univariate_insight):
        pairs = generator.generate_qa_pairs(univariate_insight, "answer")

        assert pairs[0]["provenance"]["analyzer"] == "UnivariateAnalyzer"

    def test_provenance_carries_the_computation_log(
        self, generator, univariate_insight
    ):
        pairs = generator.generate_qa_pairs(univariate_insight, "answer")

        assert pairs[0]["provenance"]["python_commands"] == [
            "data['age'].mean()  # 50.2"
        ]

    def test_variables_are_recorded_when_supplied(self, generator, univariate_insight):
        pairs = generator.generate_qa_pairs(
            univariate_insight, "answer", variables=["age"]
        )

        assert pairs[0]["variables"] == ["age"]

    def test_visual_data_is_attached_when_supplied(self, generator, univariate_insight):
        visual = {"figure": "plots/age.png", "alt_text": "histogram"}

        pairs = generator.generate_qa_pairs(
            univariate_insight, "answer", visual_data=visual
        )

        assert pairs[0]["visual"] == visual

    def test_provenance_is_not_shared_between_pairs(
        self, generator, univariate_insight
    ):
        pairs = generator.generate_qa_pairs(univariate_insight, "answer")
        pairs[0]["provenance"]["tool"] = "mutated"

        assert pairs[1]["provenance"]["tool"] == "statqa"

    def test_pairs_are_json_serialisable(self, generator, univariate_insight):
        pairs = generator.generate_qa_pairs(univariate_insight, "answer")

        assert json.loads(json.dumps(pairs)) == pairs


class TestGenerateBatch:
    def test_attaches_pairs_to_each_insight(self, generator, univariate_insight):
        results = generator.generate_batch(
            [univariate_insight, univariate_insight], ["a", "b"]
        )

        assert len(results) == 2
        assert all(r["qa_pairs"] for r in results)


class TestExport:
    @pytest.fixture
    def batch(self, generator, univariate_insight):
        return generator.generate_batch([univariate_insight], ["The mean age is 50.2."])

    def test_jsonl_lines_are_valid_json(self, generator, batch):
        lines = generator.export_qa_dataset(batch, output_format="jsonl")

        assert lines
        for line in lines:
            assert "question" in json.loads(line)

    def test_openai_format_uses_messages(self, generator, batch):
        lines = generator.export_qa_dataset(batch, output_format="openai")

        entry = json.loads(lines[0])
        roles = [m["role"] for m in entry["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_anthropic_format_uses_prompt_and_completion(self, generator, batch):
        lines = generator.export_qa_dataset(batch, output_format="anthropic")

        entry = json.loads(lines[0])
        assert set(entry) == {"prompt", "completion"}

    def test_unknown_format_produces_nothing(self, generator, batch):
        assert generator.export_qa_dataset(batch, output_format="nonsense") == []


class TestLlmDisabled:
    def test_exploratory_questions_are_empty_without_llm(
        self, generator, univariate_insight
    ):
        assert generator.generate_exploratory_questions(univariate_insight) == []

    def test_llm_provider_other_than_openai_is_rejected(self):
        with pytest.raises(ValueError, match="not yet supported"):
            QAGenerator(use_llm=True, llm_provider="anthropic")
