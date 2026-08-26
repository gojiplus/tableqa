"""Tests for LLM-backed metadata enrichment.

The LLM itself is never called: _call_llm is replaced so the tests exercise
prompt construction, response parsing and how enrichment is applied.
"""

import json

import pytest

from statqa.exceptions import LLMConnectionError, LLMResponseError
from statqa.metadata.enricher import MetadataEnricher
from statqa.metadata.schema import Codebook, Variable, VariableType


@pytest.fixture
def enricher(monkeypatch):
    """An enricher whose client construction is bypassed."""
    monkeypatch.setattr(MetadataEnricher, "__init__", lambda self, **kw: None)
    instance = MetadataEnricher()
    instance.provider = "openai"
    instance.model = "gpt-4"
    instance.kwargs = {}
    return instance


@pytest.fixture
def age():
    return Variable(
        name="age",
        label="Respondent Age",
        var_type=VariableType.UNKNOWN,
        description="Age in years",
    )


def respond_with(monkeypatch, enricher, payload):
    monkeypatch.setattr(
        MetadataEnricher, "_call_llm", lambda self, prompt, **kw: payload
    )
    return enricher


class TestResponseParsing:
    def test_plain_json(self, enricher):
        assert enricher._parse_enrichment_response('{"a": 1}') == {"a": 1}

    def test_json_fenced_block(self, enricher):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nhope that helps'

        assert enricher._parse_enrichment_response(text) == {"a": 1}

    def test_bare_fenced_block(self, enricher):
        text = '```\n{"a": 1}\n```'

        assert enricher._parse_enrichment_response(text) == {"a": 1}

    def test_unparsable_response_yields_empty_dict(self, enricher):
        assert enricher._parse_enrichment_response("not json at all") == {}


class TestPromptBuilding:
    def test_prompt_names_the_variable(self, enricher, age):
        prompt = enricher._build_variable_prompt(age, None)

        assert "age" in prompt
        assert "Respondent Age" in prompt

    def test_prompt_includes_dataset_context(self, enricher, age):
        prompt = enricher._build_variable_prompt(age, "A survey of adults")

        assert "A survey of adults" in prompt


class TestEnrichVariable:
    def test_metadata_is_recorded(self, monkeypatch, enricher, age):
        respond_with(monkeypatch, enricher, json.dumps({"topic": "demographics"}))

        result = enricher.enrich_variable(age)

        assert result.enriched_metadata["topic"] == "demographics"

    def test_high_confidence_type_is_applied(self, monkeypatch, enricher, age):
        respond_with(
            monkeypatch,
            enricher,
            json.dumps(
                {"suggested_type": "NUMERIC_CONTINUOUS", "type_confidence": 0.95}
            ),
        )

        result = enricher.enrich_variable(age)

        assert result.var_type == VariableType.NUMERIC_CONTINUOUS

    def test_low_confidence_type_is_ignored(self, monkeypatch, enricher, age):
        respond_with(
            monkeypatch,
            enricher,
            json.dumps(
                {"suggested_type": "NUMERIC_CONTINUOUS", "type_confidence": 0.2}
            ),
        )

        result = enricher.enrich_variable(age)

        assert result.var_type == VariableType.UNKNOWN

    def test_unknown_type_name_is_ignored(self, monkeypatch, enricher, age):
        respond_with(
            monkeypatch,
            enricher,
            json.dumps({"suggested_type": "NOT_A_TYPE", "type_confidence": 0.99}),
        )

        result = enricher.enrich_variable(age)

        assert result.var_type == VariableType.UNKNOWN

    def test_causal_roles_are_applied(self, monkeypatch, enricher, age):
        respond_with(
            monkeypatch,
            enricher,
            json.dumps(
                {"is_treatment": True, "is_outcome": False, "is_confounder": True}
            ),
        )

        result = enricher.enrich_variable(age)

        assert result.is_treatment is True
        assert result.is_confounder is True

    def test_connection_failure_raises_a_typed_error(self, monkeypatch, enricher, age):
        def boom(self, prompt, **kw):
            raise ConnectionError("no route to host")

        monkeypatch.setattr(MetadataEnricher, "_call_llm", boom)

        with pytest.raises(LLMConnectionError):
            enricher.enrich_variable(age)

    def test_bad_response_raises_a_typed_error(self, monkeypatch, enricher, age):
        def boom(self, prompt, **kw):
            raise ValueError("garbage")

        monkeypatch.setattr(MetadataEnricher, "_call_llm", boom)

        with pytest.raises(LLMResponseError):
            enricher.enrich_variable(age)


class TestEnrichCodebook:
    def test_every_variable_is_enriched(self, monkeypatch, enricher, age):
        respond_with(monkeypatch, enricher, json.dumps({"topic": "demographics"}))
        codebook = Codebook(
            name="Study",
            description="A survey",
            variables={
                "age": age,
                "income": Variable(
                    name="income",
                    label="Income",
                    var_type=VariableType.NUMERIC_CONTINUOUS,
                ),
            },
        )

        result = enricher.enrich_codebook(codebook)

        assert all(
            v.enriched_metadata.get("topic") == "demographics"
            for v in result.variables.values()
        )


class TestProviderValidation:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            MetadataEnricher(provider="mistral")  # type: ignore[arg-type]
