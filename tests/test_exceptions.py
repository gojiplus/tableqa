"""Tests for the exception hierarchy."""

import pytest

from statqa.exceptions import (
    ERROR_CODES,
    AnalysisError,
    CodebookParseError,
    EnrichmentError,
    ExportError,
    LLMConnectionError,
    LLMResponseError,
    ParseError,
    StatisticalAnalysisError,
    StatqaError,
    ValidationError,
    VariableParseError,
)


@pytest.mark.parametrize(
    ("subclass", "parent"),
    [
        (ParseError, StatqaError),
        (CodebookParseError, ParseError),
        (VariableParseError, ParseError),
        (AnalysisError, StatqaError),
        (StatisticalAnalysisError, AnalysisError),
        (EnrichmentError, StatqaError),
        (LLMConnectionError, EnrichmentError),
        (LLMResponseError, EnrichmentError),
        (ExportError, StatqaError),
        (ValidationError, StatqaError),
    ],
)
def test_hierarchy(subclass, parent):
    assert issubclass(subclass, parent)


def test_every_error_is_catchable_as_the_base():
    # Callers are documented to catch StatqaError; nothing may escape it.
    with pytest.raises(StatqaError):
        raise LLMConnectionError("no route to host")


def test_errors_carry_their_message():
    assert str(CodebookParseError("bad row 12")) == "bad row 12"


def test_error_codes_are_unique():
    assert len(set(ERROR_CODES.values())) == len(ERROR_CODES)


def test_error_codes_are_populated():
    assert ERROR_CODES
    assert all(isinstance(code, int) for code in ERROR_CODES.values())
