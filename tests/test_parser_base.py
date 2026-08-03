"""Tests for the field parsing shared by every codebook parser.

Type, DGP and missing-code spellings are properties of codebooks rather than
of any one file format, so BaseParser resolves them and each parser inherits
the same behaviour.
"""

import pytest

from statqa.metadata.parsers.csv import CSVParser
from statqa.metadata.parsers.text import TextParser
from statqa.metadata.schema import DataGeneratingProcess, VariableType

PARSERS = [CSVParser, TextParser]


@pytest.fixture(params=PARSERS, ids=lambda p: p.__name__)
def parser(request):
    return request.param()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("numeric", VariableType.NUMERIC_CONTINUOUS),
        ("continuous", VariableType.NUMERIC_CONTINUOUS),
        ("discrete", VariableType.NUMERIC_DISCRETE),
        ("categorical", VariableType.CATEGORICAL_NOMINAL),
        ("nominal", VariableType.CATEGORICAL_NOMINAL),
        ("ordinal", VariableType.CATEGORICAL_ORDINAL),
        ("bool", VariableType.BOOLEAN),
        ("date", VariableType.DATETIME),
        ("string", VariableType.TEXT),
        ("  NUMERIC  ", VariableType.NUMERIC_CONTINUOUS),
        ("nonsense", VariableType.UNKNOWN),
    ],
)
def test_type_spellings_agree_across_parsers(parser, text, expected):
    assert parser._parse_type(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("observational", DataGeneratingProcess.OBSERVATIONAL),
        ("experimental", DataGeneratingProcess.EXPERIMENTAL),
        ("quasi-experimental", DataGeneratingProcess.QUASI_EXPERIMENTAL),
        ("quasi_experimental", DataGeneratingProcess.QUASI_EXPERIMENTAL),
        ("survey", DataGeneratingProcess.SURVEY),
        ("administrative", DataGeneratingProcess.ADMINISTRATIVE),
        ("simulation", DataGeneratingProcess.SIMULATION),
        ("  SURVEY  ", DataGeneratingProcess.SURVEY),
        ("nonsense", DataGeneratingProcess.UNKNOWN),
    ],
)
def test_dgp_spellings_agree_across_parsers(parser, text, expected):
    assert parser._parse_dgp(text) == expected


@pytest.mark.parametrize("text", ["-1, 999", "-1; 999", "-1 | 999"])
def test_every_separator_is_understood(parser, text):
    # The text parser only split on commas before these moved to the base
    # class, so '-1; 999' became the single string '-1; 999'.
    assert parser._parse_missing(text) == {-1, 999}


def test_sentinel_labels_stay_strings(parser):
    assert parser._parse_missing("NA, -1") == {"NA", -1}


def test_blank_entries_are_dropped(parser):
    assert parser._parse_missing("-1, , 999") == {-1, 999}


def test_empty_string_yields_no_codes(parser):
    assert parser._parse_missing("") == set()
