"""Tests for the CSV codebook parser."""

import pytest

from statqa.metadata.parsers.csv import CSVParser
from statqa.metadata.schema import DataGeneratingProcess, VariableType


@pytest.fixture
def parser():
    return CSVParser()


@pytest.fixture
def codebook_csv(tmp_path):
    path = tmp_path / "codebook.csv"
    path.write_text(
        "variable_name,label,type,values,missing,units\n"
        "age,Respondent Age,numeric_continuous,,-1;999,years\n"
        "gender,Gender,categorical,1: Male; 2: Female,0,\n"
        "employed,Employed,boolean,,,\n"
    )
    return path


class TestValidate:
    def test_accepts_a_codebook_with_a_name_column(self, parser, codebook_csv):
        assert parser.validate(codebook_csv) is True

    def test_rejects_a_missing_file(self, parser):
        assert parser.validate("does_not_exist.csv") is False

    def test_rejects_a_csv_without_a_name_column(self, parser, tmp_path):
        path = tmp_path / "other.csv"
        path.write_text("foo,bar\n1,2\n")

        assert parser.validate(path) is False


class TestParse:
    def test_reads_every_row(self, parser, codebook_csv):
        codebook = parser.parse(codebook_csv)

        assert set(codebook.variables) == {"age", "gender", "employed"}

    def test_names_the_codebook_after_the_file(self, parser, codebook_csv):
        assert parser.parse(codebook_csv).name == "codebook"

    def test_carries_labels_and_units(self, parser, codebook_csv):
        age = parser.parse(codebook_csv).variables["age"]

        assert age.label == "Respondent Age"
        assert age.units == "years"

    def test_parses_value_labels(self, parser, codebook_csv):
        gender = parser.parse(codebook_csv).variables["gender"]

        assert gender.valid_values == {1: "Male", 2: "Female"}

    def test_parses_missing_codes(self, parser, codebook_csv):
        age = parser.parse(codebook_csv).variables["age"]

        assert age.missing_values == {-1, 999}

    def test_rejects_a_csv_without_a_name_column(self, parser, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("foo,bar\n1,2\n")

        with pytest.raises(ValueError, match="variable_name"):
            parser.parse(path)


class TestTypeParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("numeric", VariableType.NUMERIC_CONTINUOUS),
            ("continuous", VariableType.NUMERIC_CONTINUOUS),
            ("discrete", VariableType.NUMERIC_DISCRETE),
            ("categorical", VariableType.CATEGORICAL_NOMINAL),
            ("ordinal", VariableType.CATEGORICAL_ORDINAL),
            ("bool", VariableType.BOOLEAN),
            ("date", VariableType.DATETIME),
            ("string", VariableType.TEXT),
            ("  NUMERIC  ", VariableType.NUMERIC_CONTINUOUS),
            ("nonsense", VariableType.UNKNOWN),
        ],
    )
    def test_type_aliases(self, parser, text, expected):
        assert parser._parse_type(text) == expected


class TestValueParsing:
    @pytest.mark.parametrize(
        "text",
        ["1: Male; 2: Female", "1=Male, 2=Female", "1 : Male ; 2 : Female"],
    )
    def test_separator_variants_agree(self, parser, text):
        assert parser._parse_values(text) == {1: "Male", 2: "Female"}

    def test_non_numeric_codes_stay_strings(self, parser):
        assert parser._parse_values("M: Male; F: Female") == {
            "M": "Male",
            "F": "Female",
        }

    def test_entries_without_a_separator_are_skipped(self, parser):
        assert parser._parse_values("1: Male; garbage") == {1: "Male"}

    def test_empty_string_yields_nothing(self, parser):
        assert parser._parse_values("") == {}


class TestMissingParsing:
    @pytest.mark.parametrize("text", ["-1, 999", "-1; 999", "-1 | 999"])
    def test_separator_variants_agree(self, parser, text):
        assert parser._parse_missing(text) == {-1, 999}

    def test_non_numeric_codes_stay_strings(self, parser):
        assert parser._parse_missing("NA, -1") == {"NA", -1}


class TestDgpParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("observational", DataGeneratingProcess.OBSERVATIONAL),
            ("experimental", DataGeneratingProcess.EXPERIMENTAL),
            ("survey", DataGeneratingProcess.SURVEY),
            ("nonsense", DataGeneratingProcess.UNKNOWN),
        ],
    )
    def test_dgp_aliases(self, parser, text, expected):
        assert parser._parse_dgp(text) == expected
