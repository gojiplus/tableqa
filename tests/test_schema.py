"""Tests for metadata schema."""

import pytest

from statqa.metadata.schema import Codebook, Variable, VariableType


def test_variable_creation():
    """Test creating a variable."""
    var = Variable(
        name="test_var",
        label="Test Variable",
        var_type=VariableType.NUMERIC_CONTINUOUS,
    )

    assert var.name == "test_var"
    assert var.label == "Test Variable"
    assert var.is_numeric()
    assert not var.is_categorical()


def test_variable_with_values():
    """Test variable with valid values."""
    var = Variable(
        name="gender",
        label="Gender",
        var_type=VariableType.CATEGORICAL_NOMINAL,
        valid_values={1: "Male", 2: "Female"},
        missing_values={0, 999},
    )

    assert var.is_categorical()
    assert 1 in var.valid_values
    assert 0 in var.missing_values
    cleaned = var.get_cleaned_values()
    assert 0 not in cleaned


def test_codebook_creation(sample_codebook: Codebook):
    """Test creating a codebook."""
    assert len(sample_codebook.variables) == 2
    assert "age" in sample_codebook.variables


def test_codebook_filtering(sample_codebook: Codebook):
    """Test codebook filtering methods."""
    numeric_vars = sample_codebook.get_numeric_variables()
    categorical_vars = sample_codebook.get_categorical_variables()

    assert len(numeric_vars) == 1
    assert len(categorical_vars) == 1
    assert numeric_vars[0].name == "age"
    assert categorical_vars[0].name == "gender"


class TestCodebookFromDict:
    """from_dict accepts both codebook shapes found in the wild."""

    def test_full_shape(self):
        data = {
            "name": "Study",
            "description": "A survey",
            "variables": {
                "age": {"name": "age", "label": "Age", "var_type": "numeric_continuous"}
            },
        }

        codebook = Codebook.from_dict(data)

        assert codebook.name == "Study"
        assert codebook.description == "A survey"
        assert codebook.variables["age"].label == "Age"

    def test_bare_variable_map(self):
        # What the bundled example codebooks contain: no surrounding metadata.
        data = {
            "age": {"name": "age", "label": "Age", "var_type": "numeric_continuous"},
            "gender": {
                "name": "gender",
                "label": "Gender",
                "var_type": "categorical_nominal",
            },
        }

        codebook = Codebook.from_dict(data, name="iris")

        assert codebook.name == "iris"
        assert set(codebook.variables) == {"age", "gender"}

    def test_name_argument_fills_in_a_missing_name(self):
        assert Codebook.from_dict({}, name="fallback").name == "fallback"

    def test_an_existing_name_wins(self):
        data = {"name": "Real", "variables": {}}

        assert Codebook.from_dict(data, name="fallback").name == "Real"

    def test_non_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            Codebook.from_dict([1, 2, 3])  # type: ignore[arg-type]
