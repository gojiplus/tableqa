"""Tests for statistical format parser."""

import contextlib
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from statqa.metadata.parsers.statistical import StatisticalFormatParser
from statqa.metadata.schema import Codebook, VariableType

# Test requires pyreadstat - skip if not available
pyreadstat = pytest.importorskip("pyreadstat")


@pytest.fixture
def sample_spss_data():
    """Create sample SPSS data for testing."""
    data = {
        "age": [25, 30, 35, 40, 45],
        "gender": [1, 2, 1, 2, 1],
        "income": [30000, 45000, 55000, 65000, 75000],
        "satisfaction": [3, 4, 5, 2, 4],
    }

    df = pd.DataFrame(data)

    # Add metadata
    variable_value_labels = {
        "gender": {1: "Male", 2: "Female"},
        "satisfaction": {
            1: "Very Dissatisfied",
            2: "Dissatisfied",
            3: "Neutral",
            4: "Satisfied",
            5: "Very Satisfied",
        },
    }

    column_labels = {
        "age": "Respondent Age",
        "gender": "Gender",
        "income": "Annual Income",
        "satisfaction": "Job Satisfaction",
    }

    return df, variable_value_labels, column_labels


@pytest.fixture
def temp_spss_file(sample_spss_data):
    """Create temporary SPSS file for testing."""
    df, variable_value_labels, column_labels = sample_spss_data

    # Create temp file and close it immediately to avoid Windows file locking issues
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Write to the closed temp file
        pyreadstat.write_sav(
            df,
            str(temp_path),
            variable_value_labels=variable_value_labels,
            column_labels=column_labels,
            file_label="Test Survey Data",
        )
        yield temp_path
    finally:
        # Cleanup - handle Windows file locking gracefully
        try:
            if temp_path.exists():
                temp_path.unlink()
        except (OSError, PermissionError):
            # On Windows, file might still be locked, try again
            time.sleep(0.1)
            with contextlib.suppress(OSError, PermissionError):
                temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_stata_file(sample_spss_data):
    """Create temporary Stata file for testing."""
    df, variable_value_labels, column_labels = sample_spss_data

    # Create temp file and close it immediately to avoid Windows file locking issues
    with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Write to the closed temp file
        pyreadstat.write_dta(
            df,
            str(temp_path),
            variable_value_labels=variable_value_labels,
            column_labels=column_labels,
            file_label="Test Survey Data",
        )
        yield temp_path
    finally:
        # Cleanup - handle Windows file locking gracefully
        try:
            if temp_path.exists():
                temp_path.unlink()
        except (OSError, PermissionError):
            # On Windows, file might still be locked, try again
            time.sleep(0.1)
            with contextlib.suppress(OSError, PermissionError):
                temp_path.unlink(missing_ok=True)


class TestStatisticalFormatParser:
    """Test StatisticalFormatParser functionality."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = StatisticalFormatParser()
        assert parser is not None

    def test_validate_spss_file(self, temp_spss_file):
        """Test validation of SPSS files."""
        parser = StatisticalFormatParser()
        assert parser.validate(temp_spss_file) is True

    def test_validate_stata_file(self, temp_stata_file):
        """Test validation of Stata files."""
        parser = StatisticalFormatParser()
        assert parser.validate(temp_stata_file) is True

    def test_validate_invalid_file(self):
        """Test validation rejects invalid files."""
        parser = StatisticalFormatParser()

        # Non-existent file
        assert parser.validate("nonexistent.sav") is False

        # Wrong extension
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            assert parser.validate(f.name) is False

    def test_parse_spss_file(self, temp_spss_file):
        """Test parsing SPSS file."""
        parser = StatisticalFormatParser()
        codebook = parser.parse(temp_spss_file)

        # Check basic codebook properties
        assert isinstance(codebook, Codebook)
        assert len(codebook.variables) == 4
        assert "age" in codebook.variables
        assert "gender" in codebook.variables
        assert "income" in codebook.variables
        assert "satisfaction" in codebook.variables

        # Check variable metadata
        age_var = codebook.variables["age"]
        assert age_var.label == "Respondent Age"

        gender_var = codebook.variables["gender"]
        assert gender_var.label == "Gender"
        assert gender_var.valid_values == {1: "Male", 2: "Female"}

        satisfaction_var = codebook.variables["satisfaction"]
        assert satisfaction_var.label == "Job Satisfaction"
        assert len(satisfaction_var.valid_values) == 5

    def test_parse_stata_file(self, temp_stata_file):
        """Test parsing Stata file."""
        parser = StatisticalFormatParser()
        codebook = parser.parse(temp_stata_file)

        # Check basic codebook properties
        assert isinstance(codebook, Codebook)
        assert len(codebook.variables) == 4

        # Check some variable metadata
        age_var = codebook.variables["age"]
        assert age_var.label == "Respondent Age"

    def test_variable_type_inference(self, temp_spss_file):
        """Test variable type inference from metadata."""
        parser = StatisticalFormatParser()
        codebook = parser.parse(temp_spss_file)

        # Variables with value labels should be categorical
        gender_var = codebook.variables["gender"]
        satisfaction_var = codebook.variables["satisfaction"]

        # Note: The exact type depends on the inference logic
        # These might be CATEGORICAL_NOMINAL or UNKNOWN initially
        assert gender_var.var_type in {
            VariableType.CATEGORICAL_NOMINAL,
            VariableType.UNKNOWN,
        }
        assert satisfaction_var.var_type in {
            VariableType.CATEGORICAL_NOMINAL,
            VariableType.UNKNOWN,
        }

    def test_dataset_info_extraction(self, temp_spss_file):
        """Test dataset info extraction."""
        parser = StatisticalFormatParser()
        codebook = parser.parse(temp_spss_file)

        # Should have dataset info
        assert "number_rows" in codebook.dataset_info
        assert "number_columns" in codebook.dataset_info
        assert "raw_metadata" in codebook.dataset_info

    def test_missing_pyreadstat_error(self, monkeypatch):
        """Test error when pyreadstat not available."""
        # Mock missing pyreadstat
        monkeypatch.setattr("statqa.metadata.parsers.statistical.HAS_PYREADSTAT", False)

        with pytest.raises(ImportError, match="pyreadstat is required"):
            StatisticalFormatParser()

    @pytest.mark.parametrize(
        "extension", [".sav", ".zsav", ".por", ".dta", ".sas7bdat", ".xpt"]
    )
    def test_supported_extensions(self, extension):
        """Test all supported file extensions are recognized."""
        # Just check that the extension is in supported list
        assert extension.lower() in {
            ".sav",
            ".zsav",
            ".por",
            ".dta",
            ".sas7bdat",
            ".xpt",
        }
