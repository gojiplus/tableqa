"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest

from statqa.metadata.schema import Codebook, Variable, VariableType


@pytest.fixture
def sample_numeric_data() -> pd.Series:
    """Sample numeric data for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(50, 10, 1000))


@pytest.fixture
def sample_categorical_data() -> pd.Series:
    """Sample categorical data for testing."""
    np.random.seed(42)
    return pd.Series(np.random.choice(["A", "B", "C"], size=1000))


@pytest.fixture
def sample_variable() -> Variable:
    """Sample variable metadata."""
    return Variable(
        name="age",
        label="Respondent Age",
        var_type=VariableType.NUMERIC_CONTINUOUS,
        description="Age in years",
        units="years",
        range_min=18,
        range_max=99,
        missing_values={-1, 999},
    )


@pytest.fixture
def sample_categorical_variable() -> Variable:
    """Sample categorical variable."""
    return Variable(
        name="gender",
        label="Gender",
        var_type=VariableType.CATEGORICAL_NOMINAL,
        valid_values={1: "Male", 2: "Female", 3: "Other"},
        missing_values={0},
    )


@pytest.fixture
def sample_codebook(
    sample_variable: Variable, sample_categorical_variable: Variable
) -> Codebook:
    """Sample codebook with multiple variables."""
    return Codebook(
        name="Test Codebook",
        description="A test codebook",
        variables={
            sample_variable.name: sample_variable,
            sample_categorical_variable.name: sample_categorical_variable,
        },
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(50, 10, 100),
            "gender": np.random.choice([1, 2, 3], size=100),
            "income": np.random.normal(50000, 15000, 100),
            "year": np.random.choice([2018, 2019, 2020, 2021], size=100),
        }
    )
