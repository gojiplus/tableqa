"""
Pydantic models for metadata representation.

This module defines the core data structures for variables and codebooks,
providing type-safe, validated models with rich metadata support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VariableType(str, Enum):
    """Statistical type of a variable."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class DataGeneratingProcess(str, Enum):
    """How the data was generated."""

    OBSERVATIONAL = "observational"
    EXPERIMENTAL = "experimental"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    SURVEY = "survey"
    ADMINISTRATIVE = "administrative"
    SIMULATION = "simulation"
    UNKNOWN = "unknown"


class MissingPattern(str, Enum):
    """Pattern of missing data."""

    MCAR = "mcar"  # Missing Completely At Random
    MAR = "mar"  # Missing At Random
    MNAR = "mnar"  # Missing Not At Random
    UNKNOWN = "unknown"


class Variable(BaseModel):
    """
    Represents a single variable/column in a dataset.

    Attributes:
        name: Variable identifier (e.g., 'VCF0101', 'age', 'income')
        label: Human-readable label/description
        var_type: Statistical type of the variable
        dtype: Raw data type (from pandas/numpy)
        description: Detailed description of what this variable measures
        valid_values: Mapping of codes to descriptions (e.g., {1: "Male", 2: "Female"})
        missing_values: Set of codes representing missing data (e.g., {-1, 999})
        missing_pattern: Pattern of missingness
        units: Measurement units (e.g., "years", "USD", "percentage")
        range_min: Minimum valid value (for numeric)
        range_max: Maximum valid value (for numeric)
        is_ordinal: Whether categorical variable has meaningful order
        dgp: Data generating process
        is_treatment: Whether this is a treatment/intervention variable
        is_outcome: Whether this is an outcome/dependent variable
        is_confounder: Whether this is a potential confounder
        temporal_variable: Name of associated time variable (if longitudinal)
        notes: Additional metadata notes
        source: Data source or survey question text
        enriched_metadata: LLM-generated enrichment information
    """

    # Core identification
    name: str = Field(..., description="Variable identifier")
    label: str = Field(..., description="Human-readable label")

    # Type information
    var_type: VariableType = Field(default=VariableType.UNKNOWN, description="Statistical type")
    dtype: str | None = Field(default=None, description="Raw data type")

    # Description
    description: str | None = Field(default=None, description="Detailed description")

    # Value coding
    valid_values: dict[int | str, str] = Field(
        default_factory=dict, description="Mapping of codes to labels"
    )
    missing_values: set[int | str] = Field(default_factory=set, description="Missing value codes")
    missing_pattern: MissingPattern = Field(
        default=MissingPattern.UNKNOWN, description="Missingness pattern"
    )

    # Numeric metadata
    units: str | None = Field(default=None, description="Measurement units")
    range_min: float | None = Field(default=None, description="Minimum valid value")
    range_max: float | None = Field(default=None, description="Maximum valid value")

    # Categorical metadata
    is_ordinal: bool = Field(default=False, description="Has meaningful order")

    # Causal/experimental metadata
    dgp: DataGeneratingProcess = Field(
        default=DataGeneratingProcess.UNKNOWN, description="Data generating process"
    )
    is_treatment: bool = Field(default=False, description="Is treatment variable")
    is_outcome: bool = Field(default=False, description="Is outcome variable")
    is_confounder: bool = Field(default=False, description="Is potential confounder")

    # Temporal metadata
    temporal_variable: str | None = Field(default=None, description="Associated time variable")

    # Additional metadata
    notes: str | None = Field(default=None, description="Additional notes")
    source: str | None = Field(default=None, description="Source/survey question")
    enriched_metadata: dict[str, Any] = Field(
        default_factory=dict, description="LLM-enriched metadata"
    )

    @field_validator("missing_values", mode="before")
    @classmethod
    def ensure_set(cls, v: Any) -> set[int | str]:
        """Ensure missing_values is a set."""
        if isinstance(v, list | tuple):
            return set(v)
        if isinstance(v, set):
            return v
        return set()

    def is_numeric(self) -> bool:
        """Check if variable is numeric."""
        return self.var_type in {
            VariableType.NUMERIC_CONTINUOUS,
            VariableType.NUMERIC_DISCRETE,
        }

    def is_categorical(self) -> bool:
        """Check if variable is categorical."""
        return self.var_type in {
            VariableType.CATEGORICAL_NOMINAL,
            VariableType.CATEGORICAL_ORDINAL,
            VariableType.BOOLEAN,
        }

    def is_temporal(self) -> bool:
        """Check if variable represents time."""
        return self.var_type == VariableType.DATETIME

    def get_cleaned_values(self) -> dict[int | str, str]:
        """Get valid values excluding missing codes."""
        return {k: v for k, v in self.valid_values.items() if k not in self.missing_values}

    model_config = {"use_enum_values": True, "validate_assignment": True}


class Codebook(BaseModel):
    """
    Represents a complete codebook/data dictionary.

    Attributes:
        name: Codebook name/identifier
        description: Overall dataset description
        variables: Mapping of variable names to Variable objects
        dataset_info: General dataset metadata
        citation: How to cite this dataset
        version: Codebook version
        last_updated: Last update date
    """

    name: str = Field(..., description="Codebook identifier")
    description: str | None = Field(default=None, description="Dataset description")
    variables: dict[str, Variable] = Field(default_factory=dict, description="Variable definitions")
    dataset_info: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    citation: str | None = Field(default=None, description="Citation information")
    version: str | None = Field(default=None, description="Version")
    last_updated: str | None = Field(default=None, description="Last update date")

    def get_variable(self, name: str) -> Variable | None:
        """Get variable by name."""
        return self.variables.get(name)

    def get_numeric_variables(self) -> list[Variable]:
        """Get all numeric variables."""
        return [v for v in self.variables.values() if v.is_numeric()]

    def get_categorical_variables(self) -> list[Variable]:
        """Get all categorical variables."""
        return [v for v in self.variables.values() if v.is_categorical()]

    def get_temporal_variables(self) -> list[Variable]:
        """Get all temporal variables."""
        return [v for v in self.variables.values() if v.is_temporal()]

    def get_treatment_variables(self) -> list[Variable]:
        """Get all treatment variables."""
        return [v for v in self.variables.values() if v.is_treatment]

    def get_outcome_variables(self) -> list[Variable]:
        """Get all outcome variables."""
        return [v for v in self.variables.values() if v.is_outcome]

    def add_variable(self, variable: Variable) -> None:
        """Add a variable to the codebook."""
        self.variables[variable.name] = variable

    model_config = {"validate_assignment": True}
