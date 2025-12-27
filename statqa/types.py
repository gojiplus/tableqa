"""
Type definitions for statqa package.

This module contains TypedDict definitions for structured data
to provide better type safety than generic dict[str, Any].
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class UnivariateResult(TypedDict, total=False):
    """Result of univariate analysis."""

    variable: str
    label: str
    variable_type: str
    total_count: int
    missing_count: int
    missing_percentage: float

    # Numeric statistics
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float

    # Robust statistics
    robust_mean: float
    mad: float

    # Distribution tests
    normality_test: dict[str, Any]
    outliers: dict[str, Any]

    # Categorical statistics
    mode: str | int
    mode_count: int
    unique_count: int
    diversity_index: float
    frequencies: dict[str, int]

    # Metadata
    analysis_type: Literal["numeric", "categorical"]
    formatted_insight: str


class BivariateResult(TypedDict, total=False):
    """Result of bivariate analysis."""

    var1: str
    var2: str
    var1_label: str
    var2_label: str
    analysis_type: Literal["numeric_numeric", "categorical_categorical", "categorical_numeric"]
    sample_size: int

    # Correlation analysis
    pearson: dict[str, Any]
    spearman: dict[str, Any]

    # Categorical associations
    chi_square: dict[str, Any]
    cramers_v: float
    contingency_table: dict[str, Any]

    # Group comparisons
    t_test: dict[str, Any]
    anova: dict[str, Any]

    # Effect size
    effect_size: float
    effect_size_interpretation: str

    # Metadata
    significant: bool
    formatted_insight: str


class QAPair(TypedDict):
    """Question-answer pair with provenance metadata."""

    question: str
    answer: str
    context: str

    # Provenance metadata
    generated_at: str  # ISO 8601 timestamp
    tool: str
    tool_version: str
    generation_method: Literal["template", "llm_paraphrase"]
    analysis_type: str
    analyzer: str
    llm_model: str | None

    # Source data
    variable_name: str | None
    variable_label: str | None
    analysis_result: dict[str, Any]


class TemporalResult(TypedDict, total=False):
    """Result of temporal analysis."""

    variable: str
    label: str
    time_variable: str | None
    analysis_type: Literal["temporal_trend", "change_point", "seasonality"]

    # Trend analysis
    trend: Literal["increasing", "decreasing", "stable", "insufficient_data"]
    tau: float
    p_value: float
    trend_significance: bool

    # Change point detection
    change_points: list[dict[str, Any]]

    # Seasonality
    seasonal_component: dict[str, Any]

    formatted_insight: str


class CausalResult(TypedDict, total=False):
    """Result of causal analysis."""

    treatment: str
    outcome: str
    confounders: list[str]
    analysis_type: Literal["treatment_effect", "instrumental_variable", "regression_discontinuity"]

    # Treatment effect
    ate: float  # Average Treatment Effect
    ate_ci_lower: float
    ate_ci_upper: float
    effect_significant: bool

    # Regression results
    regression_results: dict[str, Any]

    formatted_insight: str


# Export format types
class OpenAIFormat(TypedDict):
    """OpenAI fine-tuning format."""

    prompt: str
    completion: str


class AnthropicFormat(TypedDict):
    """Anthropic fine-tuning format."""

    messages: list[dict[str, str]]


class JSONLFormat(TypedDict):
    """Standard JSONL format with provenance."""

    question: str
    answer: str
    metadata: dict[str, Any]
