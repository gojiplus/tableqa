"""Statistical utilities and helper functions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest


# Type aliases for better readability
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]
IntArray = np.ndarray[Any, np.dtype[np.integer[Any]]]


def calculate_effect_size(
    data1: pd.Series | FloatArray | float,
    data2: pd.Series | FloatArray | None = None,
    effect_type: Literal["cohen_d", "r_to_d", "cramers_v", "eta_squared"] = "cohen_d",
) -> float:
    """
    Calculate effect size for statistical tests.

    Args:
        data1: First sample or correlation coefficient
        data2: Second sample (for two-sample tests)
        effect_type: Type of effect size ('cohen_d', 'r_to_d', 'cramers_v', 'eta_squared')

    Returns:
        Effect size value

    Raises:
        ValueError: If invalid effect_type or incompatible data
        NotImplementedError: If effect_type is not yet implemented
    """
    if effect_type == "cohen_d":
        if data2 is None:
            raise ValueError("cohen_d requires two samples")
        if isinstance(data1, float):
            raise ValueError("cohen_d requires array-like data, not scalar")
        return cohens_d(data1, data2)

    elif effect_type == "r_to_d":
        # Convert correlation to Cohen's d
        if not isinstance(data1, int | float):
            raise ValueError("r_to_d expects a correlation coefficient (float)")
        r = float(data1)
        return float(2 * r / np.sqrt(1 - r**2))

    elif effect_type == "eta_squared":
        # For ANOVA - expects F-statistic and degrees of freedom
        raise NotImplementedError("eta_squared not yet implemented")

    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")


def cohens_d(group1: pd.Series | FloatArray, group2: pd.Series | FloatArray) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Args:
        group1: First group
        group2: Second group

    Returns:
        Cohen's d (standardized mean difference)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def cramers_v(contingency_table: pd.DataFrame | IntArray) -> float:
    """
    Calculate Cramér's V effect size for categorical associations.

    Args:
        contingency_table: Contingency table (crosstab)

    Returns:
        Cramér's V (0 to 1)
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = (
        contingency_table.sum().sum()
        if isinstance(contingency_table, pd.DataFrame)
        else contingency_table.sum()
    )
    min_dim = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim)))


def correct_multiple_testing(
    p_values: list[float] | FloatArray,
    method: Literal["bonferroni", "fdr_bh", "fdr_by"] = "fdr_bh",
    alpha: float = 0.05,
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], FloatArray]:
    """
    Apply multiple testing correction to p-values.

    Args:
        p_values: List or array of p-values
        method: Correction method ('bonferroni', 'fdr_bh', 'fdr_by')
            - bonferroni: Bonferroni correction (most conservative)
            - fdr_bh: Benjamini-Hochberg FDR (recommended)
            - fdr_by: Benjamini-Yekutieli FDR (more conservative)
        alpha: Significance level

    Returns:
        Tuple of (reject, corrected_p_values)
        - reject: Boolean array indicating which tests reject null
        - corrected_p_values: Adjusted p-values

    Raises:
        ValueError: If correction method is not supported
    """
    p_values = np.asarray(p_values)

    if method == "bonferroni":
        reject, corrected, _, _ = multitest.multipletests(
            p_values, alpha=alpha, method="bonferroni"
        )
    elif method == "fdr_bh":
        reject, corrected, _, _ = multitest.multipletests(p_values, alpha=alpha, method="fdr_bh")
    elif method == "fdr_by":
        reject, corrected, _, _ = multitest.multipletests(p_values, alpha=alpha, method="fdr_by")
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return reject, corrected


def robust_stats(data: pd.Series | FloatArray) -> dict[str, float]:
    """
    Calculate robust statistics for potentially outlier-heavy data.

    Args:
        data: Input data

    Returns:
        Dictionary with robust statistics:
        - median: Median (robust central tendency)
        - mad: Median Absolute Deviation (robust dispersion)
        - iqr: Interquartile Range
        - q25, q75: Quartiles
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return {
            "median": np.nan,
            "mad": np.nan,
            "iqr": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25

    return {
        "median": float(median),
        "mad": float(mad),
        "iqr": float(iqr),
        "q25": float(q25),
        "q75": float(q75),
    }


def detect_outliers(
    data: pd.Series | FloatArray,
    method: Literal["iqr", "mad", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> FloatArray:
    """
    Detect outliers in data.

    Args:
        data: Input data
        method: Detection method ('iqr', 'mad', 'zscore')
        threshold: Threshold for outlier detection
            - iqr: Multiplier for IQR (default 1.5)
            - mad: Multiplier for MAD (default 3.0 recommended)
            - zscore: Z-score threshold (default 3.0)

    Returns:
        Boolean array indicating outliers

    Raises:
        ValueError: If outlier detection method is not supported
    """
    data = np.asarray(data)
    valid = ~np.isnan(data)

    if method == "iqr":
        q25, q75 = np.percentile(data[valid], [25, 75])
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        outliers = (data < lower) | (data > upper)

    elif method == "mad":
        median = np.median(data[valid])
        mad = np.median(np.abs(data[valid] - median))
        modified_z = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z) > threshold

    elif method == "zscore":
        mean = np.mean(data[valid])
        std = np.std(data[valid])
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outliers.astype(float)


def mann_kendall_trend(series: pd.Series | FloatArray) -> dict[str, float | str]:
    """
    Perform Mann-Kendall trend test for temporal data.

    Args:
        series: Time series data

    Returns:
        Dictionary with:
        - tau: Kendall's tau statistic
        - p_value: Two-tailed p-value
        - trend: Trend direction ('increasing', 'decreasing', 'no trend')
    """
    from scipy.stats import kendalltau

    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 3:
        return {"tau": np.nan, "p_value": np.nan, "trend": "insufficient_data"}

    # Create time index
    time = np.arange(len(series))

    tau, p_value = kendalltau(time, series)

    trend = ("increasing" if tau > 0 else "decreasing") if p_value < 0.05 else "no_trend"

    return {"tau": float(tau), "p_value": float(p_value), "trend": trend}
