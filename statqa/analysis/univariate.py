"""
Univariate statistical analysis.

Performs descriptive statistics for single variables including:
- Numeric: mean, median, std, robust statistics, distribution tests
- Categorical: frequencies, mode, diversity measures
- Missing: missingness analysis
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statqa.metadata.schema import Variable
from statqa.utils.logging import get_logger
from statqa.utils.stats import detect_outliers, robust_stats


logger = get_logger(__name__)


class UnivariateAnalyzer:
    """
    Analyzer for single-variable statistics.

    Args:
        handle_outliers: Whether to detect and report outliers
        robust: Whether to include robust statistics (median, MAD)
    """

    def __init__(self, handle_outliers: bool = True, robust: bool = True) -> None:
        self.handle_outliers = handle_outliers
        self.robust = robust

    def analyze(self, data: pd.Series, variable: Variable) -> dict[str, Any]:
        """
        Analyze a single variable.

        Args:
            data: Data series
            variable: Variable metadata

        Returns:
            Analysis results as UnivariateResult
        """
        logger.debug(f"Analyzing variable '{variable.name}' (type: {variable.var_type})")

        # Clean missing values based on metadata
        clean_data = self._clean_missing(data, variable)
        logger.debug(f"Data cleaning: {len(data)} total → {len(clean_data.dropna())} valid values")

        result: dict[str, Any] = {
            "variable": variable.name,
            "label": variable.label,
            "type": (
                variable.var_type.value
                if hasattr(variable.var_type, "value")
                else variable.var_type
            ),
            "n_total": len(data),
            "n_valid": len(clean_data.dropna()),
            "n_missing": int(clean_data.isna().sum()),
            "missing_pct": float(clean_data.isna().sum() / len(clean_data) * 100),
        }

        if variable.is_numeric():
            result.update(self._analyze_numeric(clean_data, variable))
        elif variable.is_categorical():
            result.update(self._analyze_categorical(clean_data, variable))

        return result

    def _clean_missing(self, data: pd.Series, variable: Variable) -> pd.Series:
        """Replace missing value codes with NaN."""
        clean = data.copy()
        if variable.missing_values:
            clean = clean.replace(dict.fromkeys(variable.missing_values, np.nan))
        return clean

    def _analyze_numeric(self, data: pd.Series, variable: Variable) -> dict[str, Any]:
        """Analyze numeric variable."""
        valid_data = data.dropna()

        # Initialize computation log for provenance tracking
        computation_log = []
        computation_log.append("valid_data = data.dropna()")

        if len(valid_data) == 0:
            return {"error": "No valid data"}

        # Calculate statistics with computation tracking
        mean_val = float(valid_data.mean())
        computation_log.append(f"valid_data.mean()  # Result: {mean_val}")

        median_val = float(valid_data.median())
        computation_log.append(f"valid_data.median()  # Result: {median_val}")

        std_val = float(valid_data.std())
        computation_log.append(f"valid_data.std()  # Result: {std_val}")

        min_val = float(valid_data.min())
        computation_log.append(f"valid_data.min()  # Result: {min_val}")

        max_val = float(valid_data.max())
        computation_log.append(f"valid_data.max()  # Result: {max_val}")

        q25_val = float(valid_data.quantile(0.25))
        computation_log.append(f"valid_data.quantile(0.25)  # Result: {q25_val}")

        q75_val = float(valid_data.quantile(0.75))
        computation_log.append(f"valid_data.quantile(0.75)  # Result: {q75_val}")

        skew_val = float(stats.skew(valid_data))
        computation_log.append(f"scipy.stats.skew(valid_data)  # Result: {skew_val}")

        kurt_val = float(stats.kurtosis(valid_data))
        computation_log.append(f"scipy.stats.kurtosis(valid_data)  # Result: {kurt_val}")

        result: dict[str, Any] = {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "q25": q25_val,
            "q75": q75_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "computation_log": computation_log,
        }

        # Robust statistics
        if self.robust:
            robust = robust_stats(valid_data)
            result["robust"] = robust

        # Outlier detection
        if self.handle_outliers:
            outliers = detect_outliers(valid_data, method="iqr")
            result["n_outliers"] = int(outliers.sum())
            result["outlier_pct"] = float(outliers.sum() / len(valid_data) * 100)

        # Normality test (Shapiro-Wilk for n < 5000, else Anderson-Darling)
        if len(valid_data) < 5000:
            try:
                stat, p_value = stats.shapiro(valid_data)
                computation_log.append(
                    f"scipy.stats.shapiro(valid_data)  # stat={stat:.4f}, p={p_value:.4f}"
                )
                result["normality_test"] = {
                    "test": "shapiro-wilk",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": bool(p_value > 0.05),
                }
            except Exception:
                pass
        else:
            try:
                result_ad = stats.anderson(valid_data)
                computation_log.append(
                    f"scipy.stats.anderson(valid_data)  # stat={result_ad.statistic:.4f}"
                )
                result["normality_test"] = {
                    "test": "anderson-darling",
                    "statistic": float(result_ad.statistic),
                    "is_normal": bool(result_ad.statistic < result_ad.critical_values[2]),
                }
            except Exception:
                pass

        # Range validation
        if variable.range_min is not None or variable.range_max is not None:
            out_of_range = 0
            if variable.range_min is not None:
                out_of_range += int((valid_data < variable.range_min).sum())
            if variable.range_max is not None:
                out_of_range += int((valid_data > variable.range_max).sum())
            result["out_of_range"] = out_of_range

        return result

    def _analyze_categorical(self, data: pd.Series, variable: Variable) -> dict[str, Any]:
        """Analyze categorical variable."""
        valid_data = data.dropna()

        # Initialize computation log for provenance tracking
        computation_log = []
        computation_log.append("valid_data = data.dropna()")

        if len(valid_data) == 0:
            return {"error": "No valid data"}

        # Frequency counts with computation tracking
        counts = valid_data.value_counts()
        computation_log.append(f"counts = valid_data.value_counts()  # {len(counts)} unique values")

        frequencies = (counts / len(valid_data) * 100).round(2)
        computation_log.append("frequencies = (counts / len(valid_data) * 100).round(2)")

        # Mode
        mode = counts.idxmax()
        computation_log.append(f"mode = counts.idxmax()  # Result: {mode}")

        mode_freq = frequencies.max()
        computation_log.append(f"mode_frequency = frequencies.max()  # Result: {mode_freq}%")

        result: dict[str, Any] = {
            "n_unique": len(counts),
            "mode": mode,
            "mode_frequency": float(mode_freq),
            "frequencies": frequencies.to_dict(),
            "computation_log": computation_log,
        }

        # Map codes to labels if available
        if variable.valid_values:
            result["frequency_labels"] = {
                variable.valid_values.get(k, str(k)): v for k, v in frequencies.items()
            }

        # Diversity measures (entropy, Gini-Simpson)
        if len(counts) > 1:
            # Shannon entropy
            props = counts / counts.sum()
            computation_log.append("props = counts / counts.sum()")

            entropy = -np.sum(props * np.log2(props))
            computation_log.append(
                f"entropy = -np.sum(props * np.log2(props))  # Result: {entropy:.4f}"
            )
            result["shannon_entropy"] = float(entropy)

            # Gini-Simpson diversity
            gini_simpson = 1 - np.sum(props**2)
            computation_log.append(
                f"gini_simpson = 1 - np.sum(props**2)  # Result: {gini_simpson:.4f}"
            )
            result["gini_simpson"] = float(gini_simpson)

        # Check for rare categories (< 5% frequency)
        rare = frequencies[frequencies < 5.0]
        if len(rare) > 0:
            result["rare_categories"] = rare.to_dict()
            result["n_rare"] = len(rare)

        return result

    def batch_analyze(
        self, df: pd.DataFrame, variables: dict[str, Variable]
    ) -> list[dict[str, Any]]:
        """
        Analyze multiple variables at once.

        Args:
            df: DataFrame with data
            variables: Mapping of column names to Variable metadata

        Returns:
            List of analysis results as dictionaries
        """
        results = []

        for col in df.columns:
            if col in variables:
                variable = variables[col]
                result = self.analyze(df[col], variable)
                results.append(result)

        return results
