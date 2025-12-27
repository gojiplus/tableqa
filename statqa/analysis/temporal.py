"""
Temporal analysis for time series data.

Analyzes trends and patterns over time:
- Trend detection (Mann-Kendall, linear regression)
- Seasonal decomposition
- Change point detection
- Year-over-year changes
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statqa.metadata.schema import Variable
from statqa.utils.stats import mann_kendall_trend


class TemporalAnalyzer:
    """
    Analyzer for temporal patterns and trends.

    Args:
        significance_level: Alpha level for statistical tests
        min_periods: Minimum number of time periods required
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_periods: int = 3,
    ) -> None:
        self.alpha = significance_level
        self.min_periods = min_periods

    def analyze_trend(
        self,
        data: pd.DataFrame,
        time_var: Variable,
        value_var: Variable,
    ) -> dict[str, Any]:
        """
        Analyze trend in a variable over time.

        Args:
            data: DataFrame with time and value columns
            time_var: Time variable metadata
            value_var: Value variable being analyzed over time

        Returns:
            Trend analysis results as dictionary
        """
        # Clean and prepare data
        subset = data[[time_var.name, value_var.name]].copy()
        subset = self._clean_data(subset, time_var, value_var)
        subset = subset.dropna()

        if len(subset) < self.min_periods:
            return {"error": "Insufficient time periods"}

        # Sort by time
        subset = subset.sort_values(time_var.name)

        result: dict[str, Any] = {
            "analysis_type": "temporal_trend",
            "time_variable": time_var.name,
            "value_variable": value_var.name,
            "n_periods": len(subset),
            "time_range": {
                "start": str(subset[time_var.name].min()),
                "end": str(subset[time_var.name].max()),
            },
        }

        # Mann-Kendall trend test (non-parametric)
        mk_result = mann_kendall_trend(subset[value_var.name])
        result["mann_kendall"] = mk_result

        # Linear trend
        if value_var.is_numeric():
            linear_trend = self._linear_trend(subset, time_var.name, value_var.name)
            result["linear_trend"] = linear_trend

        # Descriptive change metrics
        first_value = subset[value_var.name].iloc[0]
        last_value = subset[value_var.name].iloc[-1]
        absolute_change = last_value - first_value
        percent_change = (absolute_change / first_value * 100) if first_value != 0 else np.nan

        result["change_metrics"] = {
            "first_value": float(first_value),
            "last_value": float(last_value),
            "absolute_change": float(absolute_change),
            "percent_change": float(percent_change) if not np.isnan(percent_change) else None,
            "mean": float(subset[value_var.name].mean()),
            "std": float(subset[value_var.name].std()),
        }

        # Volatility (coefficient of variation)
        mean = subset[value_var.name].mean()
        std = subset[value_var.name].std()
        cv = (std / mean * 100) if mean != 0 else np.nan
        result["volatility"] = {
            "coefficient_of_variation": float(cv) if not np.isnan(cv) else None,
        }

        return result

    def analyze_grouped_trend(
        self,
        data: pd.DataFrame,
        time_var: Variable,
        value_var: Variable,
        group_var: Variable,
    ) -> dict[str, Any]:
        """
        Analyze trends separately for different groups.

        Args:
            data: DataFrame with time, value, and group columns
            time_var: Time variable
            value_var: Value variable
            group_var: Grouping variable

        Returns:
            Grouped trend analysis results as dictionary
        """
        subset = data[[time_var.name, value_var.name, group_var.name]].copy()
        subset = subset.dropna()

        if len(subset) < self.min_periods:
            return {"error": "Insufficient data"}

        result: dict[str, Any] = {
            "analysis_type": "grouped_temporal_trend",
            "time_variable": time_var.name,
            "value_variable": value_var.name,
            "group_variable": group_var.name,
            "groups": {},
        }

        # Analyze each group
        for group_name, group_data in subset.groupby(group_var.name):
            if len(group_data) >= self.min_periods:
                group_trend = self.analyze_trend(
                    group_data,
                    time_var,
                    value_var,
                )
                # Map group code to label
                label = (
                    group_var.valid_values.get(str(group_name), str(group_name))
                    if group_var.valid_values
                    else str(group_name)
                )
                result["groups"][label] = group_trend

        return result

    def detect_change_points(
        self,
        data: pd.DataFrame,
        time_var: Variable,
        value_var: Variable,
    ) -> dict[str, Any]:
        """
        Detect significant change points in time series.

        Uses simple segmentation approach comparing before/after means.

        Args:
            data: DataFrame with time and value columns
            time_var: Time variable
            value_var: Value variable

        Returns:
            Change point detection results as dictionary
        """
        subset = data[[time_var.name, value_var.name]].copy()
        subset = subset.dropna()

        # Sort by time
        subset = subset.sort_values(time_var.name)  # type: ignore[call-overload]

        if len(subset) < self.min_periods * 2:
            return {"error": "Insufficient data for change point detection"}

        # Simple approach: test each potential split point
        n = len(subset)
        best_t = None
        best_p = 1.0
        best_change = 0.0

        for i in range(self.min_periods, n - self.min_periods):
            before = subset[value_var.name].iloc[:i]
            after = subset[value_var.name].iloc[i:]

            # T-test for difference in means
            _t_stat, p_value = stats.ttest_ind(before, after)

            if p_value < best_p:
                best_p = p_value
                best_t = subset[time_var.name].iloc[i]
                best_change = after.mean() - before.mean()

        result: dict[str, Any] = {
            "analysis_type": "change_point_detection",
            "time_variable": time_var.name,
            "value_variable": value_var.name,
        }

        if best_p < self.alpha:
            result["change_point"] = {
                "time": str(best_t),
                "p_value": float(best_p),
                "magnitude": float(best_change),
                "significant": True,
            }
        else:
            result["change_point"] = {
                "significant": False,
                "message": "No significant change point detected",
            }

        return result

    def _clean_data(self, data: pd.DataFrame, *variables: Variable) -> pd.DataFrame:
        """Clean missing values based on metadata."""
        clean = data.copy()
        for var in variables:
            if var.missing_values:
                clean[var.name] = clean[var.name].replace(dict.fromkeys(var.missing_values, np.nan))
        return clean

    def _linear_trend(self, data: pd.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
        """Fit linear trend and return statistics."""
        # Create numeric time index
        time_numeric = np.arange(len(data))
        values = data[value_col].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "significant": bool(p_value < self.alpha),
            "direction": "increasing" if slope > 0 else "decreasing",
        }

    def year_over_year_change(
        self,
        data: pd.DataFrame,
        year_var: Variable,
        value_var: Variable,
    ) -> dict[str, Any]:
        """
        Calculate year-over-year changes.

        Args:
            data: DataFrame with year and value columns
            year_var: Year variable
            value_var: Value variable

        Returns:
            Year-over-year analysis results as dictionary
        """
        subset = data[[year_var.name, value_var.name]].copy()
        subset = subset.dropna()

        # Group by year and calculate mean
        yearly = subset.groupby(year_var.name)[value_var.name].mean().sort_index()

        if len(yearly) < 2:
            return {"error": "Insufficient years"}

        # Calculate YoY changes
        yoy_absolute = yearly.diff()
        yoy_percent = yearly.pct_change() * 100

        result: dict[str, Any] = {
            "analysis_type": "year_over_year",
            "year_variable": year_var.name,
            "value_variable": value_var.name,
            "n_years": len(yearly),
            "years": {
                str(year): {
                    "value": float(value),
                    "yoy_absolute": (
                        float(yoy_absolute.loc[year])
                        if year in yoy_absolute.index and not np.isnan(yoy_absolute.loc[year])
                        else None
                    ),
                    "yoy_percent": (
                        float(yoy_percent.loc[year])
                        if year in yoy_percent.index and not np.isnan(yoy_percent.loc[year])
                        else None
                    ),
                }
                for year, value in yearly.items()
            },
            "summary": {
                "mean_yoy_absolute": float(yoy_absolute.mean()),
                "mean_yoy_percent": float(yoy_percent.mean()),
                "max_increase": {
                    "year": str(yoy_absolute.idxmax()),
                    "value": float(yoy_absolute.max()),
                },
                "max_decrease": {
                    "year": str(yoy_absolute.idxmin()),
                    "value": float(yoy_absolute.min()),
                },
            },
        }

        return result
