"""
Plotting utilities for statistical visualizations.

Creates publication-quality plots for insights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statqa.metadata.schema import Variable


class PlotFactory:
    """
    Factory for creating statistical visualizations.

    Args:
        style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        figsize: Default figure size (width, height)
        dpi: DPI for rasterized output
    """

    def __init__(
        self,
        style: Literal["whitegrid", "darkgrid", "white", "dark", "ticks"] = "whitegrid",
        context: Literal["paper", "notebook", "talk", "poster"] = "notebook",
        figsize: tuple[int, int] = (8, 6),
        dpi: int = 100,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style(style)
        sns.set_context(context)

    def plot_univariate(
        self,
        data: pd.Series,
        variable: Variable,
        output_path: str | Path | None = None,
        return_metadata: bool = False,
    ) -> Figure | tuple[Figure, dict[str, Any]]:
        """
        Create univariate plot (histogram or bar chart).

        Args:
            data: Data series
            variable: Variable metadata
            output_path: Optional path to save plot
            return_metadata: Whether to return plot metadata alongside figure

        Returns:
            Matplotlib figure, or tuple of (figure, metadata) if return_metadata=True
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Clean data
        clean_data = self._clean_data(data, variable)

        if variable.is_numeric():
            plot_type = "histogram"
            self._plot_numeric_distribution(clean_data, variable, ax)
        elif variable.is_categorical():
            plot_type = "bar_chart"
            self._plot_categorical_distribution(clean_data, variable, ax)
        else:
            plot_type = "unknown"

        ax.set_title(f"Distribution of {variable.label}")

        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)

        if return_metadata:
            metadata = self._generate_univariate_metadata(
                clean_data, variable, plot_type, output_path
            )
            return fig, metadata

        return fig

    def plot_bivariate(
        self,
        data: pd.DataFrame,
        var1: Variable,
        var2: Variable,
        output_path: str | Path | None = None,
        return_metadata: bool = False,
    ) -> Figure | tuple[Figure, dict[str, Any]]:
        """
        Create bivariate plot (scatter, box, or heatmap).

        Args:
            data: DataFrame with both variables
            var1: First variable
            var2: Second variable
            output_path: Optional path to save plot
            return_metadata: Whether to return plot metadata alongside figure

        Returns:
            Matplotlib figure, or tuple of (figure, metadata) if return_metadata=True
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Clean data
        subset = data[[var1.name, var2.name]].copy()
        subset = self._clean_dataframe(subset, [var1, var2])
        subset = subset.dropna()

        if var1.is_numeric() and var2.is_numeric():
            plot_type = "scatter"
            self._plot_scatter(subset, var1, var2, ax)
        elif var1.is_categorical() and var2.is_numeric():
            plot_type = "boxplot"
            self._plot_boxplot(subset, var1, var2, ax)
        elif var1.is_categorical() and var2.is_categorical():
            plot_type = "heatmap"
            self._plot_heatmap(subset, var1, var2, ax)
        else:
            plot_type = "unknown"

        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)

        if return_metadata:
            metadata = self._generate_bivariate_metadata(subset, var1, var2, plot_type, output_path)
            return fig, metadata

        return fig

    def plot_temporal(
        self,
        data: pd.DataFrame,
        time_var: Variable,
        value_var: Variable,
        group_var: Variable | None = None,
        output_path: str | Path | None = None,
    ) -> Figure:
        """
        Create temporal trend plot.

        Args:
            data: DataFrame with time and value
            time_var: Time variable
            value_var: Value variable
            group_var: Optional grouping variable
            output_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Clean and sort
        cols = [time_var.name, value_var.name]
        if group_var:
            cols.append(group_var.name)

        subset = data[cols].copy()
        subset = self._clean_dataframe(
            subset, [time_var, value_var] + ([group_var] if group_var else [])
        )
        subset = subset.dropna().sort_values(time_var.name)

        if group_var:
            # Grouped line plot
            for group_name, group_data in subset.groupby(group_var.name):
                label = (
                    group_var.valid_values.get(str(group_name), str(group_name))
                    if group_var.valid_values
                    else str(group_name)
                )
                ax.plot(
                    group_data[time_var.name],
                    group_data[value_var.name],
                    marker="o",
                    label=label,
                )
            ax.legend()
        else:
            # Simple line plot
            ax.plot(subset[time_var.name], subset[value_var.name], marker="o", linewidth=2)

        ax.set_xlabel(time_var.label)
        ax.set_ylabel(value_var.label)
        ax.set_title(f"{value_var.label} over {time_var.label}")
        ax.grid(True, alpha=0.3)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)

        return fig

    def _clean_data(self, data: pd.Series, variable: Variable) -> pd.Series:
        """Clean missing values from series."""
        clean = data.copy()
        if variable.missing_values:
            clean = clean.replace(dict.fromkeys(variable.missing_values, np.nan))
        return clean.dropna()

    def _clean_dataframe(self, data: pd.DataFrame, variables: list[Variable]) -> pd.DataFrame:
        """Clean missing values from dataframe."""
        clean = data.copy()
        for var in variables:
            if var.missing_values:
                clean[var.name] = clean[var.name].replace(dict.fromkeys(var.missing_values, np.nan))
        return clean

    def _plot_numeric_distribution(self, data: pd.Series, variable: Variable, ax: Axes) -> None:
        """Plot histogram/KDE for numeric variable."""
        n_unique = data.nunique()

        if n_unique > 50:
            # Use KDE for continuous data
            sns.histplot(data, kde=True, ax=ax, stat="density")
            ax.set_ylabel("Density")
        else:
            # Use count histogram for discrete data
            sns.histplot(data, kde=False, ax=ax, bins=min(n_unique, 30))
            ax.set_ylabel("Count")

        ax.set_xlabel(variable.label)

        # Add mean line
        mean = data.mean()
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}", alpha=0.7)
        ax.legend()

    def _plot_categorical_distribution(self, data: pd.Series, variable: Variable, ax: Axes) -> None:
        """Plot bar chart for categorical variable."""
        counts = data.value_counts()

        # Map to labels if available
        if variable.valid_values:
            counts.index = counts.index.map(lambda x: variable.valid_values.get(x, str(x)))

        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
        ax.set_xlabel(variable.label)
        ax.set_ylabel("Count")

        # Rotate labels if many categories
        if len(counts) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_scatter(self, data: pd.DataFrame, var1: Variable, var2: Variable, ax: Axes) -> None:
        """Plot scatter plot with regression line."""
        sns.regplot(
            x=var1.name,
            y=var2.name,
            data=data,
            ax=ax,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"},
        )
        ax.set_xlabel(var1.label)
        ax.set_ylabel(var2.label)
        ax.set_title(f"{var1.label} vs {var2.label}")

    def _plot_boxplot(
        self, data: pd.DataFrame, var_cat: Variable, var_num: Variable, ax: Axes
    ) -> None:
        """Plot box plot for categorical vs numeric."""
        # Map categories to labels
        plot_data = data.copy()
        if var_cat.valid_values:
            plot_data[var_cat.name] = plot_data[var_cat.name].map(
                lambda x: var_cat.valid_values.get(x, str(x))
            )

        sns.boxplot(x=var_cat.name, y=var_num.name, data=plot_data, ax=ax, palette="Set2")
        ax.set_xlabel(var_cat.label)
        ax.set_ylabel(var_num.label)
        ax.set_title(f"{var_num.label} by {var_cat.label}")

        if len(plot_data[var_cat.name].unique()) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_heatmap(self, data: pd.DataFrame, var1: Variable, var2: Variable, ax: Axes) -> None:
        """Plot heatmap for categorical vs categorical."""
        # Create contingency table
        contingency = pd.crosstab(data[var1.name], data[var2.name])

        # Map to labels
        if var1.valid_values:
            contingency.index = contingency.index.map(lambda x: var1.valid_values.get(x, str(x)))
        if var2.valid_values:
            contingency.columns = contingency.columns.map(
                lambda x: var2.valid_values.get(x, str(x))
            )

        sns.heatmap(contingency, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_xlabel(var2.label)
        ax.set_ylabel(var1.label)
        ax.set_title(f"{var1.label} vs {var2.label}")

    def _generate_univariate_metadata(
        self,
        data: pd.Series,
        variable: Variable,
        plot_type: str,
        output_path: str | Path | None,
    ) -> dict:
        """Generate metadata for univariate plots."""
        metadata = {
            "plot_type": plot_type,
            "caption": self._generate_univariate_caption(data, variable),
            "alt_text": self._generate_univariate_alt_text(data, variable, plot_type),
            "visual_elements": self._extract_univariate_visual_elements(data, variable, plot_type),
        }

        if output_path:
            metadata["primary_plot"] = str(output_path)
            metadata["generation_code"] = (
                f"plot_factory.plot_univariate(data['{variable.name}'], "
                f"{variable.name}_var, '{output_path}')"
            )

        return metadata

    def _generate_bivariate_metadata(
        self,
        data: pd.DataFrame,
        var1: Variable,
        var2: Variable,
        plot_type: str,
        output_path: str | Path | None,
    ) -> dict:
        """Generate metadata for bivariate plots."""
        metadata = {
            "plot_type": plot_type,
            "caption": self._generate_bivariate_caption(data, var1, var2, plot_type),
            "alt_text": self._generate_bivariate_alt_text(data, var1, var2, plot_type),
            "visual_elements": self._extract_bivariate_visual_elements(data, var1, var2, plot_type),
        }

        if output_path:
            metadata["primary_plot"] = str(output_path)
            metadata["generation_code"] = (
                f"plot_factory.plot_bivariate(data, {var1.name}_var, "
                f"{var2.name}_var, '{output_path}')"
            )

        return metadata

    def _generate_univariate_caption(self, data: pd.Series, variable: Variable) -> str:
        """Generate descriptive caption for univariate plots."""
        if variable.is_numeric():
            mean_val = data.mean()
            std_val = data.std()
            n_obs = len(data)

            # Detect distribution shape
            skewness = data.skew()
            if abs(skewness) < 0.5:
                shape = "approximately normal distribution"
            elif skewness > 0.5:
                shape = "right-skewed distribution"
            else:
                shape = "left-skewed distribution"

            return (
                f"Histogram showing {variable.label.lower()} distribution "
                f"with mean={mean_val:.2f} and std={std_val:.2f} "
                f"(N={n_obs}). The data shows a {shape}."
            )
        else:
            counts = data.value_counts()
            mode = counts.idxmax()
            mode_pct = (counts.max() / len(data)) * 100
            n_categories = len(counts)

            if variable.valid_values and mode in variable.valid_values:
                mode_label = variable.valid_values[mode]
            else:
                mode_label = str(mode)

            return (
                f"Bar chart showing {variable.label.lower()} frequencies "
                f"across {n_categories} categories (N={len(data)}). "
                f"Most common category is '{mode_label}' ({mode_pct:.1f}%)."
            )

    def _generate_bivariate_caption(
        self, data: pd.DataFrame, var1: Variable, var2: Variable, plot_type: str
    ) -> str:
        """Generate descriptive caption for bivariate plots."""
        if plot_type == "scatter":
            correlation = data.corr().iloc[0, 1]
            if abs(correlation) < 0.3:
                strength = "weak"
            elif abs(correlation) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            direction = "positive" if correlation > 0 else "negative"

            return (
                f"Scatter plot showing the relationship between {var1.label} and "
                f"{var2.label} (N={len(data)}). Shows a {strength} {direction} "
                f"correlation (r={correlation:.2f}) with regression line."
            )

        elif plot_type == "boxplot":
            n_groups = data[var1.name].nunique()
            return (
                f"Box plots comparing {var2.label} across {n_groups} "
                f"{var1.label.lower()} groups (N={len(data)}). Shows "
                f"distribution differences and potential outliers."
            )

        elif plot_type == "heatmap":
            n_var1 = data[var1.name].nunique()
            n_var2 = data[var2.name].nunique()
            return (
                f"Heatmap showing the relationship between {var1.label} "
                f"({n_var1} categories) and {var2.label} ({n_var2} categories). "
                f"Color intensity represents frequency counts."
            )

        return f"Bivariate plot showing {var1.label} vs {var2.label}"

    def _generate_univariate_alt_text(
        self, data: pd.Series, variable: Variable, plot_type: str
    ) -> str:
        """Generate accessibility alt-text for univariate plots."""
        if plot_type == "histogram":
            return (
                f"Histogram chart with {variable.label.lower()} values on x-axis "
                f"and frequency density on y-axis, showing distribution shape "
                f"with {len(data)} observations."
            )
        elif plot_type == "bar_chart":
            n_categories = data.nunique()
            return (
                f"Bar chart with {n_categories} categories of {variable.label.lower()} "
                f"on x-axis and count frequencies on y-axis."
            )
        return f"Chart showing {variable.label.lower()} distribution"

    def _generate_bivariate_alt_text(
        self, data: pd.DataFrame, var1: Variable, var2: Variable, plot_type: str
    ) -> str:
        """Generate accessibility alt-text for bivariate plots."""
        if plot_type == "scatter":
            return (
                f"Scatter plot with {var1.label} on x-axis and {var2.label} "
                f"on y-axis, showing {len(data)} data points with regression line."
            )
        elif plot_type == "boxplot":
            n_groups = data[var1.name].nunique()
            return (
                f"Box plot chart with {n_groups} {var1.label.lower()} categories "
                f"on x-axis and {var2.label} values on y-axis."
            )
        elif plot_type == "heatmap":
            return (
                f"Heatmap with {var1.label} categories on y-axis and "
                f"{var2.label} categories on x-axis, using color intensity "
                f"to show frequency counts."
            )
        return f"Chart showing relationship between {var1.label} and {var2.label}"

    def _extract_univariate_visual_elements(
        self, data: pd.Series, variable: Variable, plot_type: str
    ) -> dict:
        """Extract visual elements description for univariate plots."""
        elements = {
            "chart_type": plot_type,
            "x_axis": variable.label,
            "key_features": [],
            "colors": [],
            "annotations": [],
        }

        if plot_type == "histogram":
            elements["y_axis"] = "Density"
            elements["colors"] = ["blue bars", "red mean line"]
            elements["key_features"] = ["distribution shape", "mean line"]
            elements["annotations"] = [f"Mean: {data.mean():.2f}"]

            # Add distribution characteristics
            if abs(data.skew()) > 0.5:
                elements["key_features"].append("skewed distribution")

        elif plot_type == "bar_chart":
            elements["y_axis"] = "Count"
            elements["colors"] = ["viridis color palette"]
            elements["key_features"] = ["frequency bars"]

        return elements

    def _extract_bivariate_visual_elements(
        self, data: pd.DataFrame, var1: Variable, var2: Variable, plot_type: str
    ) -> dict:
        """Extract visual elements description for bivariate plots."""
        elements = {
            "chart_type": plot_type,
            "x_axis": var1.label if plot_type != "heatmap" else var2.label,
            "y_axis": var2.label if plot_type != "heatmap" else var1.label,
            "key_features": [],
            "colors": [],
            "annotations": [],
        }

        if plot_type == "scatter":
            elements["colors"] = ["blue points", "red regression line"]
            elements["key_features"] = ["data points", "regression line", "trend"]

        elif plot_type == "boxplot":
            elements["colors"] = ["Set2 color palette"]
            elements["key_features"] = ["boxes", "whiskers", "outliers", "medians"]

        elif plot_type == "heatmap":
            elements["colors"] = ["YlOrRd color map"]
            elements["key_features"] = ["color intensity", "frequency counts"]
            elements["annotations"] = ["count values in cells"]

        return elements

    def close_all(self) -> None:
        """Close all open figures."""
        plt.close("all")
