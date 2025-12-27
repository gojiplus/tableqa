"""
Natural language formatting for statistical insights.

Converts numerical results into human-readable statements with appropriate
statistical context and caveats.
"""

from typing import Any

import numpy as np


class InsightFormatter:
    """
    Formats statistical results as natural language insights.

    Args:
        include_caveats: Whether to include statistical caveats/warnings
        precision: Decimal precision for formatting numbers
    """

    def __init__(self, include_caveats: bool = True, precision: int = 2) -> None:
        self.include_caveats = include_caveats
        self.precision = precision

    def format_univariate(self, result: dict[str, Any]) -> str:
        """Format univariate analysis result."""
        var_label = result.get("label", result["variable"])

        if result["type"] in ["numeric_continuous", "numeric_discrete"]:
            return self._format_numeric_univariate(result, var_label)
        elif result["type"] in [
            "categorical_nominal",
            "categorical_ordinal",
            "boolean",
        ]:
            return self._format_categorical_univariate(result, var_label)
        else:
            return f"{var_label}: Analysis not available for this variable type."

    def _format_numeric_univariate(self, result: dict[str, Any], label: str) -> str:
        """Format numeric variable description."""
        mean = result.get("mean", 0)
        median = result.get("median", 0)
        std = result.get("std", 0)
        n = result.get("n_valid", 0)
        n_missing = result.get("n_missing", 0)

        # Main description
        text = (
            f"**{label}**: mean={mean:.{self.precision}f}, "
            f"median={median:.{self.precision}f}, "
            f"std={std:.{self.precision}f}"
        )

        # Add range
        if "min" in result and "max" in result:
            text += (
                f", range=[{result['min']:.{self.precision}f}, {result['max']:.{self.precision}f}]"
            )

        # Sample size
        text += f". N={n:,}"
        if n_missing > 0:
            text += f" ({n_missing:,} missing)"

        # Caveats
        if self.include_caveats:
            caveats = []

            # Normality
            if "normality_test" in result and not result["normality_test"].get("is_normal", True):
                caveats.append("non-normal distribution")

            # Outliers
            if result.get("n_outliers", 0) > 0:
                pct = result.get("outlier_pct", 0)
                caveats.append(f"{pct:.1f}% outliers")

            # Skewness
            skew = abs(result.get("skewness", 0))
            if skew > 1:
                caveats.append("highly skewed")

            if caveats:
                text += f" [{', '.join(caveats)}]"

        text += "."
        return text

    def _format_categorical_univariate(self, result: dict[str, Any], label: str) -> str:
        """Format categorical variable description."""
        mode = result.get("mode")
        mode_freq = result.get("mode_frequency", 0)
        n_unique = result.get("n_unique", 0)
        n = result.get("n_valid", 0)

        # Use labeled frequencies if available
        freq_dict = result.get("frequency_labels") or result.get("frequencies", {})

        # Main description
        if mode is not None:
            mode_label = next(iter(freq_dict.keys())) if freq_dict else str(mode)
            text = f"**{label}**: most common category is '{mode_label}' ({mode_freq:.1f}%)"
        else:
            text = f"**{label}**: {n_unique} categories"

        text += f", N={n:,}"

        # Add distribution if few categories
        if n_unique <= 5 and freq_dict:
            dist = ", ".join([f"{k}: {v:.1f}%" for k, v in list(freq_dict.items())[:5]])
            text += f". Distribution: {dist}"

        # Diversity
        if "shannon_entropy" in result:
            entropy = result["shannon_entropy"]
            max_entropy = np.log2(n_unique) if n_unique > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            if normalized_entropy < 0.5:
                text += " [low diversity]"
            elif normalized_entropy > 0.9:
                text += " [high diversity]"

        text += "."
        return text

    def format_bivariate(self, result: dict[str, Any]) -> str:
        """Format bivariate analysis result."""
        analysis_type = result.get("analysis_type")

        if analysis_type == "numeric_numeric":
            return self._format_correlation(result)
        elif analysis_type == "categorical_categorical":
            return self._format_categorical_association(result)
        elif analysis_type == "categorical_numeric":
            return self._format_group_comparison(result)
        else:
            return "Bivariate analysis result formatting not available."

    def _format_correlation(self, result: dict[str, Any]) -> str:
        """Format correlation result."""
        var1 = result["var1"]
        var2 = result["var2"]
        r = result["pearson"]["r"]
        p = result["pearson"]["p_value"]
        n = result["n"]
        strength = result.get("strength", "")

        direction = "positive" if r > 0 else "negative"

        text = (
            f"**{var1}** and **{var2}** show a {strength} {direction} correlation "
            f"(r={r:.{self.precision}f}, p={p:.3f}, N={n:,})"
        )

        if result["pearson"]["significant"]:
            text += " [statistically significant]"
        else:
            text += " [not statistically significant]"

        # Effect size
        if "effect_size" in result:
            interp = result["effect_size"]["interpretation"]
            text += f", effect size: {interp}"

        text += "."
        return text

    def _format_categorical_association(self, result: dict[str, Any]) -> str:
        """Format categorical association result."""
        var1 = result["var1"]
        var2 = result["var2"]
        chi2 = result["chi_square"]["statistic"]
        p = result["chi_square"]["p_value"]
        v = result["cramers_v"]["value"]
        v_interp = result["cramers_v"]["interpretation"]

        text = (
            f"**{var1}** and **{var2}** are associated "
            f"(χ²={chi2:.{self.precision}f}, p={p:.3f}, Cramér's V={v:.{self.precision}f})"
        )

        if result["chi_square"]["significant"]:
            text += f" [significant, {v_interp} effect]"
        else:
            text += " [not statistically significant]"

        # Assumption check
        if not result["assumptions"]["assumption_met"]:
            text += " [Warning: low expected frequencies]"

        text += "."
        return text

    def _format_group_comparison(self, result: dict[str, Any]) -> str:
        """Format group comparison result."""
        var_cat = result["var_categorical"]
        var_num = result["var_numeric"]

        # Get group means (try labeled first)
        group_means = result.get("group_means_labeled") or {}
        if not group_means:
            group_stats = result.get("group_stats", {})
            if "mean" in group_stats:
                group_means = group_stats["mean"]

        text = f"**{var_num}** differs across **{var_cat}** groups"

        # Show group means
        if group_means and len(group_means) <= 5:
            means_str = ", ".join(
                [f"{k}: {v:.{self.precision}f}" for k, v in list(group_means.items())[:5]]
            )
            text += f": {means_str}"

        # Statistical test
        if "t_test" in result:
            p = result["t_test"]["p_value"]
            text += f" (t-test: p={p:.3f})"
        elif "anova" in result:
            f_stat = result["anova"]["f_statistic"]
            p = result["anova"]["p_value"]
            text += f" (ANOVA: F={f_stat:.{self.precision}f}, p={p:.3f})"

        # Effect size
        if "effect_size" in result:
            if "cohens_d" in result["effect_size"]:
                d = result["effect_size"]["cohens_d"]
                interp = result["effect_size"]["interpretation"]
                text += f", Cohen's d={d:.{self.precision}f} ({interp})"
            elif "eta_squared" in result["effect_size"]:
                eta2 = result["effect_size"]["eta_squared"]
                interp = result["effect_size"]["interpretation"]
                text += f", η²={eta2:.{self.precision}f} ({interp})"

        text += "."
        return text

    def format_temporal(self, result: dict[str, Any]) -> str:
        """Format temporal analysis result."""
        value_var = result.get("value_variable", "Variable")
        result.get("time_variable", "time")

        if "mann_kendall" in result:
            mk = result["mann_kendall"]
            trend = mk["trend"]

            text = f"**{value_var}** shows "

            if trend == "increasing":
                text += "an increasing trend over time"
            elif trend == "decreasing":
                text += "a decreasing trend over time"
            else:
                text += "no significant trend over time"

            tau = mk["tau"]
            p = mk["p_value"]
            text += f" (Mann-Kendall: τ={tau:.{self.precision}f}, p={p:.3f})"

        # Change metrics
        if "change_metrics" in result:
            cm = result["change_metrics"]
            abs_change = cm.get("absolute_change", 0)
            pct_change = cm.get("percent_change")

            text += f". Change: {abs_change:+.{self.precision}f}"
            if pct_change is not None:
                text += f" ({pct_change:+.1f}%)"

        text += "."
        return text

    def format_causal(self, result: dict[str, Any]) -> str:
        """Format causal analysis result."""
        treatment = result.get("treatment", "Treatment")
        outcome = result.get("outcome", "Outcome")
        controls = result.get("controls", [])

        if "treatment_effect" in result:
            te = result["treatment_effect"]
            coef = te["coefficient"]
            p = te["p_value"]
            ci_lower = te["ci_lower"]
            ci_upper = te["ci_upper"]

            # Careful language for causal claims
            if controls:
                text = f"Controlling for {', '.join(controls)}, **{treatment}** is associated with "
            else:
                text = f"**{treatment}** is associated with "

            text += (
                f"a {abs(coef):.{self.precision}f}-unit {'increase' if coef > 0 else 'decrease'} "
                f"in **{outcome}** "
            )

            text += f"(β={coef:.{self.precision}f}, 95% CI=[{ci_lower:.{self.precision}f}, {ci_upper:.{self.precision}f}], p={p:.3f})"

            if te["significant"]:
                text += " [statistically significant]"
            else:
                text += " [not statistically significant]"

        # Model fit
        if "model_fit" in result:
            r2 = result["model_fit"]["adj_r_squared"]
            text += f". Model explains {r2 * 100:.1f}% of variance"

        # Sensitivity
        if "sensitivity" in result and result["sensitivity"].get("confounding", {}).get(
            "substantial"
        ):
            text += ". [Note: Substantial confounding detected]"

        text += "."
        return text

    def format_insight(self, result: dict[str, Any]) -> str:
        """
        Format any type of analysis result.

        Args:
            result: Analysis result dictionary

        Returns:
            Formatted natural language insight
        """
        analysis_type = result.get("analysis_type")

        if not analysis_type:
            # Try to infer from result structure
            if "mean" in result and "std" in result:
                return self.format_univariate(result)
            elif "pearson" in result or "spearman" in result:
                return self.format_bivariate(result)
            elif "mann_kendall" in result:
                return self.format_temporal(result)
            elif "treatment_effect" in result:
                return self.format_causal(result)
            else:
                return "Unable to format this analysis result."

        if analysis_type == "temporal_trend":
            return self.format_temporal(result)
        elif analysis_type == "treatment_effect":
            return self.format_causal(result)
        elif analysis_type in [
            "numeric_numeric",
            "categorical_categorical",
            "categorical_numeric",
        ]:
            return self.format_bivariate(result)
        else:
            return self.format_univariate(result)
