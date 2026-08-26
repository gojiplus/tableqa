"""Bivariate statistical analysis.

Analyzes relationships between pairs of variables:
- Numeric x Numeric: Pearson/Spearman correlation, regression
- Categorical x Categorical: Chi-square, Cramér's V
- Categorical x Numeric: Group comparisons, ANOVA
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from statqa.metadata.schema import Variable
from statqa.utils.cleaning import blank_missing_codes_frame
from statqa.utils.logging import get_logger
from statqa.utils.stats import calculate_effect_size, cramers_v

logger = get_logger(__name__)


class BivariateAnalyzer:
    """Analyzer for two-variable relationships."""

    def __init__(
        self,
        significance_level: float = 0.05,
        min_sample_size: int = 10,
        use_robust: bool = True,
    ) -> None:
        """Initialize the bivariate analyzer.

        Args:
            significance_level: Alpha level for statistical tests
            min_sample_size: Minimum sample size for analysis
            use_robust: Use robust methods (Spearman) when appropriate
        """
        self.alpha = significance_level
        self.min_n = min_sample_size
        self.use_robust = use_robust

    def analyze(
        self,
        data: pd.DataFrame,
        var1: Variable,
        var2: Variable,
    ) -> dict[str, Any] | None:
        """Analyze relationship between two variables.

        Args:
            data: DataFrame containing both variables
            var1: First variable metadata
            var2: Second variable metadata

        Returns:
            Analysis results, or None if analysis not applicable
        """
        logger.debug(
            f"Analyzing relationship: '{var1.name}' ({var1.var_type}) vs '{var2.name}' ({var2.var_type})"
        )

        # Extract and clean data
        subset = data[[var1.name, var2.name]].copy()  # type: ignore[assignment]
        subset = self._clean_data(subset, var1, var2)

        # Check minimum sample size
        if len(subset.dropna()) < self.min_n:
            return None

        # Route to appropriate analysis based on variable types
        if var1.is_numeric() and var2.is_numeric():
            return self._analyze_numeric_numeric(subset, var1, var2)
        if var1.is_categorical() and var2.is_categorical():
            return self._analyze_categorical_categorical(subset, var1, var2)
        if var1.is_categorical() and var2.is_numeric():
            return self._analyze_categorical_numeric(subset, var1, var2)
        if var1.is_numeric() and var2.is_categorical():
            # Swap order
            return self._analyze_categorical_numeric(
                subset[[var2.name, var1.name]], var2, var1
            )

        return None

    def _clean_data(
        self, data: pd.DataFrame, var1: Variable, var2: Variable
    ) -> pd.DataFrame:
        """Replace both variables' missing codes with NaN."""
        return blank_missing_codes_frame(data, (var1, var2))

    def _analyze_numeric_numeric(
        self, data: pd.DataFrame, var1: Variable, var2: Variable
    ) -> dict[str, Any]:
        """Analyze correlation between two numeric variables."""
        clean_data = data.dropna()

        if len(clean_data) < self.min_n:
            return {"error": "Insufficient data"}

        x = clean_data[var1.name]
        y = clean_data[var2.name]

        result: dict[str, Any] = {
            "analysis_type": "numeric_numeric",
            "var1": var1.name,
            "var2": var2.name,
            "n": len(clean_data),
        }

        # Pearson correlation. Read .statistic/.pvalue rather than unpacking:
        # the tuple form is untyped, so every downstream float() reads as an
        # error even though the values are floats.
        pearson = stats.pearsonr(x, y)
        r_pearson = float(pearson.statistic)
        p_pearson = float(pearson.pvalue)
        result["pearson"] = {
            "r": r_pearson,
            "p_value": p_pearson,
            "significant": bool(p_pearson < self.alpha),
        }

        # Spearman correlation (robust to outliers and non-linearity)
        if self.use_robust:
            spearman = stats.spearmanr(x, y)
            r_spearman = float(spearman.statistic)
            p_spearman = float(spearman.pvalue)
            result["spearman"] = {
                "rho": r_spearman,
                "p_value": p_spearman,
                "significant": bool(p_spearman < self.alpha),
            }

        # Effect size (convert r to Cohen's d)
        if abs(r_pearson) > 0.01:  # Avoid division by zero
            try:
                d = calculate_effect_size(r_pearson, effect_type="r_to_d")
                result["effect_size"] = {
                    "cohens_d": float(d),
                    "interpretation": self._interpret_cohens_d(d),
                }
            except Exception as exc:
                logger.debug("Cohen's d conversion from Pearson r failed: %s", exc)

        # Strength interpretation
        result["strength"] = self._interpret_correlation(abs(r_pearson))

        return result

    def _analyze_categorical_categorical(
        self, data: pd.DataFrame, var1: Variable, var2: Variable
    ) -> dict[str, Any]:
        """Analyze association between two categorical variables."""
        clean_data = data.dropna()

        if len(clean_data) < self.min_n:
            return {"error": "Insufficient data"}

        # Create contingency table
        contingency = pd.crosstab(clean_data[var1.name], clean_data[var2.name])

        # Chi-square test
        chi2_result = stats.chi2_contingency(contingency)
        chi2 = float(chi2_result.statistic)
        p_value = float(chi2_result.pvalue)
        dof = int(chi2_result.dof)
        expected = chi2_result.expected_freq

        result: dict[str, Any] = {
            "analysis_type": "categorical_categorical",
            "var1": var1.name,
            "var2": var2.name,
            "n": len(clean_data),
            "contingency_table": contingency.to_dict(),
            "chi_square": {
                "statistic": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "significant": bool(p_value < self.alpha),
            },
        }

        # Cramér's V (effect size for categorical associations)
        v = cramers_v(contingency)
        result["cramers_v"] = {
            "value": float(v),
            "interpretation": self._interpret_cramers_v(v),
        }

        # Check assumptions
        min_expected = expected.min()
        result["assumptions"] = {
            "min_expected_frequency": float(min_expected),
            "assumption_met": bool(min_expected >= 5),
        }

        return result

    def _analyze_categorical_numeric(
        self, data: pd.DataFrame, var_cat: Variable, var_num: Variable
    ) -> dict[str, Any]:
        """Analyze numeric variable across categorical groups."""
        clean_data = data.dropna()

        if len(clean_data) < self.min_n:
            return {"error": "Insufficient data"}

        groups = clean_data.groupby(var_cat.name)[var_num.name]

        # Group statistics
        group_stats = groups.agg(["count", "mean", "std", "median"]).round(3)

        result: dict[str, Any] = {
            "analysis_type": "categorical_numeric",
            "var_categorical": var_cat.name,
            "var_numeric": var_num.name,
            "n": len(clean_data),
            "n_groups": len(groups),
            "group_stats": group_stats.to_dict(),
        }

        # Map category codes to labels
        if var_cat.valid_values:
            labeled_stats = {}
            for code, stats_dict in group_stats.to_dict()["mean"].items():
                label = var_cat.valid_values.get(code, str(code))
                labeled_stats[label] = stats_dict
            result["group_means_labeled"] = labeled_stats

        # ANOVA (if more than 2 groups) or t-test (if 2 groups)
        group_data = [group.dropna() for _, group in groups]

        if len(group_data) == 2:
            # Two-sample t-test
            ttest = stats.ttest_ind(group_data[0], group_data[1])
            t_stat = float(ttest.statistic)
            p_value = float(ttest.pvalue)
            result["t_test"] = {
                "statistic": t_stat,
                "p_value": p_value,
                "significant": bool(p_value < self.alpha),
            }

            # Effect size (Cohen's d)
            d = calculate_effect_size(
                group_data[0], group_data[1], effect_type="cohen_d"
            )
            result["effect_size"] = {
                "cohens_d": float(d),
                "interpretation": self._interpret_cohens_d(d),
            }

        elif len(group_data) > 2:
            # One-way ANOVA
            anova = stats.f_oneway(*group_data)
            f_stat = float(anova.statistic)
            p_value = float(anova.pvalue)
            result["anova"] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": bool(p_value < self.alpha),
            }

            # Eta-squared (effect size for ANOVA)
            # SS_between / SS_total
            grand_mean = clean_data[var_num.name].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_data)
            ss_total = sum((clean_data[var_num.name] - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            result["effect_size"] = {
                "eta_squared": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared),
            }

        return result

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        if r < 0.3:
            return "weak"
        if r < 0.5:
            return "moderate"
        if r < 0.7:
            return "strong"
        return "very strong"

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"

    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        if v < 0.3:
            return "weak"
        if v < 0.5:
            return "moderate"
        return "strong"

    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        if eta2 < 0.06:
            return "small"
        if eta2 < 0.14:
            return "medium"
        return "large"

    def batch_analyze(
        self,
        df: pd.DataFrame,
        variables: dict[str, Variable],
        max_pairs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze multiple variable pairs.

        Args:
            df: DataFrame with data
            variables: Mapping of variable names to metadata
            max_pairs: Maximum number of pairs to analyze (None for all)

        Returns:
            List of analysis results as dictionaries
        """
        results = []
        var_list = list(variables.values())

        count = 0
        for i, var1 in enumerate(var_list):
            for var2 in var_list[i + 1 :]:
                if max_pairs and count >= max_pairs:
                    return results

                if var1.name in df.columns and var2.name in df.columns:
                    result = self.analyze(df, var1, var2)
                    if result:
                        results.append(result)
                        count += 1

        return results
