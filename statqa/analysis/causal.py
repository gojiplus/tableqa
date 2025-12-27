"""
Causal analysis with confounding control.

Performs regression analysis with control variables to surface
associations in causal language:
- Linear regression with controls
- Logistic regression for binary outcomes
- Confounder identification
- Sensitivity analysis
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statqa.metadata.schema import Variable


class CausalAnalyzer:
    """
    Analyzer for causal relationships with confounding control.

    Note: These are *observational* analyses and do not establish true causation
    without strong assumptions. Results should be interpreted as associations
    controlling for measured confounders.

    Args:
        significance_level: Alpha level for hypothesis tests
        min_sample_size: Minimum sample size required
        robust_se: Use heteroskedasticity-robust standard errors
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_sample_size: int = 30,
        robust_se: bool = True,
    ) -> None:
        self.alpha = significance_level
        self.min_n = min_sample_size
        self.robust_se = robust_se

    def analyze_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment_var: Variable,
        outcome_var: Variable,
        control_vars: list[Variable] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate treatment effect on outcome with optional controls.

        Args:
            data: DataFrame with variables
            treatment_var: Treatment/exposure variable
            outcome_var: Outcome variable
            control_vars: List of control/confounder variables

        Returns:
            Treatment effect analysis results as dictionary
        """
        # Prepare data
        var_names = [treatment_var.name, outcome_var.name]
        if control_vars:
            var_names.extend([v.name for v in control_vars])

        subset = data[var_names].copy()
        subset = self._clean_data(subset, [treatment_var, outcome_var] + (control_vars or []))
        subset = subset.dropna()

        if len(subset) < self.min_n:
            return {"error": f"Insufficient sample size (n={len(subset)}, min={self.min_n})"}

        result: dict[str, Any] = {
            "analysis_type": "treatment_effect",
            "treatment": treatment_var.name,
            "outcome": outcome_var.name,
            "controls": [v.name for v in (control_vars or [])],
            "n": len(subset),
        }

        # Run regression
        if outcome_var.is_numeric():
            # Linear regression
            regression_result = self._linear_regression(
                subset, treatment_var.name, outcome_var.name, control_vars
            )
            result.update(regression_result)
        else:
            # Could implement logistic regression for binary outcomes
            return {"error": "Only numeric outcomes currently supported"}

        # Sensitivity analysis if controls provided
        if control_vars:
            sensitivity = self._sensitivity_analysis(
                subset, treatment_var.name, outcome_var.name, control_vars
            )
            result["sensitivity"] = sensitivity

        return result

    def identify_confounders(
        self,
        data: pd.DataFrame,
        treatment_var: Variable,
        outcome_var: Variable,
        potential_confounders: list[Variable],
    ) -> dict[str, Any]:
        """
        Identify which variables act as confounders.

        A confounder must:
        1. Be associated with treatment
        2. Be associated with outcome
        3. Not be on causal path between treatment and outcome

        Args:
            data: DataFrame with variables
            treatment_var: Treatment variable
            outcome_var: Outcome variable
            potential_confounders: List of potential confounders to test

        Returns:
            Confounder identification results as dictionary
        """
        result: dict[str, Any] = {
            "analysis_type": "confounder_identification",
            "treatment": treatment_var.name,
            "outcome": outcome_var.name,
            "tested_variables": [v.name for v in potential_confounders],
            "confounders": [],
            "non_confounders": [],
        }

        for var in potential_confounders:
            # Test association with treatment
            from statqa.analysis.bivariate import BivariateAnalyzer

            biv_analyzer = BivariateAnalyzer(significance_level=self.alpha)

            treat_assoc = biv_analyzer.analyze(data, var, treatment_var)
            outcome_assoc = biv_analyzer.analyze(data, var, outcome_var)

            if not treat_assoc or not outcome_assoc:
                continue

            # Check if significantly associated with both
            treat_sig = self._is_significant(treat_assoc)
            outcome_sig = self._is_significant(outcome_assoc)

            if treat_sig and outcome_sig:
                result["confounders"].append(
                    {
                        "variable": var.name,
                        "label": var.label,
                        "treatment_association": treat_sig,
                        "outcome_association": outcome_sig,
                    }
                )
            else:
                result["non_confounders"].append(
                    {
                        "variable": var.name,
                        "reason": "Not associated with both treatment and outcome",
                    }
                )

        return result

    def _linear_regression(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        control_vars: list[Variable] | None,
    ) -> dict[str, Any]:
        """Run linear regression and extract results."""
        # Prepare variables
        y = data[outcome_name]
        x_vars = [treatment_name]
        if control_vars:
            x_vars.extend([v.name for v in control_vars])

        x = data[x_vars]
        x = sm.add_constant(x)  # Add intercept

        # Fit model
        model = sm.OLS(y, x).fit(cov_type="HC3") if self.robust_se else sm.OLS(y, x).fit()

        # Extract treatment effect
        treatment_coef = model.params[treatment_name]
        treatment_se = model.bse[treatment_name]
        treatment_p = model.pvalues[treatment_name]
        treatment_ci = model.conf_int().loc[treatment_name]

        result = {
            "model": "linear_regression",
            "treatment_effect": {
                "coefficient": float(treatment_coef),
                "std_error": float(treatment_se),
                "p_value": float(treatment_p),
                "ci_lower": float(treatment_ci[0]),
                "ci_upper": float(treatment_ci[1]),
                "significant": bool(treatment_p < self.alpha),
            },
            "model_fit": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue),
                "f_pvalue": float(model.f_pvalue),
            },
        }

        # Control variable effects
        if control_vars:
            controls_info = {}
            for var in control_vars:
                if var.name in model.params.index:
                    controls_info[var.name] = {
                        "coefficient": float(model.params[var.name]),
                        "p_value": float(model.pvalues[var.name]),
                        "significant": bool(model.pvalues[var.name] < self.alpha),
                    }
            result["control_effects"] = controls_info

        # Diagnostics
        result["diagnostics"] = {
            "condition_number": float(np.linalg.cond(x)),
            "multicollinearity_warning": bool(np.linalg.cond(x) > 30),
        }

        return result

    def _sensitivity_analysis(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        control_vars: list[Variable],
    ) -> dict[str, Any]:
        """
        Perform sensitivity analysis by comparing models with/without controls.

        Args:
            data: DataFrame
            treatment_name: Treatment variable name
            outcome_name: Outcome variable name
            control_vars: Control variables

        Returns:
            Sensitivity analysis results as dictionary
        """
        # Model without controls
        y = data[outcome_name]
        x_no_controls = sm.add_constant(data[[treatment_name]])
        model_no_controls = sm.OLS(y, x_no_controls).fit()

        # Model with controls
        x_vars = [treatment_name] + [v.name for v in control_vars]
        x_with_controls = sm.add_constant(data[x_vars])
        model_with_controls = sm.OLS(y, x_with_controls).fit()

        # Compare treatment coefficients
        coef_no_controls = model_no_controls.params[treatment_name]
        coef_with_controls = model_with_controls.params[treatment_name]

        percent_change = (
            (coef_with_controls - coef_no_controls) / abs(coef_no_controls) * 100
            if coef_no_controls != 0
            else np.nan
        )

        return {
            "naive_effect": {
                "coefficient": float(coef_no_controls),
                "p_value": float(model_no_controls.pvalues[treatment_name]),
            },
            "adjusted_effect": {
                "coefficient": float(coef_with_controls),
                "p_value": float(model_with_controls.pvalues[treatment_name]),
            },
            "confounding": {
                "percent_change": float(percent_change) if not np.isnan(percent_change) else None,
                "direction": ("positive" if coef_with_controls > coef_no_controls else "negative"),
                "substantial": (
                    bool(abs(percent_change) > 10) if not np.isnan(percent_change) else False
                ),
            },
            "model_comparison": {
                "r_squared_increase": float(
                    model_with_controls.rsquared - model_no_controls.rsquared
                ),
            },
        }

    def _clean_data(self, data: pd.DataFrame, variables: list[Variable]) -> pd.DataFrame:
        """Clean missing values based on metadata."""
        clean = data.copy()
        for var in variables:
            if var.missing_values:
                clean[var.name] = clean[var.name].replace(dict.fromkeys(var.missing_values, np.nan))
        return clean

    def _is_significant(self, analysis_result: dict[str, Any]) -> bool:
        """Check if analysis result shows significant association."""
        # Check various p-value locations
        if "p_value" in analysis_result:
            return analysis_result["p_value"] < self.alpha
        if "pearson" in analysis_result:
            return analysis_result["pearson"]["significant"]
        if "chi_square" in analysis_result:
            return analysis_result["chi_square"]["significant"]
        if "t_test" in analysis_result:
            return analysis_result["t_test"]["significant"]
        if "anova" in analysis_result:
            return analysis_result["anova"]["significant"]
        return False
