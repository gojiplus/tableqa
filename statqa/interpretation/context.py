"""
Context building for LLM-aware analysis.

Builds rich context strings that include metadata, statistical results,
and domain knowledge for use in LLM prompts.
"""

from typing import Any

from statqa.metadata.schema import Codebook, Variable


class ContextBuilder:
    """
    Builds context strings for LLM prompting.

    Args:
        include_metadata: Whether to include variable metadata
        max_variables: Maximum variables to include in context
    """

    def __init__(self, include_metadata: bool = True, max_variables: int = 50) -> None:
        self.include_metadata = include_metadata
        self.max_variables = max_variables

    def build_dataset_context(self, codebook: Codebook) -> str:
        """
        Build high-level dataset context.

        Args:
            codebook: Codebook with metadata

        Returns:
            Context string describing the dataset
        """
        context_parts = []

        # Dataset description
        if codebook.description:
            context_parts.append(f"Dataset: {codebook.description}")

        # Variable summary
        n_vars = len(codebook.variables)
        n_numeric = len(codebook.get_numeric_variables())
        n_categorical = len(codebook.get_categorical_variables())

        context_parts.append(
            f"Variables: {n_vars} total ({n_numeric} numeric, {n_categorical} categorical)"
        )

        # Data generating process
        dgp_counts: dict[str, int] = {}
        for var in codebook.variables.values():
            dgp = var.dgp.value
            dgp_counts[dgp] = dgp_counts.get(dgp, 0) + 1

        if dgp_counts and dgp_counts.get("unknown", 0) < len(codebook.variables):
            dominant_dgp = max(
                (k for k in dgp_counts if k != "unknown"),
                key=lambda k: dgp_counts[k],
                default="unknown",
            )
            if dominant_dgp != "unknown":
                context_parts.append(f"Primary data source: {dominant_dgp}")

        # Treatment/outcome variables
        treatments = codebook.get_treatment_variables()
        outcomes = codebook.get_outcome_variables()

        if treatments:
            treatment_names = [v.label for v in treatments[:3]]
            context_parts.append(f"Treatment variables: {', '.join(treatment_names)}")

        if outcomes:
            outcome_names = [v.label for v in outcomes[:3]]
            context_parts.append(f"Outcome variables: {', '.join(outcome_names)}")

        return ". ".join(context_parts) + "."

    def build_variable_context(self, variable: Variable, detailed: bool = True) -> str:
        """
        Build context for a single variable.

        Args:
            variable: Variable metadata
            detailed: Include detailed metadata

        Returns:
            Context string describing the variable
        """
        parts = [f"{variable.label} ({variable.name})"]

        if detailed:
            # Type
            parts.append(f"Type: {variable.var_type.value}")

            # Description
            if variable.description:
                parts.append(f"Description: {variable.description}")

            # Values for categorical
            if variable.is_categorical() and variable.valid_values:
                values_str = ", ".join(
                    [f"{k}: {v}" for k, v in list(variable.valid_values.items())[:5]]
                )
                if len(variable.valid_values) > 5:
                    values_str += ", ..."
                parts.append(f"Categories: {values_str}")

            # Range for numeric
            if variable.is_numeric():
                if variable.range_min is not None or variable.range_max is not None:
                    range_str = f"Range: [{variable.range_min or '?'}, {variable.range_max or '?'}]"
                    parts.append(range_str)
                if variable.units:
                    parts.append(f"Units: {variable.units}")

            # Role
            roles = []
            if variable.is_treatment:
                roles.append("treatment")
            if variable.is_outcome:
                roles.append("outcome")
            if variable.is_confounder:
                roles.append("confounder")
            if roles:
                parts.append(f"Role: {', '.join(roles)}")

        return ". ".join(parts)

    def build_analysis_context(
        self,
        variables: list[Variable],
        codebook: Codebook | None = None,
    ) -> str:
        """
        Build context for analysis involving specific variables.

        Args:
            variables: Variables involved in analysis
            codebook: Optional codebook for dataset context

        Returns:
            Context string for analysis
        """
        parts = []

        # Dataset context if available
        if codebook:
            parts.append(self.build_dataset_context(codebook))

        # Variable context
        if len(variables) == 1:
            parts.append("Analyzing: " + self.build_variable_context(variables[0]))
        else:
            var_summary = ", ".join([f"{v.label} ({v.var_type.value})" for v in variables])
            parts.append(f"Analyzing relationship between: {var_summary}")

        return "\n\n".join(parts)

    def build_codebook_summary(self, codebook: Codebook, max_variables: int | None = None) -> str:
        """
        Build comprehensive codebook summary for LLM context.

        Args:
            codebook: Codebook to summarize
            max_variables: Maximum variables to include (uses instance default if None)

        Returns:
            Formatted codebook summary
        """
        max_vars = max_variables or self.max_variables

        lines = []
        lines.append(f"# {codebook.name}")

        if codebook.description:
            lines.append(f"\n{codebook.description}")

        lines.append(f"\n## Variables ({len(codebook.variables)} total)")

        # List variables (up to max)
        for _i, (var_name, var) in enumerate(list(codebook.variables.items())[:max_vars]):
            lines.append(f"\n### {var.label} ({var_name})")
            lines.append(f"- Type: {var.var_type.value}")

            if var.description:
                lines.append(f"- Description: {var.description}")

            if var.is_categorical() and var.valid_values:
                lines.append("- Categories:")
                for code, label in list(var.valid_values.items())[:10]:
                    lines.append(f"  - {code}: {label}")

            if var.is_numeric():
                if var.range_min is not None and var.range_max is not None:
                    lines.append(f"- Range: [{var.range_min}, {var.range_max}]")
                if var.units:
                    lines.append(f"- Units: {var.units}")

            if var.missing_values:
                missing_str = ", ".join([str(m) for m in list(var.missing_values)[:5]])
                lines.append(f"- Missing codes: {missing_str}")

        if len(codebook.variables) > max_vars:
            lines.append(f"\n... and {len(codebook.variables) - max_vars} more variables")

        return "\n".join(lines)

    def build_insight_prompt(
        self,
        analysis_result: dict[str, Any],
        variables: list[Variable],
        task: str = "interpret",
    ) -> str:
        """
        Build prompt for LLM to interpret or enhance analysis results.

        Args:
            analysis_result: Statistical analysis result
            variables: Variables involved
            task: Task type ('interpret', 'enhance', 'question')

        Returns:
            LLM prompt string

        Raises:
            ValueError: If task type is not supported
        """
        if task == "interpret":
            return self._build_interpretation_prompt(analysis_result, variables)
        elif task == "enhance":
            return self._build_enhancement_prompt(analysis_result, variables)
        elif task == "question":
            return self._build_question_prompt(analysis_result, variables)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _build_interpretation_prompt(
        self, result: dict[str, Any], variables: list[Variable]
    ) -> str:
        """Build prompt for result interpretation."""
        var_context = ", ".join([v.label for v in variables])

        prompt = f"""Interpret this statistical analysis result in plain language.

Variables: {var_context}

Analysis Result:
{self._format_result_for_prompt(result)}

Provide a clear, accurate interpretation suitable for a non-technical audience.
Include:
1. What the result means in practical terms
2. The strength of the finding
3. Any important caveats or limitations

Keep the interpretation concise (2-3 sentences).
"""
        return prompt

    def _build_enhancement_prompt(self, result: dict[str, Any], variables: list[Variable]) -> str:
        """Build prompt for enhancing analysis with domain knowledge."""
        var_context = "\n".join(
            [f"- {v.label}: {v.description or 'No description'}" for v in variables]
        )

        prompt = f"""Enhance this statistical finding with domain knowledge and context.

Variables:
{var_context}

Statistical Result:
{self._format_result_for_prompt(result)}

Provide:
1. Potential explanations for this finding
2. Practical implications
3. Suggestions for follow-up analyses

Format as a brief expert commentary (3-4 sentences).
"""
        return prompt

    def _build_question_prompt(self, result: dict[str, Any], variables: list[Variable]) -> str:
        """Build prompt for generating questions from results."""
        var_context = ", ".join([v.label for v in variables])

        prompt = f"""Based on this statistical finding, generate insightful follow-up questions.

Variables: {var_context}

Finding:
{self._format_result_for_prompt(result)}

Generate 3-5 research questions that:
1. Probe deeper into this relationship
2. Explore potential mechanisms
3. Identify confounders or moderators
4. Suggest practical applications

Format as a numbered list.
"""
        return prompt

    def _format_result_for_prompt(self, result: dict[str, Any]) -> str:
        """Format analysis result for LLM prompt."""
        # Extract key information
        lines = []

        if "analysis_type" in result:
            lines.append(f"Analysis Type: {result['analysis_type']}")

        if "n" in result:
            lines.append(f"Sample Size: {result['n']}")

        # Type-specific formatting
        if "pearson" in result:
            lines.append(
                f"Correlation: r={result['pearson']['r']:.3f}, p={result['pearson']['p_value']:.3f}"
            )

        if "t_test" in result:
            lines.append(
                f"T-test: t={result['t_test']['statistic']:.3f}, p={result['t_test']['p_value']:.3f}"
            )

        if "anova" in result:
            lines.append(
                f"ANOVA: F={result['anova']['f_statistic']:.3f}, p={result['anova']['p_value']:.3f}"
            )

        if "treatment_effect" in result:
            te = result["treatment_effect"]
            lines.append(f"Treatment Effect: β={te['coefficient']:.3f}, p={te['p_value']:.3f}")

        if "mann_kendall" in result:
            mk = result["mann_kendall"]
            lines.append(f"Trend: {mk['trend']} (τ={mk['tau']:.3f}, p={mk['p_value']:.3f})")

        return "\n".join(lines)
