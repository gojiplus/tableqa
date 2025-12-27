"""
LLM-based metadata enrichment.

Uses language models to verify, infer, and enrich variable metadata including:
- Type inference and validation
- Relationship suggestions
- Causal structure hints
- Missing pattern detection
- Variable importance ranking
"""

import json
from typing import Any, Literal

from statqa.exceptions import EnrichmentError, LLMConnectionError, LLMResponseError
from statqa.utils.logging import get_logger


try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from statqa.metadata.schema import Codebook, Variable, VariableType


logger = get_logger(__name__)


class MetadataEnricher:
    """
    Enrich metadata using LLM capabilities.

    Supports both OpenAI and Anthropic models.

    Args:
        provider: LLM provider ('openai' or 'anthropic')
        model: Model name (defaults to gpt-4 or claude-3-sonnet)
        api_key: API key (or use environment variable)
        **kwargs: Additional provider-specific options

    Raises:
        ImportError: If required LLM package not installed
        ValueError: If provider is not supported
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.provider = provider.lower()
        self.kwargs = kwargs

        match self.provider:
            case "openai":
                if not HAS_OPENAI:
                    raise ImportError("openai package required. Install with: pip install openai")
                self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
                self.model = model or "gpt-4"
            case "anthropic":
                if not HAS_ANTHROPIC:
                    raise ImportError(
                        "anthropic package required. Install with: pip install anthropic"
                    )
                self.client = (
                    anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
                )
                self.model = model or "claude-3-sonnet-20240229"
            case _:
                raise ValueError(f"Unknown provider: {provider}")

    def enrich_variable(self, variable: Variable, dataset_context: str | None = None) -> Variable:
        """
        Enrich a single variable's metadata.

        Args:
            variable: Variable to enrich
            dataset_context: Optional context about the dataset

        Returns:
            Enriched Variable with updated metadata

        Raises:
            EnrichmentError: If enrichment process fails
            LLMConnectionError: If LLM connection fails
            LLMResponseError: If LLM response is invalid
        """
        prompt = self._build_variable_prompt(variable, dataset_context)

        try:
            response = self._call_llm(prompt)
            enrichment = self._parse_enrichment_response(response)

            # Update variable with enriched data
            if enrichment:
                variable.enriched_metadata.update(enrichment)

                # Apply high-confidence suggestions
                if "suggested_type" in enrichment and enrichment.get("type_confidence", 0) > 0.8:
                    suggested_type = enrichment["suggested_type"]
                    if suggested_type in VariableType.__members__:
                        variable.var_type = VariableType[suggested_type]

                if "is_treatment" in enrichment:
                    variable.is_treatment = enrichment["is_treatment"]
                if "is_outcome" in enrichment:
                    variable.is_outcome = enrichment["is_outcome"]
                if "is_confounder" in enrichment:
                    variable.is_confounder = enrichment["is_confounder"]

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"LLM connection failed for variable {variable.name}: {e}")
            raise LLMConnectionError(f"Failed to connect to LLM service: {e}") from e
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response for variable {variable.name}: {e}")
            raise LLMResponseError(f"Invalid LLM response format: {e}") from e
        except Exception as e:
            logger.warning(f"Unexpected error enriching variable {variable.name}: {e}")
            raise EnrichmentError(f"Unexpected enrichment error: {e}") from e

        return variable

    def enrich_codebook(self, codebook: Codebook) -> Codebook:
        """
        Enrich entire codebook metadata.

        Args:
            codebook: Codebook to enrich

        Returns:
            Enriched Codebook
        """
        # Get dataset-level context first
        dataset_context = self._get_dataset_context(codebook)

        # Enrich each variable
        for var_name, variable in codebook.variables.items():
            logger.info(f"Enriching variable: {var_name}")
            enriched = self.enrich_variable(variable, dataset_context)
            codebook.variables[var_name] = enriched

        # Get dataset-level relationships
        relationships = self._infer_relationships(codebook)
        codebook.dataset_info["inferred_relationships"] = relationships

        return codebook

    def _build_variable_prompt(self, variable: Variable, dataset_context: str | None = None) -> str:
        """Build prompt for variable enrichment."""
        context = f"\nDataset context: {dataset_context}" if dataset_context else ""

        prompt = f"""Analyze this survey/dataset variable and provide enriched metadata.

Variable Information:
- Name: {variable.name}
- Label: {variable.label}
- Current Type: {variable.var_type.value}
- Description: {variable.description or "Not provided"}
- Valid Values: {variable.valid_values or "Not provided"}
- Missing Values: {variable.missing_values or "Not provided"}
- Units: {variable.units or "Not provided"}{context}

Please provide a JSON response with the following enrichments:

1. **Type Verification**:
   - suggested_type: Most appropriate VariableType
   - type_confidence: Confidence score (0-1)
   - type_reasoning: Why this type is appropriate

2. **Role Inference**:
   - is_treatment: Is this likely a treatment/intervention variable? (boolean)
   - is_outcome: Is this likely an outcome/dependent variable? (boolean)
   - is_confounder: Is this likely a confounder? (boolean)
   - role_reasoning: Explanation of variable's role

3. **Relationships**:
   - likely_related_to: List of variable names this might correlate with
   - causal_direction: Suggested causal direction if applicable

4. **Quality Assessment**:
   - data_quality_concerns: Any potential issues
   - recommended_transformations: Suggested preprocessing

Return ONLY a valid JSON object with these fields.
"""
        return prompt

    def _get_dataset_context(self, codebook: Codebook) -> str:
        """Extract high-level dataset context."""
        var_summary = f"{len(codebook.variables)} variables"
        numeric_count = len(codebook.get_numeric_variables())
        categorical_count = len(codebook.get_categorical_variables())

        return (
            f"{codebook.description or 'Dataset'} with {var_summary} "
            f"({numeric_count} numeric, {categorical_count} categorical)"
        )

    def _infer_relationships(self, codebook: Codebook) -> dict[str, Any]:
        """Infer relationships between variables."""
        # Build summary of all variables
        var_list = [
            f"- {v.name}: {v.label} ({v.var_type.value})"
            for v in list(codebook.variables.values())[:50]  # Limit to avoid token limits
        ]

        prompt = f"""Given these variables from a dataset, suggest likely relationships and causal structure.

Variables:
{chr(10).join(var_list)}

Provide JSON with:
- treatment_variables: List of likely treatment/intervention variables
- outcome_variables: List of likely outcome variables
- confounders: List of likely confounding variables
- temporal_ordering: Suggested temporal order if applicable
- interaction_candidates: Pairs of variables that might have interactions

Return ONLY valid JSON.
"""

        try:
            response = self._call_llm(prompt, max_tokens=800)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Failed to infer relationships: {e}")
            return {}

    def _call_llm(self, prompt: str, max_tokens: int = 600) -> str:
        """Call the configured LLM."""
        match self.provider:
            case "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a statistical data analyst helping to understand and classify dataset variables.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,  # Lower temperature for more deterministic output
                    **self.kwargs,
                )
                return response.choices[0].message.content or ""

            case "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    system="You are a statistical data analyst helping to understand and classify dataset variables.",
                    messages=[{"role": "user", "content": prompt}],
                    **self.kwargs,
                )
                return response.content[0].text
            case _:
                raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_enrichment_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into enrichment data."""
        try:
            # Try to extract JSON from response
            # LLMs sometimes wrap JSON in markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enrichment response: {e}")
            logger.debug(f"Response was: {response}")
            return {}
