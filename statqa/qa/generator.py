"""
Q/A pair generation from statistical insights.

Converts facts into multiple question/answer pairs using:
1. Template-based generation
2. LLM paraphrasing and augmentation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from statqa.qa.templates import QuestionTemplate, infer_question_type
from statqa.utils.logging import get_logger


logger = get_logger(__name__)


def _get_statqa_version() -> str:
    """Get the statqa package version."""
    try:
        from importlib.metadata import version

        return version("statqa")
    except Exception:
        return "unknown"


class QAGenerator:
    """
    Generates Q/A pairs from statistical insights.

    Args:
        use_llm: Whether to use LLM for paraphrasing
        llm_provider: LLM provider ('openai' or 'anthropic')
        llm_model: Model name
        api_key: API key for LLM
        paraphrase_count: Number of paraphrased versions per question

    Raises:
        ImportError: If required LLM package not installed
        ValueError: If LLM provider configuration is invalid
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: Literal["openai", "anthropic"] = "openai",
        llm_model: str | None = None,
        api_key: str | None = None,
        paraphrase_count: int = 2,
    ) -> None:
        self.use_llm = use_llm
        self.paraphrase_count = paraphrase_count

        if use_llm:
            if llm_provider == "openai":
                if not HAS_OPENAI:
                    raise ImportError("openai required for LLM features")
                self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
                self.model = llm_model or "gpt-4"
            else:
                raise ValueError(
                    f"LLM provider {llm_provider} not yet supported for Q/A generation"
                )

    def _create_provenance(
        self,
        insight: dict[str, Any],
        method: str = "template",
        variables: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create provenance metadata for a Q/A pair.

        Args:
            insight: Statistical analysis result
            method: Generation method ('template' or 'llm_paraphrase')
            variables: List of variable names involved in the analysis

        Returns:
            Dictionary with provenance information
        """
        provenance = {
            "generated_at": datetime.now(UTC).isoformat(),
            "tool": "statqa",
            "tool_version": _get_statqa_version(),
            "generation_method": method,
        }

        # Add variables if provided
        if variables:
            provenance["variables"] = variables

        # Add analyzer information if available
        if "analyzer" in insight:
            provenance["analyzer"] = insight["analyzer"]

        # Add computational commands for reproducibility
        if "computation_log" in insight:
            provenance["python_commands"] = insight["computation_log"]

        # Add LLM information if used
        if method == "llm_paraphrase" and self.use_llm:
            provenance["llm_model"] = getattr(self, "model", None)

        return provenance

    def generate_qa_pairs(
        self,
        insight: dict[str, Any],
        formatted_answer: str,
        variables: list[str] | None = None,
        visual_data: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """
        Generate Q/A pairs from a statistical insight.

        Args:
            insight: Statistical analysis result
            formatted_answer: Natural language answer
            variables: List of variable names involved in the analysis
            visual_data: Optional visual metadata to include with Q/A pairs

        Returns:
            List of Q/A pair dictionaries with keys: question, answer, type, provenance, visual
        """
        # Infer question type
        q_type = infer_question_type(insight)

        # Generate template-based questions
        template = QuestionTemplate(q_type)
        qa_pairs = template.generate(insight, formatted_answer)

        # Add provenance to template-based questions
        template_provenance = self._create_provenance(
            insight, method="template", variables=variables
        )
        for qa in qa_pairs:
            qa["provenance"] = template_provenance.copy()

            # Add variables at top level for easy access
            if variables:
                qa["variables"] = variables

            # Add visual data if provided
            if visual_data:
                qa["visual"] = visual_data.copy()

        # LLM paraphrasing
        if self.use_llm and qa_pairs:
            try:
                paraphrased = self._paraphrase_questions(qa_pairs, insight, variables, visual_data)
                qa_pairs.extend(paraphrased)
            except Exception as e:
                logger.warning(f"LLM paraphrasing failed: {e}")

        return qa_pairs

    def generate_batch(
        self, insights: list[dict[str, Any]], formatted_answers: list[str]
    ) -> list[dict[str, Any]]:
        """
        Generate Q/A pairs for multiple insights.

        Args:
            insights: List of statistical insights
            formatted_answers: Corresponding natural language answers

        Returns:
            List of insight dictionaries with added 'qa_pairs' field
        """
        results = []

        for insight, answer in zip(insights, formatted_answers):
            qa_pairs = self.generate_qa_pairs(insight, answer)

            result = insight.copy()
            result["formatted_answer"] = answer
            result["qa_pairs"] = qa_pairs
            results.append(result)

        return results

    def _paraphrase_questions(
        self,
        qa_pairs: list[dict[str, str]],
        insight: dict[str, Any],
        variables: list[str] | None = None,
        visual_data: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """
        Use LLM to generate paraphrased questions.

        Args:
            qa_pairs: Original Q/A pairs
            insight: Statistical insight for context
            variables: List of variable names involved in the analysis
            visual_data: Optional visual metadata for multimodal context

        Returns:
            List of paraphrased Q/A pairs
        """
        # Take first few original questions
        original_questions = [qa["question"] for qa in qa_pairs[:3]]
        answer = qa_pairs[0]["answer"] if qa_pairs else ""

        prompt = f"""Given these questions about a statistical finding, generate {self.paraphrase_count} natural paraphrases for each.

Original Questions:
{chr(10).join(f"{i + 1}. {q}" for i, q in enumerate(original_questions))}

Answer (for context):
{answer}

Generate paraphrased questions that:
1. Ask for the same information in different ways
2. Vary in formality and structure
3. Could include domain-specific terminology
4. Remain clear and answerable

Return as JSON array with format:
[
  {{"original": "question 1", "paraphrases": ["p1", "p2"]}},
  ...
]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are helping create a diverse Q/A dataset for data analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.7,  # Higher temperature for diversity
            )

            content = response.choices[0].message.content or ""

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            paraphrase_data = json.loads(content.strip())

            # Build Q/A pairs from paraphrases
            paraphrased_pairs = []
            llm_provenance = self._create_provenance(
                insight, method="llm_paraphrase", variables=variables
            )

            for item in paraphrase_data:
                for paraphrase in item.get("paraphrases", []):
                    qa_pair = {
                        "question": paraphrase,
                        "answer": answer,
                        "type": qa_pairs[0]["type"] if qa_pairs else "descriptive",
                        "source": "llm_paraphrase",
                        "provenance": llm_provenance.copy(),
                    }

                    # Add variables at top level for easy access
                    if variables:
                        qa_pair["variables"] = variables

                    # Add visual data if provided
                    if visual_data:
                        qa_pair["visual"] = visual_data.copy()

                    paraphrased_pairs.append(qa_pair)

            return paraphrased_pairs

        except Exception as e:
            logger.warning(f"Failed to paraphrase questions: {e}")
            return []

    def generate_exploratory_questions(
        self, insight: dict[str, Any], context: str | None = None
    ) -> list[str]:
        """
        Generate exploratory follow-up questions using LLM.

        Args:
            insight: Statistical insight
            context: Optional dataset/domain context

        Returns:
            List of exploratory questions
        """
        if not self.use_llm:
            return []

        context_str = f"\n\nContext: {context}" if context else ""

        prompt = f"""Based on this statistical finding, generate 5 insightful follow-up questions that would deepen understanding.

Finding:
{json.dumps(insight, indent=2)}{context_str}

Generate questions that:
1. Explore mechanisms or explanations
2. Identify potential confounders or moderators
3. Suggest practical implications
4. Consider alternative explanations
5. Propose related analyses

Return as a JSON array of strings: ["question 1", "question 2", ...]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research methodologist helping design data analysis studies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.8,
            )

            content = response.choices[0].message.content or ""

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            questions = json.loads(content.strip())
            return questions if isinstance(questions, list) else []

        except Exception as e:
            logger.warning(f"Failed to generate exploratory questions: {e}")
            return []

    def generate_visual_metadata(
        self,
        insight: dict[str, Any],
        variables: list[str] | None = None,
        plot_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate visual metadata for a statistical insight.

        Args:
            insight: Statistical analysis result
            variables: List of variable names involved in the analysis
            plot_data: Optional plot data (data and variable objects)

        Returns:
            Visual metadata dictionary or None if no visualization appropriate
        """
        if not plot_data or not variables:
            return None

        # Get analysis info for future use
        _ = insight.get("analysis_type", "unknown")
        _ = infer_question_type(insight)

        # Import PlotFactory here to avoid circular imports
        from statqa.visualization.plots import PlotFactory

        plot_factory = PlotFactory()

        try:
            if len(variables) == 1:
                # Univariate analysis
                var_name = variables[0]
                data_series = plot_data["data"][var_name]
                variable_obj = plot_data["variables"][var_name]

                output_path = plot_data.get("output_path", f"plots/univariate_{var_name}.png")

                fig, metadata = plot_factory.plot_univariate(
                    data_series, variable_obj, output_path, return_metadata=True
                )

                # Close figure to free memory
                import matplotlib.pyplot as plt

                plt.close(fig)

                # Rationalize the visual metadata structure
                return self._rationalize_visual_metadata(metadata, output_path)

            elif len(variables) == 2:
                # Bivariate analysis
                var1_name, var2_name = variables[:2]
                dataframe = plot_data["data"]
                var1_obj = plot_data["variables"][var1_name]
                var2_obj = plot_data["variables"][var2_name]

                output_path = plot_data.get(
                    "output_path", f"plots/bivariate_{var1_name}_{var2_name}.png"
                )

                fig, metadata = plot_factory.plot_bivariate(
                    dataframe, var1_obj, var2_obj, output_path, return_metadata=True
                )

                # Close figure to free memory
                import matplotlib.pyplot as plt

                plt.close(fig)

                # Rationalize the visual metadata structure
                return self._rationalize_visual_metadata(metadata, output_path)

        except Exception as e:
            logger.warning(f"Failed to generate visualization for {variables}: {e}")

        return None

    def _rationalize_visual_metadata(
        self, metadata: dict[str, Any], output_path: str | Path
    ) -> dict[str, Any]:
        """
        Rationalize visual metadata to a simpler, flatter structure.

        Args:
            metadata: Original metadata from PlotFactory
            output_path: Path to the generated plot file

        Returns:
            Simplified visual metadata dictionary
        """
        # Convert absolute path to relative path
        path_obj = Path(output_path)
        if path_obj.is_absolute():
            # Try to make it relative to current working directory
            try:
                relative_path = path_obj.relative_to(Path.cwd())
                file_path = str(relative_path)
            except ValueError:
                # If can't make relative, just use the filename with plots/ prefix
                file_path = f"plots/{path_obj.name}"
        else:
            file_path = str(path_obj)

        # Extract visual elements and flatten structure
        visual_elements = metadata.get("visual_elements", {})
        features = visual_elements.get("key_features", [])
        if isinstance(features, list):
            features = [feature.replace(" ", "_") for feature in features]

        # Create simplified structure
        rationalized = {
            "type": metadata.get("plot_type", "unknown"),
            "file": file_path,
            "caption": metadata.get("caption", ""),
            "alt_text": metadata.get("alt_text", ""),
            "features": features,
        }

        return rationalized

    def export_qa_dataset(
        self, qa_results: list[dict[str, Any]], output_format: str = "jsonl"
    ) -> list[str]:
        """
        Export Q/A pairs in format suitable for LLM fine-tuning.

        Args:
            qa_results: Results from generate_batch
            output_format: 'jsonl', 'openai', or 'anthropic'

        Returns:
            List of formatted strings (one per line for JSONL)
        """
        lines = []

        for result in qa_results:
            for qa in result.get("qa_pairs", []):
                if output_format == "jsonl":
                    lines.append(json.dumps(qa, ensure_ascii=False))

                elif output_format == "openai":
                    # OpenAI fine-tuning format
                    entry = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a data analyst answering questions about statistical findings.",
                            },
                            {"role": "user", "content": qa["question"]},
                            {"role": "assistant", "content": qa["answer"]},
                        ]
                    }
                    lines.append(json.dumps(entry, ensure_ascii=False))

                elif output_format == "anthropic":
                    # Anthropic format (simpler)
                    entry = {
                        "prompt": qa["question"],
                        "completion": qa["answer"],
                    }
                    lines.append(json.dumps(entry, ensure_ascii=False))

        return lines
