"""
Question templates for Q/A pair generation.

Defines templates for converting facts into question/answer pairs.
"""

from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    """Types of questions that can be generated."""

    DESCRIPTIVE = "descriptive"  # What is the average...?
    COMPARATIVE = "comparative"  # How does X compare to Y...?
    TEMPORAL = "temporal"  # Has X changed over time...?
    CAUSAL = "causal"  # What is the effect of X on Y...?
    CORRELATIONAL = "correlational"  # Are X and Y related...?
    DISTRIBUTIONAL = "distributional"  # What is the distribution of...?


class QuestionTemplate:
    """
    Template for generating questions from statistical insights.

    Args:
        question_type: Type of question to generate
    """

    def __init__(self, question_type: QuestionType) -> None:
        self.question_type = question_type

    def generate(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """
        Generate question/answer pairs from an insight.

        Args:
            insight: Statistical insight dictionary
            answer: Formatted natural language answer

        Returns:
            List of Q/A pair dictionaries

        Raises:
            ValueError: If question type is not supported
        """
        if self.question_type == QuestionType.DESCRIPTIVE:
            return self._generate_descriptive(insight, answer)
        elif self.question_type == QuestionType.COMPARATIVE:
            return self._generate_comparative(insight, answer)
        elif self.question_type == QuestionType.TEMPORAL:
            return self._generate_temporal(insight, answer)
        elif self.question_type == QuestionType.CAUSAL:
            return self._generate_causal(insight, answer)
        elif self.question_type == QuestionType.CORRELATIONAL:
            return self._generate_correlational(insight, answer)
        elif self.question_type == QuestionType.DISTRIBUTIONAL:
            return self._generate_distributional(insight, answer)

        # This should never be reached due to enum constraint
        raise ValueError(f"Unknown question type: {self.question_type}")

    def _generate_descriptive(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """Generate descriptive questions (univariate statistics)."""
        questions = []
        var_label = insight.get("label", insight.get("variable", "Variable"))

        if "mean" in insight:
            questions.extend(
                [
                    {
                        "question": f"What is the average {var_label}?",
                        "answer": answer,
                        "type": "descriptive",
                    },
                    {
                        "question": f"What is the mean value of {var_label}?",
                        "answer": answer,
                        "type": "descriptive",
                    },
                    {
                        "question": f"Describe the central tendency of {var_label}.",
                        "answer": answer,
                        "type": "descriptive",
                    },
                ]
            )

        if "mode" in insight:
            questions.extend(
                [
                    {
                        "question": f"What is the most common category for {var_label}?",
                        "answer": answer,
                        "type": "descriptive",
                    },
                    {
                        "question": f"Which {var_label} value appears most frequently?",
                        "answer": answer,
                        "type": "descriptive",
                    },
                ]
            )

        return questions

    def _generate_comparative(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """Generate comparative questions (group comparisons)."""
        questions = []

        var_cat = insight.get("var_categorical")
        var_num = insight.get("var_numeric")

        if var_cat and var_num:
            questions.extend(
                [
                    {
                        "question": f"How does {var_num} differ across {var_cat} groups?",
                        "answer": answer,
                        "type": "comparative",
                    },
                    {
                        "question": f"What is the relationship between {var_cat} and {var_num}?",
                        "answer": answer,
                        "type": "comparative",
                    },
                    {
                        "question": f"Does {var_num} vary by {var_cat}?",
                        "answer": answer,
                        "type": "comparative",
                    },
                ]
            )

        return questions

    def _generate_temporal(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """Generate temporal questions (trends over time)."""
        questions = []

        value_var = insight.get("value_variable", "Variable")
        time_var = insight.get("time_variable", "time")

        questions.extend(
            [
                {
                    "question": f"How has {value_var} changed over {time_var}?",
                    "answer": answer,
                    "type": "temporal",
                },
                {
                    "question": f"Is there a trend in {value_var} over {time_var}?",
                    "answer": answer,
                    "type": "temporal",
                },
                {
                    "question": f"Has {value_var} increased or decreased over {time_var}?",
                    "answer": answer,
                    "type": "temporal",
                },
            ]
        )

        return questions

    def _generate_causal(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """Generate causal questions (treatment effects)."""
        questions = []

        treatment = insight.get("treatment", "Treatment")
        outcome = insight.get("outcome", "Outcome")
        controls = insight.get("controls", [])

        base_questions = [
            {
                "question": f"What is the effect of {treatment} on {outcome}?",
                "answer": answer,
                "type": "causal",
            },
            {
                "question": f"How does {treatment} affect {outcome}?",
                "answer": answer,
                "type": "causal",
            },
        ]

        if controls:
            controls_str = ", ".join(controls)
            base_questions.append(
                {
                    "question": f"Controlling for {controls_str}, what is the effect of {treatment} on {outcome}?",
                    "answer": answer,
                    "type": "causal",
                }
            )

        questions.extend(base_questions)
        return questions

    def _generate_correlational(self, insight: dict[str, Any], answer: str) -> list[dict[str, str]]:
        """Generate correlational questions (associations)."""
        questions = []

        var1 = insight.get("var1", "Variable 1")
        var2 = insight.get("var2", "Variable 2")

        questions.extend(
            [
                {
                    "question": f"Are {var1} and {var2} correlated?",
                    "answer": answer,
                    "type": "correlational",
                },
                {
                    "question": f"What is the relationship between {var1} and {var2}?",
                    "answer": answer,
                    "type": "correlational",
                },
                {
                    "question": f"How strongly are {var1} and {var2} associated?",
                    "answer": answer,
                    "type": "correlational",
                },
            ]
        )

        return questions

    def _generate_distributional(
        self, insight: dict[str, Any], answer: str
    ) -> list[dict[str, str]]:
        """Generate distributional questions (shape, spread)."""
        questions = []
        var_label = insight.get("label", insight.get("variable", "Variable"))

        if "std" in insight or "skewness" in insight:
            questions.extend(
                [
                    {
                        "question": f"What is the distribution of {var_label}?",
                        "answer": answer,
                        "type": "distributional",
                    },
                    {
                        "question": f"How variable is {var_label}?",
                        "answer": answer,
                        "type": "distributional",
                    },
                ]
            )

        if "frequencies" in insight:
            questions.append(
                {
                    "question": f"What is the frequency distribution of {var_label}?",
                    "answer": answer,
                    "type": "distributional",
                }
            )

        return questions


def infer_question_type(insight: dict[str, Any]) -> QuestionType:
    """
    Infer the appropriate question type from an insight.

    Args:
        insight: Statistical insight dictionary

    Returns:
        Inferred question type
    """
    analysis_type = insight.get("analysis_type", "")

    # Temporal
    if analysis_type in ["temporal_trend", "year_over_year"] or "mann_kendall" in insight:
        return QuestionType.TEMPORAL

    # Causal
    if analysis_type == "treatment_effect" or "treatment_effect" in insight:
        return QuestionType.CAUSAL

    # Correlational
    if analysis_type == "numeric_numeric" or "pearson" in insight:
        return QuestionType.CORRELATIONAL

    # Comparative
    if analysis_type == "categorical_numeric" or "group_stats" in insight:
        return QuestionType.COMPARATIVE

    # Distributional
    if "skewness" in insight or "frequencies" in insight:
        return QuestionType.DISTRIBUTIONAL

    # Default to descriptive
    return QuestionType.DESCRIPTIVE
