#!/usr/bin/env python3
"""
Simple Survey Analysis - Quick Start Example

Demonstrates tableqa framework with synthetic survey data.
No external data files required - everything is generated on-the-fly.

Usage:
    python quick_start.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from statqa.analysis.bivariate import BivariateAnalyzer
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.schema import Codebook, Variable, VariableType
from statqa.visualization.plots import PlotFactory


def generate_synthetic_survey(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic survey data for demonstration.

    Args:
        n: Number of respondents
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic survey responses
    """
    np.random.seed(seed)

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 81, n),
            "gender": np.random.choice(["Male", "Female"], n),
            "education": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.4, 0.2, 0.1]
            ),
            "income": np.random.randint(20000, 200000, n),
            "satisfaction": np.random.randint(1, 6, n),
            "political_interest": np.random.randint(1, 6, n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "year": np.random.choice([2020, 2021, 2022, 2023], n),
        }
    )

    # Add some correlations for interesting insights
    # Education → Income correlation
    edu_income_map = {
        "High School": 50000,
        "Bachelor": 70000,
        "Master": 95000,
        "PhD": 140000,
    }
    base_income = data["education"].map(edu_income_map)
    data["income"] = (
        (base_income + np.random.normal(0, 20000, n)).clip(20000, 200000).astype(int)
    )

    # Income → Satisfaction correlation
    income_effect = (data["income"] - data["income"].mean()) / data["income"].std()
    satisfaction_base = 3 + 0.5 * income_effect + np.random.normal(0, 0.8, n)
    data["satisfaction"] = satisfaction_base.clip(1, 5).round().astype(int)

    # Age → Political interest correlation
    age_effect = (data["age"] - data["age"].mean()) / data["age"].std()
    interest_base = 3 + 0.3 * age_effect + np.random.normal(0, 0.9, n)
    data["political_interest"] = interest_base.clip(1, 5).round().astype(int)

    return data


def create_codebook() -> Codebook:
    """
    Create a codebook with metadata for all survey variables.

    Returns:
        Codebook object with variable definitions
    """
    variables = {
        "age": Variable(
            name="age",
            label="Respondent Age",
            var_type=VariableType.NUMERIC_CONTINUOUS,
            description="Age of respondent in years",
        ),
        "gender": Variable(
            name="gender",
            label="Gender",
            var_type=VariableType.CATEGORICAL_NOMINAL,
            description="Self-reported gender",
            valid_values={"Male": "Male", "Female": "Female"},
        ),
        "education": Variable(
            name="education",
            label="Education Level",
            var_type=VariableType.CATEGORICAL_ORDINAL,
            description="Highest level of education completed",
            valid_values={
                "High School": "High school diploma or equivalent",
                "Bachelor": "Bachelor's degree",
                "Master": "Master's degree",
                "PhD": "Doctoral degree",
            },
        ),
        "income": Variable(
            name="income",
            label="Annual Income",
            var_type=VariableType.NUMERIC_CONTINUOUS,
            description="Total annual household income in USD",
        ),
        "satisfaction": Variable(
            name="satisfaction",
            label="Job Satisfaction",
            var_type=VariableType.CATEGORICAL_ORDINAL,
            description="Overall job satisfaction rating",
            valid_values={
                "1": "Very Dissatisfied",
                "2": "Dissatisfied",
                "3": "Neutral",
                "4": "Satisfied",
                "5": "Very Satisfied",
            },
        ),
        "political_interest": Variable(
            name="political_interest",
            label="Political Interest",
            var_type=VariableType.CATEGORICAL_ORDINAL,
            description="Level of interest in politics and current events",
            valid_values={
                "1": "Not at all interested",
                "2": "Slightly interested",
                "3": "Moderately interested",
                "4": "Very interested",
                "5": "Extremely interested",
            },
        ),
        "region": Variable(
            name="region",
            label="Geographic Region",
            var_type=VariableType.CATEGORICAL_NOMINAL,
            description="Region of residence",
            valid_values={
                "North": "Northern region",
                "South": "Southern region",
                "East": "Eastern region",
                "West": "Western region",
            },
        ),
        "year": Variable(
            name="year",
            label="Survey Year",
            var_type=VariableType.NUMERIC_DISCRETE,
            description="Year the survey was conducted",
        ),
    }

    return Codebook(
        name="Simple Survey",
        description="Synthetic survey data for demonstration purposes",
        variables=variables,
    )


def main():
    """Run the complete analysis pipeline."""
    print("=" * 70)
    print("TableQA Simple Survey Example")
    print("=" * 70)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic survey data...")
    data = generate_synthetic_survey(n=1000)
    print(f"✓ Generated {len(data):,} respondents with {len(data.columns)} variables")

    # Save data
    data_path = output_dir / "survey_data.csv"
    data.to_csv(data_path, index=False)
    print(f"✓ Saved data to {data_path}")

    # Step 2: Create codebook
    print("\n[2/5] Creating codebook...")
    codebook = create_codebook()
    print(f"✓ Created codebook with {len(codebook.variables)} variables")

    # Save codebook
    codebook_path = output_dir / "codebook.json"
    with open(codebook_path, "w") as f:
        json.dump(
            {
                var.name: var.model_dump(mode="json")
                for var in codebook.variables.values()
            },
            f,
            indent=2,
        )
    print(f"✓ Saved codebook to {codebook_path}")

    # Step 3: Run univariate analyses
    print("\n[3/5] Running univariate analyses...")
    univariate_analyzer = UnivariateAnalyzer()
    formatter = InsightFormatter()
    plotter = PlotFactory(style="whitegrid", figsize=(8, 6))

    insights = []
    for var_name, variable in codebook.variables.items():
        try:
            result = univariate_analyzer.analyze(data[var_name], variable)

            # Generate visualization
            fig_path = output_dir / f"univariate_{var_name}.png"
            plotter.plot_univariate(data[var_name], variable, output_path=str(fig_path))

            # Format insight
            insight_text = formatter.format_univariate(result)

            insights.append(
                {
                    "type": "univariate",
                    "vars": [var_name],
                    "insight": insight_text,
                    "figure": str(fig_path),
                }
            )

        except Exception as e:
            print(f"  ⚠ Skipped {var_name}: {e}")

    print(f"✓ Completed {len(insights)} univariate analyses")

    # Step 4: Run bivariate analyses
    print("\n[4/5] Running bivariate analyses...")
    bivariate_analyzer = BivariateAnalyzer()

    var_list = list(codebook.variables.keys())
    bivariate_count = 0

    for i in range(len(var_list)):
        for j in range(i + 1, len(var_list)):
            var1_name = var_list[i]
            var2_name = var_list[j]

            var1 = codebook.variables[var1_name]
            var2 = codebook.variables[var2_name]

            try:
                result = bivariate_analyzer.analyze(data, var1, var2)

                # Generate visualization (if applicable)
                fig_path = None
                if result.get("visualization_type"):
                    fig_path = output_dir / f"bivariate_{var1_name}_{var2_name}.png"
                    plotter.plot_bivariate(data, var1, var2, output_path=str(fig_path))

                # Format insight
                insight_text = formatter.format_bivariate(result)

                insights.append(
                    {
                        "type": "bivariate",
                        "vars": [var1_name, var2_name],
                        "insight": insight_text,
                        "figure": str(fig_path) if fig_path else None,
                    }
                )

                bivariate_count += 1

            except Exception as exc:
                print(f"  skipped {var1_name} x {var2_name}: {exc}")

    print(f"✓ Completed {bivariate_count} bivariate analyses")

    # Step 5: Save insights
    print("\n[5/5] Saving results...")
    insights_path = output_dir / "insights.json"
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)

    print(f"✓ Saved {len(insights)} total insights to {insights_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"  Total insights: {len(insights)}")
    print(f"  Univariate: {len([i for i in insights if i['type'] == 'univariate'])}")
    print(f"  Bivariate: {len([i for i in insights if i['type'] == 'bivariate'])}")
    print(f"  Output directory: {output_dir}/")
    print()
    print("Next steps:")
    print("  1. Browse output/ to see generated files")
    print("  2. Open insights.json to see all findings")
    print("  3. View *.png files for visualizations")
    print("  4. Try modifying this script to customize the analysis")
    print("=" * 70)

    # Print a few example insights
    print("\nExample Insights:")
    print("-" * 70)
    for insight in insights[:5]:
        print(f"• {insight['insight'][:100]}...")
        if insight["figure"]:
            print(f"  → Figure: {Path(insight['figure']).name}")
        print()


if __name__ == "__main__":
    main()
