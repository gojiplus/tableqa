"""Run analysis on Employee Survey dataset and generate insights."""

import json
from pathlib import Path

import pandas as pd

from statqa.analysis.bivariate import BivariateAnalyzer
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.schema import Codebook, Variable
from statqa.qa.generator import QAGenerator

# Get script directory
script_dir = Path(__file__).parent
plots_dir = script_dir / "plots"
plots_dir.mkdir(exist_ok=True)

# Load data and codebook
data = pd.read_csv(script_dir / "data.csv")
with open(script_dir / "codebook.json") as f:
    variables_dict = json.load(f)

# Convert variables dict to Codebook
variables = {}
for var_name, var_data in variables_dict.items():
    # Fix missing_values if it's a string representation of set()
    if isinstance(var_data.get("missing_values"), str):
        var_data["missing_values"] = set()
    variables[var_name] = Variable(**var_data)

codebook = Codebook(name="employee_survey", variables=variables)

print(f"Loaded {len(data)} rows and {len(codebook.variables)} variables")

# Initialize analyzers
univ_analyzer = UnivariateAnalyzer()
biv_analyzer = BivariateAnalyzer()
formatter = InsightFormatter()
qa_gen = QAGenerator(use_llm=False)

# Run univariate analyses
insights = []
all_qa_pairs = []

print("\nRunning univariate analyses...")
for var_name, variable in codebook.variables.items():
    if var_name not in data.columns:
        continue

    result = univ_analyzer.analyze(data[var_name], variable)
    if result:
        insight_text = formatter.format_univariate(result)
        insights.append(
            {"vars": [var_name], "insight": insight_text, "type": "univariate"}
        )

        # Generate visual metadata
        plot_data = {
            "data": data,
            "variables": codebook.variables,
            "output_path": plots_dir / f"univariate_{var_name}.png",
        }
        visual_metadata = qa_gen.generate_visual_metadata(
            result, variables=[var_name], plot_data=plot_data
        )

        # Generate QA pairs with visual data
        qa_pairs = qa_gen.generate_qa_pairs(
            result, insight_text, variables=[var_name], visual_data=visual_metadata
        )
        all_qa_pairs.extend(qa_pairs)

        print(f"  ✓ {var_name}")

# Run bivariate analyses
print("\nRunning bivariate analyses...")
var_list = [v for v in codebook.variables.values() if v.name in data.columns]
for i, var1 in enumerate(var_list):
    for var2 in var_list[i + 1 :]:
        result = biv_analyzer.analyze(data, var1, var2)
        if result:
            insight_text = formatter.format_bivariate(result)
            insights.append(
                {
                    "vars": [var1.name, var2.name],
                    "insight": insight_text,
                    "type": "bivariate",
                }
            )

            # Generate visual metadata
            plot_data = {
                "data": data,
                "variables": codebook.variables,
                "output_path": plots_dir / f"bivariate_{var1.name}_{var2.name}.png",
            }
            visual_metadata = qa_gen.generate_visual_metadata(
                result, variables=[var1.name, var2.name], plot_data=plot_data
            )

            # Generate QA pairs with visual data
            qa_pairs = qa_gen.generate_qa_pairs(
                result,
                insight_text,
                variables=[var1.name, var2.name],
                visual_data=visual_metadata,
            )
            all_qa_pairs.extend(qa_pairs)

            print(f"  ✓ {var1.name} x {var2.name}")

# Save insights
insights_path = script_dir / "insights.json"
with open(insights_path, "w") as f:
    json.dump(insights, f, indent=2)
print(f"\n✓ Saved {len(insights)} insights to {insights_path}")

# Save QA pairs
qa_path = script_dir / "qa_pairs.json"
with open(qa_path, "w") as f:
    json.dump(all_qa_pairs, f, indent=2)
print(f"✓ Saved {len(all_qa_pairs)} QA pairs to {qa_path}")

# Also save as JSONL
qa_jsonl_path = script_dir / "qa_pairs.jsonl"
with open(qa_jsonl_path, "w") as f:
    for qa in all_qa_pairs:
        f.write(json.dumps(qa) + "\n")
print(f"✓ Saved {len(all_qa_pairs)} QA pairs to {qa_jsonl_path}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
