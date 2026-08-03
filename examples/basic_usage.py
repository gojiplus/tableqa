"""
Basic usage example for tableqa.

This example demonstrates:
1. Creating a codebook from text
2. Loading data
3. Running analyses
4. Generating insights and Q/A pairs
"""

import pandas as pd

from statqa.analysis.bivariate import BivariateAnalyzer
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.parsers.text import TextParser
from statqa.qa.generator import QAGenerator

# 1. Define codebook as text
codebook_text = """
# Codebook: Survey Data

# Variable: age
Label: Respondent Age
Type: numeric_continuous
Units: years
Range: 18-99
Missing: -1, 999
Description: Age of survey respondent in years

# Variable: gender
Label: Gender
Type: categorical_nominal
Values:
  1: Male
  2: Female
  3: Other
Missing: 0

# Variable: satisfaction
Label: Job Satisfaction
Type: categorical_ordinal
Values:
  1: Very Dissatisfied
  2: Dissatisfied
  3: Neutral
  4: Satisfied
  5: Very Satisfied
Missing: -1
"""

# 2. Parse codebook
parser = TextParser()
codebook = parser.parse(codebook_text)

print(f"Parsed {len(codebook.variables)} variables:")
for var_name, var in codebook.variables.items():
    print(f"  - {var.label} ({var_name}): {var.var_type}")

# 3. Create sample data
data = pd.DataFrame(
    {
        "age": [25, 32, 45, 28, 51, 39, 44, 29, 35, 48],
        "gender": [1, 2, 2, 1, 2, 1, 2, 2, 1, 2],
        "satisfaction": [4, 5, 3, 4, 2, 5, 4, 5, 3, 4],
    }
)

print(f"\nData shape: {data.shape}")

# 4. Run univariate analyses
print("\n" + "=" * 60)
print("UNIVARIATE ANALYSES")
print("=" * 60)

analyzer = UnivariateAnalyzer()
formatter = InsightFormatter()

for var_name in codebook.variables:
    if var_name in data.columns:
        result = analyzer.analyze(data[var_name], codebook.variables[var_name])
        insight = formatter.format_univariate(result)
        print(f"\n{insight}")

# 5. Run bivariate analysis
print("\n" + "=" * 60)
print("BIVARIATE ANALYSES")
print("=" * 60)

biv_analyzer = BivariateAnalyzer()
age_var = codebook.variables["age"]
satisfaction_var = codebook.variables["satisfaction"]
gender_var = codebook.variables["gender"]

# Age vs Satisfaction (correlation)
result1 = biv_analyzer.analyze(data, age_var, satisfaction_var)
if result1:
    insight1 = formatter.format_bivariate(result1)
    print(f"\n{insight1}")

# Gender vs Satisfaction (group comparison)
result2 = biv_analyzer.analyze(data, gender_var, satisfaction_var)
if result2:
    insight2 = formatter.format_bivariate(result2)
    print(f"\n{insight2}")

# 6. Generate Q/A pairs
print("\n" + "=" * 60)
print("GENERATED Q/A PAIRS")
print("=" * 60)

qa_gen = QAGenerator(use_llm=False)  # Template-based only

# Get first univariate result
result = analyzer.analyze(data["age"], age_var)
answer = formatter.format_univariate(result)

qa_pairs = qa_gen.generate_qa_pairs(result, answer)

print(f"\nGenerated {len(qa_pairs)} Q/A pairs for 'age' variable:\n")
for i, qa in enumerate(qa_pairs[:3], 1):
    print(f"Q{i}: {qa['question']}")
    print(f"A{i}: {qa['answer']}")
    print()

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
