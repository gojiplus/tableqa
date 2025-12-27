# Statistical Analysis

Learn how to run comprehensive statistical analyses with StatQA.

## Overview

StatQA provides four types of statistical analysis:

- **Univariate**: Single variable analysis (descriptive statistics)
- **Bivariate**: Two-variable relationships (correlations, tests)
- **Temporal**: Time series analysis (trends, change points)
- **Causal**: Causal inference with confounding control

## Univariate Analysis

Analyze individual variables:

```python
from statqa.analysis.univariate import UnivariateAnalyzer

analyzer = UnivariateAnalyzer(handle_outliers=True, robust=True)
result = analyzer.analyze(data['age'], variable_metadata)

print(result['statistics'])  # Mean, median, std, etc.
print(result['outliers'])    # Outlier detection
```

### CLI Usage

```bash
# Analyze all variables in dataset
statqa analyze data.csv codebook.json --output-dir results/
```

## Bivariate Analysis

Analyze relationships between variable pairs:

```python
from statqa.analysis.bivariate import BivariateAnalyzer

analyzer = BivariateAnalyzer(significance_level=0.05)
result = analyzer.analyze(data, var1, var2)

# Automatic method selection based on variable types:
# - Numeric × Numeric: Pearson/Spearman correlation
# - Categorical × Categorical: Chi-square test
# - Categorical × Numeric: ANOVA/t-tests
```

## Temporal Analysis

Analyze trends and patterns over time:

```python
from statqa.analysis.temporal import TemporalAnalyzer

analyzer = TemporalAnalyzer(significance_level=0.05)
result = analyzer.analyze_trend(data, time_var, value_var)

print(result['trend'])        # Direction and significance
print(result['change_points']) # Detected change points
```

## Causal Analysis

Perform observational causal analysis with confounding control:

```python
from statqa.analysis.causal import CausalAnalyzer

analyzer = CausalAnalyzer(robust_se=True)
result = analyzer.analyze_treatment_effect(
    data,
    treatment_var,
    outcome_var,
    control_vars=[age_var, gender_var]
)

print(result['effect_size'])   # Treatment effect
print(result['confidence_interval'])
```

## Batch Analysis

Process multiple variables efficiently:

```python
# Univariate analysis for all variables
results = analyzer.batch_analyze(data, variables)

# Bivariate analysis for all pairs
bivariate_results = bivariate_analyzer.batch_analyze(
    data, variables, max_pairs=100
)
```

## Result Interpretation

All analyses return structured dictionaries with:

- **statistics**: Core statistical measures
- **tests**: Hypothesis test results (p-values, effect sizes)
- **interpretation**: Natural language summaries
- **metadata**: Analysis configuration and provenance
