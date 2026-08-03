# StatQA Examples

This directory contains examples demonstrating various features of the statqa library, including **multimodal Q/A database generation** with rich visual metadata and publication-quality plots.

## Quick Start

Start with `basic_usage.py` for a simple introduction to the core concepts.

## Examples Overview

### Basic Usage (`basic_usage.py`)
A standalone script showing the fundamental statqa workflow:
- Creating codebooks from text
- Running univariate and bivariate analyses
- Generating insights and Q/A pairs

**Best for**: First-time users, understanding core concepts

### Simple Survey (`simple_survey/`)
Generate and analyze synthetic survey data.
- Quick setup (no downloads required)
- Demonstrates end-to-end pipeline
- Good for testing and experimentation

**Best for**: Learning the full pipeline, prototyping

### Iris Dataset (`iris/`)
Classic dataset with flower measurements.
- Small, well-understood dataset
- Continuous numeric variables
- Multi-class classification context
- **39 multimodal Q/A pairs** with 15 visualizations

**Best for**: Testing statistical analyses, visualization, multimodal Q/A generation

### Employee Survey (`employee/`)
Synthetic employee workplace survey.
- Mix of categorical and numeric variables
- Realistic survey structure
- Common workplace metrics
- **35 multimodal Q/A pairs** with 15 visualizations

**Best for**: Understanding survey analysis, practicing with mixed data types, multimodal datasets

### Titanic Dataset (`titanic/`)
Historical passenger survival data.
- Binary outcome (survival)
- Mix of demographics and ticket information
- Well-known dataset for comparison
- **28 multimodal Q/A pairs** with 15 visualizations

**Best for**: Classification analysis, survival analysis, CLIP-style training data

### ANES Dataset (`anes/`)
American National Election Studies political survey (1948-2020).
- Large-scale real-world dataset
- Complex codebook with 1000+ variables
- Demonstrates advanced features

**Best for**: Production use cases, large-scale analysis, research applications

## Running Examples

### Basic Usage
```bash
python examples/basic_usage.py
```

### Simple Survey
```bash
cd examples/simple_survey
python quick_start.py
```

### Dataset Examples

**Quick Analysis:**
```bash
# Iris
cd examples/iris
python run_analysis.py
```

**Custom Analysis:**
```bash
cd examples/iris
python -c "
import pandas as pd
import json
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.schema import Codebook

data = pd.read_csv('data.csv')
with open('codebook.json') as f:
    codebook = Codebook.from_dict(json.load(f))

analyzer = UnivariateAnalyzer()
formatter = InsightFormatter()

for var_name, variable in codebook.variables.items():
    if var_name in data.columns:
        result = analyzer.analyze(data[var_name], variable)
        print(formatter.format_univariate(result))
"
```

**Generated Output:**
Each example produces:
- `qa_pairs.jsonl`: Enhanced Q/A pairs with visual metadata
- `plots/`: Publication-quality visualizations (PNG files)
- `insights.json`: Statistical analysis results

### ANES (Advanced)
```bash
cd examples/anes

# Step 1: Parse metadata (requires PDF files)
python parse_metadata.py \
    --codebook data/raw/anes_timeseries_cdf_codebook_var_20220916.pdf \
    --output-metadata data/anes_metadata.csv \
    --skip-questions

# Step 2: Extract insights
python extract_insights.py \
    --data-zip data/raw/anes_timeseries_cdf_csv_20220916.csv.zip \
    --metadata data/anes_metadata.csv \
    --output-dir output \
    --max-vars 50
```

## Complexity Levels

| Example | Complexity | Setup Time | Best For |
|---------|-----------|------------|----------|
| `basic_usage.py` | Beginner | < 1 min | Learning basics |
| `simple_survey/` | Beginner | < 1 min | Full pipeline |
| `iris/` | Beginner | < 1 min | Statistics |
| `employee/` | Intermediate | < 1 min | Survey analysis |
| `titanic/` | Intermediate | < 1 min | Classification |
| `anes/` | Advanced | ~10 min | Research |

## Common Workflows

### Generate Q/A Pairs
```python
from statqa.qa.generator import QAGenerator

qa_gen = QAGenerator(use_llm=False)  # Template-based
qa_pairs = qa_gen.generate_qa_pairs(result, answer)

for qa in qa_pairs:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}\n")
```

### Export Insights
```python
from statqa.utils.io import save_json

save_json(insights, "insights.json")
```

### Create Visualizations
```python
from statqa.visualization.plots import PlotFactory

plotter = PlotFactory(style="whitegrid")

# Univariate takes the column and its metadata; bivariate takes the frame
# and both variables.
plotter.plot_univariate(
    data["sepal_length"], sepal_length_var, output_path="distribution.png"
)
plotter.plot_bivariate(
    data, sepal_length_var, petal_length_var, output_path="relationship.png"
)
```

## File Structure

```
examples/
├── README.md              # This file
├── basic_usage.py         # Simple starting point
├── iris/                  # Iris flowers dataset
│   ├── README.md
│   ├── data.csv
│   ├── codebook.json
│   ├── run_analysis.py    # Complete multimodal analysis
│   ├── qa_pairs.jsonl     # 39 Q/A pairs with visual metadata
│   ├── insights.json      # Statistical results
│   └── plots/             # 15 visualizations (PNG files)
├── employee/              # Employee survey data
│   ├── README.md
│   ├── data.csv
│   ├── codebook.json
│   ├── run_analysis.py    # Complete multimodal analysis
│   ├── qa_pairs.jsonl     # 35 Q/A pairs with visual metadata
│   ├── insights.json      # Statistical results
│   └── plots/             # 15 visualizations (PNG files)
├── titanic/               # Titanic survival data
│   ├── README.md
│   ├── data.csv
│   ├── codebook.json
│   ├── run_analysis.py    # Complete multimodal analysis
│   ├── qa_pairs.jsonl     # 28 Q/A pairs with visual metadata
│   ├── insights.json      # Statistical results
│   └── plots/             # 15 visualizations (PNG files)
├── anes/                  # ANES political survey (advanced)
│   ├── README.md
│   ├── parse_metadata.py
│   ├── extract_insights.py
│   ├── data/
│   └── output/
└── simple_survey/         # Synthetic survey generation
    ├── README.md
    └── quick_start.py
```

## Getting Help

- **Documentation**: https://gojiplus.github.io/statqa
- **Issues**: https://github.com/gojiplus/statqa/issues

## Next Steps

1. Start with `basic_usage.py` to learn the fundamentals
2. Try `simple_survey/` for a complete workflow
3. **Run multimodal examples**: `cd iris && python run_analysis.py` for enhanced Q/A generation
4. **Explore visual metadata**: Check the `plots/` directories and `qa_pairs.jsonl` format
5. Explore domain-specific examples (iris, employee, titanic) for different data types
6. **Build CLIP-style datasets**: Use the multimodal Q/A pairs for AI training
7. Tackle `anes/` for advanced, real-world usage
8. Adapt examples to your own datasets with visualization requirements
