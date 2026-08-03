# Iris Flowers Dataset

A classic dataset demonstrating StatQA's **multimodal Q/A database generation** with the famous Iris flower measurements. This example generates 39 enhanced Q/A pairs with rich visual metadata and 15 publication-quality visualizations.

## Dataset

The Iris dataset contains measurements of 150 iris flowers from three species:
- Iris setosa
- Iris versicolor
- Iris virginica

## Files

- `data.csv`: Flower measurements (sepal length, sepal width, petal length, petal width, species)
- `codebook.json`: Variable metadata and descriptions
- `run_analysis.py`: Complete multimodal analysis script
- `qa_pairs.jsonl`: **39 multimodal Q/A pairs** with visual metadata
- `insights.json`: Statistical analysis results
- `plots/`: **15 publication-quality visualizations** (histograms, scatter plots, box plots, bar charts)

## Quick Start

**Run the complete multimodal analysis:**
```bash
python run_analysis.py
```

This generates:
- Statistical insights for all 5 variables
- 15 publication-quality plots in `plots/`
- 39 enhanced Q/A pairs with visual metadata in `qa_pairs.jsonl`

## Enhanced Q/A Format

Each Q/A pair includes rich multimodal metadata:

```json
{
  "question": "What is the distribution of Sepal Length?",
  "answer": "**Sepal Length**: mean=5.84, median=5.80, std=0.83, range=[4.30, 7.90]. N=150 [non-normal distribution].",
  "type": "distributional",
  "provenance": {
    "generated_at": "2025-11-19T19:21:28+00:00",
    "tool": "statqa",
    "tool_version": "0.2.0",
    "python_commands": ["valid_data.mean()  # Result: 5.84", "valid_data.std()  # Result: 0.83"]
  },
  "visual": {
    "plot_type": "histogram",
    "caption": "Histogram showing sepal length distribution with mean=5.84 and std=0.83 (N=150). The data shows a approximately normal distribution.",
    "alt_text": "Histogram chart with sepal length values on x-axis and frequency density on y-axis.",
    "visual_elements": {
      "chart_type": "histogram",
      "colors": ["blue bars", "red mean line"],
      "key_features": ["distribution shape", "mean line"]
    },
    "primary_plot": "plots/univariate_sepal_length.png"
  },
  "vars": ["sepal_length"]
}
```

## Custom Usage

```python
import pandas as pd
import json
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.qa import QAGenerator
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.model import Codebook

# Load data and codebook
data = pd.read_csv("data.csv")
with open("codebook.json") as f:
    codebook_dict = json.load(f)
codebook = Codebook.from_dict(codebook_dict)

# Generate multimodal Q/A pairs
analyzer = UnivariateAnalyzer()
formatter = InsightFormatter()
qa_gen = QAGenerator()

for var_name, variable in codebook.variables.items():
    if var_name in data.columns:
        result = analyzer.analyze(data[var_name], variable)
        insight = formatter.format_univariate(result)

        # Generate visual metadata
        plot_data = {
            "data": data,
            "variables": codebook.variables,
            "output_path": f"plots/univariate_{var_name}.png",
        }
        visual_metadata = qa_gen.generate_visual_metadata(
            result, variables=[var_name], plot_data=plot_data
        )

        # Generate enhanced Q/A pairs
        qa_pairs = qa_gen.generate_qa_pairs(
            result, insight, variables=[var_name], visual_data=visual_metadata
        )
        print(f"Generated {len(qa_pairs)} Q/A pairs for {var_name}")
```

## Generated Visualizations

The example creates 15 plots in the `plots/` directory:

**Univariate (5 plots):**
- `univariate_sepal_length.png`: Histogram of sepal length distribution
- `univariate_sepal_width.png`: Histogram of sepal width distribution
- `univariate_petal_length.png`: Histogram of petal length distribution
- `univariate_petal_width.png`: Histogram of petal width distribution
- `univariate_species.png`: Bar chart of species frequencies

**Bivariate (10 plots):**
- `bivariate_sepal_length_petal_length.png`: Scatter plot showing strong correlation (r=0.87)
- `bivariate_petal_length_petal_width.png`: Scatter plot showing very strong correlation (r=0.96)
- `bivariate_sepal_length_species.png`: Box plots comparing sepal length across species
- And 7 more relationship visualizations...

## Use Cases

This multimodal dataset is perfect for:
- **CLIP-style multimodal AI training** with visual-text pairs
- **Accessibility research** with comprehensive alt-text and captions
- **Statistical education** with reproducible analysis examples
- **Visualization benchmarking** with publication-quality plots
- **Question-answering systems** trained on statistical data

## Next Steps

- Explore the enhanced Q/A pairs in `qa_pairs.jsonl`
- Use the visual metadata for multimodal AI training
- Compare this multimodal approach with traditional text-only datasets
- Adapt the pipeline for your own datasets with visualization requirements
