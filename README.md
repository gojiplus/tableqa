# StatQA: Extract Multimodal Stats Q/A from Tables With Provenance

[![PyPI version](https://img.shields.io/pypi/v/statqa.svg)](https://pypi.org/project/statqa/)
[![CI](https://github.com/gojiplus/statqa/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/statqa/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/statqa)](https://pepy.tech/project/statqa)
[![Documentation](https://github.com/gojiplus/statqa/actions/workflows/docs.yml/badge.svg)](https://gojiplus.github.io/statqa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StatQA** is a modern Python framework for automatically extracting structured facts, statistical insights, and **multimodal Q/A pairs** from tabular datasets. It converts raw columns and values into clear, human-readable statements paired with rich visualizations, enabling rapid knowledge discovery, CLIP-style multimodal RAG corpus construction, and LLM training.

## 🎯 Key Features

- **📋 Flexible Metadata Parsing**: Parse codebooks from text, CSV, or PDF formats
- **🤖 LLM-Powered Enrichment**: Automatically infer variable types and relationships
- **📊 Comprehensive Statistical Analysis**:
  - Univariate: descriptive statistics, distribution tests, robust estimators
  - Bivariate: correlations, chi-square, group comparisons with effect sizes
  - Temporal: trend detection (Mann-Kendall), change points, year-over-year analysis
  - Causal: regression with confounding control, sensitivity analysis
- **💬 Natural Language Insights**: Convert statistics to publication-ready text
- **❓ Multimodal Q/A Generation**: Create CLIP-style visual-text pairs with template-based and LLM-paraphrased questions
- **🖼️ Rich Visual Metadata**: Captions, alt-text, and visual elements for each plot (colors, annotations, features)
- **🔍 Provenance Tracking**: Full metadata for reproducibility (timestamps, tools, methods, analysis types, plot generation)
- **📈 Publication-Quality Visualizations**: Automated plots for all analyses with question-plot association mapping
- **🔬 Statistical Rigor**: Multiple testing correction, effect sizes, normality tests
- **⚡ Modern Python**: Type-safe (Pydantic), async-ready, fully typed

> **📖 For detailed documentation, tutorials, and API reference, visit [StatQA Documentation](https://gojiplus.github.io/statqa)**

## 📦 Installation

### Basic Installation

```bash
pip install statqa
```

### With Optional Features

```bash
# Include LLM support (OpenAI/Anthropic)
pip install statqa[llm]

# Include PDF parsing
pip install statqa[pdf]

# Development installation
pip install statqa[dev]

# Complete installation
pip install statqa[all]
```

### From Source

```bash
git clone https://github.com/gojiplus/statqa.git
cd statqa
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Create a Codebook

```python
from statqa.metadata.parsers import TextParser

codebook_text = """
# Variable: age
Label: Respondent Age
Type: numeric_continuous
Units: years
Range: 18-99
Missing: -1, 999

# Variable: satisfaction
Label: Job Satisfaction
Type: categorical_ordinal
Values:
  1: Very Dissatisfied
  2: Dissatisfied
  3: Neutral
  4: Satisfied
  5: Very Satisfied
"""

parser = TextParser()
codebook = parser.parse(codebook_text)
```

### 2. Run Statistical Analyses

```python
import pandas as pd
from statqa.analysis import UnivariateAnalyzer, BivariateAnalyzer

# Load your data
data = pd.read_csv("survey_data.csv")

# Univariate analysis
analyzer = UnivariateAnalyzer()
result = analyzer.analyze(data["age"], codebook.variables["age"])

print(result)
# Output: {'mean': 42.5, 'median': 41.0, 'std': 12.3, ...}

# Bivariate analysis
biv_analyzer = BivariateAnalyzer()
result = biv_analyzer.analyze(
    data,
    codebook.variables["age"],
    codebook.variables["satisfaction"]
)
```

### 3. Generate Natural Language Insights

```python
from statqa.interpretation import InsightFormatter

formatter = InsightFormatter()
insight = formatter.format_univariate(result)

print(insight)
# Output: "**Respondent Age**: mean=42.5, median=41.0, std=12.3, range=[18, 95]. N=1,000 [2.3% outliers]."
```

### 4. Create Multimodal Q/A Pairs for LLM Training

```python
from statqa.qa import QAGenerator
from statqa.visualization import PlotFactory

qa_gen = QAGenerator(use_llm=False)  # Template-based

# Generate Q/A pairs with visual metadata
plot_data = {
    "data": data,
    "variables": codebook.variables,
    "output_path": "plots/univariate_age.png"
}
visual_metadata = qa_gen.generate_visual_metadata(result, variables=["age"], plot_data=plot_data)
qa_pairs = qa_gen.generate_qa_pairs(result, insight, variables=["age"], visual_data=visual_metadata)

for qa in qa_pairs:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print(f"Plot: {qa['visual']['primary_plot']}")
    print(f"Caption: {qa['visual']['caption']}")
    print(f"Provenance: {qa['provenance']}\n")
```

Each Q/A pair includes **provenance metadata** and **visual metadata** tracking:
- **When** the answer was generated (timestamp)
- **What tool** was used (statqa version)
- **What compute** was performed (analysis type, analyzer, Python commands)
- **How** it was generated (template vs. LLM paraphrase)
- **Which LLM** was used (if applicable)
- **What visualization** was created (plot type, file path, generation code)
- **Visual elements** (captions, alt-text, colors, annotations, accessibility features)

## 🖼️ Multimodal Q/A Database

StatQA creates **CLIP-style multimodal databases** where each statistical question is paired with both textual answers AND rich visual metadata. This enables training of multimodal AI systems that understand both statistical text and visual representations.

### Enhanced Q/A Format

```json
{
  "question": "What is the distribution of Sepal Length?",
  "answer": "**Sepal Length**: mean=5.84, median=5.80, std=0.83, range=[4.30, 7.90]. N=150 [non-normal distribution].",
  "type": "distributional",
  "provenance": {
    "generated_at": "2025-11-19T19:21:28+00:00",
    "tool": "statqa",
    "tool_version": "0.2.0",
    "generation_method": "template",
    "analysis_type": "unknown",
    "variables": ["sepal_length"],
    "python_commands": ["valid_data.mean()  # Result: 5.84", "valid_data.std()  # Result: 0.83"]
  },
  "visual": {
    "plot_type": "histogram",
    "caption": "Histogram showing sepal length distribution with mean=5.84 and std=0.83 (N=150). The data shows a approximately normal distribution.",
    "alt_text": "Histogram chart with sepal length values on x-axis and frequency density on y-axis, showing distribution shape with 150 observations.",
    "visual_elements": {
      "chart_type": "histogram",
      "x_axis": "Sepal Length",
      "y_axis": "Density",
      "key_features": ["distribution shape", "mean line"],
      "colors": ["blue bars", "red mean line"],
      "annotations": ["Mean: 5.84"]
    },
    "primary_plot": "/path/to/univariate_sepal_length.png",
    "generation_code": "plot_factory.plot_univariate(data['sepal_length'], sepal_length_var, 'plot.png')"
  },
  "vars": ["sepal_length"]
}
```

### Question-Plot Association Mapping

StatQA automatically associates relevant visualizations with each statistical insight:

- **Distribution questions** → Histograms for numeric data, bar charts for categorical
- **Correlation questions** → Scatter plots with regression lines
- **Group comparison questions** → Box plots showing group differences
- **Categorical relationships** → Heatmaps with frequency counts

### Accessibility & Multimodal Features

Every visualization includes:
- **Descriptive captions** with statistical context and interpretation
- **Alt-text** for screen readers and accessibility compliance
- **Visual elements extraction** for computer vision training (colors, features, annotations)
- **Reproducible generation code** for programmatic recreation

## 🎨 Complete Pipeline Example

```python
from statqa import Codebook, UnivariateAnalyzer
from statqa.metadata.parsers import CSVParser
from statqa.interpretation import InsightFormatter
from statqa.qa import QAGenerator
from statqa.utils.io import load_data, save_json

# 1. Parse codebook
parser = CSVParser()
codebook = parser.parse("codebook.csv")

# 2. Load data
data = load_data("data.csv")

# 3. Run analyses
analyzer = UnivariateAnalyzer()
results = analyzer.batch_analyze(data, codebook.variables)

# 4. Format insights
formatter = InsightFormatter()
for result in results:
    result["insight"] = formatter.format_insight(result)

# 5. Generate multimodal Q/A pairs with visualizations
from pathlib import Path
qa_gen = QAGenerator(use_llm=True, api_key="your-api-key")
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

all_qa_pairs = []
for result in results:
    # Generate visual metadata
    plot_data = {
        "data": data,
        "variables": codebook.variables,
        "output_path": plots_dir / f"univariate_{result['variable']}.png"
    }
    visual_metadata = qa_gen.generate_visual_metadata(result, variables=[result['variable']], plot_data=plot_data)

    # Generate Q/A pairs with visual data
    qa_pairs = qa_gen.generate_qa_pairs(result, result["insight"], variables=[result['variable']], visual_data=visual_metadata)
    all_qa_pairs.extend(qa_pairs)

# 6. Export multimodal Q/A dataset
import json
with open("multimodal_qa_dataset.jsonl", "w") as f:
    for qa in all_qa_pairs:
        f.write(json.dumps(qa) + "\n")

# Export in OpenAI fine-tuning format (visual metadata preserved in messages)
lines = qa_gen.export_qa_dataset([{"qa_pairs": all_qa_pairs}], format="openai")
with open("training_data.jsonl", "w") as f:
    f.write("\n".join(lines))
```

## 📝 Q/A Provenance & Visual Tracking

Every Q/A pair generated by StatQA includes detailed **provenance metadata** and **visual metadata** to ensure reproducibility and traceability:

```json
{
  "question": "What is the average Respondent Age?",
  "answer": "The mean age is 42.5 years (median=41.0, std=12.3).",
  "type": "descriptive",
  "provenance": {
    "generated_at": "2025-11-19T10:30:45.123456+00:00",
    "tool": "statqa",
    "tool_version": "0.2.0",
    "generation_method": "template",
    "analysis_type": "univariate",
    "analyzer": "UnivariateAnalyzer",
    "variables": ["age"],
    "python_commands": ["valid_data.mean()  # Result: 42.5", "valid_data.std()  # Result: 12.3"]
  },
  "visual": {
    "plot_type": "histogram",
    "caption": "Histogram showing age distribution with mean=42.5 and std=12.3 (N=1000).",
    "alt_text": "Histogram chart with age values on x-axis and frequency density on y-axis.",
    "visual_elements": {
      "chart_type": "histogram",
      "x_axis": "Age",
      "y_axis": "Density",
      "colors": ["blue bars", "red mean line"],
      "key_features": ["distribution shape", "mean line"],
      "annotations": ["Mean: 42.5"]
    },
    "primary_plot": "plots/univariate_age.png",
    "generation_code": "plot_factory.plot_univariate(data['age'], age_var, 'plots/univariate_age.png')"
  }
}
```

### Metadata Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| **Provenance Fields** | | |
| `generated_at` | ISO 8601 timestamp (UTC) | `2025-11-19T10:30:45+00:00` |
| `tool` | Software used for generation | `statqa` |
| `tool_version` | Version of statqa | `0.2.0` |
| `generation_method` | How the Q/A was created | `template`, `llm_paraphrase` |
| `analysis_type` | Statistical analysis performed | `univariate`, `bivariate`, `temporal`, `causal` |
| `analyzer` | Specific analyzer class used | `UnivariateAnalyzer`, `BivariateAnalyzer` |
| `variables` | Variables involved in analysis | `["age"]`, `["age", "income"]` |
| `python_commands` | Computational commands executed | `["data.mean()  # Result: 42.5"]` |
| `llm_model` | LLM model (if applicable) | `gpt-4`, `claude-3-opus` |
| **Visual Fields** | | |
| `plot_type` | Type of visualization | `histogram`, `scatter`, `boxplot`, `heatmap` |
| `caption` | Descriptive caption with context | `"Histogram showing age distribution..."` |
| `alt_text` | Accessibility alt-text | `"Histogram chart with age values on x-axis..."` |
| `visual_elements` | Chart components and features | `{"colors": ["blue bars"], "annotations": [...]}` |
| `primary_plot` | Path to generated plot file | `"plots/univariate_age.png"` |
| `generation_code` | Code to reproduce the plot | `"plot_factory.plot_univariate(...)"` |

This comprehensive metadata tracking enables:
- **Reproducibility**: Recreate Q/A pairs and visualizations from original data
- **Quality Control**: Filter by generation method, analysis type, or plot quality
- **Multimodal Training**: Rich visual-text pairs for CLIP-style model training
- **Accessibility**: Alt-text and captions for inclusive AI applications
- **Auditing**: Track when and how answers and plots were computed
- **Citation**: Properly attribute computational and visualization methods in research

## 🖥️ Command-Line Interface

StatQA provides a powerful CLI for common workflows:

```bash
# Parse a codebook
statqa parse-codebook codebook.csv --output codebook.json --enrich

# Run full analysis pipeline with plots and visual metadata
statqa analyze data.csv codebook.json --output-dir results/ --plots --multimodal

# Generate multimodal Q/A pairs
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl --llm --visual-metadata

# Complete multimodal pipeline
statqa pipeline data.csv codebook.csv --output-dir output/ --enrich --qa --plots --multimodal
```

## 📊 Supported Analyses

### Univariate Statistics
- Central tendency: mean, median, mode
- Dispersion: std, IQR, MAD (robust)
- Distribution: skewness, kurtosis, normality tests
- Categorical: frequencies, entropy, diversity indices

### Bivariate Relationships
- **Numeric × Numeric**: Pearson/Spearman correlation, effect sizes
- **Categorical × Categorical**: Chi-square, Cramér's V
- **Categorical × Numeric**: t-tests, ANOVA, Cohen's d

### Temporal Analysis
- Trend detection: Mann-Kendall test, linear regression
- Change point detection
- Year-over-year comparisons
- Seasonal decomposition

### Causal Inference
- Regression with control variables
- Confounder identification
- Sensitivity analysis
- Treatment effect estimation

## 🔧 Advanced Features

### LLM-Powered Metadata Enrichment

```python
from statqa.metadata import MetadataEnricher

enricher = MetadataEnricher(provider="openai", api_key="your-key")
enriched_codebook = enricher.enrich_codebook(codebook)

# LLM infers variable types, suggests relationships, identifies confounders
```

### Multiple Testing Correction

```python
from statqa.utils.stats import correct_multiple_testing

p_values = [0.03, 0.01, 0.15, 0.002]
reject, corrected_p = correct_multiple_testing(p_values, method="fdr_bh")
```

### Custom Visualizations

```python
from statqa.visualization import PlotFactory

plotter = PlotFactory(style="publication", figsize=(10, 6))
fig = plotter.plot_bivariate(data, var1, var2, output_path="plot.png")
```

## 📚 Documentation

- **Full Documentation**: [https://gojiplus.github.io/statqa](https://gojiplus.github.io/statqa)
- **API Reference**: [API Docs](https://gojiplus.github.io/statqa/api/)
- **Examples**: See [examples/](examples/) directory

## 🧪 Development

### Running Tests

```bash
pytest --cov=statqa --cov-report=html
```

### Code Quality

```bash
# Linting and formatting
ruff check statqa tests
ruff format statqa tests
```

### Building Documentation

```bash
cd docs
make html
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests and linting
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] Support for additional codebook formats (SPSS, Stata, SAS)
- [ ] Web interface for interactive analysis
