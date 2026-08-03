# Multimodal Q/A Dataset Generation Pipeline

## Overview

This document describes StatQA's pipeline for generating **multimodal Q/A-style datasets** from tabular data with metadata. The pipeline converts raw tables into structured question-answer pairs enriched with rich visual metadata and associated plots, suitable for training or fine-tuning multimodal LLMs on data analysis tasks. This creates CLIP-style visual-text databases for advanced AI training.

## Pipeline Architecture

### Complete Flow

```
Input: Table + Metadata
         ↓
[1] Parse Codebook (Variable Metadata)
         ↓
[2] Run Statistical Analyses
    - Univariate (descriptive stats)
    - Bivariate (relationships)
    - Temporal (trends)
    - Causal (effects)
         ↓
[3] Format Insights (Natural Language)
         ↓
[4] Generate Visualizations
    - Create publication-quality plots
    - Generate visual metadata (captions, alt-text, elements)
    - Associate plots with statistical insights
         ↓
[5] Generate Multimodal Q/A Pairs
    - Template-based questions
    - LLM paraphrasing (optional)
    - Rich visual metadata integration
    - Question-plot association mapping
         ↓
[6] Export Enhanced Dataset
    - JSONL format with visual metadata
    - OpenAI fine-tuning format
    - Anthropic format
    - CLIP-style visual-text pairs
         ↓
Output: Multimodal Q/A Dataset for Advanced AI Training
```

## Pipeline Components

### 1. Metadata/Codebook Parsing

**Purpose**: Extract variable-level metadata including:
- Variable name and label
- Type (numeric continuous/discrete, categorical nominal/ordinal, etc.)
- Value labels (for categorical variables)
- Missing value codes
- Units, range, description
- Data generating process information

**Supported Formats**:
- Text format (custom syntax)
- CSV format
- PDF codebooks (with optical parsing)

**Example**:
```python
from statqa.metadata.parsers import CSVParser

parser = CSVParser()
codebook = parser.parse("codebook.csv")
```

### 2. Statistical Analyses

**Univariate Analysis**:
- Numeric: mean, median, std, min/max, quartiles, skewness, kurtosis
- Categorical: frequencies, mode, diversity indices, entropy
- Distribution: normality tests, outlier detection
- Robust statistics: MAD, robust z-scores

**Bivariate Analysis**:
- Numeric × Numeric: Pearson/Spearman correlation, effect sizes
- Categorical × Categorical: Chi-square test, Cramér's V
- Categorical × Numeric: t-tests, ANOVA, Cohen's d, η²

**Temporal Analysis** (optional):
- Trend detection: Mann-Kendall test
- Change point detection
- Year-over-year comparisons
- Seasonal decomposition

**Causal Analysis** (optional):
- Regression with confounding control
- Treatment effect estimation
- Sensitivity analysis

**Example**:
```python
from statqa.analysis import UnivariateAnalyzer, BivariateAnalyzer

# Univariate
analyzer = UnivariateAnalyzer()
result = analyzer.analyze(data["age"], codebook.variables["age"])

# Bivariate
biv_analyzer = BivariateAnalyzer()
result = biv_analyzer.analyze(data, var1, var2)
```

### 3. Insight Formatting

**Purpose**: Convert statistical results into natural language insights that are:
- Human-readable and publication-quality
- Concise but complete
- Interpretable by domain experts and LLMs

**Formatting Rules**:
- Include variable labels (not just codes)
- Report effect sizes and statistical significance
- Add context (sample size, normality, outliers)
- Use consistent notation

**Example Output**:
```
**Annual Income**: mean=65,897.15, median=64,511.00, std=23,807.36,
range=[25,000.00, 133,843.00]. N=500 [non-normal distribution].
```

**Example**:
```python
from statqa.interpretation import InsightFormatter

formatter = InsightFormatter()
insight = formatter.format_univariate(result)
```

### 4. Visualization Generation & Visual Metadata

**Purpose**: Create publication-quality visualizations and extract rich metadata for each statistical insight, enabling multimodal AI training and accessibility.

**Plot Types**:
- **Univariate**: Histograms for numeric data, bar charts for categorical data
- **Bivariate**: Scatter plots (numeric×numeric), box plots (categorical×numeric), heatmaps (categorical×categorical)
- **Advanced**: Temporal plots, regression visualizations

**Visual Metadata Generated**:
- **Descriptive captions** with statistical context and key findings
- **Accessibility alt-text** for screen readers and inclusive applications
- **Visual elements** (chart type, axes, colors, annotations, key features)
- **Plot generation code** for programmatic reproduction

**Question-Plot Association**:
- Distribution questions → Histograms/bar charts
- Correlation questions → Scatter plots with regression lines
- Group comparison questions → Box plots showing differences
- Categorical relationship questions → Heatmaps with frequency counts

**Example**:
```python
from statqa.qa import QAGenerator

qa_gen = QAGenerator()

# Generate visual metadata for a statistical insight
plot_data = {
    "data": data,
    "variables": codebook.variables,
    "output_path": "plots/univariate_age.png",
}
visual_metadata = qa_gen.generate_visual_metadata(
    result, variables=["age"], plot_data=plot_data
)

# Example output
{
    "plot_type": "histogram",
    "caption": "Histogram showing age distribution with mean=42.5 and std=12.3 (N=1000). The data shows a approximately normal distribution.",
    "alt_text": "Histogram chart with age values on x-axis and frequency density on y-axis, showing distribution shape with 1000 observations.",
    "visual_elements": {
        "chart_type": "histogram",
        "x_axis": "Age",
        "y_axis": "Density",
        "colors": ["blue bars", "red mean line"],
        "key_features": ["distribution shape", "mean line"],
        "annotations": ["Mean: 42.5"],
    },
    "primary_plot": "plots/univariate_age.png",
    "generation_code": "plot_factory.plot_univariate(data['age'], age_var, 'plots/univariate_age.png')",
}
```

### 5. Multimodal Q/A Pair Generation

**Template-Based Generation**:
- Uses pre-defined templates based on analysis type
- Generates 2-3 questions per insight with associated visualizations
- Question types:
  - Descriptive: "What is the average...?" → Histogram
  - Comparative: "How does X differ across Y groups?" → Box plot
  - Correlational: "Are X and Y correlated?" → Scatter plot
  - Temporal: "Has X changed over time?" → Time series plot
  - Distributional: "What is the distribution of X?" → Histogram/bar chart
  - Causal: "What is the effect of X on Y?" → Regression plot

**LLM-Powered Paraphrasing** (optional):
- Generates diverse phrasings of the same question
- Adds domain-specific terminology
- Varies formality and structure
- Creates 2+ paraphrases per original question
- Preserves visual metadata associations

**Enhanced Q/A Structure**:
Each Q/A pair now includes rich multimodal metadata:
- Original provenance tracking (timestamps, tools, methods)
- Statistical computation log (Python commands executed)
- Complete visual metadata (plot type, captions, alt-text, visual elements)
- Question-plot association mapping
- Accessibility features for inclusive AI

**Example**:
```python
from statqa.qa import QAGenerator

# Template-based multimodal generation
qa_gen = QAGenerator(use_llm=False)

# Create plot data specification
plot_data = {
    "data": data,
    "variables": codebook.variables,
    "output_path": "plots/univariate_age.png",
}

# Generate visual metadata
visual_metadata = qa_gen.generate_visual_metadata(
    result, variables=["age"], plot_data=plot_data
)

# Generate Q/A pairs with visual data
qa_pairs = qa_gen.generate_qa_pairs(
    result, formatted_answer, variables=["age"], visual_data=visual_metadata
)

# With LLM paraphrasing (visual metadata preserved)
qa_gen = QAGenerator(use_llm=True, api_key="your-key")
qa_pairs = qa_gen.generate_qa_pairs(
    result, formatted_answer, variables=["age"], visual_data=visual_metadata
)
```

**Example Multimodal Q/A Pairs**:

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
    "analysis_type": "univariate",
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
    "primary_plot": "/plots/univariate_sepal_length.png",
    "generation_code": "plot_factory.plot_univariate(data['sepal_length'], sepal_length_var, 'plot.png')"
  },
  "vars": ["sepal_length"]
}
```

```json
{
  "question": "Are sepal_length and petal_length correlated?",
  "answer": "**sepal_length** and **petal_length** show a very strong positive correlation (r=0.87, p=0.000, N=150) [statistically significant], effect size: large.",
  "type": "correlational",
  "provenance": {
    "generated_at": "2025-11-19T19:21:28+00:00",
    "tool": "statqa",
    "tool_version": "0.2.0",
    "generation_method": "template",
    "analysis_type": "numeric_numeric",
    "variables": ["sepal_length", "petal_length"]
  },
  "visual": {
    "plot_type": "scatter",
    "caption": "Scatter plot showing the relationship between Sepal Length and Petal Length (N=150). Shows a strong positive correlation (r=0.87) with regression line.",
    "alt_text": "Scatter plot with Sepal Length on x-axis and Petal Length on y-axis, showing 150 data points with regression line.",
    "visual_elements": {
      "chart_type": "scatter",
      "x_axis": "Sepal Length",
      "y_axis": "Petal Length",
      "key_features": ["data points", "regression line", "trend"],
      "colors": ["blue points", "red regression line"],
      "annotations": []
    },
    "primary_plot": "/plots/bivariate_sepal_length_petal_length.png",
    "generation_code": "plot_factory.plot_bivariate(data, sepal_length_var, petal_length_var, 'plot.png')"
  },
  "vars": ["sepal_length", "petal_length"]
}
```

### 6. Export Formats

**JSONL (Enhanced Multimodal)**:
```json
{"question": "...", "answer": "...", "type": "...", "provenance": {...}, "visual": {...}, "vars": [...]}
```

**OpenAI Fine-Tuning Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a data analyst..."},
    {"role": "user", "content": "What is the average income?"},
    {"role": "assistant", "content": "**Annual Income**: mean=65,897..."}
  ]
}
```

**Anthropic Format**:
```json
{"prompt": "What is the average income?", "completion": "**Annual Income**: mean=65,897..."}
```

## Multimodal Demo Results

We tested the enhanced pipeline on 3 public datasets with comprehensive multimodal output:

### Dataset 1: Employee Survey (500 rows, 5 variables)

**Variables**: age, education, income, job_satisfaction, work_hours

**Results**:
- 5 univariate insights + 10 bivariate insights
- 15 publication-quality visualizations generated
- **35 multimodal Q/A pairs** with visual metadata
- Plot types: histograms, box plots, heatmaps

**Enhanced Q/A Example**:
```json
{
  "question": "What is the distribution of Annual Income?",
  "answer": "**Annual Income**: mean=65,897.15, median=64,511.00, std=23,807.36, range=[25,000.00, 133,843.00]. N=500 [non-normal distribution].",
  "visual": {
    "plot_type": "histogram",
    "caption": "Histogram showing income distribution with right-skewed pattern...",
    "primary_plot": "/plots/univariate_income.png"
  }
}
```

### Dataset 2: Iris Flowers (150 rows, 5 variables)

**Variables**: sepal_length, sepal_width, petal_length, petal_width, species

**Results**:
- 5 univariate insights + 10 bivariate insights
- 15 publication-quality visualizations generated
- **39 multimodal Q/A pairs** with visual metadata
- Plot types: histograms, scatter plots, box plots, bar charts

**Enhanced Q/A Example**:
```json
{
  "question": "Are petal_length and petal_width correlated?",
  "answer": "**petal_length** and **petal_width** show a very strong positive correlation (r=0.96, p<0.001, N=150), effect size: very large.",
  "visual": {
    "plot_type": "scatter",
    "caption": "Scatter plot showing strong positive relationship between petal measurements with regression line (r=0.96).",
    "primary_plot": "/plots/bivariate_petal_length_petal_width.png"
  }
}
```

### Dataset 3: Titanic Passengers (400 rows, 5 variables)

**Variables**: survived, pclass, sex, age, fare

**Results**:
- 5 univariate insights + 10 bivariate insights
- 15 publication-quality visualizations generated
- **28 multimodal Q/A pairs** with visual metadata
- Plot types: bar charts, box plots, heatmaps, scatter plots

**Enhanced Q/A Example**:
```json
{
  "question": "What is the frequency distribution of Survived?",
  "answer": "**Survived**: most common category is '0' (60.2%), N=400. Distribution: 0: 60.2%, 1: 39.8% [high diversity].",
  "visual": {
    "plot_type": "bar_chart",
    "caption": "Bar chart showing survival frequencies across 2 categories (N=400). Most common category is 'No survival' (60.2%).",
    "primary_plot": "/plots/univariate_survived.png"
  }
}
```

### Enhanced Overall Results

- **Total Datasets**: 3
- **Total Insights**: 45 (15 univariate + 30 bivariate)
- **Total Visualizations**: 45 publication-quality plots
- **Total Multimodal Q/A Pairs**: 102 (39 + 35 + 28)
- **Average Q/A per Insight**: 2.3
- **Visual Coverage**: 100% (every Q/A pair includes rich visual metadata)
- **Accessibility**: Full alt-text and captions for all visualizations
- **CLIP-style Pairs**: Complete visual-text associations for multimodal AI training

## Pipeline Validation

### ✅ Pipeline Makes Sense

**Strengths**:

1. **Comprehensive Metadata Integration**
   - Uses rich variable metadata (types, labels, descriptions)
   - Handles missing values and outliers appropriately
   - Supports multiple variable types (numeric, categorical, ordinal)

2. **Rigorous Statistical Analysis**
   - Appropriate tests for each variable type combination
   - Reports effect sizes (not just p-values)
   - Includes normality tests and distribution information
   - Handles edge cases (e.g., insufficient data, all missing)

3. **High-Quality Natural Language**
   - Insights are clear, concise, and informative
   - Uses proper statistical notation
   - Includes context (N, significance, effect size)
   - Publication-ready formatting

4. **Diverse Q/A Generation**
   - Multiple question types (descriptive, comparative, correlational, etc.)
   - Template-based ensures grammatical correctness
   - Optional LLM paraphrasing adds diversity
   - Maintains question-answer consistency

5. **Flexible Export Formats**
   - Supports major LLM training platforms
   - JSONL for custom training pipelines
   - Includes metadata for filtering/analysis

6. **Scalable Architecture**
   - Batch processing for multiple variables
   - Handles large codebooks efficiently
   - Modular components (easy to extend)

7. **Enhanced Multimodal Capabilities** ✅ **IMPLEMENTED**
   - Rich visual metadata with captions and alt-text
   - Question-plot association mapping
   - CLIP-style visual-text pairing for AI training
   - Accessibility features for inclusive applications
   - Comprehensive visual elements extraction

### 🔍 Future Enhancements

1. **Advanced Visual Analysis**
   - Chart pattern recognition and description
   - Automated insight extraction from visual elements
   - Multi-chart comparative visualizations

2. **LLM-Generated Follow-ups**
   - Add exploratory questions ("Why might this correlation exist?")
   - Generate hypothesis-generating questions
   - Create multi-hop reasoning chains

3. **Domain-Specific Templates**
   - Healthcare-specific question patterns
   - Social science terminology
   - Business/finance language

4. **Quality Filtering**
   - Skip non-significant findings (optional)
   - Prioritize large effect sizes
   - Filter out redundant Q/A pairs

5. **Interactive Visualizations**
   - Generate interactive plots with metadata
   - Support for dynamic filtering and exploration
   - Web-based visualization components

## Usage Examples

### Enhanced Multimodal Usage

```python
from pathlib import Path
import json
from statqa.metadata.parsers import CSVParser
from statqa.analysis import UnivariateAnalyzer, BivariateAnalyzer
from statqa.interpretation import InsightFormatter
from statqa.qa import QAGenerator

# 1. Parse codebook
parser = CSVParser()
codebook = parser.parse("codebook.csv")

# 2. Load data
import pandas as pd

data = pd.read_csv("data.csv")

# 3. Setup for multimodal generation
analyzer = UnivariateAnalyzer()
formatter = InsightFormatter()
qa_gen = QAGenerator(use_llm=False)

# Create plots directory
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

all_qa_pairs = []

for var_name, variable in codebook.variables.items():
    # Analyze
    result = analyzer.analyze(data[var_name], variable)

    # Format insight
    insight = formatter.format_univariate(result)

    # Generate visual metadata
    plot_data = {
        "data": data,
        "variables": codebook.variables,
        "output_path": plots_dir / f"univariate_{var_name}.png",
    }
    visual_metadata = qa_gen.generate_visual_metadata(
        result, variables=[var_name], plot_data=plot_data
    )

    # Generate multimodal Q/A pairs
    qa_pairs = qa_gen.generate_qa_pairs(
        result, insight, variables=[var_name], visual_data=visual_metadata
    )
    for qa in qa_pairs:
        qa["vars"] = [var_name]
    all_qa_pairs.extend(qa_pairs)

# 4. Export multimodal dataset
with open("multimodal_qa_dataset.jsonl", "w") as f:
    for qa in all_qa_pairs:
        f.write(json.dumps(qa) + "\n")

print(f"Generated {len(all_qa_pairs)} multimodal Q/A pairs with visualizations")
```

### Using the CLI

```bash
# Complete multimodal pipeline
statqa pipeline data.csv codebook.csv \
    --output-dir output/ \
    --qa \
    --plots \
    --multimodal \
    --enrich

# Generate multimodal Q/A pairs from existing insights
statqa generate-qa insights.json \
    --output qa_pairs.jsonl \
    --format openai \
    --visual-metadata \
    --llm
```

### Running the Demo

```bash
# Run the complete demo with 3 datasets
python examples/qa_dataset_generation_demo.py

# View results
ls -lh output/qa_generation_demo/
cat output/qa_generation_demo/combined_qa_dataset.jsonl
```

## Enhanced File Structure

After running the multimodal pipeline, you'll have:

```
output/multimodal_qa_demo/
├── employee_survey/
│   ├── data.csv                    # Original data
│   ├── codebook.json               # Variable metadata
│   ├── insights.json               # All insights
│   ├── qa_pairs.json               # Multimodal Q/A pairs (JSON)
│   ├── qa_pairs.jsonl              # Multimodal Q/A pairs (JSONL)
│   └── plots/                      # Generated visualizations
│       ├── univariate_age.png
│       ├── univariate_income.png
│       ├── bivariate_age_income.png
│       └── ...
├── iris_flowers/
│   ├── data.csv
│   ├── codebook.json
│   ├── insights.json
│   ├── qa_pairs.json
│   ├── qa_pairs.jsonl
│   └── plots/                      # 15 visualization files
│       ├── univariate_sepal_length.png
│       ├── bivariate_petal_length_petal_width.png
│       └── ...
├── titanic_passengers/
│   └── ...                         # Similar structure
├── combined_multimodal_dataset.jsonl    # All Q/A pairs with visual metadata
├── openai_training_data.jsonl           # OpenAI fine-tuning format
└── visualizations_summary.json          # Plot metadata index
```

## Best Practices

1. **Always include metadata**: The pipeline works best with rich variable descriptions
2. **Use appropriate variable types**: Correctly classifying variables improves analysis and visualization quality
3. **Create plots directory**: Ensure output directory structure accommodates visualizations
4. **Start template-based**: Test with templates before adding LLM paraphrasing
5. **Review visual quality**: Validate generated plots and visual metadata for accuracy
6. **Filter insights**: Consider filtering non-significant or trivial findings
7. **Accessibility first**: Leverage alt-text and captions for inclusive AI applications
8. **Domain customization**: Adapt templates and visual styles for your specific domain
9. **CLIP compatibility**: Use visual-text pairs for multimodal AI training pipelines
10. **Quality assurance**: Always review generated Q/A pairs and associated visualizations

## Conclusion

The enhanced multimodal Q/A dataset generation pipeline successfully:

✅ Processes tabular data with rich metadata parsing
✅ Runs comprehensive statistical analyses (univariate, bivariate, temporal, causal)
✅ Generates publication-quality insights and visualizations
✅ Creates diverse, high-quality multimodal Q/A pairs with visual metadata
✅ Provides complete accessibility support (alt-text, captions)
✅ Enables CLIP-style visual-text pairing for advanced AI training
✅ Exports in multiple formats optimized for multimodal LLM training
✅ Maintains full provenance and reproducibility tracking

**Key Achievements**:
- **102 multimodal Q/A pairs** generated across 3 datasets
- **45 publication-quality visualizations** with rich metadata
- **100% visual coverage** - every statistical insight paired with appropriate plots
- **Full accessibility compliance** with comprehensive alt-text and captions
- **Question-plot association mapping** for intelligent visual pairing
- **Enhanced provenance tracking** including computational commands and plot generation

The pipeline is production-ready for creating advanced multimodal datasets and can be customized for specific domains, accessibility requirements, or multimodal AI training use cases.
