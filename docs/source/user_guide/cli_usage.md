# CLI Usage

Complete command-line workflows for StatQA.

## Overview

StatQA provides a comprehensive CLI for all operations, from parsing codebooks to generating Q/A pairs. Commands can be used individually or chained together in pipelines.

## Installation & Setup

```bash
# Install StatQA with all features
pip install statqa[all]

# Or with UV
uv pip install statqa[all]

# Verify installation
statqa --version
```

## Core Commands

### Parse Codebook

Convert codebooks to structured JSON format:

```bash
# Parse CSV codebook
statqa parse-codebook codebook.csv --output codebook.json

# Parse text codebook with LLM enhancement
statqa parse-codebook codebook.txt --output codebook.json --enrich --llm

# Parse statistical format file
statqa parse-codebook data.sav --output codebook.json --enrich
```

**Options:**
- `--enrich`: Use heuristic enrichment for variable type inference
- `--llm`: Use LLM-powered enhancement (requires API keys)
- `--output`: Output JSON file path

### Analyze Data

Run comprehensive statistical analysis:

```bash
# Basic analysis
statqa analyze data.csv codebook.json --output-dir results/

# With plots and extended analysis
statqa analyze data.csv codebook.json --output-dir results/ --plots --temporal --causal

# Specific analysis types only
statqa analyze data.csv codebook.json --output-dir results/ --univariate-only
```

**Options:**
- `--plots`: Generate visualization plots
- `--temporal`: Include temporal analysis (requires datetime variables)
- `--causal`: Include causal analysis (requires treatment/outcome variables)
- `--univariate-only`: Skip bivariate analysis
- `--max-pairs`: Limit bivariate pairs (default: 100)

### Generate Q/A Pairs

Create training data from analysis results:

```bash
# Generate Q/A pairs with templates
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl

# Generate with LLM paraphrasing
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl --llm

# Export in specific format
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl --format openai
```

**Options:**
- `--llm`: Use LLM for question paraphrasing
- `--format`: Export format (jsonl, openai, anthropic)
- `--max-questions`: Limit number of questions generated
- `--filter-significance`: Only include statistically significant results

## Pipeline Command

Run complete end-to-end workflow:

```bash
# Complete pipeline
statqa pipeline data.csv codebook.csv --output-dir output/

# Pipeline with all features
statqa pipeline data.csv codebook.csv \
  --output-dir output/ \
  --enrich \
  --plots \
  --qa \
  --llm

# Pipeline with custom settings
statqa pipeline data.csv codebook.csv \
  --output-dir output/ \
  --max-pairs 50 \
  --significance-level 0.01 \
  --format openai
```

**Pipeline Steps:**
1. Parse codebook (with optional LLM enrichment)
2. Run statistical analyses (univariate, bivariate, temporal, causal)
3. Generate plots (if `--plots`)
4. Create Q/A pairs (if `--qa`)
5. Export all results

## Configuration

### Environment Variables

```bash
# LLM API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Debug logging
export STATQA_DEBUG=1

# Custom output format
export STATQA_OUTPUT_FORMAT="openai"
```

### Configuration File

Create `~/.statqa/config.yaml`:

```yaml
llm:
  provider: "openai"
  model: "gpt-4"

analysis:
  significance_level: 0.05
  min_sample_size: 30
  max_bivariate_pairs: 100

output:
  format: "jsonl"
  include_plots: true
  include_provenance: true
```

## Data Formats

### Supported Input Formats

**Data Files:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- SPSS (`.sav`, `.zsav`, `.por`)
- Stata (`.dta`)
- SAS (`.sas7bdat`, `.xpt`)

**Codebook Files:**
- CSV (structured)
- Text (unstructured)
- PDF (parsed with LLM)
- Statistical format metadata

### Output Structure

```
output/
├── codebook.json              # Parsed codebook
├── data_overview.json         # Dataset summary
├── univariate/               # Single variable analyses
│   ├── age.json
│   ├── income.json
│   └── plots/
├── bivariate/               # Variable relationships
│   ├── age_vs_income.json
│   └── plots/
├── temporal/                # Time series analyses
│   └── plots/
├── causal/                 # Causal analyses
├── all_insights.json      # Combined results
└── qa_pairs.jsonl         # Generated Q/A pairs
```

## Advanced Usage

### Filtering and Sampling

```bash
# Analyze subset of variables
statqa analyze data.csv codebook.json \
  --variables age,income,education \
  --output-dir results/

# Random sampling for large datasets
statqa analyze data.csv codebook.json \
  --sample 10000 \
  --output-dir results/
```

### Parallel Processing

```bash
# Use multiple cores
statqa pipeline data.csv codebook.csv \
  --output-dir output/ \
  --workers 4
```

### Custom Templates

```bash
# Use custom Q/A templates
statqa generate-qa results.json \
  --output qa_pairs.jsonl \
  --templates custom_templates.yaml
```

## Troubleshooting

### Common Issues

**Memory errors with large datasets:**
```bash
# Use sampling
statqa analyze large_data.csv codebook.json --sample 50000
```

**Missing dependencies:**
```bash
# Install specific features
pip install statqa[llm,pdf,statistical-formats]
```

**LLM API errors:**
```bash
# Check API keys
statqa generate-qa results.json --output qa.jsonl --debug
```

### Debug Mode

```bash
# Enable detailed logging
STATQA_DEBUG=1 statqa pipeline data.csv codebook.csv --output-dir debug_output/
```

This provides detailed logging for troubleshooting analysis issues.
