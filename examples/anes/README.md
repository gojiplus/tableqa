# ANES Example: Complete Analysis Pipeline

This example demonstrates a complete analysis pipeline using the American National Election Studies (ANES) Time Series Cumulative Data File (1948-2020).

## Overview

The ANES dataset is a comprehensive survey dataset covering American political behavior and attitudes over seven decades. This example shows how to use **statqa** to:

1. Parse codebook metadata from PDF format
2. Extract statistical insights (univariate and bivariate)
3. Generate natural language descriptions
4. Create publication-quality visualizations
5. Produce research questions using LLMs

## Directory Structure

```
anes/
├── README.md              # This file
├── parse_metadata.py      # Parse codebook and generate questions
├── extract_insights.py    # Extract statistical insights
├── data/                  # Data files
│   ├── README.md          # Data download instructions
│   ├── anes_metadata.csv  # Parsed metadata
│   └── raw/               # Raw ANES files
└── output/                # Generated outputs
    ├── README.md          # Output file descriptions
    ├── insights.json      # All extracted insights
    ├── variable_profile.csv
    └── *.png              # Visualization plots
```

## Quick Start

### Prerequisites

```bash
# Install statqa with optional dependencies
pip install statqa[pdf,llm]

# Or install from source
git clone https://github.com/gojiplus/statqa.git
cd statqa
pip install -e ".[pdf,llm]"
```

### Step 1: Parse Metadata

Parse the ANES codebook PDF and optionally generate research questions:

```bash
cd examples/anes

# Parse codebook only
python parse_metadata.py \
    --codebook data/raw/anes_timeseries_cdf_codebook_var_20220916.pdf \
    --output-metadata data/anes_metadata.csv \
    --skip-questions

# Parse codebook AND generate questions with LLM
export OPENAI_API_KEY=your_key_here
python parse_metadata.py \
    --codebook data/raw/anes_timeseries_cdf_codebook_var_20220916.pdf \
    --output-metadata data/anes_metadata.csv \
    --output-templates templates/question_templates.txt
```

### Step 2: Extract Insights

Run comprehensive statistical analysis:

```bash
# Univariate analysis only (faster)
python extract_insights.py \
    --data-zip data/raw/anes_timeseries_cdf_csv_20220916.csv.zip \
    --metadata data/anes_metadata.csv \
    --output-dir output \
    --skip-bivariate

# Full analysis (univariate + bivariate)
python extract_insights.py \
    --data-zip data/raw/anes_timeseries_cdf_csv_20220916.csv.zip \
    --metadata data/anes_metadata.csv \
    --output-dir output \
    --max-vars 50
```

## Script Details

### parse_metadata.py

Parses the ANES codebook PDF and extracts variable metadata:

**Features:**
- PDF parsing with pdfplumber
- Extraction of variable names, labels, valid values, and missing codes
- LLM-powered research question generation
- Chunked API calls to handle large codebooks

**Arguments:**
- `--codebook`: Path to ANES codebook PDF (required)
- `--output-metadata`: Output CSV for metadata (default: `../data/anes_metadata.csv`)
- `--output-templates`: Output file for questions (default: `../templates/question_templates.txt`)
- `--api-key`: OpenAI API key (or use `OPENAI_API_KEY` env var)
- `--chunk-size`: Variables per LLM prompt (default: 20)
- `--max-questions`: Questions per chunk (default: 20)
- `--skip-questions`: Skip LLM question generation

**Output:**
- `anes_metadata.csv`: Structured metadata for all variables
- `question_templates.txt`: Research questions (if generated)

### extract_insights.py

Runs statistical analysis and generates insights:

**Features:**
- Variable profiling (type inference, missingness)
- Univariate analysis (descriptive statistics, distributions)
- Bivariate analysis (correlations, group comparisons)
- Automatic visualization generation
- Natural language insight formatting

**Arguments:**
- `--data-zip`: Path to ANES data ZIP (required)
- `--metadata`: Path to metadata CSV (required)
- `--output-dir`: Output directory (required)
- `--max-vars`: Max variables for bivariate analysis (default: 50)
- `--skip-bivariate`: Skip bivariate analysis

**Output:**
- `insights.json`: All extracted insights with text and figure paths
- `variable_profile.csv`: Variable metadata and statistics
- `univariate_*.png`: Univariate distribution plots
- `bivariate_*.png`: Bivariate relationship plots

## Output Examples

### Variable Profile (variable_profile.csv)

```csv
varname,label,dtype,unique,missing_pct
VCF0004,Year of study,int64,38,0.0
VCF0006,Respondent sequence number,int64,72341,0.0
VCF0102,Age of respondent,int64,87,3.2
VCF0104,Gender,int64,2,0.1
```

### Insights (insights.json)

```json
[
  {
    "vars": ["VCF0102"],
    "insight": "**Age of respondent** (VCF0102): mean=47.32, median=46.00, std=17.45. N=68234, dropped 2107 missing. (No weights applied.)",
    "figure": "output/univariate_VCF0102.png"
  },
  {
    "vars": ["VCF0104", "VCF0110"],
    "insight": "Correlation **Gender** ↔ **Political Interest**: r=0.12 (N=65432), p=0.001. (No weights applied.)",
    "figure": null
  }
]
```

## Data Sources

The ANES Time Series Cumulative Data File is available from:
- **Official source**: https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/
- **Dataset**: 1948-2020 Cumulative Data File
- **Codebooks**: Variable and appendix codebooks (PDF format)

See `data/README.md` for download instructions.

## Customization

### Adding Custom Analyses

You can extend the scripts to include additional analyses:

```python
from statqa.analysis.temporal import TemporalAnalyzer
from statqa.analysis.causal import CausalAnalyzer

# Add temporal trend analysis
temporal_analyzer = TemporalAnalyzer()
trends = temporal_analyzer.analyze(data, time_var, target_var)

# Add causal analysis
causal_analyzer = CausalAnalyzer()
effects = causal_analyzer.analyze(data, treatment, outcome, confounders)
```

### Custom Visualizations

```python
from statqa.visualization.plots import PlotFactory

plotter = PlotFactory(style="publication", figsize=(10, 6))
plotter.plot_temporal(data, time_var, target_var, output_path="trend.png")
```

## Tips and Best Practices

1. **Start small**: Use `--max-vars 10` for initial testing
2. **Skip bivariate**: Use `--skip-bivariate` for faster univariate-only analysis
3. **API costs**: Question generation uses OpenAI API; adjust `--chunk-size` and `--max-questions` to control costs
4. **Missing data**: ANES uses various missing codes; the scripts handle these automatically
5. **Memory**: Full bivariate analysis can be memory-intensive; reduce `--max-vars` if needed

## Troubleshooting

**Problem**: `pdfplumber` not found
**Solution**: Install with `pip install statqa[pdf]`

**Problem**: OpenAI API errors
**Solution**: Check API key, quota, and use `--skip-questions` to bypass

**Problem**: Memory errors during bivariate analysis
**Solution**: Reduce `--max-vars` or use `--skip-bivariate`

**Problem**: Missing data file
**Solution**: Download from ANES website (see `data/README.md`)

## Citation

If you use this example or the ANES data in your research, please cite:

```bibtex
@misc{anes2022,
  title={ANES Time Series Cumulative Data File [dataset and documentation]},
  author={American National Election Studies},
  year={2022},
  url={https://electionstudies.org}
}
```

## License

- **statqa**: MIT License
- **ANES Data**: See [ANES usage terms](https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/)

## Support

For issues or questions:
- statqa package: [GitHub Issues](https://github.com/gojiplus/statqa/issues)
- ANES data: [ANES Contact](https://electionstudies.org/contact/)
