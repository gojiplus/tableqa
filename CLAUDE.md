# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Installation
```bash
# Development installation (includes all dev tools)
uv pip install -e ".[dev]"

# With specific optional dependencies
uv pip install -e ".[llm]"                # OpenAI/Anthropic LLM support
uv pip install -e ".[pdf]"                # PDF parsing with pdfplumber
uv pip install -e ".[statistical-formats]" # SPSS/Stata/SAS file support
uv pip install -e ".[all]"                # All features

# Or use uv sync for development environment
uv sync --all-extras
```

### Code Quality & Testing
```bash
# Run all linting checks
ruff check statqa tests

# Type checking
mypy statqa --ignore-missing-imports

# Format code
ruff format statqa tests

# Run tests with coverage
pytest --cov=statqa --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_schema.py

# Run tests with specific markers
pytest -m "not slow"        # Exclude slow tests
pytest -m "not llm"         # Exclude LLM API tests
```

### Documentation
```bash
# Build documentation
cd docs && make html
```

### CLI Usage
```bash
# Parse codebook
statqa parse-codebook codebook.csv --output codebook.json --enrich

# Run full analysis
statqa analyze data.csv codebook.json --output-dir results/ --plots

# Generate Q/A pairs
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl --llm

# Complete pipeline
statqa pipeline data.csv codebook.csv --output-dir output/ --enrich --qa
```

## Code Architecture

### Core Components

**Metadata System** (`statqa/metadata/`)
- `schema.py`: Pydantic models defining Variable and Codebook structures
- `parsers/`: Text, CSV, PDF, and statistical format parsers
  - `statistical.py`: SPSS (.sav/.zsav/.por), Stata (.dta), SAS (.sas7bdat/.xpt) support via pyreadstat
- `enricher.py`: LLM-powered metadata enhancement

**Analysis Pipeline** (`statqa/analysis/`)
- `univariate.py`: Single variable statistics (mean, median, distribution tests)
- `bivariate.py`: Relationship analysis (correlation, chi-square, t-tests)
- `temporal.py`: Time series analysis (trends, change points)
- `causal.py`: Causal inference with confounding control

**Q/A Generation** (`statqa/qa/`)
- `generator.py`: Creates Q/A pairs from statistical insights
- `templates.py`: Template-based question generation
- Supports both template-based and LLM-paraphrased questions

**Natural Language Output** (`statqa/interpretation/`)
- `formatter.py`: Converts statistical results to human-readable text
- `context.py`: Contextual interpretation helpers

### Data Flow

1. **Parse Codebook**: Text/CSV/PDF → Variable/Codebook objects (Pydantic models)
2. **LLM Enrichment** (optional): Infer variable types and relationships
3. **Statistical Analysis**: Run univariate/bivariate/temporal/causal analyses
4. **Format Insights**: Convert statistical results to natural language
5. **Generate Q/A Pairs**: Create training data with provenance metadata
6. **Export**: JSONL, OpenAI format, or Anthropic format

### Variable Types & Handling

The system uses a rich type system (`VariableType` enum):
- `NUMERIC_CONTINUOUS`/`NUMERIC_DISCRETE`: Standard statistical analysis
- `CATEGORICAL_NOMINAL`/`CATEGORICAL_ORDINAL`: Frequency analysis, chi-square
- `DATETIME`: Temporal analysis
- `BOOLEAN`, `TEXT`: Specialized handling

Variables include metadata for causal analysis:
- `is_treatment`, `is_outcome`, `is_confounder`: Supports causal inference
- `missing_pattern`: MCAR/MAR/MNAR classification

### Provenance Tracking

All Q/A pairs include detailed provenance metadata:
- `generated_at`: ISO 8601 timestamp
- `tool`/`tool_version`: statqa version
- `generation_method`: template vs. LLM paraphrase
- `analysis_type`: univariate/bivariate/temporal/causal
- `analyzer`: Specific analyzer class used
- `llm_model`: Model used (if applicable)

This enables reproducibility and quality control for LLM training datasets.

### Testing & Configuration

- Uses pytest with coverage reporting
- ruff for linting (replaces flake8/isort)
- mypy for type checking
- black for formatting
- GitHub Actions CI across Python 3.11-3.13 and multiple OS

Key test markers:
- `slow`: Long-running tests
- `llm`: Tests requiring API keys
- `integration`: End-to-end tests

### Examples & Use Cases

The `examples/` directory contains real-world usage patterns:
- `basic_usage.py`: Complete pipeline demonstration
- `anes/`, `titanic/`, `iris/`: Domain-specific examples with actual datasets
- Each example includes codebook.json, data.csv, and generated outputs

## Working with the Codebase

### Adding New Analyzers

1. Create analyzer class in appropriate `analysis/` module
2. Implement `analyze()` method returning standardized dict structure
3. Add formatter support in `interpretation/formatter.py`
4. Add Q/A templates in `qa/templates.py`
5. Update CLI commands if needed

### Extending Metadata Parsing

1. Create new parser in `metadata/parsers/`
2. Inherit from `BaseParser` and implement `parse()` method
3. Add validation logic and format detection
4. Register in CLI auto-detection logic

### Adding LLM Providers

1. Update `metadata/enricher.py` with new provider
2. Add API client integration
3. Update CLI options and error handling
4. Add optional dependency in pyproject.toml

The codebase emphasizes type safety (Pydantic models), modular design, and comprehensive provenance tracking for reproducible research workflows.
