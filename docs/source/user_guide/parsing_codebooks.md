# Parsing Codebooks

Learn how to parse and work with variable metadata in StatQA.

## Overview

StatQA can parse codebooks from various formats to understand your dataset's structure and variable types. This metadata enables intelligent statistical analysis and interpretation.

## Supported Formats

### CSV Codebooks
Parse structured codebooks from CSV files:

```python
from statqa.metadata.parsers.csv import CSVParser

parser = CSVParser()
codebook = parser.parse("codebook.csv")
```

### Text Codebooks
Parse unstructured codebooks from text files:

```python
from statqa.metadata.parsers.text import TextParser

parser = TextParser()
codebook = parser.parse("codebook.txt")
```

### Statistical Format Files
Parse codebooks directly from SPSS, Stata, or SAS files:

```python
from statqa.metadata.parsers.statistical import StatisticalFormatParser

parser = StatisticalFormatParser()
codebook = parser.parse("data.sav")  # SPSS
# Also supports: .dta (Stata), .sas7bdat (SAS)
```

## CLI Usage

```bash
# Parse and enrich a codebook
statqa parse-codebook codebook.csv --output codebook.json --enrich

# Parse from text with LLM enhancement
statqa parse-codebook codebook.txt --output codebook.json --enrich --llm
```

## Codebook Structure

The parsed codebook follows this schema:

```python
from statqa.metadata.schema import Variable, Codebook

variable = Variable(
    name="age",
    var_type="numeric_continuous",
    description="Respondent age in years",
    value_labels={},
    missing_values=[999, 998],
    is_treatment=False,
    is_outcome=False,
    is_confounder=True
)
```

## LLM Enhancement

Use LLM-powered enhancement to automatically infer variable types and relationships:

```python
from statqa.metadata.enricher import MetadataEnricher

enricher = MetadataEnricher(llm_provider="openai")
enhanced_codebook = enricher.enrich(codebook, data)
```

This adds intelligent type inference, missing value detection, and causal relationship identification.
