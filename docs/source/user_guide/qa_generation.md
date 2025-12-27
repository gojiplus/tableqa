# Q/A Generation

Generate question-answer pairs from statistical insights for LLM training.

## Overview

StatQA can automatically generate high-quality Q/A pairs from your statistical analyses, complete with provenance tracking for reproducible LLM training datasets.

## Basic Q/A Generation

Generate Q/A pairs from analysis results:

```python
from statqa.qa.generator import QAGenerator

generator = QAGenerator(
    llm_provider="openai",  # or "anthropic", "none" for templates only
    paraphrase_questions=True,
    include_provenance=True
)

qa_pairs = generator.generate(analysis_results, codebook)
```

## CLI Usage

```bash
# Generate Q/A pairs from analysis results
statqa generate-qa results/all_insights.json --output qa_pairs.jsonl --llm

# Complete pipeline: analyze + generate Q/A
statqa pipeline data.csv codebook.csv --output-dir output/ --qa
```

## Q/A Formats

### Template-Based Questions

Fast generation using predefined templates:

```python
generator = QAGenerator(llm_provider="none")
qa_pairs = generator.generate(results, codebook)

# Example output:
{
    "question": "What is the mean age in the dataset?",
    "answer": "The mean age is 42.3 years (SD = 12.1).",
    "analysis_type": "univariate",
    "variable": "age"
}
```

### LLM-Paraphrased Questions

Natural language variations using LLMs:

```python
generator = QAGenerator(
    llm_provider="openai",
    paraphrase_questions=True
)

# Example output:
{
    "question": "How old are people in this study on average?",
    "answer": "The average age of participants is 42.3 years...",
    "generation_method": "llm_paraphrase"
}
```

## Export Formats

### JSONL Format (Default)

```bash
statqa generate-qa results.json --output qa_pairs.jsonl
```

### OpenAI Fine-tuning Format

```bash
statqa generate-qa results.json --output qa_pairs.jsonl --format openai
```

### Anthropic Format

```bash
statqa generate-qa results.json --output qa_pairs.jsonl --format anthropic
```

## Question Types

StatQA generates diverse question types:

### Descriptive Questions
- "What is the mean/median of variable X?"
- "How many missing values does variable Y have?"

### Comparative Questions
- "Is there a significant difference between groups A and B?"
- "Which variable has the strongest correlation with outcome Z?"

### Temporal Questions
- "Is there an increasing trend in variable X over time?"
- "When did the most significant change occur?"

### Causal Questions
- "What is the effect of treatment T on outcome O?"
- "Which variables act as confounders?"

## Provenance Tracking

Every Q/A pair includes detailed metadata:

```python
qa_pair = {
    "question": "...",
    "answer": "...",
    "provenance": {
        "generated_at": "2025-12-27T04:30:00Z",
        "tool": "statqa",
        "tool_version": "0.3.0",
        "analysis_type": "bivariate",
        "analyzer": "BivariateAnalyzer",
        "generation_method": "template",
        "llm_model": "gpt-4",
        "variables": ["age", "income"],
        "statistical_test": "pearson_correlation"
    }
}
```

## Quality Control

### Filtering Options

```python
generator = QAGenerator(
    min_effect_size=0.1,      # Filter weak effects
    min_sample_size=30,       # Ensure adequate power
    significance_level=0.05,  # Only significant results
    exclude_missing_heavy=True # Skip variables with >50% missing
)
```

### Validation

```python
# Validate generated Q/A pairs
from statqa.qa.validator import QAValidator

validator = QAValidator()
validated_pairs = validator.validate(qa_pairs)
```

This ensures generated Q/A pairs are factually accurate and well-formed for training.
