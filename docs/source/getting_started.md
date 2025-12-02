# Getting Started

```{include} _shared/installation.md
```

```{include} _shared/quickstart.md
```

## Command-Line Interface

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

## Next Steps

- **{doc}`user_guide/index`** - Comprehensive guides for all features
- **{doc}`examples/index`** - Real-world examples with datasets
- **{doc}`api/index`** - Complete API reference
- **{doc}`concepts/index`** - Core concepts and architecture
