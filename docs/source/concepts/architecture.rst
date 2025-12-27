Architecture
============

StatQA's modular design enables flexible workflows and easy extension.

Core Components
---------------

Metadata System (``statqa/metadata/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Schema**: Pydantic models for Variables and Codebooks
- **Parsers**: Support for text, CSV, PDF, and statistical formats  
- **Enrichment**: LLM-powered metadata enhancement

Analysis Pipeline (``statqa/analysis/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Univariate**: Descriptive statistics and distribution testing
- **Bivariate**: Correlations, group comparisons, effect sizes
- **Temporal**: Trend detection and change point analysis
- **Causal**: Regression with confounding control

Q/A Generation (``statqa/qa/``) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Templates**: Structured question generation patterns
- **LLM Integration**: Natural language paraphrasing
- **Visual Metadata**: Plot association and descriptions

Data Flow
---------

1. **Parse Metadata** → Variable/Codebook objects
2. **LLM Enrichment** → Enhanced type inference (optional)
3. **Statistical Analysis** → Numerical results
4. **Format Insights** → Human-readable text
5. **Generate Q/A** → Training-ready pairs
6. **Export** → JSONL/OpenAI/Anthropic formats

Extension Points
----------------

- **Custom Analyzers**: Implement analyze() method
- **New Parsers**: Inherit from BaseParser
- **Output Formats**: Add formatters and exporters
- **LLM Providers**: Extend enricher with new APIs