Changelog
=========

All notable changes to StatQA will be documented here.

v0.3.0 (2025-01-XX)
-------------------

**Added**

- Modern Python features (pathlib, match/case statements)
- Comprehensive API documentation with Sphinx autodoc
- Python 3.12+ type hint modernization
- Dead link detection and fixes
- Enhanced pre-commit configuration

**Changed** 

- Documentation converted from mixed MD/RST to pure RST
- Removed MyST parser dependency for simpler builds
- Updated minimum Python version to 3.12
- Improved type safety with zero pyright violations

**Fixed**

- Pre-commit environment issues with Python version conflicts
- Documentation build warnings and dead links
- Pydoclint configuration for function signature type hints

v0.2.0 (2024-12-XX)
-------------------

**Added**

- Multimodal Q/A generation with visual metadata
- Provenance tracking for all generated content  
- Support for statistical file formats (SPSS, Stata, SAS)
- Causal inference analysis module
- Temporal trend detection

**Changed**

- Enhanced Q/A format with visual elements
- Improved statistical rigor with effect sizes
- Better error handling and logging

v0.1.0 (2024-11-XX)
-------------------

**Added**

- Initial release
- Basic statistical analysis (univariate, bivariate)
- Codebook parsing (text, CSV, PDF)
- Template-based Q/A generation
- CLI interface
- Publication-quality visualizations