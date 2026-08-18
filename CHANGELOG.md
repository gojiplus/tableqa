# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Split test dependencies into the canonical `test` group so shared CI can
  validate the built wheel without reinstalling the source checkout.

## [0.4.0] - 2026-08-03

### Fixed
- `statqa analyze data.csv codebook.json` crashed on the bundled example
  codebooks. Those files are a bare `{variable_name: {...}}` map while the CLI
  required the nested `{"name": ..., "variables": {...}}` shape that
  `statqa parse-codebook` writes. `Codebook.from_dict` now accepts either.
- `InsightFormatter.format_temporal` and `format_causal` raised
  `UnboundLocalError`, and `format_univariate` raised `KeyError`, on any result
  lacking an optional section. Analyzers return `{"error": ...}` on their
  insufficient-data paths, so all three were reachable from ordinary data.
- Reading a `.xpt` (SAS transport) file called `pyreadstat.read_xpt`, which does
  not exist; the function is `read_xport`.
- Cohen's d is undefined at a perfect correlation, where the conversion returned
  `inf` and serialized as `Infinity` — not valid JSON. It is now omitted.
- Reading `.value` from `Variable.var_type` and `Variable.dgp` raised
  `AttributeError`: `use_enum_values` means those fields already hold the value.
- The text parser split missing-value codes on commas only, so
  `Missing: -1; 999` became the single code `-1; 999`. It now accepts `;` and
  `|` as well, matching the CSV parser.
- The bundled example scripts passed `type=` and `value_labels=` to `Variable`,
  which are named `var_type` and `valid_values`; pydantic ignored both, so every
  variable came out untyped and unlabelled. `PlotFactory(style="seaborn")` is
  also not a valid style and raised.

### Changed
- **Breaking:** the `dev` and `docs` extras are gone. Development dependencies
  moved to PEP 735 `[dependency-groups]`, so `pip install statqa[dev]` no longer
  resolves; use `uv sync --all-groups`. The remaining feature extras (`llm`,
  `pdf`, `statistical-formats`, `examples`, `all`) are unchanged.
- The version is derived from the git tag (hatchling + uv-dynamic-versioning)
  rather than being written in `pyproject.toml`.
- `pytest` no longer collects coverage or writes reports unless asked; use
  `pytest --cov`.
- `pydoclint` is no longer a dev dependency. It requires `docstring-parser-fork`,
  which collides with the `docstring-parser` that `anthropic` requires; run it
  isolated with `uvx pydoclint statqa/`.
- Type, DGP and missing-code parsing moved to `BaseParser`, and the five copies
  of the missing-code replacement to `statqa.utils.cleaning`.
- Documentation sources moved from `docs/source/` to `docs/`.

### Added
- `Codebook.from_dict`, accepting either codebook shape.
- Adopted the py-canon fleet standard: shared CI, docs and release workflows,
  `CITATION.cff`, dependabot and zizmor configuration.
- Test coverage raised from 22% to 75%, with CI enforcing a 70% floor.

## [0.3.0] - 2025-12-14

### Added
- **New Logging System**: Added simple, practical logging utility with `STATQA_DEBUG` environment variable support
  - `statqa.utils.logging.get_logger()` for standardized logging across modules
  - Debug mode enabled with `STATQA_DEBUG=1` for enhanced debugging
  - Structured logging in core analysis modules (univariate, bivariate)
  - Computational provenance logging in analysis operations

- **Enhanced Type System**: Comprehensive type hints using Python 3.12+ features
  - `Final` annotations for enum constants and immutable values
  - `Literal` types for constrained parameters (providers, analysis types)
  - `Self` type annotations for better IDE support
  - Pattern matching (`match/case`) replacing if/elif chains in parsers and enrichers

- **Structured Exception Hierarchy**: New `statqa.exceptions` module with specific exception types
  - `StatQAError`, `ParseError`, `ValidationError`, `AnalysisError`, `LLMError`
  - Error codes dictionary for programmatic error handling
  - Better error context and debugging information

- **Modern Type Definitions**: Comprehensive TypedDict definitions in `statqa.types`
  - `UnivariateResult`, `BivariateResult`, `QAPair` types
  - Enhanced type safety across analysis pipeline
  - Better IDE autocomplete and static analysis

- **Dependency Management**: Added `examples` optional dependency group
  - Moved `tqdm` to examples group (only used in example scripts)
  - Clean separation of core vs example dependencies
  - Comprehensive `all` installation option

### Changed
- **Toolchain Simplification**: Removed mypy static type checker
  - Eliminated complex mypy configuration and CI overhead
  - Preserved type hints for IDE support and documentation
  - Simplified development workflow while maintaining type safety
  - Removed TCH (type checking) rules from ruff configuration

- **Development Status**: Promoted from Alpha to Beta
  - Indicates increased stability and feature completeness
  - API stabilization for core analysis functionality
  - Production readiness for basic workflows

- **Code Modernization**: Updated codebase to leverage Python 3.12+ features
  - Pattern matching in `enricher.py` for provider selection
  - Enhanced error handling with structured exceptions
  - Improved type hints throughout codebase

### Improved
- **Debug Experience**: Enhanced debugging capabilities across the framework
  - Debug logging in core analysis modules
  - Computational step tracking for reproducibility
  - Environment-based debug control without complexity

- **Code Quality**: Comprehensive linting and formatting improvements
  - All ruff checks pass with zero issues
  - Cleaned up dependency management with deptry
  - Consistent code formatting and import organization

- **Example Scripts**: Updated example scripts to use structured logging
  - Replaced print statements with proper logging calls
  - Consistent logging patterns across all examples
  - Better debugging experience for users learning the framework

### Fixed
- **Dependency Issues**: Resolved all dependency validation problems
  - Fixed DEP002 violations by moving non-core dependencies to optional groups
  - Configured deptry to properly handle optional dependencies
  - Clean dependency separation between core, examples, and development needs

- **Import Issues**: Fixed Python 3.12+ compatibility
  - Added `from __future__ import annotations` where needed
  - Resolved pandas type subscripting errors
  - Enhanced compatibility across Python 3.12-3.14

### Technical Details
This release focuses on **developer experience** and **code modernization** without breaking existing APIs. Key technical improvements include:

- **Logging Architecture**: Simple, non-intrusive logging that respects the principle of not over-engineering
- **Type Safety**: Enhanced without the complexity overhead of mypy
- **Modern Python**: Leverages Python 3.12+ features for better performance and developer experience
- **Dependency Hygiene**: Clean separation of concerns in package dependencies

### Migration Guide
- **For Users**: No breaking changes - all existing code continues to work
- **For Contributors**:
  - No more mypy checks - ruff handles all linting
  - Use `STATQA_DEBUG=1` for enhanced debugging
  - Optional dependencies now properly organized by use case

## [0.2.0] - Previous Release
- Core statistical analysis framework
- Q/A generation with provenance tracking
- Multimodal visualization support
- LLM-powered metadata enrichment
