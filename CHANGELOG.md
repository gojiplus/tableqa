# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

---

**Full Changelog**: https://github.com/gojiplus/statqa/compare/v0.2.0...v0.3.0
