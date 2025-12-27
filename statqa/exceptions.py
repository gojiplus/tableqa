"""
Custom exceptions for statqa package.

This module defines exception hierarchy for better error handling
using Python 3.12+ features like exception groups.
"""

from typing import Final


class StatqaError(Exception):
    """Base exception for statqa package."""

    pass


class ParseError(StatqaError):
    """Base exception for parsing errors."""

    pass


class CodebookParseError(ParseError):
    """Error parsing codebook files."""

    pass


class VariableParseError(ParseError):
    """Error parsing individual variables."""

    pass


class AnalysisError(StatqaError):
    """Base exception for analysis errors."""

    pass


class StatisticalAnalysisError(AnalysisError):
    """Error in statistical computations."""

    pass


class EnrichmentError(StatqaError):
    """Base exception for metadata enrichment errors."""

    pass


class LLMConnectionError(EnrichmentError):
    """LLM API connection or authentication error."""

    pass


class LLMResponseError(EnrichmentError):
    """Error parsing LLM response."""

    pass


class ExportError(StatqaError):
    """Error exporting data or results."""

    pass


class ValidationError(StatqaError):
    """Data validation error."""

    pass


# Error codes for programmatic handling
ERROR_CODES: Final[dict[str, int]] = {
    "UNKNOWN": 1000,
    "PARSE_ERROR": 1001,
    "CODEBOOK_PARSE_ERROR": 1002,
    "VARIABLE_PARSE_ERROR": 1003,
    "ANALYSIS_ERROR": 2001,
    "STATISTICAL_ANALYSIS_ERROR": 2002,
    "ENRICHMENT_ERROR": 3001,
    "LLM_CONNECTION_ERROR": 3002,
    "LLM_RESPONSE_ERROR": 3003,
    "EXPORT_ERROR": 4001,
    "VALIDATION_ERROR": 5001,
}
