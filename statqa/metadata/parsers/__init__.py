"""Codebook parsers for various formats."""

from statqa.metadata.parsers.base import BaseParser
from statqa.metadata.parsers.csv import CSVParser
from statqa.metadata.parsers.text import TextParser

# Optional statistical format parser
try:
    from statqa.metadata.parsers.statistical import StatisticalFormatParser

    __all__ = ["BaseParser", "CSVParser", "StatisticalFormatParser", "TextParser"]
except ImportError:
    # pyreadstat not available
    __all__ = ["BaseParser", "CSVParser", "TextParser"]
