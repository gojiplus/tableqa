"""
Base parser interface for codebook parsing.

Defines the abstract interface that all codebook parsers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from statqa.metadata.schema import Codebook


class BaseParser(ABC):
    """
    Abstract base class for codebook parsers.

    Args:
        **kwargs: Parser-specific configuration options
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def parse(self, source: str | Path) -> Codebook:
        """
        Parse a codebook from the given source.

        Args:
            source: Path to codebook file or string content

        Returns:
            Parsed Codebook object

        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If source file doesn't exist
        """
        pass

    @abstractmethod
    def validate(self, source: str | Path) -> bool:
        """
        Check if this parser can handle the given source.

        Args:
            source: Path to codebook file or string content

        Returns:
            True if parser can handle this source
        """
        pass

    def parse_file(self, file_path: str | Path) -> Codebook:
        """
        Convenience method to parse from file path.

        Args:
            file_path: Path to codebook file

        Returns:
            Parsed Codebook object
        """
        return self.parse(file_path)

    def parse_string(self, content: str) -> Codebook:
        """
        Convenience method to parse from string content.

        Args:
            content: Codebook content as string

        Returns:
            Parsed Codebook object
        """
        return self.parse(content)
