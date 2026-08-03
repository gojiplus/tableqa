"""Base parser interface for codebook parsing.

Defines the abstract interface that all codebook parsers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from statqa.metadata.schema import Codebook, DataGeneratingProcess, VariableType

#: Spellings accepted for a variable type, whatever the codebook format.
_TYPE_ALIASES: dict[str, VariableType] = {
    "numeric_continuous": VariableType.NUMERIC_CONTINUOUS,
    "numeric_discrete": VariableType.NUMERIC_DISCRETE,
    "numeric": VariableType.NUMERIC_CONTINUOUS,
    "continuous": VariableType.NUMERIC_CONTINUOUS,
    "discrete": VariableType.NUMERIC_DISCRETE,
    "categorical_nominal": VariableType.CATEGORICAL_NOMINAL,
    "categorical_ordinal": VariableType.CATEGORICAL_ORDINAL,
    "categorical": VariableType.CATEGORICAL_NOMINAL,
    "nominal": VariableType.CATEGORICAL_NOMINAL,
    "ordinal": VariableType.CATEGORICAL_ORDINAL,
    "boolean": VariableType.BOOLEAN,
    "bool": VariableType.BOOLEAN,
    "datetime": VariableType.DATETIME,
    "date": VariableType.DATETIME,
    "text": VariableType.TEXT,
    "string": VariableType.TEXT,
}

#: Spellings accepted for a data generating process.
_DGP_ALIASES: dict[str, DataGeneratingProcess] = {
    "observational": DataGeneratingProcess.OBSERVATIONAL,
    "experimental": DataGeneratingProcess.EXPERIMENTAL,
    "quasi_experimental": DataGeneratingProcess.QUASI_EXPERIMENTAL,
    "quasi-experimental": DataGeneratingProcess.QUASI_EXPERIMENTAL,
    "survey": DataGeneratingProcess.SURVEY,
    "administrative": DataGeneratingProcess.ADMINISTRATIVE,
    "simulation": DataGeneratingProcess.SIMULATION,
}

#: Separators a codebook may use between missing-value codes.
_MISSING_SEPARATORS = (";", ",", "|")


class BaseParser(ABC):
    """Abstract base class for codebook parsers."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the parser with format-specific configuration.

        Args:
            **kwargs: Parser-specific configuration options
        """
        self.config = kwargs

    @abstractmethod
    def parse(self, source: str | Path) -> Codebook:
        """Parse a codebook from the given source.

        Args:
            source: Path to codebook file or string content

        Returns:
            Parsed Codebook object

        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If source file doesn't exist
        """

    @abstractmethod
    def validate(self, source: str | Path) -> bool:
        """Check if this parser can handle the given source.

        Args:
            source: Path to codebook file or string content

        Returns:
            True if parser can handle this source
        """

    def parse_file(self, file_path: str | Path) -> Codebook:
        """Convenience method to parse from file path.

        Args:
            file_path: Path to codebook file

        Returns:
            Parsed Codebook object
        """
        return self.parse(file_path)

    def parse_string(self, content: str) -> Codebook:
        """Convenience method to parse from string content.

        Args:
            content: Codebook content as string

        Returns:
            Parsed Codebook object
        """
        return self.parse(content)

    # The field spellings below are properties of codebooks rather than of any
    # one file format, so every parser resolves them the same way.

    def _parse_type(self, type_str: str) -> VariableType:
        """Resolve a variable type spelling.

        Args:
            type_str: Type as written in the codebook.

        Returns:
            The matching type, or UNKNOWN if the spelling is not recognised.
        """
        return _TYPE_ALIASES.get(type_str.lower().strip(), VariableType.UNKNOWN)

    def _parse_dgp(self, dgp_str: str) -> DataGeneratingProcess:
        """Resolve a data generating process spelling.

        Args:
            dgp_str: Process as written in the codebook.

        Returns:
            The matching process, or UNKNOWN if the spelling is not recognised.
        """
        return _DGP_ALIASES.get(dgp_str.lower().strip(), DataGeneratingProcess.UNKNOWN)

    def _parse_missing(self, missing_str: str) -> set[int | str]:
        """Parse missing-value codes, e.g. '-1, 999, NA'.

        Codes are numeric where they parse as integers and left as strings
        otherwise, so sentinel labels such as NA survive.

        Args:
            missing_str: Codes as written in the codebook.

        Returns:
            The set of missing-value codes.
        """
        separator = next(
            (s for s in _MISSING_SEPARATORS if s in missing_str),
            ",",
        )

        missing: set[int | str] = set()
        for raw in missing_str.split(separator):
            item = raw.strip()
            if not item:
                continue
            try:
                missing.add(int(item))
            except ValueError:
                missing.add(item)
        return missing
