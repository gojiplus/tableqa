"""
Statistical format parser for SPSS, Stata, and SAS files.

Uses pyreadstat library to parse statistical data files and extract rich metadata
including variable labels, value labels, and missing value definitions.
"""

from pathlib import Path
from typing import Any


try:
    import pyreadstat

    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

from statqa.metadata.parsers.base import BaseParser
from statqa.metadata.schema import Codebook, DataGeneratingProcess, Variable, VariableType


class StatisticalFormatParser(BaseParser):
    """Parser for statistical data files (SPSS, Stata, SAS)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not HAS_PYREADSTAT:
            raise ImportError(
                "pyreadstat is required for statistical format parsing. "
                "Install with: pip install statqa[statistical-formats]"
            )

    def validate(self, source: str | Path) -> bool:
        """Check if source is a supported statistical format."""
        if not HAS_PYREADSTAT:
            return False

        if isinstance(source, str) and not Path(source).exists():
            return False

        path = Path(source)
        supported_extensions = {".sav", ".zsav", ".por", ".dta", ".sas7bdat", ".xpt"}
        return path.suffix.lower() in supported_extensions

    def parse(self, source: str | Path) -> Codebook:
        """Parse statistical format file."""
        path = Path(source)

        # Read metadata only first for efficiency
        try:
            metadata = self._read_metadata_only(path)
        except Exception as e:
            raise ValueError(f"Failed to read metadata from {path}: {e}") from e

        # Extract codebook info
        codebook_name = path.stem
        codebook_description = getattr(metadata, "file_label", None)

        # Parse variables from metadata
        variables = self._extract_variables(metadata)

        # Build dataset info from metadata
        dataset_info = self._extract_dataset_info(metadata)

        return Codebook(
            name=codebook_name,
            description=codebook_description,
            variables={v.name: v for v in variables},
            dataset_info=dataset_info,
        )

    def _read_metadata_only(self, path: Path) -> Any:
        """Read only metadata from statistical file."""
        suffix = path.suffix.lower()

        match suffix:
            case ".sav" | ".zsav":
                _, metadata = pyreadstat.read_sav(str(path), metadataonly=True)
            case ".por":
                _, metadata = pyreadstat.read_por(str(path), metadataonly=True)
            case ".dta":
                _, metadata = pyreadstat.read_dta(str(path), metadataonly=True)
            case ".sas7bdat":
                _, metadata = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
            case ".xpt":
                _, metadata = pyreadstat.read_xpt(str(path), metadataonly=True)
            case _:
                raise ValueError(f"Unsupported file format: {suffix}")

        return metadata

    def _extract_variables(self, metadata: Any) -> list[Variable]:
        """Extract Variable objects from pyreadstat metadata."""
        variables = []

        # Get column information
        column_names = getattr(metadata, "column_names", [])
        column_labels = getattr(metadata, "column_names_to_labels", {})
        variable_value_labels = getattr(metadata, "variable_value_labels", {})
        missing_ranges = getattr(metadata, "missing_ranges", {})
        original_variable_types = getattr(metadata, "original_variable_types", {})

        for col_name in column_names:
            # Basic variable info
            var_data = {
                "name": col_name,
                "label": column_labels.get(col_name) or col_name,
            }

            # Variable type inference
            var_data["var_type"] = self._infer_variable_type(
                col_name, variable_value_labels, original_variable_types
            )

            # Value labels
            if col_name in variable_value_labels:
                var_data["valid_values"] = variable_value_labels[col_name]
                # If has value labels, likely categorical
                if var_data["var_type"] == VariableType.UNKNOWN:
                    var_data["var_type"] = VariableType.CATEGORICAL_NOMINAL

            # Missing values from missing ranges
            if col_name in missing_ranges:
                missing_range = missing_ranges[col_name]
                missing_values = set()

                # Handle different missing range formats
                if isinstance(missing_range, dict):
                    lo = missing_range.get("lo")
                    hi = missing_range.get("hi")
                    if lo is not None:
                        missing_values.add(lo)
                    if hi is not None and hi != lo:
                        missing_values.add(hi)
                elif isinstance(missing_range, list | tuple):
                    missing_values.update(missing_range)

                var_data["missing_values"] = missing_values

            # Infer data generating process based on file type
            var_data["dgp"] = DataGeneratingProcess.SURVEY  # Most statistical files are surveys

            variables.append(Variable(**var_data))

        return variables

    def _infer_variable_type(
        self,
        col_name: str,
        variable_value_labels: dict[str, dict],
        original_variable_types: dict[str, Any],
    ) -> VariableType:
        """Infer StatQA VariableType from metadata."""
        # Check if has value labels (likely categorical)
        if variable_value_labels.get(col_name):
            return VariableType.CATEGORICAL_NOMINAL

        # Check original type if available
        if col_name in original_variable_types:
            orig_type = str(original_variable_types[col_name]).lower()

            # Map common type names
            if "string" in orig_type or "char" in orig_type or "str" in orig_type:
                return VariableType.TEXT
            elif "date" in orig_type or "time" in orig_type:
                return VariableType.DATETIME
            elif "int" in orig_type or "long" in orig_type:
                return VariableType.NUMERIC_DISCRETE
            elif "float" in orig_type or "double" in orig_type or "numeric" in orig_type:
                return VariableType.NUMERIC_CONTINUOUS

        # Default to unknown - will be inferred during analysis
        return VariableType.UNKNOWN

    def _extract_dataset_info(self, metadata: Any) -> dict[str, Any]:
        """Extract general dataset information from metadata."""
        dataset_info: dict[str, Any] = {}

        # Basic file information
        if hasattr(metadata, "number_rows"):
            dataset_info["number_rows"] = metadata.number_rows
        if hasattr(metadata, "number_columns"):
            dataset_info["number_columns"] = metadata.number_columns
        if hasattr(metadata, "file_encoding"):
            dataset_info["file_encoding"] = metadata.file_encoding

        # Timestamps
        if hasattr(metadata, "creation_time"):
            dataset_info["creation_time"] = str(metadata.creation_time)
        if hasattr(metadata, "modification_time"):
            dataset_info["modification_time"] = str(metadata.modification_time)

        # File-specific metadata
        if hasattr(metadata, "file_label"):
            dataset_info["file_label"] = metadata.file_label
        if hasattr(metadata, "notes"):
            dataset_info["notes"] = metadata.notes

        # Additional metadata as raw dict
        dataset_info["raw_metadata"] = vars(metadata)

        return dataset_info
