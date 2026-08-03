"""CSV-based codebook parser.

Parses codebooks stored in CSV format with columns like:
- variable_name
- label
- type
- description
- valid_values
- missing_values
- units
- etc.
"""

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from statqa.metadata.parsers.base import BaseParser
from statqa.metadata.schema import (
    Codebook,
    Variable,
)


class CSVParser(BaseParser):
    """Parser for CSV codebooks."""

    def validate(self, source: str | Path) -> bool:
        """Check if source is valid CSV."""
        try:
            if isinstance(source, str) and not Path(source).exists():
                return False
            df = pd.read_csv(source, nrows=1)
            # Check for required columns
            required = {"variable_name", "label"} | {"varname", "name"}
            return bool(required & set(df.columns.str.lower()))
        except Exception:
            return False

    def parse(self, source: str | Path) -> Codebook:
        """Parse CSV codebook."""
        df = pd.read_csv(source)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Determine name column
        name_col = self._get_name_column(df)
        if not name_col:
            raise ValueError(
                "CSV must have 'variable_name', 'varname', or 'name' column"
            )

        # Extract codebook metadata
        codebook_name = (
            Path(source).stem if isinstance(source, str | Path) else "codebook"
        )

        # Parse variables
        variables = []
        for _, row in df.iterrows():
            variable = self._parse_variable_row(row, name_col)
            if variable:
                variables.append(variable)

        return Codebook(
            name=codebook_name,
            variables={v.name: v for v in variables},
        )

    def _get_name_column(self, df: pd.DataFrame) -> str | None:
        """Find the variable name column."""
        candidates = ["variable_name", "varname", "name", "variable", "var"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _parse_variable_row(self, row: pd.Series, name_col: str) -> Variable | None:
        """Parse a single row into a Variable."""
        name = str(row[name_col]).strip()
        if not name or pd.isna(row[name_col]):
            return None

        # Build variable data
        data: dict[str, Any] = {
            "name": name,
            "label": self._get_value(row, ["label", "description", "question"], name),
        }

        # Type
        if type_val := self._get_value(row, ["type", "var_type", "variable_type"]):
            data["var_type"] = self._parse_type(type_val)

        # Description
        if desc := self._get_value(
            row, ["description", "detailed_description", "notes", "question_text"]
        ):
            data["description"] = desc

        # Valid values
        if valid_str := self._get_value(
            row, ["valid_values", "values", "value_labels"]
        ):
            data["valid_values"] = self._parse_values(valid_str)

        # Missing values
        if missing_str := self._get_value(
            row, ["missing_values", "missing", "missing_codes"]
        ):
            data["missing_values"] = self._parse_missing(missing_str)

        # Units
        if units := self._get_value(row, ["units", "unit", "measurement_unit"]):
            data["units"] = units

        # Range
        if range_min := self._get_value(row, ["range_min", "min", "minimum"]):
            with contextlib.suppress(ValueError, TypeError):
                data["range_min"] = float(range_min)

        if range_max := self._get_value(row, ["range_max", "max", "maximum"]):
            with contextlib.suppress(ValueError, TypeError):
                data["range_max"] = float(range_max)

        # DGP
        if dgp := self._get_value(row, ["dgp", "data_generating_process"]):
            data["dgp"] = self._parse_dgp(dgp)

        # Flags
        data["is_treatment"] = self._get_bool(row, ["is_treatment", "treatment"])
        data["is_outcome"] = self._get_bool(row, ["is_outcome", "outcome"])
        data["is_confounder"] = self._get_bool(row, ["is_confounder", "confounder"])
        data["is_ordinal"] = self._get_bool(row, ["is_ordinal", "ordinal"])

        # Source
        if source := self._get_value(row, ["source", "question_source"]):
            data["source"] = source

        # Additional notes
        if notes := self._get_value(row, ["notes", "comments", "additional_notes"]):
            data["notes"] = notes

        return Variable(**data)

    def _get_value(
        self, row: pd.Series, columns: list[str], default: Any = None
    ) -> str | None:
        """Get value from first available column."""
        for col in columns:
            if col in row.index and not pd.isna(row[col]):
                return str(row[col]).strip()
        return default

    def _get_bool(self, row: pd.Series, columns: list[str]) -> bool:
        """Get boolean value from first available column."""
        val = self._get_value(row, columns, "false")
        if not val:
            return False
        val_lower = val.lower()
        return val_lower in {"true", "1", "yes", "y", "t"}

    def _parse_values(self, values_str: str) -> dict[int | str, str]:
        """Parse value mappings from string like '1: Male; 2: Female' or '1=Male, 2=Female'."""
        values = {}

        # Try semicolon separator first
        items = values_str.split(";") if ";" in values_str else values_str.split(",")

        for item in items:
            item = item.strip()
            if not item:
                continue

            # Try colon separator
            if ":" in item:
                code, label = item.split(":", 1)
            # Try equals separator
            elif "=" in item:
                code, label = item.split("=", 1)
            else:
                continue

            code = code.strip()
            label = label.strip()

            # Try to convert to int
            try:
                values[int(code)] = label
            except ValueError:
                values[code] = label

        return values
