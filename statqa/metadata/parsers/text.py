"""
Text-based codebook parser.

Parses structured text codebooks with variable definitions.
Supports formats like:

```
# Variable: age
Label: Respondent Age
Type: numeric_continuous
Units: years
Range: 18-99
Missing: -1, 999
Description: Age of respondent at time of survey

# Variable: gender
Label: Gender
Type: categorical_nominal
Values:
  1: Male
  2: Female
  3: Other
Missing: 0
```
"""

import re
from pathlib import Path
from typing import Any

from statqa.metadata.parsers.base import BaseParser
from statqa.metadata.schema import Codebook, DataGeneratingProcess, Variable, VariableType


class TextParser(BaseParser):
    """Parser for structured text codebooks."""

    def validate(self, source: str | Path) -> bool:
        """Check if source is valid text format."""
        try:
            content = self._read_source(source)
            # Simple check: does it have variable markers?
            return bool(re.search(r"(?:^|\n)#\s*Variable:", content, re.MULTILINE))
        except Exception:
            return False

    def parse(self, source: str | Path) -> Codebook:
        """Parse text codebook."""
        content = self._read_source(source)

        # Extract codebook-level metadata
        name = self._extract_codebook_name(source, content)
        description = self._extract_codebook_description(content)

        # Parse variables
        variables = self._parse_variables(content)

        return Codebook(
            name=name,
            description=description,
            variables={v.name: v for v in variables},
        )

    def _read_source(self, source: str | Path) -> str:
        """Read content from source."""
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")
        # Check if it looks like a file path (not too long and no newlines)
        if isinstance(source, str) and len(source) < 4096 and "\n" not in source:
            path = Path(source)
            try:
                if path.exists() and path.is_file():
                    return path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                pass
        # Treat as string content
        return str(source)

    def _extract_codebook_name(self, source: str | Path, content: str) -> str:
        """Extract codebook name."""
        # Try to find explicit name
        match = re.search(r"^#\s*Codebook:\s*(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Use filename if available
        if isinstance(source, str | Path):
            path = Path(source)
            if path.exists():
                return path.stem

        return "codebook"

    def _extract_codebook_description(self, content: str) -> str | None:
        """Extract codebook description."""
        match = re.search(
            r"^#\s*Description:\s*(.+?)(?=^#\s*Variable:|\Z)",
            content,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    def _parse_variables(self, content: str) -> list[Variable]:
        """Parse all variables from content."""
        variables = []

        # Split by variable markers
        var_blocks = re.split(r"^#\s*Variable:\s*", content, flags=re.MULTILINE)[1:]

        for block in var_blocks:
            variable = self._parse_variable_block(block)
            if variable:
                variables.append(variable)

        return variables

    def _parse_variable_block(self, block: str) -> Variable | None:
        """Parse a single variable block."""
        lines = block.split("\n")
        if not lines:
            return None

        # First line is variable name
        name = lines[0].strip()
        if not name:
            return None

        # Initialize variable data
        data: dict[str, Any] = {"name": name}

        # Parse remaining lines
        i = 1
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith("#"):
                continue

            # Parse key-value pairs
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                match key:
                    case "label":
                        data["label"] = value
                    case "type":
                        data["var_type"] = self._parse_type(value)
                    case "units":
                        data["units"] = value
                    case "range":
                        min_val, max_val = self._parse_range(value)
                        data["range_min"] = min_val
                        data["range_max"] = max_val
                    case "missing":
                        data["missing_values"] = self._parse_missing(value)
                    case "description":
                        # Description might be multi-line
                        desc_lines = [value]
                        while i < len(lines) and not re.match(r"^[A-Z][a-z]+:", lines[i]):
                            desc_lines.append(lines[i].strip())
                            i += 1
                        data["description"] = " ".join(desc_lines)
                    case "values":
                        # Parse value mappings (next lines indented)
                        values = {}
                        while i < len(lines) and lines[i].startswith(" "):
                            val_line = lines[i].strip()
                            if ":" in val_line:
                                code, label = val_line.split(":", 1)
                                try:
                                    values[int(code.strip())] = label.strip()
                                except ValueError:
                                    values[code.strip()] = label.strip()
                            i += 1
                        data["valid_values"] = values
                    case "dgp":
                        data["dgp"] = self._parse_dgp(value)
                    case "notes":
                        data["notes"] = value

        # Set default label if not provided
        if "label" not in data:
            data["label"] = name

        return Variable(**data)

    def _parse_type(self, type_str: str) -> VariableType:
        """Parse variable type string."""
        type_str = type_str.lower().strip()
        type_map = {
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
        return type_map.get(type_str, VariableType.UNKNOWN)

    def _parse_range(self, range_str: str) -> tuple[float | None, float | None]:
        """Parse range string like '18-99' or '0 to 100'."""
        # Try dash separator
        if "-" in range_str:
            parts = range_str.split("-")
            if len(parts) == 2:
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass

        # Try 'to' separator
        if " to " in range_str.lower():
            parts = range_str.lower().split(" to ")
            if len(parts) == 2:
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass

        return None, None

    def _parse_missing(self, missing_str: str) -> set[int | str]:
        """Parse missing values like '-1, 999' or 'NA, -1'."""
        missing = set()
        for item in missing_str.split(","):
            item = item.strip()
            try:
                missing.add(int(item))
            except ValueError:
                if item:  # Add non-empty strings
                    missing.add(item)
        return missing

    def _parse_dgp(self, dgp_str: str) -> DataGeneratingProcess:
        """Parse data generating process."""
        dgp_str = dgp_str.lower().strip()
        dgp_map = {
            "observational": DataGeneratingProcess.OBSERVATIONAL,
            "experimental": DataGeneratingProcess.EXPERIMENTAL,
            "quasi_experimental": DataGeneratingProcess.QUASI_EXPERIMENTAL,
            "quasi-experimental": DataGeneratingProcess.QUASI_EXPERIMENTAL,
            "survey": DataGeneratingProcess.SURVEY,
            "administrative": DataGeneratingProcess.ADMINISTRATIVE,
            "simulation": DataGeneratingProcess.SIMULATION,
        }
        return dgp_map.get(dgp_str, DataGeneratingProcess.UNKNOWN)
