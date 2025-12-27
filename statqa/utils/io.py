"""I/O utilities for loading and saving data."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def load_data(
    source: str | Path,
    file_pattern: str = r"(?i)\.csv$",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load data from various sources.

    Args:
        source: Path to file (CSV, ZIP containing CSVs, etc.)
        file_pattern: Regex pattern for files in ZIP
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If source doesn't exist
    """
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Handle ZIP files
    if source_path.suffix.lower() == ".zip":
        return _load_from_zip(source_path, file_pattern, **kwargs)

    # Handle CSV
    if source_path.suffix.lower() == ".csv":
        return pd.read_csv(source_path, **kwargs)

    # Try to load as CSV anyway
    return pd.read_csv(source_path, **kwargs)


def _load_from_zip(
    zip_path: Path,
    file_pattern: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load CSVs from ZIP file."""
    import re

    dfs = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.startswith("__MACOSX/") or not re.search(file_pattern, member):
                continue
            with z.open(member) as f:
                dfs.append(pd.read_csv(f, **kwargs))

    if not dfs:
        raise ValueError(f"No files matching {file_pattern!r} found in {zip_path}")

    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def save_json(data: Any, output_path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON-serializable)
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_json(input_path: str | Path) -> Any:
    """
    Load data from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Loaded data
    """
    return json.loads(Path(input_path).read_text(encoding="utf-8"))
