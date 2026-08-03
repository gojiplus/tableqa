"""Applying codebook missing-value codes to data.

Survey data encodes missingness as sentinel values -- -1 for "refused", 999
for "not ascertained" -- which are ordinary numbers to pandas and would
otherwise be averaged in with real responses. Every analyzer and plot has to
replace them with NaN first, so the rule lives here once.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from statqa.metadata.schema import Variable


def blank_missing_codes(data: pd.Series, variable: Variable) -> pd.Series:
    """Replace a variable's missing codes with NaN.

    Args:
        data: The column to clean.
        variable: Metadata carrying the missing-value codes.

    Returns:
        A copy with missing codes replaced; the original is untouched.
    """
    if not variable.missing_values:
        return data.copy()
    return data.replace(dict.fromkeys(variable.missing_values, np.nan))


def blank_missing_codes_frame(
    data: pd.DataFrame, variables: Iterable[Variable]
) -> pd.DataFrame:
    """Replace each variable's missing codes with NaN across a frame.

    Args:
        data: The frame to clean.
        variables: Metadata for the columns to clean; others are left alone.

    Returns:
        A copy with missing codes replaced; the original is untouched.
    """
    clean = data.copy()
    for variable in variables:
        if variable.missing_values:
            clean[variable.name] = blank_missing_codes(clean[variable.name], variable)
    return clean
