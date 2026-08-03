"""Builders for the Variable metadata the analyzers take.

Every analysis test needs a Variable of some type, and each one built its own
one-line helper. They live here instead so a schema change lands in one place.
"""

from statqa.metadata.schema import Variable, VariableType


def numeric_var(name: str, label: str | None = None, **kwargs) -> Variable:
    """Build a continuous numeric variable."""
    return Variable(
        name=name,
        label=label or name,
        var_type=VariableType.NUMERIC_CONTINUOUS,
        **kwargs,
    )


def discrete_var(name: str, label: str | None = None, **kwargs) -> Variable:
    """Build a discrete numeric variable."""
    return Variable(
        name=name,
        label=label or name,
        var_type=VariableType.NUMERIC_DISCRETE,
        **kwargs,
    )


def categorical_var(
    name: str,
    values: dict | None = None,
    label: str | None = None,
    **kwargs,
) -> Variable:
    """Build a nominal categorical variable."""
    return Variable(
        name=name,
        label=label or name,
        var_type=VariableType.CATEGORICAL_NOMINAL,
        valid_values=values or {},
        **kwargs,
    )
