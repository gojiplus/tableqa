"""Sphinx configuration — fleet standard via py-canon, plus repo specifics."""

from py_canon.sphinx import configure

configure(
    globals(),
    # The API reference is full of pandas and numpy types, so resolve them
    # rather than rendering them as bare unlinked names.
    intersphinx_mapping={
        "python": ("https://docs.python.org/3", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
    },
    autodoc_default_options={
        "members": True,
        "undoc-members": True,
        "show-inheritance": True,
    },
    # The pydantic models document their fields in an Attributes: section and
    # autodoc emits each field as well. Rendering the section as :ivar: fields
    # keeps the prose without declaring every attribute a second time.
    napoleon_use_ivar=True,
    # Furo's "edit this page" links.
    html_theme_options={
        "source_repository": "https://github.com/gojiplus/statqa",
        "source_branch": "main",
        "source_directory": "docs/",
    },
    html_static_path=["_static"],
)
