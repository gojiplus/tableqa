"""Sphinx configuration for statqa documentation."""

import sys
from pathlib import Path

from statqa import __version__


# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Project information
project = "statqa"
copyright = "2025, StatQA Contributors"
author = "StatQA Contributors"

# Version
version = __version__
release = __version__

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Templates
templates_path = ["_templates"]
exclude_patterns = []

# HTML output
html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/gojiplus/statqa",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
