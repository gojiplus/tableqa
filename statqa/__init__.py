"""TableQA: Automatically extract structured facts from tabular datasets.

This package provides tools for:
- Parsing codebooks and metadata
- Running statistical analyses (univariate, bivariate, temporal, causal)
- Generating natural language insights
- Creating visualizations
- Generating Q/A pairs for LLM fine-tuning or RAG
"""

from importlib.metadata import version

from statqa.analysis.bivariate import BivariateAnalyzer
from statqa.analysis.causal import CausalAnalyzer
from statqa.analysis.temporal import TemporalAnalyzer
from statqa.analysis.univariate import UnivariateAnalyzer

# Import main public APIs
from statqa.metadata.schema import Codebook, Variable, VariableType

# Get version from package metadata
__version__ = version("statqa")
__version_info__ = tuple(int(i) for i in __version__.split(".") if i.isdigit())


__all__ = [
    "BivariateAnalyzer",
    "CausalAnalyzer",
    # Metadata
    "Codebook",
    "TemporalAnalyzer",
    # Analysis
    "UnivariateAnalyzer",
    "Variable",
    "VariableType",
    "__version__",
    "__version_info__",
]
