"""Consistent theming for visualizations."""

from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns


def setup_theme(style: Literal["publication", "presentation", "notebook"] = "publication") -> None:
    """
    Set up matplotlib/seaborn theme.

    Args:
        style: Theme style ('publication', 'presentation', 'notebook')

    Raises:
        ValueError: If style is not supported
    """
    if style == "publication":
        # Publication-ready theme
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans"],
                "axes.linewidth": 1.0,
                "grid.linewidth": 0.5,
                "lines.linewidth": 1.5,
                "patch.linewidth": 0.5,
                "xtick.major.width": 1.0,
                "ytick.major.width": 1.0,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

    elif style == "presentation":
        # Presentation theme (larger fonts, higher contrast)
        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=1.3)
        plt.rcParams.update(
            {
                "figure.dpi": 100,
                "savefig.dpi": 200,
                "lines.linewidth": 2.5,
            }
        )

    elif style == "notebook":
        # Jupyter notebook theme
        sns.set_style("whitegrid")
        sns.set_context("notebook")
        plt.rcParams.update(
            {
                "figure.dpi": 100,
                "savefig.dpi": 150,
            }
        )

    else:
        raise ValueError(f"Unknown style: {style}")
