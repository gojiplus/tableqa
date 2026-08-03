"""Tests for plot theming."""

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from statqa.visualization.themes import setup_theme


@pytest.fixture(autouse=True)
def restore_rcparams():
    original = plt.rcParams.copy()
    yield
    plt.rcParams.update(original)


@pytest.mark.parametrize("style", ["publication", "presentation", "notebook"])
def test_supported_styles_apply(style):
    setup_theme(style)

    assert plt.rcParams["font.family"]


def test_styles_differ_from_each_other():
    setup_theme("publication")
    publication = plt.rcParams["lines.linewidth"]
    setup_theme("presentation")
    presentation = plt.rcParams["lines.linewidth"]

    assert publication != presentation


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="style"):
        setup_theme("neon")  # type: ignore[arg-type]
