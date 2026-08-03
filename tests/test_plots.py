"""Tests for plot generation and the visual metadata that ships with it."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from statqa.metadata.schema import Variable, VariableType
from statqa.visualization.plots import PlotFactory
from tests.factories import numeric_var


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def factory():
    return PlotFactory()


@pytest.fixture
def continuous():
    return Variable(
        name="age", label="Age", var_type=VariableType.NUMERIC_CONTINUOUS, units="years"
    )


@pytest.fixture
def categorical():
    return Variable(
        name="gender",
        label="Gender",
        var_type=VariableType.CATEGORICAL_NOMINAL,
        valid_values={1: "Male", 2: "Female"},
    )


@pytest.fixture
def frame():
    rng = np.random.default_rng(31)
    n = 240
    age = rng.normal(50, 12, n)
    return pd.DataFrame(
        {
            "age": age,
            "gender": rng.choice([1, 2], size=n),
            "income": age * 800 + rng.normal(0, 3000, n),
            "year": rng.choice([2018, 2019, 2020, 2021], size=n),
        }
    )


class TestUnivariate:
    def test_continuous_renders(self, factory, frame, continuous):
        fig = factory.plot_univariate(frame["age"], continuous)

        assert fig.axes

    def test_categorical_renders(self, factory, frame, categorical):
        fig = factory.plot_univariate(frame["gender"], categorical)

        assert fig.axes[0].patches

    def test_axis_is_labelled_with_the_variable_label(self, factory, frame, continuous):
        fig = factory.plot_univariate(frame["age"], continuous)

        assert fig.axes[0].get_xlabel() == "Age"

    def test_discrete_numeric_uses_counts(self, factory, frame):
        var = Variable(
            name="year", label="Year", var_type=VariableType.NUMERIC_DISCRETE
        )

        fig = factory.plot_univariate(frame["year"], var)

        assert fig.axes

    def test_writes_the_file_when_given_a_path(
        self, factory, frame, continuous, tmp_path
    ):
        out = tmp_path / "plot.png"

        factory.plot_univariate(frame["age"], continuous, output_path=out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_metadata_describes_the_plot(self, factory, frame, continuous):
        _, meta = factory.plot_univariate(
            frame["age"], continuous, return_metadata=True
        )

        assert meta["plot_type"]
        # Captions lowercase the label to read correctly mid-sentence.
        assert "age" in meta["caption"].lower()
        assert meta["alt_text"]


class TestBivariate:
    def test_numeric_pair_is_a_scatter(self, factory, frame, continuous):
        _, meta = factory.plot_bivariate(
            frame, continuous, numeric_var("income"), return_metadata=True
        )

        assert meta["plot_type"] == "scatter"

    def test_categorical_numeric_is_a_boxplot(
        self, factory, frame, categorical, continuous
    ):
        _, meta = factory.plot_bivariate(
            frame, categorical, continuous, return_metadata=True
        )

        assert meta["plot_type"] == "boxplot"

    def test_categorical_pair_is_a_heatmap(self, factory, frame, categorical):
        other = Variable(
            name="year", label="Year", var_type=VariableType.CATEGORICAL_NOMINAL
        )

        _, meta = factory.plot_bivariate(
            frame, categorical, other, return_metadata=True
        )

        assert meta["plot_type"] == "heatmap"

    def test_caption_mentions_both_variables(self, factory, frame, continuous):
        _, meta = factory.plot_bivariate(
            frame, continuous, numeric_var("income"), return_metadata=True
        )

        assert "Age" in meta["caption"]
        assert "income" in meta["caption"]

    def test_writes_the_file_when_given_a_path(
        self, factory, frame, continuous, tmp_path
    ):
        out = tmp_path / "biv.png"

        factory.plot_bivariate(
            frame, continuous, numeric_var("income"), output_path=out
        )

        assert out.exists()


class TestTemporal:
    def test_renders_a_trend(self, factory, frame):
        time_var = Variable(
            name="year", label="Year", var_type=VariableType.NUMERIC_DISCRETE
        )

        fig = factory.plot_temporal(frame, time_var, numeric_var("income"))

        assert fig.axes

    def test_grouped_trend_renders(self, factory, frame, categorical):
        time_var = Variable(
            name="year", label="Year", var_type=VariableType.NUMERIC_DISCRETE
        )

        fig = factory.plot_temporal(
            frame, time_var, numeric_var("income"), group_var=categorical
        )

        assert fig.axes

    def test_writes_the_file_when_given_a_path(self, factory, frame, tmp_path):
        time_var = Variable(
            name="year", label="Year", var_type=VariableType.NUMERIC_DISCRETE
        )
        out = tmp_path / "temporal.png"

        factory.plot_temporal(frame, time_var, numeric_var("income"), output_path=out)

        assert out.exists()


class TestFactoryOptions:
    def test_figsize_and_dpi_are_applied(self, frame, continuous):
        factory = PlotFactory(figsize=(4, 3), dpi=72)

        fig = factory.plot_univariate(frame["age"], continuous)

        assert tuple(fig.get_size_inches()) == (4.0, 3.0)
        assert fig.dpi == 72
