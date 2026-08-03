"""Tests for applying codebook missing-value codes."""

import numpy as np
import pandas as pd
import pytest

from statqa.utils.cleaning import blank_missing_codes, blank_missing_codes_frame
from tests.factories import numeric_var


@pytest.fixture
def age():
    return numeric_var("age", missing_values={-1, 999})


def test_sentinels_become_nan(age):
    series = pd.Series([25.0, -1.0, 40.0, 999.0])

    cleaned = blank_missing_codes(series, age)

    assert cleaned.isna().tolist() == [False, True, False, True]


def test_real_values_are_untouched(age):
    series = pd.Series([25.0, -1.0, 40.0])

    assert blank_missing_codes(series, age).dropna().tolist() == [25.0, 40.0]


def test_the_input_is_not_mutated(age):
    series = pd.Series([25.0, -1.0])

    blank_missing_codes(series, age)

    assert series.tolist() == [25.0, -1.0]


def test_a_variable_without_codes_is_a_passthrough():
    plain = numeric_var("age")
    series = pd.Series([1.0, -1.0])

    assert blank_missing_codes(series, plain).tolist() == [1.0, -1.0]


def test_sentinels_would_otherwise_skew_the_mean(age):
    # The point of the exercise: -1 and 999 are ordinary numbers to pandas.
    series = pd.Series([40.0, 50.0, 999.0])

    assert series.mean() == pytest.approx(363.0)
    assert blank_missing_codes(series, age).mean() == pytest.approx(45.0)


class TestFrame:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"age": [25.0, -1.0], "income": [100.0, 999.0]})

    def test_each_column_uses_its_own_codes(self, frame):
        variables = [
            numeric_var("age", missing_values={-1}),
            numeric_var("income", missing_values={999}),
        ]

        cleaned = blank_missing_codes_frame(frame, variables)

        assert np.isnan(cleaned["age"].iloc[1])
        assert np.isnan(cleaned["income"].iloc[1])

    def test_columns_without_metadata_are_left_alone(self, frame):
        cleaned = blank_missing_codes_frame(
            frame, [numeric_var("age", missing_values={-1})]
        )

        assert cleaned["income"].tolist() == [100.0, 999.0]

    def test_the_input_frame_is_not_mutated(self, frame):
        blank_missing_codes_frame(frame, [numeric_var("age", missing_values={-1})])

        assert frame["age"].tolist() == [25.0, -1.0]
