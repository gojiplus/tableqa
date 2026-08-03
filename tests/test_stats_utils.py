"""Tests for the statistical helper functions."""

import numpy as np
import pandas as pd
import pytest

from statqa.utils.stats import (
    calculate_effect_size,
    cohens_d,
    correct_multiple_testing,
    cramers_v,
    detect_outliers,
    mann_kendall_trend,
    robust_stats,
)


class TestCohensD:
    def test_identical_groups_have_no_effect(self):
        group = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        assert cohens_d(group, group) == pytest.approx(0.0)

    def test_separated_groups_have_a_large_effect(self):
        rng = np.random.default_rng(51)
        a = pd.Series(rng.normal(0, 1, 200))
        b = pd.Series(rng.normal(3, 1, 200))

        assert abs(cohens_d(a, b)) > 2.0

    def test_sign_follows_argument_order(self):
        rng = np.random.default_rng(52)
        a = pd.Series(rng.normal(0, 1, 200))
        b = pd.Series(rng.normal(3, 1, 200))

        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))


class TestEffectSize:
    def test_cohen_d_dispatch(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([4.0, 5.0, 6.0])

        assert calculate_effect_size(a, b, effect_type="cohen_d") < 0

    def test_cohen_d_requires_two_samples(self):
        with pytest.raises(ValueError, match="two samples"):
            calculate_effect_size(pd.Series([1.0, 2.0]), effect_type="cohen_d")

    def test_cohen_d_rejects_a_scalar(self):
        with pytest.raises(ValueError, match="array-like"):
            calculate_effect_size(0.5, pd.Series([1.0]), effect_type="cohen_d")

    def test_r_to_d_converts(self):
        # r = 0.5 -> d = 2*0.5/sqrt(1-0.25)
        assert calculate_effect_size(0.5, effect_type="r_to_d") == pytest.approx(
            1.1547, abs=1e-3
        )

    def test_r_to_d_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="correlation coefficient"):
            calculate_effect_size(pd.Series([0.5]), effect_type="r_to_d")

    @pytest.mark.parametrize("r", [1.0, -1.0, 1.5])
    def test_r_to_d_rejects_perfect_correlation(self, r):
        # Otherwise the result is inf, which serialises as invalid JSON.
        with pytest.raises(ValueError, match="perfect correlation"):
            calculate_effect_size(r, effect_type="r_to_d")

    def test_unknown_effect_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown effect_type"):
            calculate_effect_size(0.5, effect_type="nonsense")  # type: ignore[arg-type]

    def test_eta_squared_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            calculate_effect_size(0.5, effect_type="eta_squared")


class TestCramersV:
    def test_independent_table_is_near_zero(self):
        table = pd.DataFrame([[50, 50], [50, 50]])

        assert cramers_v(table) == pytest.approx(0.0, abs=0.01)

    def test_perfectly_associated_table_is_near_one(self):
        table = pd.DataFrame([[100, 0], [0, 100]])

        # Just under 1.0: chi2_contingency applies Yates' continuity correction
        # to 2x2 tables by default, which shrinks the statistic slightly.
        assert cramers_v(table) == pytest.approx(1.0, abs=0.02)


class TestMultipleTesting:
    @pytest.mark.parametrize("method", ["bonferroni", "fdr_bh", "fdr_by"])
    def test_correction_never_lowers_a_p_value(self, method):
        p_values = [0.001, 0.01, 0.04, 0.2, 0.9]

        _, corrected = correct_multiple_testing(p_values, method=method)

        assert all(c >= p for c, p in zip(corrected, p_values, strict=True))

    def test_bonferroni_is_the_most_conservative(self):
        p_values = [0.01, 0.02, 0.03, 0.04]

        _, bonf = correct_multiple_testing(p_values, method="bonferroni")
        _, bh = correct_multiple_testing(p_values, method="fdr_bh")

        assert all(b >= f for b, f in zip(bonf, bh, strict=True))

    def test_rejection_flags_match_alpha(self):
        reject, _ = correct_multiple_testing(
            [0.0001, 0.5], method="bonferroni", alpha=0.05
        )

        assert bool(reject[0]) is True
        assert bool(reject[1]) is False


class TestRobustStats:
    def test_reports_median_and_mad(self):
        result = robust_stats(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert result["median"] == pytest.approx(3.0)
        assert result["mad"] > 0

    def test_median_resists_an_outlier(self):
        clean = robust_stats(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        contaminated = robust_stats(pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0]))

        assert clean["median"] == contaminated["median"]


class TestOutlierDetection:
    @pytest.mark.parametrize("method", ["iqr", "mad", "zscore"])
    def test_flags_an_extreme_value(self, method):
        data = pd.Series([*[1.0, 2.0, 3.0, 4.0, 5.0] * 10, 500.0])

        flags = detect_outliers(data, method=method, threshold=3.0)

        assert bool(flags[-1]) is True

    def test_clean_data_has_no_outliers(self):
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10)

        assert not detect_outliers(data, method="iqr").any()

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method"):
            detect_outliers(pd.Series([1.0, 2.0]), method="nonsense")  # type: ignore[arg-type]


class TestMannKendall:
    def test_detects_an_increasing_series(self):
        result = mann_kendall_trend(pd.Series(range(30)))

        assert result["trend"] == "increasing"
        assert result["tau"] == pytest.approx(1.0)

    def test_detects_a_decreasing_series(self):
        result = mann_kendall_trend(pd.Series(range(30, 0, -1)))

        assert result["trend"] == "decreasing"

    def test_flat_series_has_no_trend(self):
        result = mann_kendall_trend(pd.Series([5.0] * 30))

        assert result["trend"] == "no_trend"
