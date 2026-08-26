"""Tests for the command line interface."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from typer.testing import CliRunner

from statqa.cli.main import app

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(21)
    n = 120
    age = rng.normal(50, 10, n)
    frame = pd.DataFrame(
        {
            "age": age,
            "gender": rng.choice([1, 2], size=n),
            "income": age * 900 + rng.normal(0, 2000, n),
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def codebook_json(tmp_path):
    path = tmp_path / "codebook.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "variables": {
                    "age": {
                        "name": "age",
                        "label": "Age",
                        "var_type": "numeric_continuous",
                    },
                    "gender": {
                        "name": "gender",
                        "label": "Gender",
                        "var_type": "categorical_nominal",
                        "valid_values": {"1": "Male", "2": "Female"},
                    },
                    "income": {
                        "name": "income",
                        "label": "Income",
                        "var_type": "numeric_continuous",
                    },
                },
            }
        )
    )
    return path


@pytest.fixture
def codebook_csv(tmp_path):
    path = tmp_path / "codebook.csv"
    path.write_text(
        "variable_name,label,type\n"
        "age,Age,numeric_continuous\n"
        "gender,Gender,categorical\n"
        "income,Income,numeric_continuous\n"
    )
    return path


class TestVersion:
    def test_reports_a_version(self):
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "TableQA version" in result.stdout


class TestHelp:
    @pytest.mark.parametrize(
        "command", ["parse-codebook", "analyze", "generate-qa", "pipeline"]
    )
    def test_every_command_has_help(self, command):
        result = runner.invoke(app, [command, "--help"])

        assert result.exit_code == 0


class TestParseCodebook:
    def test_writes_a_codebook_json(self, codebook_csv, tmp_path):
        out = tmp_path / "parsed.json"

        result = runner.invoke(
            app, ["parse-codebook", str(codebook_csv), "--output", str(out)]
        )

        assert result.exit_code == 0, result.stdout
        assert set(json.loads(out.read_text())["variables"]) == {
            "age",
            "gender",
            "income",
        }

    def test_unknown_format_is_rejected(self, codebook_csv, tmp_path):
        result = runner.invoke(
            app,
            [
                "parse-codebook",
                str(codebook_csv),
                "--format",
                "nonsense",
                "--output",
                str(tmp_path / "x.json"),
            ],
        )

        assert result.exit_code != 0


class TestAnalyze:
    def test_writes_insights(self, dataset, codebook_json, tmp_path):
        out = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(out),
                "--analyses",
                "univariate",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (out / "all_insights.json").exists()

    def test_insights_are_valid_json(self, dataset, codebook_json, tmp_path):
        out = tmp_path / "results"
        runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(out),
                "--analyses",
                "univariate,bivariate",
            ],
        )

        insights = json.loads((out / "all_insights.json").read_text())

        assert isinstance(insights, list)
        assert insights

    def test_missing_data_file_fails(self, codebook_json, tmp_path):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path / "nope.csv"),
                str(codebook_json),
                "--output-dir",
                str(tmp_path / "o"),
            ],
        )

        assert result.exit_code != 0


class TestGenerateQa:
    def test_writes_jsonl(self, dataset, codebook_json, tmp_path):
        results_dir = tmp_path / "results"
        runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(results_dir),
                "--analyses",
                "univariate",
            ],
        )
        out = tmp_path / "qa.jsonl"

        result = runner.invoke(
            app,
            [
                "generate-qa",
                str(results_dir / "all_insights.json"),
                "--output",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.stdout
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert lines
        for line in lines:
            json.loads(line)


class TestPipeline:
    """`pipeline` used to be a stub that printed step headers and exited 0.

    The test that covered it asserted `exit_code == 0` and `out.exists()` —
    both of which the stub satisfied, since it never failed and `mkdir` was its
    one real statement. These assert the artifacts instead.
    """

    def test_runs_end_to_end(self, dataset, codebook_csv, tmp_path):
        out = tmp_path / "pipeline_out"

        result = runner.invoke(
            app,
            [
                "pipeline",
                str(dataset),
                str(codebook_csv),
                "--output-dir",
                str(out),
                "--no-plots",
            ],
        )

        assert result.exit_code == 0, result.stdout

        codebook = json.loads((out / "codebook.json").read_text())
        insights = json.loads((out / "all_insights.json").read_text())
        qa_lines = (out / "qa_pairs.jsonl").read_text().splitlines()

        assert codebook["variables"], "the parsed codebook has no variables"
        assert insights, "the pipeline produced no insights"
        assert qa_lines, "the pipeline produced no Q/A pairs"
        for line in qa_lines:
            pair = json.loads(line)
            assert pair["question"]
            assert pair["answer"]

    def test_a_missing_data_file_fails(self, codebook_csv, tmp_path):
        """The stub reported success against files that did not exist."""
        result = runner.invoke(
            app,
            [
                "pipeline",
                str(tmp_path / "nope.csv"),
                str(codebook_csv),
                "--output-dir",
                str(tmp_path / "o"),
            ],
        )

        assert result.exit_code != 0

    def test_an_unreadable_codebook_fails(self, dataset, tmp_path):
        result = runner.invoke(
            app,
            [
                "pipeline",
                str(dataset),
                str(tmp_path / "nope.csv"),
                "--output-dir",
                str(tmp_path / "o"),
            ],
        )

        assert result.exit_code != 0

    def test_no_qa_skips_the_third_step(self, dataset, codebook_csv, tmp_path):
        out = tmp_path / "pipeline_out"

        result = runner.invoke(
            app,
            [
                "pipeline",
                str(dataset),
                str(codebook_csv),
                "--output-dir",
                str(out),
                "--no-qa",
                "--no-plots",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (out / "all_insights.json").exists()
        assert not (out / "qa_pairs.jsonl").exists()

    def test_plots_are_written_when_asked(self, dataset, codebook_csv, tmp_path):
        out = tmp_path / "pipeline_out"

        runner.invoke(
            app,
            [
                "pipeline",
                str(dataset),
                str(codebook_csv),
                "--output-dir",
                str(out),
                "--no-qa",
            ],
        )

        assert list((out / "plots").glob("*.png"))


class TestAnalysisSelection:
    """`--analyses` accepted names it then ignored.

    `all` expanded to include temporal and causal, neither of which had a
    branch, so asking for them printed nothing and exited 0 — indistinguishable
    from an analysis that found nothing to say.
    """

    def test_causal_is_rejected_with_a_reason(self, dataset, codebook_json, tmp_path):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(tmp_path / "o"),
                "--analyses",
                "causal",
            ],
        )

        assert result.exit_code != 0
        assert "treatment" in result.stdout

    def test_an_unknown_analysis_is_rejected(self, dataset, codebook_json, tmp_path):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(tmp_path / "o"),
                "--analyses",
                "univarite",
            ],
        )

        assert result.exit_code != 0

    def test_temporal_trends_a_datetime_variable(self, tmp_path):
        """Wired, not merely accepted: the fit must recover a planted slope."""
        rng = np.random.default_rng(3)
        n = 120
        frame = pd.DataFrame(
            {
                "visit_date": pd.date_range("2020-01-01", periods=n, freq="7D"),
                "score": 50 + np.arange(n) * 0.4 + rng.normal(0, 3, n),
            }
        )
        data = tmp_path / "data.csv"
        frame.to_csv(data, index=False)
        codebook = tmp_path / "codebook.json"
        codebook.write_text(
            json.dumps(
                {
                    "name": "trend",
                    "variables": {
                        "visit_date": {
                            "name": "visit_date",
                            "label": "Visit date",
                            "var_type": "datetime",
                        },
                        "score": {
                            "name": "score",
                            "label": "Score",
                            "var_type": "numeric_continuous",
                        },
                    },
                }
            )
        )
        out = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "analyze",
                str(data),
                str(codebook),
                "--output-dir",
                str(out),
                "--analyses",
                "temporal",
                "--no-plots",
            ],
        )

        assert result.exit_code == 0, result.stdout
        temporal = json.loads((out / "temporal.json").read_text())

        assert len(temporal) == 1
        trend = temporal[0]
        assert trend["mann_kendall"]["trend"] == "increasing"
        assert trend["linear_trend"]["slope"] == pytest.approx(0.4, abs=0.05)

    def test_a_dataset_with_no_time_variable_says_so(
        self, dataset, codebook_json, tmp_path
    ):
        out = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                str(codebook_json),
                "--output-dir",
                str(out),
                "--analyses",
                "temporal",
                "--no-plots",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert json.loads((out / "temporal.json").read_text()) == []


class TestWithoutPyreadstat:
    """A plain `pip install statqa` has no pyreadstat.

    statistical.py keeps importing without it -- that is what makes the
    dependency optional -- so the class is always importable and only its
    constructor raises. Auto-detection therefore has to consult availability
    rather than assume an importable class is a usable one.
    """

    @pytest.fixture
    def without_pyreadstat(self, monkeypatch):
        class Unusable:
            def __init__(self, **kwargs):
                raise ImportError(
                    "pyreadstat is required for statistical format parsing."
                )

        monkeypatch.setattr("statqa.cli.main.StatisticalFormatParser", Unusable)
        monkeypatch.setattr("statqa.cli.main.HAS_STATISTICAL_PARSER", False)

    def test_auto_detection_still_parses_a_csv(
        self, without_pyreadstat, codebook_csv, tmp_path
    ):
        out = tmp_path / "parsed.json"

        result = runner.invoke(
            app, ["parse-codebook", str(codebook_csv), "--output", str(out)]
        )

        assert result.exit_code == 0, result.stdout
        assert set(json.loads(out.read_text())["variables"]) == {
            "age",
            "gender",
            "income",
        }

    def test_asking_for_the_statistical_format_reports_it_is_unavailable(
        self, without_pyreadstat, codebook_csv, tmp_path
    ):
        result = runner.invoke(
            app,
            [
                "parse-codebook",
                str(codebook_csv),
                "--format",
                "statistical",
                "--output",
                str(tmp_path / "x.json"),
            ],
        )

        assert result.exit_code == 1
        # rich reads [...] as markup, so an unescaped extra silently vanishes
        # and the message would read "pip install statqa".
        assert "statistical-formats" in " ".join(result.stdout.split())
