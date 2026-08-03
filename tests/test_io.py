"""Tests for data and JSON I/O helpers."""

import json
import zipfile

import pandas as pd
import pytest

from statqa.utils.io import load_data, load_json, save_json


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestLoadData:
    def test_reads_a_csv(self, frame, tmp_path):
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False)

        assert load_data(path).equals(frame)

    def test_reads_a_csv_from_a_zip(self, frame, tmp_path):
        csv_path = tmp_path / "inner.csv"
        frame.to_csv(csv_path, index=False)
        zip_path = tmp_path / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(csv_path, arcname="inner.csv")

        assert load_data(zip_path).equals(frame)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            load_data(tmp_path / "absent.csv")

    def test_unknown_extension_falls_back_to_csv(self, tmp_path):
        # load_data documents a deliberate "try to load as CSV anyway" fallback.
        path = tmp_path / "data.tsvish"
        path.write_text("a,b\n1,2\n")

        assert list(load_data(path).columns) == ["a", "b"]


class TestJson:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"a": 1, "b": [1, 2, 3], "c": "text"}

        save_json(payload, path)

        assert load_json(path) == payload

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.json"

        save_json({"a": 1}, path)

        assert path.exists()

    def test_writes_readable_indentation(self, tmp_path):
        path = tmp_path / "out.json"

        save_json({"a": 1}, path, indent=2)

        assert "\n" in path.read_text()

    def test_non_ascii_survives(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"label": "Åge — τ"}

        save_json(payload, path)

        assert load_json(path) == payload

    def test_output_is_standard_json(self, tmp_path):
        path = tmp_path / "out.json"

        save_json({"a": [1, 2]}, path)

        json.loads(path.read_text())
