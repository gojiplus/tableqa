"""Checks that packaging metadata, CI config and the install docs agree.

These are the mismatches that do not fail until a fresh environment builds
them: a CI matrix below requires-python, a documented extra that no longer
exists, a dependency floor older than the API the code calls.
"""

import re
import tomllib
from pathlib import Path

import pytest

# These assert on repository configuration rather than on installed behaviour.
# The wheel smoke test installs only the wheel and pytest, so PyYAML is absent
# there and the whole module is out of scope; skip rather than fail collection.
yaml = pytest.importorskip("yaml", reason="PyYAML is in the dev dependency group")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def ci_workflow():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())


def _floor(spec: str) -> tuple[int, ...]:
    """Return the lower bound of a requirement specifier as a version tuple."""
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", spec)
    assert match, f"no lower bound in {spec!r}"
    return tuple(int(p) for p in match.group(1).split("."))


class TestPythonVersions:
    def test_ci_matrix_is_declared(self, ci_workflow):
        # The reusable workflow defaults to ["3.11", "3.14"]; relying on that
        # default runs a version this package does not support.
        assert "python-versions" in ci_workflow["jobs"]["ci"]["with"]

    def test_every_ci_version_satisfies_requires_python(self, pyproject, ci_workflow):
        import json

        floor = _floor(pyproject["project"]["requires-python"])
        versions = json.loads(ci_workflow["jobs"]["ci"]["with"]["python-versions"])

        for version in versions:
            assert tuple(int(p) for p in version.split(".")) >= floor, (
                f"CI tests Python {version}, below requires-python "
                f"{pyproject['project']['requires-python']}"
            )

    def test_classifiers_match_the_ci_matrix(self, pyproject, ci_workflow):
        import json

        classified = {
            c.rsplit(" :: ", 1)[-1]
            for c in pyproject["project"]["classifiers"]
            if c.startswith("Programming Language :: Python :: ") and "." in c
        }
        tested = set(json.loads(ci_workflow["jobs"]["ci"]["with"]["python-versions"]))

        assert classified == tested


#: Documents that tell a reader how to install the package.
INSTALL_DOCS = [
    "README.md",
    "CLAUDE.md",
    "docs/_shared/installation.rst",
    "examples/README.md",
]


class TestDocumentedExtras:
    """Every extra named in the docs must actually exist."""

    @pytest.mark.parametrize("relative", INSTALL_DOCS)
    def test_no_document_advertises_a_missing_extra(self, pyproject, relative):
        declared = set(pyproject["project"].get("optional-dependencies", {}))
        text = (ROOT / relative).read_text()

        referenced = set(re.findall(r"statqa\[([a-z,\-]+)\]", text))
        referenced |= {name for name in re.findall(r'\.\[([a-z,\-]+)\]"', text) if name}

        for group in referenced:
            for extra in group.split(","):
                assert extra in declared, (
                    f"{relative} installs statqa[{extra}], which pyproject.toml "
                    f"does not declare (has: {sorted(declared)})"
                )


class TestDependencyFloors:
    def test_seaborn_floor_supports_the_legend_argument(self, pyproject):
        # `legend` reached barplot/boxplot in seaborn 0.13; on 0.12 it is passed
        # through to matplotlib and raises AttributeError.
        spec = next(
            d for d in pyproject["project"]["dependencies"] if d.startswith("seaborn")
        )

        assert _floor(spec) >= (0, 13)

    def test_installed_seaborn_satisfies_the_declared_floor(self, pyproject):
        import seaborn

        spec = next(
            d for d in pyproject["project"]["dependencies"] if d.startswith("seaborn")
        )
        installed = tuple(int(p) for p in seaborn.__version__.split(".")[:2])

        assert installed >= _floor(spec)[:2]
