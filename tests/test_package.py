"""Top-level `rscm` package behaviour."""

import importlib
import importlib.metadata
import subprocess
import sys

import pytest

import rscm


def _run(code: str) -> str:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_import_rscm_does_not_load_submodules():
    loaded = _run("import sys, rscm; print('rscm.core' in sys.modules)")
    assert loaded == "False"


def test_submodule_attribute_access_after_bare_import():
    assert _run("import rscm; print(rscm.core.TimeAxis.__name__)") == "TimeAxis"
    assert _run("import rscm; print(rscm.two_layer.TwoLayerBuilder.__name__)") == (
        "TwoLayerBuilder"
    )


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'not_a_module'"):
        rscm.not_a_module


def test_dir_lists_submodules():
    assert {"core", "magicc", "two_layer"} <= set(dir(rscm))


def test_version_falls_back_to_extension_version(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert importlib.reload(rscm).__version__ == rscm._lib.__version__
    monkeypatch.undo()
    importlib.reload(rscm)
