"""Offline baseline contracts, including malformed fixtures and endpoints."""

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from regression import magicc_baseline as baseline


@pytest.fixture
def fixture_dir(tmp_path):
    for suffix in (".csv", "_config.json"):
        shutil.copy(baseline.DATA_DIR / (baseline.CASE + suffix), tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "damage",
    [
        "csv",
        "config",
        "variable",
        "region",
        "duplicate",
        "identity",
        "unit",
        "nan",
        "inf",
        "negative",
        "missing_year",
        "duplicate_year",
        "shift",
    ],
)
def test_invalid_fixture_fails(fixture_dir, damage):
    path = fixture_dir / (baseline.CASE + ".csv")
    if damage in {"csv", "config"}:
        (
            path if damage == "csv" else fixture_dir / (baseline.CASE + "_config.json")
        ).unlink()
    else:
        with path.open() as handle:
            rows = list(csv.reader(handle))
        if damage == "variable":
            rows.pop(1)
        elif damage == "duplicate":
            rows.append(rows[1])
        elif damage in {"region", "identity", "unit"}:
            field = {"region": "region", "identity": "scenario", "unit": "unit"}[damage]
            rows[1][rows[0].index(field)] = "invalid"
        elif damage in {"nan", "inf", "negative"}:
            rows[1][8] = {"nan": "nan", "inf": "inf", "negative": "-1"}[damage]
        elif damage == "missing_year":
            rows = [row[:-1] for row in rows]
        elif damage == "duplicate_year":
            rows[0][-1] = rows[0][-2]
        elif damage == "shift":
            rows[0][-1] = "2101-01-01 00:00:00"
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        baseline.load_case(fixture_dir)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_integer_and_float_comparisons(dtype):
    signed, relative = baseline.compare_values(
        np.array([2, 1], dtype=dtype), np.array([0, 2], dtype=dtype)
    )
    np.testing.assert_array_equal(signed, [2.0, -1.0])
    assert np.isnan(relative[0])
    assert relative[1] == -0.5


def test_zero_and_small_reference_errors():
    signed, relative = baseline.compare_values(
        np.array([2.0, 2e-30]), np.array([0.0, 1e-30])
    )
    np.testing.assert_allclose(signed, [2.0, 1e-30], atol=0)
    assert np.isnan(relative[0])
    assert relative[1] == 1


def test_real_report_and_residual_wiring(monkeypatch):
    original = baseline.run_magicc

    def capture(years, concentrations, **kwargs):
        _, reference, _, _ = baseline.load_case(baseline.DATA_DIR)
        expected = reference[baseline.ERF] - sum(
            reference[f"{baseline.ERF}|{g}"] for g in baseline.GASES
        )
        np.testing.assert_array_equal(kwargs["other_forcing"], expected)
        assert set(concentrations) == set(baseline.GASES)
        assert (
            kwargs["forcing_parameters"]["delq2xco2"]
            == kwargs["climate_parameters"]["rf_2xco2"]
        )
        return original(years, concentrations, **kwargs)

    monkeypatch.setattr(baseline, "run_magicc", capture)
    table, metadata, summary = baseline.build_report()
    assert len(table) == 1755
    assert set(table.variable) == set(baseline.OBSERVABLES)
    assert table.groupby("variable").size().eq(351).all()
    assert table.comparison_year.min() == 1750
    assert table.comparison_year.max() == 2100
    assert metadata["scientific_parity"] == "not_evaluated"
    assert metadata["effective_parameters"]["ClimateUDEB"]["n_layers"] == 50
    assert metadata["reference_provenance_unknown"]
    assert "Surface Temperature" in summary


def test_endpoints_remain_in_summary(monkeypatch):
    original = baseline.run_magicc

    def perturb(*args, **kwargs):
        result = original(*args, **kwargs)
        result.values["Surface Temperature"][[0, -1]] = [1000.0, 2000.0]
        return result

    monkeypatch.setattr(baseline, "run_magicc", perturb)
    table, _, summary = baseline.build_report()
    temperature = table[table.variable == "Surface Temperature"]
    assert temperature.iloc[0].absolute_error >= 999
    assert temperature.iloc[-1].absolute_error >= 1900
    assert "| 2100 |" in summary
    assert "1000" in summary


def test_deterministic_bundle_and_collision(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    baseline.write_report(a)
    baseline.write_report(b)
    for filename in ("comparison.csv", "metadata.json", "summary.md"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()
    assert len(pd.read_csv(a / "comparison.csv")) == 1755
    with pytest.raises(FileExistsError):
        baseline.write_report(a)


def test_failed_serialization_leaves_no_bundle(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        msg = "disk failure"
        raise OSError(msg)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk failure"):
        baseline.write_report(tmp_path / "output")
    assert not list(tmp_path.iterdir())


def test_dangling_output_symlink_rejected_before_run(tmp_path, monkeypatch):
    output = tmp_path / "output"
    target = tmp_path / "missing"
    output.symlink_to(target)

    def unexpected_run(*args, **kwargs):
        pytest.fail("Existing output must be rejected before running the model")

    monkeypatch.setattr(baseline, "build_report", unexpected_run)
    with pytest.raises(FileExistsError):
        baseline.write_report(output)
    assert output.is_symlink()
    assert output.readlink() == target
    assert not target.exists()


def test_cli_offline_from_another_directory(tmp_path):
    command = [
        sys.executable,
        str(baseline.ROOT / "scripts/regression/compare_magicc_baseline.py"),
        "--output-dir",
        str(tmp_path / "report"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "MAGICC_ROOT"}
    result = subprocess.run(  # noqa: S603
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads((tmp_path / "report/metadata.json").read_text())["execution_status"]
        == "complete"
    )
    failed = subprocess.run(  # noqa: S603
        [
            *command[:-1],
            str(tmp_path / "failed"),
            "--data-dir",
            str(tmp_path / "missing"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("core_climatesensitivity", "3"),
        ("core_climatesensitivity", True),
        ("core_delq2xco2", float("nan")),
        ("core_co2ch4n2o_rfmethod", "OLBL"),
    ],
)
def test_bad_parameters_fail(fixture_dir, key, value):
    path = fixture_dir / (baseline.CASE + "_config.json")
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        baseline.build_report(fixture_dir)


def test_failed_model_leaves_no_bundle(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        msg = "component failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(baseline, "run_magicc", fail)
    with pytest.raises(RuntimeError, match="component failed"):
        baseline.write_report(tmp_path / "output")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ["core_delq2xco2", "core_climatesensitivity"])
def test_missing_mapped_parameter_fails(fixture_dir, key):
    path = fixture_dir / (baseline.CASE + "_config.json")
    config = json.loads(path.read_text())
    del config[key]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        baseline.build_report(fixture_dir)
