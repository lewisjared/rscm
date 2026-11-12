import numpy as np
import pytest

from rscm.validation import ExitCode, InputValidationError, IssueKind, Validator


def test_no_issues_does_not_raise():
    v = Validator()
    assert v.finite("x", [1.0, 2.0])
    assert v.check(True, IssueKind.MISSING, "y", "absent")
    v.raise_if_any()


def test_collects_every_issue():
    v = Validator()
    assert not v.finite("Emissions|CO2", [1.0, np.nan, np.inf])
    v.add(IssueKind.UNKNOWN_KEY, "core_climatesensitivty", "not a MAGICC7 key")
    with pytest.raises(InputValidationError) as info:
        v.raise_if_any()
    error = info.value
    assert len(error.issues) == 2
    assert "2 non-finite value(s), first nan at index 1" in str(error)
    assert "[unknown-key] core_climatesensitivty" in str(error)
    assert error.exit_code == ExitCode.DATA


def test_finite_names_2d_index():
    v = Validator()
    v.finite("boxes", np.array([[1.0, 2.0], [3.0, np.nan]]))
    assert "index (1, 1)" in v.issues[0].message


def test_exit_codes():
    assert IssueKind.UNSUPPORTED.exit_code == ExitCode.UNSUPPORTED == 69
    assert IssueKind.MISSING.exit_code == ExitCode.CONFIG == 78


def test_is_value_error():
    with pytest.raises(ValueError, match="at least one issue"):
        InputValidationError([])
    assert issubclass(InputValidationError, ValueError)
