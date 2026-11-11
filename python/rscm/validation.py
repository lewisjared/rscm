"""
Checks on model inputs and configuration before a run.

Front ends such as `rscm.fair` and `rscm.magicc` collect every problem with a
[Validator][rscm.validation.Validator] and raise them together as one
[InputValidationError][rscm.validation.InputValidationError],
so a bad input fails with a message naming it rather than as NaN in the results.
Scripts can exit with the error's `exit_code`:

```python
try:
    model = setup_model(config, inputs, time_axis)
except InputValidationError as error:
    print(error, file=sys.stderr)
    sys.exit(error.exit_code)
```
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ExitCode",
    "InputValidationError",
    "Issue",
    "IssueKind",
    "Validator",
]


class ExitCode(IntEnum):
    """Process exit codes for a failed validation, following BSD `sysexits.h`."""

    OK = 0
    DATA = 65
    """The input data is invalid (`EX_DATAERR`)."""
    UNSUPPORTED = 69
    """The configuration asks for an option RSCM lacks (`EX_UNAVAILABLE`)."""
    CONFIG = 78
    """The configuration is invalid (`EX_CONFIG`)."""


class IssueKind(Enum):
    """What is wrong, and the exit code it maps to."""

    NON_FINITE = ("non-finite", ExitCode.DATA)
    """An input holds NaN or infinity."""
    OUT_OF_RANGE = ("out-of-range", ExitCode.DATA)
    """A value is outside its physical or numerical domain."""
    SHAPE = ("shape", ExitCode.DATA)
    """An input has the wrong length or does not cover the time axis."""
    MISSING = ("missing", ExitCode.CONFIG)
    """A required input, parameter or species is absent."""
    UNKNOWN_KEY = ("unknown-key", ExitCode.CONFIG)
    """A configuration key is not recognised, such as a misspelling."""
    UNSUPPORTED = ("unsupported", ExitCode.UNSUPPORTED)
    """A recognised option that RSCM does not implement."""

    def __init__(self, label: str, exit_code: ExitCode) -> None:
        self.label = label
        self.exit_code = exit_code


@dataclass(frozen=True)
class Issue:
    """One problem found in the inputs or configuration."""

    kind: IssueKind
    location: str
    """What the problem is in, such as a variable, parameter or configuration key."""
    message: str

    def __str__(self) -> str:
        return f"[{self.kind.label}] {self.location}: {self.message}"


class InputValidationError(ValueError):
    """Raised before a run when its inputs or configuration are invalid."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            msg = "InputValidationError needs at least one issue"
            raise ValueError(msg)
        lines = "\n".join(f"  {issue}" for issue in self.issues)
        super().__init__(f"{len(self.issues)} invalid input(s):\n{lines}")

    @property
    def exit_code(self) -> ExitCode:
        """The first issue's exit code, since earlier checks find root causes."""
        return self.issues[0].kind.exit_code


class Validator:
    """Collects issues so a single error reports all of them."""

    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, kind: IssueKind, location: str, message: str) -> None:
        """Record an issue."""
        self.issues.append(Issue(kind, location, message))

    def check(self, ok: bool, kind: IssueKind, location: str, message: str) -> bool:
        """Record an issue unless `ok`, and return `ok`."""
        if not ok:
            self.add(kind, location, message)
        return ok

    def finite(self, location: str, values: ArrayLike) -> bool:
        """Check that every value is finite, naming the first bad index."""
        array = np.asarray(values, dtype=np.float64)
        bad = ~np.isfinite(array)
        if not bad.any():
            return True
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        index = first[0] if len(first) == 1 else first
        self.add(
            IssueKind.NON_FINITE,
            location,
            f"{int(bad.sum())} non-finite value(s), "
            f"first {array[first]} at index {index}",
        )
        return False

    def raise_if_any(self) -> None:
        """Raise an `InputValidationError` holding every issue recorded so far."""
        if self.issues:
            raise InputValidationError(self.issues)
