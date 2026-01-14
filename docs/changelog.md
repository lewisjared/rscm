## rscm 0.2.1 (2026-01-14)

No significant changes.


## rscm 0.2.0 (2026-01-14)

### ⚠️ Breaking Changes ⚠️

- Refactored `InputState` to include references to `TimeSeries` instead of scalar values.
  This is requires a change to the `Component` interface. ([#17](https://github.com/lewisjared/rscm/pulls/17))

### Features

- Add Ocean Surface Partial Pressure (OSPP) component to the `rscm-components` crate. ([#10](https://github.com/lewisjared/rscm/pulls/10))
- Added automated release workflow for publishing packages to crates.io and PyPI with cross-platform wheel builds. ([#47](https://github.com/lewisjared/rscm/pulls/47))

### Bug Fixes

- Fixed CI failure on Python 3.13 by upgrading dependencies (pandas 2.2.2 -> 2.3.3 which includes Python 3.13 wheels). ([#39](https://github.com/lewisjared/rscm/pulls/39))

### Improved Documentation

- Add the basic framework for a `mkdocs`-based documentation site in the `docs/` directory. ([#18](https://github.com/lewisjared/rscm/pulls/18))
- Added MAGICC module documentation ([#46](https://github.com/lewisjared/rscm/pulls/46))

### Trivial/Internal Changes

- [#39](https://github.com/lewisjared/rscm/pulls/39)


## rscm 0.1.0 (2024-09-24)

### Improvements

- Add changelog management to the release process ([#9](https://github.com/lewisjared/rscm/pulls/9))
