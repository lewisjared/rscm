<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0 (initial ratification)
Modified principles: N/A (initial creation)
Added sections:
  - Core Principles (4 principles)
  - Quality Gates section
  - Development Workflow section
  - Governance section
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ Compatible (success criteria align with performance principle)
  - .specify/templates/tasks-template.md: ✅ Compatible (test-first workflow matches testing principle)
Follow-up TODOs: None
-->

# RSCM Constitution

## Core Principles

### I. Code Quality First

All code MUST meet these non-negotiable standards:

- **No partial implementations**: Every feature MUST be complete before merge. Stub code, TODO markers in production paths, or "simplified for now" patterns are prohibited.
- **No code duplication**: Before writing new functions, developers MUST search existing codebase for reusable utilities. Common-sense naming enables discoverability.
- **No dead code**: Unused functions, imports, or modules MUST be removed immediately. Code exists to be used or deleted.
- **Explicit error handling**: Generic `catch(err)` or `.unwrap()` in non-test code is prohibited. All errors MUST have specific types and proper propagation paths.
- **Consistent naming**: Follow existing codebase patterns. Rust code uses snake_case for functions/variables, PascalCase for types. Python follows PEP 8.
- **Separation of concerns**: Validation logic, data access, and business rules MUST reside in appropriate modules. No mixing concerns within single functions.

**Rationale**: Climate models require scientific reproducibility. Sloppy code leads to incorrect results that undermine research validity.

### II. Testing Standards (NON-NEGOTIABLE)

Every function MUST have corresponding tests:

- **Test coverage requirement**: All public functions MUST have at least one test. Complex functions MUST have edge case coverage.
- **No cheater tests**: Tests MUST reflect real usage patterns and be designed to reveal flaws. Tests that always pass or test trivial conditions are prohibited.
- **Verbose test output**: Tests MUST provide sufficient diagnostic output for debugging. Silent pass/fail is insufficient.
- **Test location**: Rust unit tests reside alongside implementation (`#[cfg(test)]` modules). Integration tests go in `tests/` directory. Python tests in `tests/` with pytest.
- **Red-Green-Refactor**: When adding features, write failing tests first, implement until tests pass, then refactor.

**Rationale**: Climate models must produce verifiable, reproducible results. Comprehensive testing ensures model correctness and prevents regressions.

### III. User Experience Consistency

All interfaces (CLI, Python API, Rust API) MUST provide consistent, predictable behaviour:

- **API consistency**: Similar operations MUST have similar signatures. Parameter ordering, return types, and error handling MUST be uniform across the codebase.
- **Documentation parity**: Every public API in Rust MUST have rustdoc with examples. Every Python binding MUST have numpy-style docstrings.
- **Error messages**: All user-facing errors MUST be actionable. Messages MUST explain what went wrong and suggest resolution paths.
- **British English**: All user-facing text, documentation, and comments MUST use British English spelling (colour, behaviour, organisation).

**Rationale**: RSCM targets both Rust developers and Python scientists. Consistent interfaces reduce cognitive load and enable cross-language workflows.

### IV. Performance Requirements

As a climate modelling framework, performance is a core feature:

- **No unnecessary allocations**: Hot paths MUST avoid heap allocations where stack allocation suffices. Use `&str` over `String`, slices over `Vec` where appropriate.
- **Benchmark before optimising**: Performance changes MUST be justified by profiling data, not assumptions.
- **Python binding efficiency**: PyO3 bindings MUST minimise data copying between Rust and Python. Use numpy arrays and memory views where possible.
- **Complexity documentation**: O(n) or worse algorithms MUST document their complexity in rustdoc/docstrings.

**Rationale**: Climate models often run thousands of simulations. Performance directly impacts research throughput and energy consumption.

## Quality Gates

All code changes MUST pass these gates before merge:

| Gate          | Requirement                                | Verification |
| ------------- | ------------------------------------------ | ------------ |
| Build         | `cargo build --workspace` succeeds         | CI           |
| Rust Tests    | `cargo test --workspace` passes            | CI           |
| Python Tests  | `uv run pytest` passes                     | CI           |
| Linting       | `cargo clippy --tests` reports no warnings | CI           |
| Formatting    | `cargo fmt --check` and `ruff check` pass  | CI           |
| Documentation | No broken rustdoc links                    | CI           |

## Development Workflow

### Code Review Requirements

- All changes require review before merge to main
- Reviewers MUST verify compliance with all four core principles
- Complex changes (>200 lines) SHOULD include architecture discussion

### Commit Standards

- Use conventional commits format: `type(scope): description`
- Changelog fragments in `changelog/` directory via towncrier
- Atomic commits: one logical change per commit

### Dependency Management

- Rust dependencies: Cargo.toml with version constraints
- Python dependencies: managed via uv and pyproject.toml
- Security: Dependencies MUST be reviewed for known vulnerabilities before adoption

## Governance

This constitution supersedes all other development practices for the RSCM project. Compliance is mandatory, not advisory.

### Amendment Procedure

1. Propose amendment via pull request modifying this file
2. Amendment MUST include rationale and impact assessment
3. Dependent templates MUST be updated in the same PR
4. Version bump follows semantic versioning:
   - MAJOR: Principle removal or incompatible redefinition
   - MINOR: New principle or material expansion
   - PATCH: Clarification or wording refinement

### Compliance Review

- All PRs MUST reference applicable principles in description
- Reviewers MUST verify principle compliance before approval
- Violations MUST be fixed, not waived

### Guidance Files

- `CLAUDE.md` provides runtime development guidance and build commands
- This constitution provides non-negotiable principles
- When conflicts arise, this constitution takes precedence

**Version**: 1.0.0 | **Ratified**: 2026-01-14 | **Last Amended**: 2026-01-14
