# Tasks: Publish Packages to PyPI and crates.io

**Input**: Design documents from `/specs/001-publish-packages/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Not explicitly requested - no test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

This feature modifies configuration files at repository root:
- `LICENSE` - New file
- `Cargo.toml`, `rscm-core/Cargo.toml`, `rscm-components/Cargo.toml` - Rust metadata
- `pyproject.toml` - Python metadata
- `.github/workflows/release.yaml` - CI/CD workflow

---

## Phase 1: Setup (License and Shared Metadata)

**Purpose**: Create license file and establish common metadata patterns

- [x] T001 Create Apache-2.0 license file at LICENSE
- [x] T002 [P] Verify existing Cargo.toml metadata fields (description, readme, keywords, categories) in all crates

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Package metadata that MUST be complete before ANY publishing can work

**CRITICAL**: No publishing will succeed until all Cargo.toml and pyproject.toml metadata is complete

- [x] T003 [P] Add repository, license, and authors fields to Cargo.toml (root crate)
- [x] T004 [P] Add repository, license, and authors fields to rscm-core/Cargo.toml
- [x] T005 [P] Add repository, license, and authors fields to rscm-components/Cargo.toml
- [x] T006 Add authors, license, and project.urls to pyproject.toml
- [x] T007 Validate metadata with `cargo publish --dry-run -p rscm-core` and `cargo publish --dry-run -p rscm-components`

**Checkpoint**: All package metadata complete - workflow implementation can now begin

---

## Phase 3: User Story 1 - Rust Developer Installs Core Crate (Priority: P1) MVP

**Goal**: Enable `rscm-core` to be published to crates.io with complete metadata visible on the crate page.

**Independent Test**: Run `cargo publish --dry-run -p rscm-core` locally; after first release, verify crate page at https://crates.io/crates/rscm-core shows all metadata.

### Implementation for User Story 1

- [x] T008 [US1] Create build-wheels job with matrix strategy for 5 targets in .github/workflows/release.yaml
- [x] T009 [US1] Add maturin-action build step with target and manylinux parameters in .github/workflows/release.yaml
- [x] T010 [US1] Add artifact upload step for wheels in .github/workflows/release.yaml
- [x] T011 [US1] Create publish-crates job using katyo/publish-crates@v2 in .github/workflows/release.yaml
- [x] T012 [US1] Add version existence check for crates.io before publishing in .github/workflows/release.yaml
- [x] T013 [US1] Configure publish-delay for workspace dependency propagation in .github/workflows/release.yaml

**Checkpoint**: At this point, rscm-core can be published to crates.io via the workflow

---

## Phase 4: User Story 2 - Python User Installs Package (Priority: P1)

**Goal**: Enable `rscm` Python package to be published to PyPI with cross-platform wheels.

**Independent Test**: Run `uv build` locally; after first release, verify package at https://pypi.org/project/rscm/ shows wheels for all platforms.

### Implementation for User Story 2

- [x] T014 [US2] Create publish-pypi job with environment: release in .github/workflows/release.yaml
- [x] T015 [US2] Add permissions: id-token: write for OIDC authentication in .github/workflows/release.yaml
- [x] T016 [US2] Add artifact download and flatten steps in .github/workflows/release.yaml
- [x] T017 [US2] Add version existence check for PyPI before publishing in .github/workflows/release.yaml
- [x] T018 [US2] Add pypa/gh-action-pypi-publish step in .github/workflows/release.yaml

**Checkpoint**: At this point, rscm can be published to PyPI with cross-platform wheels

---

## Phase 5: User Story 3 - Rust Developer Uses Components Crate (Priority: P2)

**Goal**: Enable `rscm-components` to be published to crates.io after `rscm-core`.

**Independent Test**: After release, verify crate page at https://crates.io/crates/rscm-components shows dependency on rscm-core.

### Implementation for User Story 3

- [x] T019 [US3] Verify rscm-components/Cargo.toml has path dependency that will resolve to published version
- [x] T020 [US3] Confirm katyo/publish-crates handles rscm-core → rscm-components ordering automatically

**Checkpoint**: At this point, both Rust crates can be published in correct dependency order

---

## Phase 6: User Story 4 - Maintainer Publishes New Release (Priority: P2)

**Goal**: Complete end-to-end release workflow that publishes to all registries atomically.

**Independent Test**: Create a test tag, trigger workflow, verify all artifacts appear on crates.io, PyPI, and GitHub Releases.

### Implementation for User Story 4

- [x] T021 [US4] Add job dependencies: publish-pypi needs build-wheels, publish-crates runs independently in .github/workflows/release.yaml
- [x] T022 [US4] Add fail-fast: false to matrix to attempt all builds in .github/workflows/release.yaml
- [x] T023 [US4] Configure workflow to fail if any publish step fails (atomic release) in .github/workflows/release.yaml
- [x] T024 [US4] Update GitHub release creation job to depend on publish jobs in .github/workflows/release.yaml
- [x] T025 [US4] Add clear step names and echo statements for status visibility in .github/workflows/release.yaml

**Checkpoint**: Complete release workflow operational

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and validation

- [x] T026 [P] Document PyPI Trusted Publisher setup steps in specs/001-publish-packages/quickstart.md
- [x] T027 [P] Document CARGO_REGISTRY_TOKEN secret setup in specs/001-publish-packages/quickstart.md
- [x] T028 [P] Document GitHub release environment creation in specs/001-publish-packages/quickstart.md
- [x] T029 Run full workflow validation with `cargo publish --dry-run` for both crates
- [x] T030 Update CLAUDE.md if any new conventions established

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 and US2 are both P1 and can proceed in parallel
  - US3 depends on US1 (rscm-components needs rscm-core)
  - US4 depends on US1, US2, US3 (combines all publishing)
- **Polish (Phase 7)**: Can start after Phase 2, complete after all stories done

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Depends on US1 (rscm-components depends on rscm-core being publishable)
- **User Story 4 (P2)**: Depends on US1, US2, US3 (integrates all publishing into single workflow)

### Parallel Opportunities

- T003, T004, T005 can run in parallel (different Cargo.toml files)
- T026, T027, T028 can run in parallel (different sections of quickstart.md)
- US1 (T008-T013) and US2 (T014-T018) can proceed in parallel after Phase 2
- All build-wheels matrix jobs run in parallel during workflow execution

---

## Parallel Example: Phase 2 (Foundational)

```bash
# Launch all Cargo.toml metadata updates together:
Task: "Add repository, license, and authors fields to Cargo.toml (root crate)"
Task: "Add repository, license, and authors fields to rscm-core/Cargo.toml"
Task: "Add repository, license, and authors fields to rscm-components/Cargo.toml"
```

---

## Parallel Example: User Story 1 + User Story 2

```bash
# After Phase 2 complete, launch US1 and US2 in parallel:

# US1 tasks (crates.io):
Task: "Create build-wheels job with matrix strategy in .github/workflows/release.yaml"
Task: "Create publish-crates job using katyo/publish-crates@v2 in .github/workflows/release.yaml"

# US2 tasks (PyPI):
Task: "Create publish-pypi job with environment: release in .github/workflows/release.yaml"
Task: "Add pypa/gh-action-pypi-publish step in .github/workflows/release.yaml"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup (LICENSE file)
2. Complete Phase 2: Foundational (all metadata)
3. Complete Phase 3: User Story 1 (crates.io publishing)
4. Complete Phase 4: User Story 2 (PyPI publishing)
5. **STOP and VALIDATE**: Test with `cargo publish --dry-run` and `uv build`
6. Deploy initial release

### Incremental Delivery

1. Setup + Foundational → Metadata ready
2. Add User Story 1 → crates.io ready → Test dry-run
3. Add User Story 2 → PyPI ready → Test dry-run
4. Add User Story 3 → Dependency ordering confirmed
5. Add User Story 4 → Full atomic release workflow
6. Polish → Documentation complete

---

## Notes

- All workflow changes are to a single file: `.github/workflows/release.yaml`
- Recommend implementing workflow incrementally, testing each job before adding next
- Use `act` or GitHub Actions dry-run features to test workflow locally if possible
- First real release should be triggered with a test version (e.g., 0.1.1-alpha.0)
- Commit after each task to enable rollback if issues discovered
