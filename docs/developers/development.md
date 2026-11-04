# Development

Notes for developers. If you want to get involved, please do!
We welcome all kinds of contributions, for example:

- docs fixes/clarifications
- bug reports
- bug fixes
- feature requests
- pull requests
- tutorials

## Workflows

We don't mind whether you use a branching or forking workflow.
However, please only push to your own branches,
pushing to other people's branches is often a recipe for disaster,
is never required in our experience
so is best avoided.

Try and keep your pull requests as small as possible
(focus on one thing if you can).
This makes life much easier for reviewers
which allows contributions to be accepted at a faster rate.

## Installation

Follow [Development Setup](index.md#development-setup) in the Developer Guide.

## Language

We use British English for our development.
We do this for consistency with the broader work context of our lead developers.

## Versioning

This package follows the version format
described in [PEP440](https://peps.python.org/pep-0440/)
and [Semantic Versioning](https://semver.org/)
to describe how the version should change
depending on the updates to the code base.

Commit messages follow the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) standard
(`feat:`, `fix:`, `chore:`).
This makes it easy to find the commits that matter in the history.

/// admonition | Note
The changelog is not generated from commit messages.
Add a [changelog fragment](https://github.com/lewisjared/rscm/blob/main/changelog/README.md)
in `changelog/` for each user-facing change.
`towncrier` compiles the fragments into [the changelog](../changelog.md) when a release is made.
///

## Releasing

Releasing is semi-automated via GitHub Actions.
The version bump is chosen by hand when the workflow is triggered.

### Standard process

The steps required are the following:

1. Run the [Bump version workflow](https://github.com/lewisjared/rscm/actions/workflows/bump.yaml)
   on `main` and choose a `bump_rule` (`patch`, `minor` or `major`).
   It runs `bump-my-version`, which updates the versions, compiles the changelog with `towncrier`,
   commits and pushes a `v*` tag.

1. The tag triggers the [Release workflow](https://github.com/lewisjared/rscm/actions/workflows/release.yaml).
   It builds the wheels and sdist, publishes the crates to crates.io and the package to PyPI,
   then creates a GitHub release with the artifacts.

1. That's it, release done, make noise on social media of choice, do whatever
   else

1. Enjoy the newly available version
