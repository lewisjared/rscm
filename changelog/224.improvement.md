`import rscm` now gives access to its submodules, so `rscm.core.TimeAxis` works without `import rscm.core`.
Submodules still load only on first use.
`rscm.__version__` falls back to the extension version when the package metadata is missing.
