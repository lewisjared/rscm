## RustRover setup

[RustRover](https://www.jetbrains.com/rust/) is the Pycharm equivalent for Rust
and provides a great IDE for Rust development if you are already familiar with PyCharm.

### Code analysis and formatting

RustRover integrates with `rustfmt` and `clippy` to provide code formatting and linting
capabilities on save.
This provides contextual information in the editor to quickly see invalid code.
For a longer description of any warnings or errors,
you can run `make lint-rust` in the terminal.
You can enable these features by going to `File -> Settings -> Rust`.

For the linter configuration, it is recommended to change to use `clippy`
to get more information about how to write better rust.
Additional, add
`--target-dir=target/analyzer` to the additional arguments field.
This will ensure that the linter runs in a dedicated target directory
and does not interfere with the build process leading to unnecessary recompilation.

![Recommended RustRover Linter settings](images/rr_linter_settings.png)

### Python integration

RustRover also provides Python integration which enables code completion and
highlighting for Python code in the same project.
To enable this add the "Python Community Edition" plugin in the IDE
and then add the local virtual environment (in `.venv`) as the "Python Interpreter".

The `Ruff` plugin which automatically formats and lints Python code on save does not currently support RustRover
([GH issue](https://github.com/koxudaxi/ruff-pycharm-plugin/issues/309)).
A workaround until support is added is to add the
[File Watchers](https://plugins.jetbrains.com/plugin/7177-file-watchers) plugin.
This is optional as `ruff` is run as part of the pre-commit hooks.

To ensure the `.py` files and the `.pyi` files are formatted and linted on save,
a new file watcher can be added with the following settings to run `ruff` via `uv`
on a changed file:

![Recommended RustRover Linter settings](images/rr_ruff_filewatcher.png)

A new "Scope" should be created with the following filter:

```
(file:*.py||file:*.pyi)&&!ext:*
```

This will catch the `.py` and `.pyi` files within the project.
The default "Python" scope will not format the `.pyi` files.
