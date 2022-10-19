<!-- Content based on https://github.com/explosion/spaCy/blob/master/CONTRIBUTING.md -->

<!-- NICE TO HAVE: https://github.com/explosion/spaCy/blob/master/extra/DEVELOPER_DOCS/Code%20Conventions.md -->

# Contributing to srai

## Contributing to the code base

### What belongs in srai?

### Getting started

To make changes to srai's code base, you need to fork and then clone the GitHub repository.

For first setup of the project locally, the following commands have to be executed.

0. Make sure you have installed at least version **3.8+** of Python.

1. Install [PDM](https://pdm.fming.dev/latest) (only if not already installed)

```sh
pip install pdm
```

2. Install package locally (will download all dev packages and create a local venv)

```sh
pdm install
```

3. Activate pdm venv

```sh
pdm venv activate

# or

source ./venv/bin/activate
```

4. Activate [pre-commit](https://pre-commit.com/) hooks

```sh
pre-commit install && pre-commit install -t commit-msg
```

### Fixing bugs

### Code conventions

### Pre-Commit Hooks

### Code formatting

### Code linting

### Python conventions

All Python code must be written **compatible with Python 3.8+**.
<!-- More detailed code conventions can be found in the developer docs. -->

<!-- ## Adding tests -->
