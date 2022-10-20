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
# Optional if you want to create venv in a specific version. More info: https://pdm.fming.dev/latest/usage/venv/#create-a-virtualenv-yourself
pdm venv create 3.8 # or any higher version of Python

pdm install -G:all
```

3. Activate pdm venv

```sh
eval $(pdm venv activate)

# or

source ./venv/bin/activate
```

4. Activate [pre-commit](https://pre-commit.com/) hooks

```sh
pre-commit install && pre-commit install -t commit-msg
```

### Testing

For testing, [tox](https://tox.wiki/en/latest/) is used to allow testing on multiple Python versions.

To test code locally before committing, run

```sh
tox -e python3.8 # put your python version here
```

<!-- ### Pre-commit hooks
 This repository uses [pre-commit](https://pre-commit.com/) for managing pre-commit hooks.
 They are configured in .pre-commit-config.yaml.
 To install them use `pre-commit install && pre-commit install -t commit-msg` after initial setup with `pdm`.

### Documentation
 This repository uses [MkDocs](https://www.mkdocs.org) as a documentation generator. To use it locally, run  `pdm install -G docs` to download all required packages.

 Docstrings should be written following the [google convention](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e). To ease development one can use [autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) to generate the docstrings. -->

### Fixing bugs

### Code conventions

### Pre-Commit Hooks

### Code formatting

### Code linting

### Python conventions

All Python code must be written **compatible with Python 3.8+**.
<!-- More detailed code conventions can be found in the developer docs. -->

<!-- ## Adding tests -->

## Deployment
### Releasing a new version
To release a new version:
```sh
bumpver update --patch
```
This command will update the version strings across the project, commit and tag the commit with the new version. All you need to do is to push the changes.
