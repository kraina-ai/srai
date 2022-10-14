# srai
Spatial Representations for Artificial Intelligence

## Install
 * [pyenv](https://github.com/pyenv/pyenv)
 * [PDM](https://github.com/pdm-project/pdm)

 Run `pdm install`

## Development
 To start development install with `pdm install -d`.

### Pre-commit hooks
 This repository uses [pre-commit](https://pre-commit.com/) for managing pre-commit hooks.
 They are configured in .pre-commit-config.yaml.
 To install them use `pre-commit install && pre-commit install -t commit-msg` after initial setup with `pdm`.

### Optional dependencies

#### Documentation
 This repository uses [MkDocs](https://www.mkdocs.org) as a documentation generator. To use it locally, run  `pdm install -G docs` to download all required packages.
