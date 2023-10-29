SHELL := /bin/bash

install:
	pdm install -dG:all

bump:
	bumpver update --patch

docs:
	mkdocs serve --livereload

test:
	pytest -n auto

.PHONY: docs