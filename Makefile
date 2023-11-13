SHELL := /bin/bash

install:
	pdm install -dG:all

bump:
	bumpver update --patch

docs:
	mkdocs serve --livereload -w srai

test:
	pytest -n auto

.PHONY: install bump docs test
