SHELL := /bin/bash

install:
	pdm install -dG:all

docs:
	mkdocs serve --livereload -w srai

test:
	pytest --durations=20 --doctest-modules --doctest-continue-on-failure srai tests

.PHONY: install docs test
