SHELL := /bin/bash

install:
	pdm install -dG:all

docs:
	mkdocs serve --livereload -w srai

test:
	pytest --durations=20 --doctest-modules --doctest-continue-on-failure srai
	pytest --durations=20 tests

.PHONY: install docs test
