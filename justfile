# Justfile for lair
#
# Layout is flat: the package lives in `lair/` (not `src/`). Lint/type targets
# point at `lair`; tests live in `tests/` (containerized per module).

# Show available commands
list:
    @just --list

# Sync the local uv environment with the (lean) dev dependency group
sync:
	@echo "Syncing development environment with uv..."
	uv sync --group dev

# Refresh the uv lockfile
lock:
	@echo "Refreshing uv.lock..."
	uv lock

# Build the HTML documentation (docs/_build/html)
build-docs:
	@echo "Building HTML documentation..."
	rm -rf docs/_build/ docs/_autosummary/
	LAIR_SKIP_CCG_DOWNLOAD=1 uv run sphinx-build -M html docs docs/_build

# Show the version setuptools-scm derives from git right now (tags + commits since)
version:
	@uv run python -m setuptools_scm

# Tag + push the next CalVer release vYYYY.MM.PATCH (MM = 05/08/12; new month -> .0)
release:
	#!/usr/bin/env bash
	set -euo pipefail
	if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
	    echo "Commit or stash your changes first." >&2; exit 1
	fi
	if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
	    echo "Releases are tagged from main." >&2; exit 1
	fi
	git fetch --tags origin
	if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
	    echo "main differs from origin/main; push or pull first." >&2; exit 1
	fi
	last=$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || echo v0.0.0)
	year=$(date +%Y); month=$(date +%-m)
	if [ "$month" -le 5 ]; then rel=05; elif [ "$month" -le 8 ]; then rel=08; else rel=12; fi
	IFS=. read -r ly lm lp <<< "${last#v}"
	if [ "$ly" = "$year" ] && [ "$((10#$lm))" = "$((10#$rel))" ]; then
	    next="$year.$rel.$((lp + 1))"
	else
	    next="$year.$rel.0"
	fi
	echo "Tagging v$next (previous: $last)"
	git tag -a "v$next" -m "lair $next"
	git push origin "v$next"

# Clean up build artifacts and cache files
clean:
	@echo "Cleaning up generated files..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .pyrefly_cache/
	rm -rf .coverage coverage.xml junit.xml
	rm -rf docs/_build/ docs/_autosummary/
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name '*.py[co]' -delete

# Run pre-commit hooks on all files
pre-commit:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files

# Lint with ruff (the enforced gate)
lint:
	@echo "Linting with ruff..."
	uv run ruff check lair tests

# Apply ruff autofixes and formatting (opt-in; will reformat — review the diff)
format:
	@echo "Applying ruff fixes and formatting..."
	uv run ruff check --fix lair tests
	uv run ruff format lair tests

# Type-check with pyrefly (enforced gate)
type-check:
	@echo "Type checking with pyrefly..."
	uv run pyrefly check

# Run the default test suite (excludes slow tests)
test:
	@echo "Running default tests (excluding slow)..."
	uv run pytest -v -m "not slow"

# Run the hermetic subset: no network, no slow, no CHPC paths (what CI runs)
test-hermetic:
	@echo "Running hermetic tests (no network / slow / chpc)..."
	uv run pytest -v -m "not network and not slow and not chpc"

# Run tests without live network access
test-no-network:
	@echo "Running non-network tests..."
	uv run pytest -v -m "not network"

# Run only tests that require live network access
test-network:
	@echo "Running network tests..."
	uv run pytest -v -m network

# Run only tests that require CHPC filesystem paths / group data
test-chpc:
	@echo "Running CHPC tests..."
	uv run pytest -v -m chpc

# Run the hermetic suite with coverage
cov:
	@echo "Running tests with coverage..."
	uv run pytest -m "not network and not slow and not chpc" --cov=lair --cov-report=term-missing

# Lint + type-check + hermetic tests
quality-check:
	@echo "Running quality checks..."
	just lint
	just type-check
	just test-hermetic
