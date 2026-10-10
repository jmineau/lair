# lair development tasks. CI runs these same recipes.
#
# The layout is flat: the package lives in `lair/` (not `src/`).

set positional-arguments

# Show available recipes
list:
    @just --list

# Install the project and the (lean) dev tools into .venv
sync:
    uv sync

# Update uv.lock after changing dependencies
lock:
    uv lock

# Lint and check formatting (no changes)
lint:
    uv run ruff check
    uv run ruff format --check

# Fix lint and format the code
format:
    uv run ruff check --fix
    uv run ruff format

# Type check with pyrefly
type-check:
    uv run pyrefly check

# Require docstrings on the public API (raise --fail-under as docstrings are added)
docstr:
    uv run docstr-coverage lair --skip-magic --skip-init --exclude "lair/_ccg_filter.py" --fail-under 88

# Run the tests in parallel (up to 8 workers; `-n 0` for serial), skipping network and slow ones
test *args:
    uv run pytest -n auto --maxprocesses=8 -m "not network and not slow" "$@"

# Run the tests with coverage (coverage.xml, junit.xml for Codecov)
cov *args:
    uv run pytest -n auto --maxprocesses=8 -m "not network and not slow" --cov --cov-report=term --cov-report=xml --junitxml=junit.xml -o junit_family=legacy "$@"

# Build the HTML docs, running every example, failing on warnings
build-docs:
    rm -rf docs/_build docs/_autosummary
    LAIR_SKIP_CCG_DOWNLOAD=1 uv run sphinx-build -M html docs docs/_build -W --keep-going

# Serve the docs at http://127.0.0.1:PORT, rebuilding on every save (Ctrl-C stops)
docs-serve port="8000":
    LAIR_SKIP_CCG_DOWNLOAD=1 uv run sphinx-autobuild docs docs/_build/html --port "$1" --watch lair --re-ignore '_autosummary/' --re-ignore '_build/' --re-ignore '__pycache__/'

# Everything the Code Quality workflow checks, plus the tests
quality-check: lint type-check docstr test

# Run every pre-commit hook on every file
pre-commit:
    uv run pre-commit run --all-files

# Build the sdist and wheel into dist/ and check them
dist:
    rm -rf dist
    uv build
    uv run twine check --strict dist/*

# Draft CHANGELOG entries from the commits since the last release
changelog:
    @uv run git-cliff --unreleased --strip all

# Print the version setuptools-scm computes from git
version:
    @uv run python -m setuptools_scm

# Print the next release's version (CalVer YYYY.MM.PATCH, MM = 05, 08 or 12)
next-version:
    #!/usr/bin/env bash
    set -euo pipefail
    # The latest vYYYY.MM.PATCH tag on GitHub is the source of truth.
    last=$(git ls-remote --tags --refs origin 'v*' | sed 's|.*refs/tags/||' \
        | { grep -E '^v[0-9]{4}\.[0-9]{2}\.[0-9]+$' || true; } | sort -V | tail -n 1)
    last=${last:-v0.0.0}
    year=$(date +%Y); month=$(date +%-m)
    if [ "$month" -le 5 ]; then rel=05; elif [ "$month" -le 8 ]; then rel=08; else rel=12; fi
    IFS=. read -r ly lm lp <<< "${last#v}"
    if [ "$ly" = "$year" ] && [ "$((10#$lm))" = "$((10#$rel))" ]; then
        echo "$year.$rel.$((lp + 1))"
    else
        echo "$year.$rel.0"
    fi

# Tag and push the next release (`just next-version`); CI creates the GitHub Release
release:
    #!/usr/bin/env bash
    set -euo pipefail
    v="$(just next-version)"
    test -z "$(git status --porcelain)" || { echo "Working tree is not clean." >&2; exit 1; }
    test "$(git branch --show-current)" = main || { echo "Release from main." >&2; exit 1; }
    git fetch --quiet --tags origin main
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" || { echo "main is not in sync with origin/main." >&2; exit 1; }
    grep -q "^## \[$v\]" CHANGELOG.md || { echo "CHANGELOG.md has no '## [$v]' section." >&2; exit 1; }
    # PEP 440: the version must be newer than every v* tag, or installers
    # would not see it as the latest release.
    uv run --no-sync python - "$v" <<'PY'
    import subprocess
    import sys

    from packaging.version import InvalidVersion, Version

    new = Version(sys.argv[1])
    tags = subprocess.run(["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True).stdout.split()
    old = []
    for tag in tags:
        try:
            old.append(Version(tag[1:]))
        except InvalidVersion:
            pass
    if old and new <= max(old):
        sys.exit(f"{new} is not newer than the latest release, v{max(old)}.")
    PY
    git tag --annotate "v$v" --message "lair $v"
    git push origin "v$v"
    echo "Pushed v$v; the Publish workflow builds it and creates the GitHub Release."

# Remove build artifacts and caches
clean:
    rm -rf build dist *.egg-info .pytest_cache .ruff_cache .pyrefly_cache
    rm -rf .coverage coverage.xml junit.xml htmlcov docs/_build docs/_autosummary
    find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
