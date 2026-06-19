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

# Clean up build artifacts and cache files
clean:
	@echo "Cleaning up generated files..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .pyrefly_cache/
	rm -rf .coverage coverage.xml junit.xml
	rm -rf docs/build/
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

# Type-check with pyrefly (ADVISORY — lenient baseline, not yet a hard gate)
type-check:
	@echo "Type checking with pyrefly (advisory)..."
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

# Lint (gate) + type-check (advisory) + hermetic tests
quality-check:
	@echo "Running quality checks..."
	just lint
	-just type-check
	just test-hermetic
