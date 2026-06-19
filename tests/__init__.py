"""Test suite for lair.

Tests are *containerized* by module: every ``lair`` submodule has its own
self-contained directory under ``tests/`` (e.g. ``tests/clock/`` for
``lair.clock``). A module's tests, fixtures (``conftest.py``), and sample data
(``data/``) all live together so that when a module is pulled out of lair into
its own package, its tests come with it — no digging through a shared test file
to separate things out.

See ``tests/README.md`` for the layout convention and the extraction procedure.
"""
