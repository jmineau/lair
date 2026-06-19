"""Tests for lair.parallel.

Only the sequential (num_processes=1) path is exercised here: it is
deterministic and avoids spinning up a multiprocessing Pool in the test suite.
"""

from lair.parallel import parallelize


def _square(x):
    return x * x


def test_sequential_results():
    run = parallelize(_square, num_processes=1)
    assert run([1, 2, 3, 4]) == [1, 4, 9, 16]


def test_sequential_passes_kwargs():
    def scale(x, *, factor=1):
        return x * factor

    run = parallelize(scale, num_processes=1)
    assert run([1, 2, 3], factor=10) == [10, 20, 30]


def test_returns_a_callable():
    assert callable(parallelize(_square, num_processes=1))


# TODO: cover the multiprocessing path (num_processes > 1, 'max') and the
# CPU-count clamping warnings. Those need a module-level worker function so the
# target is picklable. NOTE: an empty iterable currently clamps `processes` to 0
# and raises ValueError from Pool(processes=0) — likely a bug worth a guard.

