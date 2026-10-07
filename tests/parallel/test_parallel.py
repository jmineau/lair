"""
Tests for lair.parallel.

Only the sequential (num_processes=1) path is exercised here: it is
deterministic and avoids spinning up a multiprocessing Pool in the test suite.
"""

import multiprocessing

import pytest

from lair.parallel import parallelize


def _square(x):
    return x * x


def _fail(x):
    raise ValueError(f"bad item {x}")


def _add(x, *, b=0):
    # Module-level (picklable) worker for the multiprocessing path.
    return x + b


def test_sequential_results():
    run = parallelize(_square, num_processes=1)
    assert run([1, 2, 3, 4]) == [1, 4, 9, 16]


def test_sequential_passes_kwargs():
    def scale(x, *, factor=1):
        return x * factor

    run = parallelize(scale, num_processes=1)
    assert run([1, 2, 3], factor=10) == [10, 20, 30]


def test_empty_iterable_returns_empty_list():
    # Must not try to start a Pool with 0 processes
    run = parallelize(_square, num_processes=4)
    assert run([]) == []


def test_sequential_accepts_generator():
    run = parallelize(_square, num_processes=1)
    assert run(i for i in range(4)) == [0, 1, 4, 9]


def test_returns_a_callable():
    assert callable(parallelize(_square, num_processes=1))


class TestMultiprocessingPath:
    def test_two_processes(self):
        run = parallelize(_square, num_processes=2)
        assert run([1, 2, 3, 4]) == [1, 4, 9, 16]

    def test_max_processes(self):
        run = parallelize(_square, num_processes="max")
        assert run([1, 2, 3, 4]) == [1, 4, 9, 16]

    def test_more_processes_than_cpus_is_clamped(self, caplog):
        # Requesting an absurd count clamps to cpu_count (and logs a
        # warning); results are still correct.
        run = parallelize(_square, num_processes=10_000)
        assert run([1, 2, 3]) == [1, 4, 9]
        assert any(
            r.levelname == "WARNING" and r.name == "lair.parallel"
            for r in caplog.records
        )

    def test_kwargs_passed_through_pool(self):
        run = parallelize(_add, num_processes=2)
        assert run([1, 2, 3], b=100) == [101, 102, 103]

    def test_accepts_generator(self):
        run = parallelize(_square, num_processes=2)
        assert run(i for i in range(4)) == [0, 1, 4, 9]

    def test_worker_error_does_not_leak_processes(self):
        run = parallelize(_fail, num_processes=2)
        before = len(multiprocessing.active_children())
        for _ in range(2):
            with pytest.raises(ValueError, match="bad item"):
                run([1, 2])
        assert len(multiprocessing.active_children()) == before
