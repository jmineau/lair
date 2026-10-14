"""
Tests for lair.parallel.

Most tests run the sequential (num_processes=1) path, which is deterministic and
avoids a multiprocessing Pool; a few start a real Pool, and `TestAllowedCPUs`
replaces the Pool and the CPU counts so it does not depend on the machine.
"""

import logging
import multiprocessing
import os

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
        # Requesting an absurd count clamps to the CPUs this process may use
        # (and logs a warning); results are still correct.
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


class TestAllowedCPUs:
    """The CPUs this process may use cap the processes, not the machine's CPUs."""

    @pytest.fixture
    def pool_sizes(self, monkeypatch):
        """Replace Pool with one that runs in-process and records its size."""
        sizes = []

        class RecordingPool:
            def __init__(self, processes):
                sizes.append(processes)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, func, iterable):
                return [func(i) for i in iterable]

        monkeypatch.setattr(multiprocessing, "Pool", RecordingPool)
        return sizes

    @pytest.fixture
    def limit_to(self, monkeypatch):
        """Return a function that sets the allowed CPUs (and a 104-CPU machine)."""

        def limit(allowed):
            # raising=False: macOS and Windows have no sched_getaffinity
            monkeypatch.setattr(
                os, "sched_getaffinity", lambda pid: set(range(allowed)), raising=False
            )
            monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 104)

        return limit

    def test_max_uses_the_allowed_cpus_not_the_machine(self, pool_sizes, limit_to):
        limit_to(4)
        run = parallelize(_square, num_processes="max")
        assert run(range(500)) == [i * i for i in range(500)]
        assert pool_sizes == [4]

    def test_integer_request_is_capped_at_the_allowed_cpus(
        self, pool_sizes, limit_to, caplog
    ):
        limit_to(4)
        run = parallelize(_square, num_processes=8)
        with caplog.at_level(logging.WARNING, logger="lair.parallel"):
            run(range(500))
        assert pool_sizes == [4]
        assert "8 processes requested" in caplog.text
        assert "only 4 CPU(s)" in caplog.text

    def test_request_within_the_allowed_cpus_is_kept(self, pool_sizes, limit_to):
        limit_to(4)
        parallelize(_square, num_processes=3)(range(500))
        assert pool_sizes == [3]

    def test_fewer_items_than_allowed_cpus(self, pool_sizes, limit_to):
        limit_to(4)
        parallelize(_square, num_processes="max")(range(3))
        assert pool_sizes == [3]

    def test_one_allowed_cpu_runs_sequentially(self, pool_sizes, limit_to):
        limit_to(1)
        run = parallelize(_square, num_processes="max")
        assert run(range(5)) == [0, 1, 4, 9, 16]
        assert pool_sizes == []

    def test_without_sched_getaffinity_it_counts_the_machine(
        self, pool_sizes, monkeypatch
    ):
        # macOS and Windows have no os.sched_getaffinity
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 6)
        parallelize(_square, num_processes="max")(range(500))
        assert pool_sizes == [6]
