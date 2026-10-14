"""
Parallelization utilities.
"""

import logging
import multiprocessing
import os
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _allowed_cpus() -> int:
    """
    Count the CPUs this process may run on.

    A batch job or ``taskset`` can limit a process to a few CPUs of a machine
    that has many; ``multiprocessing.cpu_count()`` counts them all.
    Where the system does not report the allowed CPUs (macOS and Windows), this
    is the machine's CPU count.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return multiprocessing.cpu_count()


def parallelize(func: Callable, num_processes: int | Literal["max"] = 1) -> Callable:
    """
    Parallelize a function across an iterable.

    Parameters
    ----------
    func : function
        The function to parallelize.
    num_processes : int or 'max', optional
        The number of processes to use. Uses the minimum of the number of
        items in the iterable and the number of CPUs requested, and never
        more than the CPUs this process may run on (a Slurm job's CPUs, not
        every CPU of the node; macOS and Windows count the machine's CPUs). If
        'max', uses all of those CPUs. Default is 1.

    Returns
    -------
    parallelized : function
        A function that will execute the input function in parallel across
        an iterable.

    Notes
    -----
    Use the call form only, binding the result to a new name::

        run = parallelize(func, num_processes=4)
        results = run(items)

    Don't use it as a decorator (``@parallelize``) or rebind the original
    name (``func = parallelize(func, 4)``): the worker processes find ``func``
    by its module-level name, so it then no longer pickles. ``func`` must be
    defined at module level (no lambdas or nested functions) for the same
    reason.
    """
    func_name = func.__name__

    def parallelized(iterable, **kwargs) -> list[Any]:
        """
        Execute the input function in parallel across an iterable.

        Parameters
        ----------
        iterable : iterable
            The iterable to parallelize the function across.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        results : list
            The results of the function applied to each item in the iterable.
        """
        # Materialize the iterable so generators work and len() is defined
        items = list(iterable)

        # Determine the number of processes to use
        cpu_count = _allowed_cpus()
        if num_processes == "max":
            processes = cpu_count
        elif num_processes > cpu_count:
            logger.warning(
                "%s processes requested, but this process may use only %d CPU(s).",
                num_processes,
                cpu_count,
            )
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(items):
            logger.info(
                "%s processes requested, but there are only %d items in the iterable.",
                num_processes,
                len(items),
            )
            processes = len(items)

        # If only one process is requested (or there is nothing to do),
        # execute the function sequentially
        if processes <= 1:
            logger.info("Executing %s sequentially...", func_name)
            results = [func(i, **kwargs) for i in items]
            return results

        logger.info(
            "Executing %s in parallel with %d processes...", func_name, processes
        )

        # Map the function across the items. The context manager terminates
        # the workers on exit, even when a worker raises.
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(func=partial(func, **kwargs), iterable=items)

        return results

    return parallelized
