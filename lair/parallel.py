"""
Parallelization utilities.
"""

from functools import partial
import multiprocessing
from typing import Any, Callable, Literal

from lair.config import vprint


def parallelize(func: Callable, num_processes: int | Literal["max"] = 1) -> Callable:
    """
    Parallelize a function across an iterable.

    Parameters
    ----------
    func : function
        The function to parallelize.
    num_processes : int or 'max', optional
        The number of processes to use. Uses the minimum of the number of
        items in the iterable and the number of CPUs requested. If 'max',
        uses all available CPUs. Default is 1.

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
        cpu_count = multiprocessing.cpu_count()
        if num_processes == "max":
            processes = cpu_count
        elif num_processes > cpu_count:
            vprint(
                f"Warning: {num_processes} processes requested, "
                f"but there are only {cpu_count} CPU(s) available."
            )
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(items):
            vprint(
                f"Info: {num_processes} processes requested, "
                f"but there are only {len(items)} items in the iterable."
            )
            processes = len(items)

        # If only one process is requested (or there is nothing to do),
        # execute the function sequentially
        if processes <= 1:
            vprint(f"Executing {func_name} sequentially...")
            results = [func(i, **kwargs) for i in items]
            return results

        vprint(f"Executing {func_name} in parallel with {processes} processes...")

        # Map the function across the items. The context manager terminates
        # the workers on exit, even when a worker raises.
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(func=partial(func, **kwargs), iterable=items)

        return results

    return parallelized
