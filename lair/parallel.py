"""
Parallelization utilities.
"""

from functools import partial
import multiprocessing
from typing import Any, Callable, Literal

from lair.config import vprint

def parallelize(func: Callable, num_processes: int | Literal['max'] = 1
                ) -> Callable:
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
        # Determine the number of processes to use
        cpu_count = multiprocessing.cpu_count()
        if num_processes == 'max':
            processes = cpu_count
        elif num_processes > cpu_count:
            vprint(f'Warning: {num_processes} processes requested, '
                    f'but there are only {cpu_count} CPU(s) available.')
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(iterable):
            vprint(f'Info: {num_processes} processes requested, '
                    f'but there are only {len(iterable)} items in the iterable.')
            processes = len(iterable)

        # If only one process is requested, execute the function sequentially
        if processes == 1:
            vprint(f'Executing {func_name} sequentially...')
            results = [func(i, **kwargs) for i in iterable]
            return results

        vprint(f'Executing {func_name} in parallel with {processes} processes...')

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=processes)

        # Use the pool to map the function across the iterable
        results = pool.map(func=partial(func, **kwargs), iterable=iterable)

        # Close the pool to free resources
        pool.close()
        pool.join()

        return results

    return parallelized