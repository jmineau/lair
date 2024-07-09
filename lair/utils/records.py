"""
lair.utils.records
~~~~~~~~~~~~~~~~~~

Utilities for working with files and directories.
"""

from typing import Callable, Literal

from lair.config import vprint


def unzip(zf, dir_path=None):
    '''Unzip file into dir_path if given'''
    import zipfile

    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(dir_path)


def list_files(path: str = '.', pattern: str = None, ignore_case: bool = False, all_files: bool = False,
               full_names: bool = False, recursive: bool = False) -> list[str]:
    """
    Returns a list of files in the specified directory that match the specified pattern.

    Args:
        path (str): The directory to search for files. Defaults to the current directory.
        pattern (str): The glob-style pattern to match against file names. Defaults to None, which matches all files.
        ignore_case (bool): Whether to ignore case when matching file names. Defaults to False.
        all_files (bool): Whether to include hidden files (files that start with a dot). Defaults to False.
        full_names (bool): Whether to return the full path of each file. Defaults to False.
        recursive (bool): Whether to search for files recursively in subdirectories. Defaults to False.

    Returns:
        List[str]: A list of file names or full paths that match the specified pattern.
    """
    import fnmatch
    import os
    
    result = []
    if recursive:
        walk = os.walk(path)
    else:
        walk = [(path, None, os.listdir(path))]
        
    if ignore_case and pattern is not None:
        pattern = pattern.lower()

    for root, _, files in walk:
        for file in files:
            if all_files or not file.startswith('.'):
                fn = file.lower() if ignore_case else file
                if pattern is None or fnmatch.fnmatch(fn, pattern):
                    if full_names:
                        result.append(os.path.abspath(os.path.join(root, file)))
                    else:
                        result.append(file)
    return result


def read_kml(path):
    '''Read kml file from path'''
    from fastkml import kml

    with open(path, 'rt') as KML:
        k = kml.KML()
        k.from_string(KML.read().encode('utf-8'))
    return k


def ftp_download(host, paths, download_dir,
                 username='anonymous', password='',
                 prefix=None,
                 pattern=None):
    import ftplib
    import os

    if username == 'anonymous' and password == '':
        password = 'anonymous@'

    ftp = ftplib.FTP(host)
    ftp.login(username, password)

    if isinstance(paths, str):
        paths = [paths]  # Convert a single path to a list

    for path in paths:
        # Start in root for every path
        ftp.cwd('/')

        PATH = '/' + path.strip('/')  # path should start from root on ftp

        # Redefine download func for each path to pass PATH
        def download(path):
            try:
                # Try changing to the specified path
                ftp.cwd(path)

            except ftplib.error_perm as e:
                assert 'Failed to change directory' in str(e)
                # If it's not a directory, download the file

                if pattern is not None and pattern not in path:
                    # Exit if pattern is not in path
                    vprint(f'Skipping {path} - pattern does not match')
                    return None

                # Get common path to append to download_dir
                if prefix is not None:
                    if prefix == '':
                        # Recreate the entire structure
                        common = path.strip('/')  # Remove leading '/'
                    else:
                        # Get the relative strucuture from prefix
                        common = os.path.relpath(path, prefix)
                else:
                    # Drop each PATH directory into the download_dir
                    common = os.path.relpath(path, os.path.dirname(PATH))

                # Create the local directory structure
                local = os.path.join(download_dir, common)
                os.makedirs(os.path.dirname(local), exist_ok=True)

                # Download the file
                with open(local, 'wb') as local_file:
                    print(f'Downloading {path}')
                    ftp.retrbinary(f'RETR {path}', local_file.write)

                return 'f'

            else:  # path is a directory
                files = ftp.nlst()  # Get a list of files in that directory

                for file in files:
                    # recursively download files
                    f_d = download('/'.join([path, file]))

                    if f_d == 'd':  # file is a directory
                        # download changed to a subdirectory
                        # restart in the above directory to be able to traverse multiple dirs
                        ftp.cwd(path)
                return 'd'

        download(PATH)

    ftp.quit()
    return True


def parallelize_file_parser(file_parser: Callable,
                            num_processes: int | Literal['max'] = 1):
    """
    Parallelizes a file parser function to read multiple files in parallel.
    
    Args:
        file_parser (function): The function to be parallelized. Must be picklable.
        num_processes (int): The number of processes to use for parallelization. Defaults to 1.

    Returns:
        function: A parallelized version of the file parser function.
    """

    import multiprocessing
    from functools import partial

    def parallelized_parser(files: list, **kwargs):
        """
        Parses multiple files in parallel using the file parser function.

        Args:
            files (list): A list of file to be parsed. Format is determined by the file parser function.
            **kwargs: Additional keyword arguments to be passed to the file parser function.

        Returns:
            list: A list of datasets parsed from the input files.
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

        if processes > len(files):
            vprint(f'Info: {num_processes} processes requested, '
                   f'but there are only {len(files)} file(s) to parse.')
            processes = len(files) 

        # If only one process is requested, read files sequentially
        if processes == 1:
            vprint('Parsing files sequentially...')
            datasets = [file_parser(file, **kwargs) for file in files]
            return datasets

        vprint(f'Parsing files in parallel with {processes} processes...')

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=processes)

        # Use the pool to parse the files in parallel
        datasets = pool.map(partial(file_parser, **kwargs), files)

        # Close the pool to free resources
        pool.close()
        pool.join()

        return datasets

    return parallelized_parser


class Cacher:
    """
    A class that caches function results to a file for future use.

    Parameters
    ----------
    func : function
        The function to be cached.
    cache_file : str
        The name of the file to cache results to.
    """

    import pickle as pkl

    def __init__(self, func, cache_file, reload=False):
        """
        Initializes a Cacher object.

        Parameters
        ----------
        func : function
            The function to be cached.
        cache_file : str
            The name of the file to cache results to.
        """
        assert cache_file.endswith('.pkl')

        self.func = func
        self.cache_file = cache_file
        self.reload = reload

        self.cache_index = self.load_cache_index()

    def load_cache_index(self):
        """
        Loads the cache index from a file.

        Returns
        -------
        dict
            A dictionary of cached results and their corresponding file
            positions.
        """
        if self.reload:
            # TODO causes previous caches to become unreadable
            #   need to delete previous cache or append to index_file
            # Force func to be executed by setting cache_index to empty
            vprint(f'Forcing {self.func.__name__} to execute')
            return {}

        try:
            with open(self.cache_file + '.index', 'rb') as f:
                cache_index = self.pkl.load(f)
        except FileNotFoundError:
            cache_index = {}

        return cache_index

    def save_cache_index(self):
        """
        Saves the cache index to a file.
        """
        with open(self.cache_file + '.index', 'wb') as f:
            self.pkl.dump(self.cache_index, f,
                          protocol=self.pkl.HIGHEST_PROTOCOL)

    def __call__(self, *args, **kwargs):
        """
        Returns the cached result if available, otherwise calls the function
        and caches the result for future use.

        Parameters
        ----------
        *args : tuple
            The arguments to be passed to the function.

        Returns
        -------
        Any
            The result of the function call.
        """

        key = self.pkl.dumps((args, kwargs))  # serialize func args

        if key in self.cache_index:
            # Load the result from the cache_file using its index
            vprint(f"Returning cached result for {args} {kwargs}")

            with open(self.cache_file, 'rb') as f:
                f.seek(self.cache_index[key])  # go to index in cache_file
                result = self.pkl.load(f)

        else:
            # Call the function and update the cache
            result = self.func(*args, **kwargs)

            with open(self.cache_file, 'ab') as f:
                f.seek(0, 2)  # go to the end of the pickle file
                pos = f.tell()  # get position of the end

                # dump result to the end of the pickle
                self.pkl.dump(result, f, protocol=self.pkl.HIGHEST_PROTOCOL)

            self.cache_index[key] = pos

            vprint(f"Added {args} {kwargs} to cache")

            self.save_cache_index()  # update cache_index
        return result
