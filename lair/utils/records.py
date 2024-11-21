"""
Utilities for working with files and directories.
"""

import os
from pathlib import Path
from typing import Callable, Literal

from fastkml import KML

from lair.config import vprint


def unzip(zf: str, dir_path: str | None=None):
    '''
    Unzip file into dir_path if given
    
    Parameters
    ----------
    zf : str
        Path to zip file
    dir_path : str, optional
        Path to directory to unzip to, by default None
    '''
    import zipfile

    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(dir_path or os.path.dirname(zf))


def list_files(path: str | Path = '.', pattern: str|None = None, ignore_case: bool = False, all_files: bool = False,
               full_names: bool = False, recursive: bool = False, followlinks: bool = False) -> list[str]:
    """
    Returns a list of files in the specified directory that match the specified pattern.

    Parameters
    ----------
    path : str, optional
        The directory to search for files. Defaults to the current directory.
    pattern : str, optional
        The glob-style pattern to match against file names. Defaults to None, which matches all files.
    ignore_case : bool, optional
        Whether to ignore case when matching file names. Defaults to False.
    all_files : bool, optional
        Whether to include hidden files (files that start with a dot). Defaults to False.
    full_names : bool, optional
        Whether to return the full path of each file. Defaults to False.
    recursive : bool, optional
        Whether to search for files recursively in subdirectories. Defaults to False.
    followlinks : bool, optional
        Whether to follow symbolic links. Defaults to False.

    Returns
    -------
    List[str]
        A list of file names or full paths that match the specified pattern.
    """
    import fnmatch

    result = []
    if recursive:
        walk = os.walk(path, followlinks=followlinks)
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


def read_kml(path: str) -> KML:
    """
    Read kml file from path

    Parameters
    ----------
    path : str
        The path to the KML file.

    Returns
    -------
    KML
        The KML object.
    """
    from fastkml import kml

    with open(path, 'rt') as KML:
        k = kml.KML()
        k.from_string(KML.read().encode('utf-8'))
    return k


def wget_download(urls: str | list[str],
                  download_dir: str | None,
                  prefix: str | None = None, num_threads: int = 1,
                  unzip: bool = True):
    """
    Download multiple files from given URLs using wget and extract them if they are ZIP files.

    Parameters
    ----------
    urls : str | list[str]
        List of URLs to download.
    download_dir : str
        The local directory to download files to.
    prefix : str, optional
        The common prefix to use for the local directory structure. Defaults to None.
        If None, the files will be downloaded directly into the download_dir.
        If an empty string, the entire structure will be recreated.
    num_threads : int, optional
        Maximum number of threads to use for downloading, by default 4.
    unzip : bool, optional
        Whether to unzip the downloaded files if they are ZIP files. Defaults to True.
    """
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    from urllib.parse import urlparse

    def download_file(url: str):
        """
        Download a file from a given URL and unzip it if it's a ZIP file.

        Parameters
        ----------
        url : str
            URL of the file to download.
        """
        # Parse the URL to extract the filename
        parsed_url = urlparse(url)
        
        # Get common path to append to download_dir
        if prefix is not None:
            if prefix == '':
                # Recreate the entire structure
                common = parsed_url.path.strip('/')  # Remove leading '/'
            else:
                # Get the relative strucuture from prefix
                common = os.path.relpath(parsed_url.path.strip('/'), prefix)
        else:
            # Drop each file into the download_dir
            common = os.path.basename(parsed_url.path)
        local_path = os.path.join(download_dir, common)

        # Ensure the output directory exists
        output_dir = os.path.dirname(local_path)
        os.makedirs(output_dir, exist_ok=True)

        vprint(f"Downloading: {url} to {local_path}")
        try:
            # Use wget with the -O option to specify the output file
            subprocess.run(['wget', '-O', local_path, url], check=True)
        except subprocess.CalledProcessError as e:
            vprint(f"Failed to download {url}: {e}")
            return

        if unzip and local_path.endswith('.zip'):
            vprint('Unzipping...')
            subprocess.run(['unzip', '-d', output_dir, local_path], check=True)
            os.remove(local_path)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(download_file, urls)


def ftp_download(host: str, paths: str | list[str], download_dir: str,
                 username: str='anonymous', password: str='',
                 prefix: str | None=None,
                 pattern: str | None=None):
    """
    Recursively download files from an FTP server.

    Parameters
    ----------
    host : str
        The FTP server host.
    paths : str | list[str]
        The path(s) to download from the FTP server.
    download_dir : str
        The local directory to download files to.
    username : str, optional
        The username to use for the FTP server. Defaults to 'anonymous'.
    password : str, optional
        The password to use for the FTP server. Defaults to ''.
    prefix : str, optional
        The common prefix to use for the local directory structure. Defaults to None.
    pattern : str, optional
        The pattern to match against file names. Defaults to None.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    import ftplib

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
                    vprint(f'Downloading {path} to {os.path.dirname(local)}')
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


class Cacher:
    """
    A class that caches function results to a file for future use.

    Attributes
    ----------
    func : function
        The function to be cached.
    cache_file : str
        The name of the file to cache results to.
    reload : bool
        Whether to reload the cache index from the index file.
    """

    import pickle as pkl

    def __init__(self, func: Callable, cache_file: str, reload=False):
        """
        Initializes a Cacher object.

        Parameters
        ----------
        func : function
            The function to be cached.
        cache_file : str
            The name of the file to cache results to.
        reload : bool, optional
            Whether to reload the cache index from the index file. Defaults to False.
        """
        assert cache_file.endswith('.pkl')

        self.func = func
        self.cache_file = cache_file
        self.reload = reload

        # Make sure the directory exists for the cache file
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))

        head, tail = os.path.split(cache_file)
        self.index_file = f'{head}/.{tail}.index'
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
            with open(self.index_file, 'rb') as f:
                cache_index = self.pkl.load(f)
        except FileNotFoundError:
            cache_index = {}

        return cache_index

    def save_cache_index(self):
        """
        Saves the cache index to a file.
        """
        with open(self.index_file, 'wb') as f:
            self.pkl.dump(self.cache_index, f,
                          protocol=self.pkl.HIGHEST_PROTOCOL)

    def __call__(self, *args, **kwargs):
        """
        Returns the cached result if available, otherwise calls the function
        and caches the result for future use.

        Parameters
        ----------
        args : tuple
            The arguments to be passed to the function.

        Returns
        -------
        Any
            The result of the function call.
        """

        key = self.pkl.dumps((args, kwargs))  # serialize func args

        if key in self.cache_index:
            # Load the result from the cache_file using its index
            vprint(f"Returning cached result for {self.func.__name__} with args: {args} {kwargs}")

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
