"""
Files, downloads and caching.

Download by FTP (:func:`ftp_download`) or with ``wget``, unzip, list the files
in a folder that match a pattern, read a KML file, and cache a function's
results in a pickle file (:class:`Cacher`).
"""

import fnmatch
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from lair._optional import import_optional_dependency

if TYPE_CHECKING:
    from fastkml.kml import KML

logger = logging.getLogger(__name__)


def unzip(zf: str, dir_path: str | None = None):
    """
    Unzip a file into a folder.

    Parameters
    ----------
    zf : str
        Path to zip file
    dir_path : str, optional
        Folder to unzip into; by default the folder that holds the zip file.

    See Also
    --------
    wget_download : Downloads files and unzips them.

    Examples
    --------
    >>> import tempfile, zipfile
    >>> from pathlib import Path
    >>> from lair.records import unzip
    >>> folder = Path(tempfile.mkdtemp())
    >>> with zipfile.ZipFile(folder / "data.zip", "w") as z:
    ...     z.writestr("a.txt", "hello")
    >>> unzip(str(folder / "data.zip"))
    >>> (folder / "a.txt").read_text()
    'hello'
    """
    import zipfile

    with zipfile.ZipFile(zf, "r") as zip_ref:
        zip_ref.extractall(dir_path or os.path.dirname(zf))


def list_files(
    path: str | Path = ".",
    pattern: str | None = None,
    ignore_case: bool = False,
    all_files: bool = False,
    full_names: bool = False,
    recursive: bool = False,
    followlinks: bool = False,
) -> list[str]:
    """
    Return a list of files in the specified directory that match the specified pattern.

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

    See Also
    --------
    glob.glob : Matches a pattern in a path, which can hold wildcards.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from lair.records import list_files
    >>> folder = Path(tempfile.mkdtemp())
    >>> for name in ["a.nc", "b.nc", "Notes.txt", ".hidden.nc"]:
    ...     (folder / name).touch()
    >>> sorted(list_files(folder, "*.nc"))
    ['a.nc', 'b.nc']
    >>> sorted(list_files(folder, "*.TXT", ignore_case=True))
    ['Notes.txt']
    >>> sorted(list_files(folder, "*.nc", all_files=True))
    ['.hidden.nc', 'a.nc', 'b.nc']
    """
    result = []
    if recursive:
        walk = os.walk(path, followlinks=followlinks)
    else:
        walk = [(path, None, os.listdir(path))]

    if ignore_case and pattern is not None:
        pattern = pattern.lower()

    for root, _, files in walk:
        for file in files:
            if all_files or not file.startswith("."):
                fn = file.lower() if ignore_case else file
                # fnmatchcase: plain fnmatch ignores case on Windows, so
                # ignore_case=False would not be case-sensitive there
                if pattern is None or fnmatch.fnmatchcase(fn, pattern):
                    if full_names:
                        result.append(os.path.abspath(os.path.join(root, file)))
                    else:
                        result.append(file)
    return result


def read_kml(path: str) -> "KML":
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

    See Also
    --------
    lair.transects : Metrics along a route, such as one read from a KML file.

    Notes
    -----
    Needs the ``formats`` extra (fastkml).

    Examples
    --------
    >>> from lair.records import read_kml
    >>> kml = read_kml("route.kml")  # doctest: +SKIP
    """
    # Optional dependency for KML file support
    import_optional_dependency("fastkml")
    from fastkml import kml

    if hasattr(kml.KML, "parse"):
        # fastkml >= 1.0: from_string/parse are classmethods returning a new KML
        return kml.KML.parse(path)

    # fastkml < 1.0: from_string fills the instance in place
    with open(path) as f:
        k = kml.KML()
        k.from_string(f.read().encode("utf-8"))
    return k


def wget_download(
    urls: str | list[str],
    download_dir: str,
    prefix: str | None = None,
    num_threads: int = 1,
    unzip: bool = True,
):
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
        Maximum number of threads to use for downloading. Defaults to 1.
    unzip : bool, optional
        Whether to unzip the downloaded files if they are ZIP files. Defaults to True.

    See Also
    --------
    ftp_download : The same for an FTP server.
    unzip : Unzips one file.

    Notes
    -----
    Needs ``wget`` (and ``unzip``, to unzip) on the PATH. A URL that fails is
    logged as a warning and skipped.

    Examples
    --------
    >>> from lair.records import wget_download
    >>> wget_download(
    ...     "https://example.com/data/file.zip", "downloads"
    ... )  # doctest: +SKIP
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
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
            if prefix == "":
                # Recreate the entire structure
                common = parsed_url.path.strip("/")  # Remove leading '/'
            else:
                # Get the relative strucuture from prefix
                common = os.path.relpath(parsed_url.path.strip("/"), prefix.strip("/"))
        else:
            # Drop each file into the download_dir
            common = os.path.basename(parsed_url.path)
        local_path = os.path.join(download_dir, common)

        # Ensure the output directory exists
        output_dir = os.path.dirname(local_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Downloading: %s to %s", url, local_path)
        try:
            # Use wget with the -O option to specify the output file
            subprocess.run(["wget", "-O", local_path, url], check=True)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to download %s: %s", url, e)
            return

        if unzip and local_path.endswith(".zip"):
            logger.info("Unzipping %s", local_path)
            subprocess.run(["unzip", "-d", output_dir, local_path], check=True)
            os.remove(local_path)

    if isinstance(urls, str):
        urls = [urls]  # a bare string would otherwise be iterated per character

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Consume the results so exceptions raised in the workers surface here
        list(executor.map(download_file, urls))


def _path_matches(path: str, pattern: str) -> bool:
    """Glob match on the full path if ``pattern`` has wildcards, else a substring match."""
    if any(char in pattern for char in "*?["):
        return fnmatch.fnmatch(path, pattern)
    return pattern in path


def ftp_download(
    host: str,
    paths: str | list[str],
    download_dir: str,
    username: str = "anonymous",
    password: str = "",
    prefix: str | None = None,
    pattern: str | None = None,
):
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
        With None, each path is saved under its own name in ``download_dir`` (a
        folder keeps its name and what is below it). With a prefix, the remote
        folders below the prefix are kept. With an empty string, the whole
        remote path is recreated.
    pattern : str, optional
        Only download files whose remote path matches. With wildcards (``*``,
        ``?``, ``[``) it is a glob against the full remote path (e.g.
        ``'*2015-06*.nc'``); otherwise any path containing it matches.
        Defaults to None (all files).

    Returns
    -------
    bool
        True when the download finishes. A failed transfer raises.

    See Also
    --------
    wget_download : The same for URLs.
    lair.noaa.CarbonTracker.download : Calls it with NOAA's CarbonTracker folders.
    lair.noaa.GMLData.download : Calls it with NOAA GML's data folders.

    Examples
    --------
    >>> from lair.records import ftp_download
    >>> ftp_download(
    ...     "ftp.gml.noaa.gov", "user/thoning/ccgcrv/ccg_filter.zip", "downloads"
    ... )  # doctest: +SKIP
    True
    """
    import ftplib

    if username == "anonymous" and password == "":
        password = "anonymous@"

    # The with block closes the connection however the download ends (a failed
    # transfer, or a QUIT the server never answers).
    with ftplib.FTP(host) as ftp:
        ftp.login(username, password)

        if isinstance(paths, str):
            paths = [paths]  # Convert a single path to a list

        for path in paths:
            # Start in root for every path
            ftp.cwd("/")

            PATH = "/" + path.strip("/")  # path should start from root on ftp

            # Redefine download func for each path; it is called within this iteration
            def download(path, root=PATH):
                try:
                    # Try changing to the specified path
                    ftp.cwd(path)

                except ftplib.error_perm as e:
                    # 550 = not a directory (the wording varies between servers)
                    if not str(e).startswith("550"):
                        raise
                    # If it's not a directory, download the file

                    if pattern is not None and not _path_matches(path, pattern):
                        # Exit if pattern is not in path
                        logger.debug("Skipping %s - pattern does not match", path)
                        return None

                    # Get common path to append to download_dir
                    if prefix is not None:
                        if prefix == "":
                            # Recreate the entire structure
                            common = path.strip("/")  # Remove leading '/'
                        else:
                            # Get the relative strucuture from prefix
                            common = os.path.relpath(path.strip("/"), prefix.strip("/"))
                    else:
                        # Drop each PATH directory into the download_dir
                        common = os.path.relpath(path, os.path.dirname(root))

                    # Create the local directory structure
                    local = os.path.join(download_dir, common)
                    os.makedirs(os.path.dirname(local), exist_ok=True)

                    # Download the file
                    with open(local, "wb") as local_file:
                        logger.info(
                            "Downloading %s to %s", path, os.path.dirname(local)
                        )
                        ftp.retrbinary(f"RETR {path}", local_file.write)

                    return "f"

                else:  # path is a directory
                    files = ftp.nlst()  # Get a list of files in that directory

                    for file in files:
                        # recursively download files
                        f_d = download("/".join([path, file]))

                        if f_d == "d":  # file is a directory
                            # download changed to a subdirectory
                            # restart in the above directory to be able to traverse multiple dirs
                            ftp.cwd(path)
                    return "d"

            download(PATH)

    return True


class Cacher:
    """
    A class that caches function results to a file for future use.

    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    func : function
        The function to be cached.
    cache_file : str
        The name of the file to cache results to; it must end in ``.pkl``.
    reload : bool, optional
        Whether to re-run the function instead of returning cached results.
        Each set of arguments is re-run once per Cacher; the fresh result
        replaces the old one in the index and every other cached entry is
        kept. (The old result stays in ``cache_file`` but is no longer
        referenced.) Defaults to False.
    """

    import pickle as pkl

    def __init__(self, func: Callable, cache_file: str, reload=False):
        """Initialize a Cacher; the class docstring describes the parameters."""
        if not cache_file.endswith(".pkl"):
            raise ValueError(f"cache_file must end in .pkl: {cache_file!r}")

        self.func = func
        self.cache_file = cache_file
        self.reload = reload

        # Make sure the directory exists for the cache file
        head, tail = os.path.split(cache_file)
        os.makedirs(head or ".", exist_ok=True)

        self.index_file = os.path.join(head, f".{tail}.index")
        self.cache_index = self.load_cache_index()
        self._refreshed = set()  # keys re-run by this Cacher when reload=True

    def load_cache_index(self):
        """
        Load the cache index from a file.

        Returns
        -------
        dict
            A dictionary of cached results and their corresponding file
            positions.
        """
        try:
            with open(self.index_file, "rb") as f:
                cache_index = self.pkl.load(f)
        except FileNotFoundError:
            cache_index = {}

        return cache_index

    def save_cache_index(self):
        """
        Save the cache index to a file.
        """
        with open(self.index_file, "wb") as f:
            self.pkl.dump(self.cache_index, f, protocol=self.pkl.HIGHEST_PROTOCOL)

    def __call__(self, *args, **kwargs):
        """
        Return the cached result, or call the function and cache its result.

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

        if self.reload and key not in self._refreshed:
            logger.info("Forcing %s to execute", self.func.__name__)
            cached = False
        else:
            cached = key in self.cache_index

        if cached:
            # Load the result from the cache_file using its index
            logger.debug(
                "Returning cached result for %s with args: %s %s",
                self.func.__name__,
                args,
                kwargs,
            )

            with open(self.cache_file, "rb") as f:
                f.seek(self.cache_index[key])  # go to index in cache_file
                result = self.pkl.load(f)

        else:
            # Call the function and update the cache
            result = self.func(*args, **kwargs)

            with open(self.cache_file, "ab") as f:
                f.seek(0, 2)  # go to the end of the pickle file
                pos = f.tell()  # get position of the end

                # dump result to the end of the pickle
                self.pkl.dump(result, f, protocol=self.pkl.HIGHEST_PROTOCOL)

            self.cache_index[key] = pos
            self._refreshed.add(key)

            logger.debug("Added %s %s to cache", args, kwargs)

            self.save_cache_index()  # update cache_index
        return result
