"""
Sphinx configuration for the lair docs.

The full list of settings: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime as dt
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath(".."))

# Never reach for the NOAA GML FTP download (CCG filter) while building docs;
# lair._ccg_filter is mocked below instead.
os.environ.setdefault("LAIR_SKIP_CCG_DOWNLOAD", "1")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LAIR"
copyright = f"{dt.date.today():%Y}, James Mineau"
author = "James Mineau"


def _detect_release() -> str:
    # setuptools-scm writes the version into the installed package metadata
    try:
        return package_version("lair")
    except PackageNotFoundError:
        return "0+unknown"


release = _detect_release()
version = release
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Extension configuration -------------------------------------------------

# Docstrings are numpydoc style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_default_options = {
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# Optional extras (and the generated CCG filter) are mocked so the docs build in
# the lean uv environment. Core dependencies are imported for real.
autodoc_mock_imports = [
    "boto3",
    "botocore",
    "cartopy",
    "fastkml",
    "lair._ccg_filter",
    "molmass",
    "networkx",
    "numcodecs",
    "pyproj",
    "rasterio",
    "requests",
    "rioxarray",
    "s3fs",
    "scipy",
    "shapely",
    "siphon",
    "xesmf",
    "zarr",
]

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
}

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"lair {version_match}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/jmineau/lair",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/lair/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at stable
    "logo": {
        "image_light": "_static/lair_forlight_r.png",
        "image_dark": "_static/lair_fordark_r.png",
        "alt_text": "LAIR - Home",
    },
}


# -- Class docstrings ----------------------------------------------------------
# A class docstring's NumPy "Methods" section, and the "Attributes" entries that
# are properties, describe members that autodoc's `:members:` documents again
# (Sphinx then warns of a duplicate object description). Drop them before napoleon
# turns the sections into directives; plain attributes, which autodoc does not
# document, stay.


def _is_underline(line: str) -> bool:
    return bool(line.strip()) and set(line.strip()) == {"-"}


def _section_end(lines: list[str], start: int) -> int:
    """Index of the next section header after *start*, or the end of *lines*."""
    for k in range(start, len(lines) - 1):
        if lines[k].strip() and _is_underline(lines[k + 1]):
            return k
    return len(lines)


def _is_documented_member(cls: object, name: str) -> bool:
    import functools
    import inspect

    member = inspect.getattr_static(cls, name, None)
    return isinstance(member, (property, functools.cached_property))


def drop_member_sections(app, what, name, obj, options, lines) -> None:
    """Remove what autodoc documents twice from a class docstring."""
    if what != "class":
        return
    i = 0
    while i < len(lines) - 1:
        title = lines[i].strip()
        if title not in ("Methods", "Attributes") or not _is_underline(lines[i + 1]):
            i += 1
            continue
        end = _section_end(lines, i + 2)
        if title == "Methods":
            del lines[i:end]
            continue
        # Attributes: keep the entries autodoc does not document.
        indent = len(lines[i]) - len(lines[i].lstrip())
        kept, entry, keep = [], [], True
        for line in lines[i + 2 : end]:
            starts_entry = line.strip() and len(line) - len(line.lstrip()) == indent
            if starts_entry:
                if keep:
                    kept += entry
                attr = line.strip().split(":")[0].strip()
                entry, keep = [line], not _is_documented_member(obj, attr)
            else:
                entry.append(line)
        if keep:
            kept += entry
        if any(line.strip() for line in kept):
            lines[i + 2 : end] = kept
            i += 2 + len(kept)
        else:
            del lines[i:end]


def setup(app) -> None:
    """Register the class-docstring hook ahead of napoleon (priority 500)."""
    app.connect("autodoc-process-docstring", drop_member_sections, priority=400)
