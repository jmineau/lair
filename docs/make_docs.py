"""
Build the documentation for the project.
"""

import argparse
import os
from pathlib import Path
import subprocess
import requests


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument(
    '--version',
    nargs='*',  # Accepts zero or more arguments
    default='all',  # Default to 'all' if no version is specified
    help='The version of the documentation to build. Specify one or more versions, or "all" for all versions. Default: all',
    type=str
)

args = parser.parse_args()


def sphinx_build(version: str, source: str | Path, build: str | Path):
    """
    Build the documentation for a specific version of the project.

    Parameters
    ----------
    version : str
        The version of the project to build the documentation for.
    source : str
        The path to the source documentation.
    build : str
        The path to the built documentation.
    latest : str, optional
        The path to the latest documentation. Default: None
    """

    # Set the version in the environment
    os.environ['LAIR_DOCS_VERSION'] = version

    # Build the docs
    os.system(f'sphinx-build {source} {build}')


def checkout_version(version: str):
    """
    Checkout the specified version of the project.

    Parameters
    ----------
    version : str
        The version of the project to checkout.
    """
    target = 'main' if version == 'latest' else f'v{version}'
    try:
        subprocess.run(['git', 'checkout', target], check=True)
        print(f"Checked out {target} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while checking out {target}: {e}")


host = Path('~/public_html/lair').expanduser()
dev = Path('~/lair/docs').expanduser()
docs = Path('~/.lair-docs/docs').expanduser()
source = docs / 'source'

# Get list of versions from switcher
switcher_url = 'https://raw.githubusercontent.com/jmineau/lair/main/docs/versions.json'
switcher = requests.get(switcher_url).json()

# Determine which versions to build
to_build: list = args.version
if to_build == 'all':
    to_build = [v['version'] for v in switcher]

print('to_build:', to_build)

# Change to the docs directory
os.chdir(docs)

# Pull the latest changes from the git repository
subprocess.run(['git', 'pull'], check=True)

# Iterate over the versions in the switcher
for v in switcher:
    version = v['version']
    name = v.get('name', '')

    if version == 'dev' and 'dev' in to_build:
        print(f"Building version {version} ({name})")
        # Build the dev version from my local clone
        sphinx_build('dev', source=dev / 'source', build=host / 'dev')

    elif 'latest' in name:
        if 'latest' not in to_build and version not in to_build:
            # Skip building the latest version if it's not specified
            continue
        print(f"Building version {version} ({name})")
        # Build the latest version from the head of main branch
        checkout_version('latest')
        sphinx_build(version, source=source, build=host / 'latest')
    else:
        if version not in to_build and name != '' and name not in to_build:
            # Skip building the version if it's not specified
            continue
        print(f"Building version {version} ({name})")
        # Checkout the specified version
        checkout_version(version)
        sphinx_build(version, source=source, build=host / version)

# Checkout the main branch
checkout_version('latest')
