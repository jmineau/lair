"""
Automatically update the version of the package.
To be used with pre-commit hooks.

CalVer format: YYYY.MM.PATCH
Publishing versions in May, August, and December.
"""
import datetime
import os
import re

pyproject_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml')

# Get the current version
with open(pyproject_file, 'r') as f:
    content = f.read()
    current_version = re.search(r'version = \"(\d{4}\.\d{2}\.\d+)\"', content).group(1)
cur_year, cur_month, cur_patch = current_version.split('.')

is_major_change = False

def get_version() -> str:
    global is_major_change
    now = datetime.datetime.now()
    year = now.year
    month = now.month

    if month <= 5:
        month = 5
    elif month > 5 and month <= 8:
        month = 8
    else:
        month = 12

    # If the month has changed, reset the patch
    if month != int(cur_month):
        is_major_change = True
        return f'{year}.{month:02d}.0'

    patch = int(cur_patch) + 1

    return f'{year}.{month:02d}.{patch}'


def update(version: str):
    with open(pyproject_file, 'r') as f:
        content = f.read()

    # Update the version line in pyproject.toml
    updated_content = re.sub(
        r'version = \"\d{4}\.\d{2}\.\d+\"',
        f'version = "{version}"',
        content
    )

    with open(pyproject_file, 'w') as f:
        f.write(updated_content)


if __name__ == "__main__":
    version = get_version()
    update(version)
    print(f'Updated version to {version}')
