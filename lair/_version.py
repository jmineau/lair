"""
Automatically update the version of the package.
To be used with pre-commit hooks.

CalVer format: YYYY.MM.PATCH
Publishing versions in May, August, and December.
"""
import datetime
import os
import re

init_file = os.path.join(os.path.dirname(__file__), '__init__.py')

# Get the current version
with open(init_file, 'r') as f:
    current_version = re.search(r'__version__ = [\'\"](\d{4}\.\d{2}\.\d+)[\'\"]',
                                f.read()).group(1)
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
    with open(init_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith('__version__'):
            updated_lines.append(f"__version__ = '{version}'\n")
        else:
            updated_lines.append(line)

    with open(init_file, 'w') as f:
        f.writelines(updated_lines)


if __name__ == "__main__":
    version = get_version()
    update(version)
    print(f'Updated version to {version}')
