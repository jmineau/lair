"""
Shared pytest configuration and fixtures for the lair test suite.

Keep this file *minimal*. Anything specific to a single module belongs in that
module's own ``tests/<module>/conftest.py`` so it travels with the module when
the module is extracted into its own package (see ``tests/README.md``).
"""

import os

# Make `import lair` hermetic: never reach for the NOAA GML FTP download during
# tests. If lair/_ccg_filter.py is absent (fresh checkout / CI), importing lair
# would otherwise try to fetch it over FTP. This only takes effect when the file
# is missing; when present (the usual case on CHPC) the real module is used.
os.environ.setdefault("LAIR_SKIP_CCG_DOWNLOAD", "1")
