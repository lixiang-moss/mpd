"""
Adapter layer package.

Adapters are responsible for:
- Parsing the system-under-test's artifact directory (discover/load/extract)
- Emitting per-trial row dicts with a unified field schema for aggregation/ranking/plotting

In this L1 build, the first adapter is `MPDAdapter`. When adding support for other planners,
add new adapter modules in this directory.
"""

from .mpd import MPDAdapter

__all__ = ["MPDAdapter"]
