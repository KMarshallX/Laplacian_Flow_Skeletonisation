"""Main init for crispy package."""

from . import _version
from .laplskel import _main

__version__ = _version.get_versions()['version']
