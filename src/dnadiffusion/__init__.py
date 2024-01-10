"""
dnadiffusion.
"""

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "dnadiffusion package may not be installed"

del metadata


import pathlib

# Get path to the data directory in the same level as the src directory
DATA_DIR = str(pathlib.Path(__file__).parent.parent.parent / "data")
