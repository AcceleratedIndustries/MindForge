"""MindForge: A semantic memory engine for AI conversation transcripts."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mindforge-kb")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
