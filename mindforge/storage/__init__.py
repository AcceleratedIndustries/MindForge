"""Storage seam. Minimal protocol with a FilesystemStorage implementation.

Phase 3 will migrate concept/graph/provenance writes behind this interface.
This session introduces the seam; deferred broader adoption is intentional.
"""

from mindforge.storage.fs import FilesystemStorage, Storage

__all__ = ["Storage", "FilesystemStorage"]
