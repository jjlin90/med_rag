"""Medical RAG package and Windows native dependency initialization."""

import sys

if sys.platform == "win32":
    # Initialize Arrow at shallow import depth before Transformers/FlagEmbedding.
    # On the audited CPython 3.10 environment, their nested pandas -> Arrow
    # import caused 0xC0000005; preloading Arrow passed isolated-process probes.
    # This does not load models or connect to external services.
    try:
        import pyarrow as _pyarrow
    except ImportError:
        # Minimal installations can still use modules that do not need Arrow.
        pass
