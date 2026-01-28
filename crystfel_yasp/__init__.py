"""
CrystFEL YASP - Yet Another Stream Parser

Memory-efficient CrystFEL stream file parsing and indexing.

Basic usage:
    from crystfel_yasp import iter_chunks, StreamIndex

    # Memory-efficient iteration
    for chunk in iter_chunks('file.stream'):
        print(chunk.filename, chunk.serial)

    # Build and use an index for random access
    index = StreamIndex.build('file.stream')
    index.save('file.stream.idx')

    index = StreamIndex.load('file.stream.idx')
    chunk = index.get_chunk(0)

    # Parallel parsing with shared memory
    from crystfel_yasp import parse_chunks_shared
    reflections, panels, metadata = parse_chunks_shared(index, entries)
"""

# Parser exports
from .parser import (
    Chunk,
    Crystal,
    Peak,
    Reflection,
    UnitCell,
    StreamFile,
    iter_chunks,
    parse_stream,
    HAS_NUMPY,
)

# Indexing exports
from .indexing import (
    StreamIndex,
    ChunkEntry,
)

# Parallel parsing exports
from .parallel import (
    parse_chunks_shared,
    parse_chunks_sequential,
    REFL_COLUMNS,
)

__version__ = "0.1.0"

__all__ = [
    # Parser
    "Chunk",
    "Crystal",
    "Peak",
    "Reflection",
    "UnitCell",
    "StreamFile",
    "iter_chunks",
    "parse_stream",
    "HAS_NUMPY",
    # Indexing
    "StreamIndex",
    "ChunkEntry",
    # Parallel
    "parse_chunks_shared",
    "parse_chunks_sequential",
    "REFL_COLUMNS",
]
