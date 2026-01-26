"""
Shared Memory Parallel Chunk Parsing

Pre-allocates exact-sized shared numpy arrays from index metadata, then spawns
workers that write reflection data directly into their assigned slice — zero
pickle, zero copy for the bulk numeric data.

Why shared memory instead of ProcessPoolExecutor + pickle:
  Returning List[Chunk] with thousands of Reflection objects per worker costs
  more to pickle than the parallelism saves. Shared memory avoids this by
  writing numeric data directly into a pre-allocated array.

Usage:
    from parallel_parse import parse_chunks_shared
    from indexing import StreamIndex

    index = StreamIndex.load("file.stream.idx")
    entries = index.filter(hit=True)
    reflections, panels, metadata = parse_chunks_shared(index, entries, num_workers=8)
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

# Ensure we can import from the scripts directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_parser import CHUNK_START, _parse_chunk

# Column names for the reflection array (9 numeric fields)
REFL_COLUMNS = ["h", "k", "l", "intensity", "sigma", "peak", "background", "fs", "ss"]


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable)
# ---------------------------------------------------------------------------

def _shared_parse_worker(args):
    """Parse assigned chunks and write reflection numerics into shared memory.

    Args is a tuple: (stream_path, entry_tuples, shm_name, shm_shape, offsets)
      - stream_path: path to the .stream file
      - entry_tuples: list of (start_offset, end_offset) for each chunk
      - shm_name: name of the SharedMemory block
      - shm_shape: (total_rows, 9) shape of the shared array
      - offsets: list of row-start indices, one per entry in entry_tuples

    Returns:
      - panels: list of panel strings in reflection order (one per reflection)
      - metadata: list of dicts with lightweight chunk info (one per entry)
    """
    stream_path, entry_tuples, shm_name, shm_shape, offsets = args

    # Attach to existing shared memory (no copy)
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=np.float64, buffer=shm.buf)

    panels = []
    metadata = []

    with open(stream_path, "rb") as f:
        for i, (start, end) in enumerate(entry_tuples):
            f.seek(start)
            raw = f.read(end - start)
            text = raw.decode("utf-8", errors="replace")
            lines_list = text.splitlines()

            # _parse_chunk expects the CHUNK_START marker already consumed
            line_iter = iter(lines_list)
            for line in line_iter:
                if line.strip() == CHUNK_START:
                    break

            chunk = _parse_chunk(line_iter)
            if chunk is None:
                metadata.append(None)
                continue

            # Write reflection numeric data directly into shared array
            row = offsets[i]
            for crystal in chunk.crystals:
                for refl in crystal.reflections:
                    arr[row] = [
                        refl.h, refl.k, refl.l, refl.intensity,
                        refl.sigma, refl.peak, refl.background,
                        refl.fs, refl.ss,
                    ]
                    panels.append(refl.panel)
                    row += 1

            # Collect lightweight metadata (no reflection objects)
            crystal_meta = []
            for crystal in chunk.crystals:
                crystal_meta.append({
                    "cell_a": crystal.cell.a,
                    "cell_b": crystal.cell.b,
                    "cell_c": crystal.cell.c,
                    "cell_alpha": crystal.cell.alpha,
                    "cell_beta": crystal.cell.beta,
                    "cell_gamma": crystal.cell.gamma,
                    "lattice_type": crystal.cell.lattice_type,
                    "centering": crystal.cell.centering,
                    "num_reflections": crystal.num_reflections,
                    "resolution_limit": crystal.resolution_limit,
                })
            metadata.append({
                "filename": chunk.filename,
                "event": chunk.event,
                "serial": chunk.serial,
                "hit": chunk.hit,
                "indexed_by": chunk.indexed_by,
                "num_peaks": chunk.num_peaks,
                "crystals": crystal_meta,
            })

    shm.close()  # detach (don't unlink — parent owns it)
    return panels, metadata


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def parse_chunks_shared(index, entries=None, num_workers=None):
    """Parse chunks using shared-memory parallel approach.

    Args:
        index: StreamIndex with stream_path set
        entries: list of ChunkEntry to parse (default: all)
        num_workers: number of worker processes (default: min(cpu_count, len(entries)))

    Returns:
        reflections: np.ndarray shape (N, 9) — h, k, l, I, sigma, peak, bg, fs, ss
        panels: list of str, length N — panel name per reflection
        metadata: list of dicts, one per chunk
    """
    entries = entries if entries is not None else index.entries
    if not entries:
        return np.empty((0, 9), dtype=np.float64), [], []

    num_workers = num_workers or min(os.cpu_count() or 1, len(entries))
    num_workers = max(1, min(num_workers, len(entries)))

    # 1. Pre-compute row offsets from index metadata
    counts = [e.num_reflections for e in entries]
    cumulative = np.cumsum([0] + counts)
    total = int(cumulative[-1])

    if total == 0:
        return np.empty((0, 9), dtype=np.float64), [], [None] * len(entries)

    # 2. Allocate shared memory for reflection array
    shm_size = total * 9 * 8  # float64 = 8 bytes
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    shm_shape = (total, 9)

    try:
        # Zero-fill via numpy view
        arr = np.ndarray(shm_shape, dtype=np.float64, buffer=shm.buf)
        arr[:] = 0.0

        # 3. Split work into contiguous slices for each worker
        n = len(entries)
        slice_size = (n + num_workers - 1) // num_workers
        worker_args = []
        for w in range(num_workers):
            start_idx = w * slice_size
            end_idx = min(start_idx + slice_size, n)
            if start_idx >= n:
                break
            entry_tuples = [
                (entries[j].start_offset, entries[j].end_offset)
                for j in range(start_idx, end_idx)
            ]
            offsets = [int(cumulative[j]) for j in range(start_idx, end_idx)]
            worker_args.append((
                index.stream_path,
                entry_tuples,
                shm.name,
                shm_shape,
                offsets,
            ))

        # 4. Run workers via ProcessPoolExecutor
        all_panels = []
        all_metadata = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_shared_parse_worker, worker_args))

        for panels_chunk, meta_chunk in results:
            all_panels.extend(panels_chunk)
            all_metadata.extend(meta_chunk)

        # 5. Copy result out of shared memory into a regular numpy array
        result_arr = arr.copy()

    finally:
        shm.close()
        shm.unlink()

    return result_arr, all_panels, all_metadata


# ---------------------------------------------------------------------------
# Sequential baseline (same output format for comparison)
# ---------------------------------------------------------------------------

def parse_chunks_sequential(index, entries=None):
    """Parse chunks sequentially, returning the same format as parse_chunks_shared.

    Uses the same offset-based array layout so results are directly comparable.

    Returns:
        reflections: np.ndarray shape (N, 9)
        panels: list of str, length N
        metadata: list of dicts, one per chunk
    """
    entries = entries if entries is not None else index.entries
    if not entries:
        return np.empty((0, 9), dtype=np.float64), [], []

    # Use the same offset-based layout as the parallel version
    counts = [e.num_reflections for e in entries]
    cumulative = np.cumsum([0] + counts)
    total = int(cumulative[-1])

    if total == 0:
        return np.empty((0, 9), dtype=np.float64), [], [None] * len(entries)

    arr = np.zeros((total, 9), dtype=np.float64)
    panels = []
    metadata = []

    with open(index.stream_path, "rb") as f:
        for idx, entry in enumerate(entries):
            f.seek(entry.start_offset)
            raw = f.read(entry.end_offset - entry.start_offset)
            text = raw.decode("utf-8", errors="replace")
            lines_list = text.splitlines()

            # _parse_chunk expects CHUNK_START already consumed
            line_iter = iter(lines_list)
            for line in line_iter:
                if line.strip() == CHUNK_START:
                    break

            chunk = _parse_chunk(line_iter)
            if chunk is None:
                metadata.append(None)
                continue

            # Write at the same offset the parallel version would use
            row = int(cumulative[idx])
            for crystal in chunk.crystals:
                for refl in crystal.reflections:
                    arr[row] = [
                        refl.h, refl.k, refl.l, refl.intensity,
                        refl.sigma, refl.peak, refl.background,
                        refl.fs, refl.ss,
                    ]
                    panels.append(refl.panel)
                    row += 1

            crystal_meta = []
            for crystal in chunk.crystals:
                crystal_meta.append({
                    "cell_a": crystal.cell.a,
                    "cell_b": crystal.cell.b,
                    "cell_c": crystal.cell.c,
                    "cell_alpha": crystal.cell.alpha,
                    "cell_beta": crystal.cell.beta,
                    "cell_gamma": crystal.cell.gamma,
                    "lattice_type": crystal.cell.lattice_type,
                    "centering": crystal.cell.centering,
                    "num_reflections": crystal.num_reflections,
                    "resolution_limit": crystal.resolution_limit,
                })
            metadata.append({
                "filename": chunk.filename,
                "event": chunk.event,
                "serial": chunk.serial,
                "hit": chunk.hit,
                "indexed_by": chunk.indexed_by,
                "num_peaks": chunk.num_peaks,
                "crystals": crystal_meta,
            })

    return arr, panels, metadata
