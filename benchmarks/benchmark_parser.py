#!/usr/bin/env python3
"""
Benchmark script for crystfel_yasp parser

Measures parsing performance in lines/second across different modes:
- Skip reflections: Fastest mode for stats-only use cases
- Lazy (no access): Reflections collected but not parsed
- Full access: All reflections parsed

Usage:
    export UV_CACHE_DIR=/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/.UV_CACHE
    uv run --python 3.11 --with numpy python benchmarks/benchmark_parser.py /path/to/file.stream
    uv run --python 3.11 --with numpy python benchmarks/benchmark_parser.py /path/to/file.stream -n 100
"""

import argparse
import os
import time

from crystfel_yasp.parser import iter_chunks, HAS_NUMPY


def count_lines_in_chunks(filepath: str, num_chunks: int) -> int:
    """Count lines in the first num_chunks of a stream file."""
    chunk_count = 0
    line_count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line_count += 1
            if '----- Begin chunk -----' in line:
                chunk_count += 1
            if chunk_count > num_chunks:
                break
    return line_count


def benchmark_skip_mode(filepath: str, num_chunks: int) -> tuple:
    """Benchmark with load_reflections=False."""
    start = time.time()
    n_chunks, n_crystals = 0, 0
    for chunk in iter_chunks(filepath, load_reflections=False):
        n_chunks += 1
        n_crystals += len(chunk.crystals)
        if n_chunks >= num_chunks:
            break
    elapsed = time.time() - start
    return elapsed, n_chunks, n_crystals


def benchmark_lazy_mode(filepath: str, num_chunks: int) -> tuple:
    """Benchmark with reflections collected but not accessed."""
    start = time.time()
    n_chunks, n_crystals = 0, 0
    for chunk in iter_chunks(filepath):
        n_chunks += 1
        n_crystals += len(chunk.crystals)
        if n_chunks >= num_chunks:
            break
    elapsed = time.time() - start
    return elapsed, n_chunks, n_crystals


def benchmark_full_access(filepath: str, num_chunks: int) -> tuple:
    """Benchmark with all reflections accessed (triggers parsing)."""
    start = time.time()
    n_chunks, total_refls = 0, 0
    for chunk in iter_chunks(filepath):
        n_chunks += 1
        for c in chunk.crystals:
            total_refls += len(c.reflections)
        if n_chunks >= num_chunks:
            break
    elapsed = time.time() - start
    return elapsed, n_chunks, total_refls


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark crystfel_yasp parser performance'
    )
    parser.add_argument('stream_file', help='Path to the stream file to benchmark')
    parser.add_argument('-n', '--chunks', type=int, default=1000,
                        help='Number of chunks to parse (default: 1000)')
    args = parser.parse_args()

    filepath = args.stream_file
    num_chunks = args.chunks

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1

    file_size_gb = os.path.getsize(filepath) / (1024**3)
    filename = os.path.basename(filepath)

    print(f"File: {filename} ({file_size_gb:.2f} GB)")
    print(f"Numpy available: {HAS_NUMPY}")

    # Count lines in test portion
    print(f"Counting lines in first {num_chunks} chunks...")
    line_count = count_lines_in_chunks(filepath, num_chunks)
    print(f"Test: {line_count:,} lines, {num_chunks} chunks")
    print("-" * 60)

    # Benchmark skip mode
    elapsed, n_chunks, n_crystals = benchmark_skip_mode(filepath, num_chunks)
    lps = line_count / elapsed if elapsed > 0 else 0
    print(f"Skip reflections:  {lps:>10,.0f} lines/sec  ({elapsed:.2f}s)")

    # Benchmark lazy mode
    elapsed, n_chunks, n_crystals = benchmark_lazy_mode(filepath, num_chunks)
    lps = line_count / elapsed if elapsed > 0 else 0
    print(f"Lazy (no access):  {lps:>10,.0f} lines/sec  ({elapsed:.2f}s)")

    # Benchmark full access
    elapsed, n_chunks, total_refls = benchmark_full_access(filepath, num_chunks)
    lps = line_count / elapsed if elapsed > 0 else 0
    print(f"Full access:       {lps:>10,.0f} lines/sec  ({elapsed:.2f}s, {total_refls:,} refls)")

    return 0


if __name__ == '__main__':
    exit(main())
