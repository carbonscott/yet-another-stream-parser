#!/usr/bin/env python3
"""
Benchmark: Shared Memory Parallel Parsing

Compares sequential vs shared-memory parallel chunk parsing at various worker
counts. Verifies correctness (numeric + panel string match) and prints a
speedup table.

Usage:
    export UV_CACHE_DIR=/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/.UV_CACHE
    uv run --python 3.11 --with numpy python benchmarks/benchmark_parallel_parse.py
    uv run --python 3.11 --with numpy python benchmarks/benchmark_parallel_parse.py --chunks 5000
    uv run --python 3.11 --with numpy python benchmarks/benchmark_parallel_parse.py --workers 2 4 8
"""

import argparse
import os
import sys
import time

import numpy as np

from crystfel_yasp.indexing import StreamIndex
from crystfel_yasp.parallel import parse_chunks_shared, parse_chunks_sequential


def verify_correctness(seq_arr, seq_panels, par_arr, par_panels):
    """Compare sequential vs parallel results and return a status string."""
    # Check numeric array
    if seq_arr.shape != par_arr.shape:
        return (False,
                f"FAIL: shape mismatch {seq_arr.shape} vs {par_arr.shape}")

    if not np.allclose(seq_arr, par_arr, equal_nan=True):
        mismatches = np.where(~np.isclose(seq_arr, par_arr, equal_nan=True))
        n_diff = len(mismatches[0])
        return (False, f"FAIL: {n_diff} numeric values differ")

    # Check panel strings
    if seq_panels != par_panels:
        n_panels = min(len(seq_panels), len(par_panels))
        n_diff = sum(1 for a, b in zip(seq_panels, par_panels) if a != b)
        return (False,
                f"FAIL: panels differ (len {len(seq_panels)} vs {len(par_panels)}, "
                f"{n_diff} mismatches in first {n_panels})")

    return (True, "PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark shared-memory parallel chunk parsing"
    )
    parser.add_argument(
        "--stream",
        default="/sdf/data/lcls/ds/mfx/mfxl1025422/results/btx/streams/ox112it2.stream",
        help="Path to the .stream file",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Path to pre-built .idx file (default: <stream>.idx)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1000,
        help="Number of chunks to parse (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Worker counts to benchmark (default: 2 4 8 16)",
    )
    args = parser.parse_args()

    stream_path = args.stream
    idx_path = args.index or stream_path + ".idx"

    # --- Load or build index ---
    if os.path.exists(idx_path):
        print(f"Loading index: {idx_path}")
        index = StreamIndex.load(idx_path)
    else:
        print(f"Building index: {stream_path}")
        index = StreamIndex.build(stream_path)
        index.save(idx_path)

    total_entries = len(index.entries)
    print(f"Index: {total_entries:,} entries")

    # Select entries with reflections for a meaningful test
    entries_with_refls = [e for e in index.entries if e.num_reflections > 0]
    n_test = min(args.chunks, len(entries_with_refls))
    test_entries = entries_with_refls[:n_test]
    total_refls = sum(e.num_reflections for e in test_entries)

    print(f"Test set: {n_test:,} chunks, {total_refls:,} expected reflections")
    print("=" * 70)

    # --- Sequential baseline ---
    print("\nSequential parse...")
    t0 = time.perf_counter()
    seq_arr, seq_panels, seq_meta = parse_chunks_sequential(index, test_entries)
    t_seq = time.perf_counter() - t0
    print(f"  Time: {t_seq:.3f}s  "
          f"({total_refls / t_seq:,.0f} refls/s)  "
          f"array shape: {seq_arr.shape}")

    # --- Shared-memory parallel at various worker counts ---
    results = {}
    for nw in args.workers:
        if nw > n_test:
            print(f"\nSkipping {nw} workers (more workers than chunks)")
            continue

        print(f"\nShared-memory parallel ({nw} workers)...")
        t0 = time.perf_counter()
        par_arr, par_panels, par_meta = parse_chunks_shared(
            index, test_entries, num_workers=nw
        )
        t_par = time.perf_counter() - t0
        results[nw] = t_par

        # Correctness check
        ok, status = verify_correctness(seq_arr, seq_panels, par_arr, par_panels)
        speedup = t_seq / t_par if t_par > 0 else float("inf")
        print(f"  Time: {t_par:.3f}s  speedup: {speedup:.2f}x  correctness: {status}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"  Chunks: {n_test:,}   Reflections: {total_refls:,}")
    print(f"  Stream: {os.path.basename(stream_path)}")
    print("-" * 70)
    print(f"{'Workers':>8}  {'Time (s)':>10}  {'Speedup':>8}  {'Refls/s':>14}")
    print("-" * 70)
    rps_seq = total_refls / t_seq if t_seq > 0 else 0
    print(f"{'seq':>8}  {t_seq:>10.3f}  {'1.00x':>8}  {rps_seq:>14,.0f}")
    for nw in sorted(results):
        t = results[nw]
        speedup = t_seq / t if t > 0 else float("inf")
        rps = total_refls / t if t > 0 else 0
        print(f"{nw:>8}  {t:>10.3f}  {speedup:>7.2f}x  {rps:>14,.0f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
