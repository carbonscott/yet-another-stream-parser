#!/usr/bin/env python3
"""
Stream File Indexing Module

Builds a persistent index for CrystFEL stream files enabling O(1) seeking
to any chunk and metadata queries without full parsing.

Usage:
    # Build and save index
    uv run python indexing.py build file.stream

    # Show index summary
    uv run python indexing.py info file.stream.idx

    # Seek to a specific chunk
    uv run python indexing.py get file.stream.idx --position 1000
    uv run python indexing.py get file.stream.idx --serial 12345

    # Filter entries
    uv run python indexing.py filter file.stream.idx --hit --min-crystals 1
"""

import json
import mmap
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_parser import (
    CHUNK_END,
    CHUNK_START,
    CRYSTAL_START,
    Chunk,
    _parse_chunk,
)

# Byte-encoded markers for mmap scanning
CHUNK_START_B = CHUNK_START.encode()
CHUNK_END_B = CHUNK_END.encode()
CRYSTAL_START_B = CRYSTAL_START.encode()


@dataclass
class ChunkEntry:
    """Lightweight metadata for one chunk, extracted during indexing."""

    position: int  # 0-indexed order in file
    start_offset: int  # byte offset of chunk start
    end_offset: int  # byte offset of chunk end
    filename: str
    event: str
    serial: int
    hit: bool
    indexed_by: str
    num_peaks: int
    num_crystals: int
    num_reflections: int


class StreamIndex:
    """Persistent index for a CrystFEL stream file.

    Enables O(1) seeking to any chunk and metadata queries without full parsing.
    """

    def __init__(self):
        self.entries: List[ChunkEntry] = []
        self.stream_path: str = ""
        self.file_size: int = 0
        self.created_at: str = ""
        self._serial_map: Optional[Dict[int, int]] = None

    @classmethod
    def build(cls, stream_path: str, progress_interval: int = 10000) -> "StreamIndex":
        """Scan a stream file and build an index of all chunks.

        Uses memory-mapped I/O with C-speed byte scanning to jump between
        chunk markers. Only decodes the ~6 metadata values per chunk.
        """
        index = cls()
        index.stream_path = os.path.abspath(stream_path)
        index.file_size = os.path.getsize(stream_path)
        index.created_at = datetime.now(timezone.utc).isoformat()

        position = 0

        with open(stream_path, "rb") as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = 0
                buf_len = len(buf)
                while pos < buf_len:
                    start = buf.find(CHUNK_START_B, pos)
                    if start == -1:
                        break

                    end_marker = buf.find(CHUNK_END_B, start + len(CHUNK_START_B))
                    if end_marker == -1:
                        break

                    # end_offset includes the CHUNK_END line and its newline
                    nl = buf.find(b"\n", end_marker + len(CHUNK_END_B))
                    end_offset = (nl + 1) if nl != -1 else buf_len

                    entry = _scan_chunk_metadata_mmap(
                        buf, position, start, end_offset
                    )
                    index.entries.append(entry)
                    position += 1

                    if progress_interval and position % progress_interval == 0:
                        print(f"  Scanned {position:,} chunks...")

                    pos = end_offset
            finally:
                buf.close()

        index._serial_map = {e.serial: i for i, e in enumerate(index.entries)}
        return index

    def save(self, path: str) -> None:
        """Save index to a compact JSON file with string deduplication."""
        # Build string tables for repeated values
        filenames = sorted(set(e.filename for e in self.entries))
        indexed_by_vals = sorted(set(e.indexed_by for e in self.entries))
        fn_to_idx = {s: i for i, s in enumerate(filenames)}
        ib_to_idx = {s: i for i, s in enumerate(indexed_by_vals)}

        # Entries as compact arrays (position omitted — equals array index)
        # [start_offset, end_offset, filename_idx, event, serial, hit,
        #  indexed_by_idx, num_peaks, num_crystals, num_reflections]
        compact = []
        for e in self.entries:
            compact.append([
                e.start_offset,
                e.end_offset,
                fn_to_idx[e.filename],
                e.event,
                e.serial,
                e.hit,
                ib_to_idx[e.indexed_by],
                e.num_peaks,
                e.num_crystals,
                e.num_reflections,
            ])

        data = {
            "stream_path": self.stream_path,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "num_entries": len(self.entries),
            "string_tables": {
                "filenames": filenames,
                "indexed_by": indexed_by_vals,
            },
            "entries": compact,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved index ({len(self.entries):,} entries) to {path}")

    @classmethod
    def load(cls, path: str) -> "StreamIndex":
        """Load index from a compact JSON file with string tables."""
        with open(path, "r") as f:
            data = json.load(f)

        index = cls()
        index.stream_path = data["stream_path"]
        index.file_size = data["file_size"]
        index.created_at = data["created_at"]

        filenames = data["string_tables"]["filenames"]
        indexed_by_vals = data["string_tables"]["indexed_by"]

        entries = []
        for i, arr in enumerate(data["entries"]):
            entries.append(ChunkEntry(
                position=i,
                start_offset=arr[0],
                end_offset=arr[1],
                filename=filenames[arr[2]],
                event=arr[3],
                serial=arr[4],
                hit=arr[5],
                indexed_by=indexed_by_vals[arr[6]],
                num_peaks=arr[7],
                num_crystals=arr[8],
                num_reflections=arr[9] if len(arr) > 9 else 0,
            ))
        index.entries = entries

        index._serial_map = {e.serial: i for i, e in enumerate(index.entries)}
        return index

    def get_chunk(self, position: int) -> Chunk:
        """Seek to a chunk by position index and parse it fully."""
        if position < 0 or position >= len(self.entries):
            raise IndexError(
                f"Position {position} out of range (0-{len(self.entries) - 1})"
            )
        entry = self.entries[position]
        return self._read_and_parse(entry)

    def get_chunk_by_serial(self, serial: int) -> Chunk:
        """Find and parse a chunk by its serial number (O(1) dict lookup)."""
        if self._serial_map is None:
            self._serial_map = {e.serial: i for i, e in enumerate(self.entries)}
        idx = self._serial_map.get(serial)
        if idx is None:
            raise KeyError(f"No chunk with serial {serial}")
        return self.get_chunk(idx)

    def _read_and_parse(self, entry: ChunkEntry) -> Chunk:
        """Read raw bytes for a chunk and parse it with _parse_chunk."""
        with open(self.stream_path, "rb") as f:
            f.seek(entry.start_offset)
            raw = f.read(entry.end_offset - entry.start_offset)

        text = raw.decode("utf-8", errors="replace")
        lines_list = text.splitlines()

        # _parse_chunk expects the CHUNK_START marker already consumed
        line_iter = iter(lines_list)
        for line in line_iter:
            if line.strip() == CHUNK_START:
                break

        chunk = _parse_chunk(line_iter)
        if chunk is None:
            raise RuntimeError(
                f"Failed to parse chunk at position {entry.position}"
            )
        return chunk

    def filter(
        self,
        hit: Optional[bool] = None,
        min_crystals: Optional[int] = None,
        indexed_by: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> List[ChunkEntry]:
        """Filter entries by metadata criteria."""
        results = self.entries
        if hit is not None:
            results = [e for e in results if e.hit == hit]
        if min_crystals is not None:
            results = [e for e in results if e.num_crystals >= min_crystals]
        if indexed_by is not None:
            results = [e for e in results if indexed_by in e.indexed_by]
        if filename is not None:
            results = [e for e in results if filename in e.filename]
        return results

    def summary(self) -> dict:
        """Return summary statistics about the index."""
        n = len(self.entries)
        if n == 0:
            return {"num_chunks": 0}

        n_hits = sum(1 for e in self.entries if e.hit)
        n_indexed = sum(
            1 for e in self.entries if e.indexed_by and e.indexed_by != "none"
        )
        n_crystals = sum(e.num_crystals for e in self.entries)
        total_peaks = sum(e.num_peaks for e in self.entries)
        total_reflections = sum(e.num_reflections for e in self.entries)

        return {
            "stream_path": self.stream_path,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "num_chunks": n,
            "num_hits": n_hits,
            "hit_rate": n_hits / n,
            "num_indexed": n_indexed,
            "index_rate": n_indexed / n,
            "total_crystals": n_crystals,
            "total_peaks": total_peaks,
            "total_reflections": total_reflections,
        }


def _extract_line_value(buf, marker: bytes, start: int, end: int) -> str:
    """Find a marker in buf[start:end] and return the value after it.

    Locates the marker, then extracts and decodes the text between the end
    of the marker and the next newline. Returns "" if marker is not found.
    """
    pos = buf.find(marker, start, end)
    if pos == -1:
        return ""
    val_start = pos + len(marker)
    val_end = buf.find(b"\n", val_start, end)
    if val_end == -1:
        val_end = end
    return buf[val_start:val_end].decode("utf-8", errors="replace").strip()


def _scan_chunk_metadata_mmap(
    buf, position: int, chunk_start: int, chunk_end: int
) -> ChunkEntry:
    """Extract chunk metadata from an mmap buffer region.

    Uses mmap.find() to locate each metadata field directly in the byte
    buffer — no per-line iteration or decoding of irrelevant lines.
    """
    filename = _extract_line_value(buf, b"Image filename: ", chunk_start, chunk_end)
    event = _extract_line_value(buf, b"Event: ", chunk_start, chunk_end)

    serial_str = _extract_line_value(
        buf, b"Image serial number: ", chunk_start, chunk_end
    )
    try:
        serial = int(serial_str)
    except (ValueError, TypeError):
        serial = 0

    hit_str = _extract_line_value(buf, b"hit = ", chunk_start, chunk_end)
    hit = hit_str == "1"

    indexed_by = _extract_line_value(buf, b"indexed_by = ", chunk_start, chunk_end)

    num_peaks_str = _extract_line_value(buf, b"num_peaks = ", chunk_start, chunk_end)
    try:
        num_peaks = int(num_peaks_str)
    except (ValueError, TypeError):
        num_peaks = 0

    # Count crystals and sum num_reflections in a single pass.
    # We record each crystal start position, then search for the
    # "num_reflections = " marker only from those positions forward
    # (it appears once per crystal, shortly after the crystal header).
    num_crystals = 0
    num_reflections = 0
    refl_marker = b"num_reflections = "
    search_pos = chunk_start
    while True:
        crystal_pos = buf.find(CRYSTAL_START_B, search_pos, chunk_end)
        if crystal_pos == -1:
            break
        num_crystals += 1
        # Search for num_reflections starting from this crystal header
        found = buf.find(refl_marker, crystal_pos, chunk_end)
        if found != -1:
            val_start = found + len(refl_marker)
            val_end = buf.find(b"\n", val_start, chunk_end)
            if val_end == -1:
                val_end = chunk_end
            try:
                num_reflections += int(buf[val_start:val_end].strip())
            except (ValueError, TypeError):
                pass
        search_pos = crystal_pos + len(CRYSTAL_START_B)

    return ChunkEntry(
        position=position,
        start_offset=chunk_start,
        end_offset=chunk_end,
        filename=filename,
        event=event,
        serial=serial,
        hit=hit,
        indexed_by=indexed_by,
        num_peaks=num_peaks,
        num_crystals=num_crystals,
        num_reflections=num_reflections,
    )


def _print_chunk(chunk: Chunk) -> None:
    """Print a chunk's full details."""
    print(f"Filename:       {chunk.filename}")
    print(f"Event:          {chunk.event}")
    print(f"Serial:         {chunk.serial}")
    print(f"Hit:            {chunk.hit}")
    print(f"Indexed by:     {chunk.indexed_by}")
    print(f"Photon energy:  {chunk.photon_energy_eV:.2f} eV")
    print(f"Num peaks:      {chunk.num_peaks}")
    print(f"Peaks:          {len(chunk.peaks)}")
    print(f"Crystals:       {len(chunk.crystals)}")
    for i, crystal in enumerate(chunk.crystals):
        print(f"  Crystal {i}:")
        print(
            f"    Cell: {crystal.cell.a:.3f} {crystal.cell.b:.3f} "
            f"{crystal.cell.c:.3f} nm"
        )
        print(
            f"    Angles: {crystal.cell.alpha:.2f} {crystal.cell.beta:.2f} "
            f"{crystal.cell.gamma:.2f} deg"
        )
        print(f"    Reflections: {len(crystal.reflections)}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream file indexing — build, query, and filter chunk metadata"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = subparsers.add_parser("build", help="Build index from a stream file")
    build_p.add_argument("stream_file", help="Path to the .stream file")
    build_p.add_argument(
        "-o", "--output", help="Output index file path (default: <stream>.idx)"
    )

    # info
    info_p = subparsers.add_parser("info", help="Show index summary")
    info_p.add_argument("index_file", help="Path to the .idx file")

    # get
    get_p = subparsers.add_parser("get", help="Seek to and print a specific chunk")
    get_p.add_argument("index_file", help="Path to the .idx file")
    get_p.add_argument("--position", type=int, help="Chunk position (0-indexed)")
    get_p.add_argument("--serial", type=int, help="Chunk serial number")

    # filter
    filter_p = subparsers.add_parser("filter", help="Filter index entries")
    filter_p.add_argument("index_file", help="Path to the .idx file")
    filter_p.add_argument("--hit", action="store_true", help="Only hit chunks")
    filter_p.add_argument(
        "--min-crystals", type=int, help="Minimum number of crystals"
    )
    filter_p.add_argument(
        "--indexed-by", type=str, help="Filter by indexing method (substring)"
    )
    filter_p.add_argument(
        "--filename", type=str, help="Filter by filename (substring)"
    )
    filter_p.add_argument(
        "--limit", type=int, default=20, help="Max entries to print (default 20)"
    )

    args = parser.parse_args()

    if args.command == "build":
        print(f"Building index for: {args.stream_file}")
        index = StreamIndex.build(args.stream_file)
        out_path = args.output or args.stream_file + ".idx"
        index.save(out_path)

        stats = index.summary()
        print(f"\n  Chunks:  {stats['num_chunks']:,}")
        print(f"  Hits:    {stats['num_hits']:,} ({stats['hit_rate']:.1%})")
        print(f"  Indexed: {stats['num_indexed']:,} ({stats['index_rate']:.1%})")

    elif args.command == "info":
        index = StreamIndex.load(args.index_file)
        stats = index.summary()
        print(f"Stream:     {stats['stream_path']}")
        print(f"File size:  {stats['file_size']:,} bytes")
        print(f"Created:    {stats['created_at']}")
        print(f"Chunks:     {stats['num_chunks']:,}")
        print(f"Hits:       {stats['num_hits']:,} ({stats['hit_rate']:.1%})")
        print(f"Indexed:    {stats['num_indexed']:,} ({stats['index_rate']:.1%})")
        print(f"Crystals:   {stats['total_crystals']:,}")
        print(f"Peaks:      {stats['total_peaks']:,}")
        print(f"Reflections:{stats['total_reflections']:,}")

    elif args.command == "get":
        index = StreamIndex.load(args.index_file)

        if args.position is not None:
            chunk = index.get_chunk(args.position)
        elif args.serial is not None:
            chunk = index.get_chunk_by_serial(args.serial)
        else:
            print("Error: specify --position or --serial", file=sys.stderr)
            sys.exit(1)

        _print_chunk(chunk)

    elif args.command == "filter":
        index = StreamIndex.load(args.index_file)
        hit_flag = True if args.hit else None
        entries = index.filter(
            hit=hit_flag,
            min_crystals=args.min_crystals,
            indexed_by=args.indexed_by,
            filename=args.filename,
        )
        print(f"Matched {len(entries):,} entries")
        for entry in entries[: args.limit]:
            print(
                f"  pos={entry.position} serial={entry.serial} "
                f"hit={entry.hit} crystals={entry.num_crystals} "
                f"peaks={entry.num_peaks} indexed_by={entry.indexed_by} "
                f"file={entry.filename}"
            )
        if len(entries) > args.limit:
            print(f"  ... ({len(entries) - args.limit} more)")


if __name__ == "__main__":
    main()
