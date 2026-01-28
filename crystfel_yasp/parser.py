"""
CrystFEL Stream File Parser

A memory-efficient parser for CrystFEL stream files (.stream).
Provides both streaming iteration and full-file loading APIs.

Usage:
    # Memory-efficient iteration
    for chunk in iter_chunks('file.stream'):
        print(chunk.filename, chunk.serial)

    # Load all chunks (for smaller files)
    stream = parse_stream('file.stream')
    print(f"Parsed {len(stream.chunks)} chunks")
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
import re

# Optional numpy for faster reflection parsing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Section markers
CHUNK_START = "----- Begin chunk -----"
CHUNK_END = "----- End chunk -----"
PEAK_START = "Peaks from peak search"
PEAK_END = "End of peak list"
CRYSTAL_START = "--- Begin crystal"
CRYSTAL_END = "--- End crystal"
REFLECTION_START = "Reflections measured after indexing"
REFLECTION_END = "End of reflections"
GEOM_START = "----- Begin geometry file -----"
GEOM_END = "----- End geometry file -----"
CELL_START = "----- Begin unit cell -----"
CELL_END = "----- End unit cell -----"


@dataclass
class Peak:
    """A peak from peak search."""
    fs: float          # fast-scan (pixels)
    ss: float          # slow-scan (pixels)
    resolution: float  # 1/d (nm^-1)
    intensity: float
    panel: str


@dataclass
class Reflection:
    """A measured reflection after indexing."""
    h: int
    k: int
    l: int
    intensity: float
    sigma: float
    peak: float
    background: float
    fs: float
    ss: float
    panel: str


@dataclass
class UnitCell:
    """Unit cell parameters."""
    a: float = 0.0           # nm
    b: float = 0.0           # nm
    c: float = 0.0           # nm
    alpha: float = 0.0       # degrees
    beta: float = 0.0        # degrees
    gamma: float = 0.0       # degrees
    lattice_type: str = ""
    centering: str = ""
    unique_axis: str = ""


@dataclass
class Crystal:
    """Crystal indexing result."""
    cell: UnitCell = field(default_factory=UnitCell)
    astar: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # nm^-1
    bstar: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # nm^-1
    cstar: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # nm^-1
    profile_radius: float = 0.0        # nm^-1
    resolution_limit: float = 0.0      # nm^-1
    num_reflections: int = 0
    num_saturated: int = 0
    num_implausible: int = 0
    det_shift_x: float = 0.0           # mm
    det_shift_y: float = 0.0           # mm
    # Lazy reflection parsing - store raw lines, parse on-demand
    _raw_reflection_lines: List[str] = field(default_factory=list, repr=False)
    _reflections_parsed: bool = field(default=False, repr=False)
    _reflections_cache: List[Reflection] = field(default_factory=list, repr=False)

    @property
    def reflections(self) -> List[Reflection]:
        """Reflections are parsed lazily on first access."""
        if not self._reflections_parsed:
            self._reflections_cache = _parse_reflection_lines(self._raw_reflection_lines)
            self._reflections_parsed = True
            self._raw_reflection_lines = []  # Free memory
        return self._reflections_cache

    @reflections.setter
    def reflections(self, value: List[Reflection]):
        """Allow direct assignment of reflections."""
        self._reflections_cache = value
        self._reflections_parsed = True
        self._raw_reflection_lines = []


@dataclass
class Chunk:
    """A single image/event chunk from the stream."""
    filename: str = ""
    event: str = ""
    serial: int = 0
    hit: bool = False
    indexed_by: str = ""
    n_indexing_tries: int = 0
    photon_energy_eV: float = 0.0
    beam_divergence: float = 0.0       # rad
    beam_bandwidth: float = 0.0        # fraction
    average_camera_length: float = 0.0 # m
    num_peaks: int = 0
    peak_resolution: float = 0.0       # nm^-1
    peaks: List[Peak] = field(default_factory=list)
    crystals: List[Crystal] = field(default_factory=list)


@dataclass
class StreamFile:
    """Container for all chunks in a stream file."""
    chunks: List[Chunk] = field(default_factory=list)


def _parse_peaks(lines: Iterator[str]) -> List[Peak]:
    """Parse peak list section until end marker."""
    peaks = []
    # Skip header line
    try:
        next(lines)
    except StopIteration:
        return peaks

    for line in lines:
        line = line.strip()
        if line == PEAK_END:
            break
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 5:
            try:
                peak = Peak(
                    fs=float(parts[0]),
                    ss=float(parts[1]),
                    resolution=float(parts[2]),
                    intensity=float(parts[3]),
                    panel=parts[4]
                )
                peaks.append(peak)
            except (ValueError, IndexError):
                continue

    return peaks


def _collect_reflection_lines(lines: Iterator[str]) -> List[str]:
    """Collect raw reflection lines until end marker (for lazy parsing)."""
    raw_lines = []
    # Skip header line
    try:
        next(lines)
    except StopIteration:
        return raw_lines

    for line in lines:
        stripped = line.strip()
        if stripped == REFLECTION_END:
            break
        if stripped:
            raw_lines.append(stripped)

    return raw_lines


def _parse_reflection_lines(lines: List[str]) -> List[Reflection]:
    """Parse collected reflection lines into Reflection objects."""
    if not lines:
        return []

    if HAS_NUMPY:
        return _parse_reflections_numpy(lines)
    else:
        return _parse_reflections_python(lines)


def _parse_reflections_numpy(lines: List[str]) -> List[Reflection]:
    """Parse reflections using numpy for faster number conversion."""
    from io import StringIO

    # Extract numeric columns with numpy
    text = '\n'.join(lines)
    try:
        arr = np.loadtxt(StringIO(text), usecols=range(9))
    except ValueError:
        # Fall back to Python parser if numpy fails
        return _parse_reflections_python(lines)

    # Handle single reflection case (numpy returns 1D array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Extract panels separately
    panels = [line.split()[9] for line in lines]

    # Build Reflection objects
    reflections = []
    for i, row in enumerate(arr):
        reflections.append(Reflection(
            h=int(row[0]),
            k=int(row[1]),
            l=int(row[2]),
            intensity=row[3],
            sigma=row[4],
            peak=row[5],
            background=row[6],
            fs=row[7],
            ss=row[8],
            panel=panels[i]
        ))
    return reflections


def _parse_reflections_python(lines: List[str]) -> List[Reflection]:
    """Parse reflections using pure Python (fallback when numpy unavailable)."""
    reflections = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 10:
            try:
                refl = Reflection(
                    h=int(parts[0]),
                    k=int(parts[1]),
                    l=int(parts[2]),
                    intensity=float(parts[3]),
                    sigma=float(parts[4]),
                    peak=float(parts[5]),
                    background=float(parts[6]),
                    fs=float(parts[7]),
                    ss=float(parts[8]),
                    panel=parts[9]
                )
                reflections.append(refl)
            except (ValueError, IndexError):
                continue
    return reflections


def _parse_crystal(lines: Iterator[str], load_reflections: bool = True) -> Crystal:
    """Parse crystal section until end marker.

    Args:
        lines: Iterator of lines to parse
        load_reflections: If True, collect reflection lines for lazy parsing.
                         If False, skip reflection section entirely (faster).
    """
    crystal = Crystal()
    cell = UnitCell()

    # Regex patterns
    cell_params_re = re.compile(
        r'Cell parameters\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+nm,\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+deg'
    )
    star_re = re.compile(r'([abc])star\s*=\s*([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)')
    det_shift_re = re.compile(r'predict_refine/det_shift\s+x\s*=\s*([+-]?[\d.]+)\s+y\s*=\s*([+-]?[\d.]+)')
    resolution_re = re.compile(r'diffraction_resolution_limit\s*=\s*([\d.]+)\s*nm\^-1')
    profile_re = re.compile(r'profile_radius\s*=\s*([\d.e+-]+)\s*nm\^-1')

    for line in lines:
        line = line.strip()
        if line == CRYSTAL_END:
            break

        if line.startswith(REFLECTION_START):
            if load_reflections:
                crystal._raw_reflection_lines = _collect_reflection_lines(lines)
            else:
                # Skip reflection section entirely
                for skip_line in lines:
                    if skip_line.strip() == REFLECTION_END:
                        break
            continue

        # Cell parameters
        m = cell_params_re.match(line)
        if m:
            cell.a = float(m.group(1))
            cell.b = float(m.group(2))
            cell.c = float(m.group(3))
            cell.alpha = float(m.group(4))
            cell.beta = float(m.group(5))
            cell.gamma = float(m.group(6))
            continue

        # Reciprocal lattice vectors
        m = star_re.match(line)
        if m:
            vec = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            if m.group(1) == 'a':
                crystal.astar = vec
            elif m.group(1) == 'b':
                crystal.bstar = vec
            elif m.group(1) == 'c':
                crystal.cstar = vec
            continue

        # Lattice type
        if line.startswith('lattice_type = '):
            cell.lattice_type = line.split('=', 1)[1].strip()
            continue

        # Centering
        if line.startswith('centering = '):
            cell.centering = line.split('=', 1)[1].strip()
            continue

        # Unique axis
        if line.startswith('unique_axis = '):
            cell.unique_axis = line.split('=', 1)[1].strip()
            continue

        # Profile radius
        m = profile_re.match(line)
        if m:
            crystal.profile_radius = float(m.group(1))
            continue

        # Detector shift
        m = det_shift_re.search(line)
        if m:
            crystal.det_shift_x = float(m.group(1))
            crystal.det_shift_y = float(m.group(2))
            continue

        # Resolution limit
        m = resolution_re.match(line)
        if m:
            crystal.resolution_limit = float(m.group(1))
            continue

        # Reflection counts
        if line.startswith('num_reflections = '):
            try:
                crystal.num_reflections = int(line.split('=', 1)[1].strip())
            except ValueError:
                pass
            continue

        if line.startswith('num_saturated_reflections = '):
            try:
                crystal.num_saturated = int(line.split('=', 1)[1].strip())
            except ValueError:
                pass
            continue

        if line.startswith('num_implausible_reflections = '):
            try:
                crystal.num_implausible = int(line.split('=', 1)[1].strip())
            except ValueError:
                pass
            continue

    crystal.cell = cell
    return crystal


def _parse_chunk(lines: Iterator[str], load_reflections: bool = True) -> Optional[Chunk]:
    """Parse a single chunk from lines iterator. Assumes chunk start marker already consumed.

    Args:
        lines: Iterator of lines to parse
        load_reflections: If True, collect reflection lines for lazy parsing.
                         If False, skip reflection section entirely (faster).
    """
    chunk = Chunk()

    for line in lines:
        line = line.strip()

        if line == CHUNK_END:
            return chunk

        # Filename
        if line.startswith('Image filename: '):
            chunk.filename = line[16:]
            continue

        # Event
        if line.startswith('Event: '):
            chunk.event = line[7:]
            continue

        # Serial number
        if line.startswith('Image serial number: '):
            try:
                chunk.serial = int(line[21:])
            except ValueError:
                pass
            continue

        # Hit flag
        if line.startswith('hit = '):
            chunk.hit = line[6:].strip() == '1'
            continue

        # Indexed by
        if line.startswith('indexed_by = '):
            chunk.indexed_by = line[13:]
            continue

        # Number of indexing tries
        if line.startswith('n_indexing_tries = '):
            try:
                chunk.n_indexing_tries = int(line[19:])
            except ValueError:
                pass
            continue

        # Photon energy
        if line.startswith('photon_energy_eV = '):
            try:
                chunk.photon_energy_eV = float(line[19:])
            except ValueError:
                pass
            continue

        # Beam divergence
        if line.startswith('beam_divergence = '):
            try:
                val = line[18:].replace(' rad', '')
                chunk.beam_divergence = float(val)
            except ValueError:
                pass
            continue

        # Beam bandwidth
        if line.startswith('beam_bandwidth = '):
            try:
                val = line[17:].replace(' (fraction)', '')
                chunk.beam_bandwidth = float(val)
            except ValueError:
                pass
            continue

        # Camera length
        if line.startswith('average_camera_length = '):
            try:
                val = line[24:].replace(' m', '')
                chunk.average_camera_length = float(val)
            except ValueError:
                pass
            continue

        # Number of peaks
        if line.startswith('num_peaks = '):
            try:
                chunk.num_peaks = int(line[12:])
            except ValueError:
                pass
            continue

        # Peak resolution
        if line.startswith('peak_resolution = '):
            try:
                # Format: "6.998420 nm^-1 or 1.428894 A"
                val = line[18:].split(' nm^-1')[0]
                chunk.peak_resolution = float(val)
            except (ValueError, IndexError):
                pass
            continue

        # Peak list
        if line == PEAK_START:
            chunk.peaks = _parse_peaks(lines)
            continue

        # Crystal
        if line.startswith(CRYSTAL_START):
            crystal = _parse_crystal(lines, load_reflections)
            chunk.crystals.append(crystal)
            continue

    return None  # Incomplete chunk


def iter_chunks(filepath: str, load_reflections: bool = True) -> Iterator[Chunk]:
    """
    Memory-efficient iterator that yields chunks one at a time.

    Args:
        filepath: Path to the stream file
        load_reflections: If True (default), reflection lines are collected
                         and parsed lazily when crystal.reflections is accessed.
                         If False, reflection sections are skipped entirely
                         (faster for stats-only use cases).

    Yields:
        Chunk objects parsed from the stream

    Example:
        for chunk in iter_chunks('data.stream'):
            if chunk.crystals:
                print(f"Indexed: {chunk.filename}")

        # Fast stats-only mode (no reflection data)
        for chunk in iter_chunks('data.stream', load_reflections=False):
            print(f"Crystals: {len(chunk.crystals)}")
    """
    with open(filepath, 'r') as f:
        lines = iter(f)
        for line in lines:
            if line.strip() == CHUNK_START:
                chunk = _parse_chunk(lines, load_reflections)
                if chunk is not None:
                    yield chunk


def parse_stream(filepath: str, load_reflections: bool = True) -> StreamFile:
    """
    Parse entire stream file into memory.

    Warning: For large files (>1GB), use iter_chunks() instead.

    Args:
        filepath: Path to the stream file
        load_reflections: If True (default), reflection lines are collected
                         and parsed lazily when crystal.reflections is accessed.
                         If False, reflection sections are skipped entirely.

    Returns:
        StreamFile containing all parsed chunks
    """
    stream = StreamFile()
    for chunk in iter_chunks(filepath, load_reflections):
        stream.chunks.append(chunk)
    return stream


def main():
    """CLI entry point for testing."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse CrystFEL stream files and print summary statistics'
    )
    parser.add_argument('stream_file', help='Path to the stream file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print details of first chunk')
    parser.add_argument('-n', '--limit', type=int, default=None,
                        help='Limit number of chunks to parse')
    args = parser.parse_args()

    filepath = args.stream_file

    print(f"Parsing: {filepath}")
    print("-" * 60)

    # Stats counters
    n_chunks = 0
    n_hits = 0
    n_indexed = 0
    total_peaks = 0
    total_crystals = 0
    total_reflections = 0
    first_chunk = None

    for chunk in iter_chunks(filepath):
        if first_chunk is None:
            first_chunk = chunk

        n_chunks += 1
        if chunk.hit:
            n_hits += 1
        if chunk.indexed_by and chunk.indexed_by != 'none':
            n_indexed += 1

        total_peaks += len(chunk.peaks)
        total_crystals += len(chunk.crystals)

        for crystal in chunk.crystals:
            total_reflections += len(crystal.reflections)

        if args.limit and n_chunks >= args.limit:
            print(f"(Stopped after {args.limit} chunks)")
            break

        if n_chunks % 10000 == 0:
            print(f"  Processed {n_chunks} chunks...")

    print()
    print("Summary Statistics:")
    print("-" * 60)
    print(f"Total chunks:       {n_chunks:,}")
    print(f"Hits:               {n_hits:,} ({100*n_hits/n_chunks:.1f}%)" if n_chunks > 0 else "Hits: 0")
    print(f"Indexed:            {n_indexed:,} ({100*n_indexed/n_chunks:.1f}%)" if n_chunks > 0 else "Indexed: 0")
    print(f"Total peaks:        {total_peaks:,}")
    print(f"Total crystals:     {total_crystals:,}")
    print(f"Total reflections:  {total_reflections:,}")

    if args.verbose and first_chunk:
        print()
        print("First Chunk Details:")
        print("-" * 60)
        print(f"  Filename:         {first_chunk.filename}")
        print(f"  Event:            {first_chunk.event}")
        print(f"  Serial:           {first_chunk.serial}")
        print(f"  Hit:              {first_chunk.hit}")
        print(f"  Indexed by:       {first_chunk.indexed_by}")
        print(f"  Photon energy:    {first_chunk.photon_energy_eV:.2f} eV")
        print(f"  Camera length:    {first_chunk.average_camera_length:.4f} m")
        print(f"  Num peaks:        {len(first_chunk.peaks)}")
        print(f"  Num crystals:     {len(first_chunk.crystals)}")

        if first_chunk.peaks:
            print()
            print("  First 3 peaks:")
            for i, peak in enumerate(first_chunk.peaks[:3]):
                print(f"    [{i}] fs={peak.fs:.1f}, ss={peak.ss:.1f}, "
                      f"I={peak.intensity:.1f}, panel={peak.panel}")

        if first_chunk.crystals:
            crystal = first_chunk.crystals[0]
            print()
            print("  First crystal:")
            print(f"    Cell: {crystal.cell.a:.3f} {crystal.cell.b:.3f} {crystal.cell.c:.3f} nm")
            print(f"    Angles: {crystal.cell.alpha:.2f} {crystal.cell.beta:.2f} {crystal.cell.gamma:.2f} deg")
            print(f"    Lattice type: {crystal.cell.lattice_type}")
            print(f"    Centering: {crystal.cell.centering}")
            print(f"    Resolution limit: {crystal.resolution_limit:.2f} nm^-1")
            print(f"    Num reflections: {len(crystal.reflections)}")


if __name__ == '__main__':
    main()
