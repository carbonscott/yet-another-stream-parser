# Stream File Indexing — Usage Guide

`scripts/indexing.py` builds a persistent index for CrystFEL stream files.
The index enables instant metadata queries and direct seeking to any chunk
without reading the entire file.

## Building an Index

```bash
# Build index (writes .idx file next to the stream)
uv run --with numpy python scripts/indexing.py build /path/to/file.stream

# Custom output path
uv run --with numpy python scripts/indexing.py build /path/to/file.stream -o /path/to/output.idx
```

## CLI Commands

```bash
# Summary stats
uv run --with numpy python scripts/indexing.py info file.stream.idx

# Fetch a specific chunk
uv run --with numpy python scripts/indexing.py get file.stream.idx --position 1000
uv run --with numpy python scripts/indexing.py get file.stream.idx --serial 12345

# Filter entries
uv run --with numpy python scripts/indexing.py filter file.stream.idx --hit
uv run --with numpy python scripts/indexing.py filter file.stream.idx --min-crystals 2
uv run --with numpy python scripts/indexing.py filter file.stream.idx --indexed-by mosflm
uv run --with numpy python scripts/indexing.py filter file.stream.idx --filename r0084
uv run --with numpy python scripts/indexing.py filter file.stream.idx --hit --min-crystals 1 --limit 50
```

## Python API

### Load an index

For interactive use (e.g., ipython), add the scripts directory to the import
path with `sys`:

```python
import sys
sys.path.insert(0, "scripts")
from indexing import StreamIndex

idx = StreamIndex.load("file.stream.idx")
```

### Quick stats (no stream file access)

```python
stats = idx.summary()
print(f"Chunks:  {stats['num_chunks']:,}")
print(f"Hits:    {stats['num_hits']:,} ({stats['hit_rate']:.1%})")
print(f"Indexed: {stats['num_indexed']:,} ({stats['index_rate']:.1%})")
print(f"Crystals: {stats['total_crystals']:,}")
```

### Filter by metadata (no stream file access)

```python
# Chunks with multiple crystals
multi = idx.filter(min_crystals=2)
print(f"{len(multi)} chunks with >=2 crystals")

# Chunks indexed by a specific method
mosflm = idx.filter(indexed_by="mosflm")

# Combine filters
hits_with_crystals = idx.filter(hit=True, min_crystals=1)

# Filter by filename substring (e.g. a specific run)
run84 = idx.filter(filename="r0084")
```

### Seek to a specific chunk (reads only that chunk from stream)

```python
# By position (0-indexed order in file)
chunk = idx.get_chunk(1000)

# By serial number (O(1) dict lookup)
chunk = idx.get_chunk_by_serial(12345)

# chunk is a fully parsed Chunk object from stream_parser.py
print(chunk.filename, chunk.serial, chunk.hit)
print(f"Peaks: {len(chunk.peaks)}")
for crystal in chunk.crystals:
    print(f"  Cell: {crystal.cell.a:.2f} {crystal.cell.b:.2f} {crystal.cell.c:.2f} nm")
    print(f"  Reflections: {len(crystal.reflections)}")
```

### Access individual reflection data

Each `Reflection` has 10 fields: `h`, `k`, `l` (Miller indices), `intensity`,
`sigma`, `peak`, `background`, `fs`, `ss` (detector position), and `panel`.

```python
chunk = idx.get_chunk(1000)

for crystal in chunk.crystals:
    for refl in crystal.reflections:
        print(f"  {refl.h} {refl.k} {refl.l}  I={refl.intensity:.1f}  "
              f"sigma={refl.sigma:.1f}  panel={refl.panel}")
```

## Use Cases

### Compare indexing methods

```python
from collections import Counter

method_counts = Counter(e.indexed_by for e in idx.entries)
for method, count in method_counts.most_common():
    print(f"  {method}: {count:,} ({count/len(idx.entries):.1%})")
```

### Crystal count distribution

```python
from collections import Counter

dist = Counter(e.num_crystals for e in idx.entries)
for n in sorted(dist):
    print(f"  {n} crystals: {dist[n]:,} chunks")
```

### Per-file indexing rate

```python
from collections import defaultdict

by_file = defaultdict(lambda: {"total": 0, "indexed": 0})
for e in idx.entries:
    by_file[e.filename]["total"] += 1
    if e.num_crystals > 0:
        by_file[e.filename]["indexed"] += 1

for fn, c in sorted(by_file.items()):
    rate = c["indexed"] / c["total"]
    print(f"  {rate:5.1%}  {c['indexed']:>5}/{c['total']:<5}  {fn}")
```

### Parse only multi-crystal chunks

```python
multi = idx.filter(min_crystals=2)
for entry in multi:
    chunk = idx.get_chunk(entry.position)
    for crystal in chunk.crystals:
        cell = crystal.cell
        # ... analyze unit cells, reflections, etc.
```

### Find a rare event (e.g. the chunk with the most crystals)

```python
best = max(idx.entries, key=lambda e: e.num_crystals)
print(f"Position {best.position}: {best.num_crystals} crystals")
chunk = idx.get_chunk(best.position)
```

## Performance

Benchmarked on a 40 GB stream file (139,161 chunks):

| Operation | Time |
|---|---|
| `build()` | 37s (1.09 GB/s) |
| `save()` | 0.6s (9 MB index) |
| `load()` | 0.4s |
| `get_chunk()` | 0.7 ms avg |
| `get_chunk_by_serial()` | 0.2 ms |
| `filter()` | <20 ms |
| `summary()` | <25 ms |

Compared to `iter_chunks()` from `stream_parser.py`, the index provides
up to ~2000x faster access for single chunks deep in the file, and ~600x
faster for sparse random access to 100 scattered chunks.

## When to use what

- **Parse every chunk sequentially** — use `stream_parser.iter_chunks()` directly.
  The index doesn't help here.
- **Query metadata (hit rate, crystal counts, indexing methods)** — use the index.
  No stream file access needed.
- **Parse specific chunks (by position, serial, or filter criteria)** — use the
  index. Filter first, then `get_chunk()` only what you need.
