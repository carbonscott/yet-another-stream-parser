# crystfel-yasp

Memory-efficient CrystFEL stream file parsing and indexing.

**Demo:** [marimo.app](https://static.marimo.app/static/yasp-cjg6) | **Notebook:** [examples/demo.py](examples/demo.py)

## CLI Usage

### Build an index

```bash
uv run --with numpy crystfel-yasp build file.stream
# Creates file.stream.idx
```

### Show index summary

```bash
uv run --with numpy crystfel-yasp info file.stream.idx
```

### Retrieve a chunk

```bash
# By position (0-indexed)
uv run --with numpy crystfel-yasp get file.stream.idx --position 0

# By serial number
uv run --with numpy crystfel-yasp get file.stream.idx --serial 12345
```

### Filter chunks

```bash
# Show chunks with at least 1 crystal
uv run --with numpy crystfel-yasp filter file.stream.idx --min-crystals 1

# Show only hits
uv run --with numpy crystfel-yasp filter file.stream.idx --hit

# Combine filters
uv run --with numpy crystfel-yasp filter file.stream.idx --hit --min-crystals 2
```

## Python API

For custom applications, use the Python API directly. See [examples/demo.py](examples/demo.py) for a full interactive notebook.

### Streaming iteration

```python
from crystfel_yasp import iter_chunks

for chunk in iter_chunks('file.stream'):
    print(chunk.filename, chunk.serial, len(chunk.crystals))
```

### Using an index

```python
from crystfel_yasp import StreamIndex

index = StreamIndex.load('file.stream.idx')

# Retrieve a chunk
chunk = index.get_chunk(0)
chunk = index.get_chunk_by_serial(12345)

# Filter and iterate
for entry in index.filter(min_crystals=1):
    chunk = index.get_chunk(entry.position)
    print(f"Serial {chunk.serial}: {len(chunk.crystals)} crystals")
```
