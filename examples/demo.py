# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "altair",
#     "crystfel-yasp @ git+https://github.com/carbonscott/yet-another-stream-parser.git",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import numpy as np
    from crystfel_yasp import iter_chunks, StreamIndex, parse_chunks_shared, REFL_COLUMNS
    return StreamIndex, iter_chunks, mo, np, parse_chunks_shared, REFL_COLUMNS, Path


@app.cell
def _():
    # --- Configuration constants ---
    FILTER_PREVIEW_LIMIT = 100      # max rows in filter results table
    STREAMING_DEMO_CHUNKS = 20      # chunks to iterate in streaming demo
    PARALLEL_DEMO_CHUNKS = 50       # chunks for parallel parsing demo
    UNIT_CELL_SAMPLE_SIZE = 200     # chunks to sample for unit cell stats
    return FILTER_PREVIEW_LIMIT, STREAMING_DEMO_CHUNKS, UNIT_CELL_SAMPLE_SIZE


@app.cell
def _(Path):
    # --- Path configuration (adjust for your environment) ---
    project_root = Path(__file__).resolve().parent.parent
    INDEX_PATH = str(project_root / "benchmarks" / "ox112it2.stream.idx")
    ITER_STREAM = (
        "/sdf/data/lcls/ds/mfx/mfxl1025422/results/btx/streams/"
        "ox112it2.stream"
    )
    return INDEX_PATH, ITER_STREAM, project_root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CrystFEL Stream Processing Toolkit

    This notebook demonstrates the key features of the stream processing
    library for working with CrystFEL `.stream` files — large text-based
    outputs from serial crystallography experiments.

    **The challenge:** Stream files can be tens of gigabytes, containing
    hundreds of thousands of diffraction image results ("chunks"), each with
    peaks, crystal parameters, and millions of indexed reflections.

    **This toolkit provides:**

    1. **Streaming parser** — memory-efficient iteration over chunks
    2. **Persistent index** — O(1) random access to any chunk without scanning
    3. **Parallel parsing** — shared-memory multiprocessing for bulk reflection extraction

    ### Data Model

    ```
    StreamFile
    └── Chunk  (one per diffraction image)
        ├── metadata: filename, serial, hit, photon_energy, …
        ├── peaks: List[Peak]  (detected spots)
        └── crystals: List[Crystal]  (indexing results)
            ├── cell: UnitCell  (a, b, c, α, β, γ)
            ├── reciprocal vectors: astar, bstar, cstar
            └── reflections: List[Reflection]  (h, k, l, I, σ, …)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Index Summary
    """)
    return


@app.cell
def _(INDEX_PATH, StreamIndex, mo):
    mo.output.replace(mo.md("Loading index…"))

    index = StreamIndex.load(INDEX_PATH)
    stats = index.summary()

    summary_rows = [
        {"Metric": "Stream file", "Value": stats["stream_path"]},
        {"Metric": "File size", "Value": f'{stats["file_size"] / 1e9:.1f} GB'},
        {"Metric": "Total chunks", "Value": f'{stats["num_chunks"]:,}'},
        {
            "Metric": "Hits",
            "Value": f'{stats["num_hits"]:,} ({stats["hit_rate"]:.1%})',
        },
        {
            "Metric": "Indexed",
            "Value": f'{stats["num_indexed"]:,} ({stats["index_rate"]:.1%})',
        },
        {"Metric": "Total crystals", "Value": f'{stats["total_crystals"]:,}'},
        {"Metric": "Total peaks", "Value": f'{stats["total_peaks"]:,}'},
        {
            "Metric": "Total reflections",
            "Value": f'{stats["total_reflections"]:,}',
        },
    ]

    mo.vstack(
        [
            mo.md(
                """
                A `StreamIndex` scans the stream file once with memory-mapped I/O,
                recording byte offsets and metadata for every chunk.  Once built,
                the index enables **O(1) chunk lookup** (~0.2 ms), **fast metadata
                filtering** (~20 ms), and **random-access parsing** (~0.7 ms).
                """
            ),
            mo.ui.table(summary_rows, selection=None),
        ]
    )
    return (index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive Filtering
    """)
    return


@app.cell
def _(index):
    index.entries[0]  # Each entry is a chunk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Byte offset is used to locate the chunks.  The example below shows how to fetch a chunk using offset directly with `dd` command.
    """)
    return


@app.cell
def _(ITER_STREAM, index):
    import os
    _start_offset = index.entries[0].start_offset
    _end_offset = index.entries[0].end_offset
    _ = os.system(f"dd if={ITER_STREAM} bs=1 skip={_start_offset} count={_end_offset-_start_offset} 2>/dev/null")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, back to the Python toolkit
    """)
    return


@app.cell
def _(index, mo):
    # Discover available indexing methods for the dropdown
    methods = sorted(
        set(e.indexed_by for e in index.entries if e.indexed_by and e.indexed_by != "none")
    )
    method_options = {"Any": None}
    method_options.update({m: m for m in methods})

    hit_filter = mo.ui.dropdown(
        options={"Any": None, "Hits only": True, "Non-hits only": False},
        value="Any",
        label="Hit status",
    )
    method_filter = mo.ui.dropdown(
        options=method_options,
        value="Any",
        label="Indexing method",
    )
    min_crystals_input = mo.ui.number(
        start=1, stop=20, value=1, step=1, label="Min crystals",
    )

    mo.vstack(
        [
            mo.md(
                """
                Filter chunks by hit status, crystal count, and indexing method —
                all without reading the stream file.
                """
            ),
            mo.hstack([hit_filter, method_filter, min_crystals_input]),
        ]
    )
    return hit_filter, method_filter, min_crystals_input


@app.cell
def _(
    FILTER_PREVIEW_LIMIT,
    hit_filter,
    index,
    method_filter,
    min_crystals_input,
    mo,
):
    filtered = index.filter(
        hit=hit_filter.value,
        min_crystals=min_crystals_input.value if min_crystals_input.value > 0 else None,
        indexed_by=method_filter.value,
    )

    preview = [
        {
            "Position": e.position,
            "Serial": e.serial,
            "Hit": e.hit,
            "Crystals": e.num_crystals,
            "Peaks": e.num_peaks,
            "Reflections": e.num_reflections,
            "Indexed by": e.indexed_by,
        }
        for e in filtered[:FILTER_PREVIEW_LIMIT]
    ]

    mo.vstack(
        [
            mo.md(
                f"**{len(filtered):,}** chunks match the current filters"
                f" (showing first {min(FILTER_PREVIEW_LIMIT, len(filtered))}):"
            ),
            mo.ui.table(preview, selection=None) if preview else mo.md("_No matching chunks._"),
        ]
    )
    return (filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chunk Explorer
    """)
    return


@app.cell
def _(filtered, mo):
    max_idx = max(len(filtered) - 1, 0)
    chunk_index = mo.ui.number(
        start=0,
        stop=max_idx,
        value=0,
        step=1,
        label=f"Chunk index (0–{max_idx})",
    )

    mo.vstack(
        [
            mo.md(
                """
                Pick a chunk from the filtered results.  The index seeks directly to
                the chunk's byte offset in the stream file and parses it on demand.

                This demonstrates extracting **all data** from a single chunk:
                metadata, peaks, and crystals (with unit cell and reflections).
                """
            ),
            chunk_index,
        ]
    )
    return (chunk_index,)


@app.cell
def _(chunk_index, filtered, index, mo):
    if not filtered:
        _output = mo.md("_No chunks to display.  Adjust filters above._")
    else:
        _entry = filtered[chunk_index.value]
        _chunk = index.get_chunk(_entry.position)

        _sections = [
            mo.md(f"### Chunk at position {_entry.position}"),
            mo.md(
                f"**Selected:** serial {_entry.serial}, "
                f"{_entry.num_crystals} crystals, {_entry.num_peaks} peaks"
            ),
            mo.md("**Metadata:**"),
            mo.ui.table(
                [
                    {"Field": "Filename", "Value": _chunk.filename},
                    {"Field": "Event", "Value": _chunk.event},
                    {"Field": "Serial", "Value": str(_chunk.serial)},
                    {"Field": "Hit", "Value": str(_chunk.hit)},
                    {"Field": "Indexed by", "Value": _chunk.indexed_by},
                    {"Field": "Photon energy", "Value": f"{_chunk.photon_energy_eV:.2f} eV"},
                    {"Field": "Beam divergence", "Value": f"{_chunk.beam_divergence:.2e} rad"},
                    {"Field": "Camera length", "Value": f"{_chunk.average_camera_length:.4f} m"},
                    {"Field": "Num peaks", "Value": str(_chunk.num_peaks)},
                    {"Field": "Num crystals", "Value": str(len(_chunk.crystals))},
                ],
                selection=None,
            ),
        ]

        # Peaks section (paginated table)
        if _chunk.peaks:
            _sections.append(mo.md(f"**Peaks** ({len(_chunk.peaks)} total):"))
            _sections.append(
                mo.ui.table(
                    [
                        {
                            "fs (px)": f"{p.fs:.1f}",
                            "ss (px)": f"{p.ss:.1f}",
                            "1/d (nm⁻¹)": f"{p.resolution:.2f}",
                            "Intensity": f"{p.intensity:.1f}",
                            "Panel": p.panel,
                        }
                        for p in _chunk.peaks
                    ],
                    selection=None,
                )
            )

        # Crystals section
        for _i, _crystal in enumerate(_chunk.crystals):
            _cell = _crystal.cell
            _sections.append(
                mo.md(f"**Crystal {_i}** — {_cell.lattice_type} {_cell.centering}")
            )
            _sections.append(
                mo.ui.table(
                    [
                        {"Param": "a (nm)", "Value": f"{_cell.a:.4f}"},
                        {"Param": "b (nm)", "Value": f"{_cell.b:.4f}"},
                        {"Param": "c (nm)", "Value": f"{_cell.c:.4f}"},
                        {"Param": "alpha (deg)", "Value": f"{_cell.alpha:.2f}"},
                        {"Param": "beta (deg)", "Value": f"{_cell.beta:.2f}"},
                        {"Param": "gamma (deg)", "Value": f"{_cell.gamma:.2f}"},
                        {"Param": "Resolution limit", "Value": f"{_crystal.resolution_limit:.2f} nm⁻¹"},
                        {"Param": "Num reflections", "Value": str(_crystal.num_reflections)},
                    ],
                    selection=None,
                )
            )

            # Reflections (paginated table)
            if _crystal.reflections:
                _sections.append(
                    mo.md(f"Reflections ({_crystal.num_reflections} total):")
                )
                _sections.append(
                    mo.ui.table(
                        [
                            {
                                "h": r.h,
                                "k": r.k,
                                "l": r.l,
                                "I": f"{r.intensity:.1f}",
                                "sigma": f"{r.sigma:.1f}",
                                "peak": f"{r.peak:.1f}",
                                "bg": f"{r.background:.1f}",
                                "fs": f"{r.fs:.1f}",
                                "ss": f"{r.ss:.1f}",
                                "panel": r.panel,
                            }
                            for r in _crystal.reflections
                        ],
                        selection=None,
                    )
                )

        _output = mo.vstack(_sections)

    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming Parser
    """)
    return


@app.cell
def _(ITER_STREAM, STREAMING_DEMO_CHUNKS, iter_chunks, mo):
    _chunks = []
    for _i, _c in enumerate(iter_chunks(ITER_STREAM, load_reflections=False)):
        _chunks.append(_c)
        if _i >= STREAMING_DEMO_CHUNKS - 1:
            break

    _rows = [
        {
            "Serial": c.serial,
            "Hit": c.hit,
            "Peaks": c.num_peaks,
            "Crystals": len(c.crystals),
            "Indexed by": c.indexed_by,
            "Energy (eV)": f"{c.photon_energy_eV:.1f}",
        }
        for c in _chunks
    ]

    mo.vstack(
        [
            mo.md(
                """
                `iter_chunks()` reads a stream file sequentially, yielding one
                `Chunk` at a time — only one chunk is in memory at any moment.

                Reflection data is **parsed lazily**: raw text lines are collected
                but converted to `Reflection` objects only when `crystal.reflections`
                is first accessed.  Set `load_reflections=False` to skip reflections
                entirely for metadata-only workflows.
                """
            ),
            mo.md(
                f"First **{len(_chunks)}** chunks from stream file"
                " (metadata only, no reflections):"
            ),
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Use Cases
    """)
    return


@app.cell
def _(index, mo):
    from collections import Counter

    _entries = index.entries
    _n = len(_entries)
    _n_hits = sum(1 for e in _entries if e.hit)
    _n_indexed = sum(
        1 for e in _entries if e.indexed_by and e.indexed_by != "none"
    )

    _method_counts = Counter(
        e.indexed_by
        for e in _entries
        if e.indexed_by and e.indexed_by != "none"
    )
    _method_rows = [
        {
            "Method": method,
            "Count": f"{count:,}",
            "Fraction": f"{count / _n:.1%}",
        }
        for method, count in _method_counts.most_common()
    ]

    mo.vstack(
        [
            mo.md(
                """
                The index enables fast analytical queries over the entire dataset
                without re-parsing the stream file.

                #### Hit Rate & Indexing Breakdown
                """
            ),
            mo.md(
                f"- Total chunks: **{_n:,}**\n"
                f"- Hits: **{_n_hits:,}** ({_n_hits / _n:.1%})\n"
                f"- Indexed: **{_n_indexed:,}** ({_n_indexed / _n:.1%})"
            ),
            mo.md("**Indexing method breakdown:**"),
            mo.ui.table(_method_rows, selection=None)
            if _method_rows
            else mo.md("_No indexed chunks._"),
        ]
    )
    return


@app.cell
def _(UNIT_CELL_SAMPLE_SIZE, index, mo, np):
    _indexed = index.filter(hit=True, min_crystals=1)
    _sample_n = min(UNIT_CELL_SAMPLE_SIZE, len(_indexed))
    _sample = _indexed[:_sample_n]

    _a, _b, _c = [], [], []
    for _entry in _sample:
        _chunk = index.get_chunk(_entry.position)
        for _crystal in _chunk.crystals:
            _a.append(_crystal.cell.a)
            _b.append(_crystal.cell.b)
            _c.append(_crystal.cell.c)

    if _a:
        _a_arr, _b_arr, _c_arr = np.array(_a), np.array(_b), np.array(_c)
        _cell_stats = [
            {
                "Axis": name,
                "Mean (nm)": f"{np.mean(arr):.4f}",
                "Std (nm)": f"{np.std(arr):.4f}",
                "Min (nm)": f"{np.min(arr):.4f}",
                "Max (nm)": f"{np.max(arr):.4f}",
            }
            for name, arr in [("a", _a_arr), ("b", _b_arr), ("c", _c_arr)]
        ]

        _output = mo.vstack(
            [
                mo.md(
                    f"#### Unit Cell Distribution"
                    f" ({len(_a)} crystals from {_sample_n} chunks)"
                ),
                mo.ui.table(_cell_stats, selection=None),
            ]
        )
    else:
        _output = mo.md("_No indexed crystals found in sample._")
    _output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
