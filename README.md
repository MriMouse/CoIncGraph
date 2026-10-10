# CoIncGraph

CPU–GPU Cooperative Incremental Processing of Out-of-Memory Streaming Graphs

## Build

Install CMake, CUDA and a compatible C++17 compiler, then build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

If needed, use CMake configuration options to select your installed CUDA toolkit
and compilers.

## Input

Use dense vertex IDs starting at 0. Files contain no headers or comment lines.

- `graph.txt`: one `source destination` edge per line (`--weight_num=1`).
- `updates.txt`: one `a source destination weight` (insert) or
  `d source destination weight` (delete) per line. Use `1` for unweighted data.
- `batches.txt`: one `addition_count deletion_count` per batch, delimiting
  the corresponding records in `updates.txt`.

For example, a batch deleting `1 -> 3` and inserting `2 -> 1`:

```text
# updates.txt (omit this comment in the actual file)
d 1 3 1
a 2 1 1
```

The corresponding line in `batches.txt` is `1 1`.
`--weight_num=1` generates SSSP graph weights from vertex IDs; it does not mean
unit weights. With `--weight_num=0`, graph rows must include a third weight column.

## Run

```sh
CUDA_VISIBLE_DEVICES=0 CG_BATCH_MAINTENANCE=auto CG_MERGE_PUBLICATION_SOURCES=1 \
./build/hybrid_bfs --graphfile=graph.txt --format=market_big \
  --updatefile=updates.txt --update_size=batches.txt \
  --weight_num=1 --weight=true --source_node=0 \
  --SEGMENT=1 --n_stream=3 --hybrid=0 --cache=2 \
  --check=true --verbose=false
```

Replace `hybrid_bfs` with `hybrid_sssp`, `hybrid_cc` or `hybrid_pr` as needed.
`--cache` sets the GPU cache budget in GB. Use `--check=true` for correctness
and `--check=false` for timing. Set `--bfs_max_batches=N`, `--sssp_max_batches=N`,
`--cc_max_batches=N` or `--pr_max_batches=N` to select the batch limit (default 10).
BFS ignores weights; CC uses directed minimum-label propagation (supply symmetric
edges for undirected components). Use `--help` for more options.
