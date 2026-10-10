# Dynamic CPU–GPU BFS

`hybrid_bfs` computes directed, unweighted hop distances. Every edge has cost 1,
including updates, regardless of input weights. Unreachable vertices are
`UINT32_MAX`; output columns are vertex, hop distance, parent, internal relaxation buffer.
The fourth column is diagnostic state, not an edge weight or a distance change.
Reachable non-source parents are checked as graph edges satisfying
`distance[parent] + 1 == distance[vertex]`; ties can select different valid parents.

The implementation shares the current SSSP chunk store, reverse index, GPU
deletion repair, insertion closure, hotness/cache maintenance and timers.
`TraversalEdgeWeight` and `DeletionEdgeWeight` specialize to constants at compile
time; the SSSP default weight and scheduling remain unchanged.

Build with the same CUDA 12.x toolchain as SSSP (the V100 target is sm_70):

```sh
cmake --build build --target hybrid_bfs -j2
```

Example (select an idle GPU/NUMA node before running):

```sh
CUDA_VISIBLE_DEVICES=0 CG_MUTATION_WORKERS=2 CG_REVERSE_SHARDS=64 \
CG_BATCH_MAINTENANCE=auto CG_ORDERED_REPAIR=0 \
build/hybrid_bfs \
  --graphfile=GRAPH --updatefile=UPDATES --update_size=BATCH_SIZES \
  --format=market_big --weight_num=1 --weight=1 --source_node=0 \
  --SEGMENT=512 --n_stream=3 --hybrid=0 --cache=2 \
  --bfs_max_batches=10 --check=false --bfs_print_checksum=true
```

Application flags use `bfs_`: `bfs_max_batches`, `bfs_print_checksum`.

`CG_INSERTION_SCHEDULE=block` remains the default. `ordered` is
available explicitly; `CG_ORDERED_REPAIR=1` also selects ordered
insertion. Weighted SSSP's large-diameter performance conclusions do not establish
which BFS schedule is fastest. Publication merge and other topology options are
inherited without changing production defaults.

Use `--check=true` for correctness runs; it adds full-graph validation work.
Run correctness separately from performance.

The large-batch mode is `CG_BATCH_MAINTENANCE=large` (or `auto`, which
selects large for at least 1,000,000 total updates). Road mode is `CG_ORDERED_REPAIR=1`; it combines ordered deletion
repair and ordered insertion and can be used with either maintenance mode.

Non-unit graph and update weights are ignored by BFS operators. The shared
`--weight=true` flag describes the update file format.
