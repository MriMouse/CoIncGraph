# Incremental PageRank

`hybrid_pr` computes the same unnormalized PageRank variant as the old PR sample:

```
x[v] = 0.15 + 0.85 * sum(x[u] / out_degree[u], for each edge u -> v)
```

Edge weights are ignored; parallel edges count separately. A dangling vertex
receives rank but sends none. There is no dangling-mass redistribution and no
final normalization. Output contains `vertex rank signed_residual` with enough
digits to round-trip float values.

```sh
cmake --build build --target hybrid_pr -j2
CUDA_VISIBLE_DEVICES=0 CG_MUTATION_WORKERS=2 \
CG_BATCH_MAINTENANCE=auto CG_MERGE_PUBLICATION_SOURCES=1 \
build/hybrid_pr --graphfile=GRAPH --format=market_big --weight_num=0 \
  --updatefile=UPDATES --update_size=BATCH_SIZES --weight=true \
  --SEGMENT=32 --n_stream=3 --hybrid=0 --cache=2 \
  --pr_max_batches=10 --error=1e-6 --check=true --output=ranks.txt
```

Update records use the existing `a/d source destination weight` format; batch
size records are `addition_count deletion_count`. Each batch applies deletion
before insertion and solves the final graph, as the old PR sample did. It does
not expose a separate deletion-only fixed point. Missing deletes follow the
shared topology store's occurrence semantics. Omit both update paths, or use
`--pr_max_batches=0`, for static computation. Batch count is clamped to available
input; empty phases and empty batches are supported.

`--error` is an absolute per-vertex residual threshold, **not** a relative rank
error. Both signs are active when `abs(residual) > error`. Subthreshold residual
is retained across rounds and batches. `--pr_max_rounds` defaults to 100 (accepted range 1–100), for both initialization
and every incremental batch. Stop when no active residual remains or this cap
is reached. Logs distinguish `stop=converged` from `stop=iteration_limit` and
include the remaining frontier size. Capped solves preserve residuals and pending
work across batches. A limit stop is **not** proof of convergence.
`--check=true` still requires residual convergence and fails on an inaccurate
capped result; `--check=false` permits capped experiments to continue.
`--check=true` independently recomputes the current graph in double precision
at initialization, after each batch and at completion, checking both rank error
and the residual invariant. This checking is outside batch timing.

## Retained mechanisms and PR adaptations

- Source-local pinned chunk adjacency, parallel source-group mutation and
  regular/large/auto batch maintenance remain authoritative.
- Source publication sort/merge, epoch/version validation, changed-row cache
  patching/invalidation and reclamation remain in the shared engine.
- PR uses the existing CTA/warp degree scheduler for both cache and zero-copy
  adjacency, the activity-window hotness score with stable vertex-ID pairing,
  cache candidate selection, refresh gate, eviction and compaction.
- Each batch subtracts old contributions and adds new contributions **only for
  touched sources**, including all their surviving edges whose normalization
  changes. Old rows are consumed before CPU mutation; new rows are consumed
  after ordered descriptor/cache publication. Rank and previous residual are
  preserved. No full-graph cancellation, rank reset or adjacency reupload occurs.
- An additive sparse frontier replaces SSSP's minimum-distance frontier. Separate
  consume/scatter kernels avoid lost residuals; a vertex is queued at most once
  per next frontier. There is no per-round full-vertex scan or duplicate sort.
  Runtime storage is 16 bytes per vertex plus 4 bytes per maximum touched source
  and two counters, allocated once/grown as needed. This is additional to the
  shared engine's existing arrays.
- Communication memcpy instrumentation and the external PCIe window are wired
  into the PR target. Batch timers include update preparation, mutation,
  publication, propagation and cache service.

SSSP-specific parent invalidation, reverse minimum-distance repair and CPU
`atomicMin` boundary propagation are incompatible with PR. PR keeps CPU topology
maintenance and GPU propagation. `CG_INSERTION_SCHEDULE` and `CG_ORDERED_REPAIR` select SSSP/CC executors
and do not select a PR executor. `--sparse` is accepted for compatibility: PR
always uses sparse residual frontiers. `--hybrid` does not switch the PR residual
executor; cached chunks/zero-copy remain its adjacency paths. Use
`--pr_max_rounds` to set its convergence limit.

Float atomics permit small run-to-run summation differences. Compare ranks and
residuals with a tolerance; integer checksums and parent witnesses are not PR
correctness criteria. Retaining infrastructure is not evidence of a speedup on
real datasets: benchmark matched inputs, tolerance, resources and full batch
cost before claiming one.
