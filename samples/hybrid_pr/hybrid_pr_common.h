#ifndef HYBRID_PR_COMMON_H
#define HYBRID_PR_COMMON_H
#include <groute/graphs/source_local_chunk_store.h>
#include <vector>
using rank_t = float;
#define IDENTITY_ELEMENT 0.0f
#define ALPHA 0.85f
bool PageRankCheck(const sepgraph::topology::SourceLocalChunkStore &graph,
                  const std::vector<rank_t> &ranks, const std::vector<rank_t> &residual,
                  double epsilon, const char *stage, int batch);
bool PageRankOutput(const char *path, const std::vector<rank_t> &ranks,
                    const std::vector<rank_t> &residual);
#endif
