#ifndef HYBRID_CC_COMMON_H
#define HYBRID_CC_COMMON_H

#include <cstdint>
#include <vector>
#include <groute/graphs/source_local_chunk_store.h>

using label_t = uint32_t;
#define IDENTITY_ELEMENT UINT32_MAX

// Original-system semantics: minimum ID reaching each vertex along directed edges.
uint64_t CCCheck(const sepgraph::topology::SourceLocalChunkStore &graph,
                 const std::vector<label_t> &labels);
bool CCOutput(const char *path, const std::vector<label_t> &labels);
#endif
