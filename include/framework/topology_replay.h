#ifndef SEPGRAPH_TOPOLOGY_REPLAY_H
#define SEPGRAPH_TOPOLOGY_REPLAY_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <groute/graphs/common.h>

namespace sepgraph {
namespace topology {

struct TopologyMutation {
    index_t source;
    index_t destination;
};

struct TopologyMutationBatch {
    std::vector<TopologyMutation> deletions;
    std::vector<TopologyMutation> additions;

    std::vector<index_t> TouchedSources() const {
        std::vector<index_t> sources;
        sources.reserve(deletions.size() + additions.size());
        for (const auto &mutation : deletions) sources.push_back(mutation.source);
        for (const auto &mutation : additions) sources.push_back(mutation.source);
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        return sources;
    }
};

struct SourceTopologyDigest {
    index_t source = 0;
    uint64_t degree = 0;
    uint64_t ordered_hash = 0;
    uint64_t multiset_hash = 0;

    bool operator==(const SourceTopologyDigest &other) const {
        return source == other.source && degree == other.degree &&
               ordered_hash == other.ordered_hash &&
               multiset_hash == other.multiset_hash;
    }
};

inline uint64_t HashSequence(const std::vector<index_t> &destinations) {
    uint64_t hash = 1469598103934665603ULL;
    for (const index_t destination : destinations) {
        hash ^= static_cast<uint64_t>(destination);
        hash *= 1099511628211ULL;
    }
    hash ^= static_cast<uint64_t>(destinations.size());
    hash *= 1099511628211ULL;
    return hash;
}

inline SourceTopologyDigest DigestSource(index_t source,
                                         const std::vector<index_t> &destinations) {
    SourceTopologyDigest digest;
    digest.source = source;
    digest.degree = destinations.size();
    digest.ordered_hash = HashSequence(destinations);
    std::vector<index_t> sorted(destinations);
    std::sort(sorted.begin(), sorted.end());
    digest.multiset_hash = HashSequence(sorted);
    return digest;
}

template <typename AdjacencyView>
std::vector<index_t> MaterializeNeighbors(const AdjacencyView &view, index_t source) {
    std::vector<index_t> destinations;
    if (!view.Contains(source)) return destinations;
    destinations.reserve(view.Degree(source));
    for (uint64_t offset = 0; offset < view.Degree(source); ++offset) {
        destinations.push_back(view.EdgeAt(source, offset));
    }
    return destinations;
}

template <typename AdjacencyView>
SourceTopologyDigest DigestSource(const AdjacencyView &view, index_t source) {
    return DigestSource(source, MaterializeNeighbors(view, source));
}

} // namespace topology
} // namespace sepgraph

#endif // SEPGRAPH_TOPOLOGY_REPLAY_H
