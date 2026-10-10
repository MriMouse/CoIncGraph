#ifndef SEPGRAPH_TOPOLOGY_CONTRACT_CUH
#define SEPGRAPH_TOPOLOGY_CONTRACT_CUH

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <groute/graphs/common.h>

#if defined(__CUDACC__)
#define SEPGRAPH_HOST_DEVICE __host__ __device__
#define SEPGRAPH_FORCEINLINE __forceinline__
#else
#define SEPGRAPH_HOST_DEVICE
#define SEPGRAPH_FORCEINLINE inline
#endif

namespace sepgraph {
namespace topology {

// The full mirror stays compact. Sparse publication transfers this extended
// descriptor as sparse records instead of expanding every vertex entry.
struct TopologyDescriptor {
    uint64_t index;
    index_t degree;
    uint32_t slab_id;
    uint32_t version;
};

struct TopologyPatchRecord {
    index_t source;
    uint32_t slab_id;
    uint64_t offset;
    index_t degree;
    uint32_t version;
};

struct TopologyDeviceDigest {
    uint64_t degree;
    uint64_t ordered_hash;
};

static_assert(std::is_standard_layout<TopologyDescriptor>::value,
              "topology descriptors must be transferable as POD records");
static_assert(std::is_trivially_copyable<TopologyDescriptor>::value,
              "topology descriptors must be transferable as POD records");
static_assert(sizeof(TopologyDescriptor) == 24,
              "topology descriptor ABI changed; update host/device publication together");
static_assert(sizeof(TopologyPatchRecord) == 24,
              "sparse topology patch ABI must remain compact");

template <typename Descriptor, typename Edge>
struct ContiguousAdjacencyView {
    const Descriptor *descriptors;
    const Edge *edges;
    index_t nnodes;

    SEPGRAPH_HOST_DEVICE SEPGRAPH_FORCEINLINE bool Contains(index_t source) const {
        return source < nnodes;
    }

    SEPGRAPH_HOST_DEVICE SEPGRAPH_FORCEINLINE uint64_t Begin(index_t source) const {
        assert(Contains(source));
        return descriptors[source].index;
    }

    SEPGRAPH_HOST_DEVICE SEPGRAPH_FORCEINLINE index_t Degree(index_t source) const {
        assert(Contains(source));
        return descriptors[source].degree;
    }

    SEPGRAPH_HOST_DEVICE SEPGRAPH_FORCEINLINE Edge EdgeAt(index_t source,
                                                           uint64_t offset) const {
        assert(offset < Degree(source));
        return edges[Begin(source) + offset];
    }
};

enum class TopologyEpochPhase : uint8_t {
    kPublished,
    kMutating,
};

// CPU mutations join one pending epoch. GPU traversal is valid only after its
// descriptor publication advances published_epoch to that pending epoch.
class TopologyEpochContract {
public:
    TopologyEpochContract()
        : published_epoch_(0), pending_epoch_(0), phase_(TopologyEpochPhase::kPublished) {}

    uint64_t MarkMutation() {
        if (phase_ == TopologyEpochPhase::kPublished) {
            pending_epoch_ = published_epoch_ + 1;
            phase_ = TopologyEpochPhase::kMutating;
        }
        return pending_epoch_;
    }

    uint64_t MarkPublished() {
        if (phase_ == TopologyEpochPhase::kMutating) {
            published_epoch_ = pending_epoch_;
            phase_ = TopologyEpochPhase::kPublished;
        }
        return published_epoch_;
    }

    bool IsPublished() const {
        return phase_ == TopologyEpochPhase::kPublished &&
               published_epoch_ == pending_epoch_;
    }

    uint64_t PublishedEpoch() const { return published_epoch_; }
    uint64_t PendingEpoch() const { return pending_epoch_; }
    TopologyEpochPhase Phase() const { return phase_; }

private:
    uint64_t published_epoch_;
    uint64_t pending_epoch_;
    TopologyEpochPhase phase_;
};

} // namespace topology
} // namespace sepgraph

#undef SEPGRAPH_HOST_DEVICE
#undef SEPGRAPH_FORCEINLINE

#endif // SEPGRAPH_TOPOLOGY_CONTRACT_CUH
