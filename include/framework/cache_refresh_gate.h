#ifndef SEPGRAPH_CACHE_REFRESH_GATE_H
#define SEPGRAPH_CACHE_REFRESH_GATE_H

#include <cstdint>

namespace sepgraph {
namespace cache_patch {

class RefreshGate {
public:
    bool Observe(uint64_t desired_vertices, bool missing_desired_vertex) {
        pending_vertices_ = desired_vertices;
        refresh_required_ = !published_ || missing_desired_vertex ||
                            desired_vertices != published_vertices_;
        return refresh_required_;
    }
    void Publish() {
        published_vertices_ = pending_vertices_;
        published_ = true;
        refresh_required_ = false;
    }
    bool refresh_required() const { return refresh_required_; }
    bool published() const { return published_; }
    uint64_t published_vertices() const { return published_vertices_; }
private:
    uint64_t published_vertices_ = 0;
    uint64_t pending_vertices_ = 0;
    bool published_ = false;
    bool refresh_required_ = true;
};

} // namespace cache_patch
} // namespace sepgraph

#endif
