#pragma once

namespace sepgraph { namespace cc_repair {

// The affected set is a union of entire OLD components. After deletion no edge
// can leave it. Values were reset to vertex IDs, so they can serve as a disjoint
// set without allocating another vertex array. Link only roots, always toward
// smaller IDs: stale roots are retried and cycles cannot be introduced.
__device__ inline unsigned Root(unsigned *parents, unsigned v) {
    // Volatile loads observe concurrent root links without serializing every
    // edge on the atomic unit of the giant component's minimum vertex.
    volatile unsigned *visible = parents;
    unsigned next = visible[v];
    while (next != v) {
        const unsigned grandparent = visible[next];
        if (grandparent != next) atomicCAS(parents + v, next, grandparent);
        v = next;
        next = visible[v];
    }
    return v;
}

__device__ inline void Unite(unsigned *parents, unsigned u, unsigned v) {
    for (;;) {
        u = Root(parents, u);
        v = Root(parents, v);
        if (u == v) return;
        const unsigned hi = u > v ? u : v;
        const unsigned lo = u > v ? v : u;
        if (atomicCAS(parents + hi, hi, lo) == hi) return;
    }
}

template<class Patch>
__global__ void MarkChanged(const Patch *patches, unsigned count, unsigned *witness) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x)
        witness[patches[i].source] = 0;
}

// Sampling is only a connectivity certificate, never an approximation. Freeze
// its roots before finishing: an edge internal to a sampled component is already
// redundant. For the selected component, symmetry lets its other endpoint own
// every outgoing cut edge. Do not apply the usual u<v filter to those cut edges.
template<class Graph>
__global__ void Sample(Graph graph, const unsigned *affected, unsigned count,
                       unsigned *parents, const unsigned *witness,
                       const unsigned *cache) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const unsigned u = affected[i];
        const auto begin = graph.begin_edge(u);
        const auto end = graph.end_edge(u);
        const bool cached = witness[u] == UINT32_MAX && graph.vertices_[u].cache;
        const auto degree = cached ? graph.vertices_[u].virtual_degree : 0;
        const auto offset = cached ? graph.vertices_[u].virtual_start : 0;
        for (unsigned j = 0; j < 2 && begin + j < end; ++j) {
            const unsigned v = j < degree ? cache[offset + j] : graph.edge_dest(begin + j);
            Unite(parents, u, v);
        }
    }
}

__global__ void Snapshot(const unsigned *affected, unsigned count,
                         unsigned *parents, unsigned *roots) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const unsigned u = affected[i];
        roots[u] = Root(parents, u);
    }
}

// A deterministic, evenly spaced sample selects a useful component without a
// V-sized histogram or host round trip. A poor selection affects speed only.
__global__ void SelectComponent(const unsigned *affected, unsigned count,
                                const unsigned *roots, unsigned *selection) {
    __shared__ unsigned candidates[256];
    __shared__ unsigned votes[256];
    const unsigned t = threadIdx.x;
    candidates[t] = roots[affected[(static_cast<unsigned long long>(t) * count) / 256]];
    __syncthreads();
    unsigned n = 0;
    for (unsigned j = 0; j < 256; ++j) n += candidates[j] == candidates[t];
    votes[t] = n;
    __syncthreads();
    if (t == 0) {
        unsigned best = 0;
        for (unsigned j = 1; j < 256; ++j)
            if (votes[j] > votes[best]) best = j;
        selection[0] = candidates[best];
        selection[1] = votes[best];
    }
}

template<bool Sampled = false, class Graph>
__global__ void Hook(Graph graph, const unsigned *affected, unsigned count,
                     unsigned *parents, const unsigned *witness,
                     const unsigned *cache, const unsigned *roots = nullptr,
                     const unsigned *selection = nullptr) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned stride = gridDim.x * blockDim.x / 32;
    for (unsigned i = warp; i < count; i += stride) {
        const unsigned u = affected[i];
        if (Sampled && roots[u] == selection[0]) continue;
        const auto begin = graph.begin_edge(u);
        const auto end = graph.end_edge(u);
        // Collection set CC's unused witness to UINT_MAX. Deletion staging
        // marks changed rows, whose old cache is not a valid adjacency view.
        const bool cached = witness[u] == UINT32_MAX && graph.vertices_[u].cache;
        const auto cached_degree = cached ? graph.vertices_[u].virtual_degree : 0;
        const auto cached_begin = cached ? graph.vertices_[u].virtual_start : 0;
        for (auto e = begin + lane; e < end; e += 32) {
            const unsigned v = e - begin < cached_degree ?
                cache[cached_begin + e - begin] : graph.edge_dest(e);
            // Symmetric multigraph: one orientation suffices, including duplicates.
            if (Sampled) {
                if (roots[u] != roots[v] && (u < v || roots[v] == selection[0]))
                    Unite(parents, u, v);
            } else if (u < v) Unite(parents, u, v);
        }
    }
}

__global__ void Flatten(const unsigned *affected, unsigned count,
                        unsigned *parents, unsigned *buffers, bool *reset,
                        unsigned *witness) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const unsigned v = affected[i];
        const unsigned root = Root(parents, v);
        atomicMin(parents + v, root);
        buffers[v] = root;
        reset[v] = false;
        witness[v] = UINT32_MAX;
    }
}

}} // namespace sepgraph::cc_repair
