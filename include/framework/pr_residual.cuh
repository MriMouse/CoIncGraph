// Sparse signed-residual PageRank on the authoritative chunk/cache adjacency.
#ifndef CG_PR_RESIDUAL_CUH
#define CG_PR_RESIDUAL_CUH
#include <groute/graphs/csr_graph.cuh>
#include <groute/device/cta_scheduler_hybrid.cuh>
#include <utils/cuda_utils.h>
#include <utils/communication_meter.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sepgraph { namespace pr {
constexpr float kAlpha = 0.85f;
constexpr float kBase = 0.15f;

struct Frontier {
    float *residual;
    index_t *vertices;
    unsigned *count, *queued, *invalid;
    float epsilon;
    __device__ void Add(index_t dst, float contribution) const {
        const float old = atomicAdd(residual + dst, contribution);
        const float value = old + contribution;
        if (!isfinite(value)) atomicExch(invalid, 1u);
        // The flag is reset only in the separate consume kernel. Each next
        // frontier contains at most V distinct vertices, including cancellations.
        if (fabsf(old) <= epsilon && fabsf(value) > epsilon &&
            atomicCAS(queued + dst, 0u, 1u) == 0)
            vertices[atomicAdd(count, 1u)] = dst;
    }
};
struct Payload { float delta; };
struct Push {
    groute::graphs::dev::PMAGraph graph;
    const index_t *cache;
    Frontier frontier;
    bool cached;
    __device__ bool operator()(uint64_t edge, Payload p) const {
        frontier.Add(cached ? cache[edge] : graph.edge_dest(edge), p.delta);
        return true;
    }
};

static __global__ void Initialize(index_t n, float *ranks, float *residual,
                                  index_t *queue, unsigned *queued) {
    for (uint64_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n;
         v += uint64_t(blockDim.x) * gridDim.x) {
        ranks[v] = 0;
        residual[v] = kBase;
        queue[v] = v;
        queued[v] = 1;
    }
}
static __global__ void Consume(index_t count, const index_t *queue,
                               unsigned *queued, float *ranks, float *residual,
                               float *deltas, float epsilon, unsigned *invalid) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(blockDim.x) * gridDim.x) {
        const index_t v = queue[i];
        const float r = residual[v];
        queued[v] = 0;
        deltas[i] = 0;
        if (!isfinite(r) || !isfinite(ranks[v])) atomicExch(invalid, 1u);
        
        if (fabsf(r) > epsilon) {
            ranks[v] += r;
            residual[v] = 0;
            deltas[i] = r;
        }
    }
}
// Both sparse repair seeding and propagation retain the existing CTA/warp
// degree scheduler, GPU cached rows, zero-copy chunks and activity hotness.
static __global__ void Scatter(groute::graphs::dev::PMAGraph graph,
                               const index_t *cache, const index_t *sources,
                               index_t count, const float *values, bool amend,
                               float sign, Frontier frontier) {
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    const uint64_t rounded = (uint64_t(count) + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (uint64_t i = tid; i < rounded; i += stride) {
        groute::dev::np_local<Payload> cold = {0, 0}, hot = {0, 0};
        if (i < count) {
            const index_t src = sources[i];
            const index_t degree = graph.degree(src);
            const float value = amend ? values[src] * sign : values[i];
            if (degree && value != 0) {
                auto &row = graph.vertices_[src];
                row.hotness[0]++;
                auto &work = row.cache ? hot : cold;
                work.start = row.cache ? row.virtual_start : graph.begin_edge(src);
                work.size = degree;
                work.meta_data.delta = kAlpha * value / degree;
            }
        }
        using Scheduler = groute::dev::CTAWorkSchedulerNew<Payload, groute::dev::LB_COARSE_GRAINED>;
        Scheduler::schedule(hot, Push{graph, cache, frontier, true}, true);
        Scheduler::schedule(cold, Push{graph, cache, frontier, false}, true);
    }
}

// Buffers are allocated once and reused across all batches. No graph scan or
// per-edge queue append/duplicate sort is needed to rebuild a sparse frontier.
class Runtime {
    index_t n_;
    index_t *in_ = nullptr, *out_ = nullptr, *sources_ = nullptr;
    unsigned *queued_ = nullptr, *counts_ = nullptr;
    float *deltas_ = nullptr;
    size_t source_capacity_ = 0;
    unsigned count_ = 0;
    static unsigned Blocks(size_t n) { return static_cast<unsigned>(std::min<size_t>(4096, (n + 255) / 256)); }
    Frontier Target(float *residual, float epsilon) {
        return {residual, out_, counts_, queued_, counts_ + 1, epsilon};
    }
    unsigned ReadCount(cudaStream_t stream) {
        cgcomm::Scope scope(cgcomm::Category::Control);
        unsigned result[2];
        GROUTE_CUDA_CHECK(cudaMemcpyAsync(result, counts_, sizeof(result), cudaMemcpyDeviceToHost, stream));
        GROUTE_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (result[1]) throw std::runtime_error("PR non-finite rank/residual");
        if (result[0] > n_) throw std::runtime_error("PR frontier overflow");
        return result[0];
    }
public:
    explicit Runtime(index_t n) : n_(n) {
        const size_t size = std::max<index_t>(1, n);
        GROUTE_CUDA_CHECK(cudaMalloc(&in_, size * sizeof(index_t)));
        GROUTE_CUDA_CHECK(cudaMalloc(&out_, size * sizeof(index_t)));
        GROUTE_CUDA_CHECK(cudaMalloc(&queued_, size * sizeof(unsigned)));
        GROUTE_CUDA_CHECK(cudaMalloc(&deltas_, size * sizeof(float)));
        GROUTE_CUDA_CHECK(cudaMalloc(&counts_, 2 * sizeof(unsigned)));
        GROUTE_CUDA_CHECK(cudaMemset(queued_, 0, size * sizeof(unsigned)));
        GROUTE_CUDA_CHECK(cudaMemset(counts_, 0, 2 * sizeof(unsigned)));
    }
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    ~Runtime() { cudaFree(in_); cudaFree(out_); cudaFree(queued_); cudaFree(deltas_); cudaFree(counts_); cudaFree(sources_); }
    void Init(float *ranks, float *residual, cudaStream_t stream) {
        count_ = n_;
        if (n_) Initialize<<<Blocks(n_), 256, 0, stream>>>(n_, ranks, residual, in_, queued_);
    }
    void BeginBatch(const std::vector<index_t> &sources, cudaStream_t stream) {
        cgcomm::Scope scope(cgcomm::Category::Control);
        
        if (count_) GROUTE_CUDA_CHECK(cudaMemcpyAsync(out_, in_,
            count_ * sizeof(index_t), cudaMemcpyDeviceToDevice, stream));
        if (sources.size() > source_capacity_) {
            GROUTE_CUDA_CHECK(cudaFree(sources_));
            GROUTE_CUDA_CHECK(cudaMalloc(&sources_, sources.size() * sizeof(index_t)));
            source_capacity_ = sources.size();
        }
        if (!sources.empty()) GROUTE_CUDA_CHECK(cudaMemcpyAsync(sources_, sources.data(),
            sources.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        GROUTE_CUDA_CHECK(cudaMemcpyAsync(counts_, &count_, sizeof(count_), cudaMemcpyHostToDevice, stream));
        GROUTE_CUDA_CHECK(cudaMemsetAsync(counts_ + 1, 0, sizeof(unsigned), stream));
        GROUTE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    bool Converged() const { return count_ == 0; }
    unsigned ActiveCount() const { return count_; }
    void Amend(groute::graphs::dev::PMAGraph graph, const index_t *cache,
               index_t source_count, const float *ranks, float *residual,
               float sign, float epsilon, cudaStream_t stream) {
        if (source_count) Scatter<<<Blocks(source_count), 256, 0, stream>>>(
            graph, cache, sources_, source_count, ranks, true, sign, Target(residual, epsilon));
        GROUTE_CUDA_CHECK(cudaGetLastError());
    }
    void EndBatch(cudaStream_t stream) {
        count_ = ReadCount(stream);
        std::swap(in_, out_);
    }
    unsigned Converge(groute::graphs::dev::PMAGraph graph, const index_t *cache,
                      float *ranks, float *residual, float epsilon,
                      unsigned max_rounds, cudaStream_t stream) {
        unsigned rounds = 0;
        while (count_ && rounds < max_rounds) {
            Consume<<<Blocks(count_), 256, 0, stream>>>(count_, in_, queued_, ranks,
                residual, deltas_, epsilon, counts_ + 1);
            GROUTE_CUDA_CHECK(cudaMemsetAsync(counts_, 0, sizeof(unsigned), stream));
            Scatter<<<Blocks(count_), 256, 0, stream>>>(graph, cache, in_, count_,
                deltas_, false, 1, Target(residual, epsilon));
            GROUTE_CUDA_CHECK(cudaGetLastError());
            count_ = ReadCount(stream);
            std::swap(in_, out_);
            ++rounds;
        }
        return rounds;
    }
};
}} // namespace sepgraph::pr
#endif
