// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <framework/variants/api.cuh>
#include "hybrid_pr_common.h"
#include <utils/communication_meter.h>
#include <utils/communication_window.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <map>

// Priority
DEFINE_int32(pr_max_batches, 10, "Maximum PR update batches (0 for static PR)");
DEFINE_int32(pr_max_rounds, 100, "Maximum PR rounds per initialization/batch (1..100), with early convergence");
DEFINE_double(error, 0.000001, "Absolute signed PR residual threshold");
DEFINE_bool(sparse, false, "Compatibility flag; PR always uses sparse residual frontiers");
DECLARE_bool(check);
DECLARE_string(updatefile);
DECLARE_string(update_size);
DECLARE_string(output);
DECLARE_string(graphfile);

namespace hybrid_pr
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct PageRank : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {

        /*
         * For get rid of compiler bug: It's strange that if base class has virtual function, we must add a member for subclass.
         *
         * Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
         */
        static constexpr bool kSignedResidual = true;
        double m_error;
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        PageRank(double error) : m_error(error)
        {
	      
        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return 0.0f;
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            return (1 - ALPHA);
        }

        __forceinline__ __host__
        __device__
                TBuffer

        GetIdentityElement() const override
        {
            return 0.0f;
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                TValue *value, TBuffer *residual) override {
            const TBuffer r = atomicExch(residual, TBuffer(0));
            *value += r;
            const auto degree = this->m_vcsr_graph.degree(node);
            return {degree ? TBuffer(ALPHA * r / degree) : TBuffer(0), degree && r != 0};
        }
        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node, int *sign,
                TValue *value, TBuffer *) override {
            const auto degree = this->m_vcsr_graph.degree(node);
            return {degree ? TBuffer(ALPHA * *value * sign[0] / degree) : TBuffer(0),
                    degree && *value != 0};
        }

        __forceinline__ __device__
        int AccumulateBuffer(index_t, index_t, TWeight, TValue *,
                             TBuffer *residual, TBuffer delta) override {
            atomicAdd(residual, delta);
            return delta != 0;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer, TValue value) const override
        {
            return fabsf(buffer) > m_error;
        }

        __forceinline__ __device__

        bool IsActiveNode_INC(index_t node, TBuffer buffer, TValue value) const override
        {
            return fabsf(buffer) > m_error;
        }
        
        __forceinline__ __device__

        TValue sum_value(index_t node, TValue value, TBuffer buffer) const override
        {
            return fabsf(buffer);
        }

        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return fabsf(buffer) >= current_priority;
        }
    };
}

bool HybridPageRank() {
    try {
        if (FLAGS_graphfile.empty()) throw std::invalid_argument("PR requires graphfile");
        if (!std::isfinite(FLAGS_error) || FLAGS_error <= 0 ||
            !std::isfinite(static_cast<float>(FLAGS_error)) || static_cast<float>(FLAGS_error) == 0 ||
            FLAGS_pr_max_batches < 0 || FLAGS_pr_max_rounds <= 0 || FLAGS_pr_max_rounds > 100)
            throw std::invalid_argument("PR requires finite error > 0, pr_max_batches >= 0 and 1 <= pr_max_rounds <= 100");
        if (FLAGS_updatefile.empty() != FLAGS_update_size.empty())
            throw std::invalid_argument("PR requires both updatefile and update_size, or neither");
        if (cgcomm::WindowEnabled() && FLAGS_check)
            throw std::invalid_argument("CG_COMM_WINDOW requires --check=false; validate separately");
        using Engine = sepgraph::engine::Engine<rank_t, rank_t, rank_t, hybrid_pr::PageRank, double>;
        Engine engine(sepgraph::policy::AlgoType::ITERATIVE_SCHEME);
        engine.LoadGraph();
        sepgraph::common::EngineOptions options;
        options.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH,
                                 sepgraph::common::LoadBalancing::COARSE_GRAINED);
        engine.SetOptions(options);
        engine.InitGraph(FLAGS_error);
        sepgraph::pr::Runtime runtime(engine.GetGraphDatum().nnodes);
        Stopwatch initial(true);
        engine.StartPageRank(runtime, FLAGS_error, FLAGS_pr_max_rounds);
        initial.stop();
        LOG("[PR-INITIAL] compute_ms=%.3f alpha=0.85 base=0.15 dangling=drop\n", initial.ms());
        auto check = [&](const char *stage, int batch) {
            engine.GatherValue();
            engine.GatherBuffer();
            return PageRankCheck(engine.ChunkStore(), engine.GetGraphDatum().host_value,
                engine.GetGraphDatum().host_buffer, FLAGS_error, stage, batch);
        };
        if (FLAGS_check && !check("initial", -1)) return false;
        auto cache = [&]() {
            engine.compute_hot_vertices_pr();
            engine.confirm_candidate_batch();
            if (engine.CacheRefreshRequired()) {
                engine.evication_cache();
                engine.compact_cache();
                engine.LoadCache();
                engine.MarkCachePublished();
            }
        };
        // Initial cache load has no old cache to evict/compact.
        engine.compute_hot_vertices_pr();
        engine.confirm_candidate_batch();
        engine.LoadCache();
        engine.MarkCachePublished();
        unsigned batches = 0;
        if (FLAGS_pr_max_batches && !FLAGS_updatefile.empty()) {
            engine.get_update_file();
            batches = std::min<size_t>(FLAGS_pr_max_batches, engine.GetUpdateBatchCount());
        }
        std::pair<index_t,index_t> offset{0, 0};
        cgcomm::Flush(-1, "initialization");
        if (cgcomm::WindowEnabled()) cgcomm::Window("begin");
        for (index_t batch = 0; batch < batches; ++batch) {
            Stopwatch timer(true);
            engine.UpdatePageRank(runtime, offset, batch, FLAGS_error, FLAGS_pr_max_rounds);
            cache();
            GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            LOG("[TIMER][PR][batch %u] paper_algorithm_ms=%.3f\n", batch, timer.ms());
            cgcomm::Flush(batch, "pr_update_and_cache");
            if (FLAGS_check && !check("batch", batch)) return false;
            cgcomm::Flush(batch, "batch_check");
        }
        if (cgcomm::WindowEnabled()) cgcomm::Window("end");
        engine.GatherValue();
        engine.GatherBuffer();
        const auto &ranks = engine.GetGraphDatum().host_value;
        const auto &residual = engine.GetGraphDatum().host_buffer;
        if (FLAGS_check && !PageRankCheck(engine.ChunkStore(), ranks, residual,
                                         FLAGS_error, "final", batches)) return false;
        if (!FLAGS_output.empty() && !PageRankOutput(FLAGS_output.c_str(), ranks, residual)) return false;
        LOG("[PR-FINAL] vertices=%zu batches=%u\n", ranks.size(), batches);
        cgcomm::Flush(-1, "final_output_and_check");
        return true;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[PR-ERROR] %s\n", error.what());
        return false;
    }
}
