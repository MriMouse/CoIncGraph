// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <functional>
#include <utils/communication_meter.h>
#include <utils/communication_window.h>
#include <fstream>
#include <limits>
#include <map>
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <utils/cuda_utils.h>
#include "hybrid_cc_common.h"
#include "../../include/groute/graphs/csr_graph.cuh"

DEFINE_int32(source_node,
             0, "Compatibility flag; CC seeds every vertex, source_node is ignored");
DEFINE_bool(sparse,
            false, "use async/push/dd + fusion for high-diameter");
DEFINE_int32(cc_max_batches,
             10, "Maximum number of update batches processed by hybrid_cc");
DEFINE_bool(cc_print_checksum,
            false, "Print final CC label checksum");
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_int32(prio_delta);

namespace hybrid_cc
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct CC : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        static constexpr bool kSupportsGpuDeletionRepair = true;
        static constexpr bool kVertexSeeds = true;
        static constexpr bool kRootedDeletionWitness = true;

        // Equal-label cycles remain dependencies even with another tight parent.
        // A vertex's own seed survives every deletion.
        __host__ __device__ static bool IsDeletionDependency(
                index_t src, index_t dst, TValue source, TValue destination,
                uint64_t, index_t, bool) {
            return src != dst && source == destination && destination < dst;
        }

        __host__ __device__ static TWeight DeletionEdgeWeight(index_t src, index_t dst) {
            return TWeight(0);
        }

        __host__ __device__ static TWeight TraversalEdgeWeight(index_t, index_t) {
            return TWeight(0);
        }

        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        explicit CC(index_t) {}

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            return static_cast<TBuffer>(node);
        }

        __forceinline__ __host__ __device__
        TBuffer GetIdentityElement() const override
        {
            return IDENTITY_ELEMENT;
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node,int *type_device,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {

            return utils::pair<TBuffer, bool>(*p_buffer, true);
        }
        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = *p_buffer;
            bool schedule = false;
            if (*p_value > buffer)
            {
                *p_value = buffer;
                schedule = true;
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        // Forward the minimum label unchanged, including insertion updates.
        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            
                             TWeight weight,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TBuffer buffer) override   //src_buffer
        {
            
            if (buffer == IDENTITY_ELEMENT) return 0;
            TBuffer new_buffer = buffer;
            TBuffer old_buffer = atomicMin(p_buffer, new_buffer);
            if (new_buffer < old_buffer) {
                TValue old_parent = *p_parent;
                while (new_buffer == *p_buffer && *p_parent != src) {
                    old_parent = atomicCAS(p_parent, old_parent, src);
                }
            }
            return new_buffer < old_buffer ? 1 : 0;
        }

        __forceinline__ __device__
        int AccumulateBuffer_add(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            
                             TWeight weight,
                             TValue *p_parent, //dst_parent
                             TValue *p_buffer,    //dst_value
                             TValue *buffer) override   //src_value
        {
            TValue incoming_value_curr = *buffer;
            if(incoming_value_curr ==UINT32_MAX){
                return 0;
            }
            TValue new_buffer = incoming_value_curr;
            TValue new_parent = src;
            
            TValue old_value;
            old_value = atomicMin(p_buffer, new_buffer);
            TValue old_parent = *p_parent;
        
            
            if (new_buffer < old_value) {
                while (new_buffer == *p_buffer && *p_parent != src) {
                    old_parent = atomicCAS(p_parent,old_parent,src);
                }
            }
            return 1;
        }

        // __forceinline__ __device__
        // bool reduce()

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer,TValue value) const override
        {
            return buffer < value;
        }


        __forceinline__ __device__
        TValue sum_value(index_t node, TValue value,TBuffer buffer) const override
        {
            if(buffer < value)
                return TValue(2);

            return TValue(1);
        }


        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority > buffer;
        }
    };
}

static uint64_t LabelChecksum(const std::vector<label_t> &distances) {
    uint64_t checksum = 1469598103934665603ull;
    for (index_t node = 0; node < distances.size(); ++node) {
        if (distances[node] == UINT32_MAX) {
            continue;
        }
        checksum ^= static_cast<uint64_t>(node) + 0x9e3779b97f4a7c15ull +
                    (static_cast<uint64_t>(distances[node]) << 6) +
                    (static_cast<uint64_t>(distances[node]) >> 2);
        checksum *= 1099511628211ull;
    }
    return checksum;
}


// Directed minimum-label propagation, matching old_hybrid_cc.
// Keep the shared traversal, incremental repair, scheduling and cache paths.

bool HybridCC()
{
    if (FLAGS_cc_max_batches < 0) {
        throw std::invalid_argument("cc_max_batches must be nonnegative");
    }

    if (cgcomm::Enabled()) {
        LOG("[COMM-CONTRACT] version=1 payload=successful_cuda_api_requested_bytes async=accepted coverage=cudaMemcpy,cudaMemcpyAsync physical_bytes=unavailable zc_logical_accesses=unavailable zc_cache_classification=unavailable cpu_memcpy_bytes=unavailable kernel_d2d_bytes=unavailable\n");
    }
    assert(UINT32_MAX == UINT_MAX);
    typedef sepgraph::engine::Engine<label_t, label_t, label_t, hybrid_cc::CC, index_t> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);
    engine.LoadGraph();
    LOG("[CC-SEMANTICS] directed_min_label input_edges=as_is updates=as_is\n");


    if (engine.GetGraphDatum().nnodes == 0) {
        throw std::invalid_argument("CC requires a nonempty graph");
    }
    index_t source_node = 0; 

    sepgraph::common::EngineOptions engine_opt;


    const int init_prio = FLAGS_prio_delta > 0 ? FLAGS_prio_delta : 1;

    printf("Priority delta: %u\n", init_prio);

    if (FLAGS_sparse)
    {
        engine_opt.SetFused();
        engine_opt.SetTwoLevelBasedPriority(init_prio);
        engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_DD);
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH, sepgraph::common::LoadBalancing::NONE);
    }

    if (FLAGS_prio_delta > 0)
    {
        LOG("Enable priority for scale-free dataset\n");
        engine_opt.SetTwoLevelBasedPriority(FLAGS_prio_delta);
    }

    engine.SetOptions(engine_opt);
    engine.InitGraph(source_node);
    engine.Start(init_prio);
    //PrintCacheNode() are used to detect the cache is right or not.
    bool success = true;
    engine.compute_hot_vertices_sssp();
    engine.confirm_candidate_batch();
    engine.LoadCache();
    engine.MarkCachePublished();
    
    if (FLAGS_updatefile.empty() != FLAGS_update_size.empty())
        throw std::invalid_argument("Provide both updatefile and update_size, or neither");
    const bool load_updates = FLAGS_cc_max_batches > 0 && !FLAGS_updatefile.empty();
    if (load_updates && !FLAGS_weight)
        throw std::invalid_argument("CC update reader requires --weight=true; numeric weights are ignored");
    if (load_updates) engine.get_update_file();
    std::pair<index_t,index_t> local_begin;
    local_begin.first = 0; //first is add
    local_begin.second = 0; // second is del
    index_t NumOfSnapShots = 0;
    const index_t available_batches = load_updates ? static_cast<index_t>(engine.GetUpdateBatchCount()) : 0;
    const index_t max_batches = std::min<index_t>(FLAGS_cc_max_batches, available_batches);
    if (FLAGS_cc_max_batches > available_batches) {
        std::cout << "[CC] clamp cc_max_batches from " << FLAGS_cc_max_batches
                  << " to available update batches " << available_batches << std::endl;
    }
    cgcomm::Flush(-1, "initialization");
    if (cgcomm::WindowEnabled()) {
        if (FLAGS_check) {
            std::fprintf(stderr, "CG_COMM_WINDOW requires --check=false and disabled diagnostic traces; validate separately\n");
            std::exit(EXIT_FAILURE);
        }
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        cgcomm::Window("begin");
    }
    while(true){
        if(NumOfSnapShots==max_batches) break;
        std::cout<<"batch number "<<NumOfSnapShots<<std::endl;
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        Stopwatch sw_paper_batch(true);
        Stopwatch sw_delete_stage(true);
        engine.del_edge(local_begin,NumOfSnapShots);
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        sw_delete_stage.stop();
        sw_paper_batch.stop();
        double paper_algorithm_ms = sw_paper_batch.ms();
        cgcomm::Flush(NumOfSnapShots, "deletion");
        if (FLAGS_check) {
            engine.GatherValue();
            const auto &labels = engine.GetGraphDatum().host_value;
            const auto errors = CCCheck(engine.ChunkStore(), labels);
            success = success && errors == 0;
            LOG("[CC-DELETE-STAGE-CHECK][batch %u] %s errors=%llu label_checksum=%llu\n",
                NumOfSnapShots, errors == 0 ? "passed" : "failed",
                (unsigned long long)errors, (unsigned long long)LabelChecksum(labels));
        }
        cgcomm::Flush(NumOfSnapShots, "delete_check");
        cgcomm::Stage(1);
        sw_paper_batch.start();
        Stopwatch sw_add_stage(true);
        engine.add_edge(local_begin,NumOfSnapShots);
        sw_add_stage.stop();
        // Defer ledger output until the paper timer has stopped.
        Stopwatch sw_hotness(true);
        cgcomm::Stage(2);
        engine.compute_hot_vertices_sssp();
        sw_hotness.stop();
        Stopwatch sw_candidate(true);
        cgcomm::Stage(3);
        engine.confirm_candidate_batch();
        sw_candidate.stop();
        cgcomm::Stage(4);
        Stopwatch sw_eviction(true);
        cgcomm::Stage(5);
        const bool refresh_cache = engine.CacheRefreshRequired();
        if (refresh_cache) engine.evication_cache();
        sw_eviction.stop();
        Stopwatch sw_compact(true);
        cgcomm::Stage(6);
        if (refresh_cache) engine.compact_cache();
        sw_compact.stop();
        Stopwatch sw_cache_load(true);
        cgcomm::Stage(7);
        if (refresh_cache) {
            engine.LoadCache();
            engine.MarkCachePublished();
        }
        sw_cache_load.stop();
        LOG("[CACHE-PUBLISH] batch=%u refresh=%u eviction_ms=%.3f compact_ms=%.3f load_ms=%.3f\n",
            NumOfSnapShots, refresh_cache ? 1U : 0U, sw_eviction.ms(),
            sw_compact.ms(), sw_cache_load.ms());
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        sw_paper_batch.stop();
        paper_algorithm_ms += sw_paper_batch.ms();
        cgcomm::Flush(NumOfSnapShots, "addition_and_cache");
        cgcomm::Stage(0);
        const double attributed_ms =
            sw_delete_stage.ms() + sw_add_stage.ms() + sw_hotness.ms() +
            sw_candidate.ms() + sw_eviction.ms() + sw_compact.ms() +
            sw_cache_load.ms();
        LOG("[ATTR][CC][batch %d] deletion=%.3f add=%.3f hotness=%.3f candidate=%.3f eviction=%.3f compact=%.3f cache_load=%.3f residual=%.3f total=%.3f\n",
            NumOfSnapShots,
            sw_delete_stage.ms(),
            sw_add_stage.ms(),
            sw_hotness.ms(),
            sw_candidate.ms(),
            sw_eviction.ms(),
            sw_compact.ms(),
            sw_cache_load.ms(),
            paper_algorithm_ms - attributed_ms,
            paper_algorithm_ms);
        LOG("[TIMER][CC][batch %d] total_batch: %.3f ms\n",
            NumOfSnapShots,
            paper_algorithm_ms);
        if (FLAGS_check) {
            engine.GatherValue();
            const auto &labels = engine.GetGraphDatum().host_value;
            const auto errors = CCCheck(engine.ChunkStore(), labels);
            success = success && errors == 0;
            LOG("[CC-BATCH-CHECK][batch %u] %s errors=%llu label_checksum=%llu\n",
                NumOfSnapShots, errors == 0 ? "passed" : "failed",
                (unsigned long long)errors, (unsigned long long)LabelChecksum(labels));
        }
        cgcomm::Flush(NumOfSnapShots, "batch_check");
        NumOfSnapShots+=1;
    }
    if (cgcomm::WindowEnabled()) {
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        cgcomm::Window("end");
    }
    engine.GatherValue();
    const auto &labels = engine.GetGraphDatum().host_value;
    cgcomm::Result(labels);
    if (FLAGS_check) {
        const auto errors = CCCheck(engine.ChunkStore(), labels);
        success = success && errors == 0;
        LOG("[CC-FINAL-CHECK] %s errors=%llu\n", errors == 0 ? "passed" : "failed",
            (unsigned long long)errors);
    }
    if (FLAGS_cc_print_checksum)
        LOG("[CC-FINAL-CHECK] label_checksum=%llu\n", (unsigned long long)LabelChecksum(labels));
    if (!FLAGS_output.empty()) success = CCOutput(FLAGS_output.c_str(), labels) && success;
    GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
    cgcomm::Flush(-1, "final_output_and_check");
    return success;
}
