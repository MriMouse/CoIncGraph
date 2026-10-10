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
#include "hybrid_sssp_common.h"
#include "../../include/groute/graphs/csr_graph.cuh"

DEFINE_int32(source_node,
             0, "The source node for the SSSP traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse,
            false, "use async/push/dd + fusion for high-diameter");
DEFINE_int32(sssp_max_batches,
             10, "Maximum number of update batches processed by hybrid_sssp");
DEFINE_bool(sssp_print_checksum,
            false, "Print final SSSP distance and parent checksums");
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_int32(prio_delta);

namespace hybrid_sssp
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct SSSP : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        static constexpr bool kSupportsGpuDeletionRepair = true;

        __host__ __device__ static TWeight DeletionEdgeWeight(index_t src, index_t dst) {
            return static_cast<TWeight>((src + dst) % 128 + 1);
        }

        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;

        SSSP(index_t source_node) : m_source_node(source_node)
        {

        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            TBuffer buffer;
            if (node == m_source_node)
            {
                buffer = 0;
            }
            else
            {
                buffer = IDENTITY_ELEMENT;
            }
            return buffer;
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

        // 5: repeat
        
        // 7: until (new_path!= result[dst]||old_p==parent[dst])
        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            
                             TWeight weight,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TBuffer buffer) override   //src_buffer
        {
            
            TBuffer new_buffer = buffer + weight;
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
            TValue new_buffer = incoming_value_curr + weight;
            TValue new_parent = src;
            
            TValue old_value;
            old_value = atomicMin(p_buffer, new_buffer);
            TValue old_parent = *p_parent;
        
            
            if (new_buffer <= old_value) {
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
            if(value > buffer * 2)
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

struct BellmanRelaxCheckResult {
    uint64_t relaxable_edges = 0;
    uint64_t missing_tight_witnesses = 0;
    index_t first_src = UINT32_MAX;
    index_t first_dst = UINT32_MAX;
    index_t first_missing_witness = UINT32_MAX;
    distance_t src_distance = UINT32_MAX;
    distance_t dst_distance = UINT32_MAX;
    uint64_t candidate_distance = std::numeric_limits<uint64_t>::max();
};

struct ParentWitnessCheckResult {
    uint64_t invalid_vertices = 0;
    uint64_t missing_parent_edges = 0;
    uint64_t wrong_parent_distances = 0;
    index_t first_vertex = UINT32_MAX;
    index_t first_parent = UINT32_MAX;
    distance_t vertex_distance = UINT32_MAX;
    distance_t parent_distance = UINT32_MAX;
};

static BellmanRelaxCheckResult CheckBellmanRelaxed(
        const sepgraph::topology::SourceLocalChunkStore &graph,
        const std::vector<distance_t> &distances,
        index_t source_node) {
    BellmanRelaxCheckResult result;

    const index_t nnodes = static_cast<index_t>(distances.size());
    std::vector<uint8_t> has_tight_witness(nnodes, 0);
    if (source_node < nnodes) {
        has_tight_witness[source_node] = 1;
    }
    for (index_t src = 0; src < nnodes; src++) {
        if (distances[src] == UINT32_MAX) {
            continue;
        }

        const auto &descriptor = graph.Descriptor(src);
        const uint64_t edge_start = descriptor.index;
        const uint64_t degree = descriptor.degree;
        for (uint64_t edge_offset = 0; edge_offset < degree; edge_offset++) {
            const index_t dst = graph.SlabData(descriptor.slab_id)[edge_start + edge_offset];
            if (dst >= distances.size()) {
                continue;
            }

            const uint64_t weight = (static_cast<uint64_t>(src) + dst) % 128 + 1;
            const uint64_t candidate = static_cast<uint64_t>(distances[src]) + weight;
            if (distances[dst] != UINT32_MAX && candidate == distances[dst]) {
                has_tight_witness[dst] = 1;
            }
            if (candidate < distances[dst]) {
                if (result.relaxable_edges == 0) {
                    result.first_src = src;
                    result.first_dst = dst;
                    result.src_distance = distances[src];
                    result.dst_distance = distances[dst];
                    result.candidate_distance = candidate;
                }
                result.relaxable_edges++;
            }
        }
    }
    for (index_t node = 0; node < nnodes; ++node) {
        if (distances[node] == UINT32_MAX || has_tight_witness[node]) {
            continue;
        }
        if (result.missing_tight_witnesses == 0) {
            result.first_missing_witness = node;
        }
        result.missing_tight_witnesses++;
    }

    return result;
}

static ParentWitnessCheckResult CheckParentWitness(
        const sepgraph::topology::SourceLocalChunkStore &graph,
        const std::vector<distance_t> &distances,
        const std::vector<distance_t> &parents,
        index_t source_node) {
    ParentWitnessCheckResult result;
    const index_t nnodes = static_cast<index_t>(
        std::min(distances.size(), parents.size()));
    for (index_t node = 0; node < nnodes; ++node) {
        if (node == source_node || distances[node] == UINT32_MAX) {
            continue;
        }

        const index_t parent = parents[node];
        bool valid = parent < nnodes && distances[parent] != UINT32_MAX;
        bool parent_edge_exists = false;
        if (valid) {
            const auto &descriptor = graph.Descriptor(parent);
            const uint64_t edge_start = descriptor.index;
            const uint64_t degree = descriptor.degree;
            for (uint64_t edge_offset = 0; edge_offset < degree; ++edge_offset) {
                if (graph.SlabData(descriptor.slab_id)[edge_start + edge_offset] == node) {
                    parent_edge_exists = true;
                    break;
                }
            }
            const uint64_t weight =
                (static_cast<uint64_t>(parent) + node) % 128 + 1;
            const bool distance_matches =
                static_cast<uint64_t>(distances[parent]) + weight == distances[node];
            if (!parent_edge_exists) {
                result.missing_parent_edges++;
            } else if (!distance_matches) {
                result.wrong_parent_distances++;
            }
            valid = parent_edge_exists && distance_matches;
        }

        if (!valid) {
            if (result.invalid_vertices == 0) {
                result.first_vertex = node;
                result.first_parent = parent;
                result.vertex_distance = distances[node];
                result.parent_distance =
                    parent < distances.size() ? distances[parent] : UINT32_MAX;
            }
            result.invalid_vertices++;
        }
    }
    return result;
}

static uint64_t DistanceChecksum(const std::vector<distance_t> &distances) {
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


/**
 * Δ = cw/d,
    where d is the average degree in the graph, w is the average
    edge weight, and c is the warp width (32 on our GPUs).
 * @return
 */

bool HybridSSSP()
{
    if (cgcomm::Enabled()) {
        LOG("[COMM-CONTRACT] version=1 payload=successful_cuda_api_requested_bytes async=accepted coverage=cudaMemcpy,cudaMemcpyAsync physical_bytes=unavailable zc_logical_accesses=unavailable zc_cache_classification=unavailable cpu_memcpy_bytes=unavailable kernel_d2d_bytes=unavailable\n");
    }
    assert(UINT32_MAX == UINT_MAX);
    
    typedef sepgraph::engine::Engine<distance_t, distance_t, distance_t, hybrid_sssp::SSSP, index_t> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);
    engine.LoadGraph();


    index_t source_node = min(max((index_t) 0, (index_t) FLAGS_source_node), engine.GetGraphDatum().nnodes - 1);

    sepgraph::common::EngineOptions engine_opt;


    const groute::graphs::host::PMAGraph &vcsr_graph = engine.PMAGraph();
    double weight_sum = 0;

    /**
     * We select a similar heuristic, Δ = cw/d,
        where d is the average degree in the graph, w is the average
        edge weight, and c is the warp width (32 on our GPUs)
        Link: https://people.csail.mith.edu/jshun/papers/DBGO14.pdf
     */
    
    
    int init_prio = 32 * (weight_sum / vcsr_graph.nedges) /
                (1.0 * vcsr_graph.nedges / vcsr_graph.nnodes);

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
    
    engine.get_update_file();
    std::pair<index_t,index_t> local_begin;
    local_begin.first = 0; //first is add
    local_begin.second = 0; // second is del
    index_t NumOfSnapShots = 0;
    const index_t available_batches = static_cast<index_t>(engine.GetUpdateBatchCount());
    const index_t max_batches = std::min<index_t>(FLAGS_sssp_max_batches, available_batches);
    if (FLAGS_sssp_max_batches > available_batches) {
        std::cout << "[SSSP] clamp sssp_max_batches from " << FLAGS_sssp_max_batches
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
            engine.GatherParent();
            const auto &delete_stage_distances = engine.GetGraphDatum().host_value;
            const auto &delete_stage_parents = engine.GetGraphDatum().host_parent;
            const BellmanRelaxCheckResult delete_stage_result =
                CheckBellmanRelaxed(engine.ChunkStore(), delete_stage_distances, source_node);
            const ParentWitnessCheckResult delete_parent_result =
                CheckParentWitness(engine.ChunkStore(),
                                   delete_stage_distances,
                                   delete_stage_parents,
                                   source_node);
            const bool delete_source_ok =
                source_node < delete_stage_distances.size() &&
                delete_stage_distances[source_node] == 0;
            if (!delete_source_ok || delete_stage_result.relaxable_edges != 0 ||
                delete_stage_result.missing_tight_witnesses != 0) {
                success = false;
            }
            LOG("[SSSP-DELETE-STAGE-CHECK][batch %d] %s source_ok=%d relaxable_edges=%llu missing_tight_witnesses=%llu invalid_parent_witness=%llu missing_parent_edges=%llu wrong_parent_distances=%llu first_src=%u first_dst=%u first_missing_witness=%u first_bad_vertex=%u first_bad_parent=%u distance_checksum=%llu\n",
                NumOfSnapShots,
                delete_source_ok && delete_stage_result.relaxable_edges == 0 &&
                        delete_stage_result.missing_tight_witnesses == 0
                    ? "passed"
                    : "failed",
                delete_source_ok ? 1 : 0,
                static_cast<unsigned long long>(delete_stage_result.relaxable_edges),
                static_cast<unsigned long long>(delete_stage_result.missing_tight_witnesses),
                static_cast<unsigned long long>(delete_parent_result.invalid_vertices),
                static_cast<unsigned long long>(delete_parent_result.missing_parent_edges),
                static_cast<unsigned long long>(delete_parent_result.wrong_parent_distances),
                delete_stage_result.first_src,
                delete_stage_result.first_dst,
                delete_stage_result.first_missing_witness,
                delete_parent_result.first_vertex,
                delete_parent_result.first_parent,
                static_cast<unsigned long long>(DistanceChecksum(delete_stage_distances)));
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
        LOG("[ATTR][SSSP][batch %d] deletion=%.3f add=%.3f hotness=%.3f candidate=%.3f eviction=%.3f compact=%.3f cache_load=%.3f residual=%.3f total=%.3f\n",
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
        LOG("[TIMER][SSSP][batch %d] total_batch: %.3f ms\n",
            NumOfSnapShots,
            paper_algorithm_ms);
        if (FLAGS_check) {
            engine.GatherValue();
            engine.GatherParent();
            const auto &batch_distances = engine.GetGraphDatum().host_value;
            const auto &batch_parents = engine.GetGraphDatum().host_parent;
            const BellmanRelaxCheckResult batch_bellman =
                CheckBellmanRelaxed(engine.ChunkStore(), batch_distances, source_node);
            const ParentWitnessCheckResult batch_parent =
                CheckParentWitness(engine.ChunkStore(),
                                   batch_distances,
                                   batch_parents,
                                   source_node);
            const bool batch_source_ok =
                source_node < batch_distances.size() && batch_distances[source_node] == 0;
            const bool batch_ok = batch_source_ok &&
                                  batch_bellman.relaxable_edges == 0 &&
                                  batch_bellman.missing_tight_witnesses == 0;
            if (!batch_ok) {
                success = false;
            }
            LOG("[SSSP-BATCH-CHECK][batch %d] %s source_ok=%d relaxable_edges=%llu missing_tight_witnesses=%llu invalid_parent_witness=%llu missing_parent_edges=%llu wrong_parent_distances=%llu distance_checksum=%llu\n",
                NumOfSnapShots,
                batch_ok ? "passed" : "failed",
                batch_source_ok ? 1 : 0,
                static_cast<unsigned long long>(batch_bellman.relaxable_edges),
                static_cast<unsigned long long>(batch_bellman.missing_tight_witnesses),
                static_cast<unsigned long long>(batch_parent.invalid_vertices),
                static_cast<unsigned long long>(batch_parent.missing_parent_edges),
                static_cast<unsigned long long>(batch_parent.wrong_parent_distances),
                static_cast<unsigned long long>(DistanceChecksum(batch_distances)));
        }
        cgcomm::Flush(NumOfSnapShots, "batch_check");
        NumOfSnapShots+=1;
    }
    if (cgcomm::WindowEnabled()) {
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        cgcomm::Window("end");
    }
    engine.GatherValue();
    engine.GatherParent();
    engine.GatherBuffer();
    const auto &distances = engine.GetGraphDatum().host_value;
    cgcomm::Result(distances);
    const auto &parents = engine.GetGraphDatum().host_parent;
    const auto &deltas = engine.GetGraphDatum().host_buffer;
    uint64_t reachable_count = 0;
    for (index_t node = 0; node < distances.size(); node++) {
        if (distances[node] != UINT32_MAX) {
            reachable_count++;
        }
    }

    if (FLAGS_check) {
        BellmanRelaxCheckResult bellman_result =
            CheckBellmanRelaxed(engine.ChunkStore(), distances, source_node);
        bool source_ok = source_node < distances.size() && distances[source_node] == 0;
        if (source_ok && bellman_result.relaxable_edges == 0 &&
            bellman_result.missing_tight_witnesses == 0) {
            LOG("[SSSP-BELLMAN-CHECK] passed reachable=%llu relaxable_edges=0 missing_tight_witnesses=0\n",
                static_cast<unsigned long long>(reachable_count));
        } else {
            success = false;
            LOG("[SSSP-BELLMAN-CHECK] failed source_ok=%d relaxable_edges=%llu missing_tight_witnesses=%llu first_missing_witness=%u first_src=%u first_dst=%u src_dist=%u dst_dist=%u candidate=%llu\n",
                source_ok ? 1 : 0,
                static_cast<unsigned long long>(bellman_result.relaxable_edges),
                static_cast<unsigned long long>(bellman_result.missing_tight_witnesses),
                bellman_result.first_missing_witness,
                bellman_result.first_src,
                bellman_result.first_dst,
                bellman_result.src_distance,
                bellman_result.dst_distance,
                static_cast<unsigned long long>(bellman_result.candidate_distance));
        }
    }

    LOG("[SSSP-FINAL-CHECK] final_reachable=%llu checksum=%s\n",
        static_cast<unsigned long long>(reachable_count),
        FLAGS_sssp_print_checksum ? "enabled" : "disabled");

    if (FLAGS_sssp_print_checksum) {
        uint64_t distance_checksum = 1469598103934665603ull;
        uint64_t parent_checksum = 1099511628211ull;
        for (index_t node = 0; node < distances.size(); node++) {
            if (distances[node] == UINT32_MAX) {
                continue;
            }
            distance_checksum ^= static_cast<uint64_t>(node) + 0x9e3779b97f4a7c15ull +
                                 (static_cast<uint64_t>(distances[node]) << 6) +
                                 (static_cast<uint64_t>(distances[node]) >> 2);
            distance_checksum *= 1099511628211ull;
            parent_checksum ^= static_cast<uint64_t>(node) + 0x517cc1b727220a95ull +
                               (static_cast<uint64_t>(parents[node]) << 7) +
                               (static_cast<uint64_t>(parents[node]) >> 3);
            parent_checksum *= 1469598103934665603ull;
        }
        LOG("[SSSP-FINAL-CHECK] distance_checksum=%llu parent_checksum=%llu\n",
            static_cast<unsigned long long>(distance_checksum),
            static_cast<unsigned long long>(parent_checksum));
    }

    if (!FLAGS_output.empty()) {
        std::ofstream output(FLAGS_output, std::ios::out | std::ios::trunc);
        for (index_t node = 0; node < distances.size(); node++) {
            output << node << " " << distances[node] << " " << parents[node] << " " << deltas[node] << "\n";
        }
    }

    cudaDeviceSynchronize();
    cgcomm::Flush(-1, "final_output_and_check");
    return success;
}
