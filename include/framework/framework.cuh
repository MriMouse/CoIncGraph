// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_FRAMEWORK_H
#define HYBRID_FRAMEWORK_H

#include <map>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>
#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <framework/common.h>
#include <framework/variants/api.cuh>
#include <framework/graph_datum.cuh>
#include <framework/variants/common.cuh>
#include <framework/variants/driver.cuh>
#include <framework/hybrid_policy.h>
#include <framework/dynamic_reverse_index.h>
#include <framework/directed_label_witness.h>
#include <framework/publication_sources.h>
#include <framework/topology_replay.h>
#include <framework/cache_refresh_gate.h>
#include <framework/cc_union_repair.cuh>
#include <framework/pr_residual.cuh>
#include <framework/ordered_gpu_repair.cuh>
#include <framework/algo_variants.cuh>
#include <utils/cuda_utils.h>
#include <utils/graphs/traversal.h>
#include <utils/to_json.h>
#include <groute/device/work_source.cuh>
#include "clion_cuda.cuh"
#include <groute/rmat_util.h>
#include "Loader.h"
#include <unordered_set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

DECLARE_int32(priority_a);
DECLARE_int32(hybrid);
DECLARE_int32(SEGMENT);
DECLARE_int32(n_stream);
DECLARE_string(out_wl);
DECLARE_string(lb_push);
DECLARE_string(lb_pull);
DECLARE_double(alpha);
DECLARE_bool(verbose);

DECLARE_double(edge_factor);
DECLARE_string(updatefile);
DECLARE_string(update_size);
DECLARE_bool(weight);
DECLARE_bool(check);

namespace sepgraph {
    namespace engine {
        using common::Priority;
        using common::LoadBalancing;
        using common::Scheduling;
        using common::Model;
        using common::MsgPassing;
        using common::AlgoVariant;
        using policy::AlgoType;
        using policy::PolicyDecisionMaker;
        using utils::JsonWriter;


        struct Algo {
            static const char *Name() {
                return "Hybrid Graph Engine";
            }
        };

        // CC labels have zero-cost cycles: invalidate whole old components,
        
        // the three kernels (mark, collect, reset) are separated by stream order.
        template<class Edge, class Value, class Buffer>
        __global__ void MarkDeletedComponents(const Edge *edges, uint32_t start,
                uint32_t count, uint32_t nodes, const Value *labels, Buffer *marks) {
            for (uint32_t i = TID_1D; i < count; i += TOTAL_THREADS_1D) {
                const auto e = edges[start + i];
                if (e.u >= nodes || e.v >= nodes || e.u == e.v) continue;
                // Different old components cannot contain this edge: a missing
                // deletion cannot split either component.
                if (labels[e.u] == labels[e.v])
                    atomicExch(&marks[labels[e.u]], UINT32_MAX);
            }
        }

        template<class Value, class Buffer>
        __global__ void CollectDeletedComponents(uint32_t nodes, const Value *labels,
                const Buffer *marks, Value *parents, bool *reset,
                groute::dev::Queue<index_t> affected) {
            for (uint32_t v = TID_1D; v < nodes; v += TOTAL_THREADS_1D) {
                if (marks[labels[v]] == UINT32_MAX) {
                    reset[v] = true;
                    parents[v] = UINT32_MAX;
                    affected.append(v);
                }
            }
        }

        template<class Edge, class Value>
        __global__ void MarkCcDeletionCandidates(const Edge* edges, uint32_t start,
                uint32_t count, const Value* labels, uint8_t* excluded) {
            for (uint32_t i = TID_1D; i < count; i += TOTAL_THREADS_1D) {
                const auto edge = edges[start + i];
                excluded[i] = edge.u == edge.v || labels[edge.u] != labels[edge.v] ||
                    labels[edge.v] == edge.v;
            }
        }

        template<class Value>
        __global__ void MarkCcWitnessAffected(const index_t* affected, uint32_t count,
                Value* parents, bool* reset) {
            for (uint32_t i = TID_1D; i < count; i += TOTAL_THREADS_1D) {
                parents[affected[i]] = UINT32_MAX;
                reset[affected[i]] = true;
            }
        }

        template<typename TAppInst, typename TValue, typename TBuffer>
        __global__ void GpuAffectedPullRelax(TAppInst app_inst,
                                             const index_t *affected_vertices,
                                             uint32_t affected_count,
                                             const uint64_t *incoming_offsets,
                                             const index_t *incoming_sources,
                                             TValue *node_value,
                                             TBuffer *node_buffer,
                                             TValue *node_parent,
                                             TValue infinity,
                                             unsigned int *changed_count) {
            static_assert(sizeof(TValue) == sizeof(unsigned int),
                          "GPU affected repair requires 32-bit values");
            const uint32_t tid = TID_1D;
            const uint32_t nthreads = TOTAL_THREADS_1D;
            // Cooperate on long incoming rows instead of serializing each row
            // in one lane. Unusual non-warp-sized blocks retain scalar groups.
            const uint32_t width = blockDim.x % 32 == 0 ? 32 : 1;
            const uint32_t lane = threadIdx.x % width;
            for (uint32_t i = tid / width; i < affected_count; i += nthreads / width) {
                const index_t dst = affected_vertices[i];

                TValue best = static_cast<TValue>(atomicAdd(
                    reinterpret_cast<unsigned int *>(&node_value[dst]), 0));
                index_t best_parent = node_parent[dst];
                for (uint64_t edge = incoming_offsets[i] + lane;
                     edge < incoming_offsets[i + 1]; edge += width) {
                    const index_t src = incoming_sources[edge];
                    const TValue src_value = static_cast<TValue>(atomicAdd(
                        reinterpret_cast<unsigned int *>(&node_value[src]), 0));
                    if (src_value == infinity) {
                        continue;
                    }
                    const uint64_t candidate_wide =
                        static_cast<uint64_t>(src_value) +
                        static_cast<uint64_t>(TAppInst::DeletionEdgeWeight(src, dst));
                    if (candidate_wide < static_cast<uint64_t>(best) ||
                        (candidate_wide == static_cast<uint64_t>(best) &&
                         (best_parent == UINT32_MAX || src < best_parent))) {
                        best = static_cast<TValue>(candidate_wide);
                        best_parent = src;
                    }
                }
                if (width == 32) {
                    for (uint32_t step = 16; step != 0; step >>= 1) {
                        const TValue other = __shfl_down_sync(0xffffffffU, best, step);
                        const index_t parent = __shfl_down_sync(0xffffffffU, best_parent, step);
                        if (other < best || (other == best && parent < best_parent)) {
                            best = other;
                            best_parent = parent;
                        }
                    }
                }
                if (lane != 0) continue;
                const TValue old_value = static_cast<TValue>(atomicMin(
                    reinterpret_cast<unsigned int *>(&node_value[dst]),
                    static_cast<unsigned int>(best)));
                if (best <= old_value && best_parent != UINT32_MAX) {
                    atomicExch(reinterpret_cast<unsigned int *>(&node_buffer[dst]),
                               static_cast<unsigned int>(best));
                    atomicExch(reinterpret_cast<unsigned int *>(&node_parent[dst]),
                               static_cast<unsigned int>(best_parent));
                }
                if (best < old_value) {
                    atomicAdd(changed_count, 1u);
                }
            }
        }

        template<typename TValue, typename TBuffer>
        __global__ void FinalizeGpuAffectedRepair(const index_t *affected_vertices,
                                                  uint32_t affected_count,
                                                  const TValue *node_value,
                                                  TBuffer *node_buffer,
                                                  bool *node_reset) {
            const uint32_t tid = TID_1D;
            const uint32_t nthreads = TOTAL_THREADS_1D;
            for (uint32_t i = tid; i < affected_count; i += nthreads) {
                const index_t node = affected_vertices[i];
                node_buffer[node] = static_cast<TBuffer>(node_value[node]);
                node_reset[node] = false;
            }
        }

        template<typename TValue>
        __device__ TValue AtomicMinState(TValue *address, TValue value) {
            static_assert(sizeof(TValue) == sizeof(uint32_t),
                          "AtomicMinState requires a 32-bit state type");
            uint32_t *bits = reinterpret_cast<uint32_t *>(address);
            uint32_t observed_bits = *bits;
            while (value < *reinterpret_cast<TValue *>(&observed_bits)) {
                const uint32_t desired_bits = *reinterpret_cast<uint32_t *>(&value);
                const uint32_t previous_bits = atomicCAS(bits, observed_bits, desired_bits);
                if (previous_bits == observed_bits) {
                    break;
                }
                observed_bits = previous_bits;
            }
            return *reinterpret_cast<TValue *>(&observed_bits);
        }

        template<typename TValue, typename TWEdge>
        __global__ void SeedAddedEdgeFrontier(const TWEdge *added_edges,
                                              const uint32_t *work_size,
                                              const TValue *node_value,
                                              TValue *node_buffer,
                                              TValue *node_parent,
                                              TValue infinity,
                                              BitmapDeviceObject out_active,
                                              const index_t *segment_end_nodes,
                                              uint32_t segment_count,
                                              uint32_t epoch,
                                              uint32_t *node_state_epoch,
                                              unsigned long long *seed_reject_count,
                                              groute::dev::Queue<index_t> exact_sources) {
            const uint32_t tid = TID_1D;
            const uint32_t nthreads = TOTAL_THREADS_1D;
            const uint32_t begin = work_size[0];
            const uint32_t count = work_size[1];
            for (uint32_t offset = tid; offset < count; offset += nthreads) {
                const TWEdge edge = added_edges[begin + offset];
                uint32_t lo = 0;
                uint32_t hi = segment_count;
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo) / 2;
                    if (edge.v < segment_end_nodes[mid]) {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                if (lo >= segment_count) {
                    if (seed_reject_count != nullptr) {
                        atomicAdd(seed_reject_count, 1ULL);
                    }
                    continue;
                }
                if (node_value[edge.u] == infinity) {
                    continue;
                }
                const uint64_t candidate_wide =
                    static_cast<uint64_t>(node_value[edge.u]) +
                    static_cast<uint64_t>(edge.w);
                if (candidate_wide >= static_cast<uint64_t>(infinity)) {
                    continue;
                }
                const TValue candidate = static_cast<TValue>(candidate_wide);
                const TValue old_buffer = AtomicMinState(&node_buffer[edge.v], candidate);
                if (candidate < old_buffer && candidate == node_buffer[edge.v]) {
                    atomicExch(reinterpret_cast<uint32_t *>(&node_parent[edge.v]), edge.u);
                    const uint32_t seed_ticket = epoch << 16;
                    if (atomicExch(node_state_epoch + edge.v, seed_ticket) !=
                        seed_ticket) {
                        exact_sources.append(edge.v);
                    }
                }
            }
        }

        template<typename TAppInst, typename TPMAGraph, typename TValue,
                 typename TBuffer>
        __global__ void ExpandExactGpuSources(
                TAppInst app_inst,
                const index_t *sources,
                uint32_t source_count,
                TPMAGraph graph,
                TValue *values,
                TBuffer *buffers,
                TValue *parents,
                groute::dev::Queue<index_t> changed_vertices,
                unsigned long long *processed_sources,
                unsigned long long *processed_edges) {
            const uint32_t source_index = blockIdx.x;
            if (source_index >= source_count) return;
            const index_t source = sources[source_index];
            __shared__ TBuffer source_value;
            __shared__ bool expand;
            if (threadIdx.x == 0) {
                const auto combined = app_inst.CombineValueBuffer(
                    source, &values[source], &buffers[source]);
                source_value = combined.first;
                expand = combined.second;
                if (expand) {
                    atomicAdd(processed_sources, 1ULL);
                    atomicAdd(processed_edges,
                        static_cast<unsigned long long>(graph.degree(source)));
                }
            }
            __syncthreads();
            if (!expand) return;

            const uint64_t edge_begin = graph.begin_edge(source);
            const uint32_t degree = graph.degree(source);
            for (uint32_t offset = threadIdx.x; offset < degree;
                 offset += blockDim.x) {
                const index_t destination = graph.edge_dest(edge_begin + offset);
                if (destination == static_cast<index_t>(-1)) continue;
                const index_t weight = app_inst.TraversalEdgeWeight(source, destination);
                if (app_inst.AccumulateBuffer(source, destination, weight,
                        &parents[destination], &buffers[destination], source_value)) {
                    changed_vertices.append(destination);
                }
            }
        }

        template<typename TAppInst, typename TPMAGraph, typename TValue,
                 typename TBuffer, bool Ordered = false>
        __global__ void RunExactGpuClosure(
                TAppInst app_inst,
                groute::dev::Queue<index_t> input,
                groute::dev::Queue<index_t> output,
                TPMAGraph graph,
                TValue *values,
                TBuffer *buffers,
                TValue *parents,
                uint32_t epoch,
                uint32_t *node_state_epoch,
                unsigned long long *metrics) {
            cooperative_groups::grid_group grid =
                cooperative_groups::this_grid();
            __shared__ TBuffer source_value;
            __shared__ bool expand;
            while (true) {
                grid.sync();
                const uint32_t source_count = input.count();
                if (source_count == 0) break;
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    output.reset();
                    if (Ordered) metrics[4] = ~0ULL;
                    atomicAdd(metrics + 0,
                        static_cast<unsigned long long>(source_count));
                    atomicAdd(metrics + 3, 1ULL);
                }
                grid.sync();
                if (Ordered && metrics[3] > 0xffffULL) {
                    // Fail before ticket reuse could silently suppress work.
                    asm("trap;");
                }
                // Select a distance window from pending buffers, without committing
                
                // contract, not graph identity. Reduction and carry are service costs.
                if (Ordered) {
                    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                         i < source_count; i += gridDim.x * blockDim.x) {
                        const index_t source = input.read(i);
                        atomicMin(metrics + 4,
                            static_cast<unsigned long long>(buffers[source]));
                    }
                    grid.sync();
                }
                for (uint32_t source_index = blockIdx.x;
                     source_index < source_count;
                     source_index += gridDim.x) {
                    const index_t source = input.read(source_index);
                    if (threadIdx.x == 0) {
                        const bool deferred = Ordered &&
                            static_cast<unsigned long long>(buffers[source]) > metrics[4] + 127ULL;
                        expand = false;
                        if (deferred) {
                            const uint32_t ticket = (epoch << 16) |
                                (static_cast<uint32_t>(metrics[3]) & 0xffffU);
                            
                            // a concurrent improvement remains pending next wave.
                            if (atomicExch(node_state_epoch + source, ticket) != ticket)
                                output.append(source);
                        } else {
                            const auto combined = app_inst.CombineValueBuffer(
                                source, &values[source], &buffers[source]);
                            source_value = combined.first;
                            expand = combined.second;
                        }
                        if (expand) {
                            atomicAdd(metrics + 1, 1ULL);
                            atomicAdd(metrics + 2,
                                static_cast<unsigned long long>(
                                    graph.degree(source)));
                        }
                    }
                    __syncthreads();
                    if (expand) {
                        const uint64_t edge_begin = graph.begin_edge(source);
                        const uint32_t degree = graph.degree(source);
                        for (uint32_t offset = threadIdx.x; offset < degree;
                             offset += blockDim.x) {
                            const index_t destination =
                                graph.edge_dest(edge_begin + offset);
                            if (destination == static_cast<index_t>(-1)) continue;
                            const index_t weight =
                                app_inst.TraversalEdgeWeight(source, destination);
                            if (app_inst.AccumulateBuffer(source, destination,
                                    weight, &parents[destination],
                                    &buffers[destination], source_value)) {
                                const uint32_t wave_ticket =
                                    (epoch << 16) |
                                    (static_cast<uint32_t>(metrics[3]) & 0xffffU);
                                if (atomicExch(node_state_epoch + destination,
                                        wave_ticket) != wave_ticket) {
                                    output.append(destination);
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                grid.sync();
                if (blockIdx.x == 0 && threadIdx.x == 0) input.reset();
                grid.sync();
                const auto previous = input;
                input = output;
                output = previous;
            }
            grid.sync();
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                input.reset();
                output.reset();
            }
            grid.sync();
        }

      template<typename TValue, typename TBuffer, typename TWeight, template<typename, typename, typename, typename ...> class TAppImpl, typename... UnusedData>
      class     Engine {
        private:
        typedef TAppImpl<TValue, TBuffer, TWeight, UnusedData...> AppImplDeviceObject;
        typedef graphs::GraphDatum<TValue, TBuffer, TWeight> GraphDatum;
        typedef Loader<index_t, index_t, index_t> Loader;
        
        typedef WEdgePair<uint32_t,uint32_t,uint32_t>  WEdge;
        
        cudaDeviceProp m_dev_props;

            // Graph data
            std::unique_ptr<utils::traversal::Context<Algo>> m_groute_context;
            std::unique_ptr<groute::Stream> m_stream;
            std::unique_ptr<groute::graphs::single::CSRGraphAllocator> m_csr_dev_graph_allocator;
            std::unique_ptr<groute::graphs::single::PMAGraphAllocator> m_vcsr_dev_graph_allocator;
            std::unique_ptr<topology::SourceLocalChunkStore> m_chunk_store;
            std::vector<index_t> m_topology_patch_sources;
            std::vector<index_t> m_topology_patch_scratch;
            size_t m_topology_delete_source_count = 0;
            topology::GroupedUpdateBatch m_grouped_updates;
            
            std::unique_ptr<groute::graphs::single::CSCGraphAllocator> m_csc_dev_graph_allocator;
            runtime::DynamicReverseIndex m_reverse_index;
            std::vector<uint64_t> m_gpu_repair_incoming_offsets;
            std::vector<index_t> m_gpu_repair_incoming_sources;
            runtime::RepairTopologyStorage m_gpu_repair_topology;
            groute::Queue<index_t> m_device_affected_vertices;
            std::vector<index_t> m_affected_vertices;
            unsigned int *m_device_gpu_repair_changed = nullptr;

            index_t *m_device_partition_end_nodes = nullptr;

            uint32_t *m_device_node_state_epoch = nullptr;
            unsigned long long *m_device_seed_reject_count = nullptr;
            uint32_t m_insertion_epoch = 0;
            uint64_t m_insertion_convergence_round = 0;
            bool m_insertion_epoch_active = false;
            index_t m_insertion_seed_begin = 0;
            index_t m_insertion_seed_count = 0;

            uint32_t m_last_insertion_dirty_partitions = 0;

            std::unique_ptr<AppImplDeviceObject> m_app_inst;
            groute::Stream stream[64];
            // App instance
            std::unique_ptr<GraphDatum> m_graph_datum;
            std::unique_ptr<Loader> m_load_update;
            policy::TRunningInfo m_running_info;
            TBuffer current_priority;  // put it into running info
            PolicyDecisionMaker m_policy_decision_maker;
            EngineOptions m_engine_options;
            WEdge* added_edges_h=nullptr;
            WEdge* added_edges_d=nullptr;
            WEdge* del_edges_h=nullptr;
            WEdge* del_edges_d=nullptr;
            uint32_t* work_size_d = nullptr;
            int type[2] = {0, 0};
            int *type_device=nullptr;
            bool type_host_registered = false;
            //partition_information
            unsigned int partitions_csc;
            unsigned int* partition_offset_csc;
            unsigned int max_partition_size_csc;

            unsigned int partitions_csr;
            unsigned int* partition_offset_csr;
            unsigned int max_partition_size_csr;
            std::vector<index_t> m_insertion_changed_vertices;
            cache_patch::RefreshGate m_cache_refresh_gate;

            groute::Queue<index_t> m_device_insertion_changed_vertices;
            bool m_exact_source_frontier_ready = false;
            uint32_t m_exact_source_frontier_count = 0;

            unsigned long long *m_device_exact_source_counts = nullptr;

            void RefreshSegmentActiveCountsFromQueues() {
                GraphDatum &graph_datum = *m_graph_datum;
                for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg) {
                    const index_t stream_id = seg % FLAGS_n_stream;
                    const index_t active =
                        graph_datum.m_wl_array_in_seg[seg].GetCount(stream[stream_id]);
                    graph_datum.seg_active_num[seg] = active;
                    m_running_info.input_active_count_seg[seg] = active;
                }
            }

            public:
            Engine(AlgoType algo_type) :
            m_running_info(algo_type),
            m_policy_decision_maker(m_running_info) {
                int dev_id = 0;

                GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&m_dev_props, dev_id));
                m_groute_context = std::unique_ptr<utils::traversal::Context<Algo>>
                (new utils::traversal::Context<Algo>(1, AppImplDeviceObject::kComponentLabels ||
                    AppImplDeviceObject::kVertexSeeds || AppImplDeviceObject::kSignedResidual,
                    AppImplDeviceObject::NormalizeUndirectedInput()));

                //create stream /*CODE by ax range 118 to 121*/
                for(int i = 0; i < FLAGS_n_stream; i++)
                {
                    stream[i] = m_groute_context->CreateStream(dev_id);
                }

        m_stream = std::unique_ptr<groute::Stream>(new groute::Stream(dev_id));
    }

    ~Engine() {
        if (m_device_gpu_repair_changed != nullptr) {
            cudaFree(m_device_gpu_repair_changed);
            m_device_gpu_repair_changed = nullptr;
        }

        if (m_device_partition_end_nodes != nullptr) {
            cudaFree(m_device_partition_end_nodes);
            m_device_partition_end_nodes = nullptr;
        }

        if (m_device_node_state_epoch != nullptr) {
            cudaFree(m_device_node_state_epoch);
            m_device_node_state_epoch = nullptr;
        }
        if (m_device_seed_reject_count != nullptr) {
            cudaFree(m_device_seed_reject_count);
            m_device_seed_reject_count = nullptr;
        }

        if (m_device_exact_source_counts != nullptr) {
            cudaFree(m_device_exact_source_counts);
            m_device_exact_source_counts = nullptr;
        }

    }

    void SetOptions(EngineOptions &engine_options) {
        m_engine_options = engine_options;
    }

    index_t GetNodeNum(){
      return m_groute_context->host_graph.nnodes;
    }
            void compute_hot_vertices_pr() {
                // Use the production activity window, ID/score pairing and
                
                compute_hot_vertices_sssp();
            }

            void compute_hot_vertices_sssp(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                Stopwatch sw_ch(true);
                kernel::comp_hotness_sssp<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.m_node_buffer_datum,
                    graph_datum.d_hotness,
                    graph_datum.d_id.Current());
                stream_s.Sync();
                sw_ch.stop();
                LOG("comp_hotness time: %f ms (excluded)\n", sw_ch.ms());

                graph_datum.sort_vtx_by_hotness();
                LOG("sort vtx ms (excluded)\n");
                cudaDeviceSynchronize();
                
                Stopwatch extrac(true);
                kernel::extract_vtx_degree<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_id,
                    graph_datum.d_v);
                stream_s.Sync();
                extrac.stop();
                kernel::reset_hotness<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                    stream_s.Sync();
                LOG("extract degree time: %f ms (excluded)\n", extrac.ms());
                cudaDeviceSynchronize();
            }

            index_t CacheCandidateCount() {
                GraphDatum &graph_datum = *m_graph_datum;
                const index_t nnodes = graph_datum.nnodes;
                if (nnodes == 0 || graph_datum.num_of_cache == 0) return 0;
                index_t final_prefix = 0;
                index_t final_degree = 0;
                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &final_prefix, graph_datum.d_sum + nnodes - 1,
                    sizeof(final_prefix), cudaMemcpyDeviceToHost));
                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &final_degree, graph_datum.d_v + nnodes - 1,
                    sizeof(final_degree), cudaMemcpyDeviceToHost));
                if (static_cast<uint64_t>(final_prefix) + final_degree <
                    graph_datum.num_of_cache) return nnodes;
                index_t low = 0;
                index_t high = nnodes - 1;
                while (low < high) {
                    const index_t middle = low + (high - low) / 2;
                    index_t prefix = 0;
                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        &prefix, graph_datum.d_sum + middle + 1,
                        sizeof(prefix), cudaMemcpyDeviceToHost));
                    if (static_cast<uint64_t>(prefix) < graph_datum.num_of_cache) {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                return low;
            }

            void EnsureTypeDevicePointer() {
                if (!type_host_registered) {
                    GROUTE_CUDA_CHECK(cudaHostRegister(
                        type, sizeof(type), cudaHostRegisterMapped));
                    type_host_registered = true;
                }
                if (type_device == nullptr) {
                    GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(
                        reinterpret_cast<void **>(&type_device), type, 0));
                }
            }

            void confirm_candidate_batch(){
                cgcomm::Scope comm_scope(cgcomm::Category::Cache);
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                Stopwatch extrac(true);
                graph_datum.ensure_candidate_vertex();
                extrac.stop();
                LOG("candidate v time: %f ms (excluded)\n", extrac.ms());
                const index_t admitted = CacheCandidateCount();
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, admitted);
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                
                dim3 grid_dims, block_dims;
                Stopwatch search_rebuild(true);
                type[0] = 0;
                EnsureTypeDevicePointer();
                if (admitted != 0 && m_cache_refresh_gate.published()) {
                    KernelSizing(grid_dims, block_dims, work_source.get_size());
                    kernel::search_batch<<<grid_dims, block_dims, 0, stream_s.cuda_stream>>>(
                        app_inst, vcsr_graph, work_source,
                        graph_datum.d_id.Current(), type_device);
                    stream_s.Sync();
                }
                search_rebuild.stop();
                cudaDeviceSynchronize();
                const bool missing_desired = !m_cache_refresh_gate.published() || type[0] != 0;
                const bool refresh = m_cache_refresh_gate.Observe(admitted, missing_desired);
                if (refresh && admitted != 0) {
                    KernelSizing(grid_dims, block_dims, work_source.get_size());
                    kernel::mark_cache_candidates<<<grid_dims, block_dims, 0,
                        stream_s.cuda_stream>>>(vcsr_graph, work_source,
                                                graph_datum.d_id.Current());
                    stream_s.Sync();
                }
                LOG("[CACHE-GATE] desired_vertices=%u published_vertices=%llu missing_desired=%u refresh_required=%u candidate_ms=%.3f\n",
                    admitted,
                    static_cast<unsigned long long>(m_cache_refresh_gate.published_vertices()),
                    missing_desired ? 1U : 0U, refresh ? 1U : 0U,
                    search_rebuild.ms());
            }

            bool CacheRefreshRequired() const {
                return m_cache_refresh_gate.refresh_required();
            }

            void MarkCachePublished() {
                m_cache_refresh_gate.Publish();
            }


            void LoadCache(){
                cgcomm::Scope comm_scope(cgcomm::Category::Cache);
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                Stopwatch sw_load(true);
                index_t seg_snode,seg_enode;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildWorklist_delta(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                ExecutePolicy_MF_topo(next_policy);
                cudaDeviceSynchronize();
                sw_load.stop();
                LOG("加载缓存数据时间  : %f ms \n",sw_load.ms());
            }

            void ExecutePolicy_MF_topo(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                index_t seg_snode,seg_enode;
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBAmend_cache(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                        //    type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }
//Loadgraph from the data structure to the segment_structure
        void LoadGraph() {
      Stopwatch sw_load(true);
      groute::graphs::host::PMAGraph &vcsr_graph = m_groute_context->host_pma_small;
      if (FLAGS_SEGMENT <= 0 ||
          FLAGS_SEGMENT > static_cast<int32_t>(GraphDatum::kMaxSegments)) {
          LOG("[SEGMENT-ERROR] configured=%d valid_range=[1,%u]\n",
              FLAGS_SEGMENT,
              GraphDatum::kMaxSegments);
          std::exit(EXIT_FAILURE);
      }
      index_t m_nsegs = FLAGS_SEGMENT;
      uint64_t seg_sedge_csr, seg_eedge_csr;
      index_t seg_snode,seg_enode, seg_nnodes;
      uint64_t seg_nedges_csr;
      uint64_t seg_nedges = round_up(vcsr_graph.elem_capacity, m_nsegs);
      uint64_t seg_nedges_csr_max = 0;  //dev memory
      uint64_t edge_num = 0;
      index_t node_id = 0;
      uint64_t out_degree;
      std::vector<index_t> nnodes_num;
      seg_snode = node_id;
      m_groute_context->seg_snode[0] = seg_snode;
      vcsr_graph.sync_vertices_[vcsr_graph.nnodes].index =  vcsr_graph.elem_capacity;
      for(index_t seg_idx = 0; node_id < vcsr_graph.nnodes ; seg_idx++){
          if (seg_idx >= static_cast<index_t>(FLAGS_SEGMENT)) {
              LOG("[SEGMENT-ERROR] generated_more_than_configured configured=%d node=%u\n",
                  FLAGS_SEGMENT,
                  node_id);
              std::abort();
          }
          m_groute_context->seg_snode[seg_idx] = node_id;
          while(edge_num < seg_nedges && node_id < vcsr_graph.nnodes){
            out_degree = vcsr_graph.end_edge(node_id) - vcsr_graph.begin_edge(node_id);
            edge_num = edge_num + out_degree;
            node_id++;
           }
           if(node_id == vcsr_graph.nnodes){
                seg_enode = node_id ;
            }
           else{
                seg_enode = node_id;
           }
            seg_nnodes = seg_enode - seg_snode;

            m_running_info.nnodes_seg[seg_idx] = seg_nnodes;
            nnodes_num.push_back(seg_nnodes);

            m_groute_context->seg_enode[seg_idx] = seg_enode;
            seg_sedge_csr = vcsr_graph.sync_vertices_[seg_snode].index;
            seg_eedge_csr = vcsr_graph.sync_vertices_[seg_enode].index;
            seg_nedges_csr = seg_eedge_csr - seg_sedge_csr;

            m_running_info.total_workload_seg[seg_idx] = seg_nedges_csr;
            seg_nedges_csr_max = max(seg_nedges_csr_max,seg_nedges_csr);

            m_groute_context->seg_sedge_csr[seg_idx] = seg_sedge_csr;
            m_groute_context->seg_nedge_csr[seg_idx] = seg_nedges_csr;

            
            edge_num = 0;
              seg_snode = node_id;
            }

        if (nnodes_num.size() > static_cast<size_t>(FLAGS_SEGMENT)) {
            LOG("[SEGMENT-ERROR] generated=%llu configured=%d\n",
                static_cast<unsigned long long>(nnodes_num.size()),
                FLAGS_SEGMENT);
            std::abort();
        }
        while (nnodes_num.size() < static_cast<size_t>(FLAGS_SEGMENT)) {
            const index_t seg_idx = static_cast<index_t>(nnodes_num.size());
            const index_t empty_node = vcsr_graph.nnodes;
            const uint64_t empty_edge = vcsr_graph.sync_vertices_[empty_node].index;
            m_groute_context->seg_snode[seg_idx] = empty_node;
            m_groute_context->seg_enode[seg_idx] = empty_node;
            m_groute_context->seg_sedge_csr[seg_idx] = empty_edge;
            m_groute_context->seg_nedge_csr[seg_idx] = 0;
            m_running_info.nnodes_seg[seg_idx] = 0;
            m_running_info.total_workload_seg[seg_idx] = 0;
            nnodes_num.push_back(0);
        }

        m_groute_context->segment_ct = FLAGS_SEGMENT;
        m_groute_context->SetDevice(0);

        LOG("seg_nedges_csr_max = %d\n",seg_nedges_csr_max);
        uint64_t chunk_required_edges = 0;
        for (index_t source = 0; source < vcsr_graph.nnodes; ++source) {
            chunk_required_edges += topology::SourceLocalChunkStore::RequiredChunkCapacity(
                vcsr_graph.sync_vertices_[source].degree);
        }
        const uint64_t chunk_headroom = std::max<uint64_t>(
            chunk_required_edges / 8, 1ULL << 20);
        topology::ChunkArenaOptions chunk_options;
        chunk_options.capacity_edges = chunk_required_edges + chunk_headroom;
        chunk_options.slab_capacity_edges = 1ULL << 24;
        chunk_options.minimum_chunk_edges = 4;
        chunk_options.pinned = true;
        const char *mutation_workers_env = std::getenv("CG_MUTATION_WORKERS");
        if (mutation_workers_env != nullptr && mutation_workers_env[0] != '\0') {
            const unsigned long parsed = std::strtoul(mutation_workers_env, nullptr, 10);
            if (parsed > 0 && parsed <= 256) {
                chunk_options.mutation_workers = static_cast<uint32_t>(parsed);
            }
        }
        m_chunk_store.reset(new topology::SourceLocalChunkStore(
            vcsr_graph.nnodes, chunk_options));
        for (index_t source = 0; source < vcsr_graph.nnodes; ++source) {
            const uint64_t begin = vcsr_graph.sync_vertices_[source].index;
            const index_t degree = vcsr_graph.sync_vertices_[source].degree;
            m_chunk_store->LoadSource(source, std::vector<index_t>(
                vcsr_graph.edges_ + begin, vcsr_graph.edges_ + begin + degree));
        }
        m_chunk_store->FinalizeLoad(vcsr_graph.nedges);
        m_vcsr_dev_graph_allocator = std::unique_ptr<groute::graphs::single::PMAGraphAllocator>(new groute::graphs::single::PMAGraphAllocator(vcsr_graph,seg_nedges_csr_max));
        m_vcsr_dev_graph_allocator->BindChunkStore(*m_chunk_store);
        LOG("[CHUNK-LOAD] edges=%llu capacity_edges=%llu slabs=%zu metadata_bytes=%llu pinned_edge_bytes=%llu pinned_total_bytes=%llu\n",
            static_cast<unsigned long long>(m_chunk_store->EdgeCount()),
            static_cast<unsigned long long>(chunk_options.capacity_edges),
            m_chunk_store->SlabCount(),
            static_cast<unsigned long long>(m_chunk_store->MetadataBytes()),
            static_cast<unsigned long long>(
                chunk_options.capacity_edges * sizeof(index_t)),
            static_cast<unsigned long long>(
                chunk_options.capacity_edges * sizeof(index_t) +
                m_chunk_store->MetadataBytes()));

        if constexpr (AppImplDeviceObject::kComponentLabels) {
            // CC's normalized adjacency is symmetric: it IS the reverse view.
            // Do not allocate a second CSC base or maintain reverse deltas.
            LOG("[CC-REVERSE-VIEW] mode=alias_forward adjacency_records=%llu separate_reverse_records=0\n",
                (unsigned long long)m_chunk_store->EdgeCount());
            if (AppImplDeviceObject::NormalizeUndirectedInput()) {
                std::vector<uint64_t>().swap(vcsr_graph.row_start_vec);
                std::vector<index_t>().swap(vcsr_graph.edge_dst_vec);
                vcsr_graph.row_start = nullptr;
                vcsr_graph.edge_dst = nullptr;
            }
        } else if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
            const uint32_t hardware_workers = std::thread::hardware_concurrency();
            const uint32_t reverse_workers = std::max<uint32_t>(
                1, std::min<uint32_t>(20, hardware_workers == 0 ? 1 : hardware_workers));
            Stopwatch sw_reverse_index(true);
            uint32_t reverse_shards = 1;
            if (const char *value = std::getenv("CG_REVERSE_SHARDS")) {
                const std::string setting(value);
                if (setting != "1" && setting != "64")
                    throw std::invalid_argument("CG_REVERSE_SHARDS must be 1 or 64");
                reverse_shards = setting == "64" ? 64 : 1;
            }
            m_reverse_index.Build(vcsr_graph, reverse_workers, reverse_shards);
            LOG("[REVERSE-CONFIG] shards=%u\n", reverse_shards);
            sw_reverse_index.stop();
            LOG("[CPU-REVERSE-INDEX] base_edges=%llu workers=%u build_ms=%.3f overlay_edges=0\n",
                static_cast<unsigned long long>(m_reverse_index.BaseEdgeCount()),
                reverse_workers,
                sw_reverse_index.ms());
        }

        m_graph_datum = std::unique_ptr<GraphDatum>(new GraphDatum(vcsr_graph,seg_nedges_csr_max,nnodes_num));
        m_device_affected_vertices = std::move(groute::Queue<index_t>(
            std::max<uint32_t>(vcsr_graph.nnodes, 1)));
        m_device_affected_vertices.ResetAsync(m_stream->cuda_stream);
        if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
            GROUTE_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_device_gpu_repair_changed),
                sizeof(unsigned int)));
        }
        const uint64_t changed_queue_capacity_u64 =
            std::max<uint64_t>(std::max<uint64_t>(vcsr_graph.nnodes, 1),
                               std::min<uint64_t>(vcsr_graph.elem_capacity,
                                                  seg_nedges_csr_max * std::max<uint32_t>(FLAGS_n_stream, 1)));
        m_device_insertion_changed_vertices =
            std::move(groute::Queue<index_t>(static_cast<uint32_t>(
                std::min<uint64_t>(changed_queue_capacity_u64,
                                   std::numeric_limits<uint32_t>::max()))));
        sw_load.stop();

        m_running_info.time_load_graph = sw_load.ms();

        LOG("Load graph time: %f ms (excluded)\n", sw_load.ms());

        m_running_info.nnodes = m_groute_context->nvtxs;
        m_running_info.nedges = m_groute_context->nedges;
        m_running_info.total_workload = m_groute_context->nedges * FLAGS_edge_factor;
        current_priority = m_engine_options.GetPriorityThreshold();
    }
    /*
        * Init Graph Value and buffer fields
        */

        void InitGraph(UnusedData &...data) {
            Stopwatch sw_init(true);

            m_app_inst = std::unique_ptr<AppImplDeviceObject>(new AppImplDeviceObject(data...));
            groute::Stream &stream_s = *m_stream;
            GraphDatum &graph_datum = *m_graph_datum;
            const auto &dev_vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
            const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
            dim3 grid_dims, block_dims;

            m_app_inst->m_vcsr_graph = dev_vcsr_graph;
            m_app_inst->m_nnodes = graph_datum.nnodes;
            m_app_inst->m_nedges = graph_datum.nedges;
            m_app_inst->m_p_current_round = graph_datum.m_current_round.dev_ptr;
            // Launch kernel to init value/buffer fields
            KernelSizing(grid_dims, block_dims, work_source.get_size());
            auto &app_inst = *m_app_inst;
            
            
            kernel::InitGraph
            << < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                work_source,
                graph_datum.d_id,
                graph_datum.d_hotness,
                graph_datum.GetParentDeviceObject(),
                graph_datum.GetValueDeviceObject(),
                graph_datum.GetBufferDeviceObject());
            stream_s.Sync();

            index_t seg_snode,seg_enode;
            index_t stream_id;

            for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                stream_id = seg_idx % FLAGS_n_stream;
                seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                seg_enode = m_groute_context->seg_enode[seg_idx];
                    RebuildArrayWorklist(app_inst,
                    graph_datum,
                    stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
            }

            for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                stream[stream_idx].Sync();
            }
            
            m_running_info.time_init_graph = sw_init.ms();

            sw_init.stop();

            LOG("InitGraph: %f ms (excluded)\n", sw_init.ms());
        }

        void SaveToJson() {
            JsonWriter &writer = JsonWriter::getInst();

            writer.write("time_input_active_node", m_running_info.time_overhead_input_active_node);
            writer.write("time_output_active_node", m_running_info.time_overhead_output_active_node);
            writer.write("time_input_workload", m_running_info.time_overhead_input_workload);
            writer.write("time_output_workload", m_running_info.time_overhead_output_workload);
            writer.write("time_queue2bitmap", m_running_info.time_overhead_queue2bitmap);
            writer.write("time_bitmap2queue", m_running_info.time_overhead_bitmap2queue);
            writer.write("time_rebuild_worklist", m_running_info.time_overhead_rebuild_worklist);
            writer.write("time_priority_sample", m_running_info.time_overhead_sample);
            writer.write("time_sort_worklist", m_running_info.time_overhead_wl_sort);
            writer.write("time_unique_worklist", m_running_info.time_overhead_wl_unique);
            writer.write("time_kernel", m_running_info.time_kernel);
            writer.write("time_total", m_running_info.time_total);
            writer.write("time_per_round", m_running_info.time_total / m_running_info.current_round);
            writer.write("num_iteration", (int) m_running_info.current_round);

            if (m_engine_options.IsForceVariant()) {
                writer.write("force_variant", m_engine_options.GetAlgoVariant().ToString());
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                writer.write("force_push_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)));
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                writer.write("force_pull_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)));
            }

            if (m_engine_options.GetPriorityType() == Priority::NONE) {
                writer.write("priority_type", "none");
                } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                    writer.write("priority_type", "low_high");
                    writer.write("priority_delta", m_engine_options.GetPriorityThreshold());
                    } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                        writer.write("priority_type", "sampling");
                        writer.write("cut_threshold", m_engine_options.GetCutThreshold());
                    }

                    writer.write("fused_kernel", m_engine_options.IsFused() ? "YES" : "NO");
                    writer.write("max_iteration_reached",
                     m_running_info.current_round == 1000 ? "YES" : "NO");
                
                writer.write("device", m_dev_props.name);
                writer.write("dataset", FLAGS_graphfile);
                writer.write("nnodes", (int) m_graph_datum->nnodes);
                writer.write("nedges", (int) m_graph_datum->nedges);
                writer.write("algo_type", m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME"
                   : "ITERATIVE_SCHEME");
        }

            void PrintInfo() {
                LOG("--------------Overhead--------------\n");
                LOG("Rebuild worklist: %f\n", m_running_info.time_overhead_rebuild_worklist);
                LOG("Priority sample: %f\n", m_running_info.time_overhead_sample);
                LOG("hybrid Worlist: %f\n", m_running_info.time_overhead_hybrid);
                LOG("Unique Worklist: %f\n", m_running_info.time_overhead_wl_unique);
                LOG("--------------Time statistics---------\n");
                LOG("Kernel time: %f\n", m_running_info.time_kernel);
                LOG("Total time: %f\n", m_running_info.time_total);
                LOG("Total rounds: %d\n", m_running_info.current_round);
                LOG("Time/round: %f\n", m_running_info.time_total / m_running_info.current_round);
                LOG("filter_num: %d\n", m_running_info.explicit_num);
                LOG("zerocopy_num: %d\n", m_running_info.zerocopy_num);
                LOG("compaction_num: %d\n", m_running_info.compaction_num);


                LOG("--------------Engine info-------------\n");
                if (m_engine_options.IsForceVariant()) {
                    LOG("Force variant: %s\n", m_engine_options.GetAlgoVariant().ToString().data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    LOG("Force Push Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)).data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    LOG("Force Pull Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)).data());
                }

                if (m_engine_options.GetPriorityType() == Priority::NONE) {
                    LOG("Priority type: NONE\n");
                    } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                        LOG("Priority type: LOW_HIGH\n");
                        LOG("Priority delta: %f\n", m_engine_options.GetPriorityThreshold());
                        } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                            LOG("Priority type: Sampling\n");
                            LOG("Cut threshold: %f\n", m_engine_options.GetCutThreshold());
                        }

                        LOG("Fused kernel: %s\n", m_engine_options.IsFused() ? "YES" : "NO");
                        LOG("Max iteration reached: %s\n", m_running_info.current_round == 1000 ? "YES" : "NO");


                        LOG("-------------Misc-------------------\n");
                
                LOG("Device: %s\n", m_dev_props.name);
                LOG("Dataset: %s\n", FLAGS_graphfile.data());
                LOG("Algo type: %s\n",
                    m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME" : "ITERATIVE_SCHEME");
            }

            //traditional graph processing
            void Start(index_t priority_detal = 0) {

                GraphDatum &graph_datum = *m_graph_datum;
                graph_datum.priority_detal = priority_detal;
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                bool convergence = false;
                Stopwatch sw_total(true);
                LoadOptions();
                int round = 0;
                while (!convergence) {
                  PreComputationBW();
                  ExecutePolicy_Converge(next_policy);
                
                  round++;
                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                }
               sw_total.stop();
               m_running_info.time_total = sw_total.ms();
               LOG("Iterate all time: %f ms (excluded)\n", sw_total.ms());

            }

            void Compare_to_cache(){
                GraphDatum &graph_datum = *m_graph_datum;
                graph_datum.GatherCacheL3();
                graph_datum.GatherCacheL1();
                for(uint64_t j =0; j < graph_datum.num_of_cache;j++){
                    if(graph_datum.host_cache[j]!=graph_datum.host_cache_l3[j]){
                        printf("err %d -> %d\n",graph_datum.host_cache[j],graph_datum.host_cache_l3[j]);
                    }
                }
            }

            const groute::graphs::host::CSRGraph &CSRGraph() const {
                return m_groute_context->host_graph;
            }

            // const groute::graphs::host::CSRGraph &ValidateGraph() const {

            const groute::graphs::host::PMAGraph &PMAGraph() const {
                return m_groute_context->host_pma_small;
            }

            const topology::SourceLocalChunkStore &ChunkStore() const {
                return *m_chunk_store;
            }

            const GraphDatum &GetGraphDatum() const {
                return *m_graph_datum;
            }

            void ExecutePolicy_All(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];

                    stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_ALL(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                            m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               sw_execution.stop();
               LOG("part add edge time: %f ms (excluded)\n", sw_execution.ms());
            

            }

            //memory free policy for gpu incremental graph processing
            void ExecutePolicy_MF(AlgoVariant *algo_variant,int *type_device) {
                LOG("------MF PUSH------\n");
                auto &app_inst = *m_app_inst;
                
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){

                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];

                    stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBAmend(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }

            //Cache filling
            void ExecutePolicy_MF_Cache(AlgoVariant *algo_variant,int *type_device) {
                auto &app_inst = *m_app_inst;
                
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){

                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];

                    stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBCache(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }


            void ExecutePolicy_Com(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];
                    stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncCom(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                               vcsr_graph,
                               graph_datum,
                               m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
                sw_execution.stop();
               LOG("flush cache(compact time): %f ms (excluded)\n", sw_execution.ms());
            }


            void get_update_file(){
                m_load_update = std::unique_ptr<Loader>(new Loader(FLAGS_update_size,FLAGS_weight));
                Loader &load_update = *m_load_update;
                load_update.ReadWeightList(FLAGS_updatefile);
                if constexpr (AppImplDeviceObject::kComponentLabels)
                    AppImplDeviceObject::PrepareUpdates(load_update, m_graph_datum->nnodes);
                // Empty insertion/deletion streams still need a valid mapped
                // allocation; cudaHostRegister rejects a zero-byte range.
                const size_t add_capacity = std::max<size_t>(load_update.m_add_size, 1);
                const size_t del_capacity = std::max<size_t>(load_update.m_del_size, 1);
                this->added_edges_h = (WEdge*)malloc(sizeof(WEdge) * add_capacity);
                this->del_edges_h = (WEdge*)malloc(sizeof(WEdge) * del_capacity);
                GROUTE_CUDA_CHECK(cudaHostRegister((void *)(this->added_edges_h), sizeof(WEdge) * add_capacity, cudaHostRegisterMapped));

                GROUTE_CUDA_CHECK(cudaHostRegister((void *)(this->del_edges_h), sizeof(WEdge) * del_capacity, cudaHostRegisterMapped));

                GROUTE_CUDA_CHECK(cudaMalloc(&(this->work_size_d), 2 * sizeof(uint32_t)));
                EnsureTypeDevicePointer();
                size_t max_patch_records = 0;
                for (const auto &batch_size : load_update.m_batch_size) {
                    max_patch_records = std::max<size_t>(
                        max_patch_records,
                        static_cast<size_t>(batch_size.first) + batch_size.second);
                }
                m_vcsr_dev_graph_allocator->ReserveSparsePublicationCapacity(
                    max_patch_records, FLAGS_check);
            }

            size_t GetUpdateBatchCount() const {
                return m_load_update ? m_load_update->m_batch_size.size() : 0;
            }

            void add_edge(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                Loader &load_update = *m_load_update;

                update_tree_add(local_begin,NumOfSnapShots);
            }

            void compute_hot_vertices(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::comp_hotness<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                graph_datum.sort_vtx_by_hotness();

                cudaDeviceSynchronize();
                graph_datum.CompareDeviceResult();
                cudaDeviceSynchronize();
            }

            void FinalizePublicationSources(index_t batch) {
                Stopwatch timer(true);
                const char* setting = std::getenv("CG_MERGE_PUBLICATION_SOURCES");
                const bool merge = setting && std::string(setting) == "1";
                const size_t input_count = m_topology_patch_sources.size();
                if (merge) {
                    topology::MergePublicationSources(m_topology_patch_sources,
                        m_topology_delete_source_count, m_topology_patch_scratch);
                } else {
                    std::sort(m_topology_patch_sources.begin(), m_topology_patch_sources.end());
                    m_topology_patch_sources.erase(std::unique(m_topology_patch_sources.begin(),
                        m_topology_patch_sources.end()), m_topology_patch_sources.end());
                }
                timer.stop();
                LOG("[PUBLICATION][batch %u] merge=%u input_sources=%llu output_sources=%llu source_order_ms=%.3f\n",
                    batch, unsigned(merge), static_cast<unsigned long long>(input_count),
                    static_cast<unsigned long long>(m_topology_patch_sources.size()), timer.ms());
            }

            void PrepareGroupedUpdates(const std::pair<index_t,index_t>& local_begin,
                                       index_t NumOfSnapShots) {
                Stopwatch sw_group(true);
                Loader &load_update = *m_load_update;
                topology::TopologyMutationBatch batch;
                const auto sizes = load_update.m_batch_size[NumOfSnapShots];
                batch.deletions.reserve(sizes.second);
                batch.additions.reserve(sizes.first);
                for (index_t i = local_begin.second; i < local_begin.second + sizes.second; ++i) {
                    const auto &edge = load_update.deleted_edges_w[i];
                    batch.deletions.push_back({edge.u, edge.v});
                }
                for (index_t i = local_begin.first; i < local_begin.first + sizes.first; ++i) {
                    const auto &edge = load_update.added_edges_w[i];
                    batch.additions.push_back({edge.u, edge.v});
                }
                m_grouped_updates = topology::GroupedUpdateBatch(batch);
                LOG("[MODE][batch %u] maintenance=%s requests=%llu\n",
                    NumOfSnapShots, m_grouped_updates.LargeMaintenance() ? "large" : "regular",
                    static_cast<unsigned long long>(batch.deletions.size() + batch.additions.size()));
                m_topology_patch_sources.clear();
                m_topology_delete_source_count = 0;
                m_topology_patch_sources.reserve(m_grouped_updates.sources.size() * 2);
                sw_group.stop();
                LOG("[BATCH][batch %u] sources=%llu updates=%llu group_ms=%.3f\n",
                    NumOfSnapShots, static_cast<unsigned long long>(m_grouped_updates.sources.size()),
                    static_cast<unsigned long long>(batch.deletions.size() + batch.additions.size()),
                    sw_group.ms());
            }

            topology::ChunkStoreBatchMetrics ApplyEffectivePhase(topology::UpdatePhase phase,
                                                                 index_t batch) {
                topology::ChunkStoreBatchMetrics metrics;
                if (AppImplDeviceObject::kSupportsGpuDeletionRepair &&
                    !AppImplDeviceObject::kComponentLabels) {
                    metrics = m_chunk_store->ApplyGroupedPhase(m_grouped_updates, phase, m_reverse_index);
                } else {
                    topology::IgnoreEffectiveUpdates observer;
                    metrics = m_chunk_store->ApplyGroupedPhase(m_grouped_updates, phase, observer);
                }
                LOG("[WORK][batch %u phase=%s] deletion_match_reads=%llu mutation_edge_reads=%llu effective_bytes=%llu\n",
                    batch, phase == topology::UpdatePhase::Delete ? "delete" : "add",
                    static_cast<unsigned long long>(metrics.deletion_match_reads),
                    static_cast<unsigned long long>(metrics.mutation_edge_reads),
                    static_cast<unsigned long long>(metrics.effective_records * sizeof(topology::EffectiveEdgeDelta)));
                LOG("[WORK][batch %u phase=%s] source_plan_duplicate_bytes=%llu source_plan_index_bytes=%llu reverse_input_copy_bytes=%llu\n",
                    batch, phase == topology::UpdatePhase::Delete ? "delete" : "add",
                    static_cast<unsigned long long>(metrics.source_plan_duplicate_bytes),
                    static_cast<unsigned long long>(metrics.source_plan_index_bytes),
                    static_cast<unsigned long long>(m_grouped_updates.LargeMaintenance() ? 0 :
                        metrics.effective_records * sizeof(topology::EffectiveEdgeDelta)));
                const auto &changed = m_chunk_store->ChangedSources();
                m_topology_patch_sources.insert(m_topology_patch_sources.end(), changed.begin(), changed.end());
                if (phase == topology::UpdatePhase::Delete)
                    m_topology_delete_source_count = m_topology_patch_sources.size();
                LOG("[EFFECTIVE][batch %u phase=%s] records=%llu changed_sources=%llu reverse_prepare_ms=%.3f\n",
                    batch, phase == topology::UpdatePhase::Delete ? "delete" : "add",
                    static_cast<unsigned long long>(metrics.effective_records),
                    static_cast<unsigned long long>(changed.size()), metrics.reverse_prepare_ms);
                if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
                    const auto &r = m_reverse_index.LastPrepareMetrics();
                    LOG("[REVERSE-WORK][batch %u phase=%s] input_records=%llu old_records_read=%llu output_records=%llu overlay_records=%llu\n",
                        batch, phase == topology::UpdatePhase::Delete ? "delete" : "add",
                        static_cast<unsigned long long>(r.input_records),
                        static_cast<unsigned long long>(r.old_records_read),
                        static_cast<unsigned long long>(r.output_records),
                        static_cast<unsigned long long>(r.overlay_records));
                    LOG("[REVERSE][batch %u phase=%s] reset_copy_ms=%.3f sort_ms=%.3f group_ms=%.3f slots_ms=%.3f merge_ms=%.3f release_ms=%.3f commit_ms=%.3f destinations=%zu new_destinations=%zu buckets_before=%zu buckets_after=%zu\n",
                        batch, phase == topology::UpdatePhase::Delete ? "delete" : "add",
                        r.reset_copy_ms, r.sort_ms, r.group_ms, r.slots_ms, r.merge_ms,
                        r.release_ms, r.commit_ms, r.destinations, r.new_destinations,
                        r.buckets_before, r.buckets_after);
                }
                return metrics;
            }

            void add_edge_pr(std::pair<index_t,index_t>& local_begin,index_t &NumOfSnapShots){
                const auto metrics = ApplyEffectivePhase(topology::UpdatePhase::Add, NumOfSnapShots);
                LOG("[CPU-MUTATION][batch %u phase=add] epoch=%llu workers=%llu touched=%llu changed=%llu updates=%llu max_source_updates=%llu source_work=%llu max_source_work=%llu written_bytes=%llu relocation_bytes=%llu allocations=%llu reused_blocks=%llu retired_blocks=%llu logical_edges=%llu group_ms=%.3f prepare_ms=%.3f preflight_ms=%.3f allocation_ms=%.3f commit_ms=%.3f apply_ms=%.3f retire_ms=%.3f mutation_ms=%.3f\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(metrics.epoch),
                    static_cast<unsigned long long>(metrics.worker_count),
                    static_cast<unsigned long long>(metrics.touched_sources),
                    static_cast<unsigned long long>(metrics.changed_sources),
                    static_cast<unsigned long long>(metrics.update_count),
                    static_cast<unsigned long long>(metrics.max_source_updates),
                    static_cast<unsigned long long>(metrics.source_work),
                    static_cast<unsigned long long>(metrics.max_source_work),
                    static_cast<unsigned long long>(metrics.mutation_written_bytes),
                    static_cast<unsigned long long>(metrics.relocation_copied_bytes),
                    static_cast<unsigned long long>(metrics.allocations),
                    static_cast<unsigned long long>(metrics.reused_blocks),
                    static_cast<unsigned long long>(metrics.retired_blocks),
                    static_cast<unsigned long long>(m_chunk_store->EdgeCount()),
                    metrics.group_ms, metrics.prepare_ms, metrics.preflight_ms,
                    metrics.allocation_ms, metrics.commit_ms, metrics.apply_ms,
                    metrics.retire_ms, metrics.mutation_ms);
            }

            void del_edge_pr(std::pair<index_t,index_t>& local_begin,index_t &NumOfSnapShots){
                const auto metrics = ApplyEffectivePhase(topology::UpdatePhase::Delete, NumOfSnapShots);
                LOG("[CPU-MUTATION][batch %u phase=delete] epoch=%llu workers=%llu touched=%llu changed=%llu updates=%llu max_source_updates=%llu source_work=%llu max_source_work=%llu missing_deletes=%llu written_bytes=%llu allocations=%llu reused_blocks=%llu retired_blocks=%llu logical_edges=%llu group_ms=%.3f prepare_ms=%.3f preflight_ms=%.3f allocation_ms=%.3f commit_ms=%.3f apply_ms=%.3f retire_ms=%.3f mutation_ms=%.3f\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(metrics.epoch),
                    static_cast<unsigned long long>(metrics.worker_count),
                    static_cast<unsigned long long>(metrics.touched_sources),
                    static_cast<unsigned long long>(metrics.changed_sources),
                    static_cast<unsigned long long>(metrics.update_count),
                    static_cast<unsigned long long>(metrics.max_source_updates),
                    static_cast<unsigned long long>(metrics.source_work),
                    static_cast<unsigned long long>(metrics.max_source_work),
                    static_cast<unsigned long long>(metrics.missing_deletes),
                    static_cast<unsigned long long>(metrics.mutation_written_bytes),
                    static_cast<unsigned long long>(metrics.allocations),
                    static_cast<unsigned long long>(metrics.reused_blocks),
                    static_cast<unsigned long long>(metrics.retired_blocks),
                    static_cast<unsigned long long>(m_chunk_store->EdgeCount()),
                    metrics.group_ms, metrics.prepare_ms, metrics.preflight_ms,
                    metrics.allocation_ms, metrics.commit_ms, metrics.apply_ms,
                    metrics.retire_ms, metrics.mutation_ms);
            }

            void del_edge(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                LOG("----------Batch: %d---------\n",NumOfSnapShots);
                Loader &load_update = *m_load_update;
                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
                Stopwatch sw_reclaim(true);
                const uint64_t reclaimed = m_chunk_store->ReclaimThrough(
                    m_chunk_store->PublishedEpoch());
                sw_reclaim.stop();
                LOG("[RECLAIM][batch %u] completed_epoch=%llu reclaimed_blocks=%llu reclaim_ms=%.3f\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(m_chunk_store->PublishedEpoch()),
                    static_cast<unsigned long long>(reclaimed), sw_reclaim.ms());
                PrepareGroupedUpdates(local_begin, NumOfSnapShots);
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                for(index_t i = local_begin.second; i < local_begin.second+size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    this->del_edges_h[i].u = src_del;
                    this->del_edges_h[i].v = dst_del;
                    this->del_edges_h[i].w = AppImplDeviceObject::TraversalEdgeWeight(src_del, dst_del);
                }
                update_tree_del(local_begin,NumOfSnapShots);
            }

            void read_del(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                Loader &load_update = *m_load_update;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                for(index_t i = local_begin.second; i < local_begin.second+size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    this->del_edges_h[i].u = src_del;
                    this->del_edges_h[i].v = dst_del;
                    this->del_edges_h[i].w = AppImplDeviceObject::TraversalEdgeWeight(src_del, dst_del);
                }
                

                uint32_t work_size[2];
                index_t start = local_begin.second;
                work_size[0] = start;
                work_size[1] = size;
                
                LOG("del start %d \n",work_size[0]);
                LOG("del size %d \n",work_size[1]);
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->del_edges_d), (void *)(this->del_edges_h), 0));
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                if (size == 0) {
                    return;
                }
                

                // Cache validity is changed by the version-checked topology
                // scatter so publication and invalidation share one stream.

            }

            void read_add(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                Loader &load_update = *m_load_update;
                index_t size = load_update.m_batch_size[NumOfSnapShots].first;
                for(index_t i = local_begin.first; i < local_begin.first+size; i++){
                    index_t src_add = load_update.added_edges_w[i].u;
                    index_t dst_add = load_update.added_edges_w[i].v;
                    this->added_edges_h[i].u = src_add;
                    this->added_edges_h[i].v = dst_add;
                    this->added_edges_h[i].w = AppImplDeviceObject::TraversalEdgeWeight(src_add, dst_add);
                    
                }
                
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->added_edges_d), (void *)(this->added_edges_h), 0));
                uint32_t work_size[2];
                work_size[0] = local_begin.first;
                work_size[1] = size;
                m_insertion_seed_begin = work_size[0];
                m_insertion_seed_count = work_size[1];
                
                
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                if (size == 0) {
                    return;
                }
                // Touched cached sources are invalidated by PublishSparse.
                
            }

            // PR keeps signed residuals: shortest-path parent/owner/minimum
            // repair is not a valid propagation policy for additive rank updates.
            void StartPageRank(pr::Runtime &runtime, float epsilon, unsigned max_rounds) {
                auto &datum = *m_graph_datum;
                auto &stream = *m_stream;
                runtime.Init(datum.m_node_value_datum, datum.m_node_buffer_datum, stream.cuda_stream);
                const auto rounds = runtime.Converge(m_vcsr_dev_graph_allocator->DeviceObject(),
                    datum.cache_edges_l1, datum.m_node_value_datum,
                    datum.m_node_buffer_datum, epsilon, max_rounds, stream.cuda_stream);
                LOG("[PR-CONVERGE] stage=initial rounds=%u signed_residual=1 stop=%s active=%u limit=%u\n",
                    rounds, runtime.Converged() ? "converged" : "iteration_limit", runtime.ActiveCount(), max_rounds);
            }

            void UpdatePageRank(pr::Runtime &runtime,
                                std::pair<index_t,index_t> &offset, index_t batch,
                                float epsilon, unsigned max_rounds) {
                cgcomm::Scope comm_scope(cgcomm::Category::Topology);
                auto &datum = *m_graph_datum;
                auto &stream = *m_stream;
                auto &graph = m_vcsr_dev_graph_allocator->DeviceObject();
                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
                m_chunk_store->ReclaimThrough(m_chunk_store->PublishedEpoch());
                PrepareGroupedUpdates(offset, batch);
                std::vector<index_t> sources;
                sources.reserve(m_grouped_updates.sources.size());
                for (auto src : m_grouped_updates.sources)
                    if (src < datum.nnodes) sources.push_back(src);
                runtime.BeginBatch(sources, stream.cuda_stream);
                // r' = r + alpha * (P_new^T - P_old^T) * x.
                // Consume old rows before any in-place CPU mutation or reclaim.
                runtime.Amend(graph, datum.cache_edges_l1, sources.size(),
                    datum.m_node_value_datum, datum.m_node_buffer_datum,
                    -1, epsilon, stream.cuda_stream);
                stream.Sync();
                del_edge_pr(offset, batch);
                add_edge_pr(offset, batch);
                offset.first += m_load_update->m_batch_size[batch].first;
                offset.second += m_load_update->m_batch_size[batch].second;
                FinalizePublicationSources(batch);
                m_vcsr_dev_graph_allocator->PublishSparse(m_topology_patch_sources,
                    stream.cuda_stream, FLAGS_check, datum.cache_edges_l1, datum.num_of_cache);
                // Same stream orders descriptor visibility and cache patching
                // before new-degree contribution seeding and propagation.
                runtime.Amend(graph, datum.cache_edges_l1, sources.size(),
                    datum.m_node_value_datum, datum.m_node_buffer_datum,
                    1, epsilon, stream.cuda_stream);
                runtime.EndBatch(stream.cuda_stream);
                const auto publication = m_vcsr_dev_graph_allocator->CompleteSparsePublication();
                if (publication.stale_rejects || publication.hash_mismatches)
                    throw std::runtime_error("PR topology publication failed");
                const auto rounds = runtime.Converge(graph, datum.cache_edges_l1,
                    datum.m_node_value_datum, datum.m_node_buffer_datum,
                    epsilon, max_rounds, stream.cuda_stream);
                LOG("[PR-BATCH] batch=%u touched_sources=%zu published_sources=%zu rounds=%u publication_ms=%.3f cache_invalidations=%llu stop=%s active=%u limit=%u\n",
                    batch, sources.size(), m_topology_patch_sources.size(), rounds,
                    publication.publication_ms, publication.cache_invalidations,
                    runtime.Converged() ? "converged" : "iteration_limit", runtime.ActiveCount(), max_rounds);
            }

            void Cancelation(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                //回收
                LOG("====this is %d batch====\n",NumOfSnapShots);
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                type[0] = -1;

                EnsureTypeDevicePointer();
                index_t seg_snode,seg_enode;
                index_t stream_id;
                
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildWorklist_AllVertices(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                
                
                Stopwatch sw_execution(true);
                ExecutePolicy_MF(next_policy,type_device);
                
                count();
                sw_execution.stop();
                LOG("取消时间: %f ms (excluded)\n", sw_execution.ms());

                //         RebuildWorklist_AllVertices(app_inst,
                //         graph_datum,

            }

            void Compensate(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                index_t seg_snode,seg_enode;
                index_t stream_id;
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }

                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
                Stopwatch sw_reclaim(true);
                const uint64_t reclaimed = m_chunk_store->ReclaimThrough(
                    m_chunk_store->PublishedEpoch());
                sw_reclaim.stop();
                LOG("[RECLAIM][batch %u] completed_epoch=%llu reclaimed_blocks=%llu reclaim_ms=%.3f\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(m_chunk_store->PublishedEpoch()),
                    static_cast<unsigned long long>(reclaimed), sw_reclaim.ms());

                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();

                //update graph on the cpu
                read_del(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                read_add(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();

                PrepareGroupedUpdates(local_begin, NumOfSnapShots);
                del_edge_pr(local_begin,NumOfSnapShots);
                add_edge_pr(local_begin,NumOfSnapShots);
                local_begin.second += load_update.m_batch_size[NumOfSnapShots].second;
                local_begin.first += load_update.m_batch_size[NumOfSnapShots].first;

                FinalizePublicationSources(NumOfSnapShots);
                m_vcsr_dev_graph_allocator->PublishSparse(
                    m_topology_patch_sources, stream_s.cuda_stream, FLAGS_check,
                    graph_datum.cache_edges_l1, graph_datum.num_of_cache);
                uint64_t zc_cold_edges = m_chunk_store->EdgeCount();
                if (FLAGS_cache != 0) {
                    zc_cold_edges = 0;
                    for (const index_t source : m_topology_patch_sources) {
                        zc_cold_edges += m_chunk_store->Descriptor(source).degree;
                    }
                }
                for (index_t stream_idx = 0; stream_idx < FLAGS_n_stream; ++stream_idx) {
                    m_vcsr_dev_graph_allocator->WaitForSparsePublication(
                        stream[stream_idx].cuda_stream);
                }
                type[0] = 1;
                float time_total = 0;
                EnsureTypeDevicePointer();

                //incremental computation
                Stopwatch sw_execution(true);
                ExecutePolicy_MF(next_policy,type_device);
                
                
                sw_execution.stop();
                time_total +=sw_execution.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildArrayWorklistINC(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx <  FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                const auto publication =
                    m_vcsr_dev_graph_allocator->CompleteSparsePublication();
                if (publication.stale_rejects != 0 ||
                    publication.hash_mismatches != 0) {
                    LOG("[PUBLISH] protocol_error batch=%u stale_version_rejects=%llu gpu_cpu_hash_mismatches=%llu\n",
                        NumOfSnapShots, publication.stale_rejects,
                        publication.hash_mismatches);
                    std::abort();
                }
                LOG("[PUBLISH][batch %u] epoch=%llu patch_records=%zu patch_bytes=%llu h2d_count=%u publication_ms=%.3f stale_version_rejects=%llu cache_invalidations=%llu cache_tail=%llu cache_capacity=%llu zc_cold_edges=%llu gpu_cpu_hash_mismatches=%llu audit=%u\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(m_chunk_store->PublishedEpoch()),
                    m_topology_patch_sources.size(),
                    static_cast<unsigned long long>(m_topology_patch_sources.size() * sizeof(topology::TopologyPatchRecord)),
                    m_topology_patch_sources.empty() ? 0U : 1U,
                    publication.publication_ms, publication.stale_rejects,
                    publication.cache_invalidations,
                    publication.cache_tail,
                    static_cast<unsigned long long>(graph_datum.num_of_cache),
                    static_cast<unsigned long long>(zc_cold_edges),
                    publication.hash_mismatches, FLAGS_check ? 1U : 0U);
                LOG("------------PR mf compensate-----------\n");
                count();
                for(index_t stream_idx = 0; stream_idx <  FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                bool convergence = false;
                int current_rount =0 ;
                
                while(!convergence){
                  Stopwatch sw_execution_round(true);
                  ExecutePolicy_MF_Converge(next_policy);
                
                  current_rount++;
                  sw_execution_round.stop();
                  time_total +=sw_execution_round.ms();
                  PostComputationBW_Inc();
                  if (current_rount == 100 ) {
                        convergence = true;
                        LOG("Max iterations reached\n");
                  }
                }
                

                LOG("迭代时间: %f ms (excluded)\n", time_total);
                LOG("round num : %d  (excluded)\n", current_rount);

            }


            void compact_cache() {
                cgcomm::Scope comm_scope(cgcomm::Category::Cache);
                groute::Stream &stream_s = *m_stream;
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildArrayWorklist_identify(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               cudaDeviceSynchronize();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];
                    stream_id = seg_idx % FLAGS_n_stream;

                    m_vcsr_dev_graph_allocator->SwitchZC();
                    zcflag = true;
                    RunSyncCom(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                               vcsr_graph,
                               graph_datum,
                               m_engine_options,
                        stream[stream_id]);
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               cudaDeviceSynchronize();
               m_vcsr_dev_graph_allocator->HostRiverElement();
               dim3 grid_dims, block_dims;
               const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.num_of_cache);
               KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::copy_cache<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (graph_datum.cache_edges_l1,graph_datum.cache_edges_com,work_source);
                stream_s.Sync();
                cudaDeviceSynchronize();
                const auto &work_source_flush_vertex = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                KernelSizing(grid_dims, block_dims, work_source_flush_vertex.get_size());
                kernel::copy_index<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source_flush_vertex);
                    stream_s.Sync();
                sw_execution.stop();
               LOG("合并缓存时间 cache(compact time): %f ms (excluded)\n", sw_execution.ms());
            }

            void evication_cache(){
                cgcomm::Scope comm_scope(cgcomm::Category::Cache);
                groute::Stream &stream_s = *m_stream;
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();

                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::flush_cache<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                stream_s.Sync();
                sw_execution.stop();
                LOG("缓存逐出时间 %f\n",sw_execution.ms());
            }


            void ExecutePolicy_MF_Converge(AlgoVariant *algo_variant) {
                
                auto &app_inst = *m_app_inst;
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t seg_exc = 0;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];
                    stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_MFC(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);
                    }
                    
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }

            }
            void PostComputationBW_Inc() {
                
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;

                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                    RebuildArrayWorklistINC(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                sw_rebuild.stop();
                uint64_t active_count_overall = 0;
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);
                    graph_datum.seg_active_num[seg_idx] = active_count;

                    m_running_info.input_active_count_seg[seg_idx] = active_count;

                    uint32_t work_size = active_count;
                    dim3 grid_dims, block_dims;
                    

                }
                
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

            void reset_delta_vertices(){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                index_t seg_snode,seg_enode;
                index_t stream_id;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildArrayWorklist_reset_delta(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                   stream[stream_idx].Sync();
                }
            }

            void BeginInsertionEpoch(index_t batch) {
                Stopwatch sw_epoch_prepare(true);
                GraphDatum &graph_datum = *m_graph_datum;
                if (m_device_partition_end_nodes == nullptr) {
                    std::vector<index_t> ends(FLAGS_SEGMENT);
                    for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg)
                        ends[seg] = m_groute_context->seg_enode[seg];
                    GROUTE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_device_partition_end_nodes),
                        sizeof(index_t) * FLAGS_SEGMENT));
                    GROUTE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_device_node_state_epoch),
                        sizeof(uint32_t) * graph_datum.nnodes));
                    GROUTE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_device_seed_reject_count),
                        sizeof(unsigned long long)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_device_partition_end_nodes, ends.data(),
                        sizeof(index_t) * FLAGS_SEGMENT, cudaMemcpyHostToDevice));
                    GROUTE_CUDA_CHECK(cudaMemset(m_device_node_state_epoch, 0xff,
                        sizeof(uint32_t) * graph_datum.nnodes));
                }
                if (m_insertion_epoch_active)
                    throw std::runtime_error("Nested insertion epoch");
                ++m_insertion_epoch;
                if (m_insertion_epoch == 0 || (m_insertion_epoch & 0xffffU) == 0) {
                    GROUTE_CUDA_CHECK(cudaMemset(m_device_node_state_epoch, 0xff,
                        sizeof(uint32_t) * graph_datum.nnodes));
                    if (m_insertion_epoch == 0) m_insertion_epoch = 1;
                }
                GROUTE_CUDA_CHECK(cudaMemset(m_device_seed_reject_count, 0,
                    sizeof(unsigned long long)));
                m_insertion_epoch_active = true;
                m_exact_source_frontier_ready = false;
                m_exact_source_frontier_count = 0;
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].ResetAsync(m_stream->cuda_stream);
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1].ResetAsync(m_stream->cuda_stream);
                m_insertion_convergence_round = 0;
                sw_epoch_prepare.stop();
                LOG("[GPU-EPOCH-BEGIN] batch=%u epoch=%u epoch_prepare_ms=%.3f\n",
                    batch, m_insertion_epoch, sw_epoch_prepare.ms());
            }


            void EndInsertionEpoch(index_t batch) {
                if (!m_insertion_epoch_active) {
                    return;
                }
                uint64_t local_active = 0;
                for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg) {
                    local_active += m_graph_datum->seg_active_num[seg];
                }
                unsigned long long rejected_seeds = 0;
                const uint32_t exact_input_active =
                    m_graph_datum->m_wl_array_in_seg[FLAGS_SEGMENT].GetCount(*m_stream);
                const uint32_t exact_output_active =
                    m_graph_datum->m_wl_array_in_seg[FLAGS_SEGMENT + 1].GetCount(*m_stream);
                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &rejected_seeds,
                    m_device_seed_reject_count,
                    sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));
                if (rejected_seeds != 0 || local_active != 0 ||
                    exact_input_active != 0 ||
                    exact_output_active != 0) {
                    LOG("[GPU-EPOCH] protocol_error=epoch_not_quiescent epoch=%u rejected=%llu local_active=%llu exact_input=%u exact_output=%u\n",
                        m_insertion_epoch,
                        rejected_seeds,
                        static_cast<unsigned long long>(local_active),
                        exact_input_active,
                        exact_output_active);
                    std::abort();
                }
                m_insertion_epoch_active = false;
                LOG("[GPU-EPOCH-END] batch=%u epoch=%u seed_rejects=%llu local_active=%llu\n",
                    batch,
                    m_insertion_epoch,
                    rejected_seeds,
                    static_cast<unsigned long long>(local_active));
            }

            void update_tree_add(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                //init the zero copy policy
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                Stopwatch sw_update_reset(true);
                read_del(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                read_add(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                sw_update_reset.stop();

                Stopwatch sw_cpu_mutation(true);
                add_edge_pr(local_begin,NumOfSnapShots);
                sw_cpu_mutation.stop();
                local_begin.second += load_update.m_batch_size[NumOfSnapShots].second;
                local_begin.first += load_update.m_batch_size[NumOfSnapShots].first;

                FinalizePublicationSources(NumOfSnapShots);
                m_vcsr_dev_graph_allocator->PublishSparse(
                    m_topology_patch_sources, stream_s.cuda_stream, FLAGS_check,
                    graph_datum.cache_edges_l1, graph_datum.num_of_cache);
                for (index_t stream_idx = 0; stream_idx < FLAGS_n_stream;
                     ++stream_idx) {
                    m_vcsr_dev_graph_allocator->WaitForSparsePublication(
                        stream[stream_idx].cuda_stream);
                }
                if (!m_chunk_store->IsPublished()) {
                    LOG("[TOPOLOGY-EPOCH] protocol_error=traversal_before_publication batch=%u pending=%llu published=%llu\n",
                        NumOfSnapShots,
                        static_cast<unsigned long long>(
                            m_chunk_store->PendingEpoch()),
                        static_cast<unsigned long long>(
                            m_chunk_store->PublishedEpoch()));
                    std::abort();
                }
                uint64_t zc_cold_edges = m_chunk_store->EdgeCount();
                if (FLAGS_cache != 0) {
                    zc_cold_edges = 0;
                    for (const index_t source : m_topology_patch_sources) {
                        zc_cold_edges += m_chunk_store->Descriptor(source).degree;
                    }
                }
                if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
                    BeginInsertionEpoch(NumOfSnapShots);
                }
                index_t seg_snode,seg_enode;
                index_t stream_id;
                float add_time = 0;
                Stopwatch sw_load(true);
                if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
                    const index_t add_size = load_update.m_batch_size[NumOfSnapShots].first;
                    if (add_size > 0) {
                        dim3 grid_dims, block_dims;
                        KernelSizing(grid_dims, block_dims, add_size);
                        SeedAddedEdgeFrontier<TValue><<<grid_dims,
                                                       block_dims,
                                                       0,
                                                       stream_s.cuda_stream>>>(
                            this->added_edges_d,
                            this->work_size_d,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetParentDeviceObject(),
                            std::numeric_limits<TValue>::max(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            m_device_partition_end_nodes,
                            FLAGS_SEGMENT,
                            m_insertion_epoch,
                            m_device_node_state_epoch,
                            m_device_seed_reject_count,
                            graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].DeviceObject());
                        stream_s.Sync();
                    }
                    LOG("[SSSP-SEED-FRONTIER] batch=%u added_edges=%u mode=direct_added_edge\n",
                        NumOfSnapShots,
                        add_size);
                } else {
                    for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                        stream_id = seg_idx % FLAGS_n_stream;
                        seg_snode = m_groute_context->seg_snode[seg_idx];
                        seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildWorklist_AllVertices(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                    }
                    for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                        stream[stream_idx].Sync();
                    }
                    ExecutePolicy_All(next_policy);
                }
                sw_load.stop();
                add_time+=sw_load.ms();
                Stopwatch sw_initial_rebuild(true);
                cudaDeviceSynchronize();
                const auto publication =
                    m_vcsr_dev_graph_allocator->CompleteSparsePublication();
                if (publication.stale_rejects != 0 ||
                    publication.hash_mismatches != 0) {
                    LOG("[PUBLISH] protocol_error batch=%u stale_version_rejects=%llu gpu_cpu_hash_mismatches=%llu\n",
                        NumOfSnapShots, publication.stale_rejects,
                        publication.hash_mismatches);
                    std::abort();
                }
                LOG("[PUBLISH][batch %u] epoch=%llu patch_records=%zu patch_bytes=%llu h2d_count=%u publication_ms=%.3f stale_version_rejects=%llu cache_invalidations=%llu cache_tail=%llu cache_capacity=%llu zc_cold_edges=%llu gpu_cpu_hash_mismatches=%llu audit=%u\n",
                    NumOfSnapShots,
                    static_cast<unsigned long long>(m_chunk_store->PublishedEpoch()),
                    m_topology_patch_sources.size(),
                    static_cast<unsigned long long>(m_topology_patch_sources.size() * sizeof(topology::TopologyPatchRecord)),
                    m_topology_patch_sources.empty() ? 0U : 1U,
                    publication.publication_ms, publication.stale_rejects,
                    publication.cache_invalidations,
                    publication.cache_tail,
                    static_cast<unsigned long long>(graph_datum.num_of_cache),
                    static_cast<unsigned long long>(zc_cold_edges),
                    publication.hash_mismatches, FLAGS_check ? 1U : 0U);
                const bool exact_all_gpu = m_insertion_epoch_active;
                if (exact_all_gpu) {
                    m_exact_source_frontier_ready = true;
                    m_exact_source_frontier_count =
                        graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].GetCount(stream_s);
                    for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg) {
                        graph_datum.seg_active_num[seg] = 0;
                        m_running_info.input_active_count_seg[seg] = 0;
                    }
                    if (m_exact_source_frontier_count != 0) {
                        graph_datum.seg_active_num[0] = m_exact_source_frontier_count;
                        m_running_info.input_active_count_seg[0] =
                            m_exact_source_frontier_count;
                    }
                    LOG("[EXACT-SOURCE-SEED] batch=%u exact_sources=%u seed_partition_rebuilds=0\n",
                        NumOfSnapShots, m_exact_source_frontier_count);
                } else {
                    for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                        stream_id = seg_idx % FLAGS_n_stream;

                        seg_snode = m_groute_context->seg_snode[seg_idx];
                        seg_enode = m_groute_context->seg_enode[seg_idx];
                        RebuildArrayWorklist(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx,
                            m_insertion_epoch_active ? m_device_node_state_epoch : nullptr,
                            m_insertion_epoch);
                    }
                    for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                        stream[stream_idx].Sync();
                    }
                    RefreshSegmentActiveCountsFromQueues();
                    // Seeds now live in segmented worklists. The duplicate exact queue
                    // has no consumer on this fallback path.
                    graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].ResetAsync(
                        stream_s.cuda_stream);
                    stream_s.Sync();
                }
                sw_initial_rebuild.stop();
                
                bool convergence = false;
                m_running_info.current_round = 0;
                Stopwatch sw_con(true);
                if (exact_all_gpu) {
                    RunExactSourceClosure();
                    convergence = true;
                }
                while(!convergence){
                
                  ExecutePolicy_Converge(next_policy);
                  ++m_insertion_convergence_round;
                
                
                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                  if (m_insertion_convergence_round > graph_datum.nnodes) {
                        LOG("[GPU-EPOCH] protocol_error=non_quiescent epoch=%u rounds=%llu bound=%u\n",
                            m_insertion_epoch,
                            static_cast<unsigned long long>(m_insertion_convergence_round),
                            graph_datum.nnodes);
                        std::abort();
                  }
                }
                sw_con.stop();
                if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
                    EndInsertionEpoch(NumOfSnapShots);
                }
                add_time+=sw_con.ms();
                LOG("[INSERTION-STAGE] batch=%u reset_ms=%.3f cpu_mutation_ms=%.3f topology_publication_audit_ms=%.3f initial_rebuild_ms=%.3f seed_ms=%.3f converge_ms=%.3f\n",
                    NumOfSnapShots,
                    sw_update_reset.ms(),
                    sw_cpu_mutation.ms(),
                    publication.publication_ms,
                    sw_initial_rebuild.ms(),
                    sw_load.ms(),
                    sw_con.ms());
                LOG("total add time: %f ms (excluded)\n", add_time);
                
                
            }

            void RunExactSourceClosure() {
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                auto &input = graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT];
                auto &output = graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1];
                if (!m_dev_props.cooperativeLaunch) {
                    LOG("[EXACT-SOURCE] protocol_error=cooperative_launch_unsupported device=%s\n",
                        m_dev_props.name);
                    std::abort();
                }
                if (m_device_exact_source_counts == nullptr) {
                    GROUTE_CUDA_CHECK(cudaMalloc(
                        reinterpret_cast<void **>(&m_device_exact_source_counts),
                        5 * sizeof(unsigned long long)));
                }
                GROUTE_CUDA_CHECK(cudaMemset(m_device_exact_source_counts, 0,
                    5 * sizeof(unsigned long long)));
                const char *schedule_env = std::getenv("CG_INSERTION_SCHEDULE");
                // Large-diameter mode owns both ordered phases; stale schedule
                // environment settings cannot disable its insertion policy.
                const char *ordered_env = std::getenv("CG_ORDERED_REPAIR");
                const bool large_diameter = ordered_env && std::string(ordered_env) == "1";
                const std::string schedule = large_diameter ? "ordered" :
                    (schedule_env ? schedule_env : "block");
                if (schedule != "block" && schedule != "ordered")
                    throw std::invalid_argument("CG_INSERTION_SCHEDULE must be block or ordered");
                const auto graph = m_vcsr_dev_graph_allocator->DeviceObject();
                using Graph = decltype(graph);
                auto kernel = schedule == "ordered"
                        ? RunExactGpuClosure<AppImplDeviceObject, Graph, TValue, TBuffer, true>
                        : RunExactGpuClosure<AppImplDeviceObject, Graph, TValue, TBuffer>;
                int blocks_per_sm = 0;
                GROUTE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, kernel, 128, 0));
                const int grid_blocks = blocks_per_sm * m_dev_props.multiProcessorCount;
                if (grid_blocks <= 0) {
                    LOG("[EXACT-SOURCE] protocol_error=no_resident_cooperative_blocks\n");
                    std::abort();
                }
                auto input_device = input.DeviceObject();
                auto output_device = output.DeviceObject();
                TValue *values = graph_datum.GetValueDeviceObject();
                TBuffer *buffers = graph_datum.GetBufferDeviceObject();
                TValue *parents = graph_datum.GetParentDeviceObject();
                uint32_t epoch = m_insertion_epoch;
                uint32_t *node_state_epoch = m_device_node_state_epoch;
                unsigned long long *metrics = m_device_exact_source_counts;
                void *arguments[] = {&app_inst, &input_device, &output_device,
                    const_cast<void **>(reinterpret_cast<void *const *>(&graph)),
                    &values, &buffers, &parents, &epoch, &node_state_epoch,
                    &metrics};
                GROUTE_CUDA_CHECK(cudaLaunchCooperativeKernel(
                    reinterpret_cast<void *>(kernel),
                    grid_blocks, 128, arguments, 0, m_stream->cuda_stream));
                m_stream->Sync();
                LOG("[SCHEDULE] mode=%s blocks=%d threads=128 selection=%s large_diameter=%d\n",
                    schedule.c_str(), grid_blocks,
                    large_diameter ? "large_diameter_mode" :
                        (schedule_env ? "explicit" : "mode_default"), int(large_diameter));
                unsigned long long host_metrics[4] = {0, 0, 0, 0};
                GROUTE_CUDA_CHECK(cudaMemcpy(host_metrics,
                    m_device_exact_source_counts, sizeof(host_metrics),
                    cudaMemcpyDeviceToHost));
                if (host_metrics[3] > 0xffffULL) {
                    LOG("[EXACT-SOURCE] protocol_error=wave_ticket_overflow epoch=%u waves=%llu\n",
                        m_insertion_epoch, host_metrics[3]);
                    std::abort();
                }
                m_insertion_convergence_round = host_metrics[3];
                m_exact_source_frontier_count = 0;
                for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg) {
                    graph_datum.seg_active_num[seg] = 0;
                    m_running_info.input_active_count_seg[seg] = 0;
                }
                LOG("[EXACT-SOURCE-CLOSURE] epoch=%u offered_sources=%llu processed_sources=%llu processed_edges=%llu local_waves=%llu seed_partition_rebuilds=0 host_frontier_syncs=0 final_quiescence_syncs=1\n",
                    m_insertion_epoch, host_metrics[0], host_metrics[1],
                    host_metrics[2], host_metrics[3]);
            }

            void CollectDeletionAffectedVertices(index_t batch) {
                Stopwatch sw_d2h(true);
                const uint32_t affected_count =
                    m_device_affected_vertices.GetCount(*m_stream);
                m_affected_vertices.resize(affected_count);
                if (affected_count != 0) {
                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        m_affected_vertices.data(),
                        m_device_affected_vertices.GetDeviceDataPtr(),
                        sizeof(index_t) * affected_count,
                        cudaMemcpyDeviceToHost));
                }
                sw_d2h.stop();

                std::vector<uint8_t> active_partitions(FLAGS_SEGMENT, 0);
                for (const index_t node : m_affected_vertices) {
                    const index_t *begin = m_groute_context->seg_enode;
                    const index_t *end = begin + FLAGS_SEGMENT;
                    const index_t *position = std::upper_bound(begin, end, node);
                    if (position != end) {
                        active_partitions[position - begin] = 1;
                    }
                }
                const uint32_t active_partition_count = static_cast<uint32_t>(
                    std::count(active_partitions.begin(), active_partitions.end(), uint8_t{1}));
                const uint64_t metadata_device_bytes =
                    static_cast<uint64_t>(m_graph_datum->nnodes) * sizeof(index_t);
                LOG("[AFFECTED][batch %u] vertices=%u active_partitions=%u d2h_bytes=%llu d2h_ms=%.3f metadata_device_bytes=%llu\n",
                    batch,
                    affected_count,
                    active_partition_count,
                    static_cast<unsigned long long>(sizeof(index_t) * affected_count),
                    sw_d2h.ms(),
                    static_cast<unsigned long long>(metadata_device_bytes));
            }




            bool UseCcUnionRepair() const {
                if constexpr (!AppImplDeviceObject::kComponentLabels) return false;
                const char *mode = std::getenv("CG_CC_REPAIR");
                if (mode && std::string(mode) != "union" && std::string(mode) != "pull")
                    throw std::invalid_argument("CG_CC_REPAIR must be union or pull");
                const char *ordered = std::getenv("CG_ORDERED_REPAIR");
                return (!mode || std::string(mode) == "union") &&
                    !(ordered && std::string(ordered) == "1");
            }

            void RunCcUnionRepair(index_t batch, uint32_t count) {
                if constexpr (AppImplDeviceObject::kComponentLabels) {
                auto &g = *m_graph_datum;
                Stopwatch timer(true);
                const char *sampling = std::getenv("CG_CC_SAMPLED_REPAIR");
                if (sampling && std::string(sampling) != "0" && std::string(sampling) != "1")
                    throw std::invalid_argument("CG_CC_SAMPLED_REPAIR must be 0 or 1");
                const bool sampled = !sampling || std::string(sampling) == "1";
                if (count) {
                    const auto &changed = m_chunk_store->ChangedSources();
                    const auto *patches = m_vcsr_dev_graph_allocator->StageDeletionDescriptors(
                        changed, m_stream->cuda_stream);
                    if (!changed.empty())
                        cc_repair::MarkChanged<<<(changed.size() + 255) / 256, 256, 0,
                            m_stream->cuda_stream>>>(patches, changed.size(), g.GetParentDeviceObject());
                    const unsigned blocks = std::min<uint64_t>((uint64_t(count) + 7) / 8, 65535);
                    if (sampled) {
                        cc_repair::Sample<<<blocks, 256, 0, m_stream->cuda_stream>>>(
                            m_vcsr_dev_graph_allocator->DeviceObject(),
                            m_device_affected_vertices.GetDeviceDataPtr(), count,
                            g.GetValueDeviceObject(), g.GetParentDeviceObject(), g.cache_edges_l1);
                        cc_repair::Snapshot<<<blocks, 256, 0, m_stream->cuda_stream>>>(
                            m_device_affected_vertices.GetDeviceDataPtr(), count,
                            g.GetValueDeviceObject(), g.GetBufferDeviceObject());
                        cc_repair::SelectComponent<<<1, 256, 0, m_stream->cuda_stream>>>(
                            m_device_affected_vertices.GetDeviceDataPtr(), count,
                            g.GetBufferDeviceObject(), this->work_size_d);
                        cc_repair::Hook<true><<<blocks, 256, 0, m_stream->cuda_stream>>>(
                            m_vcsr_dev_graph_allocator->DeviceObject(),
                            m_device_affected_vertices.GetDeviceDataPtr(), count,
                            g.GetValueDeviceObject(), g.GetParentDeviceObject(), g.cache_edges_l1,
                            g.GetBufferDeviceObject(), this->work_size_d);
                    } else cc_repair::Hook<<<blocks, 256, 0, m_stream->cuda_stream>>>(
                        m_vcsr_dev_graph_allocator->DeviceObject(),
                        m_device_affected_vertices.GetDeviceDataPtr(), count,
                        g.GetValueDeviceObject(), g.GetParentDeviceObject(), g.cache_edges_l1);
                    cc_repair::Flatten<<<blocks, 256, 0, m_stream->cuda_stream>>>(
                        m_device_affected_vertices.GetDeviceDataPtr(), count,
                        g.GetValueDeviceObject(), g.GetBufferDeviceObject(), g.m_node_reset_datum,
                        g.GetParentDeviceObject());
                    GROUTE_CUDA_CHECK(cudaGetLastError());
                }
                m_stream->Sync();
                timer.stop();
                LOG("[CC-UNION-REPAIR][batch %u] affected=%u sampled=%u cache_unchanged_rows=1 path_splitting=1 adjacency_copy_bytes=0 temporary_device_bytes=0 closure_ms=%.3f\n",
                    batch, count, sampled ? 1U : 0U, timer.ms());
                }
            }

            void RunGpuAffectedRepair(index_t batch) {
                if (m_affected_vertices.empty()) {
                    m_gpu_repair_incoming_offsets.assign(1, 0);
                    m_gpu_repair_incoming_sources.clear();
                    LOG("[GPU-REPAIR][batch %u] affected=0\n", batch);
                    return;
                }

                Stopwatch sw_topology(true);
                runtime::DynamicReverseIndex::MaterializeMetrics incoming_metrics;
                if constexpr (AppImplDeviceObject::kComponentLabels) {
                    m_gpu_repair_incoming_offsets.assign(m_affected_vertices.size() + 1, 0);
                    for (size_t i = 0; i < m_affected_vertices.size(); ++i)
                        m_gpu_repair_incoming_offsets[i + 1] = m_gpu_repair_incoming_offsets[i] +
                            m_chunk_store->Descriptor(m_affected_vertices[i]).degree;
                    m_gpu_repair_incoming_sources.resize(m_gpu_repair_incoming_offsets.back());
                    m_chunk_store->MutationWorkers().Run(m_affected_vertices.size(), [&](size_t i) {
                        const auto &row = m_chunk_store->Descriptor(m_affected_vertices[i]);
                        if (row.degree == 0) return;
                        std::copy_n(m_chunk_store->SlabData(row.slab_id) + row.index, row.degree,
                            m_gpu_repair_incoming_sources.data() + m_gpu_repair_incoming_offsets[i]);
                    });
                    incoming_metrics.base_edges_scanned = m_gpu_repair_incoming_sources.size();
                    incoming_metrics.output_sources = m_gpu_repair_incoming_sources.size();
                } else {
                    incoming_metrics = m_reverse_index.MaterializeIncoming(
                        m_affected_vertices, m_gpu_repair_incoming_offsets,
                        m_gpu_repair_incoming_sources);
                }
                sw_topology.stop();


                const size_t topology_budget = runtime::RepairTopologyStorage::Budget(
                    m_gpu_repair_topology.DeviceBytes());
                runtime::RepairTopologyStorage::Lease topology_lease{m_gpu_repair_topology};
                Stopwatch sw_allocate(true);
                m_gpu_repair_topology.Bind(m_gpu_repair_incoming_offsets,
                                          m_gpu_repair_incoming_sources, topology_budget);
                sw_allocate.stop();

                Stopwatch sw_h2d(true);
                {
                    cgcomm::Scope comm_scope(cgcomm::Category::Incoming);
                    m_gpu_repair_topology.Upload(m_gpu_repair_incoming_offsets,
                                                m_gpu_repair_incoming_sources);
                }
                sw_h2d.stop();
                LOG("[REPAIR-STORAGE][batch %u] budget_bytes=%llu device_bytes=%llu mapped_bytes=%llu h2d_bytes=%llu\n",
                    batch, static_cast<unsigned long long>(topology_budget),
                    static_cast<unsigned long long>(m_gpu_repair_topology.DeviceBytes()),
                    static_cast<unsigned long long>(m_gpu_repair_topology.MappedBytes()),
                    static_cast<unsigned long long>(m_gpu_repair_topology.TransferBytes()));

                GraphDatum &graph_datum = *m_graph_datum;
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, m_affected_vertices.size());
                Stopwatch sw_closure(true);


                unsigned int changed = 0;
                uint32_t iterations = 0;
                const char* ordered_env=std::getenv("CG_ORDERED_REPAIR");
                const bool ordered=ordered_env && std::string(ordered_env)=="1";
                if (ordered) {
                    const auto metrics=ordered_repair::Run<AppImplDeviceObject>(
                        graph_datum.nnodes,m_affected_vertices,m_gpu_repair_incoming_offsets,
                        m_gpu_repair_incoming_sources,m_device_affected_vertices.GetDeviceDataPtr(),
                        m_gpu_repair_topology.Offsets(),m_gpu_repair_topology.Sources(),
                        graph_datum.GetValueDeviceObject(),graph_datum.GetBufferDeviceObject(),
                        graph_datum.GetParentDeviceObject(),
                        topology_budget - m_gpu_repair_topology.DeviceBytes());
                    iterations=metrics.iterations;
                    LOG("[ORDERED][batch %u] prepare_ms=%.3f closure_ms=%.3f publish_ms=%.3f internal_scans=%llu queue_peak=%llu extra_device_bytes=%llu iterations=%u\n",
                        batch,metrics.prepare_ms,metrics.closure_ms,metrics.publish_ms,
                        (unsigned long long)metrics.scans,(unsigned long long)metrics.queue_peak,
                        (unsigned long long)metrics.device_bytes,iterations);
                } else {
                    do {
                        GROUTE_CUDA_CHECK(cudaMemset(
                            m_device_gpu_repair_changed, 0, sizeof(unsigned int)));
                        GpuAffectedPullRelax<AppImplDeviceObject, TValue, TBuffer>
                            <<<grid_dims, block_dims>>>(
                                *m_app_inst,
                                m_device_affected_vertices.GetDeviceDataPtr(),
                                static_cast<uint32_t>(m_affected_vertices.size()),
                                m_gpu_repair_topology.Offsets(),
                                m_gpu_repair_topology.Sources(),
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.GetParentDeviceObject(),
                                std::numeric_limits<TValue>::max(),
                                m_device_gpu_repair_changed);
                        GROUTE_CUDA_CHECK(cudaMemcpy(
                            &changed,
                            m_device_gpu_repair_changed,
                            sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));

                        ++iterations;
                        if (iterations > m_affected_vertices.size() + 1) {
                            LOG("[GPU-REPAIR] protocol_error=no_convergence batch=%u iterations=%u affected=%llu\n",
                                batch,
                                iterations,
                                static_cast<unsigned long long>(m_affected_vertices.size()));
                            std::abort();
                        }
                    } while (changed != 0);
                }
                FinalizeGpuAffectedRepair<TValue, TBuffer><<<grid_dims, block_dims>>>(
                    m_device_affected_vertices.GetDeviceDataPtr(),
                    static_cast<uint32_t>(m_affected_vertices.size()),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.m_node_reset_datum);
                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
                sw_closure.stop();

                const uint64_t h2d_bytes = m_gpu_repair_topology.TransferBytes();
                const uint64_t device_bytes = m_gpu_repair_topology.DeviceBytes() +
                    sizeof(unsigned int);
                LOG("[GPU-REPAIR][batch %u] affected=%llu incoming_edges=%llu base_edges_scanned=%llu delta_records_scanned=%llu merge_output_sources=%llu topology_ms=%.3f allocation_ms=%.3f h2d_bytes=%llu h2d_ms=%.3f iterations=%u closure_ms=%.3f device_bytes=%llu\n",
                    batch,
                    static_cast<unsigned long long>(m_affected_vertices.size()),
                    static_cast<unsigned long long>(m_gpu_repair_incoming_sources.size()),
                    static_cast<unsigned long long>(incoming_metrics.base_edges_scanned),
                    static_cast<unsigned long long>(incoming_metrics.delta_records_scanned),
                    static_cast<unsigned long long>(incoming_metrics.output_sources),
                    sw_topology.ms(),
                    sw_allocate.ms(),
                    static_cast<unsigned long long>(h2d_bytes),
                    sw_h2d.ms(),
                    iterations,
                    sw_closure.ms(),
                    static_cast<unsigned long long>(device_bytes));
            }

            // Only suppress a seed when its old label has a surviving rooted
            // witness after the entire deletion batch has committed.
            bool CertifyCcDeletionSeeds(index_t start, index_t size, index_t batch,
                                       thrust::device_vector<uint8_t>& device_certified) {
                if constexpr (AppImplDeviceObject::kRootedDeletionWitness) {
                    Stopwatch timer(true);
                    device_certified.resize(size);
                    dim3 grid, block;
                    KernelSizing(grid, block, size);
                    MarkCcDeletionCandidates<<<grid, block, 0, m_stream->cuda_stream>>>(
                        del_edges_d, start, size, m_graph_datum->GetValueDeviceObject(),
                        thrust::raw_pointer_cast(device_certified.data()));
                    std::vector<uint8_t> certified(size);
                    GROUTE_CUDA_CHECK(cudaMemcpyAsync(certified.data(),
                        thrust::raw_pointer_cast(device_certified.data()), size,
                        cudaMemcpyDeviceToHost, m_stream->cuda_stream));
                    m_stream->Sync();
                    const uint32_t candidates = std::count(certified.begin(), certified.end(), uint8_t{0});
                    std::vector<uint32_t> labels;
                    if (candidates != 0) {
                        labels.resize(m_graph_datum->nnodes);
                        GROUTE_CUDA_CHECK(cudaMemcpy(labels.data(),
                            m_graph_datum->GetValueDeviceObject(), labels.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
                    }
                    runtime::DirectedLabelWitness witness(labels,
                        std::max<uint64_t>(1000000, uint64_t(candidates) * 4096),
                        std::max<uint64_t>(100000, uint64_t(candidates) * 128));
                    auto outgoing = [&](uint32_t source, auto visit) {
                        const auto& row = m_chunk_store->Descriptor(source);
                        if (row.degree == 0) return;
                        const auto* destinations = m_chunk_store->SlabData(row.slab_id) + row.index;
                        for (uint32_t i = 0; i < row.degree; ++i)
                            if (!visit(destinations[i])) break;
                    };
                    std::vector<uint32_t> proof_targets;
                    for (index_t i = 0; i < size; ++i)
                        if (!certified[i]) proof_targets.push_back(del_edges_h[start + i].v);
                    const char* forward_env = std::getenv("CG_CC_FORWARD_WITNESS");
                    if (!(forward_env && forward_env[0] == '0'))
                        witness.SeedRoots(proof_targets, outgoing);
                    uint32_t proven = 0;
                    std::vector<uint32_t> affected;
                    for (index_t i = 0; i < size; ++i) {
                        if (certified[i]) continue;
                        if (witness.Prove(del_edges_h[start + i].v, m_reverse_index)) {
                            certified[i] = 1;
                            ++proven;
                        } else {
                            affected.push_back(del_edges_h[start + i].v);
                        }
                    }
                    uint64_t forward_edges = 0;
                    const bool closed = witness.ExpandAffected(affected, m_reverse_index,
                        outgoing, forward_edges);
                    if (closed && !affected.empty()) {
                        GROUTE_CUDA_CHECK(cudaMemcpy(m_device_affected_vertices.GetDeviceDataPtr(),
                            affected.data(), affected.size() * sizeof(index_t), cudaMemcpyHostToDevice));
                        m_device_affected_vertices.SetLength(*m_stream, affected.size());
                        KernelSizing(grid, block, affected.size());
                        MarkCcWitnessAffected<<<grid, block, 0, m_stream->cuda_stream>>>(
                            m_device_affected_vertices.GetDeviceDataPtr(), affected.size(),
                            m_graph_datum->GetParentDeviceObject(), m_graph_datum->m_node_reset_datum);
                        m_stream->Sync();
                    }
                    if (!closed && proven != 0) {
                        GROUTE_CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(device_certified.data()),
                            certified.data(), size, cudaMemcpyHostToDevice));
                    }
                    timer.stop();
                    LOG("[CC-FORWARD-WITNESS][batch %u] vertices=%llu edges=%llu certified=%llu\n",
                        batch, (unsigned long long)witness.SeedVertices(),
                        (unsigned long long)witness.SeedEdges(),
                        (unsigned long long)witness.SeedCertified());
                    LOG("[CC-WITNESS-SCRATCH][batch %u] reset_entries=%llu avoided_bucket_slots=%llu\n",
                        batch, (unsigned long long)witness.ResetEntries(),
                        (unsigned long long)witness.AvoidedBucketSlots());
                    LOG("[CC-WITNESS-FRONTIER][batch %u] closed=%u affected=%llu forward_edges=%llu\n",
                        batch, closed ? 1U : 0U, (unsigned long long)affected.size(),
                        (unsigned long long)forward_edges);
                    LOG("[CC-ROOTED-WITNESS][batch %u] candidates=%u certified=%u fallback=%u searched_vertices=%llu searched_edges=%llu state_d2h_bytes=%llu total_ms=%.3f\n",
                        batch, candidates, proven, candidates - proven,
                        (unsigned long long)witness.Vertices(), (unsigned long long)witness.Edges(),
                        (unsigned long long)(labels.size() * sizeof(uint32_t)), timer.ms());
                    return closed;
                }
                return false;
            }

            void update_tree_del(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->del_edges_d), (void *)(this->del_edges_h), 0));
                dim3 grid_dims, block_dims;
                uint32_t work_size[2];
                index_t start = local_begin.second;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                m_device_affected_vertices.ResetAsync(stream_s.cuda_stream);
                m_affected_vertices.clear();
                GROUTE_CUDA_CHECK(cudaMemsetAsync(graph_datum.m_node_reset_datum,
                                                  0,
                                                  sizeof(bool) * graph_datum.nnodes,
                                                  stream_s.cuda_stream));
                if (size == 0) {
                    del_edge_pr(local_begin, NumOfSnapShots);
                    stream_s.Sync();
                    m_gpu_repair_incoming_offsets.assign(1, 0);
                    m_gpu_repair_incoming_sources.clear();
                    LOG("[GPU-REPAIR][batch %u] affected=0 reason=no_deleted_edges\n",
                        NumOfSnapShots);
                    return;
                }
                work_size[0] = start;
                work_size[1] = size;
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                KernelSizing(grid_dims, block_dims, size);
                Stopwatch sw_del(true);
                // CC must validate witnesses against the final topology of this
                // deletion phase, including parallel-edge multiplicities.
                const char* witness_env = std::getenv("CG_CC_ROOTED_WITNESS");
                const bool certify_cc = AppImplDeviceObject::kRootedDeletionWitness &&
                    !(witness_env && witness_env[0] == '0');
                double early_delete_ms = 0;
                thrust::device_vector<uint8_t> certified_seeds;
                bool cc_witness_closed = false;
                if (certify_cc) {
                    Stopwatch physical(true);
                    del_edge_pr(local_begin, NumOfSnapShots);
                    physical.stop();
                    early_delete_ms = physical.ms();
                    cc_witness_closed = CertifyCcDeletionSeeds(start, size, NumOfSnapShots, certified_seeds);
                }
                Stopwatch sw_reset_seed(true);
                if constexpr (AppImplDeviceObject::kComponentLabels) {
                    MarkDeletedComponents<<<grid_dims, block_dims, 0, stream_s.cuda_stream>>>(
                        this->del_edges_d, start, size, graph_datum.nnodes,
                        graph_datum.GetValueDeviceObject(), graph_datum.GetBufferDeviceObject());
                    KernelSizing(grid_dims, block_dims, graph_datum.nnodes);
                    CollectDeletedComponents<<<grid_dims, block_dims, 0, stream_s.cuda_stream>>>(
                        graph_datum.nnodes, graph_datum.GetValueDeviceObject(),
                        graph_datum.GetBufferDeviceObject(), graph_datum.GetParentDeviceObject(),
                        graph_datum.m_node_reset_datum, m_device_affected_vertices.DeviceObject());
                } else if (!cc_witness_closed) {
                    kernel::reset_del_edges<<<grid_dims, block_dims, 0, stream_s.cuda_stream>>>(
                        app_inst,
                        graph_datum.GetParentDeviceObject(),
                        graph_datum.GetValueDeviceObject(),
                        graph_datum.GetBufferDeviceObject(),
                        this->del_edges_d,
                        this->work_size_d,
                        graph_datum.m_node_reset_datum,
                        m_device_affected_vertices.DeviceObject(),
                        certified_seeds.empty() ? nullptr :
                            thrust::raw_pointer_cast(certified_seeds.data()));
                }
                stream_s.Sync();
                sw_reset_seed.stop();

                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                m_vcsr_dev_graph_allocator->SwitchZC();
                Stopwatch sw_invalidation(true);
                uint32_t frontier_begin = 0;
                uint32_t frontier_end = m_device_affected_vertices.GetCount(stream_s);
                uint32_t invalidation_rounds = 0;
                while (!cc_witness_closed && !AppImplDeviceObject::kComponentLabels && frontier_begin < frontier_end) {
                    const uint32_t frontier_count = frontier_end - frontier_begin;
                    RunSyncPushDDB_del(
                        app_inst,
                        true,
                        vcsr_graph,
                        graph_datum,
                        m_device_affected_vertices.GetDeviceDataPtr() + frontier_begin,
                        frontier_count,
                        m_device_affected_vertices.DeviceObject(),
                        m_engine_options,
                        stream_s);
                    stream_s.Sync();
                    const uint32_t new_end =
                        m_device_affected_vertices.GetCount(stream_s);
                    if (new_end < frontier_end || new_end > graph_datum.nnodes) {
                        LOG("[INVALIDATION] protocol_error=invalid_queue_size batch=%u previous=%u current=%u nnodes=%u\n",
                            NumOfSnapShots, frontier_end, new_end, graph_datum.nnodes);
                        std::abort();
                    }
                    frontier_begin = frontier_end;
                    frontier_end = new_end;
                    ++invalidation_rounds;
                }
                sw_invalidation.stop();
                if (frontier_end != 0) {
                    KernelSizing(grid_dims, block_dims, frontier_end);
                    kernel::reset_affected_values<<<grid_dims, block_dims, 0,
                        stream_s.cuda_stream>>>(
                            app_inst,
                            m_device_affected_vertices.GetDeviceDataPtr(),
                            frontier_end,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject());
                }
                const bool cc_union = UseCcUnionRepair();
                if (!cc_union) CollectDeletionAffectedVertices(NumOfSnapShots);
                Stopwatch sw_physical_delete(true);
                if (!certify_cc) del_edge_pr(local_begin, NumOfSnapShots);
                sw_physical_delete.stop();
                const double physical_delete_ms = early_delete_ms + sw_physical_delete.ms();
                Stopwatch sw_repair(true);
                if (cc_union) {
                    RunCcUnionRepair(NumOfSnapShots, frontier_end);
                } else if (AppImplDeviceObject::kSupportsGpuDeletionRepair) {
                    RunGpuAffectedRepair(NumOfSnapShots);
                }
                sw_repair.stop();
                sw_del.stop();
                const double attributed_ms =
                    sw_reset_seed.ms() + sw_invalidation.ms() +
                    physical_delete_ms + sw_repair.ms();
                LOG("[DELETE-ATTR][batch %u] repair_executor=%s affected=%u invalidation_rounds=%u reset_seed_ms=%.3f initial_rebuild_ms=0.000 invalidation_ms=%.3f pma_delete_ms=%.3f repair_wall_ms=%.3f residual_ms=%.3f total_ms=%.3f\n",
                    NumOfSnapShots,"gpu",
                    frontier_end,
                    invalidation_rounds,
                    sw_reset_seed.ms(),
                    sw_invalidation.ms(),
                    physical_delete_ms,
                    sw_repair.ms(),
                    sw_del.ms() - attributed_ms,
                    sw_del.ms());
                LOG("删除 time: %f ms (excluded)\n", sw_del.ms());

            }

            void GatherValue() {
                cgcomm::Scope comm_scope(cgcomm::Category::State);
                return m_graph_datum->GatherValue();
            }

            void GatherReset() {
                
                return m_graph_datum->GatherReset();
                

            }

            void GatherParent() {
                cgcomm::Scope comm_scope(cgcomm::Category::State);
                return m_graph_datum->GatherParent();
            }

            void GatherBuffer() {
                cgcomm::Scope comm_scope(cgcomm::Category::State);
                return m_graph_datum->GatherBuffer();
            }

            void GatherLevel(){
                return m_graph_datum->GatherLevel();
            }

            groute::graphs::dev::CSRGraph CSRDeviceObject() const {
                return m_csr_dev_graph_allocator->DeviceObject();
            }

            const groute::Stream &getStream() const {
                return *m_stream;
            }

            private:


            void LoadOptions() {
                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    if (FLAGS_lb_push.size() == 0) {
                        if (m_groute_context->host_pma_small.avg_degree() >= 0) { //all FINE_GRAINED
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_push == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::NONE);
                                } else if (FLAGS_lb_push == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_push == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_push == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown push load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    if (FLAGS_lb_pull.size() == 0) {
                        if (m_groute_context->host_pma_small.avg_degree() >= 5) {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_pull == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::NONE);
                                } else if (FLAGS_lb_pull == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_pull == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_pull == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown pull load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                                    if (FLAGS_alpha == 0) {
                                        fprintf(stderr, "Warning: alpha = 0, A general method AsyncPushDD is used\n");
                                        m_engine_options.ForceVariant(AlgoVariant::ASYNC_PUSH_DD);
                                    }
                                }


            void PreComputationBW() {// Reorganizing nothing, just reset the round and record the workload.
                const int dev_id = 0;
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                m_running_info.current_round++;
                graph_datum.m_current_round.set_val_H2DAsync(m_running_info.current_round, stream.cuda_stream);

                stream.Sync();
            }


            void ExecutePolicy_add(AlgoVariant *algo_variant) {
                
                auto &app_inst = *m_app_inst;
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                Stopwatch sw_execution(true);

                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];

                    stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        
                        zcflag = true;
                        RunSyncPushDDB_ADD(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
                sw_execution.stop();
               LOG("插入刷新缓存 time: %f ms (excluded)\n", sw_execution.ms());
            }

            void ExecutePolicy_Converge(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                uint64_t seg_sedge_csr;
                index_t seg_snode,seg_enode;

                m_groute_context->segment_ct = FLAGS_SEGMENT;

                uint64_t active_sources = 0;
                uint64_t active_edge_span_upper_bound = 0;
                uint32_t active_partitions = 0;
                uint32_t kernel_launches = 0;
                const bool exact_all_gpu = m_insertion_epoch_active;
                groute::Queue<index_t> &exact_input =
                    graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT];
                groute::Queue<index_t> &exact_output =
                    graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1];
                if (exact_all_gpu) {
                    exact_output.ResetAsync(m_stream->cuda_stream);
                } else {
                    m_device_insertion_changed_vertices.ResetAsync(
                        m_stream->cuda_stream);
                }
                m_stream->Sync();
                if (m_insertion_epoch_active && m_device_exact_source_counts == nullptr) {
                    GROUTE_CUDA_CHECK(cudaMalloc(
                        reinterpret_cast<void **>(&m_device_exact_source_counts),
                        5 * sizeof(unsigned long long)));
                }
                if (m_insertion_epoch_active) {
                    GROUTE_CUDA_CHECK(cudaMemset(m_device_exact_source_counts, 0,
                        5 * sizeof(unsigned long long)));
                }
                if (exact_all_gpu && m_exact_source_frontier_ready) {
                    active_sources = m_exact_source_frontier_count;
                    active_partitions = active_sources == 0 ? 0 : 1;
                    const auto &vcsr_graph =
                        m_vcsr_dev_graph_allocator->DeviceObject();
                    if (active_sources != 0) {

                        ExpandExactGpuSources<<<active_sources, 128, 0,
                            stream[0].cuda_stream>>>(
                            app_inst,
                            exact_input.GetDeviceDataPtr(),
                            active_sources, vcsr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetParentDeviceObject(),
                            exact_output.DeviceObject(),
                            m_device_exact_source_counts,
                            m_device_exact_source_counts + 1);
                        kernel_launches = 1;
                    }
                } else {
                    for (index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; ++seg_idx) {
                        if (graph_datum.seg_active_num[seg_idx] == 0) continue;
                        active_sources += graph_datum.seg_active_num[seg_idx];
                        active_edge_span_upper_bound +=
                            m_groute_context->seg_nedge_csr[seg_idx];
                        active_partitions++;
                    }
                }
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    if (exact_all_gpu && m_exact_source_frontier_ready) break;
                    if (graph_datum.seg_active_num[seg_idx] == 0) {
                        continue;
                    }

                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge

                    stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){

                        if (exact_all_gpu) {
                            ExpandExactGpuSources<<<graph_datum.seg_active_num[seg_idx],
                                128, 0, stream[stream_id].cuda_stream>>>(
                                app_inst,
                                graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                graph_datum.seg_active_num[seg_idx], vcsr_graph,
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.GetParentDeviceObject(),
                                m_device_insertion_changed_vertices.DeviceObject(),
                                m_device_exact_source_counts,
                                m_device_exact_source_counts + 1);
                        } else {
                            m_vcsr_dev_graph_allocator->SwitchZC();
                            zcflag = true;
                            RunSyncPushDDB_Delta(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                               vcsr_graph,
                               graph_datum,
                                   m_engine_options,
                                   stream[stream_id],
                                   m_device_insertion_changed_vertices.DeviceObject(),
                                   m_insertion_epoch_active,
                                   graph_datum.seg_active_num[seg_idx]);
                        }
                        kernel_launches++;
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               unsigned long long exact_counts[2] = {0, 0};
               if (m_insertion_epoch_active) {
                   GROUTE_CUDA_CHECK(cudaMemcpy(exact_counts,
                       m_device_exact_source_counts, sizeof(exact_counts),
                       cudaMemcpyDeviceToHost));
                   if (exact_counts[0] == 0 && exact_counts[1] != 0) {
                       LOG("[EXACT-SOURCE] protocol_error=edges_without_processed_source epoch=%u round=%llu sources=%llu edges=%llu\n",
                           m_insertion_epoch,
                           static_cast<unsigned long long>(m_insertion_convergence_round + 1),
                           exact_counts[0], exact_counts[1]);
                       std::abort();
                   }
                   LOG("[EXACT-SOURCE-EXACT] epoch=%u round=%llu offered_sources=%llu processed_sources=%llu logical_edges=%llu processed_edges=%llu partition_rebuilds=0 frontier_syncs=1\n",
                       m_insertion_epoch,
                       static_cast<unsigned long long>(m_insertion_convergence_round + 1),
                       static_cast<unsigned long long>(active_sources),
                       exact_counts[0], exact_counts[1], exact_counts[1]);
               }
               if (exact_all_gpu) {
                   AdvanceExactSourceFrontier(exact_input, exact_output);
               } else {
                   PostComputationBW();
               }
               sw_execution.stop();
               const uint64_t reported_round = m_insertion_epoch_active
                   ? m_insertion_convergence_round + 1
                   : m_running_info.current_round;
               if (FLAGS_verbose) {
                   uint64_t next_active_sources = 0;
                   uint32_t next_active_partitions = 0;
                   for (index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; ++seg_idx) {
                       next_active_sources += graph_datum.seg_active_num[seg_idx];
                       next_active_partitions +=
                           graph_datum.seg_active_num[seg_idx] == 0 ? 0 : 1;
                   }
                   LOG("[FRONTIER][insert] state_round=%llu active_sources=%llu active_partitions=%u active_edge_span_upper_bound=%llu kernel_launches=%u post_rebuild_partitions=%d next_active_sources=%llu next_active_partitions=%u round_wall_ms=%.3f\n",
                       static_cast<unsigned long long>(reported_round),
                       static_cast<unsigned long long>(active_sources),
                       active_partitions,
                       static_cast<unsigned long long>(active_edge_span_upper_bound),
                       kernel_launches,
                       m_insertion_epoch_active
                           ? m_last_insertion_dirty_partitions
                           : static_cast<uint32_t>(FLAGS_SEGMENT),
                       static_cast<unsigned long long>(next_active_sources),
                       next_active_partitions,
                       sw_execution.ms());
               }

            }

            void PostComputationBW() {
                
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();
                std::vector<uint8_t> dirty_segments(
                    FLAGS_SEGMENT, m_insertion_epoch_active ? 0 : 1);
                if (m_insertion_epoch_active) {
                    const uint32_t changed_count =
                        m_device_insertion_changed_vertices.GetCount(*m_stream);
                    m_insertion_changed_vertices.resize(changed_count);
                    if (changed_count != 0) {
                        GROUTE_CUDA_CHECK(cudaMemcpy(
                            m_insertion_changed_vertices.data(),
                            m_device_insertion_changed_vertices.GetDeviceDataPtr(),
                            sizeof(index_t) * changed_count,
                            cudaMemcpyDeviceToHost));
                        for (const index_t dst : m_insertion_changed_vertices) {
                            const index_t seg = FindSegmentForVertexHost(dst);
                            if (seg < FLAGS_SEGMENT) dirty_segments[seg] = 1;
                        }
                    }
                }
                m_last_insertion_dirty_partitions = static_cast<uint32_t>(
                    std::count(dirty_segments.begin(), dirty_segments.end(), uint8_t{1}));
                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;

                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    if (!dirty_segments[seg_idx]) {
                        graph_datum.seg_active_num[seg_idx] = 0;
                        m_running_info.input_active_count_seg[seg_idx] = 0;
                        continue;
                    }
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                    RebuildArrayWorklist(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx,
                        m_insertion_epoch_active ? m_device_node_state_epoch : nullptr,
                        m_insertion_epoch);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                if (m_insertion_epoch_active) {
                    for (index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; ++seg_idx) {
                        if (!dirty_segments[seg_idx]) continue;
                        const index_t sid = seg_idx % FLAGS_n_stream;
                        const index_t active =
                            graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[sid]);
                        graph_datum.seg_active_num[seg_idx] = active;
                        m_running_info.input_active_count_seg[seg_idx] = active;
                    }
                } else {
                    RefreshSegmentActiveCountsFromQueues();
                }
                
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();
          }

            void AdvanceExactSourceFrontier(groute::Queue<index_t> &input,
                                            groute::Queue<index_t> &output) {
                GraphDatum &graph_datum = *m_graph_datum;
                const uint32_t changed_count = output.GetCount(*m_stream);
                input.Swap(output);
                m_exact_source_frontier_ready = true;
                m_exact_source_frontier_count = changed_count;
                m_last_insertion_dirty_partitions = 0;
                for (index_t seg = 0; seg < FLAGS_SEGMENT; ++seg) {
                    graph_datum.seg_active_num[seg] = 0;
                    m_running_info.input_active_count_seg[seg] = 0;
                }
                if (changed_count != 0) {
                    graph_datum.seg_active_num[0] = changed_count;
                    m_running_info.input_active_count_seg[0] = changed_count;
                }
                LOG("[EXACT-SOURCE-FRONTIER] epoch=%u round=%llu changed_events=%u bitmap_scanned_vertices=0 rebuilt_partitions=0\n",
                    m_insertion_epoch,
                    static_cast<unsigned long long>(m_insertion_convergence_round + 1),
                    changed_count);
            }

            index_t FindSegmentForVertexHost(index_t vertex) const {
                index_t lo = 0;
                index_t hi = FLAGS_SEGMENT;
                while (lo < hi) {
                    const index_t mid = lo + (hi - lo) / 2;
                    if (vertex < m_groute_context->seg_enode[mid]) {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                return lo;
            }


            void count() {
                
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                index_t seg_snode,seg_enode;
                index_t stream_id;
                uint64_t act = 0;
                
                
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);
                    graph_datum.seg_active_num[seg_idx] = active_count;

                    m_running_info.input_active_count_seg[seg_idx] = active_count;

                    uint32_t work_size = active_count;
                    dim3 grid_dims, block_dims;
                    

                }

          }

           void CombineTask(AlgoVariant *algo_variant) {
                
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].ResetAsync(stream_seg.cuda_stream);
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1].ResetAsync(stream_seg.cuda_stream);
                stream_seg.Sync();
                Stopwatch sw_unique(true);
                index_t seg_snode,seg_enode;
                index_t stream_id;
                index_t task = 1;// zero:0 exp_filter:1 exp_compaction:2
                bool zc = false;
                bool compaction = false;
                Stopwatch sw_rebuild(true);
                index_t seg_idx_ct = 0;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        task = 0;
                        zc = true;
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Zero_Copy && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    if(algo_variant[seg_idx] == AlgoVariant::Exp_Compaction){
                        task = 2;
                        compaction = true;
                        
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Exp_Compaction && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                    if(task == 0){
                        task = 1;
                        
                        RebuildArrayWorklist_zero(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT);
                    }
                    else if(task == 1)
                    {
                        algo_variant[seg_idx_ct] = AlgoVariant::Exp_Filter;
                        m_groute_context->segment_id_ct[seg_idx_ct++] = seg_idx;
                        
                        RebuildArrayWorklist(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                    }
                    else if(task == 2){
                        task = 1;
                        
                        RebuildArrayWorklist_compaction(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT + 1);
                    }
                }
                if(zc){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Zero_Copy;
                }
                if(compaction){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Exp_Compaction;
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                m_groute_context->segment_ct = seg_idx_ct;
                
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < seg_idx_ct ; seg_idx++){
                    uint32_t seg_idx_new = m_groute_context->segment_id_ct[seg_idx];
                    
                    stream_id = seg_idx % FLAGS_n_stream;
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx_new].GetCount(stream[stream_id]);
                    
                    uint32_t work_size = active_count;
                    dim3 grid_dims, block_dims;

                    if(FLAGS_priority_a == 1){
                        
                            Stopwatch sw_priority(true);
                            if(algo_variant[seg_idx_new] == AlgoVariant::Zero_Copy){
                                graph_datum.seg_res_num[seg_idx_new] = 1;
                                continue;
                            }
                            if(algo_variant[seg_idx_new] == AlgoVariant::Exp_Compaction){
                                graph_datum.seg_res_num[seg_idx_new] = 0;
                                continue;
                            }
                            graph_datum.m_seg_value.set_val_H2DAsync(0, stream[stream_id].cuda_stream);
                            KernelSizing(grid_dims, block_dims, work_size);
                            kernel::SumResQueue << < grid_dims, block_dims, 0, stream[stream_id].cuda_stream >> >
                                (app_inst,
                                    groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in_seg[seg_idx_new].GetDeviceDataPtr(),
                                    work_size),
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.m_seg_value.dev_ptr);

                            stream[stream_id].Sync();
                            graph_datum.seg_res_num[seg_idx_new] = graph_datum.m_seg_value.get_val_D2H();
                            sw_priority.stop();
                            m_running_info.time_overhead_sample += sw_priority.ms();
                    }

                }
                graph_datum.Compaction_num = 0;
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

          void Compaction() {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                
                auto vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                thrust::device_ptr<uint32_t> ptr_labeling(graph_datum.activeNodesLabeling.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_labeling_prefixsum(graph_datum.prefixLabeling.dev_ptr);

                graph_datum.subgraphnodes = thrust::reduce(ptr_labeling, ptr_labeling + graph_datum.nnodes);

                thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph_datum.nnodes, ptr_labeling_prefixsum);

                kernel::makeQueue<<<graph_datum.nnodes/512+1, 512>>>(vcsr_graph.subgraph_activenode, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.nnodes);

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_activenode, vcsr_graph.subgraph_activenode, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                thrust::device_ptr<uint32_t> ptr_degrees(graph_datum.activeNodesDegree.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_degrees_prefixsum(graph_datum.prefixSumDegrees.dev_ptr);

                thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph_datum.nnodes, ptr_degrees_prefixsum);

                kernel::makeActiveNodesPointer<<<graph_datum.nnodes/512+1, 512>>>(vcsr_graph.subgraph_rowstart, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.prefixSumDegrees.dev_ptr, graph_datum.nnodes);

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_rowstart, vcsr_graph.subgraph_rowstart, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numActiveEdges = 0;
                uint32_t endid = vcsr_graph_host.subgraph_activenode[graph_datum.subgraphnodes-1];
                
                uint64_t outDegree = vcsr_graph_host.sync_vertices_[endid].index;
                if(graph_datum.subgraphnodes > 0)
                    numActiveEdges = vcsr_graph_host.subgraph_rowstart[graph_datum.subgraphnodes-1] + outDegree;

                graph_datum.subgraphedges = numActiveEdges;
                uint32_t last = numActiveEdges;

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph.subgraph_rowstart + graph_datum.subgraphnodes, &last, sizeof(uint32_t), cudaMemcpyHostToDevice));

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_rowstart, vcsr_graph.subgraph_rowstart, (graph_datum.subgraphnodes + 1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numThreads = 32;

                if(graph_datum.subgraphnodes < 5000)
                    numThreads = 1;
                std::thread runThreads[numThreads];
          }

      };


  }
}

#endif //HYBRID_FRAMEWORK_H
