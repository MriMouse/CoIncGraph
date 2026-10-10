// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_API_H
#define HYBRID_API_H

#include <cuda.h>
#include <cub/grid/grid_barrier.cuh>
#include <groute/graphs/csr_graph.cuh>

#include <groute/device/queue.cuh>
#include <utils/cuda_utils.h>
#include <framework/graph_datum.cuh>

namespace sepgraph
{
    namespace api
    {
        template<typename TValue, typename TBuffer, typename TWeight>
        struct AppBase
        {
            // Compile-time propagation policy; preserve the historical default.
            __host__ __device__ static TWeight TraversalEdgeWeight(index_t src, index_t dst) {
                return static_cast<TWeight>((src + dst) % 128 + 1);
            }

            static constexpr bool kSupportsGpuDeletionRepair = false;
            static constexpr bool kComponentLabels = false;
            // Multi-source minimum-ID propagation seeds each vertex with its ID.
            static constexpr bool kVertexSeeds = false;
            static constexpr bool kRootedDeletionWitness = false;
            static constexpr bool kSignedResidual = false;
            static bool NormalizeUndirectedInput() { return false; }

            __host__ __device__ static TWeight DeletionEdgeWeight(index_t, index_t) {
                return TWeight(1);
            }

            
            __host__ __device__ static bool IsDeletionDependency(
                    index_t src, index_t dst, TValue source, TValue destination,
                    uint64_t candidate, index_t parent, bool parent_tight) {
                return parent == src ||
                    (source != static_cast<TValue>(UINT32_MAX) &&
                     candidate == static_cast<uint64_t>(destination) && !parent_tight);
            }

            // [ACCUMULATE] [CONTINUE/BREAK]
            const int ACCUMULATE_SUCCESS_BREAK = 0;         // 0 0
            const int ACCUMULATE_SUCCESS_CONTINUE = 1;      // 0 1
            const int ACCUMULATE_FAILURE_BREAK = 2;         // 1 0
            const int ACCUMULATE_FAILURE_CONTINUE = 3;      // 1 1

            groute::graphs::dev::CSRGraph m_csr_graph;
            groute::graphs::dev::CSCGraph m_csc_graph;
            groute::graphs::dev::PMAGraph m_vcsr_graph;
            uint32_t m_nnodes;
            uint32_t m_nedges;
            common::Model m_model;
            common::MsgPassing m_msg_passing;
            common::Scheduling m_scheduling;
            uint32_t *m_p_current_round;

            __forceinline__ __device__
            virtual TValue GetInitValue(index_t node) const = 0;

            __forceinline__ __device__
            virtual TBuffer GetInitBuffer(index_t node) const = 0;

            __forceinline__ __host__ __device__
            virtual TBuffer GetIdentityElement() const = 0;

            /**
             * Combine the value of node and the buffer, and return new buffer to push or be pulled by neighbor
             * @param node
             * @param p_value
             * @param p_buffer
             * @return
             */
            __forceinline__ __device__
            virtual utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                                  TValue *p_value,
                                                                  TBuffer *p_buffer) = 0;
            __forceinline__ __device__
            virtual utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node,int *type_device,
                                                                  TValue *p_value,
                                                                  TBuffer *p_buffer) = 0;

            __forceinline__ __device__
            virtual int AccumulateBuffer(index_t src,
                                         index_t dst,
                                        //  TValue level,
                                        
                                         TWeight weight,
                                         TValue *p_parent,
                                         TBuffer *p_buffer,
                                         TBuffer buffer)
            {
                assert(false);
                return 0;
            }

            __forceinline__ __device__
            virtual int AccumulateBuffer_add(index_t src,
                                         index_t dst,
                                        // TValue level,
                                        
                                         TWeight weight,
                                         TValue *p_parent,
                                         TBuffer *p_buffer,
                                         TBuffer *buffer)
            {
                assert(false);
                return 0;
            }

            /**
             * Deterimine a node is active or not.
             * This method will be called for Topology-driven variants.
             * @param node node id
             * @param buffer the buffer of node
             * @return return true if node is active.
             */
            __forceinline__ __device__
            virtual bool IsActiveNode(index_t node, TBuffer buffer, TValue value) const
            {
                assert(false);
                return true;
            }
            __forceinline__ __device__
            virtual bool IsActiveNode_INC(index_t node, TBuffer buffer, TValue value) const
            {
                assert(false);
                return true;
            }

             __forceinline__ __device__
            virtual TValue sum_value(index_t node, TValue value, TBuffer buffer) const
            {
                assert(false);
                return 0;
            }

            __forceinline__ __device__
            virtual void PostComputation()
            {
                
            }

            __forceinline__ __device__
            virtual bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const
            {
                return true;
            }

            __host__
            void SetVariant(common::Model model, common::MsgPassing msg_passing, common::Scheduling scheduling)
            {
                m_model = model;
                m_msg_passing = msg_passing;
                m_scheduling = scheduling;
            }

            __host__
            void SetVariant(common::AlgoVariant algo_variant)
            {
                m_model = algo_variant.m_model;
                m_msg_passing = algo_variant.m_msg_passing;
                m_scheduling = algo_variant.m_scheduling;
            }
        };
    }
}

#endif //HYBRID_API_H
