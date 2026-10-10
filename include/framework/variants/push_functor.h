// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_PUSH_FUNCTOR_H
#define HYBRID_PUSH_FUNCTOR_H

#include <groute/device/queue.cuh>
#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>

namespace sepgraph
{
    namespace kernel
    {
        template <typename TBuffer>
        struct Payload
        {
            index_t m_src;
            TBuffer m_buffer_to_push;
            //https://stackoverflow.com/questions/33978185/default-constructor-cannot-be-referenced-in-visual-studio-2015
            //X is a union-like class that has a variant member with a non-trivial default constructor,
            //            __device__
            //            Payload()
            
            //
            
            //
            //            __device__ __forceinline__
            //            Payload(index_t src,
            //                    TBuffer buffer_to_push) : m_src(src),
            //                                              m_buffer_to_push(buffer_to_push)
            
            //
            
        };

        template <typename T>
        __device__ __forceinline__ T AtomicMin32(T *address, T value) {
            static_assert(sizeof(T) == sizeof(uint32_t), "32-bit state required");
            uint32_t *bits = reinterpret_cast<uint32_t *>(address);
            uint32_t observed = *bits;
            while (value < *reinterpret_cast<T *>(&observed)) {
                const uint32_t desired = *reinterpret_cast<uint32_t *>(&value);
                const uint32_t previous = atomicCAS(bits, observed, desired);
                if (previous == observed) break;
                observed = previous;
            }
            return *reinterpret_cast<T *>(&observed);
        }

        template <typename TValue>
        __device__ __forceinline__ bool AtomicInvalidateParent(
                TValue *parent,
                index_t expected_parent) {
            static_assert(sizeof(TValue) == sizeof(unsigned int),
                          "deletion invalidation requires 32-bit parent state");
            const TValue expected_value = static_cast<TValue>(expected_parent);
            const TValue invalid_value = static_cast<TValue>(UINT32_MAX);
            const unsigned int expected_bits =
                *reinterpret_cast<const unsigned int *>(&expected_value);
            const unsigned int invalid_bits =
                *reinterpret_cast<const unsigned int *>(&invalid_value);
            return atomicCAS(reinterpret_cast<unsigned int *>(parent),
                             expected_bits,
                             invalid_bits) == expected_bits;
        }

        template <typename TAppInst,
                  typename PMAGraph,
                  template <typename> class GraphDatum,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctor
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            GraphDatum<TBuffer> m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;

            __device__
            PushFunctor()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        PMAGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(nullptr, nullptr, 0),
                                                            m_work_target_high(nullptr, nullptr, 0),
                                                            m_current_priority(0),
                                                            m_vcsr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(false),
                                                            m_priority(false)
            {
                m_weighted = m_weight_array.size > 0;
            }

            /**
             * Async+Push+DD+[Priority]
             * @param app_inst
             * @param work_target_low
             * @param work_target_high
             * @param current_priority
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        TWorkTarget work_target_low,
                        TWorkTarget work_target_high,
                        TBuffer current_priority,
                        PMAGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(work_target_low),
                                                            m_work_target_high(work_target_high),
                                                            m_current_priority(current_priority),
                                                            m_vcsr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(true)
            {
                m_weighted = m_weight_array.size > 0;
                m_priority = work_target_low != work_target_high;
            }

            __device__ __forceinline__ bool operator()(index_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                int status;
                bool accumulate_success;
                bool continue_push;

                if (m_weighted)
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }
                else
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }

                continue_push = (status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE ||
                                 status == m_app_inst.ACCUMULATE_FAILURE_CONTINUE);
                accumulate_success = (status == m_app_inst.ACCUMULATE_SUCCESS_BREAK ||
                                      status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE);

                if (m_data_driven && accumulate_success)
                {
                    if (m_priority)
                    {
                        if (m_app_inst.IsHighPriority(m_current_priority, m_buffer_array[dst]))
                            m_work_target_high.append(dst);
                        else
                            m_work_target_low.append(dst);
                    }
                    else
                    {
                        m_work_target_high.append(dst);
                    }
                }

                return continue_push;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBAmend_cache
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            uint64_t *m_cache_size;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBAmend_cache()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBAmend_cache(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            uint64_t *cache_size,
            TValue *parent_array,
            TBuffer *buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                            m_work_target_low(nullptr, nullptr, 0),
                                            m_work_target_high(nullptr, nullptr, 0),
                                            m_current_priority(0),
                                            m_vcsr_graph(vcsr_graph),
                                            m_cache_g(buffer),
                                            m_cache_size(cache_size),
                                            m_parent_array(parent_array),
                                            m_buffer_array(buffer_array),
                                            m_data_driven(true),
                                            m_priority(false),
                                            m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);
                
                    uint64_t offset = edge - m_vcsr_graph.begin_edge(meta_data.m_src);

                    index_t here = m_vcsr_graph.vertices_[meta_data.m_src].virtual_start + offset;
                    atomicExch((m_cache_g+here),dst);
                    
                    atomicAdd(&m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree,1);

                    //     printf("src %d dst %d offset %d vir_start %d here %d dev_dst %d\n",
                    

                return true;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDB
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            uint64_t *m_cache_size;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;
            groute::dev::Queue<index_t> m_changed_vertices;

            bool m_record_changed;

            __device__
            PushFunctorDB()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            uint64_t *cache_size,
            TValue *parent_array,
            TBuffer *buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                            m_work_target_low(nullptr, nullptr, 0),
                                            m_work_target_high(nullptr, nullptr, 0),
                                            m_current_priority(0),
                                            m_vcsr_graph(vcsr_graph),
                                            m_cache_g(buffer),
                                            m_cache_size(cache_size),
                                            m_parent_array(parent_array),
                                            m_buffer_array(buffer_array),
                                            m_data_driven(true),
                                            m_priority(false),
                                            m_out_active_high(out_active),
                                            m_changed_vertices(nullptr, nullptr, 0),

                                            m_record_changed(false)
            {
                m_weighted = true;
            }

            __device__ __forceinline__
            PushFunctorDB(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            uint64_t *cache_size,
            TValue *parent_array,
            TBuffer *buffer_array,
            BitmapDeviceObject out_active,
            groute::dev::Queue<index_t> changed_vertices,
            bool record_changed) : m_app_inst(app_inst),
                                            m_work_target_low(nullptr, nullptr, 0),
                                            m_work_target_high(nullptr, nullptr, 0),
                                            m_current_priority(0),
                                            m_vcsr_graph(vcsr_graph),
                                            m_cache_g(buffer),
                                            m_cache_size(cache_size),
                                            m_parent_array(parent_array),
                                            m_buffer_array(buffer_array),
                                            m_data_driven(true),
                                            m_priority(false),
                                            m_out_active_high(out_active),
                                            m_changed_vertices(changed_vertices),

                                            m_record_changed(record_changed)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {

                index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);

                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if ((dst!=-1))
                {

                        const int changed = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                        if (m_record_changed && changed) {
                            m_changed_vertices.append(dst);
                        }
                }

                return true;
            }
        };

        template <typename TAppInst,
                  typename PMAGraph,
                  template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctorDB_COM
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB_COM()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB_COM(TAppInst app_inst,
                          PMAGraph csr_graph,
                          TValue *parent_array,
                          TBuffer *buffer_array,
                          GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(csr_graph),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = m_weight_array.size > 0;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                
                index_t dst =  m_vcsr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;

                    m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                          &m_parent_array[dst],
                                                         &m_buffer_array[dst],
                                                         buffer_to_push);
                
                // else

                //                                          dst,
                //                                          m_buffer_array[dst],
                

                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                //   template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer,
                  typename TVec>
        struct PushFunctorAdd
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            uint64_t *m_cache_size;
            TBuffer *m_buffer_array;
            TValue *m_value_array;
            TVec m_cache_g;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorAdd()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorAdd(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
                        TValue *parent_array,
                        TBuffer *buffe_array,
                        TVec cache_l2,
                        uint64_t *cache_size,
                        BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffe_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_cache_g(cache_l2),
                                                           m_cache_size(cache_size),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                
                uint64_t offset = edge - m_vcsr_graph.begin_edge(meta_data.m_src) -m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree;

                if((m_vcsr_graph.vertices_[meta_data.m_src].secondary_start + offset) < ((*m_cache_size)/2)){
                    uint64_t here = m_vcsr_graph.vertices_[meta_data.m_src].secondary_start + offset;
                    m_cache_g[here] = dst;
                    atomicAdd(&(m_vcsr_graph.vertices_[meta_data.m_src].secondary_degree),1);
                    // LOG("cache L-2 location: %lu edges %d %lu\n",
                    // here,
                    // meta_data.m_src,
                    
                }
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBADD
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue* m_parent_datum;
            TBuffer* m_buffer_datum;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBADD()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBADD(TAppInst app_inst,
                            TPMAGraph vcsr_graph,
                            Buffer buffer,
                            TValue* parent_array,
                            TBuffer* buffer_array,
                            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_datum(parent_array),
                                                           m_buffer_datum(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                
                if (dst!=UINT32_MAX)
                {
                    //here need correct
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_datum[dst],
                                                        &m_buffer_datum[dst],
                                                        buffer_to_push);
                         //insert edges to buffer
                }

                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TBuffer>
        struct PushFunctorDBCacheFlush
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            Buffer *m_cache_g_l3;
            uint64_t *m_cache_size;
            int m_cache_level;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCacheFlush()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCacheFlush(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            Buffer *buffer_l3,
            uint64_t *cache_size,
            int &cache_level,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_cache_g_l3(buffer_l3),
                                                            m_cache_size(cache_size),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_cache_level(cache_level),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                uint64_t offset;
                offset = edge - m_vcsr_graph.vertices_[meta_data.m_src].virtual_start;
                uint64_t location = offset + m_vcsr_graph.vertices_[meta_data.m_src].third_start;
                atomicExch((m_cache_g_l3+location),dst);
                atomicAdd(&m_vcsr_graph.vertices_[meta_data.m_src].third_degree,1);
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                //   template <typename> class GraphDatum,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2_ADD
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue* m_parent_datum;
            TBuffer* m_buffer_datum;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCachel2_ADD()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2_ADD(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
            TValue* parent_array,
            TBuffer* buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_datum(parent_array),
                                                           m_buffer_datum(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);
                if (dst!=UINT32_MAX)
                {
                        
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        // m_level_array[meta_data.m_src],
                                                        // &m_level_array[dst],
                                                        weight,
                                                        
                                                        &m_parent_datum[dst],
                                                        &m_buffer_datum[dst],
                                                        buffer_to_push);
                         //insert edges to buffer
                }

                return true;
            }

        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename NodesHash,
                //   template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCache
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            NodesHash m_nodes_hash;
            
            TValue *m_parent_array;
            
            TBuffer *m_buffer_array;
            
            
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCache()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCache(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            
            Buffer buffer,
            NodesHash nodes_hash,
                        //   groute::graphs::dev::PMAGraph vcsr_graph,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                        
                        //   GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                        //    m_delta_array(delta_array),
                                                           m_cache_g(buffer),
                                                           m_nodes_hash(nodes_hash),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                        //    m_level_array(level_array),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {

                
                index_t dst = (uint32_t)m_cache_g[edge];
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;

                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        // m_level_array[meta_data.m_src],
                                                        // &m_level_array[dst],
                                                        weight,
                                                        
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                //          //insert edges to buffer
                
                // else
                
                
                //                                             dst,
                //                                             // m_buffer_array.get_item_ptr(dst),
                //                                             &m_parent_array[dst],
                //                                             &m_buffer_array[dst],

                return true;
            }

        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;
            groute::dev::Queue<index_t> m_changed_vertices;

            bool m_record_changed;

            __device__
            PushFunctorDBCachel2()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            
            Buffer *buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active),
                                                           m_changed_vertices(nullptr, nullptr, 0),

                                                           m_record_changed(false)
            {
                
            }

            __device__ __forceinline__
            PushFunctorDBCachel2(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          BitmapDeviceObject out_active,
                          groute::dev::Queue<index_t> changed_vertices,
                          bool record_changed) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active),
                                                           m_changed_vertices(changed_vertices),

                                                           m_record_changed(record_changed)
            {
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {

                index_t dst = m_cache_g[edge];
                
                index_t weight = m_app_inst.TraversalEdgeWeight(meta_data.m_src, dst);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if (dst!=-1)
                {

                        
                        const int changed = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                        if (m_record_changed && changed) {
                            m_changed_vertices.append(dst);
                        }
                         //insert edges to buffer
                }
                // else
                
                
                //                                             dst,
                //                                             // m_buffer_array.get_item_ptr(dst),
                //                                             &m_parent_array[dst],
                //                                             &m_buffer_array[dst],

                return true;
            }

        };

        template <typename TAppInst,
                    typename PMAGraph,
                    typename TVec,
                    typename TValue,
                    typename TBuffer>
        struct PushFunctorDel_l1
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            
            TBuffer *m_buffer_array;
            
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            TVec m_cache_g;
            

            __device__
            PushFunctorDel_l1()
            {
            }
            __device__ __forceinline__
            PushFunctorDel_l1(TAppInst app_inst,
            const PMAGraph vcsr_graph,
                        TVec cache_l1,
                        TValue *parent_array,
                        
                          TBuffer *buffer_array) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(cache_l1),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                index_t dst_to_del = meta_data.m_buffer_to_push;
                if(dst == dst_to_del) m_cache_g[edge] = UINT32_MAX;
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDB_Del
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue *m_parent_array;
            TValue *m_value_array;
            TBuffer *m_buffer_array;
            bool *m_reset_nodes;
            TWorkTarget m_affected_vertices;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB_Del()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB_Del(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          TValue *value_array,
                          bool *reset_nodes,
                          TWorkTarget affected_vertices,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_value_array(value_array),
                                                           m_reset_nodes(reset_nodes),
                                                           m_affected_vertices(affected_vertices),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);

                if (dst!=UINT32_MAX)
                {
                        const TValue src_value = m_value_array[meta_data.m_src];
                        const uint64_t candidate =
                            static_cast<uint64_t>(src_value) +
                            static_cast<uint64_t>(m_app_inst.DeletionEdgeWeight(
                                meta_data.m_src, dst));
                        const index_t parent = m_parent_array[dst];
                        const bool parent_tight = parent != UINT32_MAX &&
                            m_value_array[parent] != static_cast<TValue>(UINT32_MAX) &&
                            static_cast<uint64_t>(m_value_array[parent]) +
                                static_cast<uint64_t>(m_app_inst.DeletionEdgeWeight(
                                    parent, dst)) ==
                                static_cast<uint64_t>(m_value_array[dst]);
                        const bool invalid_dependency = m_app_inst.IsDeletionDependency(
                            meta_data.m_src, dst, src_value, m_value_array[dst],
                            candidate, parent, parent_tight);
                        if (invalid_dependency &&
                            atomicExch(reinterpret_cast<unsigned int *>(
                                           &m_parent_array[dst]),
                                       UINT32_MAX) != UINT32_MAX) {
                            m_reset_nodes[dst] = true;
                            m_affected_vertices.append(dst);
                        }
                }

                return true;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2_DEL
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue *m_parent_array;
            TValue *m_value_array;
            TBuffer *m_buffer_array;
            bool *m_reset_nodes;
            TWorkTarget m_affected_vertices;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCachel2_DEL()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2_DEL(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          TValue *value_array,
                          bool *reset_nodes,
                          TWorkTarget affected_vertices,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                        //    m_delta_array(delta_array),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_value_array(value_array),
                                                           m_reset_nodes(reset_nodes),
                                                           m_affected_vertices(affected_vertices),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                if (dst!=UINT32_MAX)
                {
                        const TValue src_value = m_value_array[meta_data.m_src];
                        const uint64_t candidate =
                            static_cast<uint64_t>(src_value) +
                            static_cast<uint64_t>(m_app_inst.DeletionEdgeWeight(
                                meta_data.m_src, dst));
                        const index_t parent = m_parent_array[dst];
                        const bool parent_tight = parent != UINT32_MAX &&
                            m_value_array[parent] != static_cast<TValue>(UINT32_MAX) &&
                            static_cast<uint64_t>(m_value_array[parent]) +
                                static_cast<uint64_t>(m_app_inst.DeletionEdgeWeight(
                                    parent, dst)) ==
                                static_cast<uint64_t>(m_value_array[dst]);
                        const bool invalid_dependency = m_app_inst.IsDeletionDependency(
                            meta_data.m_src, dst, src_value, m_value_array[dst],
                            candidate, parent, parent_tight);
                        if (invalid_dependency &&
                            atomicExch(reinterpret_cast<unsigned int *>(
                                           &m_parent_array[dst]),
                                       UINT32_MAX) != UINT32_MAX) {
                            m_reset_nodes[dst] = true;
                            m_affected_vertices.append(dst);
                        }
                }

                return true;
            }

        };
    } // namespace kernel
} // namespace sepgraph
#endif //HYBRID_PUSH_FUNCTOR_H
