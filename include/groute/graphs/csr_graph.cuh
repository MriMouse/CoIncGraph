// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF

// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __GROUTE_GRAPHS_CSR_GRAPH_H
#define __GROUTE_GRAPHS_CSR_GRAPH_H

#include <vector>
#include <utils/communication_meter.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <groute/context.h>
#include <groute/graphs/common.h>
#include <groute/graphs/topology_contract.cuh>
#include <groute/graphs/source_local_chunk_store.h>
#include "./util.h"
#include <cstring>
#include<iostream>

#define DEBUG 0
DECLARE_int32(hybrid);
DECLARE_int32(n_stream);
namespace groute
{
    namespace graphs
    {
        typedef struct TNoWeight
        {
            __device__ __forceinline__
            TNoWeight(int i)
            {
            } //implicit conversion
        } NoWeight;

        struct GraphBase
        {
            index_t nnodes;
            uint64_t nedges;
            index_t *edge_weights;
            index_t *node_weights;

            GraphBase(index_t nnodes, uint64_t nedges, index_t *edge_weights, index_t *node_weights) : nnodes(nnodes), nedges(nedges), edge_weights(edge_weights), node_weights(node_weights)
            {
            }

            GraphBase() : nnodes(0), nedges(0), edge_weights(nullptr), node_weights(nullptr)
            {
            }
        };

        struct CSRGraphBase : GraphBase
        {
            uint64_t *row_start;
            index_t *edge_dst;

            CSRGraphBase(index_t nnodes, uint64_t nedges, index_t *edge_weights, index_t *node_weights) : GraphBase(nnodes, nedges, edge_weights, node_weights), row_start(nullptr), edge_dst(nullptr)
            {
            }

            CSRGraphBase() : GraphBase(0, 0, nullptr, nullptr), row_start(nullptr), edge_dst(nullptr)
            {
            }
        };

        struct CSCGraphBase : GraphBase
        {
            index_t *col_start;
            index_t *edge_source;
            index_t *out_dgr;

            CSCGraphBase(index_t nnodes, uint64_t nedges, index_t *edge_weights, index_t *node_weights) : GraphBase(nnodes, nedges, edge_weights, node_weights),
                                                                                                         col_start(nullptr), edge_source(nullptr), out_dgr(nullptr)
            {
            }

            CSCGraphBase() : GraphBase(0, 0, nullptr, nullptr),
                             col_start(nullptr), edge_source(nullptr), out_dgr(nullptr)
            {
            }
        };

        namespace host
        {
            /*
            * @brief A host graph object
            */
            struct CSRGraph : public CSRGraphBase
            {
                std::vector<uint64_t> row_start_vec; // the vectors are not always in use (see Bind)
                std::vector<index_t> edge_dst_vec;
                std::vector<index_t> edge_weights_vec;
                uint32_t *subgraph_activenode;
                uint32_t *subgraph_rowstart;
                uint32_t *subgraph_edgedst;
                uint32_t *subgraph_edgeweight;

                CSRGraph(index_t nnodes, uint64_t nedges) : CSRGraphBase(nnodes, nedges, nullptr, nullptr),
                                                           row_start_vec(nnodes + 1), edge_dst_vec(nedges)
                {
                    row_start = &row_start_vec[0];
                    edge_dst = &edge_dst_vec[0];
                }

                CSRGraph()
                {
                }

                ~CSRGraph()
                {
                }

                void Subgraph_Generate(){

                }
                void Move(index_t nnodes, uint64_t nedges,
                          std::vector<uint64_t> &row_start, std::vector<index_t> &edge_dst)
                {
                    this->nnodes = nnodes;
                    this->nedges = nedges;

                    this->row_start_vec = std::move(row_start);
                    this->edge_dst_vec = std::move(edge_dst);

                    this->row_start = &this->row_start_vec[0];
                    this->edge_dst = &this->edge_dst_vec[0];
                }

                void MoveWeights(std::vector<index_t> &edge_weights)
                {
                    this->edge_weights_vec = std::move(edge_weights);
                    this->edge_weights = &this->edge_weights_vec[0];
                }

                void AllocWeights()
                {
                    this->edge_weights_vec.resize(nedges);
                    this->edge_weights = &this->edge_weights_vec[0];
                }

                void Bind(index_t nnodes, uint64_t nedges,
                          uint64_t *row_start, index_t *edge_dst,
                          index_t *edge_weights, index_t *node_weights)
                {
                    this->nnodes = nnodes;
                    this->nedges = nedges;

                    this->row_start_vec.clear();
                    this->edge_dst_vec.clear();

                    this->row_start = row_start;
                    this->edge_dst = edge_dst;

                    this->edge_weights = edge_weights;
                    this->node_weights = node_weights;

                    this->subgraph_activenode = (uint32_t *) calloc((nnodes), sizeof(uint32_t));
                    this->subgraph_rowstart = (uint32_t *) calloc((nnodes + 1), sizeof(uint32_t));
                    this->subgraph_edgedst = (uint32_t *) calloc((nedges / 4), sizeof(uint32_t));
                    this->subgraph_edgeweight = (uint32_t *) calloc((nedges / 4), sizeof(uint32_t));
                }


                index_t max_degree() const
                {
                    index_t max_degree = 0;
                    for (index_t node = 0; node < nnodes; node++)
                    {
                        max_degree = std::max(max_degree, uint32_t (end_edge(node) - begin_edge(node)));
                    }
                    return max_degree;
                }

                index_t avg_degree() const
                {
                    return nedges / nnodes;
                }

                uint64_t begin_edge(index_t node) const
                {
                    return row_start[node];
                }

                uint64_t end_edge(index_t node) const
                {
                    return row_start[node + 1];
                }

                index_t edge_dest(index_t edge) const
                {
                    return edge_dst[edge];
                }

                void PrintDegree(std::vector<index_t> node_degree)
                {
                    index_t log_counts[32];

                    for (int i = 0; i < 32; i++)
                    {
                        log_counts[i] = 0;
                    }

                    int max_log_length = -1;

                    for (index_t node = 0; node < node_degree.size(); node++)
                    {
                        index_t degree = node_degree[node];

                        int log_length = -1;
                        while (degree > 0)
                        {
                            degree >>= 1;
                            log_length++;
                        }

                        max_log_length = std::max(max_log_length, log_length);

                        if (max_log_length >= 32)
                        {
                            printf("tooooooo skew.... degree:%u\n", degree);
                            exit(1);
                        }

                        log_counts[log_length + 1]++;
                    }

                    printf("    Degree   0: %lld (%.2f%%)\n",
                           (long long)log_counts[0],
                           (float)log_counts[0] * 100.0 / node_degree.size());
                    for (int i = 0; i <= max_log_length; i++)
                    {
                        printf("    Degree 2^%i: %lld (%.2f%%)\n",
                               i, (long long)log_counts[i + 1],
                               (float)log_counts[i + 1] * 100.0 / node_degree.size());
                    }
                    printf("\n");
                }

                void PrintHistogram(uint32_t *&p_in_degree,
                                    uint32_t *&p_out_degree)
                {
                    uint32_t *in_degree = new uint32_t[nnodes];
                    uint32_t *out_degree = new uint32_t[nnodes];

                    for (index_t node = 0; node < nnodes; node++)
                    {
                        index_t begin_edge = this->begin_edge(node),
                                end_edge = this->end_edge(node);

                        out_degree[node] = end_edge - begin_edge;

                        
                        for (index_t edge = begin_edge; edge < out_degree[node]; edge++)
                        {
                            index_t dest = this->edge_dest(edge);
                            in_degree[dest]++;
                        }
                    }

                    GROUTE_CUDA_CHECK(cudaMalloc(&p_in_degree, nnodes * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_in_degree, in_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

                    GROUTE_CUDA_CHECK(cudaMalloc(&p_out_degree, nnodes * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_out_degree, out_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));
                }
            };
            struct vertex_sync_element{
                uint64_t index;
                index_t degree;
            };
            static_assert(sizeof(vertex_sync_element) == 16,
                          "Topology metadata must not expand the legacy full descriptor mirror");

            struct vertex_element{
                //third_start and third_degree not be used
                uint64_t third_start;
                index_t third_degree;
                bool delta;
                bool cache;
                uint8_t hotness[4];
                uint64_t virtual_start;
                index_t virtual_degree;
                //secondary_start and secondary_degree not be used
                uint64_t secondary_start;
                index_t secondary_degree;
            };       
            struct PMAGraph : public CSRGraphBase
            {
                using AdjacencyView = sepgraph::topology::ContiguousAdjacencyView<
                    vertex_sync_element, index_t>;
                std::vector<uint64_t> row_start_vec; // the vectors are not always in use (see Bind)
                std::vector<index_t> edge_dst_vec;
                std::vector<index_t> edge_weights_vec;
                uint32_t *subgraph_activenode;
                uint32_t *subgraph_rowstart;
                uint32_t *subgraph_edgedst;
                uint32_t *subgraph_edgeweight;
                uint64_t elem_capacity = 0;
                uint64_t elem_capacity_max = 0;
                uint32_t segment_size = 0;
                uint64_t segment_count = 0;
                uint32_t tree_height = 0;
                double delta_up;        // Delta for upper density threshold
                double delta_low;       // Delta for lower density threshold
                struct vertex_element *vertices_;
                struct vertex_sync_element *sync_vertices_;
                index_t *edges_ = nullptr;
                index_t *weights_ = nullptr;
                uint64_t *segment_edges_actual = nullptr;
                uint64_t *segment_edges_total = nullptr;
                bool *delta =nullptr;
                static constexpr double up_h = 0.75; // root
                static constexpr double up_0 = 1.00; // leaves
                // Lower density thresholds
                static constexpr double low_h = 0.50; // root
                static constexpr double low_0 = 0.25; // leaves
                index_t river = 0;
                index_t river_low = 0;
                index_t river_global = 0;
                sepgraph::topology::TopologyEpochContract topology_epoch_;
                int8_t max_sparseness = 1.0 / low_0;
                int8_t largest_empty_segment = 1.0 * max_sparseness;

                PMAGraph(index_t nnodes, uint64_t nedges) : CSRGraphBase(nnodes, nedges, nullptr, nullptr),
                                                           row_start_vec(nnodes + 1), edge_dst_vec(nedges)
                {
                    row_start = &row_start_vec[0];
                    edge_dst = &edge_dst_vec[0];
                }

                PMAGraph()
                {
                }

                AdjacencyView adjacency_view() const
                {
                    return AdjacencyView{sync_vertices_, edges_, nnodes};
                }

                const sepgraph::topology::TopologyEpochContract &topology_epoch() const
                {
                    return topology_epoch_;
                }

                ~PMAGraph()
                {
                }
                
                void compute_capacity(){
                    uint32_t &nnodes = this->nnodes;
                    auto &segment_size = this->segment_size;
                    auto &segment_count = this->segment_count;
                    auto &nedges = this->nedges;
                    auto &elem_capacity = this->elem_capacity;
                    auto &max_sparseness = this->max_sparseness;

                    segment_size = ceil_log2(nnodes);
                    segment_count = ceil_div(nnodes, segment_size); // Ideal number of segments
                    // The number of segments has to be a power of 2, though.
                    segment_count = hyperfloor(segment_count);
                    segment_size = ceil_div(nnodes, segment_count);
                    
                    
                    elem_capacity = nedges * max_sparseness;
                    auto &elem_capacity_max = this->elem_capacity_max;
                    elem_capacity_max = 2 *elem_capacity;
                }
                
                void Bind(index_t nnodes, uint64_t nedges,
                          uint64_t *row_start, index_t *edge_dst,
                          index_t *edge_weights, index_t *node_weights)
                {
                    this->nnodes = nnodes;
                    this->nedges = nedges;

                    this->row_start_vec.clear();
                    this->edge_dst_vec.clear();

                    this->row_start = row_start;
                    this->edge_dst = edge_dst;


                    this->edge_weights = edge_weights;
                    this->node_weights = node_weights;

                    this->subgraph_activenode = (uint32_t *) calloc((nnodes), sizeof(uint32_t));
                    this->subgraph_rowstart = (uint32_t *) calloc((nnodes + 1), sizeof(uint32_t));
                    this->subgraph_edgedst = (uint32_t *) calloc((nedges / 4), sizeof(uint32_t));
                    this->subgraph_edgeweight = (uint32_t *) calloc((nedges / 4), sizeof(uint32_t));

                    compute_capacity();
                    this->segment_edges_actual = (uint64_t *) malloc (sizeof(uint64_t) * segment_count * 2);
                    this->segment_edges_total = (uint64_t *) malloc (sizeof(uint64_t) * segment_count * 2);
                    memset(this->segment_edges_actual, 0, sizeof(uint64_t)*this->segment_count*2);
                    memset(this->segment_edges_total, 0, sizeof(uint64_t)*this->segment_count*2);
                    tree_height = floor_log2(segment_count);
                    delta_up = (up_0 - up_h) / tree_height;
                    delta_low = (low_h - low_0) / tree_height;

                    this->edges_ = (index_t *) malloc (sizeof(index_t) * (this->elem_capacity_max));
                    this->vertices_ = (struct vertex_element *) calloc ((nnodes + 1) , sizeof(struct vertex_element));
                    this->sync_vertices_ = (struct vertex_sync_element *) calloc ((nnodes + 1) , sizeof(struct vertex_sync_element));

                    memset(edges_, -1, sizeof(index_t)*elem_capacity_max);
                    uint32_t *out_degree = new uint32_t[nnodes];

                    for(uint32_t node = 0; node < nnodes; node++){
                        uint64_t begin_edge = row_start[node];
                        uint64_t end_edge = row_start[node+1];
                        out_degree[node] = end_edge - begin_edge;
                        sync_vertices_[node].degree = out_degree[node];
                        sync_vertices_[node].index = row_start[node];
                        vertices_[node].virtual_start = 0;
                        vertices_[node].third_start = 0;
                        vertices_[node].virtual_degree = 0;
                        vertices_[node].third_degree = 0;
                        vertices_[node].secondary_start = 0;
                        vertices_[node].secondary_degree = 0;
                        vertices_[node].delta = false;
                        vertices_[node].hotness[0] = vertices_[node].hotness[1]=vertices_[node].hotness[2]=0;
                        vertices_[node].cache = false;
                        for(uint64_t edge = begin_edge; edge < end_edge; edge++){
                            uint32_t t_segment_id = node / segment_size;
                            uint32_t j = t_segment_id + segment_count;
                            
                            while(j > 0){
                                segment_edges_actual[j] += 1;
                                j/=2;
                            }
                        }
                    }

                    for(uint64_t e = 0; e < nedges; e++){
                        edges_[e] = edge_dst[e];
                        

                    }
                    std::cout << " look edge_dst[nedges-1] : " << edge_dst[nedges-1]<< " look edge_dst[nedges-2] : " << edge_dst[nedges-1] <<  " nedges-1: " << (nedges-1)<<  " nedges-2: " << (nedges-2)<<std::endl;
                    std::cout << " edge_[nedges-1] : " << edges_[nedges-1]<< " look edge_dst[nedges-2] : " << edges_[nedges-1] <<  " nedges-1: " << (nedges-1)<<  " nedges-2: " << (nedges-2)<<std::endl;
                    for (uint32_t i = 1; i < nnodes; i++) {
                        if(sync_vertices_[i].degree == 0) sync_vertices_[i].index = sync_vertices_[i-1].index + sync_vertices_[i-1].degree;
                    }
                    spread_weighted_V1(0,nnodes);
                    
                }

                void spread_weighted_V1(uint32_t start_vertex, uint32_t end_vertex)
                {
                    assert(start_vertex == 0 && "start-vertex is expected to be 0 here.");
                    uint64_t gaps = elem_capacity - nedges;
                    uint64_t * new_positions = calculate_positions_V1(start_vertex, end_vertex, gaps, nedges);

                    uint64_t read_index, write_index, curr_degree;
                    
                    for(uint32_t curr_vertex = end_vertex-1; curr_vertex > start_vertex;curr_vertex-=1){
                        curr_degree = sync_vertices_[curr_vertex].degree;
                        read_index = sync_vertices_[curr_vertex].index + curr_degree - 1;
                        write_index = new_positions[curr_vertex] + curr_degree - 1;

                        assert(write_index >= read_index && "index anomaly occurred while spreading elements");
                    
                        for (uint32_t i = 0; i < curr_degree; i++)
                        {
                            if(curr_vertex==105153950||curr_vertex==105153951){
                                std::cout <<" 前面 "<< "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: "<< edges_[read_index] << " 写到的边： " << edges_[write_index]<<std::endl;
                            }
                            edges_[write_index] = edges_[read_index];
                            if(curr_vertex==105153950||curr_vertex==105153951){
                                std::cout <<" 后面 " << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: "<< edges_[read_index] << " 写到的边： " << edges_[write_index]<<std::endl;
                            }
                            if(read_index < new_positions[curr_vertex]) 
                                edges_[read_index] = -1;
                            if(curr_vertex==105153950||curr_vertex==105153951){
                                std::cout <<" 最后面为-1 " << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: "<< edges_[read_index] << " 写到的边： " << edges_[write_index]<<std::endl;
                            }
                            write_index--;
                            read_index--;
                        }                  

                        sync_vertices_[curr_vertex].index = new_positions[curr_vertex];  
                    }
                    free(new_positions);
                    new_positions = nullptr;
                    recount_segment_total();

                }
                
                void recount_segment_total() {
                    // count the size of each segment in the tree
                    memset(segment_edges_total, 0, sizeof(int64_t)*segment_count*2);
                    for (uint64_t i = 0; i < segment_count; i++){
                        uint64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : sync_vertices_[(i+1)*segment_size].index;
                        uint64_t segment_total_p = next_starter - sync_vertices_[i*segment_size].index;
                        uint32_t j = i + segment_count;  //tree leaves
                        while (j > 0){
                            segment_edges_total[j] += segment_total_p;
                            j /= 2;
                        }
                    }
                }

                void recount_segment_total(index_t start_vertex, index_t end_vertex) {
                    index_t start_seg = get_segment_id(start_vertex) - segment_count;
                    index_t end_seg = get_segment_id(end_vertex) - segment_count;

                    for(index_t i = start_seg; i<end_seg; i += 1) {
                        uint64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : sync_vertices_[(i+1)*segment_size].index;
                        uint64_t segment_total_p = next_starter - sync_vertices_[i*segment_size].index;
                        index_t j = i + segment_count;  //tree leaves
                        segment_total_p -= segment_edges_total[j];
                        while (j > 0){
                            segment_edges_total[j] += segment_total_p;
                            j /= 2;
                        }
                    }
                }

                uint64_t *calculate_positions_V1(uint32_t start_vertex, uint32_t end_vertex, uint64_t gaps, uint64_t total_degree) {
                    uint32_t size = end_vertex - start_vertex;
                    uint64_t *new_index = (uint64_t *) calloc(size, sizeof(uint64_t));
                    total_degree += size;

                    double index_d = sync_vertices_[start_vertex].index;
                    double step = ((double) gaps) / total_degree;  //per-edge step
                    for (uint32_t i = start_vertex; i < end_vertex; i++){
                        new_index[i-start_vertex] = index_d;
                        
                        if(i > start_vertex) {
                            
                            assert(new_index[i-start_vertex] >= new_index[(i-1)-start_vertex] + sync_vertices_[i-1].degree && "Edge-list can not be overlapped with the neighboring vertex!");
                        }
                        index_d += (sync_vertices_[i].degree + (step * (sync_vertices_[i].degree + 1)));
                    }
                    return new_index;
                }
                
                uint64_t begin_edge(index_t node) const
                {
                    return sync_vertices_[node].index;
                }
                
                uint64_t end_edge(index_t node) const
                {
                    return sync_vertices_[node + 1].index;
                }

                index_t edge_degree(index_t node) const
                {
                    return sync_vertices_[node].degree;
                }

                index_t edge_dest(uint64_t edge) const
                {
                    return edges_[edge];
                }      

                void PrintHistogram(uint32_t *&p_in_degree,uint32_t *&p_out_degree){
                    uint32_t *in_degree = new uint32_t[nnodes];
                    uint32_t *out_degree = new uint32_t[nnodes];
                    for (index_t node = 0; node < nnodes; node++)
                    {
                        index_t begin_edge = this->sync_vertices_[node].index;

                        
                        out_degree[node] = this->sync_vertices_[node].degree;
                        for (index_t edge = begin_edge; edge < begin_edge +out_degree[node] ; edge++)
                        {
                            index_t dest = this->edge_dest(edge);
                            
                                in_degree[dest]++;
                            
                        }
                    }
                    GROUTE_CUDA_CHECK(cudaMalloc(&p_in_degree, nnodes * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_in_degree, in_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

                    GROUTE_CUDA_CHECK(cudaMalloc(&p_out_degree, nnodes * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_out_degree, out_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

                }

                void PrintHistogram_update(uint32_t *&p_in_degree,uint32_t *&p_out_degree){
                    uint32_t *in_degree = new uint32_t[nnodes];
                    uint32_t *out_degree = new uint32_t[nnodes];
                    for (index_t node = 0; node < nnodes; node++)
                    {
                        index_t begin_edge = this->sync_vertices_[node].index;

                        
                        out_degree[node] = sync_vertices_[node].degree;
                        for (index_t edge = begin_edge; edge < (begin_edge +out_degree[node]) ; edge++)
                        {
                            index_t dest = this->edge_dest(edge);
                            
                                in_degree[dest]++;
                            
                        }
                    }
                    
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_in_degree, in_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

                    
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(p_out_degree, out_degree, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

                }

                index_t get_segment_id(index_t vertex_id) {
                    return (vertex_id / segment_size) + segment_count;
                }

                void update_segment_edge_total(index_t vertex_id, int count) {
                    index_t j = get_segment_id(vertex_id);
                    while (j > 0){
                    segment_edges_total[j] += count;
                    j /= 2;
                    }
                }

            void insert(index_t src, index_t dst, index_t value)
            {
                index_t current_segment = get_segment_id(src);
                // raqib: we now allow to look for space beyond the segment boundary
                assert(segment_edges_total[current_segment] > segment_edges_actual[current_segment] && "There is no enough space in the current segment for performing this insert!");

                uint64_t loc = sync_vertices_[src].index + sync_vertices_[src].degree;
                uint64_t right_free_slot = -1;
                uint64_t left_free_slot = -1;
                index_t left_vertex = src, right_vertex = src;
                index_t left_vertex_boundary, right_vertex_boundary;

                if(segment_edges_total[current_segment] > segment_edges_actual[current_segment]) {
                    left_vertex_boundary = (src / segment_size) * segment_size;
                    right_vertex_boundary = min(left_vertex_boundary + segment_size, nnodes - 1);
                }
                else {
                    index_t curr_seg_size = segment_size, j = current_segment;
                    while(j) {
                        if(segment_edges_total[j] > segment_edges_actual[j]) break;
                        j /= 2;
                        curr_seg_size *= 2;
                    }
                    left_vertex_boundary = (src / curr_seg_size) * curr_seg_size;
                    right_vertex_boundary = min(left_vertex_boundary + curr_seg_size, nnodes - 1);
                }

                // search right side for a free slot
                // raqib: shouldn't we search within the pma leaf?
                for (index_t i = src; i < right_vertex_boundary; i++) {
                    if (sync_vertices_[i + 1].index > (sync_vertices_[i].index + sync_vertices_[i].degree)) {
                        right_free_slot = sync_vertices_[i].index + sync_vertices_[i].degree;  // we get a free slot here
                        right_vertex = i;
                        break;
                    }
                }

                // in the last segment where we skipped the last vertex
                
                if (right_free_slot == -1 && right_vertex_boundary == (nnodes - 1)
                    && elem_capacity > (1 + sync_vertices_[nnodes - 1].index + sync_vertices_[nnodes - 1].degree)) {
                right_free_slot = sync_vertices_[nnodes - 1].index + sync_vertices_[nnodes - 1].degree;
                right_vertex = nnodes - 1;
                }

                // if no space on the right side, search the left side
                if (right_free_slot == -1) {
                for (index_t i = src; i > left_vertex_boundary; i--) {
                    if (sync_vertices_[i].index > (sync_vertices_[i - 1].index + sync_vertices_[i - 1].degree)) {
                    left_free_slot = sync_vertices_[i].index - 1;  // we get a free slot here
                    left_vertex = i;
                    break;
                    }
                }
                }

                // found free slot on the right
                if (right_free_slot != -1)
                {
                    if (right_free_slot >= loc)  // move elements to the right to get the free slot
                    {
                        for (index_t i = right_free_slot; i > loc; i--)
                        {
                            edges_[i] = edges_[i-1];
                        
                        }

                        for (index_t i = src+1; i <= right_vertex; i++) {
                            sync_vertices_[i].index += 1;
                        }

                        //update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
                        if(current_segment != get_segment_id(right_vertex)) {
                        update_segment_edge_total(src, 1);
                        update_segment_edge_total(right_vertex, -1);
                        }
                    }
                    edges_[loc] = dst;

                    sync_vertices_[src].degree += 1;
                }
                else if (left_free_slot != -1)
                {
                    if (left_free_slot < loc)
                    {
                        for (index_t i = left_free_slot; i < loc - 1; i++)
                        {
                            edges_[i]= edges_[i+1];
                        }

                        for (index_t i = left_vertex; i <= src; i++) {
                            sync_vertices_[i].index -= 1;
                        }

                        //update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
                        if(current_segment != get_segment_id(left_vertex)) {
                        update_segment_edge_total(src, 1);
                        update_segment_edge_total(left_vertex, -1);
                        }
                    }
                    edges_[loc - 1] = dst;

                    sync_vertices_[src].degree += 1;
                }
                else
                {
                assert(false && "Should not happen");
                }

                // we insert a new edge, increasing the degree of the whole subtree
                index_t j = current_segment;
                
                while (j > 0){
                segment_edges_actual[j] += 1;
                j /= 2;
                }

                // whether we need to update segment_edges_total
                // raqib: I think we should do the re-balance first and then make the insertion.
                // This will always ensure the empty space for the insertion.

                rebalance_wrapper(src);
                topology_epoch_.MarkMutation();
                // todo: we should also consider rebalancing the donner vertex's segment
            }

                
            bool del_edge(index_t src, index_t dst, index_t weight){
                (void)weight;

                uint64_t left = sync_vertices_[src].index;
                uint64_t right = sync_vertices_[src].index + sync_vertices_[src].degree;
                
                for(uint64_t i = left; i < right; i++){
                    if(edges_[i]== dst){
                        
                        for(uint64_t j = (i + 1); j < right ; j++){
                            edges_[j - 1] = edges_[j];
                        }
                        
                        sync_vertices_[src].degree -= 1;
                        
                        edges_[right - 1] = -1;
                        topology_epoch_.MarkMutation();
                        return true;
                    }
                }
                return false;
            }

                void rebalance_wrapper(index_t src){
                    
                    index_t height = 0;
                    index_t window = (src / segment_size) + segment_count;
                    double density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
                    double up_height = up_0 - (height * delta_up);
                    double low_height = low_0 + (height * delta_low);

                    while (window > 0 && (density >= up_height))
                    {
                        // Repeatedly check window containing an increasing amount of segments
                        
                        // so, we are going to check if the current window can fulfill the density thresholds.
                        

                        // Go one level up in our conceptual PMA tree
                        window /= 2;
                        height += 1;

                        up_height = up_0 - (height * delta_up);
                        low_height = low_0 + (height * delta_low);

                        density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
                        
                    }

                    if(!height) {
                        // rebalance is not required in the single pma leaf
                        return;
                    }
                    index_t left_index, right_index;
                    
                    if (density < up_height)
                    {
                        // Found a window within threshold
                        index_t window_size = segment_size * (1 << height);
                        left_index = (src / window_size) * window_size;
                        right_index = std::min(left_index + window_size, nnodes);

                        // do degree-based distribution of gaps
                        rebalance_weighted(left_index, right_index, window);
                    }
                    else {
                        // Rebalance not possible without increasing the underlying array size.
                        // need to resize the size of "edges_" array
                        resize();
                    }
                }

                void rebalance_weighted(index_t start_vertex,index_t end_vertex,index_t pma_idx){
                    
                    uint64_t from = sync_vertices_[start_vertex].index;
                    uint64_t to = (end_vertex >= nnodes) ? elem_capacity : sync_vertices_[end_vertex].index;
                    assert(to > from && "Invalid range found while doing weighted rebalance");
                    uint64_t capacity = to - from;
                    assert(segment_edges_total[pma_idx] == capacity && "Segment capacity is not matched with segment_edges_total");
                    uint64_t gaps = segment_edges_total[pma_idx] - segment_edges_actual[pma_idx];
                    index_t size = end_vertex - start_vertex;
                    uint64_t *new_index = calculate_positions_V1(start_vertex, end_vertex, gaps, segment_edges_actual[pma_idx]);
                    uint64_t index_boundary = (end_vertex >= nnodes) ? elem_capacity : sync_vertices_[end_vertex].index;
                    assert(new_index[size - 1] + sync_vertices_[end_vertex - 1].degree <= index_boundary && "Rebalance (weighted) index calculation is wrong!");
                    index_t ii, jj;
                    index_t curr_vertex = start_vertex + 1, next_to_start;
                    uint64_t read_index, last_read_index, write_index;

                    while (curr_vertex < end_vertex)
                    {
                        for (ii = curr_vertex; ii < end_vertex; ii++)
                        {
                            if(new_index[ii-start_vertex] <= sync_vertices_[ii].index) break;
                        }
                        if(ii == end_vertex) ii -= 1;
                        assert(new_index[ii-start_vertex] <= sync_vertices_[ii].index && "This should not happen!");
                        next_to_start = ii + 1;
                        if(new_index[ii-start_vertex] <= sync_vertices_[ii].index) {
                            // now it is guaranteed that, ii's new-starting index is less than or equal to it's old-starting index
                            jj = ii;
                            read_index = sync_vertices_[jj].index;
                            last_read_index = read_index + sync_vertices_[jj].degree;
                            write_index = new_index[jj - start_vertex];

                            while (read_index < last_read_index) {
                                edges_[write_index] = edges_[read_index];
                                write_index++;
                                read_index++;
                            }
                            // update the index to the new position
                            sync_vertices_[jj].index = new_index[jj - start_vertex];

                            ii -= 1;
                        }

                        // from current_vertex to ii, the new-starting index is greater than to it's old-starting index
                        for (jj=ii; jj>=curr_vertex; jj-=1)
                        {
                            read_index = sync_vertices_[jj].index + sync_vertices_[jj].degree - 1;
                            last_read_index = sync_vertices_[jj].index;
                            write_index = new_index[jj-start_vertex] + sync_vertices_[jj].degree - 1;

                            while(read_index >= last_read_index)
                            {
                                edges_[write_index] = edges_[read_index];
                                
                                write_index--;
                                read_index--;
                            }

                            // update the index to the new position
                            sync_vertices_[jj].index = new_index[jj-start_vertex];
                        }
                        // move current_vertex to the next position of ii
                        curr_vertex = next_to_start;
                    }
                    free(new_index);
                    new_index = nullptr;
                    
                    recount_segment_total(start_vertex, end_vertex);
                }

                void resize(){
                    
                    elem_capacity *= 2;
                    uint64_t gaps = elem_capacity - nedges;
                    uint64_t *new_indices = calculate_positions_V1(0, nnodes, gaps, nedges);
                    index_t *new_edges_ = (index_t *)malloc(sizeof(index_t) * elem_capacity);

                    for(index_t curr_vertex = nnodes - 1; curr_vertex >= 0; curr_vertex--) {
                        for(index_t i=0; i<sync_vertices_[curr_vertex].degree; i+=1) {
                            new_edges_[new_indices[curr_vertex] + i] = edges_[sync_vertices_[curr_vertex].index + i];
                            
                            
                        }
                        sync_vertices_[curr_vertex].index = new_indices[curr_vertex];
                    }
                    for(index_t edge = 0; edge < elem_capacity; edge++){
                        edges_[edge] = new_edges_[edge];
                    }
                    free(new_edges_);

                    recount_segment_total();
                }

                void print_vertices(){

                            // printf("vertices_:(%d)|%llu|%llu| \n", 741, sync_vertices_[741].index

                }


                index_t avg_degree() const
                {
                    return nedges / nnodes;
                }
            };
            /*
            * @brief A host graph object
            */
            struct CSCGraph : public CSCGraphBase
            {
                std::vector<index_t> col_start_vec;
                std::vector<index_t> edge_src_vec;
                std::vector<index_t> edge_weights_vec;
                std::vector<index_t> out_degree_vec;

                /*
                * extract from :https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
                * Compute B = A for CSR matrix A, CSC matrix B
                *
                * Also, with the appropriate arguments can also be used to:
                *   - compute B = A^t for CSR matrix A, CSR matrix B
                *   - compute B = A^t for CSC matrix A, CSC matrix B
                *   - convert CSC->CSR
                *
                * Input Arguments:
                *   I  n_row         - number of rows in A
                *   I  n_col         - number of columns in A
                *   I  Ap[n_row+1]   - row pointer
                *   I  Aj[nnz(A)]    - column indices
                *   T  Ax[nnz(A)]    - nonzeros
                *
                * Output Arguments:
                *   I  Bp[n_col+1] - column pointer
                *   I  Bj[nnz(A)]  - row indices
                *   T  Bx[nnz(A)]  - nonzeros
                *
                * Note:
                *   Output arrays Bp, Bj, Bx must be preallocated
                *
                * Note:
                *   Input:  column indices *are not* assumed to be in sorted order
                *   Output: row indices *will be* in sorted order
                *
                *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
                *
                */
                template <class I, class T>
                void csr_tocsc(const I n_row,
                               const I n_col,
                               const I Ap[],
                               const I Aj[],
                               const T Ax[],
                               I Bp[],
                               I Bi[],
                               T Bx[]) const
                {
                    const I nnz = Ap[n_row];

                    //compute number of non-zero entries per column of A
                    std::fill(Bp, Bp + n_col, 0);

                    for (I n = 0; n < nnz; n++)
                    {
                        Bp[Aj[n]]++;
                    }

                    
                    for (I col = 0, cumsum = 0; col < n_col; col++)
                    {
                        I temp = Bp[col];
                        Bp[col] = cumsum;
                        cumsum += temp;
                    }
                    Bp[n_col] = nnz;

                    //weighted graph
                    if (Bx != nullptr && Ax != nullptr)
                    {
                        for (I row = 0; row < n_row; row++)
                        {
                            for (I jj = Ap[row]; jj < Ap[row + 1]; jj++)
                            {
                                I col = Aj[jj];
                                I dest = Bp[col];

                                Bi[dest] = row;
                                Bx[dest] = Ax[jj];

                                Bp[col]++;
                            }
                        }
                    }
                    else
                    {
                        for (I row = 0; row < n_row; row++)
                        {
                            for (I jj = Ap[row]; jj < Ap[row + 1]; jj++)
                            {
                                I col = Aj[jj];
                                I dest = Bp[col];

                                Bi[dest] = row;

                                Bp[col]++;
                            }
                        }
                    }

                    for (I col = 0, last = 0; col <= n_col; col++)
                    {
                        I temp = Bp[col];
                        Bp[col] = last;
                        last = temp;
                    }
                }

                CSCGraph(groute::graphs::host::CSRGraph &csr_graph, bool undirected) : CSCGraphBase(csr_graph.nnodes,
                                                                                                    csr_graph.nedges,
                                                                                                    nullptr,
                                                                                                    nullptr)
                {
                        col_start_vec = std::vector<index_t>(csr_graph.nnodes + 1);
                        edge_src_vec = std::vector<index_t>(csr_graph.nedges);
                        out_degree_vec = std::vector<index_t>(csr_graph.nnodes);

                        col_start = &col_start_vec[0];
                        edge_source = &edge_src_vec[0];
                        out_dgr = &out_degree_vec[0];

                        //edge_weights point to csr's weight
                        //I need to alloc a new space
                        if (csr_graph.edge_weights != nullptr)
                        {
                            edge_weights_vec = std::vector<index_t>(nedges);
                            edge_weights = &edge_weights_vec[0];
                        }
                    
                }

                ~CSCGraph() = default;

                index_t max_in_degree() const
                {
                    index_t max_degree = 0;
                    for (index_t node = 0; node < nnodes; node++)
                    {
                        max_degree = std::max(max_degree, end_edge(node) - begin_edge(node));
                    }
                    return max_degree;
                }

                index_t out_degree(index_t node) const
                {
                    return out_dgr[node];
                }

                index_t begin_edge(index_t node) const
                {
                    return col_start[node];
                }

                index_t end_edge(index_t node) const
                {
                    return col_start[node + 1];
                }

                index_t edge_src(index_t edge) const
                {
                    return edge_source[edge];
                }
            };

            /*
            * @brief A host graph generator (CSR)
            * @note The generated graph is asymmetric and may have duplicated edges but no self loops
            */
            class CSRGraphGenerator
            {
            private:
                index_t m_nnodes;
                int m_gen_factor;

                std::default_random_engine m_generator;
                std::uniform_int_distribution<int> m_nneighbors_distribution;
                std::uniform_int_distribution<index_t> m_node_distribution;

                int GenNeighborsNum(index_t node)
                {
                    return m_nneighbors_distribution(m_generator);
                }

                index_t GenNeighbor(index_t node, std::set<index_t> &neighbors)
                {
                    index_t neighbor;
                    do
                    {
                        neighbor = m_node_distribution(m_generator);
                    } while (neighbor == node || neighbors.find(neighbor) != neighbors.end());

                    neighbors.insert(neighbor);
                    return neighbor;
                }

            public:
                CSRGraphGenerator(index_t nnodes, int gen_factor) : m_nnodes(nnodes), m_gen_factor(gen_factor), m_nneighbors_distribution(1, gen_factor),
                                                                    m_node_distribution(0, nnodes - 1)
                {
                    assert(nnodes > 1);
                    assert(gen_factor >= 1);
                }

                void Gen(CSRGraph &graph)
                {
                    std::vector<index_t> row_start(m_nnodes + 1, 0);
                    std::vector<index_t> edge_dst;
                    edge_dst.reserve(m_nnodes * m_gen_factor); // approximation

                    for (index_t node = 0; node < m_nnodes; ++node)
                    {
                        row_start[node] = edge_dst.size();
                        int nneighbors = GenNeighborsNum(node);
                        std::set<index_t> neighbors;
                        for (int i = 0; i < nneighbors; ++i)
                        {
                            edge_dst.push_back(GenNeighbor(node, neighbors));
                        }
                    }

                    index_t nedges = edge_dst.size();
                    row_start[m_nnodes] = nedges;

                    edge_dst.shrink_to_fit(); //

                    
                }
            };

            class NoIntersectionGraphGenerator
            {
            private:
                int m_ngpus;
                index_t m_nnodes;
                int m_gen_factor;

            public:
                NoIntersectionGraphGenerator(int ngpus, index_t nnodes, int gen_factor) : m_ngpus(ngpus), m_nnodes((nnodes / ngpus) * ngpus /*round*/), m_gen_factor(gen_factor)
                {
                    assert(nnodes >= ngpus);
                    assert(gen_factor >= 1);
                }

                void Gen(CSRGraph &graph)
                {
                    // Builds a simple two-way chain with no intersection between segments

                    std::vector<index_t> row_start(m_nnodes + 1, 0);
                    std::vector<index_t> edge_dst;

                    edge_dst.reserve(m_nnodes * 2);
                    index_t seg_nnodes = m_nnodes / m_ngpus;

                    for (index_t node = 0; node < m_nnodes; ++node)
                    {
                        index_t seg_idx = node / seg_nnodes;
                        index_t seg_snode = seg_idx * seg_nnodes;

                        row_start[node] = edge_dst.size();

                        if (node >= seg_snode + 1)
                            edge_dst.push_back(node - 1);
                        if (node + 1 < seg_snode + seg_nnodes)
                            edge_dst.push_back(node + 1);
                    }

                    index_t nedges = edge_dst.size();
                    row_start[m_nnodes] = nedges;

                    edge_dst.shrink_to_fit(); //

                    
                }
            };

            class ChainGraphGenerator
            {
            private:
                int m_ngpus;
                index_t m_nnodes;
                int m_gen_factor;

            public:
                ChainGraphGenerator(int ngpus, index_t nnodes, int gen_factor) : m_ngpus(ngpus), m_nnodes((nnodes / ngpus) * ngpus /*round*/), m_gen_factor(gen_factor)
                {
                    assert(nnodes >= ngpus);
                    assert(gen_factor >= 1);
                }

                void Gen(CSRGraph &graph)
                {
                    std::vector<index_t> row_start(m_nnodes + 1, 0);
                    std::vector<index_t> edge_dst;

                    edge_dst.reserve(m_nnodes * 2);

                    for (index_t node = 0; node < m_nnodes; ++node)
                    {
                        row_start[node] = edge_dst.size();

                        if (node >= 1)
                            edge_dst.push_back(node - 1);
                        if (node + 1 < m_nnodes)
                            edge_dst.push_back(node + 1);
                    }

                    index_t nedges = edge_dst.size();
                    row_start[m_nnodes] = nedges;

                    edge_dst.shrink_to_fit(); //

                    
                }
            };

            class CliquesNoIntersectionGraphGenerator
            {
            private:
                int m_ngpus;
                index_t m_nnodes;
                int m_gen_factor;

            public:
                CliquesNoIntersectionGraphGenerator(int ngpus, index_t nnodes, int gen_factor) : m_ngpus(ngpus), m_nnodes((nnodes / ngpus) * ngpus /*round*/), m_gen_factor(gen_factor)
                {
                    assert(nnodes >= ngpus);
                    assert(gen_factor >= 1);
                }

                void Gen(CSRGraph &graph)
                {
                    std::vector<index_t> row_start(m_nnodes + 1, 0);
                    std::vector<index_t> edge_dst;

                    index_t seg_nnodes = m_nnodes / m_ngpus;
                    edge_dst.reserve(m_nnodes * seg_nnodes);

                    for (index_t node = 0; node < m_nnodes; ++node)
                    {
                        index_t seg_idx = node / seg_nnodes;
                        index_t seg_snode = seg_idx * seg_nnodes;

                        row_start[node] = edge_dst.size();
                        for (int i = 0; i < seg_nnodes; ++i)
                        {
                            if (seg_snode + i == node)
                                continue;
                            edge_dst.push_back(seg_snode + i);
                        }
                    }

                    index_t nedges = edge_dst.size();
                    row_start[m_nnodes] = nedges;

                    edge_dst.shrink_to_fit(); //

                    
                }
            };
        } // namespace host

        namespace dev // device objects
        {
            /*
            * @brief A multi-GPU graph segment object (represents a segment allocated at one GPU)
            */
            struct CSRGraphSeg : public CSRGraphBase
            {
                int seg_idx, nsegs;

                index_t nodes_offset, edges_offset;
                index_t nnodes_local, nedges_local;

                CSRGraphSeg() : seg_idx(-1), nsegs(-1),
                                nodes_offset(0), edges_offset(0), nnodes_local(0), nedges_local(0)
                {
                }

                __device__ __host__ __forceinline__ bool owns(index_t node) const
                {
                    assert(node < nnodes);
                    return node >= nodes_offset && node < (nodes_offset + nnodes_local);
                }

                __host__ __device__ __forceinline__ index_t owned_start_node() const
                {
                    return nodes_offset;
                }

                __host__ __device__ __forceinline__ index_t owned_nnodes() const
                {
                    return nnodes_local;
                }

                __host__ __device__ __forceinline__ index_t global_nnodes() const
                {
                    return nnodes;
                }

                __device__ __forceinline__ uint64_t begin_edge(index_t node) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(row_start + node - nodes_offset);
#else
                    return row_start[node - nodes_offset];
#endif
                }

                __device__ __forceinline__ uint64_t end_edge(index_t node) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(row_start + node + 1 - nodes_offset);
#else
                    return row_start[node + 1 - nodes_offset];
#endif
                }

                __device__ __forceinline__ index_t edge_dest(uint64_t edge) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(edge_dst + edge - edges_offset);
#else
                    return edge_dst[edge - edges_offset];
#endif
                }
            };

            /*
            * @brief A single GPU graph object (a complete graph allocated at one GPU)
            */
            struct CSRGraph : public CSRGraphBase
            {
                index_t *edge_dst_zc;
		        index_t *edge_dst_exp[64];
                index_t *edge_dst_com;

                //for subgraph compaction
                uint32_t *subgraph_activenode;
                uint32_t *subgraph_rowstart;
                CSRGraph()
                {
                }

                __device__ __host__ __forceinline__ bool owns(index_t node) const
                {
                    assert(node < nnodes);
                    return true;
                }

                __host__ __device__ __forceinline__ index_t owned_start_node() const
                {
                    return 0;
                }

                __host__ __device__ __forceinline__ index_t owned_nnodes() const
                {
                    return nnodes;
                }

                __host__ __device__ __forceinline__ index_t global_nnodes() const
                {
                    return nnodes;
                }

                __device__ __forceinline__ uint64_t begin_edge(index_t node, index_t node_offset = 0) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(row_start + node - node_offset);
#else
                    return row_start[node - node_offset];
#endif
                }

                __device__ __forceinline__ uint64_t end_edge(index_t node, index_t node_offset = 0) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(row_start + node + 1 - node_offset);
#else
                    return row_start[node + 1 - node_offset];
#endif
                }

                __device__ __forceinline__ index_t edge_dest(uint64_t edge) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(edge_dst + edge);
#else
                    return edge_dst[edge];
#endif
                }
            };
            /*
            * @brief A single GPU graph object (a complete graph allocated at one GPU)
            */
            struct vertex_sync_element{
                uint64_t index;
                index_t degree;
            };     

            struct vertex_element{
                uint64_t third_start;
                index_t third_degree;
                bool delta;
                bool cache;
                uint64_t virtual_start;
                index_t virtual_degree;
                uint64_t secondary_start;
                uint8_t hotness[4];
                index_t secondary_degree;
            };       
            struct PMAGraph : public CSRGraphBase
            {
                using AdjacencyView = sepgraph::topology::ContiguousAdjacencyView<
                    groute::graphs::host::vertex_sync_element, index_t>;
                
                index_t *edge_dst_zc;
                index_t *weight_dst_zc;
		        index_t *edge_dst_exp[64];
                index_t *edge_dst_com;
                struct groute::graphs::host::vertex_element *vertices_;
                struct groute::graphs::host::vertex_sync_element *sync_vertices_;
                
                index_t *edges_;
                index_t **chunk_slabs_ = nullptr;
                index_t *weights_;
                index_t *river;
                index_t *river_low;
                index_t *river_global;
                //for subgraph compaction
                uint32_t *subgraph_activenode;
                uint32_t *subgraph_rowstart;
                // gpu unordered_map to find cache vertex
                //vertex has index
                //large edge

                
                PMAGraph()
                {
                }

                __device__ __forceinline__ AdjacencyView adjacency_view() const
                {
                    return AdjacencyView{sync_vertices_, edges_, nnodes};
                }

                __device__ __host__ __forceinline__ bool owns(index_t node) const
                {
                    assert(node < nnodes);
                    return true;
                }

                __host__ __device__ __forceinline__ index_t owned_start_node() const
                {
                    return 0;
                }

                __host__ __device__ __forceinline__ index_t owned_nnodes() const
                {
                    return nnodes;
                }

                __host__ __device__ __forceinline__ index_t global_nnodes() const
                {
                    return nnodes;
                }

                __device__ __forceinline__ uint64_t begin_edge(index_t node, index_t node_offset = 0) const
                {

                    return adjacency_view().Begin(node - node_offset);

                }

                __device__ __forceinline__ uint64_t end_edge(index_t node, index_t node_offset = 0) const
                {

                    const index_t source = node - node_offset;
                    return begin_edge(node, node_offset) +
                           sync_vertices_[source].degree;

                }

                __device__ __forceinline__ index_t edge_dest(uint64_t edge) const
                {

                    const uint32_t slab_id = static_cast<uint32_t>(edge >> 32);
                    const uint32_t offset = static_cast<uint32_t>(edge);
                    return chunk_slabs_[slab_id][offset];

                }

                __device__ __forceinline__ index_t degree(index_t node) const
                {
                    return sync_vertices_[node].degree;
                }
                // __device__ __forceinline__ bool is_delta(index_t node) const

                __device__ __forceinline__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
                {
                    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
                    unsigned long long old = *address_as_ull, assumed;

                    do {
                        assumed = old;
                        old = atomicCAS(address_as_ull, assumed,
                                        val +assumed);

                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);

                    return old;

                }
            };
            /*
            * @brief A single GPU graph object (a complete graph allocated at PinnedMemory)
            * Which is accessible on GPU
            */

            struct CSCGraph : public CSCGraphBase
            {
                bool m_undirected;
                index_t *edge_source_zc;
                index_t *edge_source_exp;

                CSCGraph(bool undirected = false) : m_undirected(undirected)
                {
                }

                __device__ __host__ __forceinline__ bool owns(index_t node) const
                {
                    assert(node < nnodes);
                    return true;
                }

                __host__ __device__ __forceinline__ index_t owned_start_node() const
                {
                    return 0;
                }

                __host__ __device__ __forceinline__ index_t owned_nnodes() const
                {
                    return nnodes;
                }

                __host__ __device__ __forceinline__ index_t global_nnodes() const
                {
                    return nnodes;
                }

                __device__ __forceinline__ index_t begin_edge(index_t node, index_t node_offset = 0) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(col_start + node - node_offset);
#else
                    return col_start[node - node_offset];
#endif
                }

                __device__ __forceinline__ index_t end_edge(index_t node, index_t node_offset = 0) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(col_start + node + 1 - node_offset);
#else
                    return col_start[node + 1 - node_offset];
#endif
                }

                __device__ __forceinline__ index_t edge_src(index_t edge) const
                {
#if __CUDA_ARCH__ >= 320
                    return __ldg(edge_source + edge);
#else
                    return edge_source[edge];
#endif
                }

                __device__ __forceinline__ index_t out_degree(index_t node) const
                {
                    if (m_undirected)
                    {
                        return col_start[node + 1] - col_start[node];
                    }
                    else
                    {
#if __CUDA_ARCH__ >= 32
                        return __ldg(out_dgr + node);
#else
                        return out_dgr[node];
#endif
                    }
                }
            };

            template <typename T>
            struct GraphDatumSeg
            {
                T *data_ptr;
                index_t offset;
                index_t size;

                GraphDatumSeg() : data_ptr(nullptr), offset(0), size(0)
                {
                }

                GraphDatumSeg(T *data_ptr, index_t offset, index_t size) : data_ptr(data_ptr), offset(offset),
                                                                           size(size)
                {
                }

                __device__ __forceinline__ T get_item(index_t idx) const
                {
                    assert(idx >= offset && idx < offset + size);
                    return data_ptr[idx - offset];
                }

                __device__ __forceinline__ T &operator[](index_t idx)
                {
                    return data_ptr[idx - offset];
                }

                __device__ __forceinline__ T *get_item_ptr(index_t idx) const
                {
                    assert(idx >= offset && idx < offset + size);
                    return data_ptr + (idx - offset);
                }

                __device__ __forceinline__ void set_item(index_t idx, const T &item) const
                {
                    assert(idx >= offset && idx < offset + size);
                    data_ptr[idx - offset] = item;
                }
            };

            template <typename T>
            struct GraphDatum
            {
                T *data_ptr;
		        T *data_ptr_zc;
		        T *data_ptr_exp[64];
                T *data_ptr_com;
		
                index_t size;

                GraphDatum() : data_ptr(nullptr), size(0)
                {
                }

                GraphDatum(T *data_ptr, index_t size) : data_ptr(data_ptr), size(size)
                {
                }

                __device__ __forceinline__ T get_item(index_t idx) const
                {
                    assert(idx < size);
                    return data_ptr[idx];
                }

                __device__ __forceinline__ T &operator[](index_t idx)
                {
                    return data_ptr[idx];
                }

                __device__ __forceinline__ T *get_item_ptr(index_t idx) const
                {
                    assert(idx < size);
                    return data_ptr + (idx);
                }
                __device__ __forceinline__ T *&operator+(index_t idx) const
                {
                    assert(idx < size);
                    return data_ptr + idx;
                }

                __device__ __forceinline__ void set_item(index_t idx, const T &item) const
                {
                    assert(idx < size);
                    data_ptr[idx] = item;
                }
            };
        } // namespace dev

        namespace single
        {

            static __global__ void ScatterTopologyPatch(
                    const sepgraph::topology::TopologyPatchRecord *patch,
                    uint32_t count,
                    host::vertex_sync_element *compat_descriptors)
            {
                const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= count) return;
                const auto record = patch[tid];
                compat_descriptors[record.source].index =
                    (static_cast<uint64_t>(record.slab_id) << 32) |
                    static_cast<uint32_t>(record.offset);
                compat_descriptors[record.source].degree = record.degree;
            }

            static __device__ bool ReserveCachePatchOrInvalidate(
                    host::vertex_element &vertex,
                    index_t degree,
                    uint64_t cache_capacity,
                    index_t *cache_tail,
                    unsigned long long *cache_invalidations)
            {
                bool patch_in_place = vertex.cache;
                if (patch_in_place && degree > vertex.virtual_degree) {
                    const index_t new_start = atomicAdd(cache_tail, degree);
                    if (static_cast<uint64_t>(new_start) + degree <= cache_capacity)
                        vertex.virtual_start = new_start;
                    else
                        patch_in_place = false;
                }
                if (patch_in_place) vertex.virtual_degree = degree;
                if (vertex.cache && !patch_in_place) {
                    vertex.cache = false;
                    vertex.virtual_degree = 0;
                    vertex.third_degree = 0;
                    atomicAdd(cache_invalidations, 1ULL);
                }
                return patch_in_place;
            }

            static __global__ void PatchOrInvalidateCachedAdjacency(
                    const sepgraph::topology::TopologyPatchRecord *patch,
                    uint32_t count,
                    index_t **slabs,
                    index_t *cache_edges,
                    uint64_t cache_capacity,
                    index_t *cache_tail,
                    host::vertex_element *vertices,
                    unsigned long long *cache_invalidations)
            {
                const uint32_t record_index = blockIdx.x;
                if (record_index >= count) return;
                const auto record = patch[record_index];
                __shared__ bool patch_in_place;
                if (threadIdx.x == 0) {
                    auto &vertex = vertices[record.source];
                    patch_in_place = ReserveCachePatchOrInvalidate(
                        vertex, record.degree, cache_capacity, cache_tail,
                        cache_invalidations);
                }
                __syncthreads();
                if (!patch_in_place) return;
                const uint64_t destination = vertices[record.source].virtual_start;
                for (uint64_t offset = threadIdx.x; offset < record.degree;
                     offset += blockDim.x) {
                    cache_edges[destination + offset] =
                        slabs[record.slab_id][record.offset + offset];
                }
            }

            static __global__ void HashPublishedAdjacency(
                    const sepgraph::topology::TopologyPatchRecord *patch,
                    uint32_t count,
                    index_t **slabs,
                    sepgraph::topology::TopologyDeviceDigest *digests)
            {
                const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= count) return;
                const auto record = patch[tid];
                uint64_t hash = 1469598103934665603ULL;
                for (uint64_t offset = 0; offset < record.degree; ++offset) {
                    hash ^= static_cast<uint64_t>(
                        slabs[record.slab_id][record.offset + offset]);
                    hash *= 1099511628211ULL;
                }
                hash ^= static_cast<uint64_t>(record.degree);
                hash *= 1099511628211ULL;
                digests[tid] = {record.degree, hash};
            }

            /*
            * @brief A single GPU graph allocator (allocates a complete mirror graph at one GPU)
            */
            struct PMAGraphAllocator
            {
                struct SparsePublicationResult {
                    double publication_ms = 0.0;
                    unsigned long long stale_rejects = 0;
                    unsigned long long cache_invalidations = 0;
                    unsigned long long hash_mismatches = 0;
                    uint64_t cache_tail = 0;
                };

                typedef dev::PMAGraph DeviceObjectType;
                
            public:
                host::PMAGraph &m_origin_graph;
                dev::PMAGraph m_dev_mirror;
                bool m_on_pinned_memory;
                sepgraph::topology::SourceLocalChunkStore *m_chunk_store = nullptr;
                sepgraph::topology::TopologyPatchRecord *m_patch_device = nullptr;
                sepgraph::topology::TopologyDeviceDigest *m_digest_device = nullptr;
                size_t m_patch_capacity = 0;
                unsigned long long *m_cache_invalidations = nullptr;
                index_t **m_chunk_slabs_device = nullptr;
                sepgraph::topology::TopologyPatchRecord *m_patch_host = nullptr;
                cudaEvent_t m_publication_begin = nullptr;
                cudaEvent_t m_publication_end = nullptr;
                size_t m_publication_count = 0;
                bool m_publication_pending = false;
                bool m_publication_audit = false;
                

            public:
                PMAGraphAllocator(host::PMAGraph &host_graph, uint64_t seg_nedges) : m_origin_graph(host_graph)
                {
                    AllocateDevMirror_node(seg_nedges);
                }

                ~PMAGraphAllocator()
                {
                    DeallocateDevMirror();
                }

                void ReloadAllocator()
                {
                    
                    AllocateDevMirror_node_update();
                    m_origin_graph.topology_epoch_.MarkPublished();
                }

                void BindChunkStore(
                        sepgraph::topology::SourceLocalChunkStore &store)
                {
                    m_chunk_store = &store;
                    const index_t nnodes = m_origin_graph.nnodes;
                    std::vector<host::vertex_sync_element> compat(nnodes + 1);
                    for (index_t source = 0; source < nnodes; ++source) {
                        const auto &descriptor = store.Descriptor(source);
                        compat[source].index = descriptor.slab_id ==
                                sepgraph::topology::SourceLocalChunkStore::kInvalidSlab
                            ? 0
                            : (static_cast<uint64_t>(descriptor.slab_id) << 32) |
                              static_cast<uint32_t>(descriptor.index);
                        compat[source].degree = descriptor.degree;
                    }
                    compat[nnodes] = {0, 0};
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.sync_vertices_, compat.data(),
                        sizeof(compat[0]) * (nnodes + 1), cudaMemcpyHostToDevice));
                    std::printf("[TOPOLOGY-MEMORY] full_extended_descriptor_bytes=0 recovered_bytes=%llu legacy_descriptor_bytes=%llu\n",
                        static_cast<unsigned long long>(
                            sizeof(sepgraph::topology::TopologyDescriptor) * nnodes),
                        static_cast<unsigned long long>(
                            sizeof(host::vertex_sync_element) * (nnodes + 1)));

                    std::vector<index_t *> mapped_slabs(store.SlabCount());
                    for (uint32_t slab = 0; slab < store.SlabCount(); ++slab) {
                        GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(
                            reinterpret_cast<void **>(&mapped_slabs[slab]),
                            const_cast<index_t *>(store.SlabData(slab)), 0));
                    }
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_chunk_slabs_device,
                        sizeof(index_t *) * mapped_slabs.size()));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_chunk_slabs_device,
                        mapped_slabs.data(), sizeof(index_t *) * mapped_slabs.size(),
                        cudaMemcpyHostToDevice));
                    m_dev_mirror.chunk_slabs_ = m_chunk_slabs_device;
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_cache_invalidations,
                        sizeof(unsigned long long)));
                }

                void ReserveSparsePublicationCapacity(size_t capacity,
                                                      bool reserve_audit)
                {
                    if (m_patch_capacity != 0 || m_patch_host != nullptr ||
                        m_patch_device != nullptr || m_digest_device != nullptr) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=publication_capacity_already_reserved\n");
                        std::abort();
                    }
                    m_patch_capacity = capacity;
                    if (capacity != 0) {
                        GROUTE_CUDA_CHECK(cudaHostAlloc(
                            reinterpret_cast<void **>(&m_patch_host),
                            sizeof(*m_patch_host) * capacity, cudaHostAllocDefault));
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_patch_device,
                            sizeof(*m_patch_device) * capacity));
                        if (reserve_audit) {
                            GROUTE_CUDA_CHECK(cudaMalloc(&m_digest_device,
                                sizeof(*m_digest_device) * capacity));
                        }
                    }
                    GROUTE_CUDA_CHECK(cudaEventCreate(&m_publication_begin));
                    GROUTE_CUDA_CHECK(cudaEventCreate(&m_publication_end));
                    std::printf("[PUBLICATION-RESERVE] patch_capacity_records=%llu pinned_patch_bytes=%llu device_patch_bytes=%llu device_digest_bytes=%llu audit=%u\n",
                        static_cast<unsigned long long>(capacity),
                        static_cast<unsigned long long>(
                            sizeof(*m_patch_host) * capacity),
                        static_cast<unsigned long long>(
                            sizeof(*m_patch_device) * capacity),
                        static_cast<unsigned long long>(
                            reserve_audit ? sizeof(*m_digest_device) * capacity : 0),
                        reserve_audit ? 1U : 0U);
                }

                // CC deletion-only executor reads chunks directly before the
                // merged delete/add epoch is published. All earlier readers
                
                // that bypasses cache for every changed source. Do not advance
                // epoch or update cache:
                // insertion still publishes the merged source set normally.
                const sepgraph::topology::TopologyPatchRecord *
                StageDeletionDescriptors(const std::vector<index_t> &sources,
                                               cudaStream_t stream) {
                    cgcomm::Scope comm_scope(cgcomm::Category::Topology);
                    if (m_publication_pending || sources.size() > m_patch_capacity ||
                        m_chunk_store == nullptr || m_chunk_store->IsPublished())
                        throw std::runtime_error("Invalid CC deletion descriptor staging");
                    for (size_t i = 0; i < sources.size(); ++i) {
                        const auto source = sources[i];
                        const auto &d = m_chunk_store->Descriptor(source);
                        m_patch_host[i] = {source, d.slab_id, d.index, d.degree, d.version};
                    }
                    if (!sources.empty()) {
                        GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_patch_device, m_patch_host,
                            sources.size() * sizeof(*m_patch_host), cudaMemcpyHostToDevice, stream));
                        ScatterTopologyPatch<<<(sources.size() + 255) / 256, 256, 0, stream>>>(
                            m_patch_device, sources.size(), m_dev_mirror.sync_vertices_);
                    }
                    return m_patch_device;
                }

                void PublishSparse(const std::vector<index_t> &sources,
                                   cudaStream_t stream,
                                   bool audit,
                                   index_t *cache_edges,
                                   uint64_t cache_capacity)
                {
                    cgcomm::Scope comm_scope(cgcomm::Category::Topology);
                    if (m_publication_pending) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=publication_already_pending\n");
                        std::abort();
                    }
                    if (m_chunk_store == nullptr || m_chunk_store->IsPublished() ||
                        m_chunk_store->PendingEpoch() !=
                            m_chunk_store->PublishedEpoch() + 1) {
                        std::fprintf(stderr, "[PUBLISH] protocol_error=invalid_pending_epoch pending=%llu published=%llu\n",
                            static_cast<unsigned long long>(
                                m_chunk_store == nullptr ? 0 : m_chunk_store->PendingEpoch()),
                            static_cast<unsigned long long>(
                                m_chunk_store == nullptr ? 0 : m_chunk_store->PublishedEpoch()));
                        std::abort();
                    }
                    for (size_t i = 0; i < sources.size(); ++i) {
                        if (sources[i] >= m_origin_graph.nnodes ||
                            (i != 0 && sources[i - 1] >= sources[i])) {
                            std::fprintf(stderr, "[PUBLISH] protocol_error=invalid_patch_sources index=%llu source=%u\n",
                                static_cast<unsigned long long>(i), sources[i]);
                            std::abort();
                        }
                        const auto &descriptor = m_chunk_store->Descriptor(sources[i]);
                        const bool invalid_slab = descriptor.slab_id ==
                            sepgraph::topology::SourceLocalChunkStore::kInvalidSlab;
                        if (descriptor.version != m_chunk_store->PendingEpoch() ||
                            (descriptor.degree != 0 && invalid_slab) ||
                            (!invalid_slab &&
                             (descriptor.slab_id >= m_chunk_store->SlabCount() ||
                              descriptor.index > std::numeric_limits<uint32_t>::max()))) {
                            std::fprintf(stderr, "[PUBLISH] protocol_error=invalid_authoritative_descriptor source=%u version=%u degree=%u slab=%u\n",
                                sources[i], descriptor.version, descriptor.degree,
                                descriptor.slab_id);
                            std::abort();
                        }
                    }
                    if (sources.size() > m_patch_capacity ||
                        (audit && sources.size() != 0 && m_digest_device == nullptr)) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=publication_capacity_exceeded records=%llu capacity=%llu audit=%u\n",
                            static_cast<unsigned long long>(sources.size()),
                            static_cast<unsigned long long>(m_patch_capacity),
                            audit ? 1U : 0U);
                        std::abort();
                    }
                    for (size_t i = 0; i < sources.size(); ++i) {
                        const index_t source = sources[i];
                        const auto &descriptor = m_chunk_store->Descriptor(source);
                        m_patch_host[i] = {source, descriptor.slab_id, descriptor.index,
                                           descriptor.degree, descriptor.version};
                    }
                    std::printf("[SPARSE-STAGING] patch_capacity_records=%llu patch_bytes=%llu digest_bytes=%llu total_bytes=%llu\n",
                        static_cast<unsigned long long>(m_patch_capacity),
                        static_cast<unsigned long long>(
                            sizeof(sepgraph::topology::TopologyPatchRecord) * m_patch_capacity),
                        static_cast<unsigned long long>(
                            m_digest_device == nullptr ? 0 :
                            sizeof(sepgraph::topology::TopologyDeviceDigest) * m_patch_capacity),
                        static_cast<unsigned long long>(
                            sizeof(sepgraph::topology::TopologyPatchRecord) *
                                m_patch_capacity +
                            (m_digest_device == nullptr ? 0 :
                             sizeof(sepgraph::topology::TopologyDeviceDigest) *
                                m_patch_capacity)));
                    m_publication_count = sources.size();
                    m_publication_audit = audit;
                    m_publication_pending = true;
                    GROUTE_CUDA_CHECK(cudaEventRecord(m_publication_begin, stream));
                    GROUTE_CUDA_CHECK(cudaMemsetAsync(m_cache_invalidations, 0,
                        sizeof(unsigned long long), stream));
                    if (m_publication_count != 0) {
                        GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_patch_device, m_patch_host,
                            sizeof(*m_patch_host) * m_publication_count,
                            cudaMemcpyHostToDevice, stream));
                        ScatterTopologyPatch<<<(m_publication_count + 255) / 256, 256, 0, stream>>>(
                            m_patch_device, m_publication_count,
                            m_dev_mirror.sync_vertices_);
                        PatchOrInvalidateCachedAdjacency<<<m_publication_count, 256, 0, stream>>>(
                            m_patch_device, m_publication_count, m_chunk_slabs_device,
                            cache_edges, cache_capacity, m_dev_mirror.river,
                            m_dev_mirror.vertices_, m_cache_invalidations);
                        if (audit) {
                            HashPublishedAdjacency<<<(m_publication_count + 255) / 256,
                                256, 0, stream>>>(m_patch_device, m_publication_count,
                                m_chunk_slabs_device, m_digest_device);
                        }
                    }
                    GROUTE_CUDA_CHECK(cudaEventRecord(m_publication_end, stream));
                    m_chunk_store->Publish();
                }

                void WaitForSparsePublication(cudaStream_t consumer_stream) const
                {
                    if (!m_publication_pending) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=no_pending_publication_wait\n");
                        std::abort();
                    }
                    GROUTE_CUDA_CHECK(cudaStreamWaitEvent(
                        consumer_stream, m_publication_end, 0));
                }

                SparsePublicationResult CompleteSparsePublication()
                {
                    if (!m_publication_pending) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=no_pending_publication_complete\n");
                        std::abort();
                    }
                    const cudaError_t ready = cudaEventQuery(m_publication_end);
                    if (ready != cudaSuccess) {
                        std::fprintf(stderr,
                            "[PUBLISH] protocol_error=completion_before_existing_sync cuda_error=%s\n",
                            cudaGetErrorString(ready));
                        std::abort();
                    }
                    SparsePublicationResult result;
                    float elapsed = 0.0f;
                    GROUTE_CUDA_CHECK(cudaEventElapsedTime(
                        &elapsed, m_publication_begin, m_publication_end));
                    result.publication_ms = elapsed;
                    GROUTE_CUDA_CHECK(cudaMemcpy(&result.cache_invalidations,
                        m_cache_invalidations, sizeof(result.cache_invalidations),
                        cudaMemcpyDeviceToHost));
                    index_t cache_tail = 0;
                    GROUTE_CUDA_CHECK(cudaMemcpy(&cache_tail, m_dev_mirror.river,
                        sizeof(cache_tail), cudaMemcpyDeviceToHost));
                    result.cache_tail = cache_tail;
                    if (m_publication_audit && m_publication_count != 0) {
                        std::vector<sepgraph::topology::TopologyDeviceDigest> gpu_digests(
                            m_publication_count);
                        GROUTE_CUDA_CHECK(cudaMemcpy(gpu_digests.data(), m_digest_device,
                            sizeof(gpu_digests[0]) * gpu_digests.size(),
                            cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < m_publication_count; ++i) {
                            const auto &descriptor = m_chunk_store->Descriptor(
                                m_patch_host[i].source);
                            if (descriptor.degree != gpu_digests[i].degree ||
                                m_chunk_store->OrderedHash(m_patch_host[i].source) !=
                                    gpu_digests[i].ordered_hash) {
                                ++result.hash_mismatches;
                            }
                        }
                    }
                    m_publication_pending = false;
                    m_publication_count = 0;
                    m_publication_audit = false;
                    return result;
                }

                const dev::PMAGraph &DeviceObject() const
                {
                    return m_dev_mirror;
                }

                // const dev::CacheGraph &DeviceObject() const

                const host::PMAGraph &HostObject() const
                {
                    return m_origin_graph;
                }

                void SwitchZC()
                {
                    m_dev_mirror.edges_ = m_dev_mirror.edge_dst_zc;
                    
                }

                void SwitchExp(index_t i)
                {
		            m_dev_mirror.edges_ = m_dev_mirror.edge_dst_exp[i];      
                }

                void SwitchCom()
                {
                    m_dev_mirror.edges_ = m_dev_mirror.edge_dst_com;      
                }

                void AllocateDevMirror_Edge_Explicit_Step(uint64_t seg_nedges, uint64_t seg_sedge,const groute::Stream &stream,index_t id)
                {
                    AllocateDevMirror_edge_explicit_step(seg_nedges, seg_sedge, stream,id);
                }

                void AllocateDevMirror_Edge_Compaction(uint64_t seg_nedges, const groute::Stream &stream)
                {
                    AllocateDevMirror_edge_compaction(seg_nedges,stream);
                }

                void AllocateDevMirror_Edge_Zero()
                {
                    AllocateDevMirror_edge_zero();
                    
                }

                void AllocateDatumObjects()
                {
                }

                template <typename TFirstGraphDatum, typename... TGraphDatum>
                void AllocateDatumObjects(TFirstGraphDatum &first_datum, TGraphDatum &... more_data)
                {
                    AllocateDatum(first_datum);
                    AllocateDatumObjects(more_data...);
                }

                template <typename TGraphDatum>
                void AllocateDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Allocate(m_origin_graph);
                }

                template <typename TGraphDatum>
                void GatherDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Gather(m_origin_graph);
                }
                

            // private:
                void AllocateDevMirror_node(uint64_t seg_nedges)
                {
                    index_t nnodes;
                    uint64_t nedges;
                    uint64_t elem_capacity = m_origin_graph.elem_capacity;
                    uint64_t elem_capacity_max = m_origin_graph.elem_capacity_max;
                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    m_dev_mirror.nedges = nedges = m_origin_graph.nedges;

                    if(FLAGS_hybrid != 0){
    		            for(index_t i = 0; i < FLAGS_n_stream; i++){
    			             GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst_exp[i], seg_nedges * sizeof(index_t))); 
    		            }
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.subgraph_activenode, (nnodes) * sizeof(uint32_t)));
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.subgraph_rowstart, (nnodes + 1) * sizeof(uint32_t)));
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst_com, nedges/4 * sizeof(index_t)));
                        GROUTE_CUDA_CHECK(cudaHostRegister((void *)m_origin_graph.subgraph_edgedst, nedges/4 * sizeof(index_t), cudaHostRegisterMapped));
		            }

                    
                    GROUTE_CUDA_CHECK(cudaHostRegister((void *)m_origin_graph.edges_, sizeof(index_t) * elem_capacity_max, cudaHostRegisterMapped));
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.vertices_, (nnodes + 1) * sizeof(host::vertex_element))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.vertices_, m_origin_graph.vertices_, (nnodes + 1) * sizeof(host::vertex_element),cudaMemcpyHostToDevice));
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.sync_vertices_, (nnodes + 1) * sizeof(host::vertex_sync_element))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.sync_vertices_, m_origin_graph.sync_vertices_, (nnodes + 1) * sizeof(host::vertex_sync_element),cudaMemcpyHostToDevice));
                    
                        
                    //     printf("vertices_:(%d)|%llu|%llu| \n", i, m_origin_graph.sync_vertices_[i].index
                    
                    
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.river,  sizeof(index_t)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.river, &m_origin_graph.river,  sizeof(index_t),cudaMemcpyHostToDevice));
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.river_low,  sizeof(index_t)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.river_low, &m_origin_graph.river_low,  sizeof(index_t),cudaMemcpyHostToDevice));
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.river_global,  sizeof(index_t)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.river_global, &m_origin_graph.river_global,  sizeof(index_t),cudaMemcpyHostToDevice));

                }

                void HostRiverElement(){
                    
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(&m_origin_graph.river_global, m_dev_mirror.river_global, 1 * sizeof(index_t),
                                   cudaMemcpyDeviceToHost));
                    this->m_origin_graph.river = this->m_origin_graph.river_global;
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.river, &m_origin_graph.river,  sizeof(index_t),cudaMemcpyHostToDevice));
                    this->m_origin_graph.river_global = 0;
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.river_global, &m_origin_graph.river_global, 1 * sizeof(index_t),
                                   cudaMemcpyHostToDevice));
                    
                }

                void AllocateDevMirror_node_update()
                {
                    index_t nnodes;
                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.sync_vertices_, m_origin_graph.sync_vertices_, (nnodes + 1) * sizeof(host::vertex_sync_element),
                                   cudaMemcpyHostToDevice));
                }

                void BackNode()
                {
                    index_t nnodes;
                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_origin_graph.vertices_, m_dev_mirror.vertices_, (nnodes + 1) * sizeof(host::vertex_element),
                                   cudaMemcpyDeviceToHost));
                }     

                void AllocateDevMirror_edge_zero()
                {
                    if (m_chunk_store == nullptr) {
                        GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_dev_mirror.edge_dst_zc, (void *)m_origin_graph.edges_, 0));
                    }
                }
                void AllocateDevMirror_weight_zero()
                {
                    GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_dev_mirror.weight_dst_zc, (void *)m_origin_graph.weights_, 0));
                }

                void AllocateDevMirror_edge_explicit_step(uint64_t seg_nedges, uint64_t seg_sedge, const groute::Stream &stream ,index_t i)
                {

		              GROUTE_CUDA_CHECK(
                            cudaMemcpyAsync(m_dev_mirror.edge_dst_exp[i], m_origin_graph.edges_ + seg_sedge,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice,stream.cuda_stream));  	    	      
                }

                void AllocateDevMirror_edge_compaction(uint64_t seg_nedges, const groute::Stream &stream)
                {
                      GROUTE_CUDA_CHECK(
                            cudaMemcpyAsync(m_dev_mirror.edge_dst_com, m_origin_graph.subgraph_edgedst,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice,stream.cuda_stream));                   
                }

                void DeallocateDevMirror()
                {
                    
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.vertices_));
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.sync_vertices_));
                    if (m_chunk_slabs_device != nullptr) cudaFree(m_chunk_slabs_device);
                    if (m_patch_host != nullptr) cudaFreeHost(m_patch_host);
                    if (m_patch_device != nullptr) cudaFree(m_patch_device);
                    if (m_digest_device != nullptr) cudaFree(m_digest_device);
                    if (m_cache_invalidations != nullptr) cudaFree(m_cache_invalidations);
                    if (m_publication_begin != nullptr) cudaEventDestroy(m_publication_begin);
                    if (m_publication_end != nullptr) cudaEventDestroy(m_publication_end);
                    m_dev_mirror.sync_vertices_ = nullptr;
                    m_dev_mirror.vertices_ = nullptr;
                    m_dev_mirror.edges_ = nullptr;
                }
            };

            /*
            * @brief A single GPU graph allocator (allocates a complete mirror graph at one GPU)
            */
            struct CSRGraphAllocator
            {
                typedef dev::CSRGraph DeviceObjectType;

            private:
                host::CSRGraph &m_origin_graph;
                dev::CSRGraph m_dev_mirror;
                bool m_on_pinned_memory;

            public:
                CSRGraphAllocator(host::CSRGraph &host_graph, uint64_t seg_nedges) : m_origin_graph(host_graph)
                {
                    AllocateDevMirror_node(seg_nedges);
                }

                ~CSRGraphAllocator()
                {
                    DeallocateDevMirror();
                }

                const dev::CSRGraph &DeviceObject() const
                {
                    return m_dev_mirror;
                }

                const host::CSRGraph &HostObject() const
                {
                    return m_origin_graph;
                }

                void SwitchZC()
                {
                    m_dev_mirror.edge_dst = m_dev_mirror.edge_dst_zc;
                }

                void SwitchExp(index_t i)
                {
		            m_dev_mirror.edge_dst = m_dev_mirror.edge_dst_exp[i];      
                }

                void SwitchCom()
                {
                    m_dev_mirror.edge_dst = m_dev_mirror.edge_dst_com;      
                }

                void AllocateDevMirror_Edge_Explicit_Step(uint64_t seg_nedges, uint64_t seg_sedge,const groute::Stream &stream,index_t id)
                {
                    AllocateDevMirror_edge_explicit_step(seg_nedges, seg_sedge, stream,id);
                }

                void AllocateDevMirror_Edge_Compaction(uint64_t seg_nedges, const groute::Stream &stream)
                {
                    AllocateDevMirror_edge_compaction(seg_nedges,stream);
                }

                void AllocateDevMirror_Edge_Zero()
                {
                    AllocateDevMirror_edge_zero();
                }

                void AllocateDatumObjects()
                {
                }

                template <typename TFirstGraphDatum, typename... TGraphDatum>
                void AllocateDatumObjects(TFirstGraphDatum &first_datum, TGraphDatum &... more_data)
                {
                    AllocateDatum(first_datum);
                    AllocateDatumObjects(more_data...);
                }

                template <typename TGraphDatum>
                void AllocateDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Allocate(m_origin_graph);
                }

                template <typename TGraphDatum>
                void GatherDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Gather(m_origin_graph);
                }

            private:
                void AllocateDevMirror_node(uint64_t seg_nedges)
                {
                    index_t nnodes;
                    uint64_t nedges;

                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    m_dev_mirror.nedges = nedges = m_origin_graph.nedges;

                    if(FLAGS_hybrid != 0){
    		            for(index_t i = 0; i < FLAGS_n_stream; i++){
    			             GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst_exp[i], seg_nedges * sizeof(index_t))); 
    		            }
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.subgraph_activenode, (nnodes) * sizeof(uint32_t)));
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.subgraph_rowstart, (nnodes + 1) * sizeof(uint32_t)));
                        GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst_com, nedges/4 * sizeof(uint32_t)));
                        GROUTE_CUDA_CHECK(cudaHostRegister((void *)m_origin_graph.subgraph_edgedst, nedges/4 * sizeof(index_t), cudaHostRegisterMapped));
		            }
                    GROUTE_CUDA_CHECK(cudaHostRegister((void *)m_origin_graph.edge_dst, sizeof(index_t) * nedges, cudaHostRegisterMapped));

                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.row_start, (nnodes + 1) * sizeof(uint64_t))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.row_start, m_origin_graph.row_start, (nnodes + 1) * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice));

                }
                void AllocateDevMirror_edge_zero()
                {
                    GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_dev_mirror.edge_dst_zc, (void *)m_origin_graph.edge_dst, 0));
                }

                void AllocateDevMirror_edge_explicit_step(uint64_t seg_nedges, uint64_t seg_sedge, const groute::Stream &stream ,index_t i)
                {
		              GROUTE_CUDA_CHECK(
                            cudaMemcpyAsync(m_dev_mirror.edge_dst_exp[i], m_origin_graph.edge_dst + seg_sedge,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice,stream.cuda_stream));  	    	      
                }

                void AllocateDevMirror_edge_compaction(uint64_t seg_nedges, const groute::Stream &stream)
                {
                      GROUTE_CUDA_CHECK(
                            cudaMemcpyAsync(m_dev_mirror.edge_dst_com, m_origin_graph.subgraph_edgedst,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice,stream.cuda_stream));                   
                }

                void DeallocateDevMirror()
                {
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.row_start));

                    m_dev_mirror.row_start = nullptr;
                    m_dev_mirror.edge_dst = nullptr;
                }
            };

            /*
            * @brief A single GPU graph allocator (allocates a complete mirror graph at one GPU)
            */
            struct CSCGraphAllocator
            {
                typedef dev::CSCGraph DeviceObjectType;

            private:
                bool m_undirected;
                host::CSCGraph &m_origin_graph;
                dev::CSCGraph m_dev_mirror;
                bool m_on_pinned_memory;

            public:
                explicit CSCGraphAllocator(host::CSCGraph &host_graph, index_t seg_nedges, bool OnPinnedMemory = true) : m_undirected(false),
                                                                                                                         m_origin_graph(host_graph),
                                                                                                                         m_dev_mirror(false),
                                                                                                                         m_on_pinned_memory(OnPinnedMemory)
                {
                    AllocateDevMirror_node(seg_nedges);
                }

                /**
                 * Construct a CSCGraphAllocator object by CSR graph, only available for undirected graph
                     * @param dev_csr_graph
                 */
                explicit CSCGraphAllocator(host::CSCGraph &host_graph,
                                           dev::CSRGraph &dev_csr_graph) : m_undirected(true),
                                                                           m_origin_graph(host_graph),
                                                                           m_dev_mirror(true)
                {
                    m_dev_mirror.nnodes = dev_csr_graph.nnodes;
                    m_dev_mirror.nedges = dev_csr_graph.nedges;
                    
                    
                    m_dev_mirror.edge_source = dev_csr_graph.edge_dst;
                    m_dev_mirror.edge_weights = dev_csr_graph.edge_weights;
                    m_dev_mirror.out_dgr = nullptr;
                }

                ~CSCGraphAllocator()
                {
                    if (!m_undirected)
                    {
                        DeallocateDevMirror();
                    }
                }

                const DeviceObjectType &DeviceObject() const
                {
                    return m_dev_mirror;
                }

                const host::CSCGraph &HostObject() const
                {
                    return m_origin_graph;
                }

                void AllocateDatumObjects()
                {
                }

                void AllocateDevMirror_Edge_Zero(index_t *edge_source_csc)
                {

                    AllocateDevMirror_edge_zero(edge_source_csc);
                }
                void SwitchZC()
                {
                    m_dev_mirror.edge_source = m_dev_mirror.edge_source_zc;
                }
                void SwitchExp()
                {
                    m_dev_mirror.edge_source = m_dev_mirror.edge_source_exp;
                }

                void AllocateDevMirror_Edge_Explicit_Step(uint64_t seg_nedges, uint64_t seg_sedge, index_t *edge_source_csc)
                {

                    AllocateDevMirror_edge_explicit_step(seg_nedges, seg_sedge, edge_source_csc);
                }
                void DeallocateDevMirror_Edge()
                {
                    DeallocateDevMirror_edge();
                }
                void DeallocateDevMirror_Edge_Zero()
                {
                    DeallocateDevMirror_edge_zero();
                }
                template <typename TFirstGraphDatum, typename... TGraphDatum>
                void AllocateDatumObjects(TFirstGraphDatum &first_datum, TGraphDatum &... more_data)
                {
                    AllocateDatum(first_datum);
                    AllocateDatumObjects(more_data...);
                }

                template <typename TGraphDatum>
                void AllocateDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Allocate(m_origin_graph);
                }

                template <typename TGraphDatum>
                void GatherDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.Gather(m_origin_graph);
                }

            private:
                void AllocateDevMirror_node(uint64_t seg_nedges)
                {
                    index_t nnodes;

                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    m_dev_mirror.edge_source = nullptr;
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_source_exp, seg_nedges * sizeof(index_t)));
		    
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.col_start, (nnodes + 1) * sizeof(index_t))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.col_start, m_origin_graph.col_start, (nnodes + 1) * sizeof(index_t),
                                   cudaMemcpyHostToDevice));

		    
                    //copy out-degree for CSC format.
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.out_dgr, nnodes * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.out_dgr, m_origin_graph.out_dgr, nnodes * sizeof(index_t),
                                   cudaMemcpyHostToDevice));
                }

                void AllocateDevMirror_edge_zero(index_t *edge_source_csc)
                {
                    GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_dev_mirror.edge_source_zc, (void *)edge_source_csc, 0));
                }

                void AllocateDevMirror_edge_explicit_step(uint64_t seg_nedges, uint64_t seg_sedge, index_t *edge_source_csc)
                {
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_mirror.edge_source_exp, edge_source_csc + seg_sedge,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice));
                }

                void DeallocateDevMirror_edge()
                {
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.edge_source_exp));
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.out_dgr));
                }
                void DeallocateDevMirror_edge_zero()
                {
                    GROUTE_CUDA_CHECK(cudaFreeHost(m_dev_mirror.edge_source_zc));
                    GROUTE_CUDA_CHECK(cudaFreeHost(m_dev_mirror.out_dgr));
                }
                void DeallocateDevMirror()
                {
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.col_start));
                    if (!m_on_pinned_memory)
                        GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.edge_source_exp));
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.out_dgr));

                    m_dev_mirror.col_start = nullptr;
                    m_dev_mirror.edge_source = nullptr;
                    m_dev_mirror.out_dgr = nullptr;
                }

            };
            

            template <typename T>
            class EdgeInputDatum
            {
            public:
                typedef dev::GraphDatum<T> DeviceObjectType;
                T *m_edge_data;
            private:
                
                std::vector<T> m_ones;

                DeviceObjectType m_dev_datum;
                bool m_on_pinned_memory;

            public:
                EdgeInputDatum()
                {
                }

                EdgeInputDatum(PMAGraphAllocator &graph_allocator, bool OnPinnedMemory = false) : m_on_pinned_memory(OnPinnedMemory)
                {
                    // Will call this->SetNumSegs and this->AllocateDevSeg with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                EdgeInputDatum(EdgeInputDatum &&other) : m_edge_data(other.m_edge_data),
                                                         m_ones(std::move(other.m_ones)),
                                                         m_dev_datum(other.m_dev_datum),
                                                         m_on_pinned_memory(other.m_on_pinned_memory)

                {
                    other.m_edge_data = nullptr;
                    other.m_dev_datum.data_ptr = nullptr;
                    other.m_dev_datum.size = 0;
                }

                EdgeInputDatum &operator=(const EdgeInputDatum &other) = delete;

                EdgeInputDatum &operator=(EdgeInputDatum &&other) = delete;

                ~EdgeInputDatum()
                {
                    Deallocate();
                }

                void Deallocate()
                {

                    m_dev_datum.data_ptr = nullptr;
                    m_dev_datum.size = 0;
                }

                void SwitchZC()
                {
		              m_dev_datum.data_ptr = m_dev_datum.data_ptr_zc;
                }

                void SwitchExp(index_t i)
                {
		              m_dev_datum.data_ptr = m_dev_datum.data_ptr_exp[i];
                }
                
                void SwitchCom()
                {
                      m_dev_datum.data_ptr = m_dev_datum.data_ptr_com;
                }

                // Ideally, when NoWeight type is used, this branch will not be executed
                void Allocate_node(const groute::graphs::host::PMAGraph &host_graph,uint64_t edge_max)
                {
                    Deallocate();

                    assert(typeid(T) != typeid(TNoWeight));

                    if (host_graph.weights_ == nullptr)
                    {
                        printf("\nWarning: Expecting edge weights, falling back to all one's weights (use gen_weights and gen_weight_range).\n\n");

                        if (m_on_pinned_memory)
                        {
                            m_ones = std::vector<T>(host_graph.nedges, 1);
                            m_edge_data = m_ones.data();
                        }
                        else
                        {
                            m_ones = std::vector<T>(host_graph.nedges, 1);
                            m_edge_data = m_ones.data();
                        }
                    }
                    else
                    {

                        m_edge_data = reinterpret_cast<T *>(host_graph.weights_); // Bind to edge_weights from the original graph

                    }
			 
                        GROUTE_CUDA_CHECK(cudaHostRegister((void *)m_edge_data, sizeof(T) * host_graph.elem_capacity, cudaHostRegisterMapped));
                        GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_dev_datum.data_ptr_zc, (void *)m_edge_data, 0));
  
                     if(FLAGS_hybrid != 0){
    			         for(index_t i = 0; i < FLAGS_n_stream; i++){
    			             GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_datum.data_ptr_exp[i], edge_max * sizeof(T))); 
    			         }
                         GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_datum.data_ptr_com, host_graph.nedges/4 * sizeof(T))); 
                     }
                        m_dev_datum.size = edge_max;
 
                }
                
		        void AllocateDevMirror_edge_explicit_step(const groute::graphs::host::PMAGraph &host_graph,uint64_t seg_nedges, uint64_t seg_sedge,const groute::Stream &stream ,index_t i)
                {
    		        GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_dev_datum.data_ptr_exp[i], m_edge_data + seg_sedge, seg_nedges * sizeof(T),
                                                         cudaMemcpyHostToDevice,stream.cuda_stream));

                }

                void AllocateDevMirror_edge_compaction(const host::PMAGraph &host_graph, uint32_t seg_nedges,const groute::Stream &stream)
                {
                    GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_dev_datum.data_ptr_com, host_graph.subgraph_edgeweight, seg_nedges * sizeof(T),
                                                         cudaMemcpyHostToDevice,stream.cuda_stream));
                }

                T *GetHostDataPtr()
                {
                    return m_edge_data;
                }

                const DeviceObjectType &DeviceObject() const
                {
                    return m_dev_datum;
                }
            };

            template <typename T>
            class NodeOutputDatum
            {
            public:
                typedef dev::GraphDatum<T> DeviceObjectType;

            private:
                std::vector<T> m_host_data;
                DeviceObjectType m_dev_datum;

            public:
                NodeOutputDatum()
                {
                }

                NodeOutputDatum(PMAGraphAllocator &graph_allocator)
                {
                    // Will call this->Allocate with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                NodeOutputDatum(NodeOutputDatum &&other) : m_host_data(std::move(other.m_host_data)),
                                                           m_dev_datum(other.m_dev_datum)
                {
                    other.m_dev_datum.data_ptr = nullptr;
                    other.m_dev_datum.size = 0;
                }

                NodeOutputDatum &operator=(const NodeOutputDatum &other) = delete;

                NodeOutputDatum &operator=(NodeOutputDatum &&other) = delete;

                ~NodeOutputDatum()
                {
                    Deallocate();
                }

                void Deallocate()
                {
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_datum.data_ptr));

                    m_dev_datum.data_ptr = nullptr;
                    m_dev_datum.size = 0;
                }

                void Allocate(const graphs::host::PMAGraph &host_graph)
                {
                    Deallocate();

                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_datum.data_ptr, host_graph.nnodes * sizeof(T)));
                    m_dev_datum.size = host_graph.nnodes;
                }

                void Gather(const graphs::host::PMAGraph &host_graph)
                {
                    m_host_data.resize(host_graph.nnodes);

                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        &m_host_data[0], m_dev_datum.data_ptr,
                        host_graph.nnodes * sizeof(T), cudaMemcpyDeviceToHost));
                }

                void Gather()
                {
                    m_host_data.resize(m_dev_datum.size);

                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        &m_host_data[0], m_dev_datum.data_ptr,
                        m_dev_datum.size * sizeof(T), cudaMemcpyDeviceToHost));
                }

                void Swap(NodeOutputDatum &other)
                {
                    std::swap(m_host_data, other.m_host_data);
                    std::swap(m_dev_datum, other.m_dev_datum);
                }

                const std::vector<T> &GetHostData()
                {
                    return m_host_data;
                }

                const DeviceObjectType &DeviceObject() const
                {
                    return m_dev_datum;
                }
            };
        
            // template <typename T>
            // struct DependencyData{

            //     DependencyData() : level(), value(), parent() {}
            

        } // namespace single

        namespace multi
        {
            struct GraphPartitioner
            {
                virtual ~GraphPartitioner()
                {
                }

                virtual host::CSRGraph &GetOriginGraph() = 0;

                virtual host::CSRGraph &GetPartitionedGraph() = 0;

                virtual void GetSegIndices(
                    int seg_idx,
                    index_t &seg_snode, index_t &seg_nnodes,
                    index_t &seg_sedge, index_t &seg_nedges) const = 0;

                virtual bool NeedsReverseLookup() = 0;

                /// Maps a node from the original index space to the new partitioned index space
                virtual index_t ReverseLookup(index_t node) = 0;
            };

            class RandomPartitioner : public GraphPartitioner
            {
                host::CSRGraph &m_origin_graph;
                int m_nsegs;

            public:
                RandomPartitioner(host::CSRGraph &origin_graph, int nsegs) : m_origin_graph(origin_graph),
                                                                             m_nsegs(nsegs)
                {
                    assert(nsegs >= 1);
                }

                host::CSRGraph &GetOriginGraph() override
                {
                    return m_origin_graph;
                }

                host::CSRGraph &GetPartitionedGraph() override
                {
                    return m_origin_graph;
                }

                void GetSegIndices(
                    int seg_idx,
                    index_t &seg_snode, index_t &seg_nnodes,
                    index_t &seg_sedge, index_t &seg_nedges) const override
                {
                    index_t seg_enode, seg_eedge;

                    seg_nnodes = round_up(m_origin_graph.nnodes,
                                          m_nsegs);                                       // general nodes seg size
                    seg_snode = seg_nnodes * seg_idx;                                     // start node
                    seg_nnodes = std::min(m_origin_graph.nnodes - seg_snode, seg_nnodes); // fix for last seg case
                    seg_enode = seg_snode + seg_nnodes;                                   // end node
                    seg_sedge = m_origin_graph.row_start[seg_snode];                      // start edge
                    seg_eedge = m_origin_graph.row_start[seg_enode];                      // end edge
                    seg_nedges = seg_eedge - seg_sedge;
                }

                bool NeedsReverseLookup() override
                {
                    return false;
                }

                index_t ReverseLookup(index_t node) override
                {
                    return node;
                }
            };

            class MetisPartitioner : public GraphPartitioner
            {
                host::CSRGraph &m_origin_graph;
                host::CSRGraph m_partitioned_graph;
                std::vector<index_t> m_reverse_lookup;
                std::vector<index_t> m_seg_offsets;
                int m_nsegs;

            public:
                MetisPartitioner(host::CSRGraph &origin_graph, int nsegs);

                host::CSRGraph &GetOriginGraph() override
                {
                    return m_origin_graph;
                }

                host::CSRGraph &GetPartitionedGraph() override
                {
                    return m_partitioned_graph;
                }

                void GetSegIndices(
                    int seg_idx,
                    index_t &seg_snode, index_t &seg_nnodes,
                    index_t &seg_sedge, index_t &seg_nedges) const override;

                bool NeedsReverseLookup() override
                {
                    return true;
                }

                index_t ReverseLookup(index_t node) override;
            };

            /*
            * @brief A multi-GPU graph segment allocator (allocates a graph segment over each GPU)
            */
            struct CSRGraphAllocator
            {
                typedef dev::CSRGraphSeg DeviceObjectType;

            private:
                groute::Context &m_context;
                std::unique_ptr<GraphPartitioner> m_partitioner;

                int m_ngpus;

                std::vector<dev::CSRGraphSeg> m_dev_segs;

            public:
                CSRGraphAllocator(groute::Context &context, host::CSRGraph &host_graph, int ngpus, bool metis_pn) : m_context(context), m_ngpus(ngpus)
                {
                    m_partitioner = metis_pn && (m_ngpus > 1)
                                        ? (std::unique_ptr<GraphPartitioner>)std::unique_ptr<MetisPartitioner>(
                                              new MetisPartitioner(host_graph, ngpus))
                                        : (std::unique_ptr<GraphPartitioner>)std::unique_ptr<RandomPartitioner>(
                                              new RandomPartitioner(host_graph, ngpus));

                    m_dev_segs.resize(m_ngpus);

                    for (int i = 0; i < m_ngpus; i++)
                    {
                        m_context.SetDevice(i);
                        AllocateDevSeg(m_ngpus, i, m_dev_segs[i]);
                    }
                }

                ~CSRGraphAllocator()
                {
                    for (auto &seg : m_dev_segs)
                    {
                        DeallocateDevSeg(seg);
                    }
                }

                GraphPartitioner *GetGraphPartitioner() const
                {
                    return m_partitioner.get();
                }

                const std::vector<dev::CSRGraphSeg> &GetDeviceObjects() const
                {
                    return m_dev_segs;
                }

                void AllocateDatumObjects()
                {
                }

                template <typename TFirstGraphDatum, typename... TGraphDatum>
                void AllocateDatumObjects(TFirstGraphDatum &first_datum, TGraphDatum &... more_data)
                {
                    AllocateDatum(first_datum);
                    AllocateDatumObjects(more_data...);
                }

                template <typename TGraphDatum>
                void AllocateDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.PrepareAllocate(m_ngpus, m_partitioner.get());

                    for (int i = 0; i < m_ngpus; i++)
                    {
                        index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                        m_partitioner->GetSegIndices(i, seg_snode, seg_nnodes, seg_sedge, seg_nedges);

                        m_context.SetDevice(i);
                        graph_datum.AllocateSeg(i, m_partitioner->GetPartitionedGraph().nnodes,
                                                m_partitioner->GetPartitionedGraph().nedges,
                                                seg_snode, seg_nnodes, seg_sedge, seg_nedges);
                    }
                }

                template <typename TGraphDatum>
                void GatherDatum(TGraphDatum &graph_datum)
                {
                    graph_datum.PrepareGather(m_ngpus, m_partitioner->GetPartitionedGraph());

                    for (int i = 0; i < m_ngpus; i++)
                    {
                        index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                        m_partitioner->GetSegIndices(i, seg_snode, seg_nnodes, seg_sedge, seg_nedges);

                        m_context.SetDevice(i);
                        graph_datum.GatherSeg(i, m_partitioner->GetPartitionedGraph().nnodes,
                                              m_partitioner->GetPartitionedGraph().nedges, seg_snode,
                                              seg_nnodes, seg_sedge, seg_nedges);
                    }

                    if (m_partitioner->NeedsReverseLookup())
                        graph_datum.FinishGather([this](index_t n) { return m_partitioner->ReverseLookup(n); });
                }

            private:
                void AllocateDevSeg(int nsegs, int seg_idx, dev::CSRGraphSeg &graph_seg) const
                {
                    graph_seg.seg_idx = seg_idx;
                    graph_seg.nsegs = nsegs;

                    index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                    m_partitioner->GetSegIndices(seg_idx, seg_snode, seg_nnodes, seg_sedge, seg_nedges);

                    graph_seg.nnodes = m_partitioner->GetPartitionedGraph().nnodes;
                    graph_seg.nedges = m_partitioner->GetPartitionedGraph().nedges;
                    graph_seg.nodes_offset = seg_snode;
                    graph_seg.edges_offset = seg_sedge;
                    graph_seg.nnodes_local = seg_nnodes;
                    graph_seg.nedges_local = seg_nedges;

                    GROUTE_CUDA_CHECK(cudaMalloc(&graph_seg.row_start,
                                                 (seg_nnodes + 1) *
                                                     sizeof(index_t))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(graph_seg.row_start, m_partitioner->GetPartitionedGraph().row_start + seg_snode,
                                   (seg_nnodes + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

                    GROUTE_CUDA_CHECK(cudaMalloc(&graph_seg.edge_dst, seg_nedges * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(graph_seg.edge_dst, m_partitioner->GetPartitionedGraph().edge_dst + seg_sedge,
                                   seg_nedges * sizeof(index_t),
                                   cudaMemcpyHostToDevice));
                }

                void DeallocateDevSeg(dev::CSRGraphSeg &graph_seg) const
                {
                    GROUTE_CUDA_CHECK(cudaFree(graph_seg.row_start));
                    GROUTE_CUDA_CHECK(cudaFree(graph_seg.edge_dst));

                    graph_seg.row_start = nullptr;
                    graph_seg.edge_dst = nullptr;

                    graph_seg.seg_idx = 0;
                    graph_seg.nsegs = 0;

                    graph_seg.nnodes = 0;
                    graph_seg.nedges = 0;
                    graph_seg.nodes_offset = 0;
                    graph_seg.edges_offset = 0;
                    graph_seg.nnodes_local = 0;
                    graph_seg.nedges_local = 0;
                }
            };

            template <typename T>
            class EdgeInputDatum
            {
            public:
                typedef dev::GraphDatumSeg<T> DeviceObjectType; // edges are scattered, so we need seg objects

            private:
                T *m_origin_data;
                T *m_partitioned_data;

                std::vector<T> m_ones;

                std::vector<DeviceObjectType> m_dev_segs;

            public:
                EdgeInputDatum() : m_origin_data(nullptr), m_partitioned_data(nullptr)
                {
                }

                EdgeInputDatum(CSRGraphAllocator &graph_allocator)
                // EdgeInputDatum(PMAGraphAllocator &graph_allocator)
                {
                    // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                ~EdgeInputDatum()
                {
                    DeallocateDevSegs();
                }

                void PrepareAllocate(int nsegs, GraphPartitioner *partitioner)
                {
                    DeallocateDevSegs();

                    if (partitioner->GetOriginGraph().edge_weights == nullptr ||
                        partitioner->GetPartitionedGraph().edge_weights == nullptr)
                    {
                        assert(partitioner->GetOriginGraph().edge_weights == nullptr);
                        assert(partitioner->GetPartitionedGraph().edge_weights == nullptr);

                        printf("\nWarning: Expecting edge weights, falling back to all one's weights (use gen_weights and gen_weight_range).\n\n");

                        m_ones = std::vector<T>(partitioner->GetOriginGraph().nedges, 1);
                        m_origin_data = m_ones.data(); // since data is all one's we can use the same for both origin and partitioned
                        m_partitioned_data = m_ones.data();
                    }
                    else
                    {
                        m_origin_data = partitioner->GetOriginGraph().edge_weights;           // Bind to edge_weights from the original graph
                        m_partitioned_data = partitioner->GetPartitionedGraph().edge_weights; // Bind to edge_weights from the partitioned graph
                    }

                    m_dev_segs.resize(nsegs);
                }

                void AllocateSeg(int seg_idx,
                                 index_t nnodes, index_t nedges,
                                 index_t seg_snode, index_t seg_nnodes,
                                 index_t seg_sedge, index_t seg_nedges)
                {
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, seg_nedges * sizeof(T)));
                    GROUTE_CUDA_CHECK(
                        cudaMemcpy(m_dev_segs[seg_idx].data_ptr, m_partitioned_data + seg_sedge,
                                   seg_nedges * sizeof(T), cudaMemcpyHostToDevice));

                    m_dev_segs[seg_idx].offset = seg_sedge;
                    m_dev_segs[seg_idx].size = seg_nedges;
                }

                T *GetHostDataPtr()
                {
                    return m_origin_data;
                }

                const std::vector<DeviceObjectType> &GetDeviceObjects() const
                {
                    return m_dev_segs;
                }

            protected:
                void DeallocateDevSegs()
                {
                    for (auto &seg : m_dev_segs)
                    {
                        DeallocateDevSeg(seg);
                    }
                    m_dev_segs.clear();
                }

                void DeallocateDevSeg(DeviceObjectType &datum_seg)
                {
                    GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                    datum_seg.data_ptr = nullptr;
                    datum_seg.offset = 0;
                    datum_seg.size = 0;
                }
            };

            std::vector<index_t> GetUniqueHalos(
                const index_t *edge_dst,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges, int &halos_counter);

            class HalosDatum
            {
            public:
                typedef dev::GraphDatum<index_t> DeviceObjectType;

            private:
                const host::CSRGraph *m_host_graph;
                std::vector<DeviceObjectType> m_dev_segs;

            public:
                HalosDatum() : m_host_graph(nullptr)
                {
                }

                HalosDatum(CSRGraphAllocator &graph_allocator) : m_host_graph(nullptr)
                {
                    // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                ~HalosDatum()
                {
                    DeallocateDevSegs();
                }

                void PrepareAllocate(int nsegs, GraphPartitioner *partitioner)
                {
                    m_host_graph = &partitioner->GetPartitionedGraph(); // cache the graph instance

                    DeallocateDevSegs();
                    m_dev_segs.resize(nsegs);
                }

                void AllocateSeg(int seg_idx,
                                 index_t nnodes, index_t nedges,
                                 index_t seg_snode, index_t seg_nnodes,
                                 index_t seg_sedge, index_t seg_nedges)
                {
                    assert(m_host_graph);

                    int halos_counter = 0;
                    std::vector<index_t> halos_vec = GetUniqueHalos(m_host_graph->edge_dst, seg_snode, seg_nnodes, seg_sedge, seg_nedges,
                                                                    halos_counter);

                    //printf(
                    //    "Halo stats -> seg: %d, seg nodes: %d, seg edges: %d, halos: %d, unique halos: %llu\n",
                    

                    if (halos_vec.size() == 0)
                    {
                        m_dev_segs[seg_idx].data_ptr = nullptr;
                    }
                    else
                    {
                        GROUTE_CUDA_CHECK(
                            cudaMalloc(&m_dev_segs[seg_idx].data_ptr, halos_vec.size() * sizeof(index_t)));
                        GROUTE_CUDA_CHECK(
                            cudaMemcpy(m_dev_segs[seg_idx].data_ptr, &halos_vec[0],
                                       halos_vec.size() * sizeof(index_t), cudaMemcpyHostToDevice));
                    }

                    m_dev_segs[seg_idx].size = halos_vec.size();
                }

                const std::vector<DeviceObjectType> &GetDeviceObjects() const
                {
                    return m_dev_segs;
                }

            protected:
                void DeallocateDevSegs()
                {
                    for (auto &seg : m_dev_segs)
                    {
                        DeallocateDevSeg(seg);
                    }
                    m_dev_segs.clear();
                }

                void DeallocateDevSeg(DeviceObjectType &datum_seg)
                {
                    GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                    datum_seg.data_ptr = nullptr;
                    datum_seg.size = 0;
                }
            };

            /*
            * @brief A node data array with global allocation for each device
            * and ownership over owned nodes data
            * Date is gathered to host from the owned segment of each device
            */
            template <typename T>
            class NodeOutputGlobalDatum
            {
            public:
                typedef dev::GraphDatum<T> DeviceObjectType;
                // no need for dev::GraphDatumSeg because data is allocated globally for each device

            private:
                std::vector<T> m_host_data;
                std::vector<DeviceObjectType> m_dev_segs;

            public:
                NodeOutputGlobalDatum()
                {
                }

                NodeOutputGlobalDatum(CSRGraphAllocator &graph_allocator)
                {
                    // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                ~NodeOutputGlobalDatum()
                {
                    DeallocateDevSegs();
                }

                void PrepareAllocate(int nsegs, GraphPartitioner *partitioner)
                {
                    DeallocateDevSegs();
                    m_dev_segs.resize(nsegs);
                }

                void AllocateSeg(int seg_idx,
                                 index_t nnodes, index_t nedges,
                                 index_t seg_snode, index_t seg_nnodes,
                                 index_t seg_sedge, index_t seg_nedges)
                {
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, nnodes * sizeof(T)));
                    m_dev_segs[seg_idx].size = nnodes;
                }

                void PrepareGather(int nsegs, const host::CSRGraph &host_graph)
                {
                    m_host_data.resize(host_graph.nnodes);
                }

                void GatherSeg(int seg_idx,
                               index_t nnodes, index_t nedges,
                               index_t seg_snode, index_t seg_nnodes,
                               index_t seg_sedge, index_t seg_nedges)
                {
                    DeviceObjectType &datum_seg = m_dev_segs[seg_idx];

                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        &m_host_data[seg_snode], datum_seg.data_ptr + seg_snode,
                        seg_nnodes * sizeof(T), cudaMemcpyDeviceToHost));
                }

                void FinishGather(const std::function<index_t(index_t)> &reverse_lookup)
                {
                    std::vector<T> temp_data(m_host_data.size());

                    for (int i = 0; i < m_host_data.size(); i++)
                    {
                        temp_data[i] = m_host_data[reverse_lookup(i)];
                    }

                    m_host_data = std::move(temp_data);
                }

                const std::vector<T> &GetHostData()
                {
                    return m_host_data;
                }

                const std::vector<DeviceObjectType> &GetDeviceObjects() const
                {
                    return m_dev_segs;
                }

            protected:
                void DeallocateDevSegs()
                {
                    for (auto &seg : m_dev_segs)
                    {
                        DeallocateDevSeg(seg);
                    }
                    m_dev_segs.clear();
                }

                void DeallocateDevSeg(DeviceObjectType &datum_seg)
                {
                    GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                    datum_seg.data_ptr = nullptr;
                    datum_seg.size = 0;
                }
            };

            /*
            * @brief A node data array with local allocation for each device
            * Each device can read/write only from/to its local nodes
            * Data is gathered to host from each segment of each device
            */
            template <typename T>
            class NodeOutputLocalDatum
            {
            public:
                typedef dev::GraphDatumSeg<T> DeviceObjectType;

            private:
                std::vector<T> m_host_data;
                std::vector<DeviceObjectType> m_dev_segs;

            public:
                NodeOutputLocalDatum()
                {
                }

                NodeOutputLocalDatum(CSRGraphAllocator &graph_allocator)
                {
                    // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context
                    graph_allocator.AllocateDatum(*this);
                }

                ~NodeOutputLocalDatum()
                {
                    DeallocateDevSegs();
                }

                void PrepareAllocate(int nsegs, GraphPartitioner *partitioner)
                {
                    DeallocateDevSegs();
                    m_dev_segs.resize(nsegs);
                }

                void AllocateSeg(int seg_idx,
                                 index_t nnodes, index_t nedges,
                                 index_t seg_snode, index_t seg_nnodes,
                                 index_t seg_sedge, index_t seg_nedges)
                {
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, seg_nnodes * sizeof(T)));

                    m_dev_segs[seg_idx].offset = seg_snode;
                    m_dev_segs[seg_idx].size = seg_nnodes;
                }

                void PrepareGather(int nsegs, const host::CSRGraph &host_graph)
                {
                    m_host_data.resize(host_graph.nnodes);
                }

                void GatherSeg(int seg_idx,
                               index_t nnodes, index_t nedges,
                               index_t seg_snode, index_t seg_nnodes,
                               index_t seg_sedge, index_t seg_nedges)
                {
                    DeviceObjectType &datum_seg = m_dev_segs[seg_idx];

                    GROUTE_CUDA_CHECK(cudaMemcpy(
                        &m_host_data[seg_snode], datum_seg.data_ptr,
                        seg_nnodes * sizeof(T), cudaMemcpyDeviceToHost));
                }

                void FinishGather(const std::function<index_t(index_t)> &reverse_lookup)
                {
                    std::vector<T> temp_data(m_host_data.size());

                    for (int i = 0; i < m_host_data.size(); i++)
                    {
                        temp_data[i] = m_host_data[reverse_lookup(i)];
                    }

                    m_host_data = std::move(temp_data);
                }

                const std::vector<T> &GetHostData()
                {
                    return m_host_data;
                }

                const std::vector<DeviceObjectType> &GetDeviceObjects() const
                {
                    return m_dev_segs;
                }

            protected:
                void DeallocateDevSegs()
                {
                    for (auto &seg : m_dev_segs)
                    {
                        DeallocateDevSeg(seg);
                    }
                    m_dev_segs.clear();
                }

                void DeallocateDevSeg(DeviceObjectType &datum_seg)
                {
                    GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                    datum_seg.data_ptr = nullptr;
                    datum_seg.offset = 0;
                    datum_seg.size = 0;
                }
            };
        } // namespace multi

    } // namespace graphs
} // namespace groute

#endif // __GROUTE_GRAPHS_CSR_GRAPH_H
