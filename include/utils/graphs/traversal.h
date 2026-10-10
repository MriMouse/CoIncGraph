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

#ifndef __GRAPHS_TRAVERSAL_H
#define __GRAPHS_TRAVERSAL_H

#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cassert>
#include <sstream>

#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.cuh>


#include <utils/parser.h>
#include <framework/undirected_input.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include <gflags/gflags.h>

DECLARE_int32(weight_num);
DECLARE_int32(SEGMENT);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_bool(verbose);

DECLARE_string(graphfile);

DECLARE_string(format);

DECLARE_bool(gen_graph);
DECLARE_int32(gen_nnodes);
DECLARE_int32(gen_factor);
DECLARE_bool(gen_weights);
DECLARE_int32(gen_weight_range);

DECLARE_int32(block_size);
DECLARE_int32(prio_delta);
DECLARE_bool(stats);
DECLARE_string(updatefile);
DECLARE_string(update_size);


using std::min;
using std::max;

inline void KernelSizing(dim3 &grid_dims, dim3 &block_dims, uint32_t work_size)
{
    dim3 bd(FLAGS_block_size, 1, 1);
    
    
    dim3 gd(round_up(work_size, bd.x), 1, 1);

    if (grid_dims.x > 480)grid_dims.x = 480;
    grid_dims = gd;
    block_dims = bd;
}

namespace utils
{
    namespace traversal
    {
        /*
        * @brief A global context for graph traversal workers
        */
        template<typename Algo>
        class Context : public groute::Context
        {
        public:
            groute::graphs::host::CSRGraph host_graph;
            
            groute::graphs::host::PMAGraph host_pma_small;

            int ngpus;
            int nvtxs;
            uint64_t nedges;
            index_t segment = FLAGS_SEGMENT;
            index_t segment_ct;
	        uint64_t seg_nedge_csr[512];
	        uint64_t seg_nedge_csc[512];
	        uint64_t seg_sedge_csr[512];
	        uint64_t seg_sedge_csc[512];
            uint64_t seg_nedge_csr_ct[512];
            uint64_t seg_sedge_csr_ct[512];
	        index_t seg_snode[512];
	        index_t seg_enode[512];
            index_t seg_snode_ct[512];
            index_t seg_enode_ct[512];
            index_t segment_id_ct[512];

            Context(int ngpus=1, bool allow_static = false, bool normalize_undirected = false) :
                    groute::Context(), ngpus(ngpus)
            {
                if (FLAGS_gen_graph) //Judge whether to generate the graph by yourself
                {
                    printf("\nGenerating graph, nnodes: %d, gen_factor: %d", FLAGS_gen_nnodes, FLAGS_gen_factor);

                }

                else
                {
                    graph_t *graph;
                    
                    if (FLAGS_graphfile == "")
                    {
                        printf("A Graph File must be provided\n");
                        exit(0);
                    }
                    if (FLAGS_updatefile == "" && !allow_static)
                    {
                        printf("Update File must be provided\n");
                        exit(0);
                    }

                    printf("\nLoading graph %s (%s)\n", FLAGS_graphfile.substr(FLAGS_graphfile.find_last_of('\\') + 1).c_str(), FLAGS_format.c_str());

                    graph = GetCachedGraph(FLAGS_graphfile, FLAGS_format,FLAGS_weight_num);

                    if (graph->nvtxs == 0)
                    {
                        printf("Empty graph!\n");
                        exit(0);
                    }

                    
                    //         graph->nvtxs, graph->nedges,
                    //         graph->xadj, graph->adjncy,
                    //         graph->readew ? graph->adjwgt : nullptr,
                    //         graph->readvw ? graph->vwgt : nullptr // avoid binding to the default 1's weight arrays
                    
                    if (normalize_undirected) {
                        Stopwatch normalization(true);
                        std::vector<sepgraph::topology::UndirectedInputEdge> records;
                        records.reserve(graph->nedges);
                        for (uint32_t u = 0; u < graph->nvtxs; ++u)
                            for (uint64_t e = graph->xadj[u]; e < graph->xadj[u + 1]; ++e)
                                records.push_back({u, graph->adjncy[e]});
                        std::vector<uint64_t> offsets;
                        std::vector<uint32_t> neighbors;
                        {
                            sepgraph::topology::UndirectedInput input(graph->nvtxs, records.size(),
                                [&](size_t i) { return records[i]; });
                            std::vector<sepgraph::topology::UndirectedInputEdge>().swap(records);
                            input.CSR(graph->nvtxs, offsets, neighbors);
                        }
                        host_pma_small.Bind(graph->nvtxs, neighbors.size(), offsets.data(),
                                            neighbors.data(), nullptr, nullptr);
                        // Keep CSR ownership valid through setup. LoadGraph releases
                        // this temporary CSR once the chunk topology is authoritative.
                        host_pma_small.row_start_vec = std::move(offsets);
                        host_pma_small.edge_dst_vec = std::move(neighbors);
                        host_pma_small.row_start = host_pma_small.row_start_vec.data();
                        host_pma_small.edge_dst = host_pma_small.edge_dst_vec.data();
                        printf("[CC-INPUT-GRAPH] input_records=%llu adjacency_records=%llu normalize_ms=%.3f\n",
                            (unsigned long long)graph->nedges,
                            (unsigned long long)host_pma_small.nedges, normalization.ms());
                    } else {
                    host_pma_small.Bind(
                            graph->nvtxs, graph->nedges,
                            graph->xadj, graph->adjncy,
                            graph->readew ? graph->adjwgt : nullptr,
                            graph->readvw ? graph->vwgt : nullptr // avoid binding to the default 1's weight arrays
                    );
                    }
                    
                    //         validate_graph->nvtxs, validate_graph->nedges,
                    //         validate_graph->xadj, validate_graph->adjncy,
                    //         validate_graph->readew ? validate_graph->adjwgt : nullptr,
                    //         validate_graph->readvw ? validate_graph->vwgt : nullptr // avoid binding to the default 1's weight arrays
                    

                    if (FLAGS_stats)
                    {
                        printf(
                                "The graph has %d vertices, and %ld edges (avg. degree: %f)\n",
                                host_pma_small.nnodes, host_pma_small.nedges, (float) host_pma_small.nedges / host_pma_small.nnodes);

                        CleanupGraphs();
                        CleanupGraphs_v();
                        exit(0);
                    }
                }

                //     // Generate edge data

                //     // for (int i = 0; i < host_graph.nedges; i++)
                //     // {
                
                //     // }

                nvtxs = host_pma_small.nnodes;
                nedges = host_pma_small.nedges;
                printf("\n----- Running %s -----\n\n", Algo::Name());

                if (FLAGS_verbose)
                {
                    printf("The graph has %d vertices, and %ld edges\n", nvtxs, nedges);
                }
            }
        };

    }
}

#endif // __GRAPHS_TRAVERSAL_H
