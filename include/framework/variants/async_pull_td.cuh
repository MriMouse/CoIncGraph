// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_ASYNC_PULL_TD_H
#define HYBRID_ASYNC_PULL_TD_H

#include <framework/variants/pull_functor.h>
#include <groute/device/cta_scheduler_hybrid.cuh>

namespace sepgraph
{
    namespace kernel
    {
        namespace async_pull_td
        {
            //TODO Is that possible using async pull for PR?
            //THINK AsyncPullTD PR?
            //Pull buffer from neighbor
            //barrier
            //CombineBuffer

            template<typename TAppInst,
                    typename WorkSource,
                    typename CSCGraph,
                    template<typename> class GraphDatum,
                    typename TBuffer,
                    typename TWeight>
            //__device__ __forceinline__
            __global__
            void Relax(TAppInst app_inst,
                       WorkSource work_source,
                       CSCGraph csc_graph,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {

//                PullFunctor <TAppInst, CSCGraph, GraphDatum, TBuffer, TWeight>

//

//

            }
        }
    }
}
#endif //HYBRID_ASYNC_PULL_TD_H
