#pragma once
#include <cuda_runtime.h>
#include <framework/repair_topology_storage.cuh>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <chrono>

// Experimental in-process compact ordered repair. All topology preparation,
// transfers, state publication and workspace destruction are in batch timing.
namespace ordered_repair {
inline void Check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
 T *p=nullptr;
 explicit Device(size_t n) { if(n) Check(cudaMalloc(&p,n*sizeof(T))); }
 ~Device() { if(p) cudaFree(p); }
 Device(const Device&)=delete;
 void Copy(const std::vector<T>& v) { if(!v.empty()) Check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
};
struct Counters { unsigned long long edges,successes,duplicates; uint32_t count,overflow; };
__global__ void ResetControl(Counters *stats,uint32_t *control) {
    *stats=Counters{}; control[0]=UINT32_MAX; control[1]=0;
}
__global__ void FindBucket(uint32_t n,const uint32_t *distance,const uint32_t *pending,uint32_t *bucket) {
    for(uint32_t u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x)
        if(pending[u] && distance[u]!=UINT32_MAX) atomicMin(bucket,distance[u]/128);
}
__global__ void SelectBucket(uint32_t n,const uint32_t *distance,uint32_t *pending,uint32_t bucket,
                             uint32_t *frontier,Counters *stats) {
    for(uint32_t u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x)
        if(pending[u] && distance[u]/128==bucket) {
            pending[u]=0; frontier[atomicAdd(&stats->count,1u)]=u;
        }
}
__global__ void FindActiveBucket(const uint32_t *active,uint32_t count,const uint32_t *distance,uint32_t *bucket) {
    for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        atomicMin(bucket,distance[active[i]]/128);
}
__global__ void PartitionActive(const uint32_t *active,uint32_t count,const uint32_t *distance,
                                uint32_t *pending,uint32_t bucket,uint32_t *selected,uint32_t *selected_count,
                                uint32_t *deferred,Counters *stats,const uint32_t *device_bucket=nullptr,
                                uint32_t reverse_capacity=0) {
    if(device_bucket) bucket=*device_bucket;
    for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const auto u=active[i];
        if(distance[u]/128==bucket) {
            pending[u]=0; selected[atomicAdd(selected_count,1u)]=u;
        } else {
            const auto pos=atomicAdd(&stats->count,1u);
            deferred[reverse_capacity?reverse_capacity-1-pos:pos]=u;
        }
    }
}
template<class App>
__global__ void Expand(const uint32_t *frontier,uint32_t count,const uint64_t *offsets,
                       const uint32_t *outgoing,const uint32_t *ids,uint32_t *distance,
                       uint32_t *queued,uint32_t epoch,uint32_t *next,uint32_t capacity,Counters *stats,
                       const uint32_t *device_count=nullptr,const uint32_t *deferred=nullptr,
                       uint32_t active_count=0) {
    if(device_count) count=*device_count;
    // Partition reserves the output prefix for deferred entries. Relaxation
    // appends only after that prefix, so these copies need no grid-wide barrier.
    if(deferred) for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<active_count-count;i+=blockDim.x*gridDim.x)
        next[i]=deferred[capacity-1-i];
    for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const auto u=frontier[i];
        const uint32_t value=atomicAdd(distance+u,0u);
        unsigned long long edges=0,success=0,duplicate=0;
        for (uint64_t e=offsets[u];e<offsets[u+1];++e) {
            ++edges; const auto v=outgoing[e];
            const uint64_t candidate=uint64_t(value)+App::DeletionEdgeWeight(ids[u],ids[v]);
            if (candidate>=UINT32_MAX) continue;
            if (candidate<atomicMin(distance+v,uint32_t(candidate))) {
                ++success;
                if (atomicExch(queued+v,1u)==0) {
                    const auto pos=atomicAdd(&stats->count,1u);
                    if(pos<capacity) next[pos]=v; else atomicExch(&stats->overflow,1u);
                } else ++duplicate;
            }
        }
        atomicAdd(&stats->edges,edges); atomicAdd(&stats->successes,success); atomicAdd(&stats->duplicates,duplicate);
    }
}

template<class App, class Value>
__global__ void Seed(uint32_t n,const uint32_t* ids,const uint64_t* offsets,const uint32_t* incoming,
                     const Value* values,uint32_t* distance,uint32_t* pending,uint32_t* queue,Counters* stats) {
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
  const auto dst=ids[i]; uint32_t best=values[dst];
  for(uint64_t e=offsets[i];e<offsets[i+1];++e) {
   auto src=incoming[e]; uint64_t candidate=uint64_t(values[src])+App::DeletionEdgeWeight(src,dst);
   if(candidate<best) best=candidate;
  }
  distance[i]=best; pending[i]=best!=UINT32_MAX;
  if(best!=UINT32_MAX) queue[atomicAdd(&stats->count,1u)]=i;
 }
}
template<class Value>
__global__ void Publish(uint32_t n,const uint32_t* ids,const uint32_t* distance,Value* values) {
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) values[ids[i]]=distance[i];
}
template<class App,class Value,class Buffer>
__global__ void Parents(uint32_t n,const uint32_t* ids,const uint64_t* offsets,const uint32_t* incoming,
                        const Value* values,Buffer* buffers,Value* parents) {
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
  auto dst=ids[i]; uint32_t best=UINT32_MAX; const auto value=values[dst];
  if(value==0) best=dst;
  else if(value!=UINT32_MAX) for(uint64_t e=offsets[i];e<offsets[i+1];++e) {
   auto src=incoming[e];
   if(uint64_t(values[src])+App::DeletionEdgeWeight(src,dst)==value) best=min(best,src);
  }
  buffers[dst]=value; parents[dst]=best;
 }
}
struct Metrics { uint32_t iterations=0; uint64_t scans=0,queue_peak=0,device_bytes=0; double prepare_ms=0,closure_ms=0,publish_ms=0; };
template<class App,class Value,class Buffer>
Metrics Run(uint32_t nodes,const std::vector<uint32_t>& ids,const std::vector<uint64_t>& incoming_offsets,
 const std::vector<uint32_t>& incoming,const uint32_t* d_ids,const uint64_t* d_in_offsets,
 const uint32_t* d_incoming,Value* values,Buffer* buffers,Value* parents, size_t topology_budget) {
 using Clock=std::chrono::steady_clock;
 auto elapsed=[](Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
 auto start=Clock::now(); Metrics m; const uint32_t n=ids.size();
 std::vector<uint32_t> local(nodes,UINT32_MAX);
 for(uint32_t i=0;i<n;++i) local[ids[i]]=i;
 std::vector<uint64_t> offsets(size_t(n)+1,0);
 for(auto src:incoming) if(local[src]!=UINT32_MAX) ++offsets[local[src]+1];
 for(uint32_t i=0;i<n;++i) offsets[i+1]+=offsets[i];
 auto cursor=offsets; std::vector<uint32_t> outgoing(offsets.back());
 for(uint32_t i=0;i<n;++i) for(uint64_t e=incoming_offsets[i];e<incoming_offsets[i+1];++e) {
  auto u=local[incoming[e]]; if(u!=UINT32_MAX) outgoing[cursor[u]++]=i;
 }
 Device<uint32_t> distance(n),q1(n),q2(n),pending(n),control(2);
 sepgraph::runtime::RepairTopologyStorage topology;
 sepgraph::runtime::RepairTopologyStorage::Lease topology_lease{topology};
 topology.Bind(offsets,outgoing,std::min(topology_budget,sepgraph::runtime::RepairTopologyStorage::Budget()));
 Device<Counters> counters(1);
 topology.Upload(offsets,outgoing);
 Check(cudaMemset(counters.p,0,sizeof(Counters)));
 const uint32_t blocks=std::max(1u,std::min(256u,(n+255)/256));
 Seed<App><<<blocks,256>>>(n,d_ids,d_in_offsets,d_incoming,values,distance.p,pending.p,q1.p,counters.p);
 Counters stats{}; Check(cudaMemcpy(&stats,counters.p,sizeof(stats),cudaMemcpyDeviceToHost));
 uint32_t count=stats.count; m.queue_peak=count;
 m.device_bytes=topology.DeviceBytes()+16*uint64_t(n)+8+sizeof(Counters);
 m.prepare_ms=elapsed(start); start=Clock::now();
 while(count) {
  ResetControl<<<1,1>>>(counters.p,control.p);
  FindActiveBucket<<<blocks,256>>>(q1.p,count,distance.p,control.p);
  PartitionActive<<<blocks,256>>>(q1.p,count,distance.p,pending.p,0,q2.p,control.p+1,q2.p,counters.p,control.p,n);
  Expand<App><<<blocks,256>>>(q2.p,0,topology.Offsets(),topology.Sources(),d_ids,distance.p,pending.p,0,q1.p,n,counters.p,control.p+1,q2.p,count);
  Check(cudaGetLastError()); Check(cudaMemcpy(&stats,counters.p,sizeof(stats),cudaMemcpyDeviceToHost));
  if(stats.overflow || stats.count>n) throw std::runtime_error("Ordered repair queue overflow");
  count=stats.count; m.queue_peak=std::max(m.queue_peak,uint64_t(count)); m.scans+=stats.edges;
  if(++m.iterations==UINT32_MAX) throw std::runtime_error("Ordered repair iteration overflow");
 }
 m.closure_ms=elapsed(start); start=Clock::now();
 Publish<<<blocks,256>>>(n,d_ids,distance.p,values);
 Parents<App><<<blocks,256>>>(n,d_ids,d_in_offsets,d_incoming,values,buffers,parents);
 Check(cudaGetLastError()); Check(cudaDeviceSynchronize()); m.publish_ms=elapsed(start);
 return m;
}
} // namespace ordered_repair
