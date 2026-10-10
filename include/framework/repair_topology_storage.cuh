#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sepgraph { namespace runtime {

// Bound repair adjacency independently of the forward hot cache. Small repairs

// vectors instead of allocating another edge-sized copy (on either processor).
class RepairTopologyStorage {
    static void Check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("repair topology: ") + cudaGetErrorString(status));
    }
    struct Buffer {
        void *device = nullptr, *mapped = nullptr, *view = nullptr;
        size_t capacity = 0, bytes = 0;
        ~Buffer() { Unmap(); FreeDevice(); }
        void Unmap() {
            if (mapped) cudaHostUnregister(mapped);
            mapped = nullptr;
            view = device;
        }
        void FreeDevice() {
            if (device) cudaFree(device);
            device = nullptr;
            capacity = 0;
            view = nullptr;
        }
        void Bind(void *host, size_t size, size_t &remaining) {
            bytes = size;
            if (capacity > remaining || bytes > remaining) FreeDevice();
            if (!bytes) { view = device; remaining -= capacity; return; }
            if (bytes <= remaining) {
                if (capacity < bytes) {
                    FreeDevice();
                    const auto status = cudaMalloc(&device, bytes);
                    if (status == cudaSuccess) capacity = bytes;
                    else if (status == cudaErrorMemoryAllocation) {
                        // Another allocation can race the free-memory snapshot.
                        cudaGetLastError();
                        device = nullptr;
                    } else Check(status);
                }
                if (device) { view = device; remaining -= capacity; return; }
            }
            Check(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
            mapped = host;
            Check(cudaHostGetDevicePointer(&view, host, 0));
        }
        void Upload(const void *host) {
            if (bytes && !mapped) Check(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
        }
    } offsets_, sources_;

public:
    RepairTopologyStorage() = default;
    RepairTopologyStorage(const RepairTopologyStorage &) = delete;
    RepairTopologyStorage &operator=(const RepairTopologyStorage &) = delete;

    // Call while the borrowed vectors are still alive and before resizing them.
    // The lease also handles exceptional exits with kernels still in flight.
    struct Lease {
        RepairTopologyStorage &storage;
        ~Lease() { storage.ReleaseMappings(); }
    };
    void ReleaseMappings() {
        if (MappedBytes()) cudaDeviceSynchronize();
        offsets_.Unmap();
        sources_.Unmap();
    }
    static size_t Budget(size_t reusable_bytes = 0) {
        size_t limit = size_t(4096) << 20;
        if (const char *value = std::getenv("CG_REPAIR_TOPOLOGY_MB")) {
            const std::string text(value);
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("CG_REPAIR_TOPOLOGY_MB must be a nonnegative integer");
            const auto mb = std::stoull(text);
            if (mb > (std::numeric_limits<size_t>::max() >> 20))
                throw std::invalid_argument("CG_REPAIR_TOPOLOGY_MB is too large");
            limit = size_t(mb) << 20;
        }
        size_t free = 0, total = 0;
        Check(cudaMemGetInfo(&free, &total));
        const size_t available = free + reusable_bytes;
        const size_t reserve = size_t(512) << 20;
        return std::min(limit, available > reserve ? (available - reserve) / 2 : size_t(0));
    }
    void Bind(std::vector<uint64_t> &offsets, std::vector<uint32_t> &sources, size_t budget) {
        ReleaseMappings();
        if (DeviceBytes() > budget) { offsets_.FreeDevice(); sources_.FreeDevice(); }
        const size_t offset_bytes = offsets.size() * sizeof(uint64_t);
        if (offset_bytes <= budget &&
            sources_.capacity > budget - std::max(offset_bytes, offsets_.capacity))
            sources_.FreeDevice();
        // Prioritize row metadata: it is small and read by every repair round.
        offsets_.Bind(offsets.data(), offset_bytes, budget);
        // A retained source allocation must fit alongside the current offsets.
        sources_.Bind(sources.data(), sources.size() * sizeof(uint32_t), budget);
    }
    void Upload(const std::vector<uint64_t> &offsets, const std::vector<uint32_t> &sources) {
        offsets_.Upload(offsets.data());
        sources_.Upload(sources.data());
    }
    const uint64_t *Offsets() const { return static_cast<const uint64_t *>(offsets_.view); }
    const uint32_t *Sources() const { return static_cast<const uint32_t *>(sources_.view); }
    size_t DeviceBytes() const { return offsets_.capacity + sources_.capacity; }
    size_t MappedBytes() const {
        return (offsets_.mapped ? offsets_.bytes : 0) + (sources_.mapped ? sources_.bytes : 0);
    }
    size_t TransferBytes() const {
        return (offsets_.mapped ? 0 : offsets_.bytes) + (sources_.mapped ? 0 : sources_.bytes);
    }
};

}} // namespace sepgraph::runtime
