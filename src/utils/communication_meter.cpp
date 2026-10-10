#include <cuda_runtime_api.h>
#include <utils/communication_meter.h>

extern "C" cudaError_t __real_cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
extern "C" cudaError_t __real_cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);

static void RecordCopy(cudaError_t status, size_t bytes, cudaMemcpyKind kind) {
    if (status != cudaSuccess || !cgcomm::Enabled()) return;
    using D = cgcomm::Direction;
    D direction = D::Default;
    switch (kind) {
        case cudaMemcpyHostToDevice: direction = D::H2D; break;
        case cudaMemcpyDeviceToHost: direction = D::D2H; break;
        case cudaMemcpyDeviceToDevice: direction = D::D2D; break;
        case cudaMemcpyHostToHost: direction = D::H2H; break;
        default: break;
    }
    cgcomm::Record(direction, bytes);
}
extern "C" cudaError_t __wrap_cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    const auto status = __real_cudaMemcpy(dst, src, bytes, kind);
    RecordCopy(status, bytes, kind);
    return status;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const auto status = __real_cudaMemcpyAsync(dst, src, bytes, kind, stream);
    RecordCopy(status, bytes, kind);
    return status;
}
