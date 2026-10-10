#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>

// Whole-cohort markers for an external NVML sampler. No device instrumentation.

namespace cgcomm {
inline bool WindowEnabled() {
    const char *v = std::getenv("CG_COMM_WINDOW");
    return v && v[0] == '1' && v[1] == '\0';
}
inline void Window(const char *event) {
    if (!WindowEnabled()) return;
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    std::fprintf(stderr, "[CG-COMM-WINDOW] event=%s monotonic_ns=%llu\n", event,
        static_cast<unsigned long long>(t.tv_sec) * 1000000000ULL + t.tv_nsec);
    std::fflush(stderr);
}

template<class Values> inline void Result(const Values &values) {
    if (!WindowEnabled()) return;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t node = 0; node < values.size(); ++node) {
        if (values[node] == UINT32_MAX) continue;
        const uint64_t value = values[node];
        hash ^= node + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        hash *= 1099511628211ULL;
    }
    std::fprintf(stderr, "[CG-COMM-RESULT] distance_checksum=%llu\n",
                 static_cast<unsigned long long>(hash));
}
}
