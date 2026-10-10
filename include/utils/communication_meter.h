#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Host API payload ledger. No synchronization, pointer queries or device counters.
// Link --wrap=cudaMemcpy/Async to capture calls across translation units.
namespace cgcomm {
enum class Category { Other, Topology, Incoming, State, Control, Cache, Count };
enum class Direction { H2D, D2H, D2D, H2H, Default, Count };
inline bool Enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("CG_COMM_METER");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
struct Counter {
    std::atomic<uint64_t> bytes{0}, calls{0};
};
inline Counter (&Counters())[8][6][5] {
    static Counter counters[8][6][5];
    return counters;
}
inline std::atomic<int> &CurrentStage() { static std::atomic<int> stage{0}; return stage; }
inline void Stage(int stage) { if (Enabled()) CurrentStage().store(stage, std::memory_order_relaxed); }
inline Category &CurrentCategory() {
    static thread_local Category category = Category::Other;
    return category;
}
class Scope {
    Category previous;
public:
    explicit Scope(Category category) : previous(CurrentCategory()) {
        CurrentCategory() = category;
    }
    ~Scope() { CurrentCategory() = previous; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};
inline void Record(Direction direction, size_t bytes) {
    if (!Enabled()) return;
    auto &counter = Counters()[CurrentStage().load(std::memory_order_relaxed)][static_cast<int>(CurrentCategory())][static_cast<int>(direction)];
    counter.bytes.fetch_add(bytes, std::memory_order_relaxed);
    counter.calls.fetch_add(1, std::memory_order_relaxed);
}

// A successful async API call means accepted payload, not physical bus traffic.
inline void Flush(int batch, const char *stage) {
    if (!Enabled()) return;
    static const char *categories[] = {"other", "topology", "repair_incoming", "state", "control", "cache"};
    static const char *directions[] = {"h2d", "d2h", "d2d", "h2h", "default_unclassified"};
    const char *stages[] = {stage, "addition", "hotness", "candidate", "candidate_trace", "eviction", "compact", "cache_load"};
    for (int s = 0; s < 8; ++s) for (int c = 0; c < 6; ++c) for (int d = 0; d < 5; ++d) {
        auto &counter = Counters()[s][c][d];
        const auto bytes = counter.bytes.exchange(0, std::memory_order_relaxed);
        const auto calls = counter.calls.exchange(0, std::memory_order_relaxed);
        if (calls) std::printf("[COMM] batch=%d stage=%s category=%s direction=%s bytes=%llu calls=%llu\n",
            batch, stages[s], categories[c], directions[d],
            static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(calls));
    }
}
} // namespace cgcomm
