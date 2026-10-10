#ifndef SEPGRAPH_EFFECTIVE_UPDATE_BATCH_H
#define SEPGRAPH_EFFECTIVE_UPDATE_BATCH_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>
#include <framework/topology_replay.h>

namespace sepgraph {
namespace topology {

enum class UpdatePhase { Mixed, Delete, Add };

struct DestinationRange {
    const index_t *first = nullptr;
    size_t count = 0;
    const index_t *begin() const { return first; }
    const index_t *end() const { return count ? first + count : first; }
    const index_t *data() const { return first; }
    size_t size() const { return count; }
    bool empty() const { return count == 0; }
    index_t front() const { return *first; }
};

struct SourceMutationView {
    DestinationRange deletions;
    DestinationRange additions;
};

class GroupedUpdateBatch {
public:
    explicit GroupedUpdateBatch(const TopologyMutationBatch &batch = {}) {
        const char *setting = std::getenv("CG_BATCH_MAINTENANCE");
        const std::string mode = setting ? setting : "regular";
        if (mode != "regular" && mode != "large" && mode != "auto")
            throw std::invalid_argument("CG_BATCH_MAINTENANCE must be regular, large or auto");
        large_ = mode == "large" || (mode == "auto" &&
            batch.deletions.size() + batch.additions.size() >= 1000000);
        struct Record { index_t source, destination; bool addition; };
        std::vector<Record> records;
        records.reserve(batch.deletions.size() + batch.additions.size());
        for (const auto &edge : batch.deletions)
            records.push_back({edge.source, edge.destination, false});
        for (const auto &edge : batch.additions)
            records.push_back({edge.source, edge.destination, true});
        if (records.size() < 4096) {
            std::stable_sort(records.begin(), records.end(),
                [](const Record &a, const Record &b) { return a.source < b.source; });
        } else {
            static_assert(sizeof(index_t) == sizeof(uint32_t), "source radix requires 32-bit IDs");
            std::vector<Record> scratch(records.size());
            // Stable passes preserve input order within each source, including
            // duplicate destinations and the delete-before-add boundary.
            for (unsigned shift = 0; shift < 32; shift += 8) {

                std::array<size_t, 256> offsets{};
                for (const auto &record : records) ++offsets[(record.source >> shift) & 255U];
                size_t total = 0;
                for (auto &offset : offsets) {
                    const size_t count = offset;
                    offset = total;
                    total += count;
                }
                for (const auto &record : records)
                    scratch[offsets[(record.source >> shift) & 255U]++] = record;
                records.swap(scratch);
            }
        }

        destinations_.reserve(records.size());
        for (const auto &record : records) {
            if (sources.empty() || sources.back() != record.source) {
                sources.push_back(record.source);
                groups_.push_back({destinations_.size(), 0, 0});
            }
            destinations_.push_back(record.destination);
            if (record.addition) ++groups_.back().adds;
            else ++groups_.back().deletes;
        }
        for (size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].deletes) delete_indices_.push_back(i);
            if (groups_[i].adds) add_indices_.push_back(i);
        }
    }

    SourceMutationView View(size_t i, UpdatePhase phase) const {
        const auto &group = groups_[i];
        const auto *data = destinations_.data() + group.offset;
        return {{data, phase == UpdatePhase::Add ? 0 : group.deletes},
                {data + group.deletes, phase == UpdatePhase::Delete ? 0 : group.adds}};
    }

    size_t PhaseSize(UpdatePhase phase) const {
        return phase == UpdatePhase::Mixed ? sources.size()
            : (phase == UpdatePhase::Delete ? delete_indices_.size() : add_indices_.size());
    }
    size_t SourceIndex(size_t i, UpdatePhase phase) const {
        return phase == UpdatePhase::Mixed ? i
            : (phase == UpdatePhase::Delete ? delete_indices_[i] : add_indices_[i]);
    }

    bool LargeMaintenance() const { return large_; }

    std::vector<index_t> sources;

private:
    bool large_ = false;
    struct Group { size_t offset, deletes, adds; };
    std::vector<Group> groups_;
    std::vector<index_t> destinations_;
    std::vector<size_t> delete_indices_, add_indices_;
};

struct EffectiveEdgeDelta {
    index_t source;
    index_t destination;
    int64_t count;
};

struct IgnoreEffectiveUpdates {
    void Prepare(const std::vector<EffectiveEdgeDelta> &) {}
    void Commit() noexcept {}
};

} // namespace topology
} // namespace sepgraph
#endif
