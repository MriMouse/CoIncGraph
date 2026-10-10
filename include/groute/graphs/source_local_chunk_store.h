#ifndef SEPGRAPH_SOURCE_LOCAL_CHUNK_STORE_H
#define SEPGRAPH_SOURCE_LOCAL_CHUNK_STORE_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <framework/topology_replay.h>
#include <framework/effective_update_batch.h>
#include <groute/graphs/topology_contract.cuh>
#include <utils/fixed_worker_pool.h>

namespace sepgraph {
namespace topology {

struct ChunkArenaOptions {
    uint64_t capacity_edges = 0;
    uint64_t slab_capacity_edges = 1ULL << 20;
    index_t minimum_chunk_edges = 4;
    bool pinned = true;
    uint32_t mutation_workers = 0;
};

struct ChunkStoreBatchMetrics {
    uint64_t epoch = 0;
    uint64_t touched_sources = 0;
    uint64_t changed_sources = 0;
    uint64_t missing_deletes = 0;
    uint64_t invalid_additions = 0;
    uint64_t mutation_written_bytes = 0;
    uint64_t relocation_copied_bytes = 0;
    uint64_t allocations = 0;
    uint64_t reused_blocks = 0;
    uint64_t retired_blocks = 0;
    uint64_t update_count = 0;
    uint64_t source_work = 0;
    uint64_t max_source_updates = 0;
    uint64_t max_source_work = 0;
    uint64_t worker_count = 0;
    double group_ms = 0.0;
    double prepare_ms = 0.0;
    double preflight_ms = 0.0;
    double commit_ms = 0.0;
    double apply_ms = 0.0;
    double retire_ms = 0.0;
    double mutation_ms = 0.0;
    double allocation_ms = 0.0;
    double reverse_prepare_ms = 0.0;
    uint64_t effective_records = 0;
    uint64_t source_plan_duplicate_bytes = 0;
    uint64_t source_plan_index_bytes = 0;
    uint64_t deletion_match_reads = 0;
    uint64_t mutation_edge_reads = 0;
};

struct ChunkArenaStats {
    uint64_t capacity_edges = 0;
    uint64_t high_water_edges = 0;
    uint64_t live_capacity_edges = 0;
    uint64_t retired_capacity_edges = 0;
    uint64_t free_capacity_edges = 0;
    uint64_t allocation_count = 0;
    uint64_t reuse_count = 0;
    uint64_t reclaimed_count = 0;
    bool pinned = false;
};

// Per-source mutation only produces these counters. Allocating the complete
// batch report per source unnecessarily zeroed and traversed 30 fields for
// every touched row in both phases (including timers never written here).
struct ChunkSourceMutationMetrics {
    uint64_t mutation_edge_reads = 0;
    uint64_t mutation_written_bytes = 0;
    uint64_t relocation_copied_bytes = 0;
};

class SourceLocalChunkStore {
public:
    static constexpr uint32_t kInvalidSlab = std::numeric_limits<uint32_t>::max();

    SourceLocalChunkStore(index_t node_count, ChunkArenaOptions options)
        : sources_(node_count), options_(options), allocation_slab_cursor_(0),
          logical_edge_count_(0), materialized_edge_count_(0),
          published_epoch_(0), pending_epoch_(0), batch_pending_(false) {
        if (options_.capacity_edges == 0) {
            throw std::invalid_argument("chunk arena capacity must be non-zero");
        }
        if (options_.slab_capacity_edges == 0) {
            throw std::invalid_argument("chunk slab capacity must be non-zero");
        }
        if (options_.minimum_chunk_edges == 0) options_.minimum_chunk_edges = 1;
        try {
            uint64_t remaining = options_.capacity_edges;
            while (remaining != 0) {
                const uint64_t slab_edges = std::min(remaining, options_.slab_capacity_edges);
                slabs_.push_back(AllocateSlab(slab_edges));
                remaining -= slab_edges;
            }
        } catch (...) {
            ReleaseSlabs();
            throw;
        }
        stats_.capacity_edges = options_.capacity_edges;
        stats_.pinned = options_.pinned;
        uint32_t worker_count = options_.mutation_workers;
        if (worker_count == 0) {
            const uint32_t hardware_workers = std::thread::hardware_concurrency();
            worker_count = std::max<uint32_t>(
                1, std::min<uint32_t>(20, hardware_workers == 0 ? 1 : hardware_workers));
        }
        mutation_workers_.reset(new concurrency::FixedWorkerPool(worker_count));
    }

    SourceLocalChunkStore(const SourceLocalChunkStore &) = delete;
    SourceLocalChunkStore &operator=(const SourceLocalChunkStore &) = delete;

    ~SourceLocalChunkStore() {
        ReleaseSlabs();
    }

    static uint64_t RequiredChunkCapacity(uint64_t degree,
                                          index_t minimum_chunk_edges = 4) {
        if (degree == 0) return 0;
        uint64_t capacity = std::max<uint64_t>(degree, minimum_chunk_edges);
        --capacity;
        capacity |= capacity >> 1;
        capacity |= capacity >> 2;
        capacity |= capacity >> 4;
        capacity |= capacity >> 8;
        capacity |= capacity >> 16;
        capacity |= capacity >> 32;
        return capacity + 1;
    }

    void LoadSource(index_t source, const std::vector<index_t> &neighbors) {
        EnsureNoPendingBatch();
        SourceState &state = Source(source);
        if (state.materialized) throw std::logic_error("source already materialized");
        state.materialized = true;
        if (!neighbors.empty()) {
            const uint64_t capacity = RequiredChunkCapacity(
                neighbors.size(), options_.minimum_chunk_edges);
            const Allocation allocation = Allocate(capacity);
            std::memcpy(BlockData(allocation.block), neighbors.data(),
                        neighbors.size() * sizeof(index_t));
            state.block = allocation.block;
            state.descriptor.index = allocation.block.offset;
            state.descriptor.degree = static_cast<index_t>(neighbors.size());
            state.descriptor.slab_id = allocation.block.slab_id;
        }
        materialized_edge_count_ += neighbors.size();
    }

    void FinalizeLoad(uint64_t graph_edge_count = std::numeric_limits<uint64_t>::max()) {
        EnsureNoPendingBatch();
        logical_edge_count_ = graph_edge_count == std::numeric_limits<uint64_t>::max()
            ? materialized_edge_count_
            : graph_edge_count;
    }

    ChunkStoreBatchMetrics ApplyBatch(const TopologyMutationBatch &batch) {
        EnsureNoPendingBatch();
        const auto begin = Clock::now();
        GroupedUpdateBatch grouped(batch);
        const double group_ms = Milliseconds(begin, Clock::now());
        IgnoreEffectiveUpdates observer;
        auto metrics = ApplyMutationPhase(grouped, UpdatePhase::Mixed, true, observer);
        metrics.group_ms += group_ms;
        return metrics;
    }

    // Deletion repair needs the host graph after deletions but before additions.
    // This appends the addition phase to the same unpublished topology epoch.
    ChunkStoreBatchMetrics ApplyPendingAdditions(
            const TopologyMutationBatch &additions) {
        if (!batch_pending_) {
            throw std::logic_error("no pending chunk-store batch");
        }
        if (!additions.deletions.empty()) {
            throw std::invalid_argument(
                "pending chunk-store phase accepts additions only");
        }
        const auto begin = Clock::now();
        GroupedUpdateBatch grouped(additions);
        const double group_ms = Milliseconds(begin, Clock::now());
        IgnoreEffectiveUpdates observer;
        auto metrics = ApplyMutationPhase(grouped, UpdatePhase::Add, false, observer);
        metrics.group_ms += group_ms;
        return metrics;
    }

    template <typename Observer>
    ChunkStoreBatchMetrics ApplyGroupedPhase(const GroupedUpdateBatch &batch,
                                             UpdatePhase phase, Observer &observer) {
        if (phase == UpdatePhase::Mixed)
            throw std::invalid_argument("grouped streaming update requires a single phase");
        if (phase == UpdatePhase::Delete) EnsureNoPendingBatch();
        else if (!batch_pending_) throw std::logic_error("no pending chunk-store batch");
        return ApplyMutationPhase(batch, phase, phase == UpdatePhase::Delete, observer);
    }

    concurrency::FixedWorkerPool& MutationWorkers() { return *mutation_workers_; }

    const std::vector<index_t> &ChangedSources() const { return changed_sources_; }

private:
    template <typename Observer>
    ChunkStoreBatchMetrics ApplyMutationPhase(const GroupedUpdateBatch &batch,
                                              UpdatePhase phase, bool begin_epoch,
                                              Observer &observer) {
        static_assert(noexcept(observer.Commit()), "effective update commit must not throw");
        const auto group_begin = Clock::now();
        const auto &sources = batch.sources;

        ChunkStoreBatchMetrics metrics;
        metrics.epoch = begin_epoch ? published_epoch_ + 1 : pending_epoch_;
        if (metrics.epoch > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("topology source version exhausted");
        }
        metrics.worker_count = mutation_workers_->WorkerCount();
        const size_t phase_size = batch.PhaseSize(phase);
        for (size_t i = 0; i < phase_size; ++i) {
            const size_t source_index = batch.SourceIndex(i, phase);
            const index_t source = sources[source_index];
            const auto mutations = batch.View(source_index, phase);
            const uint64_t source_updates = mutations.deletions.size() + mutations.additions.size();
            if (source_updates == 0) continue;
            ++metrics.touched_sources;
            metrics.update_count += source_updates;
            metrics.max_source_updates = std::max(
                metrics.max_source_updates, source_updates);
            const uint64_t source_work = source_updates +
                (source < sources_.size() && sources_[source].materialized
                    ? sources_[source].descriptor.degree
                    : 0);
            metrics.source_work += source_work;
            metrics.max_source_work = std::max(
                metrics.max_source_work, source_work);
        }
        metrics.group_ms = Milliseconds(group_begin, Clock::now());

        const auto mutation_begin = Clock::now();
        std::vector<PreparedResult> results(phase_size);
        mutation_workers_->Run(phase_size, [&](size_t phase_index) {
            const size_t source_index = batch.SourceIndex(phase_index, phase);
            const index_t source = sources[source_index];
            const auto mutations = batch.View(source_index, phase);
            PreparedResult &result = results[phase_index];
            if (source >= sources_.size() || !sources_[source].materialized) {
                result.missing_deletes = mutations.deletions.size();
                result.invalid_additions = mutations.additions.size();
                return;
            }

            result.update.source = source;
            result.update.group_index = source_index;
            BuildDeletionPlan(source, mutations.deletions, result.update);
            const uint64_t successful_deletes = result.update.successful_deletes;
            result.missing_deletes =
                mutations.deletions.size() - successful_deletes;
            result.edge_delta = static_cast<int64_t>(mutations.additions.size()) -
                                static_cast<int64_t>(successful_deletes);
            const uint64_t final_degree = sources_[source].descriptor.degree -
                successful_deletes + mutations.additions.size();
            if (final_degree > std::numeric_limits<index_t>::max()) {
                throw std::overflow_error("source degree exceeds index_t");
            }
            if (successful_deletes == 0 && mutations.additions.empty()) return;
            result.changed = true;
            result.update.final_degree = final_degree;
            result.update.expansion = final_degree > sources_[source].block.capacity;
        });

        std::vector<PreparedSource> prepared;
        const bool shared_plan = batch.LargeMaintenance();
        std::vector<size_t> prepared_indices;
        if (shared_plan) prepared_indices.reserve(phase_size);
        else prepared.reserve(phase_size);
        std::vector<uint64_t> required_allocations;
        int64_t edge_delta = 0;
        for (auto &result : results) {
            metrics.deletion_match_reads += result.update.match_reads;
            metrics.missing_deletes += result.missing_deletes;
            metrics.invalid_additions += result.invalid_additions;
            edge_delta += result.edge_delta;
            if (!result.changed) continue;
            ++metrics.changed_sources;
            if (result.update.expansion) {
                required_allocations.push_back(RequiredChunkCapacity(
                    result.update.final_degree, options_.minimum_chunk_edges));
            }
            if (shared_plan) prepared_indices.push_back(&result - results.data());
            else prepared.push_back(std::move(result.update));
        }
        const size_t prepared_count = shared_plan ? prepared_indices.size() : prepared.size();
        const auto plan = [&](size_t i) -> PreparedSource & {
            return shared_plan ? results[prepared_indices[i]].update : prepared[i];
        };
        metrics.source_plan_duplicate_bytes = prepared.size() * sizeof(PreparedSource);
        metrics.source_plan_index_bytes = prepared_indices.size() * sizeof(size_t);
        const auto prepare_end = Clock::now();
        metrics.prepare_ms = Milliseconds(mutation_begin, prepare_end);
        const auto preflight_begin = Clock::now();
        EnsureAllocationsFit(required_allocations);
        std::vector<EffectiveEdgeDelta> effective;
        effective.reserve(metrics.update_count);
        std::vector<index_t> changed;
        changed.reserve(prepared_count);
        for (size_t i = 0; i < prepared_count; ++i) {
            const auto &update = plan(i);
            changed.push_back(update.source);
            if (update.has_deletions) {
                const auto emit = [&](const DeletionRun &run) {
                    if (run.matched) effective.push_back({update.source,
                        run.destination, -static_cast<int64_t>(run.matched)});
                };
                emit(update.first_deletion);
                for (const auto &run : update.remaining_deletions) emit(run);
            }
            for (const auto destination : batch.View(update.group_index, phase).additions)
                effective.push_back({update.source, destination, 1});
        }
        // Reverse allocations/merges must succeed before forward mutation starts.
        metrics.effective_records = effective.size();
        const auto reverse_begin = Clock::now();
        if (shared_plan) observer.Prepare(std::move(effective));
        else observer.Prepare(effective);
        metrics.reverse_prepare_ms = Milliseconds(reverse_begin, Clock::now());
        std::vector<ChunkSourceMutationMetrics> apply_metrics(prepared_count);
        retired_.reserve(retired_.size() + required_allocations.size());
        metrics.preflight_ms = Milliseconds(preflight_begin, Clock::now());
        for (size_t i = 0; i < prepared_count; ++i) {
            auto &update = plan(i);
            if (!update.expansion) continue;
            const auto allocation_begin = Clock::now();
            update.allocation = Allocate(RequiredChunkCapacity(
                update.final_degree, options_.minimum_chunk_edges));
            metrics.allocation_ms += Milliseconds(allocation_begin, Clock::now());
            ++metrics.allocations;
            if (update.allocation.reused) ++metrics.reused_blocks;
        }
        const auto commit_begin = Clock::now();
        if (begin_epoch) {
            pending_epoch_ = metrics.epoch;
            batch_pending_ = true;
        }
        logical_edge_count_ = static_cast<uint64_t>(
            static_cast<int64_t>(logical_edge_count_) + edge_delta);
        metrics.commit_ms = Milliseconds(commit_begin, Clock::now());
        const auto apply_begin = Clock::now();
        mutation_workers_->Run(prepared_count, [&](size_t update_index) {
            auto &update = plan(update_index);
            if (update.expansion) {
                RewriteSourceToNewBlock(
                    update, batch.View(update.group_index, phase), apply_metrics[update_index]);
            } else {
                CompactSourceInPlace(
                    update, batch.View(update.group_index, phase), apply_metrics[update_index]);
            }
        });
        observer.Commit();
        changed_sources_.swap(changed);
        const auto apply_end = Clock::now();
        metrics.apply_ms = Milliseconds(apply_begin, apply_end);
        const auto retire_begin = Clock::now();
        for (size_t index = 0; index < prepared_count; ++index) {
            const auto &local = apply_metrics[index];
            metrics.mutation_edge_reads += local.mutation_edge_reads;
            metrics.mutation_written_bytes += local.mutation_written_bytes;
            metrics.relocation_copied_bytes += local.relocation_copied_bytes;
            if (plan(index).retired_block.capacity != 0) {
                Retire(plan(index).retired_block, pending_epoch_);
                ++metrics.retired_blocks;
            }
        }
        const auto mutation_end = Clock::now();
        metrics.retire_ms = Milliseconds(retire_begin, mutation_end);
        metrics.mutation_ms = Milliseconds(mutation_begin, mutation_end);
        return metrics;
    }

public:

    uint64_t Publish() {
        if (!batch_pending_) throw std::logic_error("no chunk-store batch to publish");
        published_epoch_ = pending_epoch_;
        batch_pending_ = false;
        return published_epoch_;
    }

    uint64_t ReclaimThrough(uint64_t completed_epoch) {
        uint64_t reclaimed = 0;
        auto output = retired_.begin();
        for (auto current = retired_.begin(); current != retired_.end(); ++current) {
            if (current->epoch <= completed_epoch) {
                free_blocks_[current->block.capacity].push_back(current->block);
                stats_.retired_capacity_edges -= current->block.capacity;
                stats_.free_capacity_edges += current->block.capacity;
                ++stats_.reclaimed_count;
                ++reclaimed;
            } else {
                *output++ = *current;
            }
        }
        retired_.erase(output, retired_.end());
        return reclaimed;
    }

    std::vector<index_t> Neighbors(index_t source) const {
        const SourceState &state = Source(source);
        if (!state.materialized) throw std::logic_error("source is not materialized");
        if (state.descriptor.degree == 0) return {};
        return std::vector<index_t>(
            slabs_.at(state.descriptor.slab_id).data + state.descriptor.index,
            slabs_.at(state.descriptor.slab_id).data +
                state.descriptor.index + state.descriptor.degree);
    }

    SourceTopologyDigest Digest(index_t source) const {
        return DigestSource(source, Neighbors(source));
    }

    uint64_t OrderedHash(index_t source) const {
        const SourceState &state = Source(source);
        uint64_t hash = 1469598103934665603ULL;
        if (state.descriptor.degree != 0) {
            const index_t *edges = slabs_.at(state.descriptor.slab_id).data +
                                   state.descriptor.index;
            for (uint64_t offset = 0; offset < state.descriptor.degree; ++offset) {
                hash ^= static_cast<uint64_t>(edges[offset]);
                hash *= 1099511628211ULL;
            }
        }
        hash ^= static_cast<uint64_t>(state.descriptor.degree);
        hash *= 1099511628211ULL;
        return hash;
    }

    const TopologyDescriptor &Descriptor(index_t source) const {
        return Source(source).descriptor;
    }

    bool IsMaterialized(index_t source) const {
        return source < sources_.size() && sources_[source].materialized;
    }

    uint64_t EdgeCount() const { return logical_edge_count_; }
    uint64_t PublishedEpoch() const { return published_epoch_; }
    uint64_t PendingEpoch() const { return pending_epoch_; }
    bool IsPublished() const { return !batch_pending_ && published_epoch_ == pending_epoch_; }
    const ChunkArenaStats &ArenaStats() const { return stats_; }
    size_t SlabCount() const { return slabs_.size(); }
    const index_t *SlabData(uint32_t slab_id) const { return slabs_.at(slab_id).data; }
    uint64_t MetadataBytes() const { return sources_.size() * sizeof(SourceState); }

private:
    using Clock = std::chrono::steady_clock;

    struct Block {
        uint32_t slab_id = kInvalidSlab;
        uint64_t offset = 0;
        uint64_t capacity = 0;
    };

    struct Slab {
        index_t *data = nullptr;
        uint64_t capacity = 0;
        uint64_t bump = 0;
    };

    struct Allocation {
        Block block;
        bool reused = false;
    };

    struct RetiredBlock {
        Block block;
        uint64_t epoch;
    };

    struct SourceState {
        TopologyDescriptor descriptor{0, 0, kInvalidSlab, 0};
        Block block;
        bool materialized = false;
    };

    struct DeletionRun {
        index_t destination = 0;
        uint64_t requested = 0;
        uint64_t matched = 0;
        uint64_t removed = 0;
        uint64_t first_match_offset = std::numeric_limits<uint64_t>::max();
    };

    struct PreparedSource {
        index_t source = 0;
        size_t group_index = 0;
        bool expansion = false;
        bool has_deletions = false;
        uint64_t successful_deletes = 0;
        uint64_t match_reads = 0;
        uint64_t final_degree = 0;
        DeletionRun first_deletion;
        std::vector<DeletionRun> remaining_deletions;
        Allocation allocation;
        Block retired_block;
    };

    struct PreparedResult {
        PreparedSource update;
        uint64_t missing_deletes = 0;
        uint64_t invalid_additions = 0;
        int64_t edge_delta = 0;
        bool changed = false;
    };

    static double Milliseconds(Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    SourceState &Source(index_t source) {
        if (source >= sources_.size()) throw std::out_of_range("source exceeds node count");
        return sources_[source];
    }

    const SourceState &Source(index_t source) const {
        if (source >= sources_.size()) throw std::out_of_range("source exceeds node count");
        return sources_[source];
    }

    void EnsureNoPendingBatch() const {
        if (batch_pending_) throw std::logic_error("publish pending chunk-store batch first");
    }

    Slab AllocateSlab(uint64_t capacity) {
        Slab slab;
        slab.capacity = capacity;
        const size_t bytes = static_cast<size_t>(capacity) * sizeof(index_t);
        if (options_.pinned) {
            const cudaError_t status = cudaHostAlloc(
                reinterpret_cast<void **>(&slab.data), bytes,
                cudaHostAllocMapped | cudaHostAllocPortable);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaHostAlloc for chunk slab failed: ") +
                    cudaGetErrorString(status));
            }
        } else {
            slab.data = new index_t[capacity];
        }
        return slab;
    }

    void ReleaseSlabs() {
        for (auto &slab : slabs_) {
            if (options_.pinned) cudaFreeHost(slab.data);
            else delete[] slab.data;
        }
        slabs_.clear();
    }

    index_t *BlockData(const Block &block) {
        return slabs_.at(block.slab_id).data + block.offset;
    }

    void EnsureAllocationsFit(const std::vector<uint64_t> &capacities) const {
        std::unordered_map<uint64_t, size_t> free_counts;
        for (const auto &entry : free_blocks_) {
            free_counts.emplace(entry.first, entry.second.size());
        }
        std::vector<uint64_t> remaining;
        remaining.reserve(slabs_.size());
        for (const auto &slab : slabs_) remaining.push_back(slab.capacity - slab.bump);
        size_t cursor = allocation_slab_cursor_;
        for (const uint64_t capacity : capacities) {
            auto free_it = free_counts.find(capacity);
            if (free_it != free_counts.end() && free_it->second != 0) {
                --free_it->second;
                continue;
            }
            bool found = false;
            for (size_t attempt = 0; attempt < remaining.size(); ++attempt) {
                const size_t slab_id = (cursor + attempt) % remaining.size();
                if (capacity <= remaining[slab_id]) {
                    remaining[slab_id] -= capacity;
                    cursor = slab_id;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error(
                    "source-local chunk arena exhausted; increase its load-stage reservation");
            }
        }
    }

    Allocation Allocate(uint64_t capacity) {
        auto free_it = free_blocks_.find(capacity);
        Allocation allocation;
        if (free_it != free_blocks_.end() && !free_it->second.empty()) {
            allocation.block = free_it->second.back();
            free_it->second.pop_back();
            allocation.reused = true;
            stats_.free_capacity_edges -= capacity;
            ++stats_.reuse_count;
        } else {
            bool found = false;
            for (size_t attempt = 0; attempt < slabs_.size(); ++attempt) {
                const size_t slab_id =
                    (allocation_slab_cursor_ + attempt) % slabs_.size();
                Slab &slab = slabs_[slab_id];
                if (capacity <= slab.capacity - slab.bump) {
                    allocation.block = {
                        static_cast<uint32_t>(slab_id), slab.bump, capacity};
                    slab.bump += capacity;
                    allocation_slab_cursor_ = slab_id;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error(
                    "source-local chunk arena exhausted; increase its load-stage reservation");
            }
            stats_.high_water_edges += capacity;
        }
        stats_.live_capacity_edges += capacity;
        ++stats_.allocation_count;
        return allocation;
    }

    void Retire(Block block, uint64_t epoch) {
        if (block.capacity == 0) return;
        retired_.push_back({block, epoch});
        stats_.live_capacity_edges -= block.capacity;
        stats_.retired_capacity_edges += block.capacity;
    }

    static DeletionRun *FindDeletionRun(
            PreparedSource &update,
            index_t destination) {
        if (!update.has_deletions) return nullptr;
        if (update.first_deletion.destination == destination) {
            return &update.first_deletion;
        }
        const auto found = std::lower_bound(
            update.remaining_deletions.begin(),
            update.remaining_deletions.end(), destination,
            [](const DeletionRun &run, index_t value) {
                return run.destination < value;
            });
        return found != update.remaining_deletions.end() &&
                found->destination == destination
            ? &*found
            : nullptr;
    }

    void BuildDeletionPlan(index_t source,
                           DestinationRange deletions,
                           PreparedSource &update) const {
        if (deletions.empty()) return;
        update.has_deletions = true;
        const SourceState &state = Source(source);
        const index_t *edges = state.descriptor.degree == 0
            ? nullptr
            : slabs_.at(state.descriptor.slab_id).data + state.descriptor.index;
        if (deletions.size() == 1) {
            update.first_deletion = {deletions.front(), 1, 0, 0};
            if (state.descriptor.degree != 0) {
                const index_t *found = std::find(
                    edges, edges + state.descriptor.degree,
                    deletions.front());
                update.match_reads = (found - edges) + (found != edges + state.descriptor.degree);
                if (found != edges + state.descriptor.degree) {
                    update.first_deletion.matched = 1;
                    update.first_deletion.first_match_offset = found - edges;
                    update.successful_deletes = 1;
                }
            }
            return;
        } else {
            std::vector<index_t> sorted(deletions.begin(), deletions.end());
            std::sort(sorted.begin(), sorted.end());
            update.first_deletion = {sorted.front(), 1, 0, 0};
            update.remaining_deletions.reserve(sorted.size() - 1);
            for (size_t i = 1; i < sorted.size(); ++i) {
                const index_t destination = sorted[i];
                if (destination == update.first_deletion.destination) {
                    ++update.first_deletion.requested;
                } else if (update.remaining_deletions.empty() ||
                           update.remaining_deletions.back().destination !=
                               destination) {
                    update.remaining_deletions.push_back(
                        {destination, 1, 0, 0});
                } else {
                    ++update.remaining_deletions.back().requested;
                }
            }
        }

        for (uint64_t offset = 0; offset < state.descriptor.degree; ++offset) {
            ++update.match_reads;
            DeletionRun *run = FindDeletionRun(update, edges[offset]);
            if (run != nullptr && run->matched < run->requested) {
                if (run->matched == 0) run->first_match_offset = offset;
                ++run->matched;
                ++update.successful_deletes;
                if (update.successful_deletes == deletions.size()) break;
            }
        }
    }

    static bool ShouldDelete(PreparedSource &update,
                             index_t destination) {
        DeletionRun *run = FindDeletionRun(update, destination);
        if (run == nullptr || run->removed == run->matched) return false;
        ++run->removed;
        return true;
    }

    void CompactSourceInPlace(PreparedSource &update,
                              const SourceMutationView &mutations,
                              ChunkSourceMutationMetrics &metrics) {
        SourceState &state = Source(update.source);
        index_t *edges = state.block.capacity == 0 ? nullptr : BlockData(state.block);
        const uint64_t old_degree = state.descriptor.degree;
        uint64_t write_offset = old_degree;
        if (update.successful_deletes == 1 &&
            update.first_deletion.requested == 1 &&
            update.remaining_deletions.empty()) {
            const uint64_t deleted_offset =
                update.first_deletion.first_match_offset;
            const uint64_t trailing = old_degree - deleted_offset - 1;
            if (trailing != 0) {
                std::memmove(edges + deleted_offset, edges + deleted_offset + 1,
                             trailing * sizeof(index_t));
                metrics.mutation_edge_reads += trailing;
                metrics.mutation_written_bytes += trailing * sizeof(index_t);
            }
            write_offset = old_degree - 1;
            update.first_deletion.removed = 1;
        } else if (update.successful_deletes != 0) {
            write_offset = 0;
            for (uint64_t read_offset = 0; read_offset < old_degree;
                 ++read_offset) {
                ++metrics.mutation_edge_reads;
                const index_t destination = edges[read_offset];
                if (ShouldDelete(update, destination)) continue;
                if (write_offset != read_offset) {
                    edges[write_offset] = destination;
                    metrics.mutation_written_bytes += sizeof(index_t);
                }
                ++write_offset;
            }
        }
        if (!mutations.additions.empty()) {
            std::memcpy(edges + write_offset, mutations.additions.data(),
                        mutations.additions.size() * sizeof(index_t));
            metrics.mutation_written_bytes +=
                mutations.additions.size() * sizeof(index_t);
            write_offset += mutations.additions.size();
        }
        if (write_offset != update.final_degree) {
            throw std::logic_error("source compaction degree mismatch");
        }
        state.descriptor.degree = static_cast<index_t>(write_offset);
        state.descriptor.version = static_cast<uint32_t>(pending_epoch_);
    }

    void RewriteSourceToNewBlock(PreparedSource &update,
                                 const SourceMutationView &mutations,
                                 ChunkSourceMutationMetrics &metrics) {
        SourceState &state = Source(update.source);
        const Allocation &allocation = update.allocation;

        const index_t *old_edges = state.descriptor.degree == 0
            ? nullptr
            : BlockData(state.block);
        index_t *new_edges = BlockData(allocation.block);
        uint64_t write_offset = state.descriptor.degree;
        if (update.successful_deletes == 0) {
            if (write_offset != 0) {
                std::memcpy(new_edges, old_edges,
                            write_offset * sizeof(index_t));
            }
        } else if (update.successful_deletes == 1 &&
                   update.first_deletion.requested == 1 &&
                   update.remaining_deletions.empty()) {
            const uint64_t deleted_offset =
                update.first_deletion.first_match_offset;
            const uint64_t trailing = state.descriptor.degree -
                deleted_offset - 1;
            if (deleted_offset != 0) {
                std::memcpy(new_edges, old_edges,
                            deleted_offset * sizeof(index_t));
            }
            if (trailing != 0) {
                std::memcpy(new_edges + deleted_offset,
                            old_edges + deleted_offset + 1,
                            trailing * sizeof(index_t));
            }
            --write_offset;
            update.first_deletion.removed = 1;
        } else {
            write_offset = 0;
            for (uint64_t read_offset = 0;
                 read_offset < state.descriptor.degree; ++read_offset) {
                const index_t destination = old_edges[read_offset];
                if (ShouldDelete(update, destination)) continue;
                new_edges[write_offset++] = destination;
            }
        }
        metrics.mutation_edge_reads += update.successful_deletes != 0 &&
            !(update.successful_deletes == 1 && update.first_deletion.requested == 1 &&
              update.remaining_deletions.empty()) ? state.descriptor.degree : write_offset;
        metrics.relocation_copied_bytes += write_offset * sizeof(index_t);
        metrics.mutation_written_bytes += write_offset * sizeof(index_t);
        if (!mutations.additions.empty()) {
            std::memcpy(new_edges + write_offset, mutations.additions.data(),
                        mutations.additions.size() * sizeof(index_t));
            metrics.mutation_written_bytes +=
                mutations.additions.size() * sizeof(index_t);
            write_offset += mutations.additions.size();
        }
        if (write_offset != update.final_degree) {
            throw std::logic_error("source rewrite degree mismatch");
        }
        if (state.block.capacity != 0) {
            update.retired_block = state.block;
        }
        state.block = allocation.block;
        state.descriptor.index = allocation.block.offset;
        state.descriptor.slab_id = allocation.block.slab_id;
        state.descriptor.degree = static_cast<index_t>(write_offset);
        state.descriptor.version = static_cast<uint32_t>(pending_epoch_);
    }

    std::vector<SourceState> sources_;
    std::vector<index_t> changed_sources_;
    ChunkArenaOptions options_;
    std::vector<Slab> slabs_;
    size_t allocation_slab_cursor_;
    std::unordered_map<uint64_t, std::vector<Block>> free_blocks_;
    std::vector<RetiredBlock> retired_;
    ChunkArenaStats stats_;
    uint64_t logical_edge_count_;
    uint64_t materialized_edge_count_;
    uint64_t published_epoch_;
    uint64_t pending_epoch_;
    bool batch_pending_;
    std::unique_ptr<concurrency::FixedWorkerPool> mutation_workers_;
};

} // namespace topology
} // namespace sepgraph

#endif // SEPGRAPH_SOURCE_LOCAL_CHUNK_STORE_H
