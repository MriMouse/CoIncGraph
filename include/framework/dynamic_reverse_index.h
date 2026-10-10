#ifndef SEPGRAPH_DYNAMIC_REVERSE_INDEX_H
#define SEPGRAPH_DYNAMIC_REVERSE_INDEX_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <groute/graphs/common.h>
#include <framework/effective_update_batch.h>
#include <framework/effective_delta_sort.h>
#include <utils/fixed_worker_pool.h>

namespace sepgraph {
namespace runtime {

class DynamicReverseIndex {
public:
    struct MaterializeMetrics {
        uint64_t base_edges_scanned = 0;
        uint64_t delta_records_scanned = 0;
        uint64_t output_sources = 0;
    };

    struct PrepareMetrics {
        double reset_copy_ms = 0, sort_ms = 0, group_ms = 0;
        double slots_ms = 0, merge_ms = 0, release_ms = 0, commit_ms = 0;
        uint64_t input_copy_bytes = 0;
        uint64_t input_records = 0, old_records_read = 0, output_records = 0, overlay_records = 0;
        size_t destinations = 0, new_destinations = 0, buckets_before = 0, buckets_after = 0;
    };
    const PrepareMetrics &LastPrepareMetrics() const { return prepare_metrics_; }

    DynamicReverseIndex() : nnodes_(0) {}
    DynamicReverseIndex(const DynamicReverseIndex &) = delete;
    DynamicReverseIndex &operator=(const DynamicReverseIndex &) = delete;

    template <typename PMAGraph>
    void Build(const PMAGraph &graph, uint32_t requested_workers, uint32_t shards = 1) {
        nnodes_ = graph.nnodes;
        overlay_records_ = 0;
        offsets_.assign(static_cast<size_t>(nnodes_) + 1, 0);
        sources_.clear();
        deltas_.clear();
        deltas_.resize(std::max<uint32_t>(1, shards));
        pending_.clear();
        merge_workers_.reset(new concurrency::FixedWorkerPool(
            std::max<uint32_t>(1, requested_workers)));

        if (nnodes_ == 0) {
            return;
        }

        const uint32_t workers = std::max<uint32_t>(
            1, std::min<uint32_t>(requested_workers, nnodes_));
        std::unique_ptr<std::atomic<uint32_t>[]> incoming_counts(
            new std::atomic<uint32_t>[nnodes_]);
        for (index_t node = 0; node < nnodes_; ++node) {
            incoming_counts[node].store(0, std::memory_order_relaxed);
        }

        ParallelForVertexRanges(graph, workers, [&](index_t, index_t dst) {
            if (dst < nnodes_) {
                incoming_counts[dst].fetch_add(1, std::memory_order_relaxed);
            }
        });

        for (index_t node = 0; node < nnodes_; ++node) {
            offsets_[static_cast<size_t>(node) + 1] =
                offsets_[node] + incoming_counts[node].load(std::memory_order_relaxed);
        }
        sources_.resize(offsets_.back());

        std::unique_ptr<std::atomic<uint64_t>[]> write_offsets(
            new std::atomic<uint64_t>[nnodes_]);
        for (index_t node = 0; node < nnodes_; ++node) {
            write_offsets[node].store(offsets_[node], std::memory_order_relaxed);
        }
        ParallelForVertexRanges(graph, workers, [&](index_t src, index_t dst) {
            if (dst < nnodes_) {
                const uint64_t position =
                    write_offsets[dst].fetch_add(1, std::memory_order_relaxed);
                sources_[position] = src;
            }
        });
        for (index_t dst = 0; dst < nnodes_; ++dst) {
            std::sort(sources_.begin() + offsets_[dst],
                      sources_.begin() + offsets_[static_cast<size_t>(dst) + 1]);
        }
    }

    // Only forward-authorized deltas enter here. Prepare leaves the visible
    
    void Prepare(const std::vector<topology::EffectiveEdgeDelta> &effective) {
        const auto start = Clock::now();
        auto records = effective;
        const double copy_ms = Elapsed(start);
        Prepare(std::move(records));
        prepare_metrics_.reset_copy_ms += copy_ms;
        prepare_metrics_.input_copy_bytes = effective.size() * sizeof(topology::EffectiveEdgeDelta);
    }

    // Consume the forward-authorized buffer: the producer no longer needs its
    // source ordering after preflight. Sorting cannot change visible topology.
    void Prepare(std::vector<topology::EffectiveEdgeDelta> &&effective) {
        prepare_metrics_ = {};
        auto start = Clock::now();
        for (const auto &update : pending_) {
            auto &shard = DeltaShard(update.destination);
            const auto found = shard.find(update.destination);
            if (found != shard.end() && found->second.empty()) shard.erase(found);
        }
        pending_.clear();
        auto records = std::move(effective);
        records.erase(std::remove_if(records.begin(), records.end(),
            [&](const topology::EffectiveEdgeDelta &edge) {
                return edge.source >= nnodes_ || edge.destination >= nnodes_ || edge.count == 0;
            }), records.end());
        prepare_metrics_.input_records = records.size();
        prepare_metrics_.reset_copy_ms = Elapsed(start);
        start = Clock::now();
        topology::SortEffectiveDeltas(records);
        prepare_metrics_.sort_ms = Elapsed(start);
        start = Clock::now();
        std::vector<size_t> shard_begin(deltas_.size() + 1, 0);
        size_t next_shard = 0;
        for (size_t begin = 0; begin < records.size();) {
            const auto dst = records[begin].destination;
            size_t end = begin + 1;
            while (end < records.size() && records[end].destination == dst) ++end;
            const size_t shard = ShardIndex(dst);
            while (next_shard <= shard) shard_begin[next_shard++] = pending_.size();
            pending_.push_back({dst, nullptr, {}, begin, end});
            begin = end;
        }
        while (next_shard <= deltas_.size()) shard_begin[next_shard++] = pending_.size();
        prepare_metrics_.group_ms = Elapsed(start);
        prepare_metrics_.destinations = pending_.size();
        for (const auto &shard : deltas_) prepare_metrics_.buckets_before += shard.bucket_count();
        start = Clock::now();
        std::vector<size_t> new_destinations(deltas_.size(), 0);
        auto prepare_shard = [&](size_t index) {
            auto &shard = deltas_[index];
            for (size_t i = shard_begin[index]; i < shard_begin[index + 1]; ++i)
                if (shard.find(pending_[i].destination) == shard.end()) ++new_destinations[index];
            const size_t required = shard.size() + new_destinations[index];
            if (required > shard.bucket_count() * shard.max_load_factor()) shard.reserve(required);
            for (size_t i = shard_begin[index]; i < shard_begin[index + 1]; ++i)
                pending_[i].target = &shard[pending_[i].destination];
        };
        // Each task owns a distinct container and a disjoint pending range.
        // Finish all rehash/insert work before merging via stable references.
        if (merge_workers_ && deltas_.size() > 1)
            merge_workers_->Run(deltas_.size(), prepare_shard, 1);
        else for (size_t i = 0; i < deltas_.size(); ++i) prepare_shard(i);
        prepare_metrics_.slots_ms = Elapsed(start);
        for (size_t i = 0; i < deltas_.size(); ++i) {
            prepare_metrics_.new_destinations += new_destinations[i];
            prepare_metrics_.buckets_after += deltas_[i].bucket_count();
        }
        start = Clock::now();
        auto merge = [&](size_t index) {
            auto &update = pending_[index];
            const size_t begin = update.begin, end = update.end;
            const auto &old = *update.target;
            auto &merged = update.records;
            merged.reserve(old.size() + end - begin);
            size_t i = 0, j = begin;
            while (i < old.size() || j < end) {
                const auto src = std::min(i < old.size() ? old[i].source :
                    std::numeric_limits<index_t>::max(), j < end ? records[j].source :
                    std::numeric_limits<index_t>::max());
                int64_t count = 0;
                if (i < old.size() && old[i].source == src) count += old[i++].count;
                while (j < end && records[j].source == src) count += records[j++].count;
                if (count) merged.push_back({src, count});
            }
        };
        if (merge_workers_) merge_workers_->Run(pending_.size(), merge);
        else for (size_t i = 0; i < pending_.size(); ++i) merge(i);
        prepare_metrics_.merge_ms = Elapsed(start);
        start = Clock::now();
        std::vector<topology::EffectiveEdgeDelta>().swap(records);
        prepare_metrics_.release_ms = Elapsed(start);
    }

    void Commit() noexcept {
        const auto start = Clock::now();
        for (auto &update : pending_) {
            prepare_metrics_.old_records_read += update.target->size();
            prepare_metrics_.output_records += update.records.size();
            overlay_records_ -= update.target->size();
            overlay_records_ += update.records.size();
            update.target->swap(update.records);
        }
        prepare_metrics_.overlay_records = overlay_records_;
        for (const auto &update : pending_) {
            if (update.target->empty()) DeltaShard(update.destination).erase(update.destination);
        }
        pending_.clear();
        prepare_metrics_.commit_ms = Elapsed(start);
    }

    template <typename Visitor>
    void ForEachIncoming(index_t dst, Visitor visitor) const {
        MergeIncoming(dst, visitor, nullptr);
    }

    // Returning false stops the row immediately (including its delta merge).
    // Useful for witness/existence queries on very high-indegree vertices.
    template <typename Visitor>
    void ForEachIncomingWhile(index_t dst, Visitor visitor) const {
        MergeIncoming(dst, visitor, nullptr);
    }

    MaterializeMetrics MaterializeIncoming(
                             const std::vector<index_t> &destinations,
                             std::vector<uint64_t> &offsets,
                             std::vector<index_t> &sources) const {
        MaterializeMetrics metrics;
        offsets.assign(destinations.size() + 1, 0);
        sources.clear();
        // Each tile merges once into private storage. Prefix/copy preserves the
        // caller's destination order (including repeats) without atomics or a
        // second base/delta traversal. Scratch is host-only and dies on return.
        if (merge_workers_ && merge_workers_->WorkerCount() > 1 && destinations.size() >= 4096) {
            constexpr size_t tile_size = 4096;
            const size_t tiles = (destinations.size() + tile_size - 1) / tile_size;
            struct Tile {
                std::vector<index_t> sources;
                MaterializeMetrics metrics;
            };
            std::vector<Tile> scratch(tiles);
            merge_workers_->Run(tiles, [&](size_t tile) {
                auto &local = scratch[tile];
                const size_t begin = tile * tile_size;
                const size_t end = std::min(begin + tile_size, destinations.size());
                for (size_t i = begin; i < end; ++i) {
                    const size_t before = local.sources.size();
                    MergeIncoming(destinations[i], [&](index_t src) {
                        local.sources.push_back(src);
                    }, &local.metrics);
                    offsets[i + 1] = local.sources.size() - before;
                }
            }, 1);
            for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
            sources.resize(offsets.back());
            merge_workers_->Run(tiles, [&](size_t tile) {
                const auto &local = scratch[tile].sources;
                std::copy(local.begin(), local.end(), sources.begin() + offsets[tile * tile_size]);
            }, 1);
            for (const auto &tile : scratch) {
                metrics.base_edges_scanned += tile.metrics.base_edges_scanned;
                metrics.delta_records_scanned += tile.metrics.delta_records_scanned;
            }
            metrics.output_sources = sources.size();
            return metrics;
        }
        for (size_t i = 0; i < destinations.size(); ++i) {
            MergeIncoming(destinations[i], [&](index_t src) {
                sources.push_back(src);
            }, &metrics);
            offsets[i + 1] = sources.size();
        }
        metrics.output_sources = sources.size();
        return metrics;
    }

    uint64_t BaseEdgeCount() const { return sources_.size(); }

private:
    using Clock = std::chrono::steady_clock;
    static double Elapsed(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    PrepareMetrics prepare_metrics_;
    uint64_t overlay_records_ = 0;
    template <typename PMAGraph, typename Visitor>
    static void ParallelForVertexRanges(const PMAGraph &graph,
                                        uint32_t workers,
                                        Visitor visitor) {
        const auto adjacency = graph.adjacency_view();
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (uint32_t worker = 0; worker < workers; ++worker) {
            const index_t begin_node = static_cast<index_t>(
                (static_cast<uint64_t>(graph.nnodes) * worker) / workers);
            const index_t end_node = static_cast<index_t>(
                (static_cast<uint64_t>(graph.nnodes) * (worker + 1)) / workers);
            threads.emplace_back([adjacency, begin_node, end_node, &visitor]() {
                for (index_t src = begin_node; src < end_node; ++src) {
                    const uint64_t degree = adjacency.Degree(src);
                    for (uint64_t offset = 0; offset < degree; ++offset) {
                        visitor(src, adjacency.EdgeAt(src, offset));
                    }
                }
            });
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }

    template <typename Visitor>
    void MergeIncoming(index_t dst, Visitor visitor,
                       MaterializeMetrics *metrics) const {
        if (dst >= nnodes_) return;
        const uint64_t begin = offsets_[dst];
        const uint64_t end = offsets_[static_cast<size_t>(dst) + 1];
        if (metrics != nullptr) metrics->base_edges_scanned += end - begin;

        const auto &shard = DeltaShard(dst);
        const auto delta_it = shard.find(dst);
        const std::vector<Delta> empty;
        const auto &sorted_delta = delta_it == shard.end() ? empty : delta_it->second;
        if (metrics != nullptr) {
            metrics->delta_records_scanned += sorted_delta.size();
        }

        uint64_t base_pos = begin;
        size_t delta_pos = 0;
        while (base_pos < end || delta_pos < sorted_delta.size()) {
            const index_t base_src = base_pos < end
                ? sources_[base_pos] : std::numeric_limits<index_t>::max();
            const index_t delta_src = delta_pos < sorted_delta.size()
                ? sorted_delta[delta_pos].source : std::numeric_limits<index_t>::max();
            const index_t src = std::min(base_src, delta_src);
            int64_t count = 0;
            while (base_pos < end && sources_[base_pos] == src) {
                ++count;
                ++base_pos;
            }
            if (delta_pos < sorted_delta.size() && sorted_delta[delta_pos].source == src) {
                count += sorted_delta[delta_pos++].count;
            }
            if (count > 0) {
                if constexpr (std::is_same<decltype(visitor(src)), bool>::value) {
                    if (!visitor(src)) return;
                } else {
                    visitor(src);
                }
            }
        }
    }

    index_t nnodes_;
    std::vector<uint64_t> offsets_;
    std::vector<index_t> sources_;
    struct Delta { index_t source; int64_t count; };
    struct PendingDestination {
        index_t destination;
        std::vector<Delta> *target;
        std::vector<Delta> records;
        size_t begin, end;
    };
    using DeltaMap = std::unordered_map<index_t, std::vector<Delta>>;
    // Monotonic mapping keeps destination-sorted pending ranges contiguous.
    size_t ShardIndex(index_t dst) const {
        return static_cast<uint64_t>(dst) * deltas_.size() / std::max<index_t>(1, nnodes_);
    }
    DeltaMap &DeltaShard(index_t dst) { return deltas_[ShardIndex(dst)]; }
    const DeltaMap &DeltaShard(index_t dst) const { return deltas_[ShardIndex(dst)]; }
    std::vector<DeltaMap> deltas_{1};
    std::vector<PendingDestination> pending_;
    std::unique_ptr<concurrency::FixedWorkerPool> merge_workers_;
};

} // namespace runtime
} // namespace sepgraph

#endif
