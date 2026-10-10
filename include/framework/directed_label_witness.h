#ifndef SEPGRAPH_DIRECTED_LABEL_WITNESS_H
#define SEPGRAPH_DIRECTED_LABEL_WITNESS_H

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sepgraph { namespace runtime {

// A batch-local proof, never an equal-label/parent heuristic. Search the
// committed post-delete reverse graph for a path from label[v] to v. Every

// A bounded search may decline a proof, in which case ordinary invalidation
// and GPU repair remain responsible for the answer.
class DirectedLabelWitness {
public:
    explicit DirectedLabelWitness(const std::vector<uint32_t>& labels,
                                  uint64_t edge_budget = 1000000,
                                  uint64_t vertex_budget = 100000)
        : labels_(labels), edge_budget_(edge_budget), vertex_budget_(vertex_budget) {}

    // Build shared proof anchors in the *forward* direction from actual roots.
    // Reverse DFS alone can spend its entire batch budget inside a giant
    // equal-label SCC before finding its (arbitrarily numbered) root entrance.
    
    // queries can stop at any of them. This is a batch-local certificate, not
    // an ID-order assumption or a surviving-parent heuristic.
    template<class Outgoing>
    void SeedRoots(const std::vector<uint32_t>& destinations, Outgoing outgoing,
                   uint64_t edge_limit = 262144, uint64_t vertex_limit = 32768) {
        std::vector<uint32_t> queue;
        for (auto v : destinations) {
            const auto root = labels_[v];
            if (root < labels_.size() && labels_[root] == root &&
                certified_.insert(root).second) queue.push_back(root);
        }
        for (std::size_t i = 0; i < queue.size() && seed_edges_ < edge_limit &&
                seed_vertices_ < vertex_limit; ++i) {
            const auto source = queue[i];
            ++seed_vertices_;
            outgoing(source, [&](uint32_t dst) {
                ++seed_edges_;
                if (labels_[dst] == labels_[source] &&
                    certified_.insert(dst).second) queue.push_back(dst);
                return seed_edges_ < edge_limit;
            });
        }
        seed_certified_ = certified_.size();
    }

    uint64_t SeedEdges() const { return seed_edges_; }
    uint64_t SeedVertices() const { return seed_vertices_; }
    uint64_t SeedCertified() const { return seed_certified_; }

    template<class Reverse>
    bool Prove(uint32_t destination, const Reverse& reverse) {
        const uint32_t label = labels_[destination];
        if (destination == label || certified_.count(destination)) return true;
        if (edges_ >= edge_budget_ || vertices_ >= vertex_budget_) return false;
        // libstdc++ unordered_map::clear also zeroes the entire retained bucket
        // array. One high-indegree search can grow it enormously; clearing it
        // for every subsequent tiny proof then costs O(peak size * queries),
        // even though the reported edge/vertex work stays small. Erase only
        // live entries so reset work follows the preceding search's work.
        reset_entries_ += child_.size();
        avoided_bucket_slots_ += child_.bucket_count();
        while (!child_.empty()) child_.erase(child_.begin());
        pending_.clear();
        child_.emplace(destination, destination);
        pending_.push_back(destination);
        while (!pending_.empty() && edges_ < edge_budget_ && vertices_ < vertex_budget_) {
            const uint32_t node = pending_.back();
            pending_.pop_back();
            ++vertices_;
            uint32_t witness = UINT32_MAX;
            neighbors_.clear();
            reverse.ForEachIncomingWhile(node, [&](uint32_t source) {
                ++edges_;
                if (labels_[source] != label) return edges_ < edge_budget_;
                if (source == label || certified_.count(source)) {
                    witness = source;
                    return false;
                } else if (edges_ <= edge_budget_ && child_.emplace(source, node).second) {
                    neighbors_.push_back(source);
                }
                return edges_ < edge_budget_;
            });
            if (witness != UINT32_MAX) {
                // source -> node -> ... -> destination is a real surviving path.
                for (uint32_t v = node;; v = child_.at(v)) {
                    certified_.insert(v);
                    if (v == destination) break;
                }
                return true;
            }
            // Reverse rows are sorted. Explore low IDs first, tending toward
            // the minimum-ID root without requiring a global graph traversal.
            pending_.insert(pending_.end(), neighbors_.rbegin(), neighbors_.rend());
        }
        return false;
    }

    uint64_t Edges() const { return edges_; }
    uint64_t Vertices() const { return vertices_; }
    uint64_t ResetEntries() const { return reset_entries_; }
    uint64_t AvoidedBucketSlots() const { return avoided_bucket_slots_; }

    // Close the uncertified region, proving live boundary vertices before
    // crossing them. A lost leaf/branch must not invalidate a still-rooted giant
    // component just because an outgoing edge carries the same old label.
    // On budget exhaustion the caller discards this partial set and uses the
    // ordinary GPU invalidation frontier with its original seeds.
    template<class Reverse, class Outgoing>
    bool ExpandAffected(std::vector<uint32_t>& affected, const Reverse& reverse,
                        Outgoing outgoing, uint64_t& forward_edges) {
        std::unordered_set<uint32_t> seen(affected.begin(), affected.end());
        affected.assign(seen.begin(), seen.end());
        forward_edges = 0;
        for (std::size_t i = 0; i < affected.size(); ++i) {
            if (affected.size() > 100000 || edges_ >= edge_budget_ ||
                vertices_ >= vertex_budget_) return false;
            const uint32_t source = affected[i];
            bool complete = true;
            outgoing(source, [&](uint32_t dst) {
                if (++forward_edges > edge_budget_) { complete = false; return false; }
                if (dst == source || labels_[dst] != labels_[source] || labels_[dst] == dst ||
                    !seen.insert(dst).second) return true;
                if (!Prove(dst, reverse)) affected.push_back(dst);
                if (edges_ >= edge_budget_ || vertices_ >= vertex_budget_) {
                    complete = false;
                    return false;
                }
                return true;
            });
            if (!complete) return false;
        }
        return true;
    }

private:
    const std::vector<uint32_t>& labels_;
    uint64_t edge_budget_, vertex_budget_, edges_ = 0, vertices_ = 0;
    uint64_t reset_entries_ = 0, avoided_bucket_slots_ = 0;
    uint64_t seed_edges_ = 0, seed_vertices_ = 0, seed_certified_ = 0;
    std::unordered_set<uint32_t> certified_;
    std::unordered_map<uint32_t, uint32_t> child_;
    std::vector<uint32_t> pending_, neighbors_;
};

}} // namespace sepgraph::runtime
#endif
