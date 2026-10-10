#ifndef SEPGRAPH_UNDIRECTED_INPUT_H
#define SEPGRAPH_UNDIRECTED_INPUT_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sepgraph { namespace topology {

struct UndirectedInputEdge { uint32_t u, v; };

// Reciprocal occurrences are two representations of the same undirected edge.
// Repeated occurrences in one direction remain parallel edges: multiplicity is
// max(count(u,v), count(v,u)), never their sum. Self-loops are emitted once.
class UndirectedInput {
public:
    template<class ReadEdge>
    UndirectedInput(uint32_t nodes, size_t count, ReadEdge read) {
        edges_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto edge = read(i);
            if (edge.u >= nodes || edge.v >= nodes)
                throw std::invalid_argument("CC input endpoint outside graph");
            edges_.push_back({edge.u, edge.v});
        }
        std::sort(edges_.begin(), edges_.end(), [](auto a, auto b) {
            const uint32_t amin = std::min(a.u, a.v), bmin = std::min(b.u, b.v);
            if (amin != bmin) return amin < bmin;
            const uint32_t amax = std::max(a.u, a.v), bmax = std::max(b.u, b.v);
            if (amax != bmax) return amax < bmax;
            return a.u < b.u;
        });
    }

    template<class Visit>
    void ForEachPair(Visit visit) const {
        for (size_t i = 0; i < edges_.size();) {
            const auto u = std::min(edges_[i].u, edges_[i].v);
            const auto v = std::max(edges_[i].u, edges_[i].v);
            uint64_t forward = 0, reverse = 0;
            do {
                if (edges_[i].u == u) ++forward; else ++reverse;
                ++i;
            } while (i < edges_.size() && std::min(edges_[i].u, edges_[i].v) == u &&
                     std::max(edges_[i].u, edges_[i].v) == v);
            visit(u, v, std::max(forward, reverse));
        }
    }

    void CSR(uint32_t nodes, std::vector<uint64_t> &offsets,
             std::vector<uint32_t> &neighbors) const {
        offsets.assign(size_t(nodes) + 1, 0);
        ForEachPair([&](uint32_t u, uint32_t v, uint64_t count) {
            offsets[u + 1] += count;
            if (u != v) offsets[v + 1] += count;
        });
        for (size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i] > UINT32_MAX)
                throw std::overflow_error("CC vertex degree exceeds uint32");
            offsets[i] += offsets[i - 1];
        }
        neighbors.resize(offsets.back());
        auto cursor = offsets;
        ForEachPair([&](uint32_t u, uint32_t v, uint64_t count) {
            for (uint64_t i = 0; i < count; ++i) {
                neighbors[cursor[u]++] = v;
                if (u != v) neighbors[cursor[v]++] = u;
            }
        });
    }

    template<class Edges>
    size_t Append(Edges &output) const {
        const size_t begin = output.size();
        uint64_t count = 0;
        ForEachPair([&](uint32_t u, uint32_t v, uint64_t n) { count += n * (u == v ? 1 : 2); });
        if (count > UINT32_MAX || begin > UINT32_MAX - count)
            throw std::overflow_error("CC normalized update stream exceeds uint32");
        output.resize(begin + count);
        size_t position = begin;
        ForEachPair([&](uint32_t u, uint32_t v, uint64_t n) {
            for (uint64_t i = 0; i < n; ++i) {
                output[position].u = u; output[position].v = v; output[position++].w = 0;
                if (u != v) {
                    output[position].u = v; output[position].v = u; output[position++].w = 0;
                }
            }
        });
        return output.size() - begin;
    }
private:
    std::vector<UndirectedInputEdge> edges_;
};

// File counts describe original records. Rewrite only the in-memory sizes and

template<class Loader>
void NormalizeUndirectedUpdates(Loader &updates, uint32_t nodes) {
    decltype(updates.added_edges_w) additions, deletions;
    size_t add_begin = 0, del_begin = 0;
    for (auto &batch : updates.m_batch_size) {
        const auto input_add = batch.first, input_del = batch.second;
        UndirectedInput add(nodes, input_add, [&](size_t i) { return updates.added_edges_w[add_begin + i]; });
        batch.first = add.Append(additions);
        UndirectedInput del(nodes, input_del, [&](size_t i) { return updates.deleted_edges_w[del_begin + i]; });
        batch.second = del.Append(deletions);
        add_begin += input_add;
        del_begin += input_del;
    }
    updates.added_edges_w.swap(additions);
    updates.deleted_edges_w.swap(deletions);
    updates.m_add_size = updates.added_edges_w.size();
    updates.m_del_size = updates.deleted_edges_w.size();
}

} }
#endif
