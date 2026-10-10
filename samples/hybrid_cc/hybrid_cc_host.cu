#include "hybrid_cc_common.h"
#include <fstream>

using sepgraph::topology::SourceLocalChunkStore;


// an already visited vertex and its descendants have a smaller reaching seed.
uint64_t CCCheck(const SourceLocalChunkStore &graph, const std::vector<label_t> &labels) {
    std::vector<label_t> expected(labels.size(), UINT32_MAX);
    std::vector<uint32_t> stack;
    for (uint32_t seed = 0; seed < labels.size(); ++seed) {
        if (expected[seed] != UINT32_MAX) continue;
        expected[seed] = seed;
        stack.push_back(seed);
        while (!stack.empty()) {
            const uint32_t u = stack.back();
            stack.pop_back();
            const auto &d = graph.Descriptor(u);
            if (d.degree == 0) continue;
            const auto *row = graph.SlabData(d.slab_id) + d.index;
            for (uint32_t i = 0; i < d.degree; ++i) {
                const uint32_t v = row[i];
                if (expected[v] != UINT32_MAX) continue;
                expected[v] = seed;
                stack.push_back(v);
            }
        }
    }
    uint64_t errors = 0;
    for (uint32_t u = 0; u < labels.size(); ++u) errors += labels[u] != expected[u];
    return errors;
}

bool CCOutput(const char *path, const std::vector<label_t> &labels) {
    std::ofstream output(path);
    for (size_t u = 0; u < labels.size(); ++u) output << u << ' ' << labels[u] << '\n';
    output.close();
    return !output.fail();
}
