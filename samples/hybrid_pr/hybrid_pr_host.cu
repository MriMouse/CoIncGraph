#include "hybrid_pr_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>

bool PageRankCheck(const sepgraph::topology::SourceLocalChunkStore &graph,
                  const std::vector<rank_t> &ranks, const std::vector<rank_t> &residual,
                  double epsilon, const char *stage, int batch) {
    const size_t n = ranks.size();
    if (residual.size() != n) return false;
    // Independent double-precision Jacobi fixed point, no incremental state.
    auto multiply = [&](const std::vector<double> &x, std::vector<double> &next) {
        std::fill(next.begin(), next.end(), 0.15);
        for (index_t src = 0; src < n; ++src) {
            const auto &row = graph.Descriptor(src);
            if (!row.degree) continue;
            const double contribution = 0.85 * x[src] / row.degree;
            const auto *edges = graph.SlabData(row.slab_id) + row.index;
            for (index_t e = 0; e < row.degree; ++e) next[edges[e]] += contribution;
        }
    };
    std::vector<double> reference(n, 0), next(n), actual(ranks.begin(), ranks.end());
    bool converged = false;
    for (unsigned round = 0; round < 10000; ++round) {
        multiply(reference, next);
        double delta = 0;
        for (size_t v = 0; v < n; ++v) delta += std::abs(next[v] - reference[v]);
        reference.swap(next);
        if (delta <= 1e-12 * std::max<size_t>(1, n)) { converged = true; break; }
    }
    multiply(actual, next);
    double l1 = 0, defect = 0, max_residual = 0, max_error = 0;
    bool finite = true;
    for (size_t v = 0; v < n; ++v) {
        finite &= std::isfinite(ranks[v]) && std::isfinite(residual[v]) && ranks[v] >= -epsilon;
        const double error = std::abs(ranks[v] - reference[v]);
        l1 += error;
        max_error = std::max(max_error, error);
        defect += std::abs(next[v] - ranks[v] - residual[v]);
        max_residual = std::max(max_residual, double(std::abs(residual[v])));
    }
    // For substochastic P, ||x-x*||_1 <= ||r||_1/(1-alpha).
    // A separate floating-point allowance covers rank accumulation and summation.
    double mass = 0;
    for (double x : reference) mass += std::abs(x);
    const double roundoff = 64 * std::numeric_limits<float>::epsilon() * std::max(1.0, mass);
    const double bound = (n * epsilon + roundoff) / 0.15;
    const bool ok = finite && converged && l1 <= bound && defect <= roundoff &&
                    max_residual <= static_cast<float>(epsilon);
    std::printf("[PR-CHECK] stage=%s batch=%d %s l1_error=%.9g bound=%.9g max_error=%.9g invariant_l1=%.9g max_residual=%.9g\n",
                stage, batch, ok ? "passed" : "failed", l1, bound, max_error, defect, max_residual);
    return ok;
}

bool PageRankOutput(const char *path, const std::vector<rank_t> &ranks,
                    const std::vector<rank_t> &residual) {
    std::ofstream output(path);
    if (!output) { std::fprintf(stderr, "Cannot write PR output: %s\n", path); return false; }
    output << std::setprecision(std::numeric_limits<rank_t>::max_digits10);
    for (size_t v = 0; v < ranks.size(); ++v) output << v << ' ' << ranks[v] << ' ' << residual[v] << '\n';
    output.close();
    return bool(output);
}
