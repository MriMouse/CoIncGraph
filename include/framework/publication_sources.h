#ifndef CG_PUBLICATION_SOURCES_H
#define CG_PUBLICATION_SOURCES_H

#include <algorithm>
#include <iterator>
#include <vector>
#include <cstddef>

namespace sepgraph { namespace topology {
// Each phase supplies source-ordered changed IDs, including only effective changes.

template <typename T>
void MergePublicationSources(std::vector<T>& sources, std::size_t split,
                             std::vector<T>& scratch) {
    scratch.clear();
    scratch.reserve(sources.size());
    std::set_union(sources.begin(), sources.begin() + split,
                   sources.begin() + split, sources.end(),
                   std::back_inserter(scratch));
    scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
    sources.swap(scratch);
}
}}
#endif
