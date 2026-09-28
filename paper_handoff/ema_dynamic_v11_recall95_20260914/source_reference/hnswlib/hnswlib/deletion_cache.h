#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hnswlib {

// One point worker owns this bounded, lazy geometry cache. Ordered keys also
// support asymmetric native distance functions; no metric symmetry is assumed.
template<class Id, class Distance>
class DeletionDistanceCache {
    struct Entry {
        uint64_t key{0};
        Distance value;
        bool valid{false};
    };
    std::unordered_map<Id, size_t> positions_;
    size_t width_{0};
    std::unique_ptr<Distance[]> values_;
    std::vector<uint64_t> valid_;
    std::vector<Entry> overflow_;

 public:
    size_t hits{0};
    size_t misses{0};

    DeletionDistanceCache(Id deleted, const std::vector<std::pair<Distance, Id>>& candidates)
        : overflow_(4096) {
        const size_t dense_limit = 512;
        positions_.reserve(std::min(dense_limit, candidates.size() + 1));
        positions_.emplace(deleted, width_++);
        for (const auto& candidate : candidates) {
            if (width_ == dense_limit) break;
            if (positions_.emplace(candidate.second, width_).second) ++width_;
        }
        values_.reset(new Distance[width_ * width_]);
        valid_.assign((width_ * width_ + 63) / 64, 0);
    }

    template<class Evaluate>
    Distance get(Id from, Id to, Evaluate evaluate) {
        auto a = positions_.find(from);
        auto b = positions_.find(to);
        if (a != positions_.end() && b != positions_.end()) {
            size_t position = a->second * width_ + b->second;
            uint64_t bit = uint64_t(1) << (position & 63);
            auto& word = valid_[position >> 6];
            if (word & bit) {
                ++hits;
                return values_[position];
            }
            ++misses;
            Distance value = evaluate();
            values_[position] = value;
            word |= bit;
            return value;
        }
        uint64_t key = (uint64_t(from) << 32) | uint64_t(to);
        uint64_t hash = key ^ (key >> 33);
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        auto& entry = overflow_[hash & (overflow_.size() - 1)];
        if (entry.valid && entry.key == key) {
            ++hits;
            return entry.value;
        }
        ++misses;
        Distance value = evaluate();
        entry.key = key;
        entry.value = value;
        entry.valid = true;
        return value;
    }
};

}  // namespace hnswlib
