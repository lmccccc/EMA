#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "btree_map.hpp"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <bitset>
#include <unordered_map>
#include <cmath>
#include <deque>
#include <tuple>
#include <chrono>
#include <limits>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

// using BTree = tlx::btree_map<int, std::pair<tableint, int>>; // value, <internal_id, order>
struct Key {
    int attr;      
    tableint id;
};

struct KeyCompare {
    bool operator()(const Key& a, const Key& b) const {
        if (a.attr < b.attr) return true;
        if (a.attr > b.attr) return false;
        return a.id < b.id;
    }
};

using BTree = tlx::btree_map<Key, int, KeyCompare>;  // <<attr, id>, rank>


template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    size_t top_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    size_t ft_bits_{128}; // number of bits per vector for filtering table
    size_t ft_bytes_{16}; // 128/8
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t M0_mul_{2};
    size_t minM0_{16};
    double attr_sort_alpha_{0.5};  // weight for attribute similarity in mixed sorting
    size_t ef_construction_{0};
    size_t ef_{ 0 };
    size_t ef_top_{ 0 };
    size_t padding{0};
    std::vector<BTree> btrees; // B+ tree for attributes   <value, <internal_id, order>>
    std::vector<std::vector<std::vector<tableint>>> ivf; // inverted file for categorical attributes

    //attributes
    int attr_size_per_item_ = 0;
    std::vector<int> attr_type_; // 0: numerical, 1: categorical
    std::vector<int> attr_pos_;
    std::vector<int> predicate_offset_;
    int predicate_size_{0};


    double mult_{0.0}, revSize_{0.0};
    double threshold_1_{0.01}, threshold_2_{0.01}, threshold_3_{0.1}; // threshold to decide whether scan the bucket or not
    int maxlevel_{0};

    double ft_routing_min_deg_{10}; // min degree: if FT-passing neighbors < this, backfill for routing

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 }, ft_offset_{0}, offsetAttr_{0}; 
    size_t nbr_size_per_element{0};

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};
    mutable std::atomic<long> metric_ft_passed_total{0};     // nodes that passed FT check
    mutable std::atomic<long> metric_ft_false_positives{0};  // passed FT but failed predicate (FP)
    mutable std::atomic<long> metric_predicate_checked{0};   // nodes that actually reached predicate_check
    mutable std::atomic<long> metric_total_neighbors{0};     // total neighbors scanned (before FT)

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // Repair candidate tracking: nodes with high dead-neighbor ratio detected during search
    double repair_dead_ratio_threshold_{0.1};  // report nodes with ≥10% deleted neighbors
    mutable std::mutex repair_candidates_lock_;
    mutable std::unordered_map<tableint, float> repair_candidates_;  // node_id → dead_ratio

    // entry points (items at 2 layer)
    std::vector<tableint> ep_ids_;

    // cluster buckets
    int bucket_size_{0};            // number of buckets(clusters)
    std::vector<tableint> bucket_data_; // buckets to id, cluster[i]: buckets[bucket_offsets_[i] ~ bucket_offsets_[i+1]-1]
    std::vector<tableint> bucket_offsets_; // offsets for bucket_data_
    std::vector<tableint> id_to_buckets_; // id to buckets
    int cate_int_byte_{0};
    int max_cate_size_{0};

    // counting hash table
    int table_size_;
    int* counting_hash_table = nullptr; // counting hash table
    std::vector<std::vector<int>> counting_hash_table_mapping;

    // search hyper-parameters
    // double total_scan_factor_{0.001}; // scan all points belonging to top buckets
    // double bucket_scan_factor_{0.01}; // scan points within the bucket whose CHT selectivity is lower than this factor
    // double esti_scan_factor_{0.1}; // scan if the estimated selectivity is lower than this factor

    int offsetNbrFt_{0};
    int size_per_ft_{0};

    bool use_ft_{true};
    bool edge_level_ft_{false};  // true = per-edge FT, false = per-node FT

    bool use_ft_routing_{true};

    // ====================================================================
    // DNF Predicate: supports arbitrary AND/OR combinations via DNF
    // (Disjunctive Normal Form = OR of AND-terms)
    //
    // Each term stores two FT bitmaps per attribute:
    //   or_bitmap  → existence check: (ft & pred) != 0
    //   and_bitmap → superset check:  (~ft & pred) == 0
    // This eliminates the per-attribute type branch in the hot path.
    //
    // Per-attribute mapping within each term:
    //   Numerical range   → or=range_bits, and=zeros
    //   Categorical AND   → or=all_ones,   and=label_bits
    //   Categorical OR    → or=label_bits, and=zeros
    //   Skip (no filter)  → attr_active=0, bitmaps unused
    // ====================================================================
    static constexpr int DNF_MAX_TERMS = 16;

    struct DNFPredicate {
        int num_terms = 0;
        int attr_count = 0;
        int ft_bytes = 0;
        int term_ft_size = 0;    // = attr_count * ft_bytes
        int term_pred_size = 0;  // = predicate_size_

        // FT check bitmaps — flat contiguous for cache locality
        // Layout: [term0_attr0 | term0_attr1 | ... | term1_attr0 | ...]
        std::vector<char> or_bitmaps;    // size = num_terms * term_ft_size
        std::vector<char> and_bitmaps;   // size = num_terms * term_ft_size

        // Per-term, per-attr: 1 if this attribute is constrained in this term
        std::vector<int8_t> attr_active; // size = num_terms * attr_count

        // Exact check: per-term predicate (same format as predicate_translate output)
        std::vector<int> exact_preds;    // size = num_terms * term_pred_size

        // Per-term, per-attr exact check mode: 0=existence, 1=superset, -1=skip
        std::vector<int8_t> exact_modes; // size = num_terms * attr_count

        // Raw numerical ranges for multi-range OR exact check
        // raw_ranges[term * attr_count + attr] = {lo1,hi1,lo2,hi2,...}
        std::vector<std::vector<int>> raw_ranges; // size = num_terms * attr_count
    };

    void set_ft_flag(bool flag){
        use_ft_ = flag;
    }

    void set_ft_routing_flag(bool flag){
        use_ft_routing_ = flag;
        std::cout << "set ft_routing flag to " << use_ft_routing_ << std::endl;
    }

    void add_ep_ids(const std::vector<tableint>& ep_ids){
        ep_ids_ = ep_ids;
    }

    void set_thresholds(double threshold_1, double threshold_2, double threshold_3){
        threshold_1_ = threshold_1;
        threshold_2_ = threshold_2;
        threshold_3_ = threshold_3;
    }

    void set_ft_routing_min_deg(double threshold){
        ft_routing_min_deg_ = threshold;
        std::cout << "set ft_routing_min_deg to " << ft_routing_min_deg_ << std::endl;
    }

    void set_min_deg(double threshold){
        set_ft_routing_min_deg(threshold);
    }

    void add_buckets(const int *bucket_data, const int * bucket_offsets, size_t offset_size){
        bucket_size_ = offset_size - 1;
        bucket_data_.resize(max_elements_);
        memcpy(bucket_data_.data(), bucket_data, sizeof(tableint) * max_elements_);

        bucket_offsets_.clear();
        assert(ep_ids_.size() > 0);
        for(size_t i = 0; i < offset_size; i++) {
            bucket_offsets_.push_back(bucket_offsets[i]);
        }
    }

    void add_id_to_bucket(const int *id_to_bucket_data){
        id_to_buckets_.resize(max_elements_);
        memcpy(id_to_buckets_.data(), id_to_bucket_data, sizeof(tableint) * max_elements_);
    }


    std::vector<int> bucketize_equal_count(const std::vector<std::vector<std::vector<int>>>& attr, const int attr_idx, int M) {
        int N = max_elements_;
        if (N == 0 || M <= 0) return {};

        std::vector<int> indexed;
        indexed.reserve(N);
        for (int i = 0; i < N; ++i) {
            indexed.push_back(attr[i][attr_idx][0]);
        }
        std::sort(indexed.begin(), indexed.end(),
                [](const auto& a, const auto& b) { return a < b; });

        std::vector<int> mapping(M, indexed[0]);

        int bucket_id = 0;
        int start = 0;
        int start_val = indexed[0];

        for (int k = 1; k < M; ++k) {
            int offset = (N - start) / (M - k + 1);
            while(start + offset < N && indexed[start + offset] == start_val) {
                offset++;
            }
            if (start + offset >= N) {
                mapping[k] = indexed[N - 1] + 1;
                for (int kk = k + 1; kk < M; ++kk)
                    mapping[kk] = mapping[k];
                break;
            } else {
                mapping[k] = indexed[start + offset];
            }
            start += offset;
            start_val = indexed[start];
        }

        std::cout << "num-attribute split positions: ";
        for (int i = 0; i < M; i++) {
            std::cout << mapping[i] << " ";
        }
        std::cout << std::endl;

        return mapping;
    }

    std::vector<int> distribute_labels(const std::vector<std::vector<std::vector<int>>>& attr, const int attr_idx, int M) {

        // count label frequencies
        std::vector<int> label_cnt(max_cate_size_, 0);
        for (int i = 0; i < max_elements_; ++i){
            // int* _attr = attr_at(i, attr_idx);
            std::vector<int> _attr = attr[i][attr_idx];

            // for(int k = 0; k < max_cate_size_; ++k){
            //     int byte_pos = k >> 5;
            //     int bit_pos = k & 31;
            //     if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
            //         label_cnt[k]++;
            //     }
            // }
            for (size_t j = 0; j < _attr.size(); ++j){
                label_cnt[_attr[j]]++;
            }
            // for(int j = 1; j <= _attr[0]; ++j){
            //     if(_attr[j] > label_cnt.size()){
            //         label_cnt.resize(_attr[j] + 1000, 0);
            //     }
            //     label_cnt[_attr[j]]++;
            // }
        }

        // sort labels by frequency, sort together with both label and frequency
        std::vector<std::pair<int, int>> label_freq;
        for (int lbl = 0; lbl < max_cate_size_; lbl++) {
            label_freq.push_back({lbl, label_cnt[lbl]});
        }
        std::sort(label_freq.begin(), label_freq.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });


        // distribute labels to M buckets, assign label[i] = bucket_id
        std::vector<int> label_to_bucket(M, -1);
        std::vector<int> bucket_count(M, 0);
        for (int i = 0; i < label_freq.size(); i++) {
            int lbl = label_freq[i].first;
            // find the bucket with the minimum count
            int min_bucket = std::min_element(bucket_count.begin(), bucket_count.end()) - bucket_count.begin();
            label_to_bucket[lbl] = min_bucket;
            bucket_count[min_bucket]++; // update bucket count
        }

        std::cout << "label distribution to " << M << " buckets: " << std::endl;
        for (int i = 0; i < M && i < (int)label_cnt.size(); i++) {
            if (bucket_count[i] == 0) continue;
            std::cout << "label=" << i << ", bucket=" << label_to_bucket[i] << ", count=" << label_cnt[i] << "; " << std::endl;
        }
        std::cout << std::endl;

        return label_to_bucket;
    }

    inline int* counting_hash_table_at(int cluster_id, int attr_idx) {
        return &counting_hash_table[cluster_id * table_size_ * attr_type_.size() + attr_idx * table_size_];
    }

    inline int* counting_hash_table_at(int cluster_id, int attr_idx) const {
        return &counting_hash_table[cluster_id * table_size_ * attr_type_.size() + attr_idx * table_size_];
    }


    inline int lower_bound(const int* mapping, int L, int val) const {
    #ifdef USE_SSE
        __m128i vval = _mm_set1_epi32(val);
        int i = 0;
        for (; i + 4 <= L; i += 4) {
            __m128i vdata = _mm_loadu_si128((const __m128i*)(mapping + i));

            // 标记 data < val
            __m128i lt = _mm_cmplt_epi32(vdata, vval);
            int mask_lt = _mm_movemask_ps(_mm_castsi128_ps(lt));

            if (mask_lt != 0xF) { // 说明这一组里有 >= val 的
                int mask_ge = (~mask_lt) & 0xF;      // 把那些 < val 的位取反，得到 >= val 的位
                int offset = __builtin_ctz(mask_ge); // 第一个 >= val
                return i + offset;
            }
        }

        // 处理尾巴
        for (; i < L; ++i) {
            if (mapping[i] >= val) return i;
        }
        return L; // 没有 >= val
    #else
        int pos = 0;
        while (pos < L && mapping[pos] < val) ++pos;
        return pos;
    #endif
    }

    inline int last_le_index(const int* mapping, int L, int val) const {
    #ifdef USE_SSE
        // -------- SSE 版本 (一次 4 个 int) --------
        __m128i vval = _mm_set1_epi32(val);
        int i = 0;
        for (; i + 4 <= L; i += 4) {
            __m128i vdata = _mm_loadu_si128((__m128i*)(mapping + i));
            __m128i gt = _mm_cmpgt_epi32(vdata, vval); // mapping[j] > val
            int mask = _mm_movemask_ps(_mm_castsi128_ps(gt));
            if (mask != 0) {
                int offset = __builtin_ctz(mask);
                return i + offset - 1;
            }
        }
        for (; i < L; i++) {
            if (mapping[i] > val) return i - 1;
        }
        return L - 1;
    #else
        // -------- 普通循环 --------
        for (int i = 0; i < L; i++) {
            if (mapping[i] > val) return i-1;
        }
        return L-1;
    #endif
    }

    void init_attr_mapping(const std::vector<std::vector<std::vector<int>>>& attr){
        table_size_ = ft_bits_;
        counting_hash_table_mapping.clear();
        for(int i = 0; i < attr_type_.size(); ++i){
            if (attr_type_[i] == 0) { // numerical
                counting_hash_table_mapping.push_back(bucketize_equal_count(attr, i, table_size_));
            } else if (attr_type_[i] == 1) { // categorical
                counting_hash_table_mapping.push_back(distribute_labels(attr, i, table_size_));
            }
        }
    }

    void update_cht(int* cht, tableint id){
        for(int i = 0; i < attr_type_.size(); ++i){
            int* _attr = attr_at(id, i);
            int base = i * table_size_;
            if (attr_type_[i] == 0) { // numerical
                int val = _attr[0];
                int corresponding_slot = lower_bound(&counting_hash_table_mapping[i][0], counting_hash_table_mapping[i].size(), val);
                if (corresponding_slot == -1) continue;
                cht[base + corresponding_slot]++;
            } else if (attr_type_[i] == 1) { // categorical
                for(int k = 0; k < max_cate_size_; ++k){
                    int byte_pos = k >> 5;
                    int bit_pos = k & 31;
                    if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
                        int corresponding_slot = counting_hash_table_mapping[i][k]; 
                        cht[base + corresponding_slot]++;
                    }
                }
            }
        }
    }

    void init_counting_hash_table() {
        assert(counting_hash_table == nullptr);
        assert(bucket_size_ > 0);
        assert(bucket_offsets_.size() == bucket_size_ + 1);
        bucket_size_ = ep_ids_.size();
        counting_hash_table = new int[table_size_ * bucket_size_ * attr_type_.size()]; // each cluster holds a CHT, table_size_ slots for attr_type_.size() attributes
        memset(counting_hash_table, 0, sizeof(int) * table_size_ * bucket_size_ * attr_type_.size());
        
        assert(counting_hash_table_mapping.size() == attr_type_.size());

        // Initialize counting hash table for each attribute
        // not used for now
        // for(int i = 0; i < attr_type_.size(); ++i){
        //     int bucket_id = 0;
        //     for(int j = 0; j < max_elements_; ++j){
        //         if (j >= bucket_offsets_[bucket_id + 1]) bucket_id++;
        //         int id = bucket_data_[j];
        //         int* _attr = attr_at(id, i);

        //         int* cht = counting_hash_table_at(bucket_id, i);
        //         if (attr_type_[i] == 0) { // numerical
        //             int val = _attr[0];
        //             int corresponding_slot = lower_bound(&counting_hash_table_mapping[i][0], counting_hash_table_mapping[i].size(), val);
        //             if (corresponding_slot == -1) continue;
        //             cht[corresponding_slot]++;
        //         } else if (attr_type_[i] == 1) { // categorical
        //             // for(int k = 1; k <= _attr[0]; ++k){
        //             //     int val = _attr[k];
        //             //     int corresponding_slot = counting_hash_table_mapping[i][val];
        //             //     cht[corresponding_slot]++;
        //             // }
        //             for(int k = 0; k < max_cate_size_; ++k){
        //                 int byte_pos = k >> 5;
        //                 int bit_pos = k & 31;
        //                 if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
        //                     int corresponding_slot = counting_hash_table_mapping[i][k]; 
        //                     cht[corresponding_slot]++;
        //                 }
        //             }
        //         }
        //     }
        // }


    }


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        size_t top_elements = 0,
        bool allow_replace_deleted = false,
        bool dynamic = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements, dynamic);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t top_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        size_t ft_bits = 128,
        std::vector<int> attr_type = {0, 1},
        size_t max_cate_size = 5,
        bool allow_replace_deleted = false,
        bool edge_level_ft = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        top_elements_ = top_elements;
        num_deleted_ = 0;
        ft_bits_ = ft_bits;
        ft_bytes_ = (ft_bits + 7) / 8;
        attr_type_ = attr_type;
        max_cate_size_ = max_cate_size;
        edge_level_ft_ = edge_level_ft;
        assert(ft_bits_ % 8 == 0);
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * M0_mul_;

        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;
        ef_top_ = 1;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        
        // init attr space
        init_attr_space();

        // per element: links(maxM0*4 + 4) + FT(size_per_ft_) + data_size_ + label(8) + padding + attr(attr_size * 4)
        // Layout: [link_list | FT | vector_data | label | padding | attr]
        // FT is placed right after link list for cache locality during search
        size_links_level0_ = maxM0_ * sizeof(tableint)+ sizeof(linklistsizeint);
        size_per_ft_ = ft_bytes_ * attr_type_.size();

        size_t ft_total = edge_level_ft_ ? (maxM0_ * size_per_ft_) : size_per_ft_;
        // New layout: FT right after link list, before vector data
        ft_offset_ = size_links_level0_;
        offsetNbrFt_ = ft_offset_;
        offsetData_ = size_links_level0_ + ft_total;
        label_offset_ = size_links_level0_ + ft_total + data_size_;
        // attr is int*, padding the start position to 4 byte alignment
        padding = sizeof(int) - ((size_links_level0_ + ft_total + data_size_ + sizeof(labeltype)) % sizeof(int));
        if (padding == sizeof(int)) padding = 0;
        offsetAttr_ = size_links_level0_ + ft_total + data_size_ + sizeof(labeltype) + padding;
        size_data_per_element_ = offsetAttr_ + attr_size_per_item_ * sizeof(int);
        nbr_size_per_element = size_links_level0_ + ft_total + data_size_ + sizeof(labeltype);
        offsetLevel0_ = 0;
        
        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);

        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        memset(data_level0_memory_, 0, max_elements_ * size_data_per_element_);

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;

        node_dominate_count_.assign(max_elements_, 0);

    }



    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);


        // counting_hash_table
        free(counting_hash_table);
        counting_hash_table = nullptr;
        
        btrees.clear();
        ivf.clear();

    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    void setEfTop(size_t eftop) {
        ef_top_ = eftop;
    }

    void init_attr_space(){
        attr_pos_.clear();
        attr_pos_.resize(attr_type_.size());
        attr_size_per_item_ = 0;
        predicate_offset_.resize(attr_type_.size());
        predicate_size_ = 0;
        if(max_cate_size_ > 0) cate_int_byte_ = (max_cate_size_ + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
        else cate_int_byte_ = 0;
        std::cout << "max_cate_size_:" << max_cate_size_ << ", cate_int_byte_: " << cate_int_byte_ << std::endl;
        for (size_t i = 0; i < attr_type_.size(); i++)
        {
            if (attr_type_[i] != 0 && attr_type_[i] != 1)
            {
                throw std::runtime_error("Attribute type should be either 0 (numerical) or 1 (categorical)");
            }
        }
        for (size_t i = 0; i < attr_type_.size(); i++)
        {
            if (attr_type_[i] == 0) { // numerical
                attr_pos_[i] = attr_size_per_item_;
                attr_size_per_item_ += 1;
                predicate_offset_[i] = predicate_size_;
                predicate_size_ += 2;
            }
            else if (attr_type_[i] == 1) { // categorical
                attr_pos_[i] = attr_size_per_item_;
                attr_size_per_item_ += cate_int_byte_;
                predicate_offset_[i] = predicate_size_;
                predicate_size_ += cate_int_byte_;
            }
        }
    }


    // void add_attr(const std::vector<std::vector<std::vector<int>>>& data) {
    //     for(size_t i = 0; i < data.size(); i++) { // for each item
    //         for(size_t j = 0; j < data[i].size(); j++) {  // for each attribute
    //             if (attr_type_[j] == 0) { // numerical
    //                 // std::cout << "add attr num data[" << i << "][" << j << "][0]: " << data[i][j][0] << std::endl;
    //                 assert(data[i][j].size() == 1);
    //                 int* num_attr = attr_at(i, j);
    //                 *num_attr = data[i][j][0];
    //             } else if (attr_type_[j] == 1) { // categorical
    //                 int* attr_space = attr_at(i, j);
    //                 // memset(attr_space, 0, cate_int_byte_ * sizeof(int));
    //                 for (int m = 0; m < cate_int_byte_; ++m)
    //                     attr_space[m] = 0;
    //                 for(int k = 0; k < data[i][j].size(); ++k){
    //                     // std::cout << "cate data[" << i << "][" << j << "][" << k << "]: " << data[i][j][k] << std::endl;
    //                     // mark bit position of attribute to 1 
    //                     int val = data[i][j][k];
    //                     int byte_pos = val >> 5; 
    //                     int bit_pos  = val & 31;
    //                     assert(byte_pos < cate_int_byte_);
    //                     attr_space[byte_pos] |= (1u << bit_pos);
    //                 }
    //             }
    //         }
    //     }
    // }

    void add_attr_to_point(int internal_id, const std::vector<std::vector<int>>& data){
        for(size_t j = 0; j < data.size(); j++) {  // for each attribute
            if (attr_type_[j] == 0) { // numerical
                // std::cout << "add attr num data[" << i << "][" << j << "][0]: " << data[i][j][0] << std::endl;
                assert(data[j].size() == 1);
                int* num_attr = attr_at(internal_id, j);
                *num_attr = data[j][0];
            } else if (attr_type_[j] == 1) { // categorical
                int* attr_space = attr_at(internal_id, j);
                // memset(attr_space, 0, cate_int_byte_ * sizeof(int));
                for (int m = 0; m < cate_int_byte_; ++m)
                    attr_space[m] = 0;
                for(int k = 0; k < data[j].size(); ++k){
                    // std::cout << "cate data[" << i << "][" << j << "][" << k << "]: " << data[i][j][k] << std::endl;
                    // mark bit position of attribute to 1 
                    int val = data[j][k];
                    int byte_pos = val >> 5; 
                    int bit_pos  = val & 31;
                    assert(byte_pos < cate_int_byte_);
                    attr_space[byte_pos] |= (1u << bit_pos);
                }
            }
        }
    }

    void generate_attr_indexes(){
        btrees.clear();
        std::cout << "Generating attribute indexes ..." << std::endl;
        std::cout << "Number of attributes: " << attr_type_.size() << std::endl;
        btrees.resize(attr_type_.size());
        for(int i = 0; i < attr_type_.size(); ++i){
            if (attr_type_[i] == 0) { // numerical
                std::cout << "Attribute " << i << ": " << "numerical" << std::endl;
                // build B+ tree for numerical attribute
                // sort all items by this attribute
                // std::vector<std::pair<int, tableint>> items;
                for (int j = 0; j < max_elements_; ++j){
                    int* _attr = attr_at(j, i);
                    // items.push_back({_attr[0], j}); // attr_value, internal_id
                    btrees[i].insert({{_attr[0], j}, -1}); // value, <internal_id, rank>
                }

                // modify each item.second.second to rank
                tableint rank = 0;
                for (auto& it : btrees[i]) {
                    it.second = rank;
                    rank++;
                }
            }
        }


        ivf.clear();
        ivf.resize(attr_type_.size());
        for(int i = 0; i < attr_type_.size(); ++i){
            if (attr_type_[i] == 1) { // categorical
                // build inverted file for categorical attribute
                std::cout << "Attribute " << i << ": " << "categorical" << std::endl;
                ivf[i].resize(max_cate_size_);
                for(int j = 0; j < max_elements_; ++j){
                    int* _attr = attr_at(j, i);
                    // std::cout << "  item " << j << ": ";
                    // for(int k = 1; k <= _attr[0]; ++k){
                    //     std::cout << _attr[k] << " ";
                    // }
                    // if (j < 5){
                    //     std::cout << "  item " << j << ": ";
                    //     for(int k = 1; k <= _attr[0]; ++k){
                    //         std::cout << _attr[k] << " ";
                    //     }
                    //     std::cout << std::endl;
                    // }
                    for(int k = 0; k < max_cate_size_; ++k){
                        int byte_pos = k >> 5;
                        int bit_pos = k & 31;
                        if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
                            if(k >= ivf[i].size()){
                                ivf[i].resize(k + 1);
                            }
                            ivf[i][k].push_back(j);
                        }
                    }
                }
            }
        }
        std::cout << "Attribute indexes generated." << std::endl;

        // for (size_t i = 0; i < cur_element_count; i++) {
        //     const int* var = attr_at(i, 0);
        //     if (var[0] <= 0 || var[0] > 100000) {
        //         std::cout << " after attr index generation, error attr at id " << i << " attr[0]=" << var[0] << std::endl;
        //     }
        // }
    }

    // void generate_id_to_bucket(){
    //     // init id_to_bucket_
    //     id_to_bucket_.resize(max_elements_, -1);
    //     int cur_bucket_id = 0;
    //     for(int i = 0; i < max_elements_; ++i){
    //         int id = bucket_data_[i];
    //         if (i >= bucket_offsets_[cur_bucket_id + 1]) cur_bucket_id++;
    //         id_to_bucket_[id] = cur_bucket_id;
    //     }
    // }




    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    
    
    // Node-level FT: single FT per node (not per edge)
    inline unsigned char* node_ft_at(tableint node_id, int attr_idx=0) const {
        return (unsigned char *) (data_level0_memory_ + node_id * size_data_per_element_ + ft_offset_ + attr_idx * ft_bytes_);
    }

    // Edge-level FT: one FT per edge slot (edge_idx in [0, maxM0_))
    inline unsigned char* edge_ft_at(tableint node_id, int edge_idx, int attr_idx=0) const {
        return (unsigned char *) (data_level0_memory_ + node_id * size_data_per_element_ + ft_offset_ + edge_idx * size_per_ft_ + attr_idx * ft_bytes_);
    }

    // inline unsigned char *getFilterTable(tableint internal_id) const {
    //     return (unsigned char *) (data_level0_memory_ + internal_id * size_data_per_element_ + ft_offset_);
    // }

    
    inline int* attr_at(int internal_id, int attr_idx) {
        return (int *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetAttr_) + attr_pos_[attr_idx];
    }

    inline const int* attr_at(int internal_id, int attr_idx) const {
        return (const int*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetAttr_) + attr_pos_[attr_idx];
    }

    // inline unsigned char* ft_at(int internal_id, int attr_idx) const {
    //     return (unsigned char*)(data_level0_memory_ + internal_id * size_data_per_element_ + ft_offset_) + attr_idx * ft_bytes_;
    // }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


//     // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
//     template <bool bare_bone_search = true, bool collect_metrics = false>
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//     searchBaseLayerST(
//         tableint ep_id,
//         const void *data_point,
//         size_t ef,
//         BaseFilterFunctor* isIdAllowed = nullptr,
//         BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
//         VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//         vl_type *visited_array = vl->mass;
//         vl_type visited_array_tag = vl->curV;

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

//         dist_t lowerBound;
//         if (bare_bone_search || 
//             !isMarkedDeleted(ep_id)) {
//             char* ep_data = getDataByInternalId(ep_id);
//             dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
//             lowerBound = dist;
//             top_candidates.emplace(dist, ep_id);
//             if (!bare_bone_search && stop_condition) {
//                 stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
//             }
//             candidate_set.emplace(-dist, ep_id);
//         } else {
//             lowerBound = std::numeric_limits<dist_t>::max();
//             candidate_set.emplace(-lowerBound, ep_id);
//         }

//         visited_array[ep_id] = visited_array_tag;

//         while (!candidate_set.empty()) {
//             std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
//             dist_t candidate_dist = -current_node_pair.first;

//             bool flag_stop_search;
//             if (bare_bone_search) {
//                 flag_stop_search = candidate_dist > lowerBound;
//             } else {
//                 if (stop_condition) {
//                     flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
//                 } else {
//                     flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
//                 }
//             }
//             if (flag_stop_search) {
//                 break;
//             }
//             candidate_set.pop();

//             tableint current_node_id = current_node_pair.second;
//             int *data = (int *) get_linklist0(current_node_id);
//             size_t size = getListCount((linklistsizeint*)data);
// //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
//             if (collect_metrics) {
//                 metric_hops++;
//                 metric_distance_computations+=size;
//             }

// #ifdef USE_SSE
//             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//             _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//             _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
// #endif

//             for (size_t j = 1; j <= size; j++) {
//                 int candidate_id = *(data + j);
// //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                                 _MM_HINT_T0);  ////////////
// #endif
//                 if (!(visited_array[candidate_id] == visited_array_tag)) {
//                     visited_array[candidate_id] = visited_array_tag;

//                     char *currObj1 = (getDataByInternalId(candidate_id));
//                     dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

//                     bool flag_consider_candidate;
//                     if (!bare_bone_search && stop_condition) {
//                         flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
//                     } else {
//                         flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
//                     }

//                     if (flag_consider_candidate) {
//                         candidate_set.emplace(-dist, candidate_id);
// #ifdef USE_SSE
//                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                         offsetLevel0_,  ///////////
//                                         _MM_HINT_T0);  ////////////////////////
// #endif

//                         if (bare_bone_search || 
//                             !isMarkedDeleted(candidate_id)) {
//                             top_candidates.emplace(dist, candidate_id);
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
//                             }
//                         }

//                         bool flag_remove_extra = false;
//                         if (!bare_bone_search && stop_condition) {
//                             flag_remove_extra = stop_condition->should_remove_extra();
//                         } else {
//                             flag_remove_extra = top_candidates.size() > ef;
//                         }
//                         while (flag_remove_extra) {
//                             tableint id = top_candidates.top().second;
//                             top_candidates.pop();
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
//                                 flag_remove_extra = stop_condition->should_remove_extra();
//                             } else {
//                                 flag_remove_extra = top_candidates.size() > ef;
//                             }
//                         }

//                         if (!top_candidates.empty())
//                             lowerBound = top_candidates.top().first;
//                     }
//                 }
//             }
//         }

//         visited_list_pool_->releaseVisitedList(vl);
//         return top_candidates;
//     }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        // tableint ep_id,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_,
        const void *data_point,
        size_t ef,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        // if (bare_bone_search || 
        //     !isMarkedDeleted(ep_id)) {
        //     char* ep_data = getDataByInternalId(ep_id);
        //     dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
        //     lowerBound = dist;
        //     top_candidates.emplace(dist, ep_id);
        //     if (!bare_bone_search && stop_condition) {
        //         stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
        //     }
        //     candidate_set.emplace(-dist, ep_id);
        // } else {
        //     lowerBound = std::numeric_limits<dist_t>::max();
        //     candidate_set.emplace(-lowerBound, ep_id);
        // }

        // visited_array[ep_id] = visited_array_tag;

        // mark top candidates as visited, and push to candidate set, and initialize lowerBound
        while(!top_candidates_.empty()) {
            tableint id = top_candidates_.top().second;
            visited_array[id] = visited_array_tag;
            candidate_set.emplace(-top_candidates_.top().first, id);
            top_candidates.emplace(top_candidates_.top());
            top_candidates_.pop();
        }
        lowerBound = top_candidates.top().first;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            !isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    inline bool numerical_attr_check(int attr, int predicate_low, int predicate_high) const {
        return (attr >= predicate_low && attr <= predicate_high);
    }

    #ifdef USE_SSE
    inline bool categorical_attr_check(const int* attr, const int* predicate, int len) const {
        if (len <= 4) {
            __m128i a = _mm_loadu_si128((const __m128i*)attr);
            __m128i p = _mm_loadu_si128((const __m128i*)predicate);
            __m128i anded = _mm_and_si128(a, p);
            __m128i cmp   = _mm_cmpeq_epi32(anded, p);
            int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));
            int expected = 0b1000 >> (len-1);  // 对应前 len 个元素匹配
            return mask & expected == expected;
        } else if (len <= 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)attr);
            __m256i p = _mm256_loadu_si256((const __m256i*)predicate);
            __m256i anded = _mm256_and_si256(a, p);
            __m256i cmp   = _mm256_cmpeq_epi32(anded, p);
            int mask = _mm256_movemask_epi8(cmp);
            // 构造一个期望值，只要求前 len 个 int 为 0xF
            int expected = 0b10000000 >> (len-1);  // 对应前 len 个元素匹配
            // 屏蔽掉后面不用的位
            return (mask & expected) == expected;
        }
        // fallback（不太可能到这）
        for (int i = 0; i < len; ++i) {
            if ((attr[i] & predicate[i]) != predicate[i]) return false;
        }
        return true;
    }
    #else
    // 普通版本
    inline bool categorical_attr_check(const int* attr, const int* predicate, int len) const {
        for (int i = 0; i < len; ++i) {
            if ((attr[i] & predicate[i]) != predicate[i]) return false;
        }
        return true;
    }
    #endif


    std::vector<int> cate_to_int(const int* cate) const{
        std::vector<int> res;
        for (int i = 0; i < max_cate_size_; ++i) {
            int byte_pos = i >> 5;
            int bit_pos = i & 31;
            if (byte_pos < cate_int_byte_ && (cate[byte_pos] & (1 << bit_pos))) {
                res.push_back(i);
            }
        }
        return res;
    }

    inline bool predicate_check(tableint id, const std::vector<int>& predicate) const {
        for(int i = 0; i < attr_type_.size(); ++i){
            const int* attr = attr_at(id, i);
            if (attr_type_[i] == 0) { // numerical, compare range
                int low = predicate[predicate_offset_[i]];
                int high = predicate[predicate_offset_[i]+1];
                if (low == -1 && high == -1) continue; // no constraint on this attribute
                if (!numerical_attr_check(attr[0], low, high)) {
                    // std::cout << "num attr " << attr[0] << " not in range [" << low << ", " << high << "]" << std::endl;
                    return false;
                }
            } else if (attr_type_[i] == 1) { // categorical
                if(!categorical_attr_check(attr, predicate.data()+predicate_offset_[i] , cate_int_byte_)) {
                    // std::cout << "cate attr not match" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // ====================================================================
    // predicate_check_dnf: exact attribute check for DNF predicates
    //
    // OR over terms, AND within term.
    // Categorical supports both AND (superset) and OR (existence) modes.
    // Numerical supports OR of multiple ranges via num_range_pairs.
    // ====================================================================
    inline bool predicate_check_dnf(tableint id,
                                     const DNFPredicate& pred) const {
        for (int t = 0; t < pred.num_terms; ++t) {
            const int*    ex_pred = pred.exact_preds.data()  + t * pred.term_pred_size;
            const int8_t* ex_mode = pred.exact_modes.data()  + t * pred.attr_count;

            bool term_pass = true;
            for (int i = 0; i < pred.attr_count; ++i) {
                if (ex_mode[i] == -1) continue;  // skip

                const int* attr = attr_at(id, i);

                if (attr_type_[i] == 0) { // numerical
                    const auto& ranges = pred.raw_ranges[t * pred.attr_count + i];
                    if (ranges.empty()) continue;
                    int n_pairs = static_cast<int>(ranges.size()) / 2;
                    bool any_range = false;
                    for (int p = 0; p < n_pairs; ++p) {
                        if (numerical_attr_check(attr[0], ranges[p*2], ranges[p*2+1])) {
                            any_range = true;
                            break;
                        }
                    }
                    if (!any_range) { term_pass = false; break; }
                } else { // categorical
                    const int* pred_data = ex_pred + predicate_offset_[i];
                    if (ex_mode[i] == 0) {
                        // OR/existence: (attr & pred) != 0
                        bool any = false;
                        for (int k = 0; k < cate_int_byte_; ++k) {
                            if (attr[k] & pred_data[k]) { any = true; break; }
                        }
                        if (!any) { term_pass = false; break; }
                    } else {
                        // AND/superset: (attr & pred) == pred
                        if (!categorical_attr_check(attr, pred_data, cate_int_byte_)) {
                            term_pass = false;
                            break;
                        }
                    }
                }
            }
            if (term_pass) return true;  // OR: any term passes
        }
        return false;
    }


#ifdef USE_SSE
//                 if (ft_bytes_ == 16) {
//                     __m128i ft_vec = _mm_loadu_si128((__m128i*)ft);
//                     __m128i pred_vec = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_));
//                     __m128i and_vec = _mm_and_si128(ft_vec, pred_vec);
//                     matched = !_mm_testz_si128(and_vec, and_vec); // 检查是否有非零
//                 } else if (ft_bytes_ == 32) {
//                     // 分两段处理
//                     __m128i ft_vec1 = _mm_loadu_si128((__m128i*)ft);
//                     __m128i pred_vec1 = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_));
//                     __m128i and1 = _mm_and_si128(ft_vec1, pred_vec1);

//                     __m128i ft_vec2 = _mm_loadu_si128((__m128i*)(ft + 16));
//                     __m128i pred_vec2 = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_ + 16));
//                     __m128i and2 = _mm_and_si128(ft_vec2, pred_vec2);

//                     matched = !_mm_testz_si128(and1, and1) || !_mm_testz_si128(and2, and2);
//                 }
//                 if (!matched) return false; // 数值属性没有匹配直接返回 false

//             } else if (attr_type_[i] == 1) { // categorical
//                 if (ft_bytes_ == 16) {
//                     __m128i ft_vec = _mm_loadu_si128((__m128i*)ft);
//                     __m128i pred_vec = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_));
//                     __m128i and_vec = _mm_and_si128(ft_vec, pred_vec);
//                     if (!_mm_testc_si128(and_vec, pred_vec)) return false; // 有不匹配直接 false
//                 } else if (ft_bytes_ == 32) {
//                     __m128i ft_vec1 = _mm_loadu_si128((__m128i*)ft);
//                     __m128i pred_vec1 = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_));
//                     __m128i and1 = _mm_and_si128(ft_vec1, pred_vec1);

//                     __m128i ft_vec2 = _mm_loadu_si128((__m128i*)(ft + 16));
//                     __m128i pred_vec2 = _mm_loadu_si128((__m128i*)(mapped_predicate + i * ft_bytes_ + 16));
//                     __m128i and2 = _mm_and_si128(ft_vec2, pred_vec2);

//                     if (!_mm_testc_si128(and1, pred_vec1) || !_mm_testc_si128(and2, pred_vec2))
//                         return false;
//                 }
//             }
//         }
//         return true;
//     }
// #else
    // inline bool filter_table_check(tableint id, char* mapped_predicate) const {
    //     for(int i = 0; i < attr_type_.size(); ++i){
    //         unsigned char* ft = ft_at(id, i);
    //         bool matched = false;
    //         if (attr_type_[i] == 0) { // numerical
    //             for (int j = 0; j < ft_bytes_; ++j) {
    //                 if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) > 0) { // any attr exist for range search
    //                     matched = true;
    //                     break;
    //                 }
    //             }
    //             if (!matched) {
    //                 // std::cout << "num attr not match" << std::endl;
    //                 return false;
    //             }
    //         }
    //         else if (attr_type_[i] == 1) { // categorical
    //             for (int j = 0; j < ft_bytes_; ++j) {
    //                 if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) != mapped_predicate[i * ft_bytes_ + j]) { // all attr exist for label search
    //                     // std::cout << "cate attr not match" << std::endl;
    //                     return false;
    //                 }
    //             }
    //         }
    //     }
    //     return true;
    // }
#endif


    inline bool node_ft_check(tableint nbr_id, char* mapped_predicate) const {
        for(int i = 0; i < attr_type_.size(); ++i){
            unsigned char* ft = node_ft_at(nbr_id, i);
            const char* pred = mapped_predicate + i * ft_bytes_;

            // Fast scalar path for small FT (16/32/64 bit)
            if (ft_bytes_ <= 8) {
                uint64_t ft_val = 0, pred_val = 0;
                switch (ft_bytes_) {
                    case 2: { uint16_t a, b; memcpy(&a, ft, 2); memcpy(&b, pred, 2); ft_val = a; pred_val = b; break; }
                    case 4: { uint32_t a, b; memcpy(&a, ft, 4); memcpy(&b, pred, 4); ft_val = a; pred_val = b; break; }
                    case 8: { memcpy(&ft_val, ft, 8); memcpy(&pred_val, pred, 8); break; }
                    default: { memcpy(&ft_val, ft, ft_bytes_); memcpy(&pred_val, pred, ft_bytes_); break; }
                }
                if (attr_type_[i] == 0) { // numerical: any overlap
                    if ((ft_val & pred_val) == 0) return false;
                } else { // categorical: pred must be subset of ft
                    if ((ft_val & pred_val) != pred_val) return false;
                }
                continue;
            }

            #ifdef USE_SSE
            size_t offset = 0;
            if (attr_type_[i] == 0) {  // numerical: exists-anywhere
                bool matched = false;

                for (; offset + 16 <= ft_bytes_; offset += 16) {
                    __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                    __m128i v_mp = _mm_loadu_si128((const __m128i*)(pred + offset));
                    __m128i r    = _mm_and_si128(v_ft, v_mp);

                    if (!_mm_testz_si128(r, r)) {  // r != 0
                        matched = true;
                        break;
                    }
                }

                // tail
                if (!matched) {
                    for (; offset < ft_bytes_; ++offset) {
                        if ((ft[offset] & pred[offset]) > 0) {
                            matched = true;
                            break;
                        }
                    }
                }

                if (!matched) return false;
            }
            else { // categorical: must-satisfy-everywhere
                for (; offset + 16 <= ft_bytes_; offset += 16) {
                    __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                    __m128i v_mp = _mm_loadu_si128((const __m128i*)(pred + offset));
                    __m128i r    = _mm_and_si128(v_ft, v_mp);

                    __m128i cmp = _mm_cmpeq_epi8(r, v_mp);
                    __m128i not_cmp = _mm_xor_si128(cmp, _mm_set1_epi8((char)0xFF));
                    if (!_mm_testz_si128(not_cmp, not_cmp)) return false;
                }

                for (; offset < ft_bytes_; ++offset) {
                    if ((ft[offset] & pred[offset]) != (unsigned char)pred[offset])
                        return false;
                }
            }
            #else
            bool matched = false;
            if (attr_type_[i] == 0) { // numerical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & pred[j]) > 0) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) return false;
            }
            else if (attr_type_[i] == 1) { // categorical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & pred[j]) != (unsigned char)pred[j])
                        return false;
                }
            }
            #endif
        }
        return true;
    }

    // Edge-level FT check: check the FT stored on edge slot edge_idx of node_id
    inline bool edge_ft_check(tableint node_id, int edge_idx, char* mapped_predicate) const {
        for (int i = 0; i < (int)attr_type_.size(); ++i) {
            unsigned char* ft = edge_ft_at(node_id, edge_idx, i);
            const char* pred = mapped_predicate + i * ft_bytes_;

            // Fast scalar path for small FT (≤8 bytes)
            if (ft_bytes_ <= 8) {
                uint64_t ft_val = 0, pred_val = 0;
                memcpy(&ft_val, ft, ft_bytes_);
                memcpy(&pred_val, pred, ft_bytes_);
                if (attr_type_[i] == 0) { // numerical: any overlap
                    if ((ft_val & pred_val) == 0) return false;
                } else { // categorical: pred subset of ft
                    if ((ft_val & pred_val) != pred_val) return false;
                }
                continue;
            }

#ifdef USE_SSE
            // Vectorized path: 16-byte chunks via SSE.
            // numerical: any overlap (any AND-bit set across whole FT)
            // categorical: pred ⊆ ft (every set bit in pred also set in ft)
            if (attr_type_[i] == 0) {
                bool matched = false;
                size_t off = 0;
                for (; off + 16 <= ft_bytes_; off += 16) {
                    __m128i v_ft   = _mm_loadu_si128((const __m128i*)(ft + off));
                    __m128i v_pred = _mm_loadu_si128((const __m128i*)(pred + off));
                    __m128i v_and  = _mm_and_si128(v_ft, v_pred);
                    if (!_mm_testz_si128(v_and, v_and)) { matched = true; break; }
                }
                // tail (only if ft_bytes_ not multiple of 16)
                for (; !matched && off < ft_bytes_; ++off) {
                    if ((ft[off] & (unsigned char)pred[off]) != 0) { matched = true; break; }
                }
                if (!matched) return false;
            } else {
                size_t off = 0;
                for (; off + 16 <= ft_bytes_; off += 16) {
                    __m128i v_ft   = _mm_loadu_si128((const __m128i*)(ft + off));
                    __m128i v_pred = _mm_loadu_si128((const __m128i*)(pred + off));
                    // _mm_testc_si128(a,b): tests (~a & b) == 0, i.e. b ⊆ a
                    if (!_mm_testc_si128(v_ft, v_pred)) return false;
                }
                for (; off < ft_bytes_; ++off) {
                    if ((ft[off] & (unsigned char)pred[off]) != (unsigned char)pred[off]) return false;
                }
            }
#else
            // Generic path for larger FT (no SSE)
            if (attr_type_[i] == 0) {
                bool matched = false;
                for (int j = 0; j < (int)ft_bytes_; ++j) {
                    if ((ft[j] & (unsigned char)pred[j]) > 0) { matched = true; break; }
                }
                if (!matched) return false;
            } else {
                for (int j = 0; j < (int)ft_bytes_; ++j) {
                    if ((ft[j] & (unsigned char)pred[j]) != (unsigned char)pred[j]) return false;
                }
            }
#endif
        }
        return true;
    }

    // Edge-level batched FT check — disabled, kept for reference
    /*
        assert(ft_bytes_ % 16 == 0); // ensure ft_bytes_ is multiple of 16 for SSE processing
        std::fill(res.begin(), res.end(), 1);
        for(int i = 0; i < attr_type_.size(); ++i){
            if (attr_type_[i] == 0) {  // numerical: exists-anywhere
                int batch_start = 0;
                for (; batch_start < size; batch_start += 4){
                    bool matched1 = false;
                    bool matched2 = false;
                    bool matched3 = false;
                    bool matched4 = false;
                    unsigned char* ft1 = nbr_ft_at(id, batch_start, i);
                    unsigned char* ft2 = nbr_ft_at(id, batch_start + 1, i);
                    unsigned char* ft3 = nbr_ft_at(id, batch_start + 2, i);
                    unsigned char* ft4 = nbr_ft_at(id, batch_start + 3, i);
                    size_t offset = 0;
                    for (; offset + 16 < ft_bytes_; offset += 16) {
                        __m128i v_ft1 = _mm_loadu_si128((const __m128i*)(ft1 + offset));
                        __m128i v_ft2 = _mm_loadu_si128((const __m128i*)(ft2 + offset));
                        __m128i v_ft3 = _mm_loadu_si128((const __m128i*)(ft3 + offset));
                        __m128i v_ft4 = _mm_loadu_si128((const __m128i*)(ft4 + offset));
                        
                        __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));

                        __m128i r1    = _mm_and_si128(v_ft1, v_mp);
                        __m128i r2    = _mm_and_si128(v_ft2, v_mp);
                        __m128i r3    = _mm_and_si128(v_ft3, v_mp);
                        __m128i r4    = _mm_and_si128(v_ft4, v_mp);  
                        
                        matched1 = matched1 || !_mm_testz_si128(r1, r1);
                        matched2 = matched2 || !_mm_testz_si128(r2, r2);
                        matched3 = matched3 || !_mm_testz_si128(r3, r3);
                        matched4 = matched4 || !_mm_testz_si128(r4, r4);

                        if (matched1 && matched2 && matched3 && matched4) break; // all matched, can stop early
                    }
                    res[batch_start] &= matched1;
                    res[batch_start + 1] &= matched2;
                    res[batch_start + 2] &= matched3;
                    res[batch_start + 3] &= matched4;
                    if (!res[batch_start] && !res[batch_start + 1] && !res[batch_start + 2] && !res[batch_start + 3]) {
                        continue; // all failed, skip to next batch
                    }
                }

                // tail
                for( int j = batch_start; j < size; ++j) {
                    bool matched = false;
                    unsigned char* ft = nbr_ft_at(id, j, i);
                    size_t offset = 0;
                    for (; offset + 16 < ft_bytes_; offset += 16) {
                        __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                        __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));
                        __m128i r    = _mm_and_si128(v_ft, v_mp);

                        if (!_mm_testz_si128(r, r)) {  // r != 0
                            matched = true;
                            break;
                        }
                    }
                    res[j] = matched;
                    if (res[j]) continue;
                }
            }
            else { // attr type = 1
                int batch_start = 0;
                for (; batch_start < size; batch_start += 4){
                    bool matched1 = false;
                    bool matched2 = false;
                    bool matched3 = false;
                    bool matched4 = false;
                    unsigned char* ft1 = nbr_ft_at(id, batch_start, i);
                    unsigned char* ft2 = nbr_ft_at(id, batch_start + 1, i);
                    unsigned char* ft3 = nbr_ft_at(id, batch_start + 2, i);
                    unsigned char* ft4 = nbr_ft_at(id, batch_start + 3, i);
                    size_t offset = 0;
                    for (; offset + 16 < ft_bytes_; offset += 16) {
                        __m128i v_ft1 = _mm_loadu_si128((const __m128i*)(ft1 + offset));
                        __m128i v_ft2 = _mm_loadu_si128((const __m128i*)(ft2 + offset));
                        __m128i v_ft3 = _mm_loadu_si128((const __m128i*)(ft3 + offset));
                        __m128i v_ft4 = _mm_loadu_si128((const __m128i*)(ft4 + offset));
                        
                        __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));

                        __m128i r1    = _mm_and_si128(v_ft1, v_mp);
                        __m128i r2    = _mm_and_si128(v_ft2, v_mp);
                        __m128i r3    = _mm_and_si128(v_ft3, v_mp);
                        __m128i r4    = _mm_and_si128(v_ft4, v_mp);  

                        __m128i cmp1 = _mm_cmpeq_epi8(r1, v_mp);
                        __m128i cmp2 = _mm_cmpeq_epi8(r2, v_mp);
                        __m128i cmp3 = _mm_cmpeq_epi8(r3, v_mp);
                        __m128i cmp4 = _mm_cmpeq_epi8(r4, v_mp);
                        
                        __m128i not_cmp1 = _mm_xor_si128(cmp1, _mm_set1_epi8((char)0xFF));
                        __m128i not_cmp2 = _mm_xor_si128(cmp2, _mm_set1_epi8((char)0xFF));
                        __m128i not_cmp3 = _mm_xor_si128(cmp3, _mm_set1_epi8((char)0xFF));
                        __m128i not_cmp4 = _mm_xor_si128(cmp4, _mm_set1_epi8((char)0xFF));

                        matched1 = matched1 && _mm_testz_si128(not_cmp1, not_cmp1);
                        matched2 = matched2 && _mm_testz_si128(not_cmp2, not_cmp2);
                        matched3 = matched3 && _mm_testz_si128(not_cmp3, not_cmp3);
                        matched4 = matched4 && _mm_testz_si128(not_cmp4, not_cmp4);

                        if (matched1 && matched2 && matched3 && matched4) break; // all matched, can stop early
                    }
                    res[batch_start] &= matched1;
                    res[batch_start + 1] &= matched2;
                    res[batch_start + 2] &= matched3;
                    res[batch_start + 3] &= matched4;
                    if (!res[batch_start] && !res[batch_start + 1] && !res[batch_start + 2] && !res[batch_start + 3]) {
                        continue; // all failed, skip to next batch
                    }
                }

                // tail
                for( int j = batch_start; j < size; ++j) {
                    bool matched = true;
                    unsigned char* ft = nbr_ft_at(id, j, i);
                    size_t offset = 0;
                    for (; offset + 16 < ft_bytes_; offset += 16) {
                        __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                        __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));
                        __m128i r    = _mm_and_si128(v_ft, v_mp);

                        __m128i cmp = _mm_cmpeq_epi8(r, v_mp);
                        __m128i not_cmp = _mm_xor_si128(cmp, _mm_set1_epi8((char)0xFF));
                        if (!_mm_testz_si128(not_cmp, not_cmp)) {
                            matched = false;
                            break;
                        }
                    }
                    res[j] = matched;
                    if (!res[j]) break;
                }
            }
        }
        #else
            std::cerr << "Batched nbr_ft_check_sse called without USE_SSE defined." << std::endl;
            exit(1);
        #endif
        
        return;
    }
    */

    inline bool node_ft_check_multi_or(tableint nbr_id, char* mapped_predicate) const {
        assert(attr_type_.size() > 1); // only one attribute is supported in multi-or query
        for(int i = 0; i < attr_type_.size(); ++i){
            unsigned char* ft = node_ft_at(nbr_id, i);
            #ifdef USE_SSE
            size_t offset = 0;
            if (attr_type_[i] == 0) {  // numerical: exists-anywhere
                bool matched = false;

                for (; offset + 16 <= ft_bytes_; offset += 16) {
                    __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                    __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));
                    __m128i r    = _mm_and_si128(v_ft, v_mp);

                    if (!_mm_testz_si128(r, r)) {  // r != 0
                        matched = true;
                        return true;
                    }
                }

                // tail
                if (!matched) {
                    for (; offset < ft_bytes_; ++offset) {
                        if ((ft[offset] & mapped_predicate[i * ft_bytes_ + offset]) > 0) {
                            matched = true;
                            return true;
                        }
                    }
                }

                // if (!matched) return false;
                // if (matched) {
                //     any_predicate_matched = true;
                //     break;   // OR：已有一个 predicate 成功
                // }
            }
            else { // categorical: must-satisfy-everywhere
                bool matched = true;
                for (; offset + 16 <= ft_bytes_; offset += 16) {
                    __m128i v_ft = _mm_loadu_si128((const __m128i*)(ft + offset));
                    __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));
                    __m128i r    = _mm_and_si128(v_ft, v_mp);

                    __m128i cmp = _mm_cmpeq_epi8(r, v_mp);
                    __m128i not_cmp = _mm_xor_si128(cmp, _mm_set1_epi8((char)0xFF));
                    if (!_mm_testz_si128(not_cmp, not_cmp)) {
                        matched = false;
                        break;
                    }
                }

                for (; offset < ft_bytes_; ++offset) {
                    if ((ft[offset] & mapped_predicate[i * ft_bytes_ + offset]) != mapped_predicate[i * ft_bytes_ + offset])
                        matched = false;
                        break;
                }
                if (matched) {
                    // any_predicate_matched = true;
                    return true;
                }
            }
            #else
            bool matched = false;
            if (attr_type_[i] == 0) { // numerical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) > 0) { // any attr exist for range search
                        matched = true;
                        return true;
                    }
                }
                // if (!matched) {
                //     // if (id == 9701 && datal[nbr_idx] == 9885)
                //     //     std::cout << "num attr not match" << std::endl;
                //     return false;
                // }
            }
            else if (attr_type_[i] == 1) { // categorical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) != mapped_predicate[i * ft_bytes_ + j]) { // all attr exist for label search
                        // std::cout << "cate attr not match" << std::endl;
                        // return false;
                        continue;
                    }
                    else {
                        // any_predicate_matched = true;
                        return true;
                    }
                }
            }
            #endif
        }
        return true;
    }

    // ====================================================================
    // node_ft_check_dnf: branchless two-bitmap FT check for DNF predicates
    //
    // Per attribute, two checks run unconditionally (no attr_type_ branch):
    //   1. Existence: (ft & or_bitmap) != 0
    //   2. Superset:  (~ft & and_bitmap) == 0
    // Unconstrained attributes are skipped via attr_active flag.
    //
    // Outer loop: OR over DNF terms (short-circuit on first pass)
    // Inner loop: AND over constrained attributes (short-circuit on fail)
    // ====================================================================
    inline bool node_ft_check_dnf(tableint nbr_id,
                                  const DNFPredicate& pred) const {
        for (int t = 0; t < pred.num_terms; ++t) {
            const char*   or_pred = pred.or_bitmaps.data()  + t * pred.term_ft_size;
            const char*   and_pred= pred.and_bitmaps.data() + t * pred.term_ft_size;
            const int8_t* active  = pred.attr_active.data() + t * pred.attr_count;

            bool term_pass = true;
            for (int i = 0; i < pred.attr_count; ++i) {
                if (!active[i]) continue;  // unconstrained — skip

                unsigned char* ft   = node_ft_at(nbr_id, i);
                const char* or_p    = or_pred  + i * ft_bytes_;
                const char* and_p   = and_pred + i * ft_bytes_;

                bool or_pass  = false;
                bool and_fail = false;

                // Fast scalar path for small FT (16/32/64 bit)
                if (ft_bytes_ <= 8) {
                    uint64_t ft_val = 0, or_val = 0, and_val = 0;
                    switch (ft_bytes_) {
                        case 2: {
                            uint16_t a, b, c;
                            memcpy(&a, ft, 2); memcpy(&b, or_p, 2); memcpy(&c, and_p, 2);
                            ft_val = a; or_val = b; and_val = c; break;
                        }
                        case 4: {
                            uint32_t a, b, c;
                            memcpy(&a, ft, 4); memcpy(&b, or_p, 4); memcpy(&c, and_p, 4);
                            ft_val = a; or_val = b; and_val = c; break;
                        }
                        case 8: {
                            memcpy(&ft_val, ft, 8); memcpy(&or_val, or_p, 8); memcpy(&and_val, and_p, 8);
                            break;
                        }
                        default: {
                            memcpy(&ft_val, ft, ft_bytes_); memcpy(&or_val, or_p, ft_bytes_); memcpy(&and_val, and_p, ft_bytes_);
                            break;
                        }
                    }
                    or_pass  = (ft_val & or_val) != 0;
                    and_fail = (~ft_val & and_val) != 0;

                    if (!or_pass || and_fail) {
                        term_pass = false;
                        break;
                    }
                    continue;
                }

                #ifdef USE_SSE
                size_t offset = 0;
                for (; offset + 16 <= ft_bytes_; offset += 16) {
                    __m128i v_ft  = _mm_loadu_si128((const __m128i*)(ft    + offset));
                    __m128i v_or  = _mm_loadu_si128((const __m128i*)(or_p  + offset));
                    __m128i v_and = _mm_loadu_si128((const __m128i*)(and_p + offset));

                    // Existence: any bit set in (ft & or_bitmap)
                    __m128i or_r = _mm_and_si128(v_ft, v_or);
                    or_pass |= !_mm_testz_si128(or_r, or_r);

                    // Superset: any bit set in (~ft & and_bitmap) means fail
                    __m128i and_missing = _mm_andnot_si128(v_ft, v_and);
                    if (!_mm_testz_si128(and_missing, and_missing)) {
                        and_fail = true;
                        break;
                    }
                }
                // Tail bytes
                if (!and_fail) {
                    for (; offset < ft_bytes_; ++offset) {
                        or_pass  |= ((ft[offset] & (unsigned char)or_p[offset]) != 0);
                        if ((~ft[offset] & (unsigned char)and_p[offset]) != 0) {
                            and_fail = true;
                            break;
                        }
                    }
                }
                #else
                for (size_t j = 0; j < ft_bytes_; ++j) {
                    or_pass |= ((ft[j] & (unsigned char)or_p[j]) != 0);
                    if ((~ft[j] & (unsigned char)and_p[j]) != 0) {
                        and_fail = true;
                        break;
                    }
                }
                #endif

                if (!or_pass || and_fail) {
                    term_pass = false;
                    break;
                }
            }
            if (term_pass) return true;  // OR: any term passes
        }
        return false;
    }

    // Edge-level DNF FT check: same logic but reads FT from edge slot
    inline bool edge_ft_check_dnf(tableint node_id, int edge_idx,
                                  const DNFPredicate& pred) const {
        for (int t = 0; t < pred.num_terms; ++t) {
            const char*   or_pred = pred.or_bitmaps.data()  + t * pred.term_ft_size;
            const char*   and_pred= pred.and_bitmaps.data() + t * pred.term_ft_size;
            const int8_t* active  = pred.attr_active.data() + t * pred.attr_count;

            bool term_pass = true;
            for (int i = 0; i < pred.attr_count; ++i) {
                if (!active[i]) continue;

                unsigned char* ft   = edge_ft_at(node_id, edge_idx, i);
                const char* or_p    = or_pred  + i * ft_bytes_;
                const char* and_p   = and_pred + i * ft_bytes_;

                bool or_pass  = false;
                bool and_fail = false;

                if (ft_bytes_ <= 8) {
                    uint64_t ft_val = 0, or_val = 0, and_val = 0;
                    memcpy(&ft_val, ft, ft_bytes_);
                    memcpy(&or_val, or_p, ft_bytes_);
                    memcpy(&and_val, and_p, ft_bytes_);
                    or_pass  = (ft_val & or_val) != 0;
                    and_fail = (~ft_val & and_val) != 0;
                    if (!or_pass || and_fail) { term_pass = false; break; }
                    continue;
                }

                for (size_t j = 0; j < ft_bytes_; ++j) {
                    or_pass |= ((ft[j] & (unsigned char)or_p[j]) != 0);
                    if ((~ft[j] & (unsigned char)and_p[j]) != 0) {
                        and_fail = true;
                        break;
                    }
                }
                if (!or_pass || and_fail) { term_pass = false; break; }
            }
            if (term_pass) return true;
        }
        return false;
    }

    void attr_check() const {
        // test
        for (size_t i = 0; i < cur_element_count; i++) {
            const int* var = attr_at(i, 0);
            if (var[0] <= 0 || var[0] > 100000) {
                std::cout << " error attr at id " << i << " attr[0]=" << var[0] << std::endl;
            }
        }
    }


    // translate 1. two dimentional to one dimentional predicate
    //           2. categorical attribute to bitmap
    std::vector<int> predicate_translate(std::vector<std::vector<int>> ori_predicate) const{
        std::vector<int> mapped_predicate(predicate_size_, 0);
        for(int i = 0; i < ori_predicate.size(); ++i){
            if (attr_type_[i] == 0) { // numerical
                if (ori_predicate[i].size() == 0){
                    // no constraint on this attribute
                    mapped_predicate[predicate_offset_[i]] = -1;
                    mapped_predicate[predicate_offset_[i]+1] = -1;
                    continue;
                }
                if (ori_predicate[i].size() != 2){
                    throw std::runtime_error("Numerical attribute predicate should have exactly two values: [low, high]");
                }
                mapped_predicate[predicate_offset_[i]] = ori_predicate[i][0];
                mapped_predicate[predicate_offset_[i]+1] = ori_predicate[i][1];
            } else if (attr_type_[i] == 1) { // categorical
                if (ori_predicate[i].size() == 0){
                    // no constraint on this attribute
                    for (int k = 0; k < cate_int_byte_; ++k){
                        mapped_predicate[predicate_offset_[i]+k] = (int)0xFFFFFFFF;
                    }
                    continue;
                }
                int byte_pos, bit_pos;
                for(int j = 0; j < ori_predicate[i].size(); ++j){
                    int val = ori_predicate[i][j];
                    byte_pos = val >> 5; 
                    bit_pos = val & 31;
                    assert(byte_pos < cate_int_byte_);
                    mapped_predicate[predicate_offset_[i]+byte_pos] |= (1 << bit_pos);
                }
            }
        }
        return mapped_predicate;
    }

    // transform predicate to filter table format
    // 1. numerical: 1 for matched range
    // 2. categorical: 1 for matched label
    std::vector<char> predicate_to_ft(std::vector<std::vector<int>> ori_predicate) const{
        std::vector<char> predicate_ft(size_per_ft_, 0);
        for(int i = 0; i < ori_predicate.size(); ++i){
            if (ori_predicate[i].size() == 0){
                // no constraint on this attribute
                for (int k = 0; k < ft_bytes_; ++k){
                    predicate_ft[i * ft_bytes_ + k] = (char)0xFF;
                }
                continue;
            }
            if (attr_type_[i] == 0) { // numerical
                if (ori_predicate[i].size() == 0){
                    // no constraint on this attribute
                    for (int k = 0; k < ft_bytes_; ++k){
                        predicate_ft[i * ft_bytes_ + k] = (char)0xFF;
                    }
                }
                else if (ori_predicate[i].size() == 2){
                    int low = ori_predicate[i][0];
                    int high = ori_predicate[i][1];
                    int low_slot = lower_bound(&counting_hash_table_mapping[i][0], counting_hash_table_mapping[i].size(), low);
                    low_slot = low_slot < 0 ? 0 : low_slot;
                    int high_slot = lower_bound(&counting_hash_table_mapping[i][0], counting_hash_table_mapping[i].size(), high);
                    for (int val = low_slot; val < high_slot; ++val){
                        int pos = val >> 3;
                        int bit = val & 7;
                        if (pos < ft_bytes_){
                            predicate_ft[i * ft_bytes_ + pos] |= (1 << bit);
                        }
                    }
                } else {
                    throw std::runtime_error("Numerical attribute predicate should have exactly two values: [low, high]");
                }
            }
            else if (attr_type_[i] == 1) { // categorical
                int byte_pos, bit_pos;
                for(int j = 0; j < ori_predicate[i].size(); ++j){
                    int val = ori_predicate[i][j];
                    int slot = counting_hash_table_mapping[i][val]; 
                    byte_pos = slot >> 3;
                    bit_pos = slot & 7;
                    predicate_ft[i * ft_bytes_ + byte_pos] |= (1 << bit_pos);
                }
            }
        }
        return predicate_ft;
    }

    // ====================================================================
    // build_dnf_predicate: convert user-facing DNF to internal representation
    //
    // dnf_raw[term][attr] = values:
    //   numerical : [lo, hi] or [lo1,hi1, lo2,hi2, ...] for OR-merged ranges
    //   categorical: [label1, label2, ...]
    //   empty     : no constraint (skip)
    //
    // modes[term][attr] (optional, default = auto):
    //   -1 = auto (numerical → existence, categorical → superset)
    //    0 = existence  (ft & pred != 0)   — use for categorical OR
    //    1 = superset   ((ft & pred) == pred)
    // ====================================================================
    DNFPredicate build_dnf_predicate(
        const std::vector<std::vector<std::vector<int>>>& dnf_raw,
        const std::vector<std::vector<int8_t>>& modes = {}
    ) const {
        int n_terms = static_cast<int>(dnf_raw.size());
        if (n_terms > DNF_MAX_TERMS) {
            throw std::runtime_error("DNF predicate exceeds maximum term count ("
                + std::to_string(DNF_MAX_TERMS) + ")");
        }
        int n_attr = static_cast<int>(attr_type_.size());

        DNFPredicate pred;
        pred.num_terms      = n_terms;
        pred.attr_count     = n_attr;
        pred.ft_bytes       = ft_bytes_;
        pred.term_ft_size   = size_per_ft_;
        pred.term_pred_size = predicate_size_;

        pred.or_bitmaps.resize(n_terms * size_per_ft_, 0);
        pred.and_bitmaps.resize(n_terms * size_per_ft_, 0);
        pred.attr_active.resize(n_terms * n_attr, 0);
        pred.exact_preds.resize(n_terms * predicate_size_, 0);
        pred.exact_modes.resize(n_terms * n_attr, -1);
        pred.raw_ranges.resize(n_terms * n_attr);

        for (int t = 0; t < n_terms; ++t) {
            const auto& term = dnf_raw[t];
            assert(static_cast<int>(term.size()) == n_attr);

            char*   or_base   = pred.or_bitmaps.data()  + t * size_per_ft_;
            char*   and_base  = pred.and_bitmaps.data() + t * size_per_ft_;
            int8_t* act_base  = pred.attr_active.data() + t * n_attr;
            int*    ex_base   = pred.exact_preds.data()  + t * predicate_size_;
            int8_t* mode_base = pred.exact_modes.data()  + t * n_attr;

            for (int i = 0; i < n_attr; ++i) {
                const auto& vals = term[i];

                // ------ determine check mode ------
                int8_t cm;
                if (!modes.empty() && modes[t][i] != -1)
                    cm = modes[t][i];
                else if (vals.empty())
                    cm = -1;  // skip
                else
                    cm = (attr_type_[i] == 0) ? 0 : 1; // num→existence, cat→superset

                mode_base[i] = cm;

                if (cm == -1 || vals.empty()) {
                    // Skip — leave bitmaps zero, active=0
                    // Fill exact predicate with "no constraint" markers
                    if (attr_type_[i] == 0) {
                        ex_base[predicate_offset_[i]]   = -1;
                        ex_base[predicate_offset_[i]+1] = -1;
                    } else {
                        for (int k = 0; k < cate_int_byte_; ++k)
                            ex_base[predicate_offset_[i]+k] = (int)0xFFFFFFFF;
                    }
                    continue;
                }

                act_base[i] = 1;  // mark constrained

                // ------ build FT bitmap ------
                char ft_buf[256] = {};  // max 2048 bits, stack-allocated
                assert(ft_bytes_ <= 256);

                if (attr_type_[i] == 0) { // numerical — possibly multi-range
                    int n_pairs = static_cast<int>(vals.size()) / 2;
                    pred.raw_ranges[t * n_attr + i] = vals;  // store all [lo,hi,...] pairs
                    for (int p = 0; p < n_pairs; ++p) {
                        int lo = vals[p*2], hi = vals[p*2+1];
                        int lo_slot = lower_bound(
                            &counting_hash_table_mapping[i][0],
                            counting_hash_table_mapping[i].size(), lo);
                        if (lo_slot < 0) lo_slot = 0;
                        int hi_slot = lower_bound(
                            &counting_hash_table_mapping[i][0],
                            counting_hash_table_mapping[i].size(), hi);
                        for (int v = lo_slot; v < hi_slot; ++v) {
                            int pos = v >> 3, bit = v & 7;
                            if (pos < ft_bytes_) ft_buf[pos] |= (1 << bit);
                        }
                    }
                    // Exact predicate: store all ranges sequentially
                    // (first pair goes into the standard [lo,hi] slot)
                    ex_base[predicate_offset_[i]]   = vals[0];
                    ex_base[predicate_offset_[i]+1] = vals[1];
                } else { // categorical
                    for (int j = 0; j < static_cast<int>(vals.size()); ++j) {
                        int val = vals[j];
                        int slot = counting_hash_table_mapping[i][val];
                        int pos = slot >> 3, bit = slot & 7;
                        ft_buf[pos] |= (1 << bit);
                    }
                    // Exact predicate: bitmap of required labels
                    for (int j = 0; j < static_cast<int>(vals.size()); ++j) {
                        int val = vals[j];
                        int bp = val >> 5, bi = val & 31;
                        assert(bp < cate_int_byte_);
                        ex_base[predicate_offset_[i]+bp] |= (1 << bi);
                    }
                }

                // ------ place into or/and bitmaps ------
                if (cm == 0) {
                    // Existence: or_bitmap = ft_bits, and_bitmap stays zero
                    memcpy(or_base + i * ft_bytes_, ft_buf, ft_bytes_);
                } else {
                    // Superset: and_bitmap = ft_bits, or_bitmap = all-ones
                    // (or=all-ones ensures existence part is trivially true
                    //  for valid edges that have ≥1 bit set in FT)
                    memcpy(and_base + i * ft_bytes_, ft_buf, ft_bytes_);
                    memset(or_base  + i * ft_bytes_, 0xFF, ft_bytes_);
                }
            }
        }
        return pred;
    }

    // Convenience: build DNF from a single AND term (backward compatible)
    DNFPredicate build_single_predicate(
        const std::vector<std::vector<int>>& raw_predicate
    ) const {
        return build_dnf_predicate({raw_predicate});
    }

    std::vector<std::vector<int>> get_raw_attr(tableint id) const {
        std::vector<std::vector<int>> res;
        for (int i = 0; i < attr_type_.size(); ++i) {
            const int* attr = attr_at(id, i);
            if (attr_type_[i] == 0) { // numerical
                res.push_back({attr[0]});
            } else if (attr_type_[i] == 1) { // categorical
                std::vector<int> cate;
                for (int j = 0; j < max_cate_size_; ++j) {
                    int byte_pos = j >> 5;
                    int bit_pos = j & 31;
                    if (byte_pos < cate_int_byte_ && (attr[byte_pos] & (1 << bit_pos))) {
                        cate.push_back(j);
                    }
                }
                res.push_back(cate);
            }
        }
        return res;
    }

    void print_raw_attr(tableint id) const {
        std::vector<std::vector<int>> attrs = get_raw_attr(id);
        std::cout << "Attr of id " << id << ":[ ";
        for (int i = 0; i < attrs.size(); ++i) {
            if (attr_type_[i] == 0) { // numerical
                std::cout << "[" << attrs[i][0] << "] ";
            } else if (attr_type_[i] == 1) { // categorical
                std::cout << " [";
                for (int val : attrs[i]) {
                    std::cout << val << ", ";
                }
                std::cout << "]";
            }
        }
        std::cout << "]";
    }

// #define DEBUG_SEARCH
// #define DEBUG_SEARCH_WORKFLOW

    std::vector<tableint> copy_to_vector(const std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& pq) const {
        auto pq_copy = pq;  // 拷贝一份
        std::vector<tableint> result;
        result.reserve(pq_copy.size());

        while (!pq_copy.empty()) {
            result.push_back(pq_copy.top().second);
            pq_copy.pop();
        }
        return result;
    }

//     // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
//     template <bool bare_bone_search = true, bool collect_metrics = false>
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//     hybridSearchBaseLayerST(
//         // tableint ep_id,
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_,
//         const void *data_point,
//         std::vector<int> predicate,
//         std::vector<char> ft_predicate,
//         size_t ef,
//         size_t search_k,
//         VisitedList *vl,
//         BaseFilterFunctor* isIdAllowed = nullptr,
//         BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
//         vl_type *visited_array = vl->mass;
//         vl_type visited_array_tag = vl->curV;

//         int backbone_edge_size = 10;

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

//         dist_t lowerBound;
//         // mark top candidates as visited, and push to candidate set, and initialize lowerBound
//         int visited = 0;

//         // test
//         // std::cout << "entry points:" << std::endl;

//         while(!top_candidates_.empty()) {
//             tableint id = top_candidates_.top().second;

//             //test 
//             // print_raw_attr(id);

//             visited_array[id] = visited_array_tag;
//             visited++;
//             candidate_set.emplace(-top_candidates_.top().first, id);
//             if (predicate_check(id, predicate)) {
//                 top_candidates.emplace(top_candidates_.top());
//                 // std::cout << " in top candidates" << std::endl;
//             }
//             if (candidate_set.size() >= ef_top_) {
//                 // prevent candidate set from being too large
//                 break;
//             }
//             top_candidates_.pop();
//         }


//         // if (!top_candidates.empty())
//         //     lowerBound = top_candidates.top().first;
//         // else
//         lowerBound = std::numeric_limits<dist_t>::max();
//         int round = 0;
//         int passed = 0;
//         int ft_passed = 0;
//         int deg = 0;
//         int max_deg = 0;
//         int ft_deg = 0;
//         int use_ft_routing_freq = 0;
//         std::vector<int> nbrs;
//         std::vector<int> not_nbrs;
//         std::vector<int> not_nbrs_idx;
//         nbrs.reserve(maxM0_);
//         not_nbrs.reserve(maxM0_);
//         not_nbrs_idx.reserve(maxM0_);
//         while (!candidate_set.empty()) {
//             round++;
//             std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
//             dist_t candidate_dist = -current_node_pair.first;

//             bool flag_stop_search;
//             if (bare_bone_search) {
//                 flag_stop_search = candidate_dist > lowerBound;
//             } else {
//                 if (stop_condition) {
//                     flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
//                 } else {
//                     flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
//                 }
//             }
//             if (flag_stop_search) {
//                 break;
//             }
//             candidate_set.pop();

//             tableint current_node_id = current_node_pair.second;
//             #ifdef DEBUG_SEARCH
//             std::cout << "round " << round << " checking node " << current_node_id << " candidate dist " << candidate_dist << " lower bound " << lowerBound << std::endl;
//             #endif

//             int *data = (int *) get_linklist0(current_node_id);
//             size_t size = getListCount((linklistsizeint*)data);
// //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
//             if (collect_metrics) {
//                 metric_hops++;
//                 metric_distance_computations+=size;
//             }


//             // test
//             deg += size;
//             // std::cout << " node " << current_node_id << " deg " << size << std::endl;
//             if (size > max_deg) {
//                 max_deg = size;
//             }
//             nbrs.clear();
//             not_nbrs.clear();
//             not_nbrs_idx.clear();
//             if (use_ft_) {
//                 for (size_t j = 1; j <= size; j++) {
//                     if (nbr_ft_check(current_node_id, j-1, ft_predicate.data())) {
//                         nbrs.push_back(*(data + j)); // indicate filtered nbr
//                     }
//                     else {
//                         not_nbrs.push_back(*(data + j));
//                         not_nbrs_idx.push_back(j);
//                     }
//                 }
//             }
//             else {
//                 for (size_t j = 1; j <= size; j++) {
//                     nbrs.push_back(*(data + j));
//                 }
//             }
//             ft_deg += nbrs.size();
//             ft_passed += (size - nbrs.size());

//             // double passed_ratio = double(size - nbrs.size()) / size;
//             int min_nbrs = ft_routing_min_deg_ * size;
//             // std::cout << "size:" << size << " ft nbrs size:" << nbrs.size() << " expected min nbrs:" << min_nbrs << std::endl;
            
//             if (use_ft_routing_ && nbrs.size() < min_nbrs) {
//                 use_ft_routing_freq++;
//                 // blind two hop expansion
//                 int two_hop_loop = 0;
//                 for (int not_nbr_j = 0; not_nbr_j < not_nbrs.size(); not_nbr_j++) {
//                     tableint candidate_nbr_id = not_nbrs[not_nbr_j];
//                     if (attr_type_.size() > 1) {
//                         if (!nbr_ft_check_multi_or(current_node_id, not_nbrs_idx[not_nbr_j]-1, ft_predicate.data())) {
//                             continue;
//                         }
//                     }

//                     // get second hop nbrs
//                     int *data2 = (int *) get_linklist0(candidate_nbr_id);
//                     size_t size2 = getListCount((linklistsizeint*)data2);
//                     for (size_t j2 = 1; j2 <= size2; j2++) { 
//                         if (nbr_ft_check(candidate_nbr_id, j2-1, ft_predicate.data())) {
//                             nbrs.push_back(*(data2 + j2)); // indicate filtered nbr
//                         }
//                     }
//                     if (nbrs.size() >= min_nbrs || ++two_hop_loop > 3) {
//                         break;
//                     }
//                 }
//             }
// #ifdef USE_SSE
//             if (nbrs.size() == 0) continue;
//             _mm_prefetch((char *) (visited_array + nbrs[0]), _MM_HINT_T0);
//             _mm_prefetch((char *) (visited_array + nbrs[0] + 64), _MM_HINT_T0);
//             _mm_prefetch(data_level0_memory_ + (nbrs[0]) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//             // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
// #endif

//             for (int nbr_idx = 0; nbr_idx < nbrs.size(); ++nbr_idx) {
//                 // size_t j = nbrs[nbr_idx];
//                 // size_t j_next = nbr_idx + 1 < nbrs.size() ? nbrs[nbr_idx + 1] : size + 1;
//                 // int candidate_id = *(data + j);
//                 int candidate_id = nbrs[nbr_idx];
//                 size_t j_next = nbr_idx + 1 < nbrs.size() ? nbrs[nbr_idx + 1] : size + 1;
                
// //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + j_next), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (j_next) * size_data_per_element_ + offsetData_,
//                                 _MM_HINT_T0);  ////////////
// #endif
//                 if (!(visited_array[candidate_id] == visited_array_tag)) {
//                     visited_array[candidate_id] = visited_array_tag;
//                     visited++;

//                     #ifdef DEBUG_SEARCH
//                     std::cout << "   visiting nbr " << candidate_id << " ";
//                     print_raw_attr(candidate_id);
//                     #endif

//                     //test
//                     // print_raw_attr(candidate_id);

//                     // check filter table predicate
//                     // if (!filter_table_check(candidate_id, ft_predicate.data())) {
//                     //     ft_passed++;
//                     //     // std::cout << " ft passed" << std::endl;
//                     //     // std::cout << " checking 2 hop nbr " << std::endl;
//                     //     int *data2 = (int *) get_linklist0(candidate_id);
//                     //     size_t size2 = getListCount((linklistsizeint*)data2);
//                     //     for (size_t j2 = 1; j2 <= size2; j2++) {
//                     //         int candidate_id2 = *(data2 + j2);
//                     //         // print_raw_attr(candidate_id2);
//                     //     }
//                     //     // std::cout << std::endl;
//                     //     continue;
//                     // }

//                     // check nbr filter table predicate
//                     // if (j > backbone_edge_size && use_ft_ && !nbr_ft_check(current_node_id, j-1, ft_predicate.data())) {
//                     //     ft_passed++;
//                     //     // std::cout << " ft passed" << std::endl;
//                     //     // std::cout << " checking 2 hop nbr " << std::endl;
//                     //     // int *data2 = (int *) get_linklist0(candidate_id);
//                     //     // size_t size2 = getListCount((linklistsizeint*)data2);
//                     //     // for (size_t j2 = 1; j2 <= size2; j2++) {
//                     //     //     int candidate_id2 = *(data2 + j2);
//                     //     //     print_raw_attr(candidate_id2);
//                     //     // }
//                     //     // std::cout << std::endl;
//                     //     #ifdef DEBUG_SEARCH
//                     //     std::cout << " ft passed " << std::endl;
//                     //     #endif
//                     //     continue;
//                     // }
                        
//                     char *currObj1 = (getDataByInternalId(candidate_id));
//                     dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

//                     #ifdef DEBUG_SEARCH
//                     std::cout << " dist " << dist;
//                     #endif

//                     bool flag_consider_candidate;
//                     if (!bare_bone_search && stop_condition) {
//                         flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
//                     } else {
//                         flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
//                     }

//                     #ifdef DEBUG_SEARCH
//                     if (!flag_consider_candidate) {
//                         std::cout << ", farther than lowerbound" << std::endl;
//                     }
//                     #endif

//                     if (flag_consider_candidate) {
//                         candidate_set.emplace(-dist, candidate_id);
// #ifdef USE_SSE
//                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                         offsetLevel0_,  ///////////
//                                         _MM_HINT_T0);  ////////////////////////
// #endif

//                         if (!predicate_check(candidate_id, predicate)) {
//                             passed++;
//                             #ifdef DEBUG_SEARCH
//                             std::cout << " predicate passed" << std::endl;
//                             #endif
//                             continue;
//                         }
                        

//                         if (bare_bone_search || 
//                             !isMarkedDeleted(candidate_id)) {
//                             // std::cout << "pushed to top candidates" << std::endl;
//                             top_candidates.emplace(dist, candidate_id);
//                             #ifdef DEBUG_SEARCH
//                             std::cout << ", pushed to top candidates" << std::endl;
//                             #endif
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
//                             }
//                         }
//                         #ifdef DEBUG_SEARCH
//                         else{
//                             std::cout << ", id not allowed" << std::endl;
//                         }
//                         #endif

//                         bool flag_remove_extra = false;
//                         if (!bare_bone_search && stop_condition) {
//                             flag_remove_extra = stop_condition->should_remove_extra();
//                         } else {
//                             flag_remove_extra = top_candidates.size() > ef;
//                         }
//                         while (flag_remove_extra) {
//                             tableint id = top_candidates.top().second;
//                             top_candidates.pop();
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
//                                 flag_remove_extra = stop_condition->should_remove_extra();
//                             } else {
//                                 flag_remove_extra = top_candidates.size() > ef;
//                             }
//                         }

//                         // if (!top_candidates.empty()) 
//                         //     lowerBound = top_candidates.top().first;


//                         if (top_candidates.size() >= search_k) {
//                             lowerBound = top_candidates.top().first;
//                         }
//                     }
//                 }
//             }
//         }
//         // std::cout << "hybrid search round: " << round
//         //              << ", visited " << visited 
//         //              << ", ft passed " << ft_passed 
//         //              << ", passed " << passed 
//         //              << ", avg degree " << deg * 1.0 / round 
//         //              << ", ft avg degree "<< ft_deg * 1.0 / round 
//         //              << ", max degree " << max_deg
//         //              << ", use two hop freq " << use_ft_routing_freq
//         //              << std::endl;

//         return top_candidates;
//     }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false, bool use_ft_routing = true, bool use_ft=true>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    hybridSearchBaseLayerST(
        // tableint ep_id,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_,
        const void *data_point,
        std::vector<int> predicate,
        std::vector<char> ft_predicate,
        size_t ef,
        size_t search_k,
        VisitedList *vl,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        const DNFPredicate* dnf_pred = nullptr) const {
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        int backbone_edge_size = 10;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        // mark top candidates as visited, and push to candidate set, and initialize lowerBound
        int visited = 0;

        // test
        // std::cout << "entry points:" << std::endl;
        
        // start time
        // auto start_time = std::chrono::high_resolution_clock::now();
        // double two_hop_collection_time = 0.0;
        // double two_hop_scan_time = 0.0;
        // double visited_array_time = 0.0;
        // double linklist_fetch_time = 0.0;
        // double nbr_ft_time = 0.0;
        // auto two_hop_coll_start = std::chrono::high_resolution_clock::now();
        // auto two_hop_coll_end = std::chrono::high_resolution_clock::now();
        // auto two_hop_scan_start = std::chrono::high_resolution_clock::now();
        // auto two_hop_scan_end = std::chrono::high_resolution_clock::now();
        // auto visited_array_start = std::chrono::high_resolution_clock::now();
        // auto visited_array_end = std::chrono::high_resolution_clock::now();
        // auto linklist_fetch_start = std::chrono::high_resolution_clock::now();
        // auto linklist_fetch_end = std::chrono::high_resolution_clock::now();
        // auto nbr_ft_start = std::chrono::high_resolution_clock::now();
        // auto nbr_ft_end = std::chrono::high_resolution_clock::now();

        while(!top_candidates_.empty()) {
            tableint id = top_candidates_.top().second;

            //test 
            // print_raw_attr(id);

            visited_array[id] = visited_array_tag;
            visited++;
            candidate_set.emplace(-top_candidates_.top().first, id);
            if (dnf_pred ? predicate_check_dnf(id, *dnf_pred) : predicate_check(id, predicate)) {
                top_candidates.emplace(top_candidates_.top());
            }
            if (candidate_set.size() >= ef_top_) {
                // prevent candidate set from being too large
                break;
            }
            top_candidates_.pop();
        }


        // Initialize lowerBound from the first candidate (entry point distance),
        // regardless of whether it matches the predicate
        // Use max to avoid premature termination at low selectivity
        lowerBound = std::numeric_limits<dist_t>::max();
        size_t lb_update_threshold = ef;
        int round = 0;
        int passed = 0;
        int ft_passed = 0;
        int predicate_checked = 0;
        int deg = 0;
        int max_deg = 0;
        int ft_deg = 0;
        int use_ft_routing_freq = 0;
        std::vector<int> nbrs;
        std::vector<int> not_nbrs;
        std::vector<int> not_nbrs_idx;
        nbrs.reserve(maxM0_);
        not_nbrs.reserve(maxM0_);
        not_nbrs_idx.reserve(maxM0_);
        while (!candidate_set.empty()) {
            round++;
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    // Stop when candidate is farther than lowerBound and we have ef qualifying results
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() >= ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;

            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

            #ifdef USE_SSE
            if (edge_level_ft_) {
                // Edge-level FT: all neighbor FTs packed in current node's element.
                // Single prefetch warms the entire FT block (matches Jan layout).
                _mm_prefetch(data_level0_memory_ + current_node_id * size_data_per_element_ + ft_offset_, _MM_HINT_T0);
            } else {
                // Node-level FT: each neighbor's FT lives in its own element. Prefetch first few.
                for (size_t pf = 1; pf <= std::min((size_t)8, size); pf++) {
                    tableint pf_id = *(data + pf);
                    _mm_prefetch(data_level0_memory_ + pf_id * size_data_per_element_ + ft_offset_, _MM_HINT_T0);
                }
            }
            #endif
            #ifdef DEBUG_SEARCH
            std::cout << "round " << round << " checking node " << current_node_id << " candidate dist " << candidate_dist << " lower bound " << lowerBound << std::endl;
            #endif
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                // metric_distance_computations counted after FT filtering below
            }


            // test
            deg += size;
            // std::cout << " node " << current_node_id << " deg " << size << std::endl;
            if (size > max_deg) {
                max_deg = size;
            }
            nbrs.clear();
            not_nbrs.clear();
            not_nbrs_idx.clear();
            size_t orig_nbrs_end = 0; // boundary in nbrs[] between original and augmented edge neighbors
            
            if (use_ft) {
                size_t orig_size = (use_augmented_edges_ && !orig_degree_.empty())
                    ? std::min((size_t)orig_degree_[current_node_id], size)
                    : size;

                auto handle_nbr = [&](size_t j, tableint nbr_id, bool ft_pass) {
                    if (ft_pass) {
                        nbrs.push_back(nbr_id);
                        #ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + nbr_id * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + nbr_id), _MM_HINT_T0);
                        #endif
                    } else if constexpr (use_ft_routing) {
                        not_nbrs.push_back(nbr_id);
                        not_nbrs_idx.push_back(j);
                    }
                };

                if (edge_level_ft_) {
                    // Edge-level: FT block already prefetched above; no per-iter neighbor-FT prefetch.
                    for (size_t j = 1; j <= size; j++) {
                        tableint nbr_id = *(data + j);
                        bool ft_pass = dnf_pred
                            ? edge_ft_check_dnf(current_node_id, j - 1, *dnf_pred)
                            : edge_ft_check(current_node_id, j - 1, ft_predicate.data());
                        handle_nbr(j, nbr_id, ft_pass);
                        if (j == orig_size) {
                            orig_nbrs_end = nbrs.size();
                            if (orig_size < size && (int)orig_nbrs_end >= augmented_min_deg_) break;
                        }
                    }
                } else {
                    for (size_t j = 1; j <= size; j++) {
                        tableint nbr_id = *(data + j);
                        #ifdef USE_SSE
                        // Deeper prefetch (j+8) for memory-level parallelism.
                        // Bulk prefetch above already warmed first 16; this keeps the pipeline filled.
                        if (j + 8 <= size) {
                            _mm_prefetch(data_level0_memory_ + *(data + j + 8) * size_data_per_element_ + ft_offset_, _MM_HINT_T0);
                        }
                        #endif
                        bool ft_pass = dnf_pred
                            ? node_ft_check_dnf(nbr_id, *dnf_pred)
                            : node_ft_check(nbr_id, ft_predicate.data());
                        handle_nbr(j, nbr_id, ft_pass);
                        if (j == orig_size) {
                            orig_nbrs_end = nbrs.size();
                            if (orig_size < size && (int)orig_nbrs_end >= augmented_min_deg_) break;
                        }
                    }
                }
                if (orig_size == size) orig_nbrs_end = nbrs.size();
            }
            else {
                for (size_t j = 1; j <= size; j++) {
                    nbrs.push_back(*(data + j));
                    #ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + (*(data + j))), _MM_HINT_T0);
                        // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
                    #endif
                }
                orig_nbrs_end = nbrs.size();
            }
            ft_deg += nbrs.size();
            ft_passed += (size - nbrs.size());

            // Repair candidate detection: count deleted neighbors (read-only, keeps const)
            if (num_deleted_.load(std::memory_order_relaxed) > 0 && size > 0) {
                int dead_count = 0;
                for (size_t j = 1; j <= size; j++) {
                    if (isMarkedDeleted(*(data + j))) dead_count++;
                }
                if (dead_count > 0) {
                    float dead_ratio = static_cast<float>(dead_count) / size;
                    if (dead_ratio >= repair_dead_ratio_threshold_) {
                        std::lock_guard<std::mutex> lock(repair_candidates_lock_);
                        auto it = repair_candidates_.find(current_node_id);
                        if (it == repair_candidates_.end() || it->second < dead_ratio) {
                            repair_candidates_[current_node_id] = dead_ratio;
                        }
                    }
                }
            }

            // double passed_ratio = double(size - nbrs.size()) / size;
            // int min_nbrs = 10;
            // std::cout << "size:" << size << " ft nbrs size:" << nbrs.size() << " expected min nbrs:" << min_nbrs << std::endl;
            // std::vector<tableint> two_hop_candidates(256);
            // ft_nbrs_end marks boundary: [0, ft_nbrs_end) = FT-passing (route+result),
            // [ft_nbrs_end, nbrs.size()) = routing-only (added by min_deg backfill)
            size_t ft_nbrs_end = nbrs.size();

            if (use_ft_routing) {
                if (nbrs.size() < ft_routing_min_deg_) {
                    size_t extra_nbrs_needed = ft_routing_min_deg_ - nbrs.size();
                    // Pick from HEAD of not_nbrs (nearest non-FT-passing neighbors)
                    // to maintain minimum local connectivity for routing
                    size_t pick_count = std::min(extra_nbrs_needed, not_nbrs.size());
                    for(size_t j = 0; j < pick_count; ++j){
                        nbrs.push_back(not_nbrs[j]);
                        _mm_prefetch(data_level0_memory_ + (not_nbrs[j]) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + (not_nbrs[j])), _MM_HINT_T0);
                    }
                }
                // for(int j = 0; j < std::min(two_hop_candidates.size(), (size_t)2); ++j){
                //     _mm_prefetch(data_level0_memory_ + (not_nbrs[j]) * size_data_per_element_, _MM_HINT_T0);  // L1 prefetch
                //     // _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_, _MM_HINT_T1); // L2 prefetch
                //     _mm_prefetch((char *) (visited_array + (not_nbrs[j])), _MM_HINT_T0);  // L1 prefetch
                //     // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
                // }
                // // prefetch visited array and link lists
                // #ifdef USE_SSE
                //     for (int prefetch_idx = 0; prefetch_idx < nbrs.size(); ++prefetch_idx) {
                //         // prefecth visited array
                //         _mm_prefetch((char *) (visited_array + nbrs[prefetch_idx]), _MM_HINT_T0);
                //     }
                //     for (int prefetch_idx = 1; prefetch_idx <= size; ++prefetch_idx) { 
                //         // prefetch all nbrs' link lists to l1 cache
                //         _mm_prefetch(data_level0_memory_ + (*(data + prefetch_idx)) * size_data_per_element_, _MM_HINT_T0);
                //         _mm_prefetch(data_level0_memory_ + (*(data + prefetch_idx)) * size_data_per_element_+64, _MM_HINT_T0);
                //         // prefetch visited array
                //         _mm_prefetch((char *) (visited_array + (*(data + prefetch_idx))), _MM_HINT_T0);
                //     }
                // #endif


                // if (nbrs.size() < min_nbrs) {
                //     two_hop_coll_start = std::chrono::high_resolution_clock::now();
                //     for (auto& nbr: nbrs) {
                //         visited_array_start = std::chrono::high_resolution_clock::now();
                //         if (!(visited_array[nbr] == visited_array_tag)) {
                //             // visited_array[nbr] = visited_array_tag;
                //             // visited++;
                //             two_hop_candidates.push_back(nbr);
                //         }
                //         visited_array_end = std::chrono::high_resolution_clock::now();
                //         visited_array_time += std::chrono::duration<double, std::milli>(visited_array_end - visited_array_start).count();

                //     }

                //     for (size_t j = 1; j <= size; j++) {
                //         tableint candidate_nbr_id = *(data + j);
                //         // if (attr_type_.size() > 1) {
                //         //     if (!nbr_ft_check_multi_or(current_node_id, not_nbrs_idx[not_nbr_j]-1, ft_predicate.data())) {
                //         //         continue;
                //         //     }
                //         // }

                //         // get second hop nbrs
                //         linklist_fetch_start = std::chrono::high_resolution_clock::now();
                //         int *data2 = (int *) get_linklist0(candidate_nbr_id);
                //         linklist_fetch_end = std::chrono::high_resolution_clock::now();
                //         linklist_fetch_time += std::chrono::duration<double, std::milli>(linklist_fetch_end - linklist_fetch_start).count();
                //         size_t size2 = getListCount((linklistsizeint*)data2);
                        
                //         std::vector<uint8_t> ft_predicate_result_2nd_hop(size2, 1);
                //         nbr_ft_start = std::chrono::high_resolution_clock::now();
                //         // batched_nbr_ft_check_sse(candidate_nbr_id, size2, ft_predicate.data(), ft_predicate_result_2nd_hop);
                //         for (size_t j2 = 1; j2 <= size2; j2++) {
                //             ft_predicate_result_2nd_hop[j2-1] = nbr_ft_check(candidate_nbr_id, j2-1, ft_predicate.data());
                //         }
                //         nbr_ft_end = std::chrono::high_resolution_clock::now();
                //         nbr_ft_time += std::chrono::duration<double, std::milli>(nbr_ft_end - nbr_ft_start).count();
                //         for (size_t j2 = 1; j2 <= size2; j2++) {
                //             linklist_fetch_start = std::chrono::high_resolution_clock::now();
                //             int candidate_two_hop_id = *(data2 + j2);
                //             linklist_fetch_end = std::chrono::high_resolution_clock::now();
                //             linklist_fetch_time += std::chrono::duration<double, std::milli>(linklist_fetch_end - linklist_fetch_start).count();
                        
                //             if (ft_predicate_result_2nd_hop[j2-1]) {
                //                 visited_array_start = std::chrono::high_resolution_clock::now();
                //                 if (!(visited_array[candidate_two_hop_id] == visited_array_tag)){
                //                     // visited_array[candidate_two_hop_id] = visited_array_tag;
                //                     // visited++;
                //                     nbrs.push_back(*(data2 + j2));
                //                     two_hop_candidates.push_back(*(data2 + j2)); // indicate filtered nbr
                //                     #ifdef USE_SSE
                //                         _mm_prefetch(data_level0_memory_ + (*(data2 + j2)) * size_data_per_element_ + offsetData_, _MM_HINT_T1);
                //                         _mm_prefetch(data_level0_memory_ + (*(data2 + j2)) * size_data_per_element_ + offsetData_ + 64, _MM_HINT_T1);
                //                         // _mm_prefetch((char *) (visited_array + (*(data2 + j2))), _MM_HINT_T0);
                //                         // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
                //                     #endif
                //                 }
                //                 visited_array_end = std::chrono::high_resolution_clock::now();
                //                 visited_array_time += std::chrono::duration<double, std::milli>(visited_array_end - visited_array_start).count();
                //             }
                //         }
                //     }
                    // two_hop_coll_end = std::chrono::high_resolution_clock::now();
                    // two_hop_collection_time += std::chrono::duration<double, std::milli>(two_hop_coll_end - two_hop_coll_start).count();
                    // use_ft_routing_freq++;
                    // two_hop_scan_start = std::chrono::high_resolution_clock::now();
                    // linear_two_hop_scan(two_hop_candidates, 
                    //          data_point, 
                    //          predicate, 
                    //          candidate_set,
                    //          top_candidates,
                    //          ef,
                    //          lowerBound,
                    //          passed,
                    //          search_k);
                    // two_hop_scan_end = std::chrono::high_resolution_clock::now();
                    // two_hop_scan_time += std::chrono::duration<double, std::milli>(two_hop_scan_end - two_hop_scan_start).count();
                    // continue;
                // }
            }


//             l2 prefetch
// #ifdef USE_SSE
//             for (int prefetch_idx = 0; prefetch_idx < nbrs.size(); ++prefetch_idx) {
//                 // prefetch all nbrs to l2 cache
//                 _mm_prefetch(data_level0_memory_ + (nbrs[prefetch_idx]) * size_data_per_element_ + offsetData_, _MM_HINT_T1);
//                 // prefecth visited array
//                 _mm_prefetch((char *) (visited_array + nbrs[prefetch_idx]), _MM_HINT_T1);
//             }
// #endif

// l1 prefetch
#ifdef USE_SSE
            // if (nbrs.size() == 0) continue;
            // _mm_prefetch((char *) (visited_array + nbrs[0]), _MM_HINT_T0);
            // _mm_prefetch((char *) (visited_array + nbrs[0] + 64), _MM_HINT_T0);
            // _mm_prefetch(data_level0_memory_ + (nbrs[0]) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            // // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
            int prefetch_n = std::min<int>(nbrs.size(), 8);
            for (int i = 0; i < prefetch_n; i++) {
                char* p = (char*)(data_level0_memory_ + nbrs[i] * size_data_per_element_ + offsetData_);
                _mm_prefetch(p + 0,   _MM_HINT_T0);
                _mm_prefetch(p + 64,  _MM_HINT_T0);
            }
#endif

            // Count actual distance computations (only FT-passing + min_deg backfill)
            if (collect_metrics) {
                metric_distance_computations += nbrs.size();
            }

            // Augmented-edge decision already made in FT loop (adaptive skip).
            // Process all neighbors in nbrs:
            // [0, ft_nbrs_end): FT-passing → route + result
            // [ft_nbrs_end, nbrs.size()): routing-only backfill → route only, skip result queue
            for (int nbr_idx = 0; nbr_idx < (int)nbrs.size(); ++nbr_idx) {
                int candidate_id = nbrs[nbr_idx];
                bool routing_only = use_ft && edge_level_ft_ && ((size_t)nbr_idx >= ft_nbrs_end);

#ifdef USE_SSE
                if (nbr_idx + 8 < (int)nbrs.size()) {
                    int next_candidate_id = nbrs[nbr_idx + 8];
                    char* p = (char*)(data_level0_memory_ + next_candidate_id * size_data_per_element_ + offsetData_);
                    _mm_prefetch(p + 0,   _MM_HINT_T0);
                    _mm_prefetch(p + 64,  _MM_HINT_T0);
                }
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    visited++;

                    #ifdef DEBUG_SEARCH
                    std::cout << "   visiting nbr " << candidate_id << " ";
                    print_raw_attr(candidate_id);
                    #endif
                        
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    #ifdef DEBUG_SEARCH
                    std::cout << " dist " << dist;
                    #endif

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    #ifdef DEBUG_SEARCH
                    if (!flag_consider_candidate) {
                        std::cout << ", farther than lowerbound" << std::endl;
                    }
                    #endif

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        // Routing-only edges: add to candidate_set for routing but skip result queue
                        if (routing_only) {
                            #ifdef DEBUG_SEARCH
                            std::cout << " (routing-only, skip result)" << std::endl;
                            #endif
                            continue;
                        }

                        predicate_checked++;
                        if (!(dnf_pred ? predicate_check_dnf(candidate_id, *dnf_pred) : predicate_check(candidate_id, predicate))) {
                            passed++;
                            #ifdef DEBUG_SEARCH
                            std::cout << " predicate passed" << std::endl;
                            #endif
                            continue;
                        }

                        if (bare_bone_search || 
                            !isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist, candidate_id);
                            #ifdef DEBUG_SEARCH
                            std::cout << ", pushed to top candidates" << std::endl;
                            #endif
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }
                        #ifdef DEBUG_SEARCH
                        else{
                            std::cout << ", id not allowed" << std::endl;
                        }
                        #endif

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        // Key change: update lowerBound as soon as top_candidates reaches lb_update_threshold
                        if (top_candidates.size() >= lb_update_threshold) 
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        // auto end_time = std::chrono::high_resolution_clock::now();
        // double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        // double coll_pct = two_hop_collection_time / total_time * 100.0;
        // double scan_pct = two_hop_scan_time / total_time * 100.0;
        // double visit_pct = visited_array_time / total_time * 100.0;
        // double linklist_pct = linklist_fetch_time / total_time * 100.0;
        // double nbr_ft_pct = nbr_ft_time / total_time * 100.0;
        // std::cout << "hybrid search round: " << round
        //              << ", visited " << visited 
        //              << ", ft passed " << ft_passed 
        //              << ", passed " << passed 
        //              << ", avg degree " << deg * 1.0 / round 
        //              << ", ft avg degree "<< ft_deg * 1.0 / round 
        //              << ", max degree " << max_deg
        //              << ", use two hop freq " << use_ft_routing_freq
        //              << std::endl;
        // FT stats: ft_deg = nodes passed FT, passed = passed FT but failed predicate (FP)
        metric_ft_passed_total.fetch_add(ft_deg, std::memory_order_relaxed);
        metric_ft_false_positives.fetch_add(passed, std::memory_order_relaxed);
        metric_predicate_checked.fetch_add(predicate_checked, std::memory_order_relaxed);
        metric_total_neighbors.fetch_add(deg, std::memory_order_relaxed);
        return top_candidates;
    }

    void linear_two_hop_scan(std::vector<tableint>& two_hop_candidates, 
                             const void *data_point, 
                             const std::vector<int>& predicate, 
                             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& candidate_set,
                             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& top_candidates,
                             size_t& ef,
                             dist_t& lowerBound,
                             int& passed,
                             int& predicate_checked,
                             size_t& search_k,
                             const DNFPredicate* dnf_pred = nullptr) const {
        for (size_t idx = 0; idx < two_hop_candidates.size(); idx++) {
            tableint candidate_id = two_hop_candidates[idx];
            char *currObj1 = (getDataByInternalId(candidate_id));
            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
            bool flag_consider_candidate;
            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
            if (flag_consider_candidate) {
                candidate_set.emplace(-dist, candidate_id);
                predicate_checked++;
                if (!(dnf_pred ? predicate_check_dnf(candidate_id, *dnf_pred) : predicate_check(candidate_id, predicate))) {
                    passed++;
                    continue;
                }

                top_candidates.emplace(dist, candidate_id);
                bool flag_remove_extra = false;
                flag_remove_extra = top_candidates.size() > ef;
                while (flag_remove_extra) {
                    tableint id = top_candidates.top().second;
                    top_candidates.pop();
                    flag_remove_extra = top_candidates.size() > ef;
                }

                if (top_candidates.size() >= search_k) 
                    lowerBound = top_candidates.top().first;
                // if (!top_candidates.empty())
                //             lowerBound = top_candidates.top().first;
            }
        }
        
    }


    // search first layer with ef_search_top
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchTopLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            !isMarkedDeleted(ep_id)) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            // int *data = (int *) get_linklist0(current_node_id);
            // size_t size = getListCount((linklistsizeint*)data);

            unsigned int *data = (unsigned int *) get_linklist(current_node_id, 1);
            int size = getListCount(data);
            metric_hops++;
            metric_distance_computations+=size;

//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            // if (collect_metrics) {
            //     metric_hops++;
            //     metric_distance_computations+=size;
            // }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);

//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            !isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // inline void updateft(tableint ft_id, tableint attr_id) {
    //     // insert "attr" of attr_id to "ft" of ft_id
    //     // unsigned char* ft = getFilterTable(ft_id);
    //     for (int attr_idx = 0; attr_idx < attr_type_.size(); attr_idx++) {
    //         unsigned char* ft = ft_at(ft_id, attr_idx);
    //         int* _attr = attr_at(attr_id, attr_idx);
    //         if (attr_type_[attr_idx] == 0) { // numerical
    //             // hash by mapping to bucket
    //             int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0], counting_hash_table_mapping[attr_idx].size(), _attr[0]);
    //             if (pos < 0) continue;
    //             set_ft_at_pos(ft, pos);
    //         }
    //         else { // categorical
    //             for (int k = 0; k <= max_cate_size_; ++k) {
    //                 int byte_pos = k >> 5; 
    //                 int bit_pos = k & 31;
    //                 if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
    //                     int pos = counting_hash_table_mapping[attr_idx][k];
    //                     if (pos < 0) continue;
    //                     set_ft_at_pos(ft, pos);
    //                 }
    //             }
    //         }
    //     }
    // }

    inline void updateft(unsigned char* ft_addr, tableint attr_id) {
        // insert "attr" of attr_id to "ft" of ft_id
        // char* ft = getFilterTable(ft_id);
        for (int attr_idx = 0; attr_idx < attr_type_.size(); attr_idx++) {
            unsigned char* ft = ft_addr + attr_idx * ft_bytes_;
            int* _attr = attr_at(attr_id, attr_idx);
            if (attr_type_[attr_idx] == 0) { // numerical
                // hash by mapping to bucket
                int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0], counting_hash_table_mapping[attr_idx].size(), _attr[0]);
                if (pos < 0) continue;
                set_ft_at_pos(ft, pos);
            }
            else { // categorical
                for (int k = 0; k <= max_cate_size_; ++k) {
                    int byte_pos = k >> 5; 
                    int bit_pos = k & 31;
                    if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
                        int pos = counting_hash_table_mapping[attr_idx][k];
                        if (pos < 0) continue;
                        set_ft_at_pos(ft, pos);
                    }
                }
            }
        }
    }


    // point_id: id of point who is getting neighbors
    // nbr_idx: index of the neighbor in the neighbor list of point_id
    // inline void updateNbrFt(tableint point_id, int nbr_idx, tableint dominated_id) {
    //     // get nbr id
    //     // insert "attr" of attr_id to "ft" of ft_id
    //     for (int attr_idx = 0; attr_idx < attr_type_.size(); attr_idx++) {
    //         unsigned char* ft = nbr_ft_at(point_id, nbr_idx, attr_idx);
    //         // add attr of dominated_id
    //         int* _attr = attr_at(dominated_id, attr_idx);
    //         if (attr_type_[attr_idx] == 0) { // numerical
    //             // hash by mapping to bucket
    //             int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0], counting_hash_table_mapping[attr_idx].size(), _attr[0]);
    //             if (pos < 0) continue;
    //             set_ft_at_pos(ft, pos);
    //         }
    //         else { // categorical
    //             for (int k = 0; k <= max_cate_size_; ++k) {
    //                 int byte_pos = k >> 5; 
    //                 int bit_pos = k & 31;
    //                 if (byte_pos < cate_int_byte_ && (_attr[byte_pos] & (1 << bit_pos))) {
    //                     int pos = counting_hash_table_mapping[attr_idx][k];
    //                     if (pos < 0) continue;
    //                     set_ft_at_pos(ft, pos);
    //                 }
    //             }
    //         }
    //     }
    // }

    int pruned_count{0};
    int attr_pruned_count{0};
    double heuristic_time{0.0};
    bool build_stats_printed_{false};
    // #define DEBUG_BUILD

    // Compute attribute distance between src and nbr, normalized to [0, 1].
    // 0 = identical attributes, 1 = maximally different.
    // For numerical: |bucket_src - bucket_nbr| / num_buckets
    // For categorical: 1 if no shared labels, 0 if identical
    double compute_attr_distance(tableint src_id, tableint nbr_id) {
        double total_dist = 0.0;
        int num_attrs = (int)attr_type_.size();
        for (int attr_idx = 0; attr_idx < num_attrs; attr_idx++) {
            int* src_attr = attr_at(src_id, attr_idx);
            int* nbr_attr = attr_at(nbr_id, attr_idx);
            if (attr_type_[attr_idx] == 0) { // numerical
                int src_pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                          counting_hash_table_mapping[attr_idx].size(), src_attr[0]);
                int nbr_pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                          counting_hash_table_mapping[attr_idx].size(), nbr_attr[0]);
                int num_buckets = (int)counting_hash_table_mapping[attr_idx].size();
                total_dist += (num_buckets > 1) ? 
                    (double)std::abs(src_pos - nbr_pos) / (num_buckets - 1) : 0.0;
            } else { // categorical: 1 - has_shared_label
                bool has_shared = false;
                for (int byte = 0; byte < cate_int_byte_; ++byte) {
                    if (((uint32_t*)src_attr)[byte] & ((uint32_t*)nbr_attr)[byte]) {
                        has_shared = true;
                        break;
                    }
                }
                total_dist += has_shared ? 0.0 : 1.0;
            }
        }
        return total_dist / num_attrs;
    }

    bool cht_low_degree(int* cht, tableint src_id, tableint nbr_id, int return_list_size) {
        if (return_list_size < maxM_ / 3) return true;

        // check if id has unique attr not covered by cht
        for (int attr_idx = 0; attr_idx < attr_type_.size(); attr_idx++) {
            int* src_attr = attr_at(src_id, attr_idx);
            int* nbr_attr = attr_at(nbr_id, attr_idx);
            if (attr_type_[attr_idx] == 0) { // numerical
                // hash by mapping to bucket
                int src_pos = lower_bound(&counting_hash_table_mapping[attr_idx][0], counting_hash_table_mapping[attr_idx].size(), src_attr[0]);
                int nbr_pos = lower_bound(&counting_hash_table_mapping[attr_idx][0], counting_hash_table_mapping[attr_idx].size(), nbr_attr[0]);
                int range_left = std::min(src_pos, nbr_pos);
                int range_right = std::max(src_pos, nbr_pos);
                
                int cnt = 0;
                for (int cht_pos = range_left; cht_pos <= range_right; ++cht_pos){
                    cnt += cht[attr_idx * table_size_ + cht_pos];
                }

                #ifdef DEBUG_BUILD
                std::cout << " num attr degree range [" << range_left << ", " << range_right << "] cnt: " << cnt << std::endl;
                if (cnt < minM0_)  std::cout << " num attr low degree range " << std::endl;
                #endif

                if (cnt < minM0_) return true; // range search degree less than minM0
            }
            else { // categorical
                int* src_attr = attr_at(src_id, attr_idx);
                int* nbr_attr = attr_at(nbr_id, attr_idx);
                for (int byte = 0; byte < cate_int_byte_; ++byte) {
                    uint32_t overlap = ((uint32_t*)src_attr)[byte] & ((uint32_t*)nbr_attr)[byte];
                    if (overlap == 0) continue; 

                    while (overlap) {
                        uint32_t lowbit = overlap & -overlap;
                        int bit_pos = __builtin_ctz(overlap);
                        int k = (byte << 5) + bit_pos;

                        int pos = counting_hash_table_mapping[attr_idx][k];
                        if (pos >= 0 && cht[attr_idx * table_size_ + pos] < minM0_) {
                            return true;   // OK
                        }

                        overlap &= (overlap - 1);
                    }
                }
            }
        }
        #ifdef DEBUG_BUILD
        std::cout << " pruned by cht " << std::endl;
        #endif
        return false;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,
        bool need_record = false, // if need_record, we will record the pruned candidates and add to filter table
        tableint cur_id = -1,      // id of point who is getting neighbors. valid only when need_record is true
        std::vector<std::vector<tableint>>* dominated_list = nullptr // return the dominate list
    ) {

        // start time
        auto start = std::chrono::high_resolution_clock::now();
        if (top_candidates.size() < M) {
            return;
        }

        // Extract all candidates and sort by mixed score (distance + alpha * attr_distance)
        std::vector<std::pair<dist_t, tableint>> all_candidates;
        all_candidates.reserve(top_candidates.size());
        while (top_candidates.size() > 0) {
            all_candidates.emplace_back(top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        // Sort by mixed score: normalized distance + alpha * attr_distance
        if (need_record && attr_sort_alpha_ > 0 && cur_id >= 0 && !counting_hash_table_mapping.empty()) {
            // Find max distance for normalization
            dist_t max_dist = 0;
            for (auto& [d, id] : all_candidates) {
                if (d > max_dist) max_dist = d;
            }
            if (max_dist <= 0) max_dist = 1.0f;

            // Compute mixed scores and sort
            std::vector<std::pair<double, size_t>> scored; // (score, original_index)
            scored.reserve(all_candidates.size());
            for (size_t idx = 0; idx < all_candidates.size(); idx++) {
                double norm_dist = (double)all_candidates[idx].first / max_dist;
                double attr_dist = compute_attr_distance(cur_id, all_candidates[idx].second);
                double score = norm_dist + attr_sort_alpha_ * attr_dist;
                scored.emplace_back(score, idx);
            }
            std::sort(scored.begin(), scored.end());

            // Reorder candidates by mixed score
            std::vector<std::pair<dist_t, tableint>> sorted_candidates;
            sorted_candidates.reserve(all_candidates.size());
            for (auto& [score, idx] : scored) {
                sorted_candidates.push_back(all_candidates[idx]);
            }
            all_candidates = std::move(sorted_candidates);
        } else {
            // Default: sort by distance (ascending)
            std::sort(all_candidates.begin(), all_candidates.end());
        }

        // Heuristic pruning on reordered candidates
        std::vector<std::pair<dist_t, tableint>> return_list;
        std::vector<int> temp_cht;
        if (need_record) temp_cht.resize(table_size_ * attr_type_.size(), 0);

        for (auto& curent_pair : all_candidates) {
            dist_t dist_to_query = curent_pair.first;
            bool good = true;

            for (int i = 0; i < return_list.size(); i++) {
                std::pair<dist_t, tableint> second_pair = return_list[i];
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;

                    if (need_record) {
                        tableint nbr_id = curent_pair.second;
                        (*dominated_list)[i].push_back(nbr_id);
                        pruned_count++;
                    }
                    break;
                }
            }
            if (good) {

                if (return_list.size() >= M) break;

                bool need_push = !need_record;
                if (need_record && cht_low_degree(temp_cht.data(), cur_id, curent_pair.second, return_list.size())) {
                    need_push = true;
                    update_cht(temp_cht.data(), curent_pair.second);
                }
                else{
                    attr_pruned_count++;
                }

                if (need_push)
                {
                    return_list.push_back(curent_pair);
                }
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        heuristic_time += diff.count();
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    void merge_ft(unsigned char* new_ft, unsigned char* old_ft){
        for (int ft_idx = 0; ft_idx < size_per_ft_; ft_idx++) {
            new_ft[ft_idx] |= old_ft[ft_idx];
        }
    }

    // Node-level FT: hash the node's own attributes into its FT slot
    void update_node_ft(tableint node_id) {
        unsigned char* ft = node_ft_at(node_id);
        memset(ft, 0, size_per_ft_);
        updateft(ft, node_id);
    }

    // Merge dominated nodes' attributes into a surviving neighbor's node FT.
    // Called during pruning: surviving_nbr won over dominated nodes,
    // so its FT should reflect that matching nodes are reachable through it.
    void merge_dominated_to_node_ft(tableint surviving_nbr, const std::vector<tableint>& dominated_nodes) {
        unsigned char* ft = node_ft_at(surviving_nbr);
        for (tableint dom_id : dominated_nodes) {
            updateft(ft, dom_id);  // OR dominated node's attr hashes into surviving neighbor's FT
        }
    }

    // Augment edges: for each node with degree < maxM0, search with large efc
    // to find more candidates, then use CHT-based selection to fill gaps.
    // Keeps all existing neighbors, only appends new ones.
    void augment_edges_cht(int efc = 2000, int num_threads = 32) {
        size_t n = cur_element_count;
        fprintf(stderr, "augment_edges_cht: n=%zu, efc=%d, maxM0=%zu, minM0=%zu, threads=%d\n",
                n, efc, maxM0_, minM0_, num_threads);

        size_t orig_efc = ef_construction_;
        ef_construction_ = efc;

        std::atomic<long long> total_added{0};
        std::atomic<size_t> nodes_augmented{0};
        std::atomic<size_t> nodes_skipped{0};
        std::atomic<size_t> progress{0};

        auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1000) num_threads(num_threads)
        for (size_t i = 0; i < n; i++) {
            size_t cur_progress = progress.fetch_add(1);
            if (cur_progress % 100000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double rate = (cur_progress > 0) ? cur_progress / elapsed : 0;
                double eta = (cur_progress > 0) ? (n - cur_progress) / rate : 0;
                fprintf(stderr, "  augment: %zu/%zu (%.1f%%) elapsed=%.0fs eta=%.0fs added=%lld augmented=%zu skipped=%zu\n",
                        cur_progress, n, 100.0 * cur_progress / n, elapsed, eta,
                        total_added.load(), nodes_augmented.load(), nodes_skipped.load());
            }

            linklistsizeint* ll = get_linklist0(i);
            int cur_deg = getListCount(ll);
            if (cur_deg >= (int)maxM0_) {
                nodes_skipped++;
                continue;
            }
            int slots_avail = (int)maxM0_ - cur_deg;

            // Collect existing neighbors into a set for fast lookup
            tableint* existing_nbrs = (tableint*)(ll + 1);
            std::unordered_set<tableint> existing_set;
            existing_set.reserve(cur_deg * 2);
            for (int j = 0; j < cur_deg; j++) {
                existing_set.insert(existing_nbrs[j]);
            }

            // Build CHT from existing neighbors
            std::vector<int> cht(table_size_ * attr_type_.size(), 0);
            for (int j = 0; j < cur_deg; j++) {
                update_cht(cht.data(), existing_nbrs[j]);
            }

            const void* data_point = getDataByInternalId(i);

            // Greedy descent through upper layers
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    linklistsizeint* ll_upper = get_linklist(currObj, level);
                    int sz = getListCount(ll_upper);
                    tableint* datal = (tableint*)(ll_upper + 1);
                    for (int j = 0; j < sz; j++) {
                        dist_t d = fstdistfunc_(data_point, getDataByInternalId(datal[j]), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = datal[j];
                            changed = true;
                        }
                    }
                }
            }

            // Search base layer with large efc (read-only traversal, thread-safe)
            auto top_candidates = searchBaseLayer(currObj, data_point, 0);

            // Sort candidates by mixed score (normalized_dist + alpha * attr_dist)
            std::vector<std::pair<dist_t, tableint>> sorted_cands;
            sorted_cands.reserve(top_candidates.size());
            while (!top_candidates.empty()) {
                auto& top = top_candidates.top();
                tableint cand_id = top.second;
                if (cand_id != (tableint)i && existing_set.find(cand_id) == existing_set.end()) {
                    sorted_cands.emplace_back(top.first, cand_id);
                }
                top_candidates.pop();
            }

            if (!sorted_cands.empty() && attr_sort_alpha_ > 0 && !counting_hash_table_mapping.empty()) {
                dist_t max_dist = 0;
                for (auto& [d, id] : sorted_cands) {
                    if (d > max_dist) max_dist = d;
                }
                if (max_dist <= 0) max_dist = 1.0f;

                std::vector<std::pair<double, size_t>> scored;
                scored.reserve(sorted_cands.size());
                for (size_t idx = 0; idx < sorted_cands.size(); idx++) {
                    double norm_dist = (double)sorted_cands[idx].first / max_dist;
                    double attr_dist = compute_attr_distance((tableint)i, sorted_cands[idx].second);
                    scored.emplace_back(norm_dist + attr_sort_alpha_ * attr_dist, idx);
                }
                std::sort(scored.begin(), scored.end());

                std::vector<std::pair<dist_t, tableint>> reordered;
                reordered.reserve(sorted_cands.size());
                for (auto& [score, idx] : scored) {
                    reordered.push_back(sorted_cands[idx]);
                }
                sorted_cands = std::move(reordered);
            } else {
                std::sort(sorted_cands.begin(), sorted_cands.end());
            }

            // Greedily pick new neighbors using CHT check + mixed-score order
            std::vector<tableint> new_nbrs;
            new_nbrs.reserve(slots_avail);
            for (auto& [dist, cand_id] : sorted_cands) {
                if ((int)new_nbrs.size() >= slots_avail) break;

                if (cht_low_degree(cht.data(), (tableint)i, cand_id, cur_deg + (int)new_nbrs.size())) {
                    new_nbrs.push_back(cand_id);
                    update_cht(cht.data(), cand_id);
                }
            }

            if (!new_nbrs.empty()) {
                // Lock this node and write augmented edges
                {
                    std::unique_lock<std::mutex> lock(link_list_locks_[i]);
                    int cur_sz = getListCount(ll);
                    tableint* data = (tableint*)(ll + 1);
                    int can_add = std::min((int)new_nbrs.size(), (int)maxM0_ - cur_sz);
                    for (int j = 0; j < can_add; j++) {
                        data[cur_sz + j] = new_nbrs[j];
                    }
                    setListCount(ll, cur_sz + can_add);
                    // Do NOT update FT for augmented edges —
                    // keep FT reflecting only original edges so it remains selective
                    total_added += can_add;
                }
                nodes_augmented++;

                // Reverse edges with per-node locks
                for (tableint nbr_id : new_nbrs) {
                    std::unique_lock<std::mutex> lock(link_list_locks_[nbr_id]);
                    linklistsizeint* ll_other = get_linklist0(nbr_id);
                    size_t sz_other = getListCount(ll_other);
                    if (sz_other < maxM0_) {
                        tableint* data_other = (tableint*)(ll_other + 1);
                        bool already = false;
                        for (size_t k = 0; k < sz_other; k++) {
                            if (data_other[k] == (tableint)i) { already = true; break; }
                        }
                        if (!already) {
                            data_other[sz_other] = (tableint)i;
                            setListCount(ll_other, sz_other + 1);
                        }
                    }
                }
            } else {
                nodes_skipped++;
            }
        }

        ef_construction_ = orig_efc;

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(end_time - start_time).count();
        fprintf(stderr, "augment_edges_cht: done in %.1fs. nodes_augmented=%zu, nodes_skipped=%zu, total_edges_added=%lld\n",
                total_sec, nodes_augmented.load(), nodes_skipped.load(), total_added.load());
    }

    // Post-construction graph augmentation: ensure each node has >= min_same
    // neighbors sharing each of its rare labels, to boost filtered-search connectivity.
    //
    // Strategy: group nodes by each label. For rare labels (frequency < freq_threshold),
    // within the group, ensure each node has >= min_same same-label neighbors.
    // If not, BFS within the group to find nearby same-label nodes and add edges.
    //
    // Returns: {edges_added, nodes_augmented}
    std::pair<long long, long long> augment_ft_neighbors(int min_same = 8, int max_hops = 2) {
        long long edges_added = 0;
        long long nodes_augmented = 0;

        int cat_attr_idx = -1;
        for (int i = 0; i < (int)attr_type_.size(); i++) {
            if (attr_type_[i] == 1) { cat_attr_idx = i; break; }
        }
        if (cat_attr_idx < 0) {
            fprintf(stderr, "augment_ft_neighbors: no categorical attribute found\n");
            return {0, 0};
        }

        size_t n = cur_element_count;
        int num_labels = max_cate_size_ + 1;
        fprintf(stderr, "augment_ft_neighbors: n=%zu, min_same=%d, max_hops=%d, maxM0=%zu, num_labels=%d\n",
                n, min_same, max_hops, maxM0_, num_labels);

        // Record original degree for each node (uint8, max 255)
        orig_degree_.resize(n);
        for (size_t i = 0; i < n; i++) {
            linklistsizeint* ll = get_linklist0(i);
            unsigned short int deg = getListCount(ll);
            orig_degree_[i] = (uint8_t)std::min((int)deg, 255);
        }

        // Step 1: Build per-label membership lists
        std::vector<std::vector<tableint>> label_members(num_labels);
        for (size_t i = 0; i < n; i++) {
            const int* cat = attr_at(i, cat_attr_idx);
            for (int k = 0; k < num_labels; k++) {
                int byte_pos = k >> 5;
                int bit_pos = k & 31;
                if (byte_pos < cate_int_byte_ && (cat[byte_pos] & (1 << bit_pos))) {
                    label_members[k].push_back((tableint)i);
                }
            }
        }

        // Step 2: Build per-label membership sets for O(1) lookup
        // Use a flat bitset: label_has[label][node] = true
        // Too much memory for 20 × 4M. Use flat visited tag instead per label.
        std::vector<unsigned char> has_label(n, 0);  // reused per label

        // Flat visited tag for BFS
        std::vector<unsigned int> visited_tag(n, 0);
        unsigned int cur_tag = 0;
        std::vector<tableint> frontier, next_frontier;
        std::vector<std::pair<dist_t, tableint>> candidates;

        // Process labels from rarest to most common (rare labels benefit most)
        std::vector<int> label_order(num_labels);
        std::iota(label_order.begin(), label_order.end(), 0);
        std::sort(label_order.begin(), label_order.end(), [&](int a, int b) {
            return label_members[a].size() < label_members[b].size();
        });

        for (int li = 0; li < num_labels; li++) {
            int label = label_order[li];
            auto& members = label_members[label];
            if (members.empty()) continue;

            double freq = (double)members.size() / n;
            // Skip very common labels (>50% of nodes) — they're useless for filtering
            if (freq > 0.5) {
                fprintf(stderr, "  label %d: %zu members (%.1f%%) — skipping (too common)\n",
                        label, members.size(), freq * 100);
                continue;
            }

            fprintf(stderr, "  label %d: %zu members (%.1f%%)\n",
                    label, members.size(), freq * 100);

            // Mark members of this label
            for (tableint m : members) has_label[m] = 1;

            // For each member, count same-label neighbors
            for (tableint node_id : members) {
                linklistsizeint* ll = get_linklist0(node_id);
                unsigned short int cur_degree = getListCount(ll);
                if (cur_degree >= maxM0_) {
                    has_label[node_id] = 0;
                    continue;
                }

                tableint* nbr_data = (tableint*)(ll + 1);

                int same_count = 0;
                for (int j = 0; j < cur_degree; j++) {
                    if (has_label[nbr_data[j]]) same_count++;
                }

                if (same_count >= min_same) {
                    has_label[node_id] = 0;
                    continue;
                }

                int needed = std::min(min_same - same_count, (int)(maxM0_ - cur_degree));
                if (needed <= 0) {
                    has_label[node_id] = 0;
                    continue;
                }

                // BFS from this node to find same-label candidates
                cur_tag++;
                if (cur_tag == 0) {
                    memset(visited_tag.data(), 0, n * sizeof(unsigned int));
                    cur_tag = 1;
                }

                visited_tag[node_id] = cur_tag;
                frontier.clear();
                for (int j = 0; j < cur_degree; j++) {
                    visited_tag[nbr_data[j]] = cur_tag;
                    frontier.push_back(nbr_data[j]);
                }

                candidates.clear();

                for (int hop = 0; hop < max_hops; hop++) {
                    next_frontier.clear();
                    for (tableint fnode : frontier) {
                        linklistsizeint* ll2 = get_linklist0(fnode);
                        unsigned short int deg2 = getListCount(ll2);
                        tableint* data2 = (tableint*)(ll2 + 1);
                        for (int k = 0; k < deg2; k++) {
                            tableint cand = data2[k];
                            if (visited_tag[cand] == cur_tag) continue;
                            visited_tag[cand] = cur_tag;

                            if (has_label[cand]) {
                                dist_t d = fstdistfunc_(getDataByInternalId(node_id),
                                                        getDataByInternalId(cand),
                                                        dist_func_param_);
                                candidates.push_back({d, cand});
                            }
                            next_frontier.push_back(cand);
                        }
                    }
                    std::swap(frontier, next_frontier);
                    if (frontier.size() > 1500) frontier.resize(1500);
                    if ((int)candidates.size() >= needed + 4) break;
                }

                // Add best candidates
                if (!candidates.empty()) {
                    std::sort(candidates.begin(), candidates.end());
                    int added = 0;
                    cur_degree = getListCount(ll);

                    for (auto& [d, cand_id] : candidates) {
                        if (added >= needed || cur_degree >= maxM0_) break;

                        nbr_data[cur_degree] = cand_id;
                        // Edge-level FT: write edge FT for the new slot
                        if (edge_level_ft_) {
                            unsigned char* eft = edge_ft_at(node_id, cur_degree);
                            memset(eft, 0, size_per_ft_);
                            updateft(eft, cand_id);
                        }
                        cur_degree++;
                        setListCount(ll, cur_degree);
                        edges_added++;
                        added++;

                        // Reverse edge
                        linklistsizeint* ll_c = get_linklist0(cand_id);
                        unsigned short int cd = getListCount(ll_c);
                        if (cd < maxM0_) {
                            tableint* cd_data = (tableint*)(ll_c + 1);
                            cd_data[cd] = (tableint)node_id;
                            // Edge-level FT: write edge FT for reverse edge
                            if (edge_level_ft_) {
                                unsigned char* eft = edge_ft_at(cand_id, cd);
                                memset(eft, 0, size_per_ft_);
                                updateft(eft, node_id);
                            }
                            setListCount(ll_c, cd + 1);
                            edges_added++;
                        }
                    }
                    if (added > 0) {
                        nodes_augmented++;
                        if (!edge_level_ft_) {
                            update_node_ft(node_id);
                        }
                    }
                }

                has_label[node_id] = 0;  // clear as we go
            }

            // Clear remaining has_label flags
            for (tableint m : members) has_label[m] = 0;
        }

        fprintf(stderr, "augment_ft_neighbors: done. edges_added=%lld, nodes_augmented=%lld\n",
                edges_added, nodes_augmented);
        
        // Enable conditional augmented-edge activation during search
        augmented_min_deg_ = min_same;
        use_augmented_edges_ = true;
        fprintf(stderr, "augment_ft_neighbors: augmented edges enabled, min_deg threshold=%d\n", min_same);
        
        return {edges_added, nodes_augmented};
    }

    /**
     * augment_ft_bfs: For every node, ensure at least `min_same` edges pass
     * the node's own "self-FT predicate" (the FT pattern matching the node's attributes).
     * Uses BFS to find nearby same-attribute nodes, replaces tail neighbors if full.
     *
     * @param min_same  Minimum number of edge-FT-passing neighbors required per node
     * @param max_hops  Maximum BFS hops to search for candidates
     * @param num_threads  Number of parallel threads (each thread processes a chunk of nodes)
     */
    std::pair<long long, long long> augment_ft_bfs(int min_same = 4, int max_hops = 3, int num_threads = 1) {
        if (!edge_level_ft_) {
            fprintf(stderr, "augment_ft_bfs: requires edge-level FT\n");
            return {0, 0};
        }

        size_t n = cur_element_count;
        fprintf(stderr, "augment_ft_bfs: n=%zu, min_same=%d, max_hops=%d, maxM0=%zu, ft_bytes=%zu, threads=%d\n",
                n, min_same, max_hops, maxM0_, ft_bytes_, num_threads);

        // Record original degree for each node before augmentation
        orig_degree_.resize(n);
        for (size_t i = 0; i < n; i++) {
            linklistsizeint* ll = get_linklist0(i);
            unsigned short int deg = getListCount(ll);
            orig_degree_[i] = (uint8_t)std::min((int)deg, 255);
        }

        std::atomic<long long> edges_added{0};
        std::atomic<long long> nodes_augmented{0};
        std::atomic<size_t> progress{0};

        auto worker = [&](size_t start, size_t end) {
            // Thread-local buffers
            std::vector<char> self_ft(size_per_ft_, 0);
            std::vector<unsigned int> visited_tag(n, 0);
            unsigned int cur_tag = 0;
            std::vector<tableint> frontier, next_frontier;
            std::vector<std::pair<dist_t, tableint>> candidates;

            for (size_t node_id = start; node_id < end; node_id++) {
                // Progress reporting
                size_t p = progress.fetch_add(1);
                if (p % 500000 == 0) {
                    fprintf(stderr, "  augment_ft_bfs progress: %zu/%zu (%.1f%%)\n", p, n, p*100.0/n);
                }

                linklistsizeint* ll = get_linklist0(node_id);
                unsigned short int cur_degree = getListCount(ll);
                tableint* nbr_data = (tableint*)(ll + 1);

                // Step 1: Build self-FT predicate for this node's attributes
                memset(self_ft.data(), 0, size_per_ft_);
                for (int attr_idx = 0; attr_idx < (int)attr_type_.size(); attr_idx++) {
                    unsigned char* ft_part = (unsigned char*)(self_ft.data() + attr_idx * ft_bytes_);
                    int* _attr = attr_at(node_id, attr_idx);
                    if (attr_type_[attr_idx] == 0) { // numerical
                        int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                              counting_hash_table_mapping[attr_idx].size(), _attr[0]);
                        if (pos >= 0) {
                            int byte_pos = pos >> 3;
                            int bit_pos = pos & 7;
                            if (byte_pos < (int)ft_bytes_) ft_part[byte_pos] |= (1 << bit_pos);
                        }
                    } else { // categorical
                        for (int k = 0; k <= max_cate_size_; ++k) {
                            int byte_pos_attr = k >> 5;
                            int bit_pos_attr = k & 31;
                            if (byte_pos_attr < cate_int_byte_ && (_attr[byte_pos_attr] & (1 << bit_pos_attr))) {
                                int pos = counting_hash_table_mapping[attr_idx][k];
                                if (pos < 0) continue;
                                int byte_pos = pos >> 3;
                                int bit_pos = pos & 7;
                                if (byte_pos < (int)ft_bytes_) ft_part[byte_pos] |= (1 << bit_pos);
                            }
                        }
                    }
                }

                // Step 2: Count how many current edges pass self-FT
                int ft_passing = 0;
                for (int j = 0; j < cur_degree; j++) {
                    if (edge_ft_check(node_id, j, self_ft.data())) {
                        ft_passing++;
                    }
                }

                if (ft_passing >= min_same) continue;

                int needed = min_same - ft_passing;

                // Step 3: BFS to find same-attribute candidates
                cur_tag++;
                if (cur_tag == 0) {
                    memset(visited_tag.data(), 0, n * sizeof(unsigned int));
                    cur_tag = 1;
                }

                visited_tag[node_id] = cur_tag;
                frontier.clear();
                candidates.clear();

                // Seed frontier with current neighbors
                for (int j = 0; j < cur_degree; j++) {
                    tableint nb = nbr_data[j];
                    visited_tag[nb] = cur_tag;
                    frontier.push_back(nb);
                }

                for (int hop = 0; hop < max_hops; hop++) {
                    next_frontier.clear();
                    for (tableint fnode : frontier) {
                        linklistsizeint* ll2 = get_linklist0(fnode);
                        unsigned short int deg2 = getListCount(ll2);
                        tableint* data2 = (tableint*)(ll2 + 1);
                        for (int k = 0; k < deg2; k++) {
                            tableint cand = data2[k];
                            if (visited_tag[cand] == cur_tag) continue;
                            visited_tag[cand] = cur_tag;

                            // Check if candidate's attributes match self-FT
                            // (i.e., if we add edge to cand, will its FT pass our self-predicate?)
                            // This is equivalent to: cand's attributes would set the same FT bits
                            bool attr_match = true;
                            for (int attr_idx = 0; attr_idx < (int)attr_type_.size(); attr_idx++) {
                                const char* pred = self_ft.data() + attr_idx * ft_bytes_;
                                int* cand_attr = attr_at(cand, attr_idx);

                                if (attr_type_[attr_idx] == 0) { // numerical: cand's bucket bit must overlap pred
                                    int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                                          counting_hash_table_mapping[attr_idx].size(), cand_attr[0]);
                                    if (pos < 0) { attr_match = false; break; }
                                    int byte_pos = pos >> 3;
                                    int bit_pos = pos & 7;
                                    if (byte_pos >= (int)ft_bytes_ || !((uint8_t)pred[byte_pos] & (1 << bit_pos))) {
                                        attr_match = false; break;
                                    }
                                } else { // categorical: cand must have all labels in pred
                                    // Check: for each bit set in pred, cand must also have that label
                                    for (int b = 0; b < (int)ft_bytes_; b++) {
                                        // pred bits that are set — cand's FT for this edge must cover them
                                        // Since we're adding edge directly to cand, the edge FT = cand's attr bits
                                        // So check if cand's categorical bits cover pred's categorical bits
                                        uint8_t pred_byte = (uint8_t)pred[b];
                                        if (pred_byte == 0) continue;
                                        // Compute what bits cand would set in this byte
                                        // We need cand's categorical labels to cover the required bits
                                        // This is complex with hash mapping; simpler: just check if cand has same labels
                                        break;
                                    }
                                    // Simplified: check if cand has ALL the categorical labels that node has
                                    int* node_attr = attr_at(node_id, attr_idx);
                                    bool cat_match = true;
                                    for (int byte_i = 0; byte_i < cate_int_byte_; byte_i++) {
                                        if ((node_attr[byte_i] & cand_attr[byte_i]) != node_attr[byte_i]) {
                                            cat_match = false; break;
                                        }
                                    }
                                    if (!cat_match) { attr_match = false; break; }
                                }
                            }

                            if (attr_match) {
                                dist_t d = fstdistfunc_(getDataByInternalId(node_id),
                                                        getDataByInternalId(cand),
                                                        dist_func_param_);
                                candidates.push_back({d, cand});
                            }
                            next_frontier.push_back(cand);
                        }
                    }
                    std::swap(frontier, next_frontier);
                    if (frontier.size() > 2000) frontier.resize(2000);
                    if ((int)candidates.size() >= needed * 4 + 4) break;
                }

                if (candidates.empty()) continue;

                // Step 4: Sort by distance, apply RNG pruning for diversity, pick top `needed`
                std::sort(candidates.begin(), candidates.end());
                
                // RNG pruning: greedily select candidates ensuring diversity
                std::vector<std::pair<dist_t, tableint>> selected;
                selected.reserve(needed + 2);
                for (auto& [d, cand_id] : candidates) {
                    if ((int)selected.size() >= needed) break;
                    
                    // Check if cand_id is already a neighbor
                    bool already_nb = false;
                    for (int j = 0; j < cur_degree; j++) {
                        if (nbr_data[j] == (tableint)cand_id) { already_nb = true; break; }
                    }
                    if (already_nb) continue;

                    // RNG rule: reject if any already-selected augment neighbor is closer
                    // to this candidate than the source node is
                    bool good = true;
                    for (auto& [sd, sid] : selected) {
                        dist_t inter_dist = fstdistfunc_(getDataByInternalId(sid),
                                                         getDataByInternalId(cand_id),
                                                         dist_func_param_);
                        if (inter_dist < d) {
                            good = false;
                            break;
                        }
                    }
                    if (good) {
                        selected.push_back({d, cand_id});
                    }
                }

                int added = 0;
                cur_degree = getListCount(ll);

                for (auto& [d, cand_id] : selected) {
                    if (cur_degree >= maxM0_) {
                        // Replace the last (tail) neighbor
                        int replace_idx = cur_degree - 1;
                        nbr_data[replace_idx] = (tableint)cand_id;
                        unsigned char* eft = edge_ft_at(node_id, replace_idx);
                        memset(eft, 0, size_per_ft_);
                        updateft(eft, cand_id);
                        added++;
                    } else {
                        // Append to neighbor list
                        nbr_data[cur_degree] = (tableint)cand_id;
                        unsigned char* eft = edge_ft_at(node_id, cur_degree);
                        memset(eft, 0, size_per_ft_);
                        updateft(eft, cand_id);
                        cur_degree++;
                        setListCount(ll, cur_degree);
                        added++;
                    }

                    // Add reverse edge (best-effort: only if space available)
                    linklistsizeint* ll_c = get_linklist0(cand_id);
                    unsigned short int cd = getListCount(ll_c);
                    if (cd < maxM0_) {
                        tableint* cd_data = (tableint*)(ll_c + 1);
                        cd_data[cd] = (tableint)node_id;
                        unsigned char* eft_r = edge_ft_at(cand_id, cd);
                        memset(eft_r, 0, size_per_ft_);
                        updateft(eft_r, node_id);
                        setListCount(ll_c, cd + 1);
                    }
                }
                if (added > 0) {
                    edges_added.fetch_add(added);
                    nodes_augmented.fetch_add(1);
                }
            }
        };

        // Run in parallel
        if (num_threads <= 1) {
            worker(0, n);
        } else {
            std::vector<std::thread> threads_vec;
            size_t chunk = (n + num_threads - 1) / num_threads;
            for (int t = 0; t < num_threads; t++) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, n);
                if (start < end) {
                    threads_vec.emplace_back(worker, start, end);
                }
            }
            for (auto& t : threads_vec) t.join();
        }

        fprintf(stderr, "augment_ft_bfs: done. edges_added=%lld, nodes_augmented=%lld\n",
                edges_added.load(), nodes_augmented.load());

        // Enable conditional augmented-edge activation during search
        augmented_min_deg_ = min_same;
        use_augmented_edges_ = true;
        fprintf(stderr, "augment_ft_bfs: augmented edges enabled, min_deg threshold=%d\n", min_same);

        return {edges_added.load(), nodes_augmented.load()};
    }

    /**
     * color_ft_bit: For a single FT bit (attr_idx, bit_idx), ensure that every
     * node whose attributes set this bit is reachable from any layer-1 entry
     * via a path of edges whose edge-FT also has this bit set.
     *
     * Phase 1: collect layer-1 nodes whose attrs include this bit -> seeds.
     * Phase 2: BFS from seeds through edges that already have this bit set;
     *          mark the reachable closure.
     * Phase 3: collect "qualifying" nodes (have this bit) that were not reached.
     * Phase 4: 0-1 BFS on the FULL graph from the reachable set, with
     *          weight 0 on edges that already have this bit set, weight 1 on
     *          edges that don't. For each unreachable qualifying node, walk the
     *          predecessor chain back and OR the bit into edge-FT (both
     *          directions, since the graph is undirected). This minimises the
     *          number of bit-flips needed to repair connectivity.
     *
     * Returns: {seeds, unreachable_before_repair, bits_flipped}
     */
    std::tuple<long long, long long, long long>
    color_ft_bit(int attr_idx, int bit_idx, int K = 1) {
        // DEPRECATED: replaced by color_ft_bit_voronoi.
        // The old "repair only unreachable" approach gave near-zero recall improvement
        // because reachable nodes (the vast majority) were never given additional FT
        // paths. Use color_ft_bit_voronoi for true per-node connectivity to nearby
        // qualifying nodes.
        (void)attr_idx; (void)bit_idx; (void)K;
        fprintf(stderr, "color_ft_bit: DEPRECATED — use color_ft_bit_voronoi\n");
        return {0, 0, 0};
    }

    // Internal legacy implementation (kept for reference, not bound).
    std::tuple<long long, long long, long long>
    color_ft_bit_legacy(int attr_idx, int bit_idx, int K = 1) {
        if (!edge_level_ft_) {
            fprintf(stderr, "color_ft_bit: requires edge-level FT\n");
            return {0, 0, 0};
        }
        if (attr_idx < 0 || attr_idx >= (int)attr_type_.size()) {
            fprintf(stderr, "color_ft_bit: attr_idx out of range\n");
            return {0, 0, 0};
        }
        int byte_pos = bit_idx >> 3;
        int bit_in_byte = bit_idx & 7;
        if (byte_pos < 0 || byte_pos >= (int)ft_bytes_) {
            fprintf(stderr, "color_ft_bit: bit_idx out of range\n");
            return {0, 0, 0};
        }
        unsigned char bit_mask = (unsigned char)(1u << bit_in_byte);
        size_t N = cur_element_count;

        // -------- helpers --------
        auto node_has_bit = [&](tableint v) -> bool {
            int* a = attr_at(v, attr_idx);
            if (attr_type_[attr_idx] == 0) {  // numerical
                int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                      counting_hash_table_mapping[attr_idx].size(), a[0]);
                return pos == bit_idx;
            } else {  // categorical
                for (int k = 0; k <= max_cate_size_; ++k) {
                    int bp = k >> 5;
                    int bi = k & 31;
                    if (bp < cate_int_byte_ && (a[bp] & (1 << bi))) {
                        int pos = counting_hash_table_mapping[attr_idx][k];
                        if (pos == bit_idx) return true;
                    }
                }
                return false;
            }
        };

        auto edge_has_bit = [&](tableint u, int j) -> bool {
            unsigned char* eft = edge_ft_at(u, j, attr_idx);
            return (eft[byte_pos] & bit_mask) != 0;
        };

        auto edge_set_bit = [&](tableint u, int j) -> bool {
            unsigned char* eft = edge_ft_at(u, j, attr_idx);
            if (eft[byte_pos] & bit_mask) return false;
            eft[byte_pos] |= bit_mask;
            return true;
        };

        // -------- Phase 1+2: BFS through bit-b edges --------
        // Sources: ALL layer-1 nodes that have this bit. (We want connectivity
        // from any layer-1 entry that has a bit-b edge to start expansion.)
        std::vector<uint8_t> reachable(N, 0);
        std::vector<tableint> queue;
        queue.reserve(N / 8 + 16);

        for (size_t v = 0; v < N; v++) {
            if (element_levels_[v] >= 1 && node_has_bit(v)) {
                reachable[v] = 1;
                queue.push_back((tableint)v);
            }
        }
        long long n_seeds = (long long)queue.size();

        size_t head = 0;
        while (head < queue.size()) {
            tableint u = queue[head++];
            linklistsizeint* ll = get_linklist0(u);
            unsigned short cur_deg = getListCount(ll);
            tableint* nbr = (tableint*)(ll + 1);
            for (int j = 0; j < cur_deg; j++) {
                if (!edge_has_bit(u, j)) continue;
                tableint w = nbr[j];
                if (!reachable[w]) {
                    reachable[w] = 1;
                    queue.push_back(w);
                }
            }
        }

        // -------- Phase 3: collect unreachable qualifying nodes --------
        std::vector<tableint> unreachable_qual;
        for (size_t v = 0; v < N; v++) {
            if (!reachable[v] && node_has_bit(v)) {
                unreachable_qual.push_back((tableint)v);
            }
        }
        long long n_unreach = (long long)unreachable_qual.size();

        if (n_unreach == 0) {
            return {n_seeds, 0, 0};
        }

        // -------- Phase 4: K rounds of repair, each forcing edge-disjoint paths --------
        // Round 1: standard 0-1 BFS, weight=0 on edges with bit set.
        // Round r > 1: edges flipped in PREVIOUS rounds get weight=1 (disqualified
        //   from being weight-0 shortcut), forcing the BFS to find an alternative
        //   path. Original bit-b edges (set BEFORE coloring) remain weight-0 and
        //   may be reused. This produces K largely edge-disjoint repair paths per
        //   unreachable node, ensuring K-redundant connectivity to the source set.
        const int INF = std::numeric_limits<int>::max();

        // Snapshot original bit-b edges (so we know which edges are "free" across
        // all rounds). Stored as a flat per-node bitset over local edge indices.
        // Memory: ~maxM0_ bits per node = ~10 bytes per node => 40MB for 4M.
        std::vector<uint8_t> orig_bit_set;  // [v * maxM0_ + j] = 1 if edge originally had bit
        orig_bit_set.assign(N * maxM0_, 0);
        for (size_t v = 0; v < N; v++) {
            linklistsizeint* ll = get_linklist0(v);
            unsigned short cur_deg = getListCount(ll);
            for (int j = 0; j < cur_deg; j++) {
                if (edge_has_bit(v, j)) orig_bit_set[v * maxM0_ + j] = 1;
            }
        }

        long long bits_flipped = 0;
        long long unrepairable_last = 0;

        // Snapshot original Phase-4 source set (do NOT add round-r repaired nodes
        // to source set, otherwise round 2 would find trivial 0-length paths).
        std::vector<uint8_t> orig_phase4_source(N, 0);
        for (size_t v = 0; v < N; v++) {
            if (reachable[v] || element_levels_[v] >= 1) {
                orig_phase4_source[v] = 1;
            }
        }

        // Track which edges have been flipped in any previous round.
        // Round r > 0 BFS will FORBID these edges (skip them entirely),
        // forcing round r's repair path to be edge-disjoint from prior rounds.
        std::vector<uint8_t> flipped_prev(N * maxM0_, 0);

        for (int round = 0; round < K; round++) {
            std::vector<int> dist(N, INF);
            std::vector<tableint> pred(N, (tableint)-1);
            std::deque<tableint> dq;

            for (size_t v = 0; v < N; v++) {
                if (orig_phase4_source[v]) {
                    dist[v] = 0;
                    pred[v] = (tableint)-1;
                    dq.push_back((tableint)v);
                }
            }

            while (!dq.empty()) {
                tableint u = dq.front();
                dq.pop_front();
                int du = dist[u];
                linklistsizeint* ll = get_linklist0(u);
                unsigned short cur_deg = getListCount(ll);
                tableint* nbr = (tableint*)(ll + 1);
                for (int j = 0; j < cur_deg; j++) {
                    // Forbid edges flipped in any previous round (forces disjoint paths)
                    if (flipped_prev[u * maxM0_ + j]) continue;
                    tableint w = nbr[j];
                    int weight = orig_bit_set[u * maxM0_ + j] ? 0 : 1;
                    int nd = du + weight;
                    if (nd < dist[w]) {
                        dist[w] = nd;
                        pred[w] = u;
                        if (weight == 0) dq.push_front(w);
                        else dq.push_back(w);
                    }
                }
            }

            // For each unreachable qualifying node, walk pred chain back to source,
            // flip bits and record flipped edges so next round forbids them.
            long long unrepairable = 0;
            long long round_flips = 0;
            for (tableint u : unreachable_qual) {
                std::vector<tableint> chain;
                tableint cur = u;
                while (cur != (tableint)-1 && !orig_phase4_source[cur]) {
                    chain.push_back(cur);
                    if (pred[cur] == cur) break;
                    cur = pred[cur];
                }
                if (cur == (tableint)-1) {
                    unrepairable++;
                    continue;
                }
                tableint child = cur;
                for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                    tableint parent = child;
                    tableint kid = *it;

                    linklistsizeint* ll_p = get_linklist0(parent);
                    unsigned short deg_p = getListCount(ll_p);
                    tableint* nbr_p = (tableint*)(ll_p + 1);
                    for (int j = 0; j < deg_p; j++) {
                        if (nbr_p[j] == kid) {
                            if (edge_set_bit(parent, j)) bits_flipped++, round_flips++;
                            flipped_prev[parent * maxM0_ + j] = 1;
                            break;
                        }
                    }
                    linklistsizeint* ll_k = get_linklist0(kid);
                    unsigned short deg_k = getListCount(ll_k);
                    tableint* nbr_k = (tableint*)(ll_k + 1);
                    for (int j = 0; j < deg_k; j++) {
                        if (nbr_k[j] == parent) {
                            if (edge_set_bit(kid, j)) bits_flipped++, round_flips++;
                            flipped_prev[kid * maxM0_ + j] = 1;
                            break;
                        }
                    }
                    child = kid;
                }
            }
            unrepairable_last = unrepairable;
            (void)round_flips;
        }

        if (unrepairable_last > 0) {
            fprintf(stderr, "  color_ft_bit(attr=%d,bit=%d,K=%d): %lld unreachable could not be repaired (disconnected components)\n",
                    attr_idx, bit_idx, K, unrepairable_last);
        }
        return {n_seeds, n_unreach, bits_flipped};
    }

    /**
     * color_all_ft_bits: iterate over all attribute bits and repair
     * connectivity per bit. Returns total bits flipped.
     */
    long long color_all_ft_bits(int K = 1, bool verbose = true) {
        // DEPRECATED: see color_ft_bit. Use color_all_ft_bits_voronoi instead.
        (void)K; (void)verbose;
        fprintf(stderr, "color_all_ft_bits: DEPRECATED — use color_all_ft_bits_voronoi\n");
        return 0;
    }

    long long color_all_ft_bits_legacy(int K = 1, bool verbose = true) {
        if (!edge_level_ft_) {
            fprintf(stderr, "color_all_ft_bits: requires edge-level FT\n");
            return 0;
        }
        long long total_flipped = 0;
        long long total_unreach = 0;
        size_t N = cur_element_count;
        int n_attrs = (int)attr_type_.size();
        int total_bits = n_attrs * (int)(ft_bytes_ * 8);
        fprintf(stderr, "color_all_ft_bits: N=%zu, attrs=%d, ft_bytes=%zu, total_bits=%d, K=%d\n",
                N, n_attrs, ft_bytes_, total_bits, K);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int a = 0; a < n_attrs; a++) {
            for (int b = 0; b < (int)ft_bytes_ * 8; b++) {
                auto bt = std::chrono::high_resolution_clock::now();
                auto [seeds, unreach, flipped] = color_ft_bit_legacy(a, b, K);
                auto et = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(et - bt).count();
                total_flipped += flipped;
                total_unreach += unreach;
                if (verbose) {
                    fprintf(stderr, "  bit (attr=%d,b=%d): seeds=%lld, unreach_qual=%lld, flipped=%lld, %.2fs\n",
                            a, b, seeds, unreach, flipped, sec);
                }
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "color_all_ft_bits: done in %.1fs. total unreach=%lld, total bits flipped=%lld\n",
                total_sec, total_unreach, total_flipped);
        return total_flipped;
    }

    /**
     * color_ft_bit_voronoi: per-bit connectivity coloring via multi-source BFS.
     *
     * Goal: for every qualifying node q (node_has_bit=true), guarantee that at
     * least K_neighbors of its "Voronoi neighbor" qualifying nodes are reachable
     * from q via FT-bit-set edges only (without traversing distant chains).
     *
     * Algorithm (per bit):
     *   1. Multi-source BFS on full HNSW from all qualifying nodes. For every
     *      node v: source[v] = nearest qualifying, parent[v] = BFS-tree parent.
     *   2. Scan all directed edges (u -> w). If source[u] != source[w], this
     *      edge bridges two Voronoi regions. Track best (shortest combined
     *      dist[u]+dist[w]+1) bridge per ordered (source[u], source[w]) pair.
     *   3. For each qualifying source s, pick the K_neighbors closest other
     *      sources (by best-bridge distance). For each picked bridge: walk
     *      parent chain from u back to source[u], from w back to source[w],
     *      and color the bridge edge itself. All edges set bit (both directions).
     *
     * Returns: (n_qualifying, n_selected_bridges, bits_flipped).
     */
    std::tuple<long long, long long, long long>
    color_ft_bit_voronoi(int attr_idx, int bit_idx, int K_neighbors = 4) {
        if (!edge_level_ft_) {
            fprintf(stderr, "color_ft_bit_voronoi: requires edge-level FT\n");
            return {0, 0, 0};
        }
        if (attr_idx < 0 || attr_idx >= (int)attr_type_.size()) {
            fprintf(stderr, "color_ft_bit_voronoi: attr_idx out of range\n");
            return {0, 0, 0};
        }
        int byte_pos = bit_idx >> 3;
        int bit_in_byte = bit_idx & 7;
        if (byte_pos < 0 || byte_pos >= (int)ft_bytes_) {
            fprintf(stderr, "color_ft_bit_voronoi: bit_idx out of range\n");
            return {0, 0, 0};
        }
        unsigned char bit_mask = (unsigned char)(1u << bit_in_byte);
        size_t N = cur_element_count;

        auto node_has_bit = [&](tableint v) -> bool {
            int* a = attr_at(v, attr_idx);
            if (attr_type_[attr_idx] == 0) {
                int pos = lower_bound(&counting_hash_table_mapping[attr_idx][0],
                                      counting_hash_table_mapping[attr_idx].size(), a[0]);
                return pos == bit_idx;
            } else {
                for (int k = 0; k <= max_cate_size_; ++k) {
                    int bp = k >> 5;
                    int bi = k & 31;
                    if (bp < cate_int_byte_ && (a[bp] & (1 << bi))) {
                        int pos = counting_hash_table_mapping[attr_idx][k];
                        if (pos == bit_idx) return true;
                    }
                }
                return false;
            }
        };

        auto edge_set_bit = [&](tableint u, int j) -> bool {
            unsigned char* eft = edge_ft_at(u, j, attr_idx);
            if (eft[byte_pos] & bit_mask) return false;
            eft[byte_pos] |= bit_mask;
            return true;
        };

        // 1) Multi-source BFS from all qualifying nodes
        std::vector<tableint> source(N, (tableint)-1);
        std::vector<tableint> parent(N, (tableint)-1);
        std::vector<int> dist(N, std::numeric_limits<int>::max());
        std::vector<tableint> bfs_q;
        bfs_q.reserve(N);

        long long n_qual = 0;
        for (size_t v = 0; v < N; v++) {
            if (node_has_bit((tableint)v)) {
                source[v] = (tableint)v;
                parent[v] = (tableint)-1;
                dist[v] = 0;
                bfs_q.push_back((tableint)v);
                n_qual++;
            }
        }
        if (n_qual == 0) return {0, 0, 0};

        size_t head = 0;
        while (head < bfs_q.size()) {
            tableint u = bfs_q[head++];
            int du = dist[u];
            linklistsizeint* ll = get_linklist0(u);
            unsigned short cur_deg = getListCount(ll);
            tableint* nbr = (tableint*)(ll + 1);
            for (int j = 0; j < cur_deg; j++) {
                tableint w = nbr[j];
                if (dist[w] == std::numeric_limits<int>::max()) {
                    dist[w] = du + 1;
                    parent[w] = u;
                    source[w] = source[u];
                    bfs_q.push_back(w);
                }
            }
        }

        // 2) Find best bridge per ordered (source_u, source_w) pair
        struct BridgeInfo {
            int dist_sum;       // dist[u] + 1 + dist[w]
            tableint u;
            tableint w;
            int j_in_u;         // edge index of w in u's neighbor list
        };
        // Key: pack two tableint into uint64
        auto pack_key = [](tableint a, tableint b) -> uint64_t {
            return ((uint64_t)a << 32) | (uint64_t)b;
        };
        std::unordered_map<uint64_t, BridgeInfo> best_bridge;
        best_bridge.reserve(N);

        for (size_t v = 0; v < N; v++) {
            tableint sv = source[v];
            if (sv == (tableint)-1) continue;
            linklistsizeint* ll = get_linklist0(v);
            unsigned short cur_deg = getListCount(ll);
            tableint* nbr = (tableint*)(ll + 1);
            int dv = dist[v];
            for (int j = 0; j < cur_deg; j++) {
                tableint w = nbr[j];
                tableint sw = source[w];
                if (sw == (tableint)-1 || sw == sv) continue;
                int bd = dv + 1 + dist[w];
                uint64_t key = pack_key(sv, sw);
                auto it = best_bridge.find(key);
                if (it == best_bridge.end() || it->second.dist_sum > bd) {
                    best_bridge[key] = {bd, (tableint)v, w, j};
                }
            }
        }

        // 3) Group bridges by source, pick top K_neighbors closest
        std::unordered_map<tableint, std::vector<std::pair<int, uint64_t>>> per_source;
        per_source.reserve(n_qual);
        for (auto& kv : best_bridge) {
            tableint s_u = (tableint)(kv.first >> 32);
            per_source[s_u].push_back({kv.second.dist_sum, kv.first});
        }

        // Selected bridges (use set of keys)
        std::unordered_set<uint64_t> selected;
        selected.reserve(per_source.size() * (size_t)K_neighbors);
        for (auto& kv : per_source) {
            auto& v = kv.second;
            if ((int)v.size() > K_neighbors) {
                std::nth_element(v.begin(), v.begin() + K_neighbors, v.end(),
                    [](const std::pair<int, uint64_t>& a, const std::pair<int, uint64_t>& b) {
                        return a.first < b.first;
                    });
                v.resize(K_neighbors);
            }
            for (auto& p : v) selected.insert(p.second);
        }

        // 4) Color: for each selected bridge, color path u->source[u], path w->source[w], bridge edge
        long long flipped = 0;
        long long n_bridges = (long long)selected.size();

        auto color_edge_bidir = [&](tableint a, tableint b) {
            linklistsizeint* ll_a = get_linklist0(a);
            unsigned short da = getListCount(ll_a);
            tableint* nbr_a = (tableint*)(ll_a + 1);
            for (int j = 0; j < da; j++) {
                if (nbr_a[j] == b) { if (edge_set_bit(a, j)) flipped++; break; }
            }
            linklistsizeint* ll_b = get_linklist0(b);
            unsigned short db = getListCount(ll_b);
            tableint* nbr_b = (tableint*)(ll_b + 1);
            for (int j = 0; j < db; j++) {
                if (nbr_b[j] == a) { if (edge_set_bit(b, j)) flipped++; break; }
            }
        };

        auto color_chain_to_source = [&](tableint v) {
            tableint cur = v;
            while (parent[cur] != (tableint)-1) {
                tableint p = parent[cur];
                color_edge_bidir(p, cur);
                cur = p;
            }
        };

        for (uint64_t key : selected) {
            const BridgeInfo& bi = best_bridge[key];
            color_edge_bidir(bi.u, bi.w);
            color_chain_to_source(bi.u);
            color_chain_to_source(bi.w);
        }

        return {n_qual, n_bridges, flipped};
    }

    long long color_all_ft_bits_voronoi(int K_neighbors = 4, bool verbose = true) {
        if (!edge_level_ft_) {
            fprintf(stderr, "color_all_ft_bits_voronoi: requires edge-level FT\n");
            return 0;
        }
        long long total_flipped = 0;
        long long total_bridges = 0;
        size_t N = cur_element_count;
        int n_attrs = (int)attr_type_.size();
        int total_bits = n_attrs * (int)(ft_bytes_ * 8);
        fprintf(stderr, "color_all_ft_bits_voronoi: N=%zu, attrs=%d, ft_bytes=%zu, total_bits=%d, K_neighbors=%d\n",
                N, n_attrs, ft_bytes_, total_bits, K_neighbors);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int a = 0; a < n_attrs; a++) {
            for (int b = 0; b < (int)ft_bytes_ * 8; b++) {
                auto bt = std::chrono::high_resolution_clock::now();
                auto [n_qual, n_bridges, flipped] = color_ft_bit_voronoi(a, b, K_neighbors);
                auto et = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(et - bt).count();
                total_flipped += flipped;
                total_bridges += n_bridges;
                if (verbose) {
                    fprintf(stderr, "  bit (attr=%d,b=%d): qual=%lld bridges=%lld flipped=%lld %.2fs\n",
                            a, b, n_qual, n_bridges, flipped, sec);
                }
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "color_all_ft_bits_voronoi: done in %.1fs. total bridges=%lld, bits flipped=%lld\n",
                total_sec, total_bridges, total_flipped);
        return total_flipped;
    }

    double update_ft_time{0.0};

    // Edge-level update_nbr_ft — replaced by node-level update_node_ft
    // Kept commented out for reference
    /*
    void update_nbr_ft(tableint id, std::vector<tableint>& selectedNeighbors, std::vector<std::vector<tableint>>& dominated_list) {
        // ... edge-level FT logic removed (see git history) ...
    }
    */

    long long dominate_count{0};
    std::vector<int> node_dominate_count_; // accumulated domination events per node during construction

    // Per-node original degree marker (uint8).
    // After augment_ft_neighbors(), neighbors[0..orig_degree_[i]) are original edges,
    // neighbors[orig_degree_[i]..total_degree) are augmented edges.
    // During search, augmented edges only activate when FT-matched original neighbors < augmented_min_deg_.
    std::vector<uint8_t> orig_degree_;
    int augmented_min_deg_{0};
    bool use_augmented_edges_{false};

    void print_build_stats() const {
        if (cur_element_count == 0) return;

        double total_degree = 0.0;
        for (size_t i = 0; i < cur_element_count; i++) {
            total_degree += getListCount(get_linklist0(i));
        }

        std::cout << "build stats: avg degree=" << (total_degree / cur_element_count)
              << ", avg dominate size=" << (static_cast<double>(dominate_count) / static_cast<double>(cur_element_count))
                  << std::endl;
    }

    void maybe_print_build_stats() {
        if (build_stats_printed_) return;
        if (cur_element_count == 0) return;
        if (node_dominate_count_.empty()) return;

        print_build_stats();
        build_stats_printed_ = true;
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        // std::cout << "connecting " << cur_c << " at level " << level << " with " << top_candidates.size() << " candidates" << std::endl;
        std::vector<std::vector<tableint>> dominated_list(level ? 0 : maxM0_); // for each selected neighbor, the list of points it dominates
        
        if (level == 0){
            for (int i = 0; i < maxM0_; i++) {
                dominated_list[i].reserve(maxM0_);
            }
        }
        // std::cout << "level==0?" << (level==0) << std::endl;
        getNeighborsByHeuristic2(top_candidates, level==0 ? Mcurmax : M_, level==0, cur_c, &dominated_list);

        for(int i = 0; i < dominated_list.size(); i++) {
            dominate_count += static_cast<long long>(dominated_list[i].size());
        }

        // std::cout << "connecting for id " << cur_c << " at level " << level << " with " << top_candidates.size() << " candidates, pruned count: " << pruned_count << std::endl;
        if (top_candidates.size() > (level==0 ? Mcurmax : M_))
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(level==0 ? Mcurmax : M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }
        std::reverse(selectedNeighbors.begin(), selectedNeighbors.end());

        // accumulate per-node domination events (dominated_list[i] was dominated by selectedNeighbors[i])
        for (int i = 0; i < (int)dominated_list.size() && i < (int)selectedNeighbors.size(); i++) {
            node_dominate_count_[selectedNeighbors[i]] += (int)dominated_list[i].size();
        }

        // FT update: node-level merges into neighbor's FT; edge-level writes to cur_c's edge slots
        if (level == 0) {
            if (edge_level_ft_) {
                // Edge-level: write edge FT for each edge slot of cur_c
                for (int i = 0; i < (int)selectedNeighbors.size(); i++) {
                    unsigned char* eft = edge_ft_at(cur_c, i);
                    memset(eft, 0, size_per_ft_);
                    updateft(eft, selectedNeighbors[i]);  // neighbor's own attr
                    for (tableint dom_id : dominated_list[i]) {
                        updateft(eft, dom_id);  // dominated nodes' attrs
                    }
                }
            } else {
                // Node-level FT: merge dominated nodes' attrs into surviving neighbors' FTs
                for (int i = 0; i < (int)dominated_list.size() && i < (int)selectedNeighbors.size(); i++) {
                    if (!dominated_list[i].empty()) {
                        merge_dominated_to_node_ft(selectedNeighbors[i], dominated_list[i]);
                    }
                }
            }
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                    // Edge-level FT: write edge FT for the new slot (cur_c's attr only, no domination)
                    if (edge_level_ft_ && level == 0) {
                        unsigned char* eft = edge_ft_at(selectedNeighbors[idx], sz_link_list_other);
                        memset(eft, 0, size_per_ft_);
                        updateft(eft, cur_c);
                    }
                } else {
                    // for bottom layer, edges within valid_M0_ follows traditional rng prune
                    // edges above valid_M0_, prune redundant edge by attribute
                    // for num attr, reserve if |{[attr(cur), attr(nbr)]}| < min_M0_
                    // for cate attr, reserve if {attr(cur) U attr(nbr)} not empty and |{attr(cur) ∪ attr(nbr)}| < min_M0_

                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    if (level == 0){
                        for (int i = 0; i < dominated_list.size(); i++) {
                            dominated_list[i].clear();
                        }
                    }
                    getNeighborsByHeuristic2(candidates, Mcurmax, level==0, selectedNeighbors[idx], &dominated_list);

                    for(int i = 0; i < dominated_list.size(); i++) {
                        dominate_count += static_cast<long long>(dominated_list[i].size());
                    }

                    int indx = 0;
                    std::vector<tableint> selectedNeighbors_other;
                    selectedNeighbors_other.reserve(M_);
                    while (candidates.size() > 0) {
                        // data[indx] = candidates.top().second;
                        selectedNeighbors_other.push_back(candidates.top().second);
                        candidates.pop();
                        indx++;
                    }
                    std::reverse(selectedNeighbors_other.begin(), selectedNeighbors_other.end());

                    // accumulate per-node domination events for reverse connection update
                    for (int i = 0; i < (int)dominated_list.size() && i < (int)selectedNeighbors_other.size(); i++) {
                        node_dominate_count_[selectedNeighbors_other[i]] += (int)dominated_list[i].size();
                    }

                    // FT update for reverse re-prune
                    if (level == 0) {
                        if (edge_level_ft_) {
                            // Edge-level: rewrite all edge FTs for selectedNeighbors[idx]
                            for (int i = 0; i < (int)selectedNeighbors_other.size(); i++) {
                                unsigned char* eft = edge_ft_at(selectedNeighbors[idx], i);
                                memset(eft, 0, size_per_ft_);
                                updateft(eft, selectedNeighbors_other[i]);  // neighbor's own attr
                                if (i < (int)dominated_list.size()) {
                                    for (tableint dom_id : dominated_list[i]) {
                                        updateft(eft, dom_id);
                                    }
                                }
                            }
                            // Clear remaining edge FT slots
                            for (int i = (int)selectedNeighbors_other.size(); i < (int)Mcurmax; i++) {
                                unsigned char* eft = edge_ft_at(selectedNeighbors[idx], i);
                                memset(eft, 0, size_per_ft_);
                            }
                        } else {
                            // Node-level FT: merge dominated nodes' attrs into surviving neighbors' FTs
                            for (int i = 0; i < (int)dominated_list.size() && i < (int)selectedNeighbors_other.size(); i++) {
                                if (!dominated_list[i].empty()) {
                                    merge_dominated_to_node_ft(selectedNeighbors_other[i], dominated_list[i]);
                                }
                            }
                        }
                    }

                    for (int i = 0; i < selectedNeighbors_other.size(); i++) {
                        data[i] = selectedNeighbors_other[i];
                    }


                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }
        // std::cout << "connect done" << std::endl;

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
        node_dominate_count_.resize(new_max_elements, 0);
    }


    /*
     * Repair a single node: re-search neighbors (skipping deleted nodes)
     * and reconnect via mutuallyConnectNewElement (isUpdate=true).
     * This rebuilds out-edges, FT, and updates reverse edges.
     * Returns true if the node was successfully repaired.
     */
    bool repairNode(tableint internal_id) {
        if (internal_id >= cur_element_count) return false;
        if (isMarkedDeleted(internal_id)) return false;

        const void* data_point = getDataByInternalId(internal_id);

        // Search for nearest non-deleted neighbors from the entry point
        // searchBaseLayer already skips deleted nodes in top_candidates
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates = searchBaseLayer(enterpoint_node_, data_point, 0);

        // Remove self from candidates
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> filtered;
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            if (id != internal_id) {
                filtered.emplace(dist, id);
            }
        }

        if (filtered.empty()) return false;

        // Reconnect: isUpdate=true allows overwriting existing edges
        mutuallyConnectNewElement(data_point, internal_id, filtered, 0, true);
        return true;
    }


    /*
     * Batch repair: process all current repair candidates.
     * For each candidate, re-search and reconnect edges (level 0 only).
     * Returns the number of nodes successfully repaired.
     */
    size_t repairCandidates() {
        auto candidates = popRepairCandidates();
        size_t repaired = 0;
        for (auto& [node_id, dead_ratio] : candidates) {
            if (repairNode(node_id)) {
                repaired++;
            }
        }
        return repaired;
    }


    /*
     * Global graph rebuild: zero all edges & FT, then re-insert all live nodes.
     * Deleted nodes are truly removed (compacted out).
     * counting_hash_table_mapping (FT codebook) is preserved.
     * Returns the number of live nodes after rebuild.
     *
     * NOT thread-safe — caller must ensure exclusive access.
     */
    size_t rebuildGraph() {
        size_t N = cur_element_count;
        if (N == 0) return 0;

        // --- Step 1: collect live nodes ---
        struct NodeInfo {
            tableint old_id;
            labeltype label;
            int level;
            std::vector<char> data;   // vector data
            std::vector<int> attr;    // raw attr bytes
        };
        std::vector<NodeInfo> live_nodes;
        live_nodes.reserve(N);

        for (tableint i = 0; i < N; i++) {
            if (isMarkedDeleted(i)) continue;
            NodeInfo ni;
            ni.old_id = i;
            ni.label = getExternalLabel(i);
            ni.level = element_levels_[i];
            ni.data.resize(data_size_);
            memcpy(ni.data.data(), getDataByInternalId(i), data_size_);
            int attr_ints = attr_size_per_item_;
            ni.attr.resize(attr_ints);
            int* attr_src = (int*)(data_level0_memory_ + i * size_data_per_element_ + offsetAttr_);
            memcpy(ni.attr.data(), attr_src, attr_ints * sizeof(int));
            live_nodes.push_back(std::move(ni));
        }

        size_t live_count = live_nodes.size();

        // --- Step 2: zero all level-0 memory ---
        memset(data_level0_memory_, 0, max_elements_ * size_data_per_element_);

        // --- Step 3: free upper-layer link lists ---
        for (size_t i = 0; i < N; i++) {
            if (linkLists_[i]) {
                free(linkLists_[i]);
                linkLists_[i] = nullptr;
            }
        }

        // --- Step 4: reset all graph state ---
        cur_element_count = 0;
        num_deleted_ = 0;
        label_lookup_.clear();
        deleted_elements.clear();
        enterpoint_node_ = -1;
        maxlevel_ = -1;
        clearRepairCandidates();
        dominate_count = 0;
        build_stats_printed_ = false;

        // Clear auxiliary ID-based structures (stale after compaction)
        ep_ids_.clear();
        bucket_data_.clear();
        bucket_offsets_.clear();
        id_to_buckets_.clear();
        bucket_size_ = 0;
        btrees.clear();
        ivf.clear();
        if (counting_hash_table) {
            delete[] counting_hash_table;
            counting_hash_table = nullptr;
        }

        std::fill(element_levels_.begin(), element_levels_.end(), 0);
        std::fill(node_dominate_count_.begin(), node_dominate_count_.end(), 0);

        if (live_count == 0) return 0;

        // --- Step 5 & 6: re-insert each live node ---
        for (size_t idx = 0; idx < live_count; idx++) {
            auto& ni = live_nodes[idx];
            tableint cur_c = cur_element_count;
            cur_element_count++;

            label_lookup_[ni.label] = cur_c;
            element_levels_[cur_c] = ni.level;

            // Write data, label, attr into compacted position
            memcpy(getDataByInternalId(cur_c), ni.data.data(), data_size_);
            memcpy(getExternalLabeLp(cur_c), &ni.label, sizeof(labeltype));
            int* attr_dst = (int*)(data_level0_memory_ + cur_c * size_data_per_element_ + offsetAttr_);
            memcpy(attr_dst, ni.attr.data(), ni.attr.size() * sizeof(int));

            // Node-level FT: rebuild after attrs are set
            update_node_ft(cur_c);

            // Allocate upper layer if needed
            if (ni.level > 0) {
                linkLists_[cur_c] = (char*) malloc(size_links_per_element_ * ni.level + 1);
                if (!linkLists_[cur_c])
                    throw std::runtime_error("rebuildGraph: failed to allocate upper layer link list");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * ni.level + 1);
            }

            // First node: just set as entry point, no edges to build
            if (cur_c == 0) {
                enterpoint_node_ = 0;
                maxlevel_ = ni.level;
                continue;
            }

            // Snapshot current state before insertion (mirrors addPoint logic)
            int maxlevelcopy = maxlevel_;
            tableint currObj = enterpoint_node_;
            const void* data_point = getDataByInternalId(cur_c);

            // Greedy descent through upper layers (only if cur node's level < current max)
            if (ni.level < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int lev = maxlevelcopy; lev > ni.level; lev--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int* data = get_linklist(currObj, lev);
                        int sz = getListCount(data);
                        tableint* datal = (tableint*)(data + 1);
                        for (int j = 0; j < sz; j++) {
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(datal[j]), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = datal[j];
                                changed = true;
                            }
                        }
                    }
                }
            }

            // Connect at each level (search + mutual connect)
            for (int lev = std::min(ni.level, maxlevelcopy); lev >= 0; lev--) {
                auto top_candidates = searchBaseLayer(currObj, data_point, lev);
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, lev, false);
            }

            // Update entrypoint AFTER insertion (fixes higher-level race)
            if (ni.level > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = ni.level;
            }
        }

        return live_count;
    }


    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;
        size += sizeof(int);  // ft_format_version

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        maybe_print_build_stats();

        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, top_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, nbr_size_per_element);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M0_mul_);
        writeBinaryPOD(output, minM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);
        writeBinaryPOD(output, ft_bits_);
        writeBinaryPOD(output, cate_int_byte_);
        writeBinaryPOD(output, max_cate_size_);
        
        

        // new
        writeBinaryPOD(output, attr_type_.size());
        for (auto &it : attr_type_) {
            writeBinaryPOD(output, it);
        }
        writeBinaryPOD(output, attr_pos_.size());
        for (auto &it : attr_pos_) {
            writeBinaryPOD(output, it);
        }
        writeBinaryPOD(output, attr_size_per_item_);
        // std::cout << "attr_size_per_item_:" << attr_size_per_item_ << ", max_elements_:" << max_elements_ << std::endl;

        // ---
        writeBinaryPOD(output, ep_ids_.size());
        for (const auto &id : ep_ids_) {
            writeBinaryPOD(output, id);
        }
        writeBinaryPOD(output, bucket_size_);
        writeBinaryPOD(output, bucket_data_.size());
        output.write((char *)bucket_data_.data(), sizeof(tableint) * bucket_data_.size());
        // std::vector<tableint> bucket_offsets_; 
        writeBinaryPOD(output, bucket_offsets_.size());
        output.write((char *)bucket_offsets_.data(), sizeof(tableint) * bucket_offsets_.size());

        writeBinaryPOD(output, id_to_buckets_.size());
        output.write((char *)id_to_buckets_.data(), sizeof(tableint) * id_to_buckets_.size());

        // int table_size_;
        writeBinaryPOD(output, table_size_);
        // int* counting_hash_table = nullptr; // counting hash table
        output.write((char *)counting_hash_table, sizeof(int) * table_size_ * bucket_size_ * attr_type_.size());
        // std::vector<std::vector<int>> counting_hash_table_mapping;
        writeBinaryPOD(output, counting_hash_table_mapping.size());
        for (const auto &vec : counting_hash_table_mapping) {
            writeBinaryPOD(output, vec.size());
            output.write((char *)vec.data(), sizeof(int) * vec.size());

            // std::cout << " itr pos:" << output.tellp() << ", vec_size:" << vec.size() << std::endl;
        }


        writeBinaryPOD(output, predicate_size_);
        writeBinaryPOD(output, predicate_offset_.size());
        for (auto &it : predicate_offset_) {
            writeBinaryPOD(output, it);
        }
        writeBinaryPOD(output, ft_offset_);
        writeBinaryPOD(output, offsetAttr_);
        
        writeBinaryPOD(output, offsetNbrFt_);
        writeBinaryPOD(output, size_per_ft_);

        // Format version: 4 = edge-level FT, 3 = node-level FT with FT-before-vector layout
        // (2 = node-level FT old layout, 1 = legacy edge-level)
        int ft_format_version = edge_level_ft_ ? 4 : 3;
        writeBinaryPOD(output, ft_format_version);

        //
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);



        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0, bool dynamic = false) {
        #ifdef USE_SSE
            std::cout << "SSE is supported." << std::endl;
        #endif
        #ifdef USE_AVX
            std::cout << "AVX2 is supported." << std::endl;
        #endif
        #ifdef USE_AVX512
            std::cout << "AVX512 is supported." << std::endl;
        #endif



        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, top_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, nbr_size_per_element);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M0_mul_);
        readBinaryPOD(input, minM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        readBinaryPOD(input, ft_bits_);
        ft_bytes_ = (ft_bits_ + 7) / 8;
        // std::cout << "ft_bits_:" << ft_bits_ << ", ft_bytes_:" << ft_bytes_ << std::endl;
        readBinaryPOD(input, cate_int_byte_);
        readBinaryPOD(input, max_cate_size_);




        size_t attr_type_size;
        readBinaryPOD(input, attr_type_size);
        attr_type_.resize(attr_type_size);
        for (int i = 0; i < attr_type_size; i++) {
            readBinaryPOD(input, attr_type_[i]);
        }
        size_t attr_pos_size;
        readBinaryPOD(input, attr_pos_size);
        attr_pos_.resize(attr_pos_size);
        for (int i = 0; i < attr_pos_size; i++) {
            readBinaryPOD(input, attr_pos_[i]);
        }
        readBinaryPOD(input, attr_size_per_item_);
        std::cout << "attr_size_per_item_:" << attr_size_per_item_ << ", max_elements_:" << max_elements_ << std::endl;


        size_t ep_ids_size;
        readBinaryPOD(input, ep_ids_size);
        ep_ids_.resize(ep_ids_size);
        for (size_t i = 0; i < ep_ids_size; i++) {
            readBinaryPOD(input, ep_ids_[i]);
        }
        readBinaryPOD(input, bucket_size_);
        size_t bucket_data_size;
        readBinaryPOD(input, bucket_data_size);
        bucket_data_.resize(bucket_data_size);
        input.read((char *)bucket_data_.data(), sizeof(tableint) * bucket_data_.size());
        size_t bucket_offsets_size;
        readBinaryPOD(input, bucket_offsets_size);
        bucket_offsets_.resize(bucket_offsets_size);
        input.read((char *) bucket_offsets_.data(), sizeof(tableint) * bucket_offsets_.size());

        size_t id_to_buckets_size;
        readBinaryPOD(input, id_to_buckets_size);
        id_to_buckets_.resize(id_to_buckets_size);
        input.read((char *) id_to_buckets_.data(), sizeof(tableint) * id_to_buckets_.size());   
        readBinaryPOD(input, table_size_);
        counting_hash_table = (int *) malloc(sizeof(int) * table_size_ * bucket_size_ * attr_type_.size());
        input.read((char *) counting_hash_table, sizeof(int) * table_size_ * bucket_size_ * attr_type_.size());
        size_t counting_hash_table_mapping_size;
        readBinaryPOD(input, counting_hash_table_mapping_size);
        counting_hash_table_mapping.resize(counting_hash_table_mapping_size);
        std::cout << "counting_hash_table_mapping_size:" << counting_hash_table_mapping_size << std::endl;
        for (size_t i = 0; i < counting_hash_table_mapping_size; i++) {
            size_t vec_size;
            readBinaryPOD(input, vec_size);
            counting_hash_table_mapping[i].resize(vec_size);
            input.read((char *) counting_hash_table_mapping[i].data(), sizeof(int) * vec_size);
            std::cout << "cht " << i << " vec:";
            for (size_t j = 0; j < vec_size; j++) {
                std::cout << counting_hash_table_mapping[i][j] << ", ";
            }
            std::cout << std::endl;
        }   

        readBinaryPOD(input, predicate_size_);
        size_t predicate_offset_size;
        readBinaryPOD(input, predicate_offset_size);
        predicate_offset_.resize(predicate_offset_size);
        for (size_t i = 0; i < predicate_offset_size; i++) {
            readBinaryPOD(input, predicate_offset_[i]);
        }
        readBinaryPOD(input, ft_offset_);
        readBinaryPOD(input, offsetAttr_);

        readBinaryPOD(input, offsetNbrFt_);
        readBinaryPOD(input, size_per_ft_);

        // Read format version: 4 = edge-level FT, 3 = node-level FT-before-vector layout
        // Version 2 = old node-level FT layout, 1 = legacy (no version field)
        int ft_format_version = 0;
        auto pre_version_pos = input.tellg();
        readBinaryPOD(input, ft_format_version);
        if (ft_format_version != 2 && ft_format_version != 3 && ft_format_version != 4) {
            // Old format (version 1): no version field, data follows size_per_ft_ directly
            ft_format_version = 1;
            input.seekg(pre_version_pos);
            std::cout << "Detected legacy index format (version 1, no version field)" << std::endl;
        }
        edge_level_ft_ = (ft_format_version == 4);
        // Version 1 & 2: old layout [link|vector|label|FT|attr]
        // Version 3 & 4: new layout [link|FT|vector|label|attr]
        bool need_layout_migration = (ft_format_version <= 2);



        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);

        // for (size_t i = 0; i < cur_element_count; i++) {
        //     if (input.tellg() < 0 || input.tellg() >= total_filesize) {
        //         // std::cout << "i=" << i << ", tellg=" << input.tellg() << ", total_filesize=" << total_filesize << std::endl;
        //         throw std::runtime_error("Index seems to be corrupted or unsupported");
        //     }

        //     unsigned int linkListSize;
        //     readBinaryPOD(input, linkListSize);
        //     if (linkListSize != 0) {
        //         input.seekg(linkListSize, input.cur);
        //     }
        // }

        // // throw exception if it either corrupted or old index
        // if (input.tellg() != total_filesize)
        //     throw std::runtime_error("Index seems to be corrupted or unsupported by tellg() != total_filesize");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        memset(data_level0_memory_, 0, max_elements * size_data_per_element_);
        // tell pointer position
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        // Compute size_links_level0_ early (needed for migration and integrity check)
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        // Migrate v2 layout [link|vector|label|FT|attr] → v3 [link|FT|vector|label|attr]
        if (need_layout_migration) {
            std::cout << "Migrating index from v2 to v3 layout (FT before vector)..." << std::endl;
            // Old offsets (from file): ft at end, vector right after link list
            size_t old_ft_offset = ft_offset_;         // e.g. 2380
            size_t old_offsetData = offsetData_;        // e.g. 324
            // New offsets: FT right after link list
            size_t new_ft_offset = size_links_level0_;  // e.g. 324
            size_t new_offsetData = size_links_level0_ + size_per_ft_;  // e.g. 356
            size_t new_label_offset = new_offsetData + data_size_;
            size_t vec_label_size = data_size_ + sizeof(labeltype);  // vector + label block

            char* ft_tmp = (char*)malloc(size_per_ft_);
            for (size_t i = 0; i < cur_element_count; i++) {
                char* elem = data_level0_memory_ + i * size_data_per_element_;
                // 1. Save FT from old position
                memcpy(ft_tmp, elem + old_ft_offset, size_per_ft_);
                // 2. Shift vector+label right by size_per_ft_ bytes (overlapping, use memmove)
                memmove(elem + new_offsetData, elem + old_offsetData, vec_label_size);
                // 3. Write FT to new position (right after link list)
                memcpy(elem + new_ft_offset, ft_tmp, size_per_ft_);
            }
            free(ft_tmp);

            // Update offsets to new layout
            ft_offset_ = new_ft_offset;
            offsetNbrFt_ = new_ft_offset;
            offsetData_ = new_offsetData;
            label_offset_ = new_label_offset;
            size_t padding_new = sizeof(int) - ((new_label_offset + sizeof(labeltype)) % sizeof(int));
            if (padding_new == sizeof(int)) padding_new = 0;
            offsetAttr_ = new_label_offset + sizeof(labeltype) + padding_new;
            nbr_size_per_element = size_links_level0_ + size_per_ft_ + data_size_ + sizeof(labeltype);

            std::cout << "Migration complete. New offsets: ft=" << ft_offset_
                      << " data=" << offsetData_ << " label=" << label_offset_
                      << " attr=" << offsetAttr_ << std::endl;
        }

        // in memory check
        for (size_t i = 0; i < cur_element_count; i++) {
            int count = getListCount(get_linklist0(i));
            if (count > maxM0_)
                throw std::runtime_error("Corrupted index file: list size larger than maxM0_");
            else if (count < 0)
                throw std::runtime_error("Corrupted index file: list size negative");
        }




        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        ef_top_ = 1;
        // for (size_t i = 0; i < cur_element_count; i++) {
        //     label_lookup_[getExternalLabel(i)] = i;
        //     unsigned int linkListSize;
        //     readBinaryPOD(input, linkListSize);
        //     if (linkListSize == 0) {
        //         element_levels_[i] = 0;
        //         linkLists_[i] = nullptr;
        //     } else {
        //         element_levels_[i] = linkListSize / size_links_per_element_;
        //         linkLists_[i] = (char *) malloc(linkListSize);
        //         if (linkLists_[i] == nullptr)
        //             throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
        //         input.read(linkLists_[i], linkListSize);
        //     }
        // }

        // for (size_t i = 0; i < cur_element_count; i++) {
        //     if (isMarkedDeleted(i)) {
        //         num_deleted_ += 1;
        //         if (allow_replace_deleted_) deleted_elements.insert(i);
        //     }
        // }

        // read all into buffer
        char *load_buffer;
        std::streamoff rest_pos = input.tellg();
        size_t remain = size_t(total_filesize - rest_pos);
        load_buffer = (char *) malloc(remain);
        if (!load_buffer)
            throw std::runtime_error("Not enough memory for load_buffer");
        input.read(load_buffer, remain);
        if (!input) {
            free(load_buffer);
            throw std::runtime_error("Failed to read index file");
        }
        char *cur_ptr = load_buffer;
        for (size_t i = 0; i < cur_element_count; i++) {
            if (dynamic) {
                label_lookup_[getExternalLabel(i)] = i;
            }
            unsigned int linkListSize;
            memcpy(&linkListSize, cur_ptr, sizeof(unsigned int));
            cur_ptr += sizeof(unsigned int);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                memcpy(linkLists_[i], cur_ptr, linkListSize);
                cur_ptr += linkListSize;
            }
            if (cur_ptr > load_buffer + remain) {
                throw std::runtime_error("Index corrupted: out of range");
            }
        }
        if (cur_ptr != load_buffer + remain) {
            throw std::runtime_error("Index corrupted: size mismatch");
        }
        free(load_buffer);

        if (dynamic) {
            for (size_t i = 0; i < cur_element_count; i++) {
                if (isMarkedDeleted(i)) {
                    num_deleted_ += 1;
                    if (allow_replace_deleted_) deleted_elements.insert(i);
                }
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    /*
     * Repair candidate tracking API.
     * During search, nodes with ≥ repair_dead_ratio_threshold_ fraction of
     * deleted neighbors are recorded (mutable, lock-protected, keeps search const).
     * Caller retrieves and decides: local repairNodes() or global rebuild.
     */
    void setRepairThreshold(double threshold) {
        repair_dead_ratio_threshold_ = threshold;
    }

    std::unordered_map<tableint, float> getRepairCandidates() const {
        std::lock_guard<std::mutex> lock(repair_candidates_lock_);
        return repair_candidates_;
    }

    std::unordered_map<tableint, float> popRepairCandidates() {
        std::lock_guard<std::mutex> lock(repair_candidates_lock_);
        std::unordered_map<tableint, float> result;
        result.swap(repair_candidates_);
        return result;
    }

    void clearRepairCandidates() {
        std::lock_guard<std::mutex> lock(repair_candidates_lock_);
        repair_candidates_.clear();
    }

    size_t repairCandidatesSize() const {
        std::lock_guard<std::mutex> lock(repair_candidates_lock_);
        return repair_candidates_.size();
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false, int level_=-1) {
        throw std::runtime_error("error, please use addPoint with attr_data");
    } 

    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, const std::vector<std::vector<int>>& attr_data, bool replace_deleted = false, int level_=-1) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("In HNSW, Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, attr_data, level_);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, attr_data, level_);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            add_attr_to_point(internal_id_replaced, attr_data);
            update_node_ft(internal_id_replaced);  // rebuild FT for reused slot
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    inline void set_ft_at_pos(unsigned char* ft, int pos) {
        int byte_pos = pos >> 3;   // 等价于 pos / 8
        int bit_pos  = pos & 7;    // 等价于 pos % 8
        assert(byte_pos >= 0 && byte_pos < (int)ft_bytes_);
        // if (byte_pos < 0 || byte_pos >= (int)ft_bytes_) return;
        ft[byte_pos] |= (1 << bit_pos);
    }

    std::string ft_to_string(unsigned char* ft) {
        std::string res = "";
        for (int attr_idx = 0; attr_idx < attr_type_.size(); attr_idx++) {
            res += "ft " + std::to_string(attr_idx) + ": ";
            for (int i = 0; i < ft_bytes_; ++i){
                std::bitset<8> b((unsigned char)ft[i]);
                for (size_t i = 0; i < b.size(); i++) {
                    res += std::to_string(b[i]);  // 注意 bitset[i] 访问的是低位 -> 高位
                }
                res += " ";
            }
            res += "\n";
        }
        return res;
    }

    // generate filter table for all elements after hnsw index generation, not used
    // void generateFT() {
    //     // new version, search for all neighbors and reserve the top ones

    //     std::cout << "before generateFT, pruned_count:" << pruned_count << ", average " << (pruned_count / (float)max_elements_) << std::endl;
    //     std::cout << "dominate_count:" << dominate_count << ", average " << (dominate_count / (float)max_elements_) << std::endl;
    //     std::cout << "heuristic time: " << heuristic_time << " s, ft update time: " << update_ft_time << " s" << std::endl;

    //     // check 9701: 9885
    //     // std::cout << "id 9701's neighbor 9885 ft:" << std::endl;
    //     // unsigned int* data = get_linklist0(9701);
    //     // int size = getListCount(data);
    //     // tableint *datal = (tableint *) (data + 1);
    //     // for (int nbr_idx = 0; nbr_idx < size; nbr_idx++) {
    //     //     if (datal[nbr_idx] == 9885) {
    //     //         unsigned char* ft_9885 = nbr_ft_at(9701, nbr_idx);
    //     //         std::cout << ft_to_string(ft_9885);
    //     //     }
    //     // }
    //     // ---------------------

    //     // #pragma omp parallel for
    //     for (int i = 0; i < max_elements_; ++i) {
    //         unsigned int* data = get_linklist0(i);
    //         int size = getListCount(data);
    //         tableint *datal = (tableint *) (data + 1);
    //         updateft(i, i);
    //         for (int nbr_idx = 0; nbr_idx < size; nbr_idx++) {
    //             updateft(i, datal[nbr_idx]);
    //         }
    //     }

    // }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, const std::vector<std::vector<int>>& attr_data, int level) {
        
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                add_attr_to_point(existingInternalId, attr_data);
                update_node_ft(existingInternalId);  // rebuild node FT after attr change
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);


        add_attr_to_point(cur_c, attr_data);
        // Node-level FT: hash own attributes into FT right after attrs are set
        update_node_ft(cur_c);
        // int curlevel = getRandomLevel(mult_);
        // if (level > 0)
        //     curlevel = level;
        assert(level == 0 || level == 1);
        int curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, nbr_size_per_element);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        // std::cout << "construct nbr ft time: " << update_ft_time << std::endl;

        // if (cur_element_count == max_elements_) {
        //     maybe_print_build_stats();
        // }

        return cur_c;
    }


    // std::priority_queue<std::pair<dist_t, labeltype >>
    // searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
    //     std::priority_queue<std::pair<dist_t, labeltype >> result;
    //     if (cur_element_count == 0) return result;

    //     tableint currObj = enterpoint_node_;
    //     dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    //     for (int level = maxlevel_; level > 0; level--) {
    //         bool changed = true;
    //         while (changed) {
    //             changed = false;
    //             unsigned int *data;

    //             data = (unsigned int *) get_linklist(currObj, level);
    //             int size = getListCount(data);
    //             metric_hops++;
    //             metric_distance_computations+=size;

    //             tableint *datal = (tableint *) (data + 1);
    //             for (int i = 0; i < size; i++) {
    //                 tableint cand = datal[i];
    //                 if (cand < 0 || cand > max_elements_)
    //                     throw std::runtime_error("cand error");
    //                 dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

    //                 if (d < curdist) {
    //                     curdist = d;
    //                     currObj = cand;
    //                     changed = true;
    //                 }
    //             }
    //         }
    //     }

    //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    //     bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    //     if (bare_bone_search) {
    //         top_candidates = searchBaseLayerST<true>(
    //                 currObj, query_data, std::max(ef_, k), isIdAllowed);
    //     } else {
    //         top_candidates = searchBaseLayerST<false>(
    //                 currObj, query_data, std::max(ef_, k), isIdAllowed);
    //     }

    //     while (top_candidates.size() > k) {
    //         top_candidates.pop();
    //     }
    //     while (top_candidates.size() > 0) {
    //         std::pair<dist_t, tableint> rez = top_candidates.top();
    //         result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
    //         top_candidates.pop();
    //     }
    //     return result;
    // }

    // searchKnn with two layers

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        assert(maxlevel_ == 1); // for two layers only
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_layer_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        top_layer_candidates = searchTopLayerST<true, true>(currObj, query_data, ef_top_);
        bool bare_bone_search = !num_deleted_;
        if (bare_bone_search) {
                top_candidates = searchBaseLayerST<true, true>(
                    top_layer_candidates, query_data, std::max(ef_, k));
        } else {
                top_candidates = searchBaseLayerST<false, true>(
                    top_layer_candidates, query_data, std::max(ef_, k));
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    int get_cnt_via_CHT(std::vector<char> ft_predicate, int bucket_id, int min_attr1, int min_attr2, int attr_type) const {
        // estimate the selectivity of a bucket by counting hash table
        // attr_idx: attribute index
        int* cht = counting_hash_table_at(bucket_id, min_attr1);

        if (attr_type == 0) {
            int sel_count = 0;
            int byte_pos = 0;
            int bit_pos = 0;
            int overall_pos = 0;
            while(overall_pos < ft_bits_) {
                byte_pos = overall_pos >> 3;   // 等价于 pos / 8
                bit_pos  = overall_pos & 7;    // 等价于 pos % 8
                if (ft_predicate[min_attr1 * ft_bytes_ + byte_pos] & (1 << bit_pos)) {
                    sel_count += cht[overall_pos];
                }
                overall_pos++;
            }
            return sel_count;
        }
        else {
            // one label
            return cht[min_attr2];
        }
    }

    // construct bucket from constructed bottom layer graph, not clustering
    // each node is assigned to at most two buckets
    void graph_partition(){
        std::vector<std::vector<tableint>> temp_buckets;
        std::vector<int> visited_queue;
        temp_buckets.resize(ep_ids_.size());
        visited_queue.resize(ep_ids_.size(), 0);
        std::vector<uint8_t> visited(cur_element_count, 0);

        for (tableint i = 0; i < ep_ids_.size(); i++) {
            tableint ep_id = ep_ids_[i];
            temp_buckets[i].push_back(ep_id);
            visited[ep_id] = 1;
        }
        while (true) {
            bool queue_empty = true;
            for (tableint i = 0; i < ep_ids_.size(); i++) {
                if(visited_queue[i] >= temp_buckets[i].size()) {
                    continue;
                }
                queue_empty = false;
                tableint cur_id = temp_buckets[i][visited_queue[i]];
                visited_queue[i]++;
                unsigned int *data = get_linklist0(cur_id);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for (int j = 0; j < size; j++) {
                    tableint nbr_id = datal[j];
                    if (visited[nbr_id] <= 1) {
                        temp_buckets[i].push_back(nbr_id);
                        visited[nbr_id]++;
                    }
                }
            }
            if (queue_empty) {
                break;
            }
        }

        // flatten to bucket storage
        int bucket_data_cnt = 0;
        bucket_data_.clear();
        bucket_offsets_.clear();
        bucket_offsets_.push_back(0);
        for (tableint i = 0; i < temp_buckets.size(); i++) {
            bucket_data_cnt += temp_buckets[i].size();
        }
        for (tableint i = 0; i < temp_buckets.size(); i++) {
            bucket_data_.insert(bucket_data_.end(), temp_buckets[i].begin(), temp_buckets[i].end());
            bucket_offsets_.push_back(bucket_data_.size());
        }
    }


    // hybrid searchKnn with two layers
    // predicate format:
    //    predicate[i]: filtering restriction for attribute i
    //      numerical:
    //        predicate[i][0] <= value <= predicate[i][1]
    //      categorical:
    //        attr[item] contains all the predicate[i][:]
    //    
    std::priority_queue<std::pair<dist_t, labeltype >>
    hybridSearch(const void *query_data, std::vector<std::vector<int>> raw_predicate, size_t k) const {
        
        std::vector<int> predicate = predicate_translate(raw_predicate);
        std::vector<char> ft_predicate = predicate_to_ft(raw_predicate);

        // estimate selectivity for all attributes
        std::vector<std::vector<double>> selectivities;
        selectivities.resize(predicate.size());
        double min_sel = 1.0;
        int min_attr1 = -1;
        int min_attr2 = -1;
        assert(predicate.size() <= attr_type_.size());
        int predicate_cnt = 0;
        
        // std::vector<std::vector<tableint>> multi_attr_bucket_candidates; // inverted file index for categorical attribute
        // multi_attr_bucket_candidates.resize(bucket_size_);
        // for (int i = 0; i < raw_predicate.size(); i++) { // iterate all predicate to check the selectivity
        //     if (raw_predicate[i].size() == 0) {
        //         // no filtering condition on this attribute
        //         selectivities[i].push_back(1.0);
        //         continue;
        //     }
        //     if (attr_type_[i] == 0) { // numerical, check btree
        //         predicate_cnt++;
        //         auto left = btrees[i].lower_bound({raw_predicate[i][0], 0});
        //         auto right = btrees[i].lower_bound({raw_predicate[i][1], max_elements_});
        //         right--;
        //         double sel = static_cast<double>(right->second - left->second) / max_elements_;        
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "left attr:" << raw_predicate[i][0] << ", right attr:" << raw_predicate[i][1] << std::endl;
        //         std::cout << "left idx:" << left->second << ", right idx:" << right->second << std::endl;
        //         std::cout << "predicate " << i << " sel:" << sel << std::endl;
        //         #endif
        //         selectivities[i].push_back(sel);
        //         if (sel < min_sel) {
        //             min_sel = sel;
        //             min_attr1 = i;
        //             min_attr2 = 0;
        //         }
        //     }
        //     else { // categorical, check ivf
        //         for (int j = 0; j < raw_predicate[i].size(); j++) {
        //             // each label performs "and" operation, so we static all labels separately
        //             int num = ivf[i][raw_predicate[i][j]].size();
        //             double sel = static_cast<double>(num) / max_elements_;
        //             selectivities[i].push_back(sel);
        //             predicate_cnt++;

        //             if (sel < min_sel) {
        //                 min_sel = sel;
        //                 min_attr1 = i;
        //                 min_attr2 = j;
        //             }
        //         }
        //     }
        // }

        #ifdef DEBUG_SEARCH_WORKFLOW
        std::cout << "predicate size:" << predicate_cnt << std::endl;
        std::cout << "min sel:" << min_sel << ", attr idx:" << min_attr1 << ", attr type:" << attr_type_[min_attr1] << " attr idx2: " << min_attr2 << std::endl;
        #endif

        // double final_sel = 1.0;
        // if (predicate_cnt > 1) {
        //     // multi-attribute

        //     // compute approximate hybrid selectivity, assume to be uniformly distributed and independent
        //     double hybrid_sel = 1.0;
        //     for (int i = 0; i < selectivities.size(); i++) {
        //         for (int j = 0; j < selectivities[i].size(); j++) {
        //             hybrid_sel *= selectivities[i][j];
        //         }
        //     }
    
        //     #ifdef DEBUG_SEARCH_WORKFLOW
        //     std::cout << "hybrid sel:" << hybrid_sel << std::endl;
        //     #endif
        //     if (hybrid_sel < threshold_3_) {
        //         // scan on subset with smallest sel
        //         int valid_cnt = 0;
        //         if (attr_type_[min_attr1] == 0) { // numerical, scan on btree
                    
        //             #ifdef DEBUG_SEARCH_WORKFLOW
        //             std::cout << "smaller than threshold 3, scan all valid attributes from btree" << std::endl;
        //             #endif
        //             auto left = btrees[min_attr1].lower_bound({raw_predicate[min_attr1][0], 0});
        //             auto right = btrees[min_attr1].lower_bound({raw_predicate[min_attr1][1], max_elements_});
        //             for (auto iter = left; iter != right; iter++) {
        //                 tableint id = iter->first.id;
        //                 if (predicate_check(id, predicate)) {
        //                     // candidate_set.push_back(id);
        //                     valid_cnt++;
        //                     multi_attr_bucket_candidates[id_to_buckets_[id]].push_back(id);
        //                 }
        //             }
        //         }
        //         else { // categorical, scan on ivf bucket
        //             #ifdef DEBUG_SEARCH_WORKFLOW
        //             std::cout << "smaller than threshold 3, collect all valid vectors from ivf" << std::endl;
        //             #endif
        //             int label = raw_predicate[min_attr1][min_attr2];
        //             for (auto id : ivf[min_attr1][label]) {
        //                 if (predicate_check(id, predicate)) {
        //                     // candidate_set.push_back(id);
        //                     valid_cnt++;
        //                     multi_attr_bucket_candidates[id_to_buckets_[id]].push_back(id);
        //                 }
        //             }
        //         }
        //         final_sel = valid_cnt * 1.0 / (double)max_elements_;

        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "final sel for multi-attribute: " << final_sel << std::endl;
        //         #endif
        //     }
        // }
        // else {
        //     final_sel = min_sel;
        //     #ifdef DEBUG_SEARCH_WORKFLOW
        //     std::cout << "final sel for one-attribute: " << final_sel << std::endl;
        //     #endif
        // }

        // smaller than threshold 1, scan on candidate set or btree/ivf
        // std::vector<tableint> candidate_set; // candidate set after filtering with all predicate
        // candidate_set.reserve(max_elements_ * final_sel + 100); // pre-allocate memory
        // if (final_sel < threshold_1_) {
        //     #ifdef DEBUG_SEARCH_WORKFLOW
        //     std::cout << "final sel smaller than threshold 1" << std::endl;
        //     #endif
        //     if (predicate_cnt == 1) {
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "one predicate, scan on btree/ivf" << std::endl;
        //         #endif

        //         int btree_cnt = 0;
        //         if (attr_type_[min_attr1] == 0) { // numerical, scan on btree
        //             auto left = btrees[min_attr1].lower_bound({raw_predicate[min_attr1][0], 0});
        //             auto right = btrees[min_attr1].lower_bound({raw_predicate[min_attr1][1], max_elements_});

        //             for (auto iter = left; iter != right; iter++) {
        //                 tableint id = iter->first.id;
        //                 btree_cnt++;
        //                 candidate_set.push_back(id);
        //             }
        //         }
        //         else { // categorical, scan on ivf bucket
        //             int label = raw_predicate[min_attr1][min_attr2];
        //             for (auto id : ivf[min_attr1][label]) {
        //                 candidate_set.push_back(id);
        //                 btree_cnt++;
        //             }
        //         }
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "btree scanned cnt: " << btree_cnt << std::endl;
        //         #endif
        //     }
        //     else{
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "multi predicate, collect multi_attr_bucket_candidates" << std::endl;
        //         #endif
        //         for (auto& bucket : multi_attr_bucket_candidates) {
        //             for (auto id : bucket) {
        //                 candidate_set.push_back(id);
        //             }
        //         }
        //     }

        //     // scan on candidate set
        //     #ifdef DEBUG_SEARCH_WORKFLOW
        //     std::cout << "scan candidate set, size:" << candidate_set.size() << std::endl;
        //     #endif
        //     std::priority_queue<std::pair<dist_t, labeltype >> result;

        //     #ifdef USE_SSE
        //         if (candidate_set.size() > 0){
        //             _mm_prefetch(getDataByInternalId(candidate_set[0]), _MM_HINT_T0);
        //         }
        //     #endif
            
            
        //     for (int i = 0; i < candidate_set.size(); i++) {
        //         tableint id = candidate_set[i];
        //         #ifdef USE_SSE
        //             if (i + 1 < candidate_set.size()){
        //                 _mm_prefetch(getDataByInternalId(candidate_set[i + 1]), _MM_HINT_T0);
        //             }
        //         #endif
        //         dist_t distance = fstdistfunc_(query_data, getDataByInternalId(id), dist_func_param_);

        //         if (result.size() < k) {
        //             result.emplace(distance, getExternalLabel(id));
        //         } else {
        //             if (distance < result.top().first) {
        //                 result.pop();
        //                 result.emplace(distance, getExternalLabel(id));
        //             }
        //         }
        //     }
        //     return result;
        // }

        // search on top layer
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);


        assert(maxlevel_ == 1); // for two layers only
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_layer_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        top_layer_candidates = searchTopLayerST<true, true>(currObj, query_data, ef_top_ * 2);
        
        // use ep as bottom layer ep
        // top_layer_candidates.emplace(curdist, currObj);
        // for (int i = 0; i < ef_top_; i++) {
        //     dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        //     top_layer_candidates.emplace(curdist, i);
        // }

        // std::vector<tableint> top_vector = copy_to_vector(top_layer_candidates);

        // search on base layer
        // bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        bool bare_bone_search = true;
        
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        if (bare_bone_search) {
            if (use_ft_routing_ && use_ft_)
            top_candidates = hybridSearchBaseLayerST<true, true, true, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else if (use_ft_routing_ && !use_ft_)
            top_candidates = hybridSearchBaseLayerST<true, true, true, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else if (!use_ft_routing_ && use_ft_)
            top_candidates = hybridSearchBaseLayerST<true, true, false, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else
            top_candidates = hybridSearchBaseLayerST<true, true, false, false>(
                top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
        } else {
            if (use_ft_routing_ && use_ft_)
            top_candidates = hybridSearchBaseLayerST<false, true, true, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else if (use_ft_routing_ && !use_ft_)
            top_candidates = hybridSearchBaseLayerST<false, true, true, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else if (!use_ft_routing_ && use_ft_)
            top_candidates = hybridSearchBaseLayerST<false, true, false, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
            else
            top_candidates = hybridSearchBaseLayerST<false, true, false, false>(
                top_layer_candidates, query_data, predicate, ft_predicate, ef_, k, vl);
        }
        // for multi-attribute, CHT is unnecessary since we have scanned on btree with smallest sel

        // top vector id to bucket id
        // bool unsearched = false;
        // if (top_candidates.size() < k) {
        //     unsearched = true;
        // }

        // double overall_sel = 0.0;
        // for (int i = 0; i < top_vector.size(); i++) {
        //     int bucket_id = id_to_buckets_[top_vector[i]];
        //     #ifdef DEBUG_SEARCH_WORKFLOW
        //     std::cout << "checking for ep " << i << ", bucket " << bucket_id << std::endl;
        //     #endif
        //     if (predicate_cnt > 1) {
                
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "multi attr, checking sel for each selected bucket" << std::endl;
        //         #endif
        //         // check candidates filtered when scanning on btree with smallest sel
        //         if (multi_attr_bucket_candidates[bucket_id].size() > 0) {
        //             int this_bucket_size = bucket_offsets_[bucket_id+1] - bucket_offsets_[bucket_id];
        //             double sel = multi_attr_bucket_candidates[bucket_id].size() * 1.0 / (double)this_bucket_size;
        //             #ifdef DEBUG_SEARCH_WORKFLOW
        //             std::cout << "bucket " << bucket_id << " sel " << sel << std::endl;
        //             #endif
        //             overall_sel += sel;
        //         }
        //         else {
        //             overall_sel = 1; // large sel, no need to estimate
        //         }
        //     }
        //     else {
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "single attr, checking local sel" << std::endl;
        //         #endif
        //         // one predicate, estimate local selectivity with CHT
        //         int this_bucket_size = bucket_offsets_[bucket_id+1] - bucket_offsets_[bucket_id];

        //         // cht size
        //         int cht_size = get_cnt_via_CHT(ft_predicate, bucket_id, min_attr1, min_attr2, attr_type_[min_attr1]);
        //         double sel = cht_size * 1.0 / (double)this_bucket_size;

        //         overall_sel += sel;
        //     }
        // }
        // overall_sel /= top_vector.size();
        
        // if (overall_sel < threshold_2_) {
        //     unsearched = true;
        // }

        // if (unsearched) {
        //     for (int i = 0; i < top_vector.size(); i++) {
        //         int bucket_id = id_to_buckets_[top_vector[i]];
        //         #ifdef DEBUG_SEARCH_WORKFLOW
        //         std::cout << "checking for ep " << i << ", bucket " << bucket_id << std::endl;
        //         #endif
        //         if (predicate_cnt > 1) {
        //             // scan on this bucket
        //             #ifdef USE_SSE
        //                 auto &cands = multi_attr_bucket_candidates[bucket_id];
        //                 int prefetch_n = std::min<int>(3, cands.size());
        //                 for (int i = 0; i < prefetch_n; i++) {
        //                     int id = cands[i];
        //                     _mm_prefetch((char*)(visited_array + id), _MM_HINT_T0);
        //                     _mm_prefetch(getDataByInternalId(id), _MM_HINT_T0);
        //                 }
        //             #endif

        //             for (int idx = 0; idx < multi_attr_bucket_candidates[bucket_id].size(); idx++) {
        //                 // skip if visited
        //                 int id = multi_attr_bucket_candidates[bucket_id][idx];

        //                 #ifdef USE_SSE
        //                     if (idx + 3 < multi_attr_bucket_candidates[bucket_id].size()) {
        //                         int next_id = multi_attr_bucket_candidates[bucket_id][idx + 3];
        //                         _mm_prefetch((char *) (visited_array + next_id), _MM_HINT_T0);
        //                         _mm_prefetch(getDataByInternalId(next_id), _MM_HINT_T0);
        //                     }
        //                 #endif

        //                 if (visited_array[id] == visited_array_tag) continue;
        //                 visited_array[id] = visited_array_tag;

        //                 dist_t distance = fstdistfunc_(query_data, getDataByInternalId(id), dist_func_param_);
        //                 if (top_candidates.size() < k) {
        //                     top_candidates.emplace(distance, id);
        //                 } else {
        //                     if (distance < top_candidates.top().first) {
        //                         top_candidates.pop();
        //                         top_candidates.emplace(distance, id);
        //                     }
        //                 }
        //             }
        //         }
        //         else {
        //             // scan on this bucket
        //             int start = bucket_offsets_[bucket_id];
        //             int end = bucket_offsets_[bucket_id+1];

        //             #ifdef USE_SSE
        //                 _mm_prefetch((char *) (visited_array + bucket_data_[start]), _MM_HINT_T0);
        //                 _mm_prefetch((char *) (visited_array + bucket_data_[start + 1]), _MM_HINT_T0);
        //                 _mm_prefetch(getDataByInternalId(bucket_data_[start]), _MM_HINT_T0);
        //                 _mm_prefetch(getDataByInternalId(bucket_data_[start + 1]), _MM_HINT_T0);
        //             #endif

        //             for (int idx = start; idx < end; idx++) {
        //                 tableint id = bucket_data_[idx];
        //                 // skip if visited
        //                 if (visited_array[id] == visited_array_tag) continue;
        //                 visited_array[id] = visited_array_tag;
        //                 if (!predicate_check(id, predicate)) continue;

        //                 dist_t distance = fstdistfunc_(query_data, getDataByInternalId(id), dist_func_param_);
        //                 if (top_candidates.size() < k) {
        //                     top_candidates.emplace(distance, id);
        //                 } else {
        //                     if (distance < top_candidates.top().first) {
        //                         top_candidates.pop();
        //                         top_candidates.emplace(distance, id);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }


        visited_list_pool_->releaseVisitedList(vl);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    // ====================================================================
    // hybridSearchDNF: entry point for DNF predicate queries
    //
    // dnf_raw[term][attr] = values (same format as build_dnf_predicate)
    // modes[term][attr] = check mode override (optional)
    // ====================================================================
    std::priority_queue<std::pair<dist_t, labeltype >>
    hybridSearchDNF(
        const void *query_data,
        const std::vector<std::vector<std::vector<int>>>& dnf_raw,
        const std::vector<std::vector<int8_t>>& modes,
        size_t k) const {

        DNFPredicate dnf_pred = build_dnf_predicate(dnf_raw, modes);
        // (they won't be used when dnf_pred is provided)
        std::vector<int> predicate(predicate_size_, 0);
        std::vector<char> ft_predicate(size_per_ft_, 0);

        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        assert(maxlevel_ == 1);
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_layer_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        top_layer_candidates = searchTopLayerST<true, true>(currObj, query_data, ef_top_ * 2);

        bool bare_bone_search = true;
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();

        if (bare_bone_search) {
            if (use_ft_routing_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, true, true, true>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else if (use_ft_routing_ && !use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, true, true, false>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else if (!use_ft_routing_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, true, false, true>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else
                top_candidates = hybridSearchBaseLayerST<true, true, false, false>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
        } else {
            if (use_ft_routing_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, true, true, true>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else if (use_ft_routing_ && !use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, true, true, false>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else if (!use_ft_routing_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, true, false, true>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
            else
                top_candidates = hybridSearchBaseLayerST<false, true, false, false>(
                    top_layer_candidates, query_data, predicate, ft_predicate,
                    ef_, k, vl, nullptr, &dnf_pred);
        }

        visited_list_pool_->releaseVisitedList(vl);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false, true>(currObj, query_data, 0, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
