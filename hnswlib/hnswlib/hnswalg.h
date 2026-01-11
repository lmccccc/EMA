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

    double two_hop_threshold{0.2}; // threshold for two hop expansion

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

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

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

    bool use_two_hop_{true};

    void set_ft_flag(bool flag){
        use_ft_ = flag;
    }

    void set_two_hop_flag(bool flag){
        use_two_hop_ = flag;
        std::cout << "set two hop flag to " << use_two_hop_ << std::endl;
    }

    void add_ep_ids(const std::vector<tableint>& ep_ids){
        ep_ids_ = ep_ids;
    }

    void set_thresholds(double threshold_1, double threshold_2, double threshold_3){
        threshold_1_ = threshold_1;
        threshold_2_ = threshold_2;
        threshold_3_ = threshold_3;
    }

    void set_two_hop_threshold(double threshold){
        two_hop_threshold = threshold;
        std::cout << "set two hop threshold to " << two_hop_threshold << std::endl;
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
            for (int j = 1; j <= _attr.size(); ++j){
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
        for (int i = 0; i < M; i++) {
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
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
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
        bool allow_replace_deleted = false)
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

        // per element: links(maxM0*4 + 4) + data_size_ + label(4) + filter table * attr size + padding + attr(attr_size * 4)
        size_links_level0_ = maxM0_ * sizeof(tableint)+ sizeof(linklistsizeint);
        size_per_ft_ = ft_bytes_ * attr_type_.size();

        // attr is int*, padding the start position to 4 byte alignment
        padding = sizeof(int) - (size_links_level0_ % sizeof(int));
        if (padding == sizeof(int)) padding = 0;
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype) + size_per_ft_* maxM0_ + padding + attr_size_per_item_ * sizeof(int);
        nbr_size_per_element = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        ft_offset_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetNbrFt_ = ft_offset_;
        offsetAttr_ = size_links_level0_ + data_size_ + sizeof(labeltype) + size_per_ft_ * maxM0_ + padding;
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

    
    
    inline unsigned char* nbr_ft_at(tableint internal_id, int nbr_idx, int attr_idx=0) const {
        // int* size = get_linklist0(internal_id);
        // assert(nbr_idx < getListCount((linklistsizeint*)size));
        return (unsigned char *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetNbrFt_ + nbr_idx * size_per_ft_ + attr_idx * ft_bytes_);
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
//             (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
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
//                             (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
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
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        // if (bare_bone_search || 
        //     (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
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
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
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

    inline bool predicate_check(tableint id, std::vector<int> predicate) const {
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


#ifdef USE_SSE
//     inline bool filter_table_check(tableint id, char* mapped_predicate) const {
//         for(int i = 0; i < attr_type_.size(); ++i) {
//             char* ft = ft_at(id, i);

//             if (attr_type_[i] == 0) { // numerical
//                 bool matched = false;
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


    inline bool nbr_ft_check(tableint id, int nbr_idx, char* mapped_predicate) const {
        for(int i = 0; i < attr_type_.size(); ++i){
            unsigned char* ft = nbr_ft_at(id, nbr_idx, i);
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
                        break;
                    }
                }

                // tail
                if (!matched) {
                    for (; offset < ft_bytes_; ++offset) {
                        if ((ft[offset] & mapped_predicate[i * ft_bytes_ + offset]) > 0) {
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
                    __m128i v_mp = _mm_loadu_si128((const __m128i*)(mapped_predicate + i * ft_bytes_ + offset));
                    __m128i r    = _mm_and_si128(v_ft, v_mp);

                    __m128i cmp = _mm_cmpeq_epi8(r, v_mp);
                    __m128i not_cmp = _mm_xor_si128(cmp, _mm_set1_epi8((char)0xFF));
                    if (!_mm_testz_si128(not_cmp, not_cmp)) return false;
                }

                for (; offset < ft_bytes_; ++offset) {
                    if ((ft[offset] & mapped_predicate[i * ft_bytes_ + offset]) != mapped_predicate[i * ft_bytes_ + offset])
                        return false;
                }
            }
            #else
            bool matched = false;
            if (attr_type_[i] == 0) { // numerical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) > 0) { // any attr exist for range search
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    // if (id == 9701 && datal[nbr_idx] == 9885)
                    //     std::cout << "num attr not match" << std::endl;
                    return false;
                }
            }
            else if (attr_type_[i] == 1) { // categorical
                for (int j = 0; j < ft_bytes_; ++j) {
                    if ((ft[j] & mapped_predicate[i * ft_bytes_ + j]) != mapped_predicate[i * ft_bytes_ + j]) { // all attr exist for label search
                        // std::cout << "cate attr not match" << std::endl;
                        return false;
                    }
                }
            }
            #endif
        }
        return true;
    }

    
    inline void batched_nbr_ft_check_sse(const tableint& id, const int& size, char* mapped_predicate, std::vector<uint8_t>& res) const {
        
        #ifdef USE_SSE
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

    inline bool nbr_ft_check_multi_or(tableint id, int nbr_idx, char* mapped_predicate) const {
        assert(attr_type_.size() > 1); // only one attribute is supported in multi-or query
        for(int i = 0; i < attr_type_.size(); ++i){
            unsigned char* ft = nbr_ft_at(id, nbr_idx, i);
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
//         int use_two_hop_freq = 0;
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
//             int min_nbrs = two_hop_threshold * size;
//             // std::cout << "size:" << size << " ft nbrs size:" << nbrs.size() << " expected min nbrs:" << min_nbrs << std::endl;
            
//             if (use_two_hop_ && nbrs.size() < min_nbrs) {
//                 use_two_hop_freq++;
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
//                             (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
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
//         //              << ", use two hop freq " << use_two_hop_freq
//         //              << std::endl;

//         return top_candidates;
//     }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false, bool use_two_hop = true, bool use_ft=true>
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
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
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
            if (predicate_check(id, predicate)) {
                top_candidates.emplace(top_candidates_.top());
                // std::cout << " in top candidates" << std::endl;
            }
            if (candidate_set.size() >= ef_top_) {
                // prevent candidate set from being too large
                break;
            }
            top_candidates_.pop();
        }


        // if (!top_candidates.empty())
        //     lowerBound = top_candidates.top().first;
        // else
            lowerBound = std::numeric_limits<dist_t>::max();
        int round = 0;
        int passed = 0;
        int ft_passed = 0;
        int deg = 0;
        int max_deg = 0;
        int ft_deg = 0;
        int use_two_hop_freq = 0;
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
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;

            #ifdef USE_SSE
            char* ft0 =
                data_level0_memory_
                + current_node_id * size_data_per_element_
                + offsetNbrFt_;
            _mm_prefetch((char*)ft0, _MM_HINT_T0);
            #endif
            #ifdef DEBUG_SEARCH
            std::cout << "round " << round << " checking node " << current_node_id << " candidate dist " << candidate_dist << " lower bound " << lowerBound << std::endl;
            #endif

            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
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
            
            if (use_ft) {
                std::vector<uint8_t> ft_predicate_result(size, 1);
                // batched_nbr_ft_check_sse(current_node_id, size, ft_predicate.data(), ft_predicate_result);
                for (size_t j = 1; j <= size; j++) {
                    ft_predicate_result[j-1] = nbr_ft_check(current_node_id, j-1, ft_predicate.data());
                }
                for (size_t j = 1; j <= size; j++) {
                    if (ft_predicate_result[j-1]) {
                        nbrs.push_back(*(data + j)); // indicate filtered nbr
                        // prefetch
                        #ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);  // L1 prefetch
                            // _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_, _MM_HINT_T1); // L2 prefetch
                            _mm_prefetch((char *) (visited_array + (*(data + j))), _MM_HINT_T0);  // L1 prefetch
                            // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
                        #endif
                    }
                    else {
                        not_nbrs.push_back(*(data + j));
                        not_nbrs_idx.push_back(j);
                        // #ifdef USE_SSE
                        //     // if (not_nbrs.size() > 5 ) continue;
                        //     _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_, _MM_HINT_T0); // L1 prefetch
                        //     // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
                        //     char* ft0 =
                        //         data_level0_memory_
                        //         + *(data + j) * size_data_per_element_
                        //         + offsetNbrFt_;
                        //     _mm_prefetch((char*)ft0, _MM_HINT_T0);
                        // #endif
                    }
                }
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
            }
            ft_deg += nbrs.size();
            ft_passed += (size - nbrs.size());

            // double passed_ratio = double(size - nbrs.size()) / size;
            int min_nbrs = two_hop_threshold * size;
            // std::cout << "size:" << size << " ft nbrs size:" << nbrs.size() << " expected min nbrs:" << min_nbrs << std::endl;
            // std::vector<tableint> two_hop_candidates(256);
            if (use_two_hop) {
                if (nbrs.size() < min_nbrs) {
                    size_t extra_nbrs_needed = 10;
                    for(int j = 0; j < std::min(not_nbrs.size(), extra_nbrs_needed); ++j){
                        nbrs.push_back(not_nbrs[j]);
                        _mm_prefetch(data_level0_memory_ + (not_nbrs[j]) * size_data_per_element_ + offsetData_, _MM_HINT_T0);  // L1 prefetch
                        // _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_, _MM_HINT_T0); // L2 prefetch
                        _mm_prefetch((char *) (visited_array + (not_nbrs[j])), _MM_HINT_T0);  // L1 prefetch
                        // _mm_prefetch((char *) nbrs[1], _MM_HINT_T0);
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
                    // use_two_hop_freq++;
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



            for (int nbr_idx = 0; nbr_idx < nbrs.size(); ++nbr_idx) {
                int candidate_id = nbrs[nbr_idx];
                // size_t j_next = nbr_idx + 1 < nbrs.size() ? nbrs[nbr_idx + 1] : size + 1;
                
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + j_next), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (j_next) * size_data_per_element_ + offsetData_,
//                                 _MM_HINT_T0);  ////////////
                if (nbr_idx + 8 < nbrs.size()) {
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

                        if (!predicate_check(candidate_id, predicate)) {
                            passed++;
                            #ifdef DEBUG_SEARCH
                            std::cout << " predicate passed" << std::endl;
                            #endif
                            continue;
                        }
                        

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            // std::cout << "pushed to top candidates" << std::endl;
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

                        // if (!top_candidates.empty()) 
                        //     lowerBound = top_candidates.top().first;


                        if (top_candidates.size() >= search_k) 
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
        //              << ", use two hop freq " << use_two_hop_freq
        //              << std::endl;
        // std::cout << " two hop collection time (ms): " << two_hop_collection_time << " (" << coll_pct << "%)" << std::endl;
        // std::cout << " two hop scan time (ms): " << two_hop_scan_time << " (" << scan_pct << "%)" << std::endl;
        // std::cout << " visited array time (ms): " << visited_array_time << " (" << visit_pct << "%)" << std::endl;
        // std::cout << " linklist fetch time (ms): " << linklist_fetch_time << " (" << linklist_pct << "%)" << std::endl;
        // std::cout << " nbr ft check time (ms): " << nbr_ft_time << " (" << nbr_ft_pct << "%)" << std::endl;
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
                             size_t& search_k) const {
        for (size_t idx = 0; idx < two_hop_candidates.size(); idx++) {
            tableint candidate_id = two_hop_candidates[idx];
            char *currObj1 = (getDataByInternalId(candidate_id));
            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
            bool flag_consider_candidate;
            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
            if (flag_consider_candidate) {
                candidate_set.emplace(-dist, candidate_id);
                if (!predicate_check(candidate_id, predicate)) {
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
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
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
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
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
    // #define DEBUG_BUILD

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

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        // modify stop condition: stop when new candidate is good
        std::vector<int> temp_cht;
        if (need_record) temp_cht.resize(table_size_ * attr_type_.size(), 0);

        while (queue_closest.size()) {
            // if (return_list.size() >= M)
            //     break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            
            for (int i = 0; i < return_list.size(); i++) {
                std::pair<dist_t, tableint> second_pair = return_list[i];
            // for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;

                    if (need_record) {
                        // add attr to ft
                        tableint nbr_id = curent_pair.second;
                        (*dominated_list)[i].push_back(nbr_id); // nbr_id is dominated by return_list[i]
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

    
    double update_ft_time{0.0};

    void update_nbr_ft(tableint id, std::vector<tableint>& selectedNeighbors, std::vector<std::vector<tableint>>& dominated_list) {
        // std::cout << "updating filter tables for " << selectedNeighbors.size() << " neighbors for id " << id << std::endl;
        

        // start time
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<unsigned char>> new_fts; // filter tables of selected neighbors
        new_fts.resize(selectedNeighbors.size(), std::vector<unsigned char>(size_per_ft_, 0));
        linklistsizeint *cur_nbrs = get_linklist0(id);               // nbr info of current point

        // initialize new ft.  With old ft if the neighbor was an old neighbor
        unsigned short int ori_nbr_size = getListCount(cur_nbrs);    // >0 if not first insert
        // std::cout << "original nbr size: " << ori_nbr_size << std::endl;
        tableint *nbr = (tableint *)(cur_nbrs + 1);

        // copy old ft to new ft
        for (size_t i = 0; i < selectedNeighbors.size(); i++) {         // for each new neighbor
            bool found = false;
            for (size_t j = 0; j < ori_nbr_size; j++) {             // for each old neighbor
                if (selectedNeighbors[i] == nbr[j]) {                 // new neighbor i was an old neighbor
                    memcpy(new_fts[i].data(), nbr_ft_at(id, j), size_per_ft_); // merge old ft to new ft
                    found = true;
                    break;
                }
            }
            if (!found) {
                // update attr into new ft
                updateft(new_fts[i].data(), selectedNeighbors[i]);
            }
        }

        // merge dominated points's ft to new ft
        for (size_t i = 0; i < selectedNeighbors.size(); i++) {         // for each new neighbor
            tableint cur_id = selectedNeighbors[i];                     // id of new neighbor i
            
            // std::cout << "dominated list size: " << dominated_list.size() << std::endl;
            // if (selectedNeighbors.size() < maxM0_) continue;
            // std::cout << " neighbor " << cur_id << " dominates " << dominated_list[i].size() << " points" << std::endl;

            for (int dominated_idx = 0; dominated_idx < dominated_list[i].size(); dominated_idx++) {
                tableint dominated_id = dominated_list[i][dominated_idx];
                // 9701: 9885
                // if (id == 9701 && cur_id == 9885) {
                //     std::cout << "9701's neighbor 9885 dominates " << dominated_id << std::endl;
                // }


                // if dominated id was a neighbor, then inherit its filter table
                unsigned char* old_ft = nullptr; 
                // if(ori_nbr_size > 0) {                        // neighbor size > 0, not first insert
                    for (size_t j = 0; j < ori_nbr_size; j++) {// for each neighbor
                        if (nbr[j] == dominated_id) {                    // dominated_id was a neighbor
                            old_ft = nbr_ft_at(id, j);        // get its filter table
                            break;
                        }
                    }
                    if (old_ft) {                                        // add old ft to new ft, if found
                        merge_ft(new_fts[i].data(), old_ft);
                    }
                    else{
                        // add dominated id's attr to new_ft, if not found
                        updateft(new_fts[i].data(), dominated_id);
                    }
                // }
            }
        }

        // add attr of itself to ft
        for (size_t i = 0; i < selectedNeighbors.size(); i++) {
            updateft(new_fts[i].data(), selectedNeighbors[i]);
        }
        
        // cover ft by new_ft
        for (size_t i = 0; i < selectedNeighbors.size(); i++) {         // for each new neighbor
            unsigned char* ft = nbr_ft_at(id, i);
            memcpy(ft, new_fts[i].data(), size_per_ft_);
        }

        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        update_ft_time += diff.count();
        // std::cout << "ft update done" << std::endl;
        return;
    }

    int dominate_count{0};


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
            dominate_count += dominated_list[i].size();
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
        
        // merge attr of dominated points to filter table of neighbors who domianate them
        // std::vector<std::vector<char>> new_fts; // filter tables of selected neighbors
        if (level == 0) {
            update_nbr_ft(cur_c, selectedNeighbors, dominated_list);
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
                    updateft(nbr_ft_at(selectedNeighbors[idx], sz_link_list_other), cur_c);
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
                        dominate_count += dominated_list[i].size();
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
                    if (level == 0) {
                        update_nbr_ft(selectedNeighbors[idx], selectedNeighbors_other, dominated_list);
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

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
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


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
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
            // label_lookup_[getExternalLabel(i)] = i; // removed, so no longer support index update
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
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        assert(maxlevel_ == 1); // for two layers only
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_layer_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        top_layer_candidates = searchTopLayerST<true>(currObj, query_data, ef_top_, isIdAllowed);
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    top_layer_candidates, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    top_layer_candidates, query_data, std::max(ef_, k), isIdAllowed);
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


    std::priority_queue<std::pair<dist_t, labeltype >>
    hybridSearch(const void *query_data, std::vector<std::vector<int>> raw_predicate, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        
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

        top_layer_candidates = searchTopLayerST<true>(currObj, query_data, ef_top_ * 2, isIdAllowed);
        
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
            if (use_two_hop_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, false, true, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else if (use_two_hop_ && !use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, false, true, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else if (!use_two_hop_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<true, false, false, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else
                top_candidates = hybridSearchBaseLayerST<true, false, false, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
        } else {
            if (use_two_hop_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, false, true, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else if (use_two_hop_ && !use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, false, true, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else if (!use_two_hop_ && use_ft_)
                top_candidates = hybridSearchBaseLayerST<false, false, false, true>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
            else
                top_candidates = hybridSearchBaseLayerST<false, false, false, false>(
                        top_layer_candidates, query_data, predicate, ft_predicate, std::max(ef_, k), k, vl, isIdAllowed);
        }

        // for each bucket, scan if CHT selectivity is smaller than threshold 2
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


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
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
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

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
