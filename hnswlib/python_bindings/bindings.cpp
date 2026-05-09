#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}




inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
            buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "The input label shape %d does not match the input data vector shape %d",
                ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;
    size_t default_ef_top;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
        default_ef_top = 1;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t topElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        size_t ft_bits,
        std::vector<int> attr_type, 
        size_t max_cate_size,
        bool allow_replace_deleted,
        bool edge_level_ft) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, topElements, M, efConstruction, random_seed, ft_bits, attr_type, max_cate_size, allow_replace_deleted, edge_level_ft);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        appr_alg->ef_top_ = default_ef_top;
        seed = random_seed;
    }

    void set_ft_flag(bool flag){
        if(appr_alg)
            appr_alg->set_ft_flag(flag);
    }

    void set_edge_level_ft(bool flag){
        if(appr_alg)
            appr_alg->edge_level_ft_ = flag;
    }

    void set_thresholds(double threshold_1, double threshold_2, double threshold_3){
        if(appr_alg)
            appr_alg->set_thresholds(threshold_1, threshold_2, threshold_3);
    }

    void set_ft_routing_flag(bool flag){
        if(appr_alg)
            appr_alg->set_ft_routing_flag(flag);
    }

    void set_ft_routing_min_deg(double threshold){
        if(appr_alg)
            appr_alg->set_ft_routing_min_deg(threshold);
    }

    void set_min_deg(double threshold){
        set_ft_routing_min_deg(threshold);
    }

    py::dict get_ft_stats() {
        py::dict stats;
        if (appr_alg) {
            long ft_total = appr_alg->metric_ft_passed_total.load();
            long ft_fp = appr_alg->metric_ft_false_positives.load();
            long pred_checked = appr_alg->metric_predicate_checked.load();
            long total_nbrs = appr_alg->metric_total_neighbors.load();
            stats["ft_passed_total"] = ft_total;
            stats["ft_false_positives"] = ft_fp;
            stats["ft_true_positives"] = ft_total - ft_fp;
            stats["ft_fp_rate"] = ft_total > 0 ? (double)ft_fp / ft_total : 0.0;
            stats["predicate_checked"] = pred_checked;
            stats["predicate_fp_rate"] = pred_checked > 0 ? (double)ft_fp / pred_checked : 0.0;
            stats["total_neighbors"] = total_nbrs;
            stats["ft_rejection_rate"] = total_nbrs > 0 ? 1.0 - (double)ft_total / total_nbrs : 0.0;
        }
        return stats;
    }

    void reset_ft_stats() {
        if (appr_alg) {
            appr_alg->metric_ft_passed_total = 0;
            appr_alg->metric_ft_false_positives = 0;
            appr_alg->metric_predicate_checked = 0;
            appr_alg->metric_total_neighbors = 0;
        }
    }

    py::dict augment_ft_neighbors(int min_same = 8, int max_hops = 3) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        auto [edges, nodes] = appr_alg->augment_ft_neighbors(min_same, max_hops);
        py::dict result;
        result["edges_added"] = edges;
        result["nodes_augmented"] = nodes;
        return result;
    }

    py::dict augment_ft_bfs(int min_same = 4, int max_hops = 3, int num_threads = 1) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        auto [edges, nodes] = appr_alg->augment_ft_bfs(min_same, max_hops, num_threads);
        py::dict result;
        result["edges_added"] = edges;
        result["nodes_augmented"] = nodes;
        return result;
    }

    py::dict color_ft_bit(int attr_idx, int bit_idx, int K = 1) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        auto [seeds, unreach, flipped] = appr_alg->color_ft_bit(attr_idx, bit_idx, K);
        py::dict result;
        result["seeds"] = seeds;
        result["unreachable_qualifying"] = unreach;
        result["bits_flipped"] = flipped;
        return result;
    }

    long long color_all_ft_bits(int K = 1, bool verbose = true) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->color_all_ft_bits(K, verbose);
    }

    py::dict color_ft_bit_voronoi(int attr_idx, int bit_idx, int K_neighbors = 4) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        auto [n_qual, n_bridges, flipped] = appr_alg->color_ft_bit_voronoi(attr_idx, bit_idx, K_neighbors);
        py::dict result;
        result["qualifying"] = n_qual;
        result["bridges"] = n_bridges;
        result["bits_flipped"] = flipped;
        return result;
    }

    long long color_all_ft_bits_voronoi(int K_neighbors = 4, bool verbose = true) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->color_all_ft_bits_voronoi(K_neighbors, verbose);
    }

    py::dict color_ft_bit_diverse_tail(int attr_idx, int bit_idx, int K_top = 16, int K_tail = 4) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        auto [n_qual, n_bridges, flipped] = appr_alg->color_ft_bit_diverse_tail(attr_idx, bit_idx, K_top, K_tail);
        py::dict result;
        result["qualifying"] = n_qual;
        result["bridges"] = n_bridges;
        result["bits_flipped"] = flipped;
        return result;
    }

    long long color_all_ft_bits_diverse_tail(int K_top = 16, int K_tail = 4,
                                              int num_threads = 0, bool verbose = true) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->color_all_ft_bits_diverse_tail(K_top, K_tail, num_threads, verbose);
    }

    void augment_edges_cht(int efc = 2000, int num_threads = 32) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        appr_alg->augment_edges_cht(efc, num_threads);
    }

    void set_attr_sort_alpha(double alpha) {
        if (appr_alg) appr_alg->attr_sort_alpha_ = alpha;
    }

    double get_attr_sort_alpha() {
        return appr_alg ? appr_alg->attr_sort_alpha_ : 0.0;
    }

    // Return degree of every L0 node as a numpy array
    py::array_t<int> get_degrees() {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        size_t n = appr_alg->cur_element_count;
        py::array_t<int> degrees(n);
        auto buf = degrees.mutable_unchecked<1>();
        for (size_t i = 0; i < n; i++) {
            auto* ll = appr_alg->get_linklist0(i);
            buf(i) = appr_alg->getListCount(ll);
        }
        return degrees;
    }

    // Return neighbors of a specific node at L0 as numpy array
    py::array_t<unsigned int> get_neighbors(unsigned int node_id) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        if (node_id >= appr_alg->cur_element_count) throw std::runtime_error("node_id out of range");
        auto* ll = appr_alg->get_linklist0(node_id);
        unsigned int size = appr_alg->getListCount(ll);
        unsigned int* data = (unsigned int*)((char*)ll + sizeof(hnswlib::linklistsizeint));
        py::array_t<unsigned int> nbrs(size);
        auto buf = nbrs.mutable_unchecked<1>();
        for (unsigned int i = 0; i < size; i++) buf(i) = data[i];
        return nbrs;
    }

    // Return popcount of each node's FT, shape (n, num_attrs)
    py::array_t<int> get_ft_bit_counts() {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        size_t n = appr_alg->cur_element_count;
        size_t num_attrs = appr_alg->attr_type_.size();
        size_t ft_bytes = appr_alg->ft_bytes_;
        
        py::array_t<int> counts({(py::ssize_t)n, (py::ssize_t)num_attrs});
        auto buf = counts.mutable_unchecked<2>();
        
        for (size_t i = 0; i < n; i++) {
            for (size_t a = 0; a < num_attrs; a++) {
                unsigned char* ft = appr_alg->node_ft_at(i, a);
                int bits = 0;
                size_t off = 0;
                for (; off + 8 <= ft_bytes; off += 8) {
                    uint64_t val;
                    memcpy(&val, ft + off, 8);
                    bits += __builtin_popcountll(val);
                }
                for (; off < ft_bytes; off++) {
                    bits += __builtin_popcount(ft[off]);
                }
                buf(i, a) = bits;
            }
        }
        return counts;
    }

    // Returns (total_bits[a], total_edges[a], total_bit_capacity[a]) per attr,
    // counted over ALL real edges (sum of degrees). Honest edge-FT density.
    py::tuple get_edge_ft_bit_stats() {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        if (!appr_alg->edge_level_ft_) throw std::runtime_error("Not edge-level FT");
        size_t n = appr_alg->cur_element_count;
        size_t num_attrs = appr_alg->attr_type_.size();
        size_t ft_bytes = appr_alg->ft_bytes_;

        std::vector<uint64_t> total_bits(num_attrs, 0);
        uint64_t total_edges = 0;
        for (size_t i = 0; i < n; i++) {
            int* data = (int*)appr_alg->get_linklist0(i);
            size_t deg = appr_alg->getListCount((hnswlib::linklistsizeint*)data);
            total_edges += deg;
            for (size_t e = 0; e < deg; e++) {
                for (size_t a = 0; a < num_attrs; a++) {
                    unsigned char* ft = appr_alg->edge_ft_at(i, e, a);
                    int bits = 0;
                    size_t off = 0;
                    for (; off + 8 <= ft_bytes; off += 8) {
                        uint64_t val;
                        memcpy(&val, ft + off, 8);
                        bits += __builtin_popcountll(val);
                    }
                    for (; off < ft_bytes; off++) bits += __builtin_popcount(ft[off]);
                    total_bits[a] += bits;
                }
            }
        }
        py::array_t<uint64_t> bits_arr(num_attrs);
        auto b = bits_arr.mutable_unchecked<1>();
        for (size_t a = 0; a < num_attrs; a++) b(a) = total_bits[a];
        size_t bit_cap_per_edge = ft_bytes * 8;
        return py::make_tuple(bits_arr, (uint64_t)total_edges, (uint64_t)bit_cap_per_edge);
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }

    
    void set_ef_top(size_t ef_top) {
      default_ef_top = ef_top;
      if (appr_alg)
          appr_alg->ef_top_ = ef_top;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void initAttrSpace(){
        appr_alg->init_attr_space();
    }

    // void addAttr(const std::vector<std::vector<std::vector<int>>>& data){
    //     appr_alg->add_attr(data);
    // }

    // void generateFT(){
    //     py::gil_scoped_release l;
    //     appr_alg->generateFT();
    // }


    void loadIndex(const std::string &path_to_index, size_t max_elements, size_t top_elements, bool allow_replace_deleted, bool dynamic = false) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, top_elements, allow_replace_deleted, dynamic);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, const std::vector<std::vector<std::vector<int>>>& attr_data, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false, const py::array_t<int>& levels = py::array_t<int>()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        std::cout << "Adding " << rows << " items with dimension " << features << std::endl;

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                int level = levels.at(0);
                float* vector_data = (float*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id, attr_data[0], replace_deleted, level);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    if(id % 1000 == 0){
                        std::cout << "Adding point id: " << id << std::endl;
                    }
                    int level = levels.size() ? levels.at(row) : 0;
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, attr_data[row], replace_deleted, level);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    if(id % 1000 == 0){
                        std::cout << "Adding point id: " << id << std::endl;
                    }
                    int level = levels.size() ? levels.at(row) : 0;
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, attr_data[row], replace_deleted, level);
                    });
            }
            cur_l += rows;
        }
    }


    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        std::vector<std::string> return_types{"numpy", "list"};
        if (std::find(std::begin(return_types), std::end(return_types), return_type) == std::end(return_types)) {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();

            if (ids_numpy.ndim == 0) {
                throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
            } else {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
        }

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        if (return_type == "list") {
            return py::cast(data);
        }
        if (return_type == "numpy") {
            return py::array_t< data_t, py::array::c_style | py::array::forcecast >(py::cast(data));
        }
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        char* link_list_npy = (char*)malloc(link_npy_size);
        int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize) {
                memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_ll(link_list_npy, [](void* f) {
            delete[] f;
            });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_node"_a = appr_alg->enterpoint_node_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            "element_levels"_a = py::array_t<int>(
                { appr_alg->element_levels_.size() },  // shape
                { sizeof(int) },  // C-style contiguous strides for each index
                element_levels_npy,  // the data pointer
                free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0),

            "link_lists"_a = py::array_t<char>(
                { link_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                link_list_npy,  // the data pointer
                free_when_done_ll));
    }

    void addEpIds(const std::vector<unsigned int>& ep_ids_){
        appr_alg->add_ep_ids(ep_ids_);
    }

    void initCountingHashTable() {
        appr_alg->init_counting_hash_table();
    }

    void attrCheck(){
        appr_alg->attr_check();
    }

    void addBuckets(py::array_t<int> buckets_, py::array_t<int> bucket_offsets_){
        py::buffer_info buf = buckets_.request();
        int* bucket_ptr = static_cast<int*>(buf.ptr);

        py::buffer_info buf2 = bucket_offsets_.request();
        int* bucket_offsets_ptr = static_cast<int*>(buf2.ptr);
        size_t length = buf2.size;   // 数组总元素数
        size_t ndim   = buf2.ndim;   // 维度

        appr_alg->add_buckets(bucket_ptr, bucket_offsets_ptr, length);
    }

    void addIdToBucket(py::array_t<int> id_to_bucket_){
        py::buffer_info buf = id_to_bucket_.request();
        int* id_to_bucket_ptr = static_cast<int*>(buf.ptr);

        appr_alg->add_id_to_bucket(id_to_bucket_ptr);
    }

    void generateAttrIndexes(){
        appr_alg->generate_attr_indexes();
    }

    void graphPartition() {
        appr_alg->graph_partition();
    }

    void initAttrMapping(const std::vector<std::vector<std::vector<int>>>& attr){
        appr_alg->init_attr_mapping(attr);
    }


    std::vector<std::vector<int>> predicateTranslate(const std::vector<std::vector<std::vector<int>>>& predicate) const {
        std::vector<std::vector<int>> result;
        for(int i = 0; i < predicate.size(); i++){
            result.push_back(appr_alg->predicate_translate(predicate[i]));
        }
        return result;
        // int rows = predicate.size();
        // if (num_threads <= 0)
        //     num_threads = num_threads_default;

        // {
        //     py::gil_scoped_release l;
        //     get_input_array_shapes(buffer, &rows, &features);

        //     // avoid using threads when the number of searches is small:
        //     if (rows <= num_threads * 4) {
        //         num_threads = 1;
        //     }
        
        //     ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
        //         std::vector<int> result = appr_alg->predicate_translate(predicate);
        //         if (result.size() != k)
        //             throw std::runtime_error(
        //                 "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
        //         for (int i = k - 1; i >= 0; i--) {
        //             auto& result_tuple = result.top();
        //             data_numpy_d[row * k + i] = result_tuple.first;
        //             data_numpy_l[row * k + i] = result_tuple.second;
        //             result.pop();
        //         }
        //     });
        // }
    }

    std::vector<std::vector<char>> predicate_to_ft(const std::vector<std::vector<std::vector<int>>>& predicate) const {
        std::vector<std::vector<char>> predicate_ft;
        for(int i = 0; i < predicate.size(); i++){
            predicate_ft.push_back(appr_alg->predicate_to_ft(predicate[i]));
        }
        return predicate_ft;
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(Index<float>::ser_version),  // serialization version
            "space"_a = space_name,
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static Index<float>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(Index<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        Index<float>* new_index = new Index<float>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static Index<float> * createFromIndex(const Index<float> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
            }
        }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];


            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }

    py::object knnQuery_return_numpy_with_stats(
        py::object input,
        size_t k = 1) {
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        int64_t* data_numpy_dc;
        int64_t* data_numpy_hops;
        size_t rows, features;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];
            data_numpy_dc = new int64_t[rows];
            data_numpy_hops = new int64_t[rows];


            if (normalize == false) {
                for (size_t row = 0; row < rows; row++) {
                    appr_alg->metric_distance_computations = 0;
                    appr_alg->metric_hops = 0;

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = appr_alg->searchKnn(
                        (void*)items.data(row), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                    data_numpy_dc[row] = appr_alg->metric_distance_computations.load();
                    data_numpy_hops[row] = appr_alg->metric_hops.load();
                }
            } else {
                std::vector<float> norm_array(dim);
                for (size_t row = 0; row < rows; row++) {
                    appr_alg->metric_distance_computations = 0;
                    appr_alg->metric_hops = 0;

                    normalize_vector((float*)items.data(row), norm_array.data());
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = appr_alg->searchKnn(
                        (void*)norm_array.data(), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                    data_numpy_dc[row] = appr_alg->metric_distance_computations.load();
                    data_numpy_hops[row] = appr_alg->metric_hops.load();
                }
            }
        }

        py::capsule free_when_done_l(data_numpy_l, [](void* f) { delete[] f; });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) { delete[] f; });
        py::capsule free_when_done_dc(data_numpy_dc, [](void* f) { delete[] f; });
        py::capsule free_when_done_hops(data_numpy_hops, [](void* f) { delete[] f; });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                {rows, k},
                {k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype)},
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                {rows, k},
                {k * sizeof(dist_t), sizeof(dist_t)},
                data_numpy_d,
                free_when_done_d),
            py::array_t<int64_t>(
                {rows},
                {sizeof(int64_t)},
                data_numpy_dc,
                free_when_done_dc),
            py::array_t<int64_t>(
                {rows},
                {sizeof(int64_t)},
                data_numpy_hops,
                free_when_done_hops));
    }

    py::object hybridKnnQuery_return_numpy(
        py::object input,
        std::vector<std::vector<std::vector<int>>> raw_predicate, 
        size_t k = 1,
        int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];


            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->hybridSearch(
                        (void*)items.data(row), raw_predicate[row], k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->hybridSearch(
                        (void*)items.data(row), raw_predicate[row], k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }

    // DNF predicate query: supports arbitrary AND/OR combinations
    py::object hybridKnnQueryDNF_return_numpy(
        py::object input,
        std::vector<std::vector<std::vector<std::vector<int>>>> dnf_predicates,  // [query][term][attr] = values
        std::vector<std::vector<std::vector<int8_t>>> check_modes,               // [query][term][attr] = mode
        size_t k = 1,
        int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];


            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    const auto& modes = check_modes.empty()
                        ? std::vector<std::vector<int8_t>>()
                        : check_modes[row];
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->hybridSearchDNF(
                        (void*)items.data(row), dnf_predicates[row], modes, k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    const auto& modes = check_modes.empty()
                        ? std::vector<std::vector<int8_t>>()
                        : check_modes[row];
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->hybridSearchDNF(
                        (void*)(norm_array.data() + start_idx), dnf_predicates[row], modes, k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },
                { k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype) },
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },
                { k * sizeof(dist_t), sizeof(dist_t) },
                data_numpy_d,
                free_when_done_d));
    }

    py::object hybridKnnQuery_return_numpy_with_stats(
        py::object input,
        std::vector<std::vector<std::vector<int>>> raw_predicate,
        size_t k = 1) {
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        int64_t* data_numpy_dc;
        int64_t* data_numpy_hops;
        size_t rows, features;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];
            data_numpy_dc = new int64_t[rows];
            data_numpy_hops = new int64_t[rows];


            if (normalize == false) {
                for (size_t row = 0; row < rows; row++) {
                    appr_alg->metric_distance_computations = 0;
                    appr_alg->metric_hops = 0;

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = appr_alg->hybridSearch(
                        (void*)items.data(row), raw_predicate[row], k);
                    size_t actual_k = result.size();
                    // Pad missing results with -1 label and max distance
                    for (size_t i = actual_k; i < k; i++) {
                        data_numpy_d[row * k + i] = std::numeric_limits<dist_t>::max();
                        data_numpy_l[row * k + i] = (hnswlib::labeltype)(-1);
                    }
                    for (int i = (int)actual_k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                    data_numpy_dc[row] = appr_alg->metric_distance_computations.load();
                    data_numpy_hops[row] = appr_alg->metric_hops.load();
                }
            } else {
                std::vector<float> norm_array(dim);
                for (size_t row = 0; row < rows; row++) {
                    appr_alg->metric_distance_computations = 0;
                    appr_alg->metric_hops = 0;

                    normalize_vector((float*)items.data(row), norm_array.data());
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = appr_alg->hybridSearch(
                        (void*)norm_array.data(), raw_predicate[row], k);
                    size_t actual_k = result.size();
                    for (size_t i = actual_k; i < k; i++) {
                        data_numpy_d[row * k + i] = std::numeric_limits<dist_t>::max();
                        data_numpy_l[row * k + i] = (hnswlib::labeltype)(-1);
                    }
                    for (int i = (int)actual_k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                    data_numpy_dc[row] = appr_alg->metric_distance_computations.load();
                    data_numpy_hops[row] = appr_alg->metric_hops.load();
                }
            }
        }

        py::capsule free_when_done_l(data_numpy_l, [](void* f) { delete[] f; });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) { delete[] f; });
        py::capsule free_when_done_dc(data_numpy_dc, [](void* f) { delete[] f; });
        py::capsule free_when_done_hops(data_numpy_hops, [](void* f) { delete[] f; });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                {rows, k},
                {k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype)},
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                {rows, k},
                {k * sizeof(dist_t), sizeof(dist_t)},
                data_numpy_d,
                free_when_done_d),
            py::array_t<int64_t>(
                {rows},
                {sizeof(int64_t)},
                data_numpy_dc,
                free_when_done_dc),
            py::array_t<int64_t>(
                {rows},
                {sizeof(int64_t)},
                data_numpy_hops,
                free_when_done_hops));
    }


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }


    void setRepairThreshold(double threshold) {
        appr_alg->setRepairThreshold(threshold);
    }

    py::dict getRepairCandidates() const {
        auto candidates = appr_alg->getRepairCandidates();
        py::dict result;
        for (auto& [id, ratio] : candidates) {
            result[py::int_(id)] = ratio;
        }
        return result;
    }

    py::dict popRepairCandidates() {
        auto candidates = appr_alg->popRepairCandidates();
        py::dict result;
        for (auto& [id, ratio] : candidates) {
            result[py::int_(id)] = ratio;
        }
        return result;
    }

    void clearRepairCandidates() {
        appr_alg->clearRepairCandidates();
    }

    size_t repairCandidatesSize() const {
        return appr_alg->repairCandidatesSize();
    }

    bool repairNode(size_t internal_id) {
        return appr_alg->repairNode(static_cast<hnswlib::tableint>(internal_id));
    }

    size_t repairCandidateNodes() {
        return appr_alg->repairCandidates();
    }

    size_t rebuildGraph() {
        return appr_alg->rebuildGraph();
    }
};

template<typename dist_t, typename data_t = float>
class NSWIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::NSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    NSWIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~NSWIndex() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::NSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::NSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = (float*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }
            
            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        std::vector<std::string> return_types{"numpy", "list"};
        if (std::find(std::begin(return_types), std::end(return_types), return_type) == std::end(return_types)) {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();

            if (ids_numpy.ndim == 0) {
                throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
            } else {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
        }

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        if (return_type == "list") {
            return py::cast(data);
        }
        if (return_type == "numpy") {
            return py::array_t< data_t, py::array::c_style | py::array::forcecast >(py::cast(data));
        }
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        // char* link_list_npy = (char*)malloc(link_npy_size);
        // int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        // memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        // memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        // for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
        //     size_t linkListSize = 0;
        //     if (linkListSize) {
        //         memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
        //     }
        // }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        // py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
        //     delete[] f;
        //     });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        // py::capsule free_when_done_ll(link_list_npy, [](void* f) {
        //     delete[] f;
        //     });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_nodes"_a = appr_alg->enterpoint_nodes_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            // "element_levels"_a = py::array_t<int>(
            //     { appr_alg->element_levels_.size() },  // shape
            //     { sizeof(int) },  // C-style contiguous strides for each index
            //     element_levels_npy,  // the data pointer
            //     free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0)

            // "link_lists"_a = py::array_t<char>(
            //     { link_npy_size },  // shape
            //     { sizeof(char) },  // C-style contiguous strides for each index
            //     link_list_npy,  // the data pointer
            //     free_when_done_ll)
            );
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(NSWIndex<float>::ser_version),  // serialization version
            "space"_a = space_name,
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static NSWIndex<float>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(NSWIndex<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        NSWIndex<float>* new_index = new NSWIndex<float>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::NSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static NSWIndex<float> * createFromIndex(const NSWIndex<float> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_nodes_ = d["enterpoint_nodes"].cast< std::vector< hnswlib::tableint > >();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        // auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        // memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        // for (size_t i = 0; i < appr_alg->max_elements_; i++) {
        //     size_t linkListSize = 0;
        //     if (linkListSize == 0) {
        //         appr_alg->linkLists_[i] = nullptr;
        //     } else {
        //         appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
        //         if (appr_alg->linkLists_[i] == nullptr)
        //             throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

        //         memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
        //     }
        // }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];


            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }
};

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

    hnswlib::labeltype cur_l;
    hnswlib::BruteforceSearch<dist_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<dist_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            for (size_t row = 0; row < rows; row++) {
                size_t id = ids.size() ? ids.at(row) : cur_l + row;
                if (!normalize) {
                    alg->addPoint((void *) items.data(row), (size_t) id);
                } else {
                    std::vector<float> normalized_vector(dim);
                    normalize_vector((float *)items.data(row), normalized_vector.data());
                    alg->addPoint((void *) normalized_vector.data(), (size_t) id);
                }
            }
            cur_l+=rows;
        }
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];


            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                    (void*)items.data(row), k);
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }

        py::capsule free_when_done_l(data_numpy_l, [](void *f) {
            delete[] f;
        });
        py::capsule free_when_done_d(data_numpy_d, [](void *f) {
            delete[] f;
        });


        return py::make_tuple(
                py::array_t<hnswlib::labeltype>(
                        { rows, k },  // shape
                        { k * sizeof(hnswlib::labeltype),
                          sizeof(hnswlib::labeltype)},  // C-style contiguous strides for each index
                        data_numpy_l,  // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        { rows, k },  // shape
                        { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                        data_numpy_d,  // the data pointer
                        free_when_done_d));
    }
};


PYBIND11_PLUGIN(hashannlib) {
        py::module m("hashannlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init(&Index<float>::createFromParams), py::arg("params"))
           /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&Index<float>::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &Index<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("top_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("ft_bits") = 128,
            py::arg("attr_type") = py::list(py::cast(std::vector<int>{0, 1})),
            py::arg("max_cate_size") = 5,
            py::arg("allow_replace_deleted") = false,
            py::arg("edge_level_ft") = false)
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("knn_query_with_stats",
            &Index<float>::knnQuery_return_numpy_with_stats,
            py::arg("data"),
            py::arg("k") = 1)
        .def("hybrid_knn_query",
            &Index<float>::hybridKnnQuery_return_numpy,
            py::arg("data"),
            py::arg("predicate"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("hybrid_knn_query_with_stats",
            &Index<float>::hybridKnnQuery_return_numpy_with_stats,
            py::arg("data"),
            py::arg("predicate"),
            py::arg("k") = 1)
        .def("hybrid_knn_query_dnf",
            &Index<float>::hybridKnnQueryDNF_return_numpy,
            py::arg("data"),
            py::arg("dnf_predicate"),
            py::arg("check_modes") = std::vector<std::vector<std::vector<int8_t>>>(),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("add_items",
            &Index<float>::addItems,
            py::arg("data"),
            py::arg("data_attr"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false,
            py::arg("levels") = py::array_t<int>() )
        .def("initAttrSpace", &Index<float>::initAttrSpace)
        // .def("addAttr", &Index<float>::addAttr, py::arg("attr_data"))
        .def("attrCheck", &Index<float>::attrCheck)
        .def("get_items", &Index<float>::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
        .def("addBuckets", &Index<float>::addBuckets, py::arg("bucket2id_data"), py::arg("bucket_offsets"))
        .def("addIdToBucket", &Index<float>::addIdToBucket, py::arg("id2bucket_data"))
        .def("get_ids_list", &Index<float>::getIdsList)
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_ft_flag", &Index<float>::set_ft_flag, py::arg("ft_flag"))
        .def("set_edge_level_ft", &Index<float>::set_edge_level_ft, py::arg("flag"))
        .def("set_thresholds", &Index<float>::set_thresholds, py::arg("threshold_1"), py::arg("threshold_2"), py::arg("threshold_3"))
        .def("set_ft_routing_flag", &Index<float>::set_ft_routing_flag, py::arg("flag"))
        .def("set_ft_routing_min_deg", &Index<float>::set_ft_routing_min_deg, py::arg("min_deg"))
        .def("set_two_hop_flag", &Index<float>::set_ft_routing_flag, py::arg("two_hop"))  // deprecated alias
        .def("set_two_hop_threshold", &Index<float>::set_ft_routing_min_deg, py::arg("two_hop_threshold"))  // deprecated alias
        .def("set_min_deg", &Index<float>::set_min_deg, py::arg("min_deg"))
        .def("get_ft_stats", &Index<float>::get_ft_stats)
        .def("reset_ft_stats", &Index<float>::reset_ft_stats)
        .def("get_degrees", &Index<float>::get_degrees)
        .def("get_neighbors", &Index<float>::get_neighbors, py::arg("node_id"))
        .def("get_ft_bit_counts", &Index<float>::get_ft_bit_counts)
        .def("get_edge_ft_bit_stats", &Index<float>::get_edge_ft_bit_stats)
        .def("augment_ft_neighbors", &Index<float>::augment_ft_neighbors,
             py::arg("min_same") = 8, py::arg("max_hops") = 3)
        .def("augment_ft_bfs", &Index<float>::augment_ft_bfs,
             py::arg("min_same") = 4, py::arg("max_hops") = 3, py::arg("num_threads") = 1)
        .def("color_ft_bit_voronoi", &Index<float>::color_ft_bit_voronoi,
             py::arg("attr_idx"), py::arg("bit_idx"), py::arg("K_neighbors") = 4)
        .def("color_all_ft_bits_voronoi", &Index<float>::color_all_ft_bits_voronoi,
             py::arg("K_neighbors") = 4, py::arg("verbose") = true)
        .def("color_ft_bit_diverse_tail", &Index<float>::color_ft_bit_diverse_tail,
             py::arg("attr_idx"), py::arg("bit_idx"),
             py::arg("K_top") = 16, py::arg("K_tail") = 4)
        .def("color_all_ft_bits_diverse_tail", &Index<float>::color_all_ft_bits_diverse_tail,
             py::arg("K_top") = 16, py::arg("K_tail") = 4,
             py::arg("num_threads") = 0, py::arg("verbose") = true)
        .def("color_ft_bit", &Index<float>::color_ft_bit,
             py::arg("attr_idx"), py::arg("bit_idx"), py::arg("K") = 1)
        .def("color_all_ft_bits", &Index<float>::color_all_ft_bits,
             py::arg("K") = 1, py::arg("verbose") = true)
        .def("augment_edges_cht", &Index<float>::augment_edges_cht,
             py::arg("efc") = 2000, py::arg("num_threads") = 32)
        .def("set_attr_sort_alpha", &Index<float>::set_attr_sort_alpha, py::arg("alpha"))
        .def("get_attr_sort_alpha", &Index<float>::get_attr_sort_alpha)
        .def("set_augmented_min_deg", [](Index<float>& self, int min_deg) {
            self.appr_alg->augmented_min_deg_ = min_deg;
        }, py::arg("min_deg"))
        .def("set_use_augmented_edges", [](Index<float>& self, bool use) {
            self.appr_alg->use_augmented_edges_ = use;
        }, py::arg("use"))
        .def("set_ef_top", &Index<float>::set_ef_top, py::arg("ef_top"))
        .def("addEpIds", &Index<float>::addEpIds, py::arg("ep_ids"))
        .def("predicateTranslate", &Index<float>::predicateTranslate, py::arg("predicate"))
        .def("predicateToFT", &Index<float>::predicate_to_ft, py::arg("predicate"))
        // .def("generateFT", &Index<float>::generateFT)
        .def("initCountingHashTable", &Index<float>::initCountingHashTable)
        .def("generateAttrIndexes", &Index<float>::generateAttrIndexes)
        .def("graphPartition", &Index<float>::graphPartition)
        .def("initAttrMapping", &Index<float>::initAttrMapping, py::arg("attr"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("index_file_size", &Index<float>::indexFileSize)
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &Index<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("top_elements") = 0,
            py::arg("allow_replace_deleted") = false,
            py::arg("dynamic") = false)
        .def("mark_deleted", &Index<float>::markDeleted, py::arg("label"))
        .def("unmark_deleted", &Index<float>::unmarkDeleted, py::arg("label"))
        .def("resize_index", &Index<float>::resizeIndex, py::arg("new_size"))
        .def("get_max_elements", &Index<float>::getMaxElements)
        .def("get_current_count", &Index<float>::getCurrentCount)
        .def("set_repair_threshold", &Index<float>::setRepairThreshold, py::arg("threshold"))
        .def("get_repair_candidates", &Index<float>::getRepairCandidates)
        .def("pop_repair_candidates", &Index<float>::popRepairCandidates)
        .def("clear_repair_candidates", &Index<float>::clearRepairCandidates)
        .def("repair_candidates_size", &Index<float>::repairCandidatesSize)
        .def("repair_node", &Index<float>::repairNode, py::arg("internal_id"))
        .def("repair_candidate_nodes", &Index<float>::repairCandidateNodes)
        .def("rebuild_graph", &Index<float>::rebuildGraph)
        .def_readonly("space", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const Index<float> &ind) {  // __getstate__
                return py::make_tuple(ind.getIndexParams()); /* Return dict (wrapped in a tuple) that fully encodes state of the Index object */
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return Index<float>::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const Index<float> &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        py::class_<NSWIndex<float>>(m, "NSWIndex")
        .def(py::init(&NSWIndex<float>::createFromParams), py::arg("params"))
           /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&NSWIndex<float>::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &NSWIndex<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("knn_query",
            &NSWIndex<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("add_items",
            &NSWIndex<float>::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_items", &NSWIndex<float>::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
        .def("get_ids_list", &NSWIndex<float>::getIdsList)
        .def("set_ef", &NSWIndex<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &NSWIndex<float>::set_num_threads, py::arg("num_threads"))
        .def("index_file_size", &NSWIndex<float>::indexFileSize)
        .def("save_index", &NSWIndex<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &NSWIndex<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("mark_deleted", &NSWIndex<float>::markDeleted, py::arg("label"))
        .def("unmark_deleted", &NSWIndex<float>::unmarkDeleted, py::arg("label"))
        .def("resize_index", &NSWIndex<float>::resizeIndex, py::arg("new_size"))
        .def("get_max_elements", &NSWIndex<float>::getMaxElements)
        .def("get_current_count", &NSWIndex<float>::getCurrentCount)
        .def_readonly("space", &NSWIndex<float>::space_name)
        .def_readonly("dim", &NSWIndex<float>::dim)
        .def_readwrite("num_threads", &NSWIndex<float>::num_threads_default)
        .def_property("ef",
          [](const NSWIndex<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](NSWIndex<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const NSWIndex<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const NSWIndex<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const NSWIndex<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const NSWIndex<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const NSWIndex<float> &ind) {  // __getstate__
                return py::make_tuple(ind.getIndexParams()); /* Return dict (wrapped in a tuple) that fully encodes state of the NSWIndex object */
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return NSWIndex<float>::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const NSWIndex<float> &a) {
            return "<hnswlib.NSWIndex(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        py::class_<BFIndex<float>>(m, "BFIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &BFIndex<float>::init_new_index, py::arg("max_elements"))
        .def("knn_query",
            &BFIndex<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("add_items", &BFIndex<float>::addItems, py::arg("data"), py::arg("ids") = py::none())
        .def("delete_vector", &BFIndex<float>::deleteVector, py::arg("label"))
        .def("set_num_threads", &BFIndex<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &BFIndex<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index", &BFIndex<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements") = 0)
        .def("__repr__", [](const BFIndex<float> &a) {
            return "<hnswlib.BFIndex(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        })
        .def("get_max_elements", &BFIndex<float>::getMaxElements)
        .def("get_current_count", &BFIndex<float>::getCurrentCount)
        .def_readwrite("num_threads", &BFIndex<float>::num_threads_default);
        return m.ptr();
}
