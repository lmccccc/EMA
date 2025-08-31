#pragma once

#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace faiss_ivfpq{

struct Index {
    using component_t = float;
    using distance_t = float;

    int d;        ///< vector dimension
    idx_t ntotal; ///< total nb of indexed vectors
    bool verbose; ///< verbosity level

    /// set if the Index does not require training, or if training is
    /// done already
    bool is_trained;

    /// type of metric this index uses for search
    MetricType metric_type;
    float metric_arg; ///< argument of the metric type

    explicit Index(idx_t d = 0, MetricType metric = METRIC_L2)
            : d(d),
              ntotal(0),
              verbose(false),
              is_trained(true),
              metric_type(metric),
              metric_arg(0) {}

    virtual ~Index() = default;

    /** Perform training on a representative set of vectors
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     */
    virtual void train(idx_t n, const float* x){
        // does nothing by default
    }

    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * This function slices the input vectors in chunks smaller than
     * blocksize_add and calls add_core.
     * @param n      number of vectors
     * @param x      input matrix, size n * d
     */
    virtual void add(idx_t n, const float* x) = 0;




    /** return the indexes of the k vectors closest to the query x.
     *
     * This function is identical as search but only return labels of
     * neighbors.
     * @param n           number of vectors
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     * @param k           number of nearest neighbours
     */
    virtual void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const;

    /// removes all elements from the database.
    virtual void reset() = 0;

    /* The standalone codec interface */

    /** size of the produced codes in bytes */
    virtual size_t sa_code_size() const;

    /** encode a set of vectors
     *
     * @param n       number of vectors
     * @param x       input vectors, size n * d
     * @param bytes   output encoded vectors, size n * sa_code_size()
     */
    virtual void sa_encode(idx_t n, const float* x, uint8_t* bytes) const;

    /** decode a set of vectors
     *
     * @param n       number of vectors
     * @param bytes   input encoded vectors, size n * sa_code_size()
     * @param x       output vectors, size n * d
     */
    virtual void sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
        std::cout << "standalone codec not implemented for this type of index\n";
        exit(-1);
    }


    /** moves the entries from another dataset to self.
     * On output, other is empty.
     * add_id is added to all moved ids
     * (for sequential ids, this would be this->ntotal) */
    virtual void merge_from(Index& otherIndex, idx_t add_id = 0){
        std::cout << "merge_from() not implemented\n";
        exit(-1);
    }

    /** check that the two indexes are compatible (ie, they are
     * trained in the same way and have the same
     * parameters). Otherwise throw. */
    virtual void check_compatible_for_merge(const Index& otherIndex) const {
        std::cout << "check_compatible_for_merge() not implemented\n";
        exit(-1);
    }

    /** return a distance computer that can compute distances
     * between the query and the indexed vectors.
     * The returned object is owned by the index and should not be deleted.
     */
    virtual DistanceComputer* get_distance_computer() const const {
        if (metric_type == METRIC_L2) {
            return new GenericDistanceComputer(*this);
        } else {
            std::cout << "get_distance_computer() not implemented\n";
            exit(-1);
        }
    }

    assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
        std::vector<float> distances(n * k);
        search(n, x, k, distances.data(), labels);
    }

    void compute_residual(const float* x, float* residual, idx_t key) const {
        reconstruct(key, residual);
        for (size_t i = 0; i < d; i++) {
            residual[i] = x[i] - residual[i];
        }
    }


    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const {
    #pragma omp parallel for
        for (idx_t i = 0; i < n; ++i) {
            compute_residual(&xs[i * d], &residuals[i * d], keys[i]);
        }
    }

};

struct Level1Quantizer {
    /// quantizer that maps vectors to inverted lists
    Index* quantizer = nullptr;

    /// number of inverted lists
    size_t nlist = 0;
    char quantizer_trains_alone = 0;
    bool own_fields = false; ///< whether object owns the quantizer

    ClusteringParameters cp; ///< to override default clustering params
    /// to override index used during clustering
    Index* clustering_index = nullptr;



    /// compute the number of bytes required to store list ids
    size_t coarse_code_size() const;
    void encode_listno(idx_t list_no, uint8_t* code) const;
    idx_t decode_listno(const uint8_t* code) const;

    Level1Quantizer(Index* quantizer, size_t nlist)
        : quantizer(quantizer), nlist(nlist) {
        // here we set a low # iterations because this is typically used
        // for large clusterings (nb this is not used for the MultiIndex,
        // for which quantizer_trains_alone = true)
        cp.niter = 10;
    }
    

    Level1Quantizer() = default;

    ~Level1Quantizer() {
        if (own_fields) {
            delete quantizer;
        }
    }

    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train_q1(
        size_t n,
        const float* x,
        bool verbose,
        DISTFUNC<dist_t> fstdistfunc_ ) { // MetricType metric_type
        size_t d = quantizer->d;
        if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
            if (verbose)
                printf("IVF quantizer does not need training.\n");
        } else if (quantizer_trains_alone == 1) {
            if (verbose)
                printf("IVF quantizer trains alone...\n");
            quantizer->verbose = verbose;
            quantizer->train(n, x);
            assert(quantizer->ntotal == nlist);
            // FAISS_THROW_IF_NOT_MSG(
            //         quantizer->ntotal == nlist,
            //         "nlist not consistent with quantizer size");
        } else if (quantizer_trains_alone == 0) {
            if (verbose)
                printf("Training level-1 quantizer on %zd vectors in %zdD\n", n, d);

            Clustering clus(d, nlist, cp);
            quantizer->reset();
            if (clustering_index) {
                clus.train(n, x, *clustering_index);
                quantizer->add(nlist, clus.centroids.data());
            } else {
                clus.train(n, x, *quantizer);
            }
            quantizer->is_trained = true;
        } else if (quantizer_trains_alone == 2) {
            if (verbose) {
                printf("Training L2 quantizer on %zd vectors in %zdD%s\n",
                    n,
                    d,
                    clustering_index ? "(user provided index)" : "");
            }
            // also accept spherical centroids because in that case
            // L2 and IP are equivalent
            // FAISS_THROW_IF_NOT(
            //         metric_type == METRIC_L2 ||
            //         (metric_type == METRIC_INNER_PRODUCT && cp.spherical));

            Clustering clus(d, nlist, cp);
            if (!clustering_index) {
                IndexFlatL2 assigner(d);
                clus.train(n, x, assigner);
            } else {
                clus.train(n, x, *clustering_index);
            }
            if (verbose) {
                printf("Adding centroids to quantizer\n");
            }
            if (!quantizer->is_trained) {
                if (verbose) {
                    printf("But training it first on centroids table...\n");
                }
                quantizer->train(nlist, clus.centroids.data());
            }
            quantizer->add(nlist, clus.centroids.data());
        }
    }
};

struct TransformedVectors {
    const float* x;
    bool own_x;
    TransformedVectors(const float* x_orig, const float* x) : x(x) {
        own_x = x_orig != x;
    }

    ~TransformedVectors() {
        if (own_x) {
            delete[] x;
        }
    }
};

const float* fvecs_maybe_subsample(
        size_t d,
        size_t* n,
        size_t nmax,
        const float* x,
        bool verbose,
        int64_t seed) {
    if (*n <= nmax)
        return x; // nothing to do

    size_t n2 = nmax;
    if (verbose) {
        printf("  Input training set too big (max size is %zd), sampling "
               "%zd / %zd vectors\n",
               nmax,
               n2,
               *n);
    }
    std::vector<int> subset(*n);
    rand_perm(subset.data(), *n, seed);
    float* x_subset = new float[n2 * d];
    for (int64_t i = 0; i < n2; i++)
        memcpy(&x_subset[i * d], &x[subset[i] * size_t(d)], sizeof(x[0]) * d);
    *n = n2;
    return x_subset;
}

struct IndexIVF{

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};


    IndexIVFIndexIVF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        DISTFUNC<dist_t> _fstdistfunc_)
        : Index(d, _fstdistfunc_),
          code_size(code_size) {
        assert(d == quantizer->d);
        is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    }

    IndexIVF::IndexIVF() = default;

    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * This function slices the input vectors in chunks smaller than
     * blocksize_add and calls add_core.
     * @param n      number of vectors
     * @param x      input matrix, size n * d
     */
    virtual void add(idx_t n, const float* x) override {
        add_with_ids(n, x, nullptr);
    }

    add_with_ids(idx_t n, const float* x, const idx_t* xids) {
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
        quantizer->assign(n, x, coarse_idx.get());
        add_core(n, x, xids, coarse_idx.get());
    }

    add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* coarse_idx,
            void* inverted_list_context) {
        // do some blocking to avoid excessive allocs
        idx_t bs = 65536;
        if (n > bs) {
            for (idx_t i0 = 0; i0 < n; i0 += bs) {
                idx_t i1 = std::min(n, i0 + bs);
                if (verbose) {
                    printf("   IndexIVF::add_with_ids %" PRId64 ":%" PRId64 "\n",
                        i0,
                        i1);
                }
                add_core(
                        i1 - i0,
                        x + i0 * d,
                        xids ? xids + i0 : nullptr,
                        coarse_idx + i0,
                        inverted_list_context);
            }
            return;
        }
        FAISS_THROW_IF_NOT(coarse_idx);
        FAISS_THROW_IF_NOT(is_trained);
        direct_map.check_can_add(xids);

        size_t nadd = 0, nminus1 = 0;

        for (size_t i = 0; i < n; i++) {
            if (coarse_idx[i] < 0)
                nminus1++;
        }

        std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
        encode_vectors(n, x, coarse_idx, flat_codes.get());

        DirectMapAdd dm_adder(direct_map, n, xids);

    #pragma omp parallel reduction(+ : nadd)
        {
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();

            // each thread takes care of a subset of lists
            for (size_t i = 0; i < n; i++) {
                idx_t list_no = coarse_idx[i];
                if (list_no >= 0 && list_no % nt == rank) {
                    idx_t id = xids ? xids[i] : ntotal + i;
                    size_t ofs = invlists->add_entry(
                            list_no,
                            id,
                            flat_codes.get() + i * code_size,
                            inverted_list_context);

                    dm_adder.add(i, list_no, ofs);

                    nadd++;
                } else if (rank == 0 && list_no == -1) {
                    dm_adder.add(i, -1, 0);
                }
            }
        }

        if (verbose) {
            printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
                nadd,
                n,
                nminus1);
        }

        ntotal += n;
    }


    train(idx_t n, const float* x) {
        if (verbose) {
            printf("Training level-1 quantizer\n");
        }

        train_q1(n, x, verbose, metric_type);

        if (verbose) {
            printf("Training IVF residual\n");
        }

        // optional subsampling
        idx_t max_nt = 0;
        if (max_nt <= 0) {
            max_nt = (size_t)1 << 35;
        }

        TransformedVectors tv(
                x, fvecs_maybe_subsample(d, (size_t*)&n, max_nt, x, verbose));

        if (by_residual) {
            std::vector<idx_t> assign(n);
            quantizer->assign(n, tv.x, assign.data());

            std::vector<float> residuals(n * d);
            quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());

            train_encoder(n, residuals.data(), assign.data());
        } else {
            train_encoder(n, tv.x, nullptr);
        }

        is_trained = true;
    }

    void train_encoder(
            idx_t /*n*/,
            const float* /*x*/,
            const idx_t* assign) {
        // does nothing by default
        if (verbose) {
            printf("IndexIVF: no residual training\n");
        }
    }

};


}
