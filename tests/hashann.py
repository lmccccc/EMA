# define HashANN filtering method

import hashannlib
import time
import os
import numpy as np
import time 
import hashlib
import json
import warnings

try:
    import faiss
except (ImportError, AttributeError):
    faiss = None


def closest_cluster_representatives(distances, groups, cluster_count, metric):
    distances, groups = np.asarray(distances), np.asarray(groups)
    if (metric not in ("l2", "ip") or distances.ndim != 1
            or groups.shape != distances.shape or not np.isfinite(distances).all()
            or not np.issubdtype(groups.dtype, np.integer)
            or cluster_count <= 0 or np.any(groups < 0) or np.any(groups >= cluster_count)):
        raise ValueError("Invalid FAISS centroid assignments")
    # FAISS IP returns similarity, unlike native1-dot distance: larger is closer.
    rank = -distances if metric == "ip" else distances
    order = np.lexsort((np.arange(len(groups)), rank, groups))
    occupied, first = np.unique(groups[order], return_index=True)
    closest = np.full(cluster_count, -1, dtype=np.int64)
    closest[occupied] = order[first]
    levels = np.zeros(len(groups), dtype=np.int32)
    levels[closest[closest >= 0]] = 1
    return closest, levels


class HashANN:
    # NSW with filtering table, B+tree, cluster, pq entry points, counting hash table
    def __init__(self):
        self.index = None
        self.k = 10
        self.threads = 64
        self.index_method = "NSW"  # default index method
        self.d = None  # dimension of the data
        self.N = None  # size of the data
        self.centroid_size = None  # centroid index
        self.closest_ids = None  # closest ids for clustering
        self.best_dist = None  # best distances for clustering
        self.layer = None # is entry point=2, otherwise 1
        self.buckets = None  # buckets for clustering
        self.offsets = None # offsets for buckets


        self.threshold_1 = 0.0001  # < threshold_1: scan on B+ tree / inverted list
        self.threshold_2 = 0.0001  # < threshold_2: scan within the bucket
        self.threshold_3 = 0.0001   # < threshold_3: hybrid search only, scan on B+ tree / inverted list to estimate selectivity

        self.save_root = ""
        
        # counting hash table
        self.bits = 32
        self.metric = "l2"



    def init_params(self, params):
        self.k = params.get("k", 10)
        self.threads = params.get("threads", 64)
        self.index_method = params.get("name", "HNSW")
        self.d = params.get("dim", None)
        self.N = params.get("N", None)
        self.max_elements = params.get("max_elements", None) or self.N
        self.metric = params.get("metric", "l2")

    def build_index(self, params, base_scalars, attr, attr_type_list, index_save_path, threads: int, name: str = "HNSW"):
        self.save_root = params.get("clustering_cache_root", os.path.dirname(index_save_path))
        self.threads = threads
        self.index_method = name
        print("name:", name)
        if name == "HNSW":
            print("building 2 layer HNSW index")
        elif name == "NSW":
            print("building HashANN index")
            exit(-1)
        else:
            print("error: unknown index method:", name)
            exit(-1)

        # get max_cate_attr_value
        print(f"Building index: {name}...")
        start = time.time()
        max_cate_value = 0
        for i in range(len(attr_type_list)):
            if attr_type_list[i] == 1:
                for j in range(len(attr)):
                    for val in attr[j][i]:
                        max_cate_value = max(max_cate_value, val+1)
                print(f"max cate value for attr {i}: {max_cate_value}")
        print(f"iterate label time: {time.time() - start}")

        centroid_size, layers, closest_ids, id2bucket = self.clustering(base_scalars)

        print("base scalar shape:", base_scalars.shape, " layer shape:", layers.shape)

        self.index = hashannlib.Index(space=self.metric, dim=params["dim"])
        self.index.init_index(max_elements=self.max_elements, 
                              top_elements=centroid_size, 
                              ef_construction=params["ef_construction"], 
                              M=params["M"], 
                              ft_bits=params["ft_bits"], 
                              attr_type=attr_type_list,
                              max_cate_size=max_cate_value,
                              edge_level_ft=bool(params.get("edge_level_ft", False)))
        self.index.set_num_threads(threads)

        print("init index done, time:", time.time() - start)

        # add attributes into index
        assert(attr_type_list is not None)
        # self.add_attr(attr, attr_type_list)
        # print("add attr done, time:", time.time() - start)
        # generate attribute indexes
        # self.index.generateAttrIndexes() # B+ tree (numerical) and inverted list (categorical)
        # print("generate attr index done, time:", time.time() - start)

        print("attr type list:", attr_type_list)
        print("attr size:", len(attr), " attr example:", attr[0:5])

        # initAttrMapping iterates 0..max_elements_ in C++ (reads attr[i] for
        # all i < max_elements_), so when max_elements > len(attr) we must
        # pad with a valid sentinel attr so we don't read garbage.
        if len(attr) < self.max_elements:
            pad_unit = attr[0]
            pad = [pad_unit] * (self.max_elements - len(attr))
            attr_padded = list(attr) + pad
            print(f"padding attr {len(attr)} -> {self.max_elements} for initAttrMapping")
        else:
            attr_padded = attr

        # generate Counting hash table
        # self.index.addEpIds(closest_ids.astype(np.uint32).tolist()) # add entry point ids to list, used for partitioning, not used. 
        self.index.initAttrMapping(attr_padded)                                # generate counting_hash_table_mapping (codebook)
        print("generate attr mapping done, time:", time.time() - start)
        # add data points into index
        self.index.add_items(base_scalars, attr, levels=layers)           # add items
        print("add items done, time:", time.time() - start)
        # self.index.addIdToBucket(id2bucket)                         # ivf for clustered scan. Currently not used
        # print("add id to bucket done, time:", time.time() - start)
        # self.index.graphPartition()                                 # generate partition based on graph clustering. Currently not used
        # print("graph partition done, time:", time.time() - start)
        # self.index.initCountingHashTable()                          # static attribute for partitioned dataset, currently not used
        # print("init counting hash table done, time:", time.time() - start)
        # self.index.generateFT()                                     # generate ft for node. new version ft is in edge, not sued.
        end = time.time()
        self.construction_duration_s = end - start
        print(f"Index built: {name}, duration: {end-start}.")
        save_start = time.perf_counter()
        self.index.save_index(index_save_path)
        self.save_duration_s = time.perf_counter() - save_start
        print("index save done, time:", time.time() - end)


        return self.index

    def load_index(self, params, attr_type_list, index_save_path, threads: int, name: str = "HNSW"):
        self.index_method = name
        print("name:", name)

        print(f"Loading index: {name}...")
        self.index = hashannlib.Index(space=self.metric, dim=params["dim"])
        self.index.set_num_threads(threads)
        start = time.time()
        assert(attr_type_list is not None)
        print("index path:", index_save_path)
        self.index.load_index(index_save_path)
        print("index loaded")
        self.index.generateAttrIndexes() # B+ tree (numerical) and inverted list (categorical)
        # self.index.generateIdToBucket()
        end = time.time()

        self.index.set_thresholds(self.threshold_1, self.threshold_2, self.threshold_3)

        print(f"Index loaded: {name}, duration: {end-start}.")

        return self.index

    def add_attr(self, attr: list, attr_type: list):
        assert(len(attr) == self.N)
        num_max_size = 1
        for i in range(self.N):
            for j in range(len(attr_type)):
                if attr_type[j] == 1:
                    num_max_size = max(num_max_size, len(attr[i][j]))

        self.index.addAttr(attr)

    def predicate_translate(self, predicate: list):
        return self.index.predicateTranslate(predicate)

    def hybrid_search(self, query, predicate):
        return self.index.hybrid_knn_query(query, predicate, self.k, num_threads=1)

    def hybrid_search_dnf(self, query, dnf_predicate, check_modes=None):
        modes = check_modes if check_modes else []
        return self.index.hybrid_knn_query_dnf(query, dnf_predicate, modes, self.k, num_threads=1)
        

    # def build_or_load_index(self, params, base_scalars, attr, index_save_path, threads: int, name: str = "NSW"):
        
    #     self.index_method = name
    #     base_size = base_scalars.shape[0]
    #     if name == "HNSW":
    #         print("building HNSW index, not in this function")
    #         exit(0)
    #         # self.index = hashannlib.Index(space=params["metric"], dim=params["dim"])
    #         # return
    #     else:
    #         print("building HashANN index")
    #         # self.index = hashannlib.NSWIndex(space=params["metric"], dim=params["dim"])

    #     centroid_size, layers, closest_ids, best_dist, buckets, offsets = self.clustering(base_scalars)
    #     self.centroid_index = hashannlib.NSWIndex(space=params["metric"], dim=params["dim"])
    #     self.centroid_index.init_index(max_elements=self.N, top_elements=centroid_size, ef_construction=params["ef_construction"], M=params["M"])
    #     self.centroid_index.set_num_threads(threads)

    #     print(f"Building centroid index: {name}...")
    #     start = time.time()
    #     self.centroid_index.add_items(base_scalars, layers=layers)
    #     end = time.time()
    #     self.centroid_index.save_index(index_save_path)

    #     print(f"Index built: {name}, duration: {end-start}.")

    #     return
    

    

    def clustering(self, base_scalars):
        if faiss is None:
            raise RuntimeError("FAISS is required to construct the representative level plan")
        if (base_scalars.shape != (self.N, self.d) or self.metric not in ("l2", "ip")
                or self.N <= 0):
            raise ValueError("Clustering vectors differ from the declared dataset/metric")
        identity_started = time.perf_counter()
        digest = hashlib.sha256()
        for start in range(0, self.N, 65_536):
            block = np.ascontiguousarray(base_scalars[start:start + 65_536], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError("Nonfinite clustering vectors")
            digest.update(memoryview(block))
        seed, max_train_size = 1234, 2_560_000
        centroid_size = min(self.N, int(np.sqrt(self.N)) * 4)
        identity = {
            "schema": "metric-nearest-representatives-v2", "metric": self.metric,
            "N": self.N, "dimension": self.d, "vectors_sha256": digest.hexdigest(),
            "seed": seed, "iterations": 25, "training_limit": max_train_size,
            "centroid_size": centroid_size, "spherical": False,
            "faiss_version": faiss.__version__, "threads": self.threads,
        }
        self.clustering_identity_s = time.perf_counter() - identity_started
        save_file = self.save_root+"_clustering_nn2.npz"
        if os.path.exists(save_file):
            print("loading clustering from file:", save_file)
            with np.load(save_file, allow_pickle=False) as data:
                if ("identity" not in data or json.loads(data["identity"].item()) != identity
                        or data["centroid_size"].item() != centroid_size):
                    raise RuntimeError(f"Clustering cache source/metric/policy mismatch: {save_file}")
                layers, closest_ids, id2bucket = data["layers"], data["closest_ids"], data["id2bucket"]
                self.clustering_metadata = json.loads(data["metadata"].item())
            if (layers.shape != (self.N,) or id2bucket.shape != (self.N,)
                    or closest_ids.shape != (centroid_size,)
                    or any(not np.issubdtype(value.dtype, np.integer)
                           for value in (layers, closest_ids, id2bucket))
                    or np.any((layers != 0) & (layers != 1))
                    or np.any(id2bucket < 0) or np.any(id2bucket >= centroid_size)):
                raise RuntimeError(f"Invalid clustering cache dimensions/assignments: {save_file}")
            representatives = closest_ids[closest_ids >= 0]
            expected_levels = np.zeros(self.N, dtype=np.int32)
            if (np.any(closest_ids < -1) or np.any(representatives >= self.N)
                    or len(np.unique(representatives)) != len(representatives)
                    or not np.array_equal(id2bucket[representatives], np.flatnonzero(closest_ids >= 0))
                    or not np.array_equal(np.unique(id2bucket), np.flatnonzero(closest_ids >= 0))):
                raise RuntimeError(f"Invalid clustering representatives: {save_file}")
            expected_levels[representatives] = 1
            if not np.array_equal(layers, expected_levels):
                raise RuntimeError(f"Clustering levels disagree with representatives: {save_file}")
            if (self.clustering_metadata["identity"] != identity
                    or self.clustering_metadata["levels_sha256"]
                    != hashlib.sha256(memoryview(np.ascontiguousarray(layers))).hexdigest()):
                raise RuntimeError(f"Clustering cache metadata/checksum mismatch: {save_file}")
            print("clustering loaded")
            self.clustering_cache_reused = True
            self.clustering_cache_path = save_file
            return centroid_size, layers, closest_ids, id2bucket

        started = time.perf_counter()
        print("d:", self.d, "N:", self.N, "centroid_size:", centroid_size)
        clustering = faiss.Clustering(self.d, centroid_size)
        clustering.seed = seed
        clustering.niter = 25

        print(f"Clustering parameters: niter={clustering.niter}, max_points_per_centroid={clustering.max_points_per_centroid}")

        if self.metric == "l2":
            print("Using L2 metric for clustering")
            index = faiss.IndexFlatL2(self.d)  # For assignment
        elif self.metric == "ip":
            print("Using IP metric for clustering")
            index = faiss.IndexFlatIP(self.d)  # For assignment
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        faiss.omp_set_num_threads(self.threads)

        random_indices = np.arange(self.N, dtype=np.int64)
        if self.N > max_train_size:
            random_indices = np.random.default_rng(seed).choice(self.N, size=max_train_size, replace=False)
            training_data = base_scalars[random_indices]
            print(f"Training clustering on a subset of size {max_train_size}")
        else:
            training_data = base_scalars
            print(f"Training clustering on the full dataset of size {self.N}")

        clustering.train(training_data, index)

        centroids = faiss.vector_to_array(clustering.centroids).reshape(centroid_size, self.d)
        print(f"Clustering done, centroids shape: {centroids.shape}")
        # print("centroids:", centroids)

        # -----------------------------
        # Step 3: Assign all points to clusters
        # -----------------------------
        if self.metric == "l2":
            index = faiss.IndexFlatL2(self.d)
        elif self.metric == "ip":
            index = faiss.IndexFlatIP(self.d)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        index.add(centroids)

        # Find nearest centroid for each point
        distances = np.empty(self.N, dtype=np.float32)
        id2bucket = np.empty(self.N, dtype=np.int64)
        for start in range(0, self.N, 65_536):
            stop = min(start + 65_536, self.N)
            values, groups = index.search(np.ascontiguousarray(base_scalars[start:stop]), 1)
            distances[start:stop], id2bucket[start:stop] = values[:, 0], groups[:, 0]
        closest_ids, layers = closest_cluster_representatives(
            distances, id2bucket, centroid_size, self.metric)
        empty = int(np.count_nonzero(closest_ids < 0))
        if empty:
            warnings.warn(f"{empty} empty FAISS clusters receive no representative; row -1 is never used",
                          RuntimeWarning)
        self.clustering_metadata = {
            "identity": identity, "generation_s": time.perf_counter() - started,
            "empty_clusters": empty, "representative_count": int(layers.sum()),
            "training_ids_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(random_indices))).hexdigest(),
            "centroids_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(centroids))).hexdigest(),
            "levels_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(layers))).hexdigest(),
        }
        pending = save_file + f".{os.getpid()}.pending"
        with open(pending, "xb") as stream:
            np.savez(stream, centroid_size=centroid_size, layers=layers, closest_ids=closest_ids,
                     id2bucket=id2bucket, identity=json.dumps(identity),
                     metadata=json.dumps(self.clustering_metadata))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, save_file)
        self.clustering_cache_reused = False
        self.clustering_cache_path = save_file
        print("clustering saved to file:", save_file)

        return centroid_size, layers, closest_ids, id2bucket
