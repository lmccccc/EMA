# define HashANN filtering method

import hashannlib
import time
import os
import faiss
import numpy as np
import time 

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
        
        # counting hash table
        self.bits = 32

    def search(self, query, k=None):
        pass


    def init_params(self, params):
        self.k = params.get("k", 10)
        self.threads = params.get("threads", 64)
        self.index_method = params.get("name", "HNSW")
        self.d = params.get("dim", None)
        self.N = params.get("N", None)

    def build_index(self, params, base_scalars, attr, attr_type_list, index_save_path, threads: int, name: str = "HNSW"):
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

        centroid_size, layers, closest_ids, best_dist, buckets, offsets = self.clustering(base_scalars)
        self.index = hashannlib.Index(space=params["metric"], dim=params["dim"])
        self.index.init_index(max_elements=self.N, 
                              top_elements=centroid_size, 
                              ef_construction=params["ef_construction"], 
                              M=params["M"], 
                              ft_bits=params["ft_bits"], 
                              attr_type=attr_type_list,
                              max_cate_size=max_cate_value)
        self.index.set_num_threads(threads)

        print("init index done, time:", time.time() - start)

        # add attributes into index
        assert(attr_type_list is not None)
        self.add_attr(attr, attr_type_list)
        print("add attr done, time:", time.time() - start)
        # generate attribute indexes
        self.index.generateAttrIndexes() # B+ tree (numerical) and inverted list (categorical)
        print("generate attr index done, time:", time.time() - start)


        # generate Counting hash table
        self.index.addEpIds(closest_ids.tolist())
        print("bucket[0:10]:", buckets[0:10])
        self.index.addBuckets(buckets, offsets)
        self.index.generateIdToBucket()
        print("add buckets done, time:", time.time() - start)
        self.index.initCountingHashTable()
        print("init counting hash table done, time:", time.time() - start)
        print("generate filter table done, time:", time.time() - start)

        # add data points into index
        self.index.add_items(base_scalars, levels=layers)
        self.index.generateFT()
        end = time.time()
        print(f"Index built: {name}, duration: {end-start}.")
        self.index.save_index(index_save_path)
        print("index save done, time:", time.time() - end)


        return self.index

    def load_index(self, params, attr_type_list, index_save_path, threads: int, name: str = "HNSW"):
        self.index_method = name
        print("name:", name)

        print(f"Loading index: {name}...")
        self.index = hashannlib.Index(space=params["metric"], dim=params["dim"])
        self.index.set_num_threads(threads)
        start = time.time()
        assert(attr_type_list is not None)
        self.index.load_index(index_save_path)
        print("index loaded")
        self.index.generateAttrIndexes() # B+ tree (numerical) and inverted list (categorical)
        self.index.generateIdToBucket()
        end = time.time()

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
        seed = 1234
        centroid_size = int(np.sqrt(self.N)) # * 10
        # centroid_size = int(np.sqrt(self.N) / 100)
        print("d:", self.d, "N:", self.N, "centroid_size:", centroid_size)
        clustering = faiss.Clustering(self.d, centroid_size)
        clustering.seed = seed
        clustering.niter = max(20, int(np.log2(self.N)) * 2)
        # clustering.max_points_per_centroid = int(np.sqrt(self.N))

        print(f"Clustering parameters: niter={clustering.niter}, max_points_per_centroid={clustering.max_points_per_centroid}")

        index = faiss.IndexFlatL2(self.d)  # For assignment
        clustering.train(base_scalars, index)

        centroids = faiss.vector_to_array(clustering.centroids).reshape(centroid_size, self.d)
        print(f"Clustering done, centroids shape: {centroids.shape}")
        # print("centroids:", centroids)

        # -----------------------------
        # Step 3: Assign all points to clusters
        # -----------------------------
        index = faiss.IndexFlatL2(self.d)
        index.add(centroids)

        # Find nearest centroid for each point
        D, I = index.search(base_scalars, 1)  # I: cluster id for each point
        D = D.reshape(-1)
        I = I.reshape(-1)

        # -----------------------------
        # Step 4: Find closest vector to each centroid
        # -----------------------------
        closest_ids = np.full(centroid_size, -1, dtype=int)
        best_dist = np.full(centroid_size, float('inf'))
        layers = np.zeros(self.N, dtype=int)  # default layer 1

        for i in range(self.N):
            cluster_id = I[i]
            dist = D[i]
            if dist < best_dist[cluster_id]:
                best_dist[cluster_id] = dist
                closest_ids[cluster_id] = i

        # -----------------------------
        # Step 5: Map closest_ids back to the original indices
        # -----------------------------
        closest_ids = closest_ids
        best_dist = best_dist
        layers[closest_ids] = 1

        buckets = np.zeros(self.N, dtype=int)
        offsets = np.zeros(centroid_size + 1, dtype=int)

        print("ep id len:", len(closest_ids))

        if centroid_size <= 0:
            print("error: centroid size <= 0")
            exit(-1)
        start = 0
        end = 0
        offsets[0] = 0
        for i in range(centroid_size):
            selected = np.where(I == i)[0]
            size = selected.shape[0]
            end += size
            buckets[start:end] = selected
            start = end
            offsets[i + 1] = end

        return centroid_size, layers, closest_ids, best_dist, buckets, offsets
