# HashANN 源代码函数级参考文档

> 自动生成于 2026-04-27，供后续修改时参考。

## 目录

- [零、核心创新（相比原版 hnswlib）](#零核心创新相比原版-hnswlib)
- [一、项目结构概览](#一项目结构概览)
- [二、C++ 核心库 (hnswlib/hnswlib/)](#二c-核心库-hnswlibhnswlib)
- [三、Python 绑定 (hnswlib/python_bindings/)](#三python-绑定-hnswlibpython_bindings)
- [四、Python 工具脚本 (tests/)](#四python-工具脚本-tests)
- [五、Shell 脚本 (exp2/exp3/exp4/)](#五shell-脚本-exp2exp3exp4)

---

## 零、核心创新（相比原版 hnswlib）

HashANN 基于 hnswlib (HNSW) 做了属性感知的改造，核心创新有两点，二者相辅相成：

### 1. 边级 Bloom Filter 结构（Neighbor Filtering Table）

每条 HNSW 边附带一个 bitmap（`ft_bits_` 位，默认 128），类似 Bloom Filter，编码了**沿该边方向可达的属性值**。

- **搜索时**：filtered query 通过扫描边的 FT 即可快速判断该方向是否存在满足属性约束的节点，无需实际访问邻居，大幅减少无效的距离计算。
- **实现**：`nbr_ft_check()` / `batched_nbr_ft_check_sse()` 在 `hybridSearchBaseLayerST()` 中对每条边做 bitmap AND 检查（含 SSE 加速）。
- **数据布局**：FT 内联存储在节点的 level-0 数据中（`offsetNbrFt_`），每个邻居占 `ft_bytes_ × attr_count` 字节。

### 2. Prune 时的属性聚合

在 `getNeighborsByHeuristic2` 做邻居裁剪时，被 prune 掉的邻居的属性信息**不是丢弃，而是 OR 合并到支配者（dominator）的 FT 中**（`update_nbr_ft()`）。

- **效果**：虽然边被裁剪了，但被裁剪节点的属性信息仍然保留在图结构中，增强了图在**属性空间上的感知能力**，避免 filtered search 时出现"属性盲区"。
- **实现**：`getNeighborsByHeuristic2()` 记录 `dominated_list`，然后 `update_nbr_ft()` 将被支配点的属性通过 `updateft()` / `merge_ft()` 聚合到对应邻居的 FT bitmap 中。

### 两者的关系

FT 提供搜索时的快速过滤能力，prune 时的属性聚合保证 FT 信息的完整性。二者是一体设计。

### 其他辅助改动

| 改动 | 说明 |
|------|------|
| **CHT (Counting Hash Table)** | 构建时统计邻居的属性覆盖度，防止图在某些属性区域度数过低（`cht_low_degree`） |
| **多入口点 + 图分区** | `ep_ids_` + `graph_partition()` 支持从多个分区入口开始搜索 |
| **Two-hop 扩展** | FT 过滤后邻居不足时，补充未过滤邻居防止搜索卡死（`two_hop_threshold`） |
| **属性索引** | B+ Tree（数值）+ IVF 倒排（分类），用于选择性估算 |
| **属性存储** | 节点内联存储属性值（`offsetAttr_`），支持数值范围查询和分类位图匹配 |
| **SIMD 优化** | FT 检查、属性匹配均有 SSE/AVX 加速版本 |
| **DNF 谓词支持** | `DNFPredicate` + `build_dnf_predicate()` + `nbr_ft_check_dnf()` + `predicate_check_dnf()`，支持任意 AND/OR 组合（见下） |

### 3. DNF 谓词扩展（OR 支持）

原有谓词仅支持跨属性 AND。通过 DNF（析取范式 = OR of ANDs）扩展，支持任意布尔组合。

**核心思路：不修改 FT 存储结构，仅改变查询侧谓词映射。**

#### 双 bitmap 无分支 FT 检查

每个 DNF term 的每个属性预计算两个 bitmap：

| 属性约束 | `or_bitmap` | `and_bitmap` | 语义 |
|----------|-------------|--------------|------|
| 数值范围 | range bits | 全零 | 存在性检查 `(ft & pred) != 0` |
| 分类 AND | 全一 | label bits | 超集检查 `(~ft & pred) == 0` |
| 分类 OR  | label bits | 全零 | 存在性检查 `(ft & pred) != 0` |
| 无约束   | `attr_active=0`，跳过 | — | — |

统一检查逻辑（无 `attr_type_` 分支）：
```cpp
// 两个 SSE 操作并行执行
or_result    = _mm_and_si128(ft, or_bitmap);    // 存在性
and_missing  = _mm_andnot_si128(ft, and_bitmap); // 超集缺失
pass = (or_result != 0) && (and_missing == 0);
```

#### 同属性 OR 优化

数值多范围 OR（如 `attr ∈ [1,5] OR attr ∈ [10,20]`）的 FT bitmap 直接 `|` 合并为单个 bitmap，**零额外开销**。

#### 关键函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `DNFPredicate` 结构体 | ~line 160 | 预计算的 DNF 谓词，扁平连续数组 |
| `build_dnf_predicate()` | ~line 1938 | 从用户格式构建 DNFPredicate |
| `build_single_predicate()` | ~line 2060 | 兼容：单个 AND term → DNF |
| `nbr_ft_check_dnf()` | ~line 1759 | 双 bitmap 无分支 FT 检查 |
| `predicate_check_dnf()` | ~line 1310 | 精确属性检查（支持分类 OR + 多范围） |
| `hybridSearchDNF()` | ~line 5148 | DNF 查询入口，构建 DNFPredicate 后调用 hybridSearchBaseLayerST |

**Python 绑定**: `hybrid_knn_query_dnf(data, dnf_predicate, check_modes=[], k=1, num_threads=-1)`

**用法示例**（Python 端）:
```python
# 单查询 DNF: (attr0 in [1,5] AND attr1 == 3) OR (attr0 in [10,20])
dnf_pred = [
    [[1, 5], [3]],        # term 0: attr0=[1,5], attr1=[3]
    [[10, 20], []],        # term 1: attr0=[10,20], attr1=don't care
]
# 批量: dnf_predicates[i] = per-query DNF
labels, dists = index.hybrid_knn_query_dnf(queries, [dnf_pred] * nq, k=10)
```

---

## 一、项目结构概览

```
hashann/
├── hnswlib/                    # 核心 C++ 库 + Python 绑定
│   ├── hnswlib/                # C++ 头文件 (header-only library)
│   │   ├── hnswlib.h           # 主头文件：接口定义、CPU检测、距离空间
│   │   ├── hnswalg.h           # HNSW 算法核心实现
│   │   ├── core.hpp/cpp        # HashANN 核心：属性索引、FT过滤
│   │   ├── nswalg.h            # NSW (非层级) 算法实现
│   │   ├── dijknsw.h           # Dijkstra-NSW 变体
│   │   ├── ivfalg.h            # IVF (倒排文件) 实现
│   │   ├── bruteforce.h        # 暴力搜索基线
│   │   ├── space_l2.h          # L2 距离实现 (含 SIMD 优化)
│   │   ├── space_ip.h          # 内积距离实现 (含 SIMD 优化)
│   │   ├── stop_condition.h    # 搜索停止条件
│   │   ├── visited_list_pool.h # 访问列表对象池
│   │   ├── btree.hpp           # B树实现 (属性索引)
│   │   └── btree_map.hpp       # B树 Map 封装
│   ├── python_bindings/        # pybind11 绑定
│   │   ├── bindings.cpp        # Index/NSWIndex/BFIndex Python API
│   │   └── LazyIndex.py        # 延迟初始化包装器
│   └── setup.py                # 构建配置 (hashannlib)
├── tests/                      # Python 工具脚本
│   ├── hashann.py              # HashANN 索引封装类
│   ├── hashann_build.py        # 构建索引 CLI
│   ├── hashann_query.py        # 查询索引 CLI
│   ├── utils.py                # 通用工具函数
│   ├── extract_results.py      # 日志结果提取
│   ├── attr_generator.py       # 属性数据生成
│   ├── predicate_generator.py  # 查询谓词生成
│   ├── groundtruth_generator.py# Ground Truth 生成
│   ├── selectivity.py          # 选择性计算
│   ├── milvus_hnsw_index.py    # Milvus HNSW 对比
│   ├── msvbase.py              # MSVBASE 对比
│   └── ...                     # 更多工具/绘图脚本
└── exp2/exp3/exp4/             # 实验目录
    ├── conf.sh                 # 实验配置 (数据集/参数)
    ├── *_index.sh              # 索引构建脚本
    ├── *_query.sh              # 查询脚本
    └── exps.sh/exp_query.sh    # 实验编排脚本
```

---

## 二、C++ 核心库 (hnswlib/hnswlib/)

---

## hnswlib.h

**Purpose**: Main header file providing core interfaces and utility functions for the HNSW library. Contains CPU capability detection, base classes for distance functions, filtering, search stop conditions, and type definitions.

### Macros & Configuration
- `HNSWERR`: Error stream output (default: std::cerr)
- `USE_SSE`, `USE_AVX`, `USE_AVX512`: SIMD feature detection macros
- `PORTABLE_ALIGN32`, `PORTABLE_ALIGN64`: Alignment macros for different compilers
- `AVXCapable()`: Detects CPU AVX capability
- `AVX512Capable()`: Detects CPU AVX-512 capability

### Classes
- **`BaseFilterFunctor`**: Abstract base class for filtering query results
  - `operator()(labeltype id)`: Returns true if element should be included
  
- **`BaseSearchStopCondition<dist_t>`**: Template abstract class for search termination control
  - `add_point_to_result(labeltype, const void*, dist_t)`: Record found point
  - `remove_point_from_result(labeltype, const void*, dist_t)`: Remove found point
  - `should_stop_search(dist_t, dist_t)`: Determine if search should terminate
  - `should_consider_candidate(dist_t, dist_t)`: Check if candidate is worth exploring
  - `should_remove_extra()`: Check if extra results need removal
  - `filter_results(vector<pair>)`: Final filtering of results

- **`pairGreater<T>`**: Comparator for pairs (first element descending)

- **`SpaceInterface<MTYPE>`**: Abstract base for distance metric spaces
  - `get_data_size()`: Returns size of data elements
  - `get_dist_func()`: Returns distance function pointer
  - `get_dist_func_param()`: Returns distance function parameters

- **`AlgorithmInterface<dist_t>`**: Abstract base for index algorithms
  - `addPoint(const void*, labeltype, bool, int)`: Insert point with label
  - `searchKnn(const void*, size_t, BaseFilterFunctor*)`: Search k nearest neighbors
  - `searchKnnCloserFirst(const void*, size_t, BaseFilterFunctor*)`: Search k NN (closer first order)
  - `saveIndex(const string&)`: Persist index to file

### Standalone Functions
- `writeBinaryPOD<T>(ostream, const T&)`: Write binary POD to stream
- `readBinaryPOD<T>(istream, T&)`: Read binary POD from stream

### Type Definitions
- `labeltype`: size_t (unique label for data points)
- `DISTFUNC<MTYPE>`: Function pointer type for distance functions

---

## hnswalg.h

**Purpose**: Implements the Hierarchical Navigable Small World (HNSW) algorithm with support for attributes, filtering tables, and advanced search techniques including bucket-based and two-hop expansion strategies.

### Classes

- **`Key`**: Structure for B-tree key (attribute value + ID)
  - `attr`: attribute value
  - `id`: tableint (internal ID)

- **`KeyCompare`**: Comparator for Key objects (lexicographic ordering)

- **`HierarchicalNSW<dist_t>`**: Main HNSW implementation (extends AlgorithmInterface)
  
  **Key Members**:
  - `max_elements_`, `cur_element_count`: Capacity and size tracking
  - `M_`, `maxM_`, `maxM0_`: Connectivity parameters
  - `ef_construction_`, `ef_`, `ef_top_`: Search parameter (ef = ef_construction, ef_, ef_top_)
  - `element_levels_`: Hierarchical level of each element
  - `visited_list_pool_`: Pool for visited tracking lists
  - `enterpoint_node_`: Entry point for search
  - `data_level0_memory_`: Memory for level-0 nodes
  - `linkLists_`: Links for higher levels
  - `btrees`: B+ tree indexes for numerical attributes
  - `ivf`: Inverted file indexes for categorical attributes
  - `bucket_data_`, `bucket_offsets_`, `id_to_buckets_`: Bucket structure for clustering
  - `counting_hash_table`: Hash table for attribute value counting
  
  **Key Methods**:
  - `HierarchicalNSW(SpaceInterface*, size_t, size_t, size_t, size_t, size_t, size_t, vector, size_t, bool)`: Constructor
  - `addPoint(const void*, labeltype, bool, int)`: Insert point
  - `deletePoint(labeltype)`: Remove point by label
  - `searchKnn(const void*, size_t, BaseFilterFunctor*)`: K-NN search
  - `searchKnnBucketed(const void*, size_t, BaseFilterFunctor*)`: K-NN with bucket filtering
  - `searchKnnWithPredicate(const void*, size_t, vector, BaseFilterFunctor*)`: Search with attribute predicates
  - `saveIndex(const string&)`: Save to file
  - `loadIndex(const string&, SpaceInterface*)`: Load from file
  - `set_ft_flag(bool)`: Enable/disable filtering table
  - `set_two_hop_flag(bool)`: Enable/disable two-hop expansion
  - `add_ep_ids(vector)`: Set entry points
  - `set_thresholds(double, double, double)`: Configure search thresholds
  - `add_buckets(const int*, const int*, size_t)`: Load bucket structure
  - `init_counting_hash_table()`: Initialize attribute counting hash table
  - `bucketize_equal_count(vector, int, int)`: Equal-count bucketing
  - `distribute_labels(vector, int, int)`: Distribute categorical labels to buckets
  - `init_attr_mapping(vector)`: Initialize attribute mapping for hash table
  - `update_cht(int*, tableint)`: Update counting hash table entry
  - `generate_attr_indexes()`: Build B+ trees and inverted files
  - `add_attr_to_point(int, vector)`: Add attributes to a point
  - `setEf(size_t)`: Set search parameter ef
  - `setEfTop(size_t)`: Set ef for top-level search

  **Utility Methods**:
  - `getExternalLabel(tableint)`: Get label from internal ID
  - `setExternalLabel(tableint, labeltype)`: Set label for internal ID
  - `getDataByInternalId(tableint)`: Get data pointer
  - `isMarkedDeleted(tableint)`: Check if element marked for deletion
  - `getLevel(tableint)`: Get hierarchical level of element
  - `getLinklist(tableint, int)`: Get neighbor list at level
  - `getLinklist0(tableint)`: Get level-0 neighbor list
  - `lower_bound(const int*, int, int)`: Find lower bound with SSE optimization
  - `last_le_index(const int*, int, int)`: Find last less-than-or-equal index

---

## core.hpp & core.cpp

**Purpose**: Part of tlx library - provides error handling and assertions for the codebase.

### Macros (core.hpp)
- `tlx_die_with_sstream(msg)`: Output error message and terminate
- `tlx_die(msg)`: Simplified die macro
- `tlx_die_unless(X)`: Assert condition is true
- `tlx_die_if(X)`: Assert condition is false
- `tlx_die_verbose_unless(X, msg)`: Assert with custom message
- `tlx_die_verbose_if(X, msg)`: Assert false with message
- `tlx_die_unequal(X, Y)`: Assert equality
- `TLX_BTREE_PRINT(x)`: Debug print (if TLX_BTREE_DEBUG defined)
- `TLX_BTREE_ASSERT(x)`: Debug assertion
- `TLX_BTREE_MAX(a, b)`: Maximum macro

### Functions (core.cpp)
- `die_with_message(const string&)`: Core error handler
- `die_with_message(const char*, const char*, size_t)`: Error handler with file/line
- `set_die_with_exception(bool)`: Toggle exception vs terminate behavior

### Classes (core.hpp)
- **`DieException`**: Runtime error exception thrown by die functions

---

## bruteforce.h

**Purpose**: Linear search index - compares query against all points without hierarchical structure.

### Classes

- **`BruteforceSearch<dist_t>`**: Brute-force nearest neighbor search (extends AlgorithmInterface)
  
  **Members**:
  - `data_`: Pointer to all vectors and labels
  - `maxelements_`, `cur_element_count`: Capacity and count
  - `size_per_element_`: Bytes per element (data + label)
  - `fstdistfunc_`: Distance function pointer
  - `dict_external_to_internal`: Label to internal ID mapping
  - `index_lock`: Mutex for thread safety
  
  **Methods**:
  - `BruteforceSearch(SpaceInterface*)`: Default constructor
  - `BruteforceSearch(SpaceInterface*, const string&)`: Load from file
  - `BruteforceSearch(SpaceInterface*, size_t)`: Create with capacity
  - `addPoint(const void*, labeltype, bool, int)`: Insert point
  - `removePoint(labeltype)`: Delete point by label
  - `searchKnn(const void*, size_t, BaseFilterFunctor*)`: K-NN search (scans all)
  - `saveIndex(const string&)`: Persist to file
  - `loadIndex(const string&, SpaceInterface*)`: Load from file

---

## nswalg.h

**Purpose**: Navigable Small World (NSW) algorithm - flat graph-based index with multiple entry points and clustering support.

### Classes

- **`NSW<dist_t>`**: NSW algorithm implementation (extends AlgorithmInterface)
  
  **Members**:
  - `max_elements_`, `cur_element_count`: Capacity and size
  - `M_`, `maxM_`, `maxM0_`: Connectivity parameters
  - `ef_construction_`, `ef_`: Search parameters
  - `visited_list_pool_`: Visited tracking pool
  - `enterpoint_nodes_`, `enterpoint_size_`: Multiple entry points
  - `label_lookup_`: Map from label to internal ID
  - `deleted_elements`: Set of deleted internal IDs
  - `allow_replace_deleted_`: Flag to reuse deleted slots
  - `metric_distance_computations`, `metric_hops`: Performance metrics
  
  **Methods**:
  - `NSW(SpaceInterface*)`: Default constructor
  - `NSW(SpaceInterface*, const string&, bool, size_t, bool)`: Load from file
  - `NSW(SpaceInterface*, size_t, size_t, size_t, size_t, bool)`: Create with params
  - `addPoint(const void*, labeltype, bool, int)`: Insert point
  - `deletePoint(labeltype)`: Remove point
  - `searchKnn(const void*, size_t, BaseFilterFunctor*)`: K-NN search
  - `searchBaseLayer(vector, const void*)`: Search on flat layer
  - `searchBaseLayerST(vector, const void*, size_t, BaseFilterFunctor*, BaseSearchStopCondition*)`: Single-threaded search with stop condition
  - `setEf(size_t)`: Set ef parameter
  - `saveIndex(const string&)`: Save index
  - `loadIndex(const string&, SpaceInterface*, size_t)`: Load index

  **Utility Methods**:
  - `getExternalLabel(tableint)`: Get label from ID
  - `setExternalLabel(tableint, labeltype)`: Set label
  - `getDataByInternalId(tableint)`: Get data pointer
  - `isMarkedDeleted(tableint)`: Check deletion status
  - `getCurrentElementCount()`: Get current count
  - `getDeletedCount()`: Get deleted count

---

## dijknsw.h

**Purpose**: Dijkstra-based graph building for incremental range construction with multi-threaded processing and heuristic-based neighbor pruning.

### Classes

- **`dijknsw_build<dist_t>`**: Graph construction using Dijkstra on incomplete graph
  
  **Members**:
  - `max_threads`: Thread pool size
  - `tree`: Segment tree for hierarchical decomposition
  - `storage`: Data loader/storage
  - `edges`: Edge lists per layer
  - `reverse_edges`: Reverse edge tracking
  - `M`, `ef_construction`: Parameters
  - `space`, `fstdistfunc_`: Distance metric
  - `visitedpool`, `visited_tag`: Visited tracking
  
  **Methods**:
  - `iRangeGraph_Build(DataLoader*, int, int)`: Constructor
  - `dis_compute(vector, vector)`: Compute distance between vectors
  - `copyfirstchild(TreeNode*)`: Copy edges from first child node
  - `search_on_incomplete_graph(TreeNode*, vector, int, int, vector)`: Search on partial graph with entry points
  - `PruneByHeuristic2(vector, vector)`: Neighbor selection heuristic
  - `process_node(TreeNode*)`: Build edges for tree node
  - `buildindex()`: Build entire graph with multi-threading
  - `buildandsave(string)`: Build and save to file

---

## ivfalg.h

**Purpose**: Inverted File (IVF) based indexing for large-scale similarity search using clustering and product quantization.

### Classes

- **`Index`**: Base index interface for search/retrieval
  
  **Members**:
  - `d`: Vector dimension
  - `ntotal`: Total indexed vectors
  - `verbose`: Verbosity flag
  - `is_trained`: Training status
  - `metric_type`: Distance metric (L2, IP)
  - `metric_arg`: Metric argument
  
  **Methods**:
  - `Index(idx_t, MetricType)`: Constructor
  - `train(idx_t, const float*)`: Train on sample data
  - `add(idx_t, const float*)`: Add vectors (pure virtual)
  - `reset()`: Clear index (pure virtual)
  - `sa_code_size() const`: Standalone codec size
  - `sa_encode(idx_t, const float*, uint8_t*)`: Encode vectors
  - `sa_decode(idx_t, const uint8_t*, float*)`: Decode vectors
  - `assign(idx_t, const float*, idx_t*, idx_t)`: Assign vectors to clusters
  - `compute_residual(const float*, float*, idx_t)`: Compute residual from reconstruction
  - `compute_residual_n(idx_t, const float*, float*, const idx_t*)`: Batch residual computation
  - `merge_from(Index&, idx_t)`: Merge another index
  - `check_compatible_for_merge(const Index&) const`: Verify compatibility

- **`Level1Quantizer`**: First-level quantizer for IVF (maps to clusters)
  
  **Members**:
  - `quantizer`: Index for clustering
  - `nlist`: Number of inverted lists
  - `clustering_index`: Custom clustering index
  
  **Methods**:
  - `train_q1(size_t, const float*, bool, DISTFUNC)`: Train quantizer
  - `coarse_code_size() const`: Size of cluster codes
  - `encode_listno(idx_t, uint8_t*)`: Encode list ID
  - `decode_listno(const uint8_t*) const`: Decode list ID

---

## space_l2.h

**Purpose**: Euclidean (L2) distance metric implementation with multiple SIMD optimizations.

### Standalone Functions (Floating Point)
- `L2Sqr(const void*, const void*, const void*)`: Basic L2 distance squared
- `L2SqrSIMD16ExtAVX512(const void*, const void*, const void*)`: AVX-512 optimized (16 floats)
- `L2SqrSIMD16ExtAVX(const void*, const void*, const void*)`: AVX optimized (16 floats)
- `L2SqrSIMD16ExtSSE(const void*, const void*, const void*)`: SSE optimized (16 floats)
- `L2SqrSIMD16Ext`: Function pointer to best SIMD variant
- `L2SqrSIMD16ExtResiduals(const void*, const void*, const void*)`: Handle remainder elements
- `L2SqrSIMD4Ext(const void*, const void*, const void*)`: SSE for 4 floats
- `L2SqrSIMD4ExtResiduals(const void*, const void*, const void*)`: With remainder

### Standalone Functions (Integer)
- `L2SqrI4x(const void*, const void*, const void*)`: Integer L2 (4 at a time)
- `L2SqrI(const void*, const void*, const void*)`: Basic integer L2

### Classes

- **`L2Space`**: L2 metric space (extends SpaceInterface<float>)
  
  **Methods**:
  - `L2Space(size_t dim)`: Initialize with dimension
  - `get_data_size()`: Return data size in bytes
  - `get_dist_func()`: Return distance function
  - `get_dist_func_param()`: Return dimension parameter

- **`L2SpaceI`**: Integer L2 metric space (extends SpaceInterface<int>)
  
  **Methods**:
  - `L2SpaceI(size_t dim)`: Initialize with dimension
  - `get_data_size()`: Return data size
  - `get_dist_func()`: Return integer distance function
  - `get_dist_func_param()`: Return dimension

---

## space_ip.h

**Purpose**: Inner Product (IP) distance metric with SIMD optimizations for different CPU capabilities.

### Standalone Functions (Basic)
- `InnerProduct(const void*, const void*, const void*)`: Compute inner product
- `InnerProductDistance(const void*, const void*, const void*)`: 1.0 - InnerProduct

### SIMD Variants

**AVX-512**:
- `InnerProductSIMD16ExtAVX512(const void*, const void*, const void*)`: 16-element AVX-512
- `InnerProductDistanceSIMD16ExtAVX512(const void*, const void*, const void*)`: With distance

**AVX**:
- `InnerProductSIMD4ExtAVX(const void*, const void*, const void*)`: AVX variants
- `InnerProductSIMD16ExtAVX(const void*, const void*, const void*)`
- `InnerProductDistanceSIMD4ExtAVX(const void*, const void*, const void*)`
- `InnerProductDistanceSIMD16ExtAVX(const void*, const void*, const void*)`

**SSE**:
- `InnerProductSIMD4ExtSSE(const void*, const void*, const void*)`
- `InnerProductSIMD16ExtSSE(const void*, const void*, const void*)`
- `InnerProductDistanceSIMD4ExtSSE(const void*, const void*, const void*)`
- `InnerProductDistanceSIMD16ExtSSE(const void*, const void*, const void*)`

**Residual Handling**:
- `InnerProductDistanceSIMD16ExtResiduals(const void*, const void*, const void*)`
- `InnerProductDistanceSIMD4ExtResiduals(const void*, const void*, const void*)`

### Classes

- **`InnerProductSpace`**: Inner product metric (extends SpaceInterface<float>)
  
  **Methods**:
  - `InnerProductSpace(size_t dim)`: Initialize with dimension
  - `get_data_size()`: Return data size
  - `get_dist_func()`: Return distance function (auto-selects best SIMD)
  - `get_dist_func_param()`: Return dimension parameter

---

## stop_condition.h

**Purpose**: Advanced search termination strategies including document-level and epsilon-based constraints.

### Classes

- **`BaseMultiVectorSpace<DOCIDTYPE>`**: Base for multi-vector spaces with document ID support
  
  **Pure Virtual Methods**:
  - `get_doc_id(const void*)`: Extract document ID from data
  - `set_doc_id(void*, DOCIDTYPE)`: Set document ID in data

- **`MultiVectorL2Space<DOCIDTYPE>`**: L2 metric with document ID (extends BaseMultiVectorSpace)
  
  **Methods**:
  - `MultiVectorL2Space(size_t dim)`: Constructor
  - `get_data_size()`: Total size including doc ID
  - `get_dist_func()`: Return L2 distance function
  - `get_dist_func_param()`: Return dimension
  - `get_doc_id(const void*)`: Extract doc ID
  - `set_doc_id(void*, DOCIDTYPE)`: Set doc ID

- **`MultiVectorInnerProductSpace<DOCIDTYPE>`**: IP metric with document ID
  
  **Methods**:
  - `MultiVectorInnerProductSpace(size_t dim)`: Constructor
  - `get_data_size()`: Total size with doc ID
  - `get_dist_func()`: Return IP distance function
  - `get_dist_func_param()`: Return dimension
  - `get_doc_id(const void*)`: Extract doc ID
  - `set_doc_id(void*, DOCIDTYPE)`: Set doc ID

- **`MultiVectorSearchStopCondition<DOCIDTYPE, dist_t>`**: Document-level search termination
  
  **Members**:
  - `curr_num_docs_`: Current unique document count
  - `num_docs_to_search_`: Target document count
  - `doc_counter_`: Track vectors per document
  - `search_results_`: Candidate result queue
  
  **Methods**:
  - `MultiVectorSearchStopCondition(BaseMultiVectorSpace&, size_t, size_t)`: Constructor
  - `add_point_to_result(labeltype, const void*, dist_t)`: Record vector
  - `remove_point_from_result(labeltype, const void*, dist_t)`: Remove vector
  - `should_stop_search(dist_t, dist_t)`: Stop if doc count reached
  - `should_consider_candidate(dist_t, dist_t)`: Evaluate candidate
  - `should_remove_extra()`: Remove excess results
  - `filter_results(vector)`: Keep top documents

- **`EpsilonSearchStopCondition<dist_t>`**: Epsilon-radius search termination
  
  **Members**:
  - `epsilon_`: Search radius threshold
  - `min_num_candidates_`, `max_num_candidates_`: Result bounds
  - `curr_num_items_`: Current result count
  
  **Methods**:
  - `EpsilonSearchStopCondition(float, size_t, size_t)`: Constructor
  - `add_point_to_result(labeltype, const void*, dist_t)`: Add result
  - `remove_point_from_result(labeltype, const void*, dist_t)`: Remove result
  - `should_stop_search(dist_t, dist_t)`: Stop if out of epsilon or max reached
  - `should_consider_candidate(dist_t, dist_t)`: Check candidate
  - `should_remove_extra()`: Remove excess results
  - `filter_results(vector)`: Keep results within epsilon

---

## visited_list_pool.h

**Purpose**: Thread-safe pool management for visited tracking lists used during graph traversal.

### Classes

- **`VisitedList`**: Single visited tracking array
  
  **Members**:
  - `curV`: Current version counter
  - `curV_ft`: Version for filtering table
  - `mass`: Array tracking visit versions
  - `numelements`: Size of mass array
  
  **Methods**:
  - `VisitedList(int numelements1)`: Constructor
  - `reset()`: Increment version counter (memset if overflow)

- **`VisitedListPool`**: Thread-safe pool of visited lists
  
  **Members**:
  - `pool`: Deque of available VisitedList objects
  - `poolguard`: Mutex for thread safety
  - `numelements`: Size of each list
  - `is_two_level`: Flag for hierarchical tracking
  
  **Methods**:
  - `VisitedListPool(int initmaxpools, int numelements1)`: Constructor
  - `getFreeVisitedList()`: Acquire list from pool (allocate if needed)
  - `releaseVisitedList(VisitedList*)`: Return list to pool

---

## btree.hpp

**Purpose**: Generic B+ tree implementation - core data structure for ordered set/map containers.

### Key Structures

- **`btree_default_traits<Key, Value>`**: Configuration for B+ tree behavior
  - `self_verify`: Enable invariant checking
  - `debug`: Enable debug output
  - `leaf_slots`: Keys per leaf (estimated ~256 bytes)
  - `inner_slots`: Keys per inner node
  - `binsearch_threshold`: Use binary search if node > threshold

### Classes

- **`BTree<Key, Value, KeyOfValue, Compare, Traits, Duplicates, Allocator>`**: Main B+ tree implementation
  
  **Type Aliases**:
  - `key_type`: Key template parameter
  - `value_type`: Value template parameter
  - `key_compare`: Comparator class
  - `iterator`: STL-like iterator
  - `const_iterator`: Const iterator
  - `size_type`: Size counter type
  - `tree_stats`: Statistics structure
  
  **Core Operations** (documented in implementation):
  - Insertion with node splitting
  - Deletion with node merging/rebalancing
  - Lower/upper bound searches
  - Iteration (forward/reverse)
  - Tree validation and statistics
  - Bulk loading and serialization

---

## btree_map.hpp

**Purpose**: STL-compatible map container implemented using B+ tree - alternative to std::map.

### Classes

- **`btree_map<Key, Data, Compare, Traits, Allocator>`**: B+ tree based map
  
  **Type Definitions**:
  - `key_type`: Key template
  - `data_type`: Value template
  - `key_compare`: Comparator
  - `value_type`: pair<key_type, data_type>
  - `iterator`: Tree iterator type
  - `const_iterator`: Const iterator
  - `size_type`: Size counter
  
  **Key Constants**:
  - `leaf_slotmax`: Max keys per leaf
  - `inner_slotmax`: Max keys per inner node
  - `leaf_slotmin`: Min keys per leaf (for merging)
  - `inner_slotmin`: Min keys per inner
  - `self_verify`: Verification flag
  - `debug`: Debug flag
  - `allow_duplicates`: Duplicate key handling
  
  **Core STL Methods** (implements standard map interface):
  - Constructor variants (default, copy, move)
  - `insert(key, value)`: Add element
  - `erase(key)`: Remove by key
  - `find(key)`: Locate element
  - `lower_bound(key)`: First >= key
  - `upper_bound(key)`: First > key
  - `begin()`, `end()`: Iterator access
  - `size()`, `empty()`: Size queries
  - `operator[]`: Element access
  - Comparison operators

---

This comprehensive analysis covers all public/important interfaces, methods, and data structures across the 14 files in the HNSW library, organized by file with clear descriptions of their purposes and functionality.

---


## 三、Python 绑定 (hnswlib/python_bindings/)

### bindings.cpp
通过 pybind11 暴露三个主要索引类到 Python。

#### Index<float> — HNSW 索引 (主要类)
- `__init__(space, dim)` / `__init__(params)` — 构造
- `init_index(max_elements, top_elements, M=16, ef_construction=200, ft_bits=128, attr_type, max_cate_size=5)` — 初始化
- `add_items(data, data_attr, ids, num_threads, replace_deleted, levels)` — 添加向量+属性
- `knn_query(data, k, num_threads, filter)` → (labels, distances) — KNN搜索
- `hybrid_knn_query(data, predicate, k, num_threads, filter)` → (labels, distances) — 混合属性搜索
- `knn_query_with_stats(data, k, filter)` → (labels, distances, dist_comps, hops) — 带统计搜索
- `hybrid_knn_query_with_stats(data, predicate, k, filter)` → 带统计混合搜索
- `initAttrSpace()` / `attrCheck()` / `addBuckets()` / `generateAttrIndexes()` — 属性空间管理
- `graphPartition()` — 图分区
- `set_ft_flag(bool)` / `set_thresholds(t1,t2,t3)` — Fast-Thinking 控制
- `set_two_hop_flag(bool)` / `set_two_hop_threshold(val)` — 两跳搜索
- `predicateTranslate(predicate)` / `predicateToFT(predicate)` — 谓词转换
- `save_index(path)` / `load_index(path, max_elements, top_elements)` — 持久化
- 属性: `space`, `dim`, `max_elements`, `element_count`, `ef`, `M`, `ef_construction`, `num_threads`

#### NSWIndex<float> — NSW 索引
- API 与 Index 类似，但无层级结构和属性功能
- `init_index(max_elements, M, ef_construction)` — 无 top_elements/ft_bits

#### BFIndex<float> — 暴力搜索索引
- `init_index(max_elements)` / `add_items(data, ids)` / `knn_query(data, k)` / `delete_vector(label)`

### LazyIndex.py
延迟初始化的 Index 包装器，首次 `add_items` 时才真正初始化索引。

---

## 四、Python 工具脚本 (tests/)

---

## hashann.py
Defines the HashANN index wrapper class for building and querying HNSW-based indexes with attribute filtering support.

### Classes
- `HashANN`: Main index class for HNSW-based filtering search
  - `__init__()`: Initialize index with default parameters (k=10, threads=64, metric="l2", etc.)
  - `init_params(params)`: Set index parameters from config dict (k, threads, index_method, dim, N, metric)
  - `build_index(params, base_scalars, attr, attr_type_list, index_save_path, threads, name)`: Build and save HNSW index with clustering and attribute mapping
  - `load_index(params, attr_type_list, index_save_path, threads, name)`: Load pre-built index and generate attribute indexes
  - `add_attr(attr, attr_type)`: Add attributes to index
  - `predicate_translate(predicate)`: Translate predicate to internal format
  - `hybrid_search(query, predicate)`: Execute kNN query with attribute filtering
  - `clustering(base_scalars)`: Perform k-means clustering using FAISS for data partitioning (caches result to .npz file)

---

## hashann_build.py
Command-line script to build HashANN indexes with attribute support.

### Functions
- `if __name__ == "__main__"`: Main entry point that initializes HashANN, loads data, and builds index

### CLI Arguments (via `arg_init()` from utils)
- `--data_path`: Path to fvecs base data file (required)
- `--query_path`: Path to fvecs query data file (required)
- `--attr_path`: Path to JSON attribute file (required)
- `--qrange_path`: Path to JSON query attribute range file (required)
- `--gt_path`: Path to JSON ground truth file (required)
- `--index_cache_path`: Path to save built index (required)
- `--N`: Number of base vectors (required)
- `--dim`: Vector dimension (required)
- `--threads`: Build thread count (default: 1, required)
- `--name`: Index name (default: "HNSW")
- `--M`: Number of graph connections (default: 16)
- `--efConstruction`: Construction ef parameter (default: 500)
- `--metric`: Distance metric "l2" or "ip" (default: "l2")
- `--attr_type_list`: List of attribute types [0=numerical, 1=categorical] (required)
- `--ft_bits`: Filter table bits (default: 128)
- `--n_query_to_use`: Number of queries to use (default: 1000)
- `--K`: Top-K value (default: 10)
- `--ef_search`: EF search parameters (default: "[100]")
- `--ef_top`: EF top (default: 10)
- `--use_ft`: Use filter table (default: "true")

---

## hashann_query.py
Command-line script to query HashANN indexes with attribute filtering and performance measurement.

### Functions
- `load_query_data(query_file, qrange_file, gt_file, N, Nq, k)`: Load and validate query data, predicates, and ground truth from files

### Main Execution
- Loads HashANN index, executes queries with varying ef_search parameters
- Measures QPS, distance comparisons (cmps), and recall
- Supports optional statistics API if available
- Prints final results as [ef_search, recall, QPS] or [ef_search, recall, QPS, cmps]

---

## hnsw_build.py
Command-line script to build standard HNSW indexes using hnswlib without attribute filtering.

### Functions
- `arg_init()`: Parse command-line arguments
- `load_base_data(data_path, expected_count, expected_dim)`: Load and validate fvecs base data
- `build_index(args)`: Build HNSW index and save to disk

### CLI Arguments
- `--data_path`: Path to fvecs base data file (required)
- `--index_cache_path`: Path to save built index (required)
- `--N`: Number of base vectors (required)
- `--dim`: Vector dimension (required)
- `--M`: Number of graph connections (default: 16)
- `--efConstruction`: Construction ef parameter (default: 500)
- `--metric`: Distance metric: "l2", "ip", "cosine" (default: "l2")
- `--threads`: Build thread count (default: 1)
- `--name`: Index name for compatibility (default: "HNSW")

---

## hnsw_query.py
Query script for standard HNSW indexes with Python-level predicate filtering.

### Functions
- `_normalize_numeric(value)`: Extract numeric value from list or scalar
- `_normalize_categorical(value)`: Convert scalar/None to list for categorical attributes
- `_match_predicate(attr_row, predicate_row, attr_type_list)`: Check if data point satisfies predicates
- `_load_query_data(query_file, qrange_file, gt_file, nq, k)`: Load query files with validation
- `main()`: Execute filtered kNN queries with varying ef_search values

### Main Features
- Uses hnswlib's filter callback for hybrid search
- Measures recall, QPS, and supports multiple ef_search values
- Python filter callbacks execute in single-threaded mode for correctness

---

## hnsw_query_navix.py
Query script for HNSW indexes using "navix" query mode (enhanced filtering algorithm).

### Functions
Same as `hnsw_query.py` (helpers: `_normalize_numeric`, `_normalize_categorical`, `_match_predicate`, `_load_query_data`, `main`)

### Key Difference
- Uses `index.knn_query_navix()` instead of standard `knn_query()`
- Applies filter at index boundary value comparison instead of after nearest neighbor search

---

## utils.py
Utility module providing common functions for data I/O and argument parsing.

### Functions
- `ivecs_read(fname)`: Read binary ivecs/fvecs files
- `fvecs_read(fname)`: Read binary fvecs files (calls ivecs_read with float32 conversion)
- `read_attr(fname)`: Read JSON attribute file and return as numpy array (int64)
- `read_multy_attr(fname)`: Read JSON file with multiple attributes, return as list
- `load_data(dataset_file, query_file, attr_file, qrange_file, gt_file, N, Nq, k)`: Load all dataset files with validation
- `check_dir(file_dir)`: Verify directory exists
- `check_file(f)`: Verify file exists
- `arg_init()`: Parse standard command-line arguments for index/query scripts

### CLI Arguments (arg_init)
- `--K` / `--k`: Top-K value for kNN (default: 10)
- `--name`: Index method "HNSW" or "NSW" (required)
- `--n_query_to_use`: Number of queries (default: 1000)
- `--data_path`: Path to fvecs data file (required)
- `--query_path`: Path to fvecs query file (required)
- `--attr_path`: Path to JSON attribute file (required)
- `--qrange_path`: Path to JSON query range file (required)
- `--gt_path`: Path to JSON ground truth file (required)
- `--M`: Number of graph connections (default: 16)
- `--ft_bits`: Filter table bits (default: 128)
- `--efConstruction`: HNSW construction parameter (default: 500)
- `--attr_type_list`: Attribute types [0=numerical, 1=categorical] (required)
- `--index_cache_path`: Index file path (required)
- `--metric`: Distance metric "l2" or "cosine" (default: "l2")
- `--N`: Number of data points (default: 0)
- `--threads`: Thread count (default: 1, required)
- `--dim`: Vector dimension (default: 1, required)
- `--ef_search`: EF search values (default: "[100]")
- `--ef_top`: EF top value (default: 10)
- `--use_ft`: Use filter table "true"/"false" (default: "true")

---

## extract_results.py
Parses log files from query execution and extracts performance results.

### Functions
- `_extract_result_rows(section_lines)`: Extract result rows in format [efs, recall, qps] or [efs, recall, qps, cmps]

### Main Features
- Parses log files for search results
- Handles both normalized summary rows and raw tail formats
- Special handling for DiskANN format
- Appends parsed results to logs.txt with file locking

### Command-line Arguments
Takes 7 or 8 positional arguments:
- `sys.argv[1]`: Log file path
- `sys.argv[2]`: Dataset name
- `sys.argv[3]`: Attribute type
- `sys.argv[4]`: Query selectivity
- `sys.argv[5]`: M parameter
- `sys.argv[6]`: Algorithm name
- `sys.argv[7]` (optional): K value (default: 10)

---

## attr_generator.py
Generate synthetic attributes for data points with Zipfian distribution.

### Functions
- `arg_init()`: Parse command-line arguments
- `zipfProb(N=5, s=1.0)`: Calculate Zipf probabilities for N categories with exponent s
- `assign_labels(probs, num_items)`: Assign labels to items following Zipf distribution

### CLI Arguments
- `--output_file`: Output attribute file path (required)
- `--attr_type_list`: Attribute types [0=numerical, 1=categorical] (required)
- `--N`: Number of data points (required)
- `--numerical_max_attr`: Max value for numerical attributes (default: 100000)
- `--categorical_attr_max_cardinality`: Max cardinality for categorical attributes (default: 5)

---

## predicate_generator.py
Generate query predicates with specified selectivity levels.

### Functions
- `arg_init()`: Parse command-line arguments
- `zipfProb(N=5, s=1.0)`: Calculate Zipf probabilities
- `assign_labels(probs, num_items)`: Assign labels following Zipf distribution
- `generate_query_selectivity(attr_type_list, query_sel)`: Convert selectivity spec to predicate format

### CLI Arguments
- `--attr_file`: Input attribute file path (required)
- `--attr_type_list`: Attribute types [0=numerical, 1=categorical] (required)
- `--N`: Number of data points (required)
- `--numerical_max_attr`: Max value for numerical attributes (default: 100000)
- `--categorical_attr_max_cardinality`: Max cardinality (default: 5)
- `--query_sel`: Selectivity for each attribute e.g., "[0.5,0.5]" (required)
- `--query_size`: Number of queries to generate (default: 100, required)
- `--predicate_file`: Output predicate file path (required)

---

## groundtruth_generator.py
Generate ground truth results by querying vector data with attribute filters using Milvus.

### Functions
- `read_data(data_file, attr_file, N, d, query_file, predicate_file, Nq)`: Load vectors, attributes, queries, predicates
- `arg_init()`: Parse command-line arguments

### CLI Arguments
- `--dataset_file`: Dataset file path (required)
- `--d`: Dimension of data (required)
- `--attr_file`: Attribute file path (required)
- `--attr_type_list`: Attribute types (required)
- `--N`: Number of data points (required)
- `--query_size`: Number of queries (default: 100, required)
- `--predicate_file`: Predicate file path (required)
- `--query_file`: Query file path (required)
- `--c_name`: Collection name (optional)
- `--mode`: "query" or "construction" (default: "query")
- `--max_cate_val`: Max cardinality (default: 5)
- `--K`: Top-K value (default: 10)
- `--gt_file`: Output ground truth file (required)
- `--metric`: Distance metric "L2" or "IP" (default: "L2")

---

## selectivity.py
Compute query selectivity (percentage of data points matching predicates).

### Functions
- `read_data(attr_file, N, d, predicate_file, Nq)`: Load attributes and predicates
- `arg_init()`: Parse command-line arguments

### Main Features
- Counts data points matching each query's predicate
- Computes overall selectivity and distribution histogram (10 bins)
- Prints selectivity bins: [0-0.1], [0.1-0.2], ..., [0.9-1.0]

### CLI Arguments
- `--d`: Dimension of data (required)
- `--attr_file`: Attribute file path (required)
- `--attr_type_list`: Attribute types (required)
- `--N`: Number of data points (required)
- `--query_size`: Number of queries (default: 100, required)
- `--predicate_file`: Predicate file path (required)

---

## milvus_hnsw_index.py
Build and query Milvus HNSW indexes with attribute filtering.

### Functions
- `read_data(data_file, attr_file, N, d, query_file, predicate_file, Nq)`: Load all data files
- `arg_init()`: Parse command-line arguments
- `drop_index(client, col_name, index_name)`: Drop index and release collection
- `check_size(client, col_name, expected_size)`: Verify collection size matches expected
- `generate_sql(schema_name, table_name, query_vector, raw_predicate, K, attr_type_list, metric)`: (msvbase.py) Generate filtered vector search SQL

### CLI Arguments
- `--dataset_file`: Dataset file path (required)
- `--d`: Dimension (required)
- `--attr_file`: Attribute file path (required)
- `--attr_type_list`: Attribute types (required)
- `--N`: Number of data points (required)
- `--query_size`: Number of queries (default: 100, required)
- `--predicate_file`: Predicate file path (required)
- `--query_file`: Query file path (required)
- `--c_name`: Collection name (optional)
- `--mode`: "construction" or "query" (default: "query")
- `--max_cate_val`: Max cardinality (default: 5)
- `--K`: Top-K value (default: 10)
- `--gt_file`: Ground truth file (required)
- `--metric`: Distance metric "L2" or "IP" (default: "L2")
- `--ef_construction`: EF construction parameter (required)
- `--M`: M parameter (required)
- `--ef_search`: EF search list (required)

### Main Features
- Supports modes: "construction" (index build) and "query"
- Creates HNSW indexes on Milvus with partitioning
- Executes filtered kNN queries with result verification
- Supports array-type categorical attributes

---

## milvus_hnsw_index_bulk.py
Build and query Milvus HNSW indexes with bulk data import support.

### Functions
Same structure as `milvus_hnsw_index.py` with bulk data handling

### Key Differences
- Saves bulk data to separate files for faster import
- Generates numpy arrays for bulk insertion
- More efficient for large datasets

---

## milvus_to_npy.py
Convert Milvus/Milvus_hnsw_index_bulk format data to numpy arrays.

### Main Features
- Reads dataset, IDs, and attributes from Milvus format
- Creates 5 partition directories with numpy arrays
- Saves vector.npy, id.npy, and attr_*.npy files per partition

---

## msvbase.py
PostgreSQL VectorDB integration for filtered vector search (DiskANN-like).

### Functions
- `load_query_data(query_file, qrange_file, gt_file, N, Nq, k)`: Load query data
- `read_data(data_file, attr_file, N, d, query_file, predicate_file, Nq)`: Load all data
- `arg_init()`: Parse command-line arguments
- `create_schema_if_not_exists(cur, conn, schema_name)`: Create PostgreSQL schema
- `create_table_file(vector_data, attr_list, attr_type_list, table_file)`: Generate TSV data file
- `table_exists(cur, schema_name, table_name)`: Check if table exists
- `index_exists(cur, schema_name, table_name)`: Check if HNSW index exists
- `create_table_if_not_exists(cur, conn, schema_name, table_name, d, attr_type_list)`: Create HNSW-indexed table
- `delete_table_if_exists(cur, conn, schema_name, table_name)`: Drop table
- `insert_data(cur, conn, schema_name, table_name, dataset, attr, attr_type_list, table_file)`: Bulk insert data
- `delete_index_if_exists(cur, conn, schema_name, table_name)`: Drop HNSW index
- `construct_index(cur, conn, schema_name, table_name, metric, d)`: Build HNSW index
- `create_extension(cur, conn)`: Create vectordb extension
- `get_count(cur, schema_name, table_name)`: Count rows in table
- `generate_sql(schema_name, table_name, query_vector, raw_predicate, K, attr_type_list, metric)`: Generate filtered kNN SQL
- `execute_query(cur, raw_sql)`: Execute query and return result IDs

### CLI Arguments
- `--dataset_file`: Dataset file path (required)
- `--d`: Dimension (required)
- `--attr_file`: Attribute file path (required)
- `--attr_type_list`: Attribute types (required)
- `--N`: Number of data points (required)
- `--query_size`: Number of queries (default: 100, required)
- `--predicate_file`: Predicate file path (required)
- `--query_file`: Query file path (required)
- `--schema_name`: Schema name (required)
- `--table_name`: Table name (required)
- `--table_file`: Table TSV file path (required)
- `--mode`: "construction" or "query" (default: "query")
- `--max_cate_val`: Max cardinality (default: 5)
- `--K`: Top-K value (default: 10)
- `--gt_file`: Ground truth file (required)
- `--metric`: Distance metric "L2" or "IP" (default: "L2")

---

## post_query.py
Query HashANN without two-hop optimization (postprocessing variant).

### Functions
- `load_query_data(query_file, qrange_file, gt_file, N, Nq, k)`: Load query data

### Main Features
- Similar to `hashann_query.py` but with `set_two_hop_flag(False)`
- Disables two-hop neighbor recovery for comparison
- Measures impact of optimization on performance

---

## get_data.py
Extract results from log files for data analysis and plotting.

### Variables
- `target_dataset`: Target dataset name (e.g., "youtube_rgb")
- `target_attr`: Attribute configuration string
- `target_algo`: List of algorithms to extract
- `sel_mapping`: Dictionary mapping selectivity strings to numeric values
- `target_recall`: Dictionary of target recall thresholds
- `target_M`, `target_k`: Parameter filters

### Functions
- `add_to_res(res, matched_algo, sel_value, target_recall, qps)`: Accumulate results in nested dict

### Main Features
- Parses "logs.txt" file generated by `extract_results.py`
- Filters results by dataset, attributes, selectivity, M, K values
- Extracts recall and QPS for each algorithm and selectivity level
- Outputs R data.frame formatted results for plotting

---

## download_navix_dataset.py
Download Navix benchmark datasets from Hugging Face.

### Main Features
- Downloads "gaurav8297/navix" dataset
- Saves to `/mnt/data/mocheng/dataset/navix_dataset`
- Supports resume downloads

---

## modify_conf.py
(Appears to be empty/placeholder)

---

## compute_false_pos.py
Calculate false positive rate from query execution logs.

### Main Features
- Parses hybrid search statistics from log strings
- Extracts "ft_passed" (filter table passed) and "passed" (final passed) counts
- Computes false positive rate: passed / (ft_passed + passed)
- Outputs average FPR across all queries

---

## compute_speedup.py
Compute speedup metrics by comparing algorithm performance.

### Functions
- `get_my_data(line_str)`: Extract data from R data.frame format (queries "bfann" method)
- `get_sota_data(line_str)`: Extract data for other methods, handling multiple methods

### Main Features
- Parses R-formatted benchmark results
- Compares "my" algorithm (bfann) against SOTA methods
- Computes average and best speedup ratios
- Outputs speedup multipliers (e.g., "1.5x")

---

## json2txt.py
Convert JSON attribute file to TXT format for DiskANN compatibility.

### Main Features
- Reads JSON categorical attributes: `[[[val1, val2, ...]], ...]`
- Writes to TXT with values separated by newlines
- For DiskANN label file format

### Command-line Arguments (positional)
1. Input JSON attribute file path
2. Output TXT file path

---

## gt_json2bin.py
Convert JSON ground truth to binary format for DiskANN.

### Main Features
- Reads JSON ground truth: list of lists of IDs (length N×K)
- Writes binary format: [N:int32][K:int32][id0][id1]...[idN*K-1:int32]
- Reshapes flat list to N rows of K columns

### Command-line Arguments (positional)
1. Input JSON ground truth file
2. Output binary file
3. Top-K value

---

## qrange_json2bin.py
Convert JSON query ranges to binary format for iRangeGraph.

### Main Features
- Reads JSON query ranges: `[[[l,u]], [[l,u]], ...]`
- Writes binary: `[l0][u0][l1][u1]...:int32`
- Flattens ranges from 2D to 1D

### Command-line Arguments (positional)
1. Input JSON range file
2. Output binary file

---

## range2keyword.py
Convert categorical predicate labels to keyword format.

### Main Features
- Reads JSON predicates: `[[[label0, label1, ...]], ...]`
- Writes text format: `label0\nlabel1\nlabel2\n...`
- For DiskANN keyword file format

### Command-line Arguments (positional)
1. Input JSON attribute file
2. Output TXT file

---

## id2od2_json2bin.py
Convert id-to-order mapping to binary format.

### Functions
- `sort_attr_get_order(attr)`: Create mapping from original ID to sorted position

### Main Features
- Sorts attribute values and gets position mapping
- Writes binary format: [N:int32][order0][order1]...[orderN-1:int32]

### CLI Arguments
- `input_json`: Input file path (positional)
- `output_bin`: Output file path (positional)

---

## two_attr_to_one_json2bin.py
Convert single attribute from multi-attribute JSON to binary.

### Main Features
- Reads JSON with 2 attributes per point
- Extracts first attribute only
- Writes to binary range format

### Command-line Arguments (positional)
1. Input JSON file
2. Output binary file

---

## two_attr_to_two_json2bin.py
Convert both attributes from JSON to separate binary files.

### Main Features
- Reads JSON with 2 attributes: `[[[val0],[val1]], ...]`
- Extracts both values
- Writes binary: `[a0][b0][a1][b1]...:int32`

### Command-line Arguments (positional)
1. Input JSON file
2. Output binary file

---

## two_predicate_json2bin.py
Convert dual-attribute predicates to binary format.

### Main Features
- Reads JSON predicates with 2 range constraints: `[[[l0,u0],[l1,u1]], ...]`
- Writes binary: `[l0][u0][l1][u1][l2][u2]...`:int32
- Length = 4 × number of queries

### Command-line Arguments (positional)
1. Input JSON predicate file
2. Output binary file

---

## wiki_attr_update.py
Update Wikipedia attribute data by extracting date information.

### Main Features
- Reads wiki dataset attributes
- Filters to only date attributes (single value)
- Extracts min/max dates
- Saves filtered attributes to new JSON file

### Hardcoded Paths
- Source: `/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M_attr.json`
- Target: `/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M_birthdate.json`

---

## alblation.py
Ablation study analysis script comparing optimization variants.

### Variables
- `exp1`: Baseline (no diverse neighbor)
- `exp2`: With diverse neighbor selection
- `exp3`: With diverse neighbor + marker/filter table
- `exp4`: With diverse neighbor + marker + neighbor recovery
- Format: [efs, recall, qps]

### Main Features
- Defines 4 experimental variants with performance data
- Outputs R data.frame format for plotting
- Prints recall vs QPS comparison across variants

---

## redcaps_plot.py
Benchmark result visualization for RedCaps dataset.

### Variables
- Hardcoded performance data for: `bfann`, `navix`, `acorn`, `milvus`
- Nested structure: `{selectivity: {recall: qps}}`

### Functions
- `to_list(method, target_sel)`: Extract data points for target recall from method

### Main Features
- Outputs R data.frame format for selected methods at 95% recall
- Prints formatted data for plotting

---

## sift10m_plot.py
Incomplete script for SIFT-10M dataset benchmark visualization (partial data).

---

## sift1m_plot.py
Incomplete script for SIFT-1M dataset benchmark visualization (minimal data).

---

## wiki_neg_plot.py
Compare HashANN vs Navix on Wikipedia negative correlation dataset.

### Functions
- `get_qps(recall_dict, sel_list)`: Extract matching selectivity/QPS pairs from recall dict

### Variables
- `hashann_9_recall`, `hashann_95_recall`, `hashann_99_recall`: HashANN QPS by selectivity
- `navix_9_recall`: Navix QPS by selectivity
- `sel_list`: Selectivity thresholds: [1.01, 5.1, 9.96, 15.02, 22.93]

### Main Features
- Plots HashANN vs Navix QPS curves at 90% recall
- Saves plot as `wiki_navix_9_recall_comparison.png`
- Uses matplotlib for visualization

---

**End of Analysis**

---

## 五、Shell 脚本 (exp2/exp3/exp4/)

### 配置脚本 (每个实验目录不同)
| 脚本 | 用途 |
|------|------|
| `conf.sh` | 主配置：数据集路径、N、K、dim、ef参数、属性类型、选择性 |
| `acorn_conf.sh` | ACORN 参数：M_beta、gamma、test_query_size |
| `diskann_conf.sh` | DiskANN 路径和参数 |
| `diskann_stitched_conf.sh` | DiskANN Stitched 参数 |
| `irange_conf.sh` | iRange 配置和路径 |
| `milvus_conf.sh` | Milvus 集合名和参数 |
| `msvbase_conf.sh` | MSVBASE 参数 |
| `navix_conf.sh` | NaVIX 参数 |
| `wiki_conf.sh` | Wiki 数据集特定配置 |

### 索引构建脚本 (已统一)
| 脚本 | 用途 |
|------|------|
| `hashann.sh` | 构建 HashANN 索引 (调用 hashann_build.py) |
| `acorn_index.sh` | 构建 ACORN 索引 (调用 acorn_build C++) |
| `navix_index.sh` | 构建 NaVIX 索引 (调用 navix_build C++) |
| `milvus_hnsw_index.sh` | 构建 Milvus HNSW 索引 (调用 milvus_hnsw_index.py) |
| `msvbase_hnsw_index.sh` | 构建 MSVBASE HNSW 索引 |
| `diskann_index.sh` | 构建 DiskANN 索引 |
| `ground_truth_index.sh` | 构建 Ground Truth 索引 |

### 查询脚本 (已统一，均传 $K 参数)
| 脚本 | 用途 |
|------|------|
| `query.sh` | HashANN 查询 (调用 hashann_query.py + extract_results.py) |
| `acorn_query.sh` | ACORN 查询 |
| `navix_query.sh` | NaVIX 查询 |
| `milvus_hnsw_query.sh` | Milvus HNSW 查询 |
| `msvbase_hnsw_query.sh` | MSVBASE HNSW 查询 |
| `diskann_query.sh` | DiskANN 查询 |

### 实验编排脚本 (每个实验目录不同)
| 脚本 | 用途 |
|------|------|
| `exps.sh` | 批量索引构建编排 |
| `exp_query.sh` | 批量查询编排 |
| `exp_loop.sh` | 循环执行实验 |
| `exp_cons.sh` | 构建+等待编排 |

### 数据准备脚本
| 脚本 | 用途 |
|------|------|
| `attr_generator.sh` | 调用 attr_generator.py 生成属性 |
| `predicate_generator.sh` | 调用 predicate_generator.py 生成谓词 |
| `ground_truth_generator.sh` | 调用 groundtruth_generator.py |
| `selectivity.sh` | 调用 selectivity.py 计算选择性 |

