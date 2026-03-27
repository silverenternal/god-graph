# God-Graph 性能基准测试报告

> 测试日期：2026-03-26  
> 测试环境：Linux, 61GB RAM, 多核 CPU  
> God-Graph 版本：v0.1.0 (向 v0.3.0-beta 推进)

## 执行摘要

God-Graph 是一个高性能 Rust 图操作库，采用桶式邻接表格式、类 Arena 槽位管理和并行计算优化。基准测试显示：

- **并行 PageRank**: 在 1000 节点图上达到 **80x 加速比**（串行 53.9ms → 并行 668µs）
- **并行连通分量**: 在 2000 节点图上耗时 357µs
- **并行度中心性**: 在 5000 节点图上耗时 68µs
- **Doctest 通过率**: 100% (23/23 通过)
- **Clippy 警告**: 0 个

**注**: 本库使用桶式邻接表变体（代码中命名为 CsrStorage）而非传统 CSR 格式，以支持 O(1) 增量更新。并行算法使用细粒度锁（Mutex/RwLock）保护共享数据，par_dijkstra 在 v0.3.0-beta 中标记为 experimental。

## 测试环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Linux |
| 内存 | 61GB |
| Rust 版本 | 1.85 (2024 edition) |
| 编译优化 | `-C opt-level=3 -C lto=thin -C codegen-units=1` |

## 并行算法加速比

### PageRank

| 节点数 | 串行时间 | 并行时间 | 加速比 |
|--------|----------|----------|--------|
| 1,000 | 53.9 ms | 668 µs | **80.7x** |
| 5,000 | 1.68 s | 待完成 | 待计算 |

**测试参数**: damping=0.85, iterations=20, avg_degree=5

**分析**: PageRank 算法在 God-Graph 中实现了显著的加速，主要得益于：
1. 反向邻接表优化，时间复杂度 O(iterations × E)
2. 细粒度锁保护下的并行节点更新
3. 桶式邻接表格式提供高效的邻居遍历

### 连通分量 (Connected Components)

| 节点数 | 并行时间 | 备注 |
|--------|----------|------|
| 200 | 35.8 µs | 4 个分量 |
| 400 | 66.5 µs | 4 个分量 |
| 1,000 | 147.8 µs | 4 个分量 |
| 2,000 | 357.8 µs | 4 个分量 |

**测试参数**: 稀疏图（环状连接），每个分量内部形成环状 + 额外边

**注意**: 并行并查集实现在多核上可能不会带来显著加速，因为并查集的 union-find 操作本质上是串行的。当前实现使用原子 CAS 操作保证安全性，移除了路径压缩以避免竞争条件。

### 度中心性 (Degree Centrality)

| 节点数 | 并行时间 |
|--------|----------|
| 500 | 27.0 µs |
| 1,000 | 49.1 µs |
| 2,000 | 68.9 µs |
| 5,000 | 待完成 |

**测试参数**: avg_degree=10

## 内存布局优化

### 桶式邻接表格式 (Bucket-based Adjacency List)

God-Graph 使用桶式邻接表变体（代码中命名为 CsrStorage）而非传统 CSR，支持 O(1) 增量更新：

```rust
pub struct AdjBucket {
    neighbors: Vec<usize>,      // 目标节点索引
    edge_indices: Vec<usize>,   // 边索引
    deleted_mask: Vec<u64>,     // 位图标记删除
    deleted_count: usize,       // 删除计数
}

pub struct CsrStorage {
    buckets: Vec<AdjBucket>,           // 每个节点的邻接桶
    reverse_buckets: Vec<AdjBucket>,   // 反向邻接（有向图）
    needs_compact: bool,               // 压缩标记
}
```

**优势**:
- O(1) 增量边插入（push 到桶）
- 惰性删除（deleted_mask 位图标记）
- 按需压缩（compact() 回收空间）
- 64 字节对齐避免 false sharing
- 软件预取支持（条件编译 nightly）

### 类 Arena 槽位管理

```rust
pub struct NodeSlot<T> {
    data: Option<T>,
    generation: u32,
}

pub struct Graph<T, E> {
    nodes: Vec<NodeSlot<T>>,
    edges: Vec<EdgeStorage<E>>,
    csr: CsrStorage,
    free_nodes: Vec<usize>,  // 空闲列表
    free_edges: Vec<usize>,
}
```

**特性**:
- generation 验证防止 ABA 问题
- 空闲列表支持索引复用
- 槽位连续存储优化缓存命中率

## 算法性能对比

### 遍历算法

| 算法 | 实现 | 时间复杂度 | 并行 |
|------|------|------------|------|
| DFS | 递归 + 迭代 | O(V+E) | ✓ |
| BFS | 队列 | O(V+E) | ✓ |
| Tarjan SCC | 迭代 | O(V+E) | ✗ |

### 最短路径

| 算法 | 实现 | 时间复杂度 | 并行 |
|------|------|------------|------|
| Dijkstra | 优先队列 | O((V+E)logV) | ✓ (delta-stepping) |
| Bellman-Ford | 迭代 | O(VE) | ✗ |
| Floyd-Warshall | 动态规划 | O(V³) | ✗ |
| A* | 启发式搜索 | O((V+E)logV) | ✗ |

### 中心性算法

| 算法 | 实现 | 时间复杂度 | 并行 |
|------|------|------------|------|
| 度中心性 | 计数 | O(V) | ✓ |
| 介数中心性 | Brandes 算法 | O(VE) | ✗ |
| 接近中心性 | BFS | O(V(V+E)) | ✗ |
| PageRank | 迭代 | O(iterations×E) | ✓ |

### 最大流

| 算法 | 实现 | 时间复杂度 | 内存优化 |
|------|------|------------|----------|
| Edmonds-Karp | BFS 增广路 | O(VE²) | O(V+E) 邻接表 |
| Dinic | 分层图 | O(V²E) | O(V+E) 邻接表 |
| Push-Relabel | 预流推进 | O(V²E) | O(V+E) 邻接表 |

**内存优化**: Flow 算法残量图从 `Vec<Vec<f64>>` 邻接矩阵优化为 `Vec<Vec<(usize, f64)>>` 邻接表，空间复杂度从 O(V²) 降至 O(V+E)。

## 测试覆盖率

### 单元测试

- **总数**: 88 个
- **通过率**: 100%

### 集成测试

- **总数**: 18 个
- **通过率**: 100%

### 属性测试 (Property-based)

- **总数**: 15 个
- **通过率**: 100%

### Doctest

- **总数**: 23 个
- **通过率**: 100%
- **忽略**: 0 个

### Clippy

- **警告数**: 0 个（已修复 23 个）

## 与 petgraph 对比

| 特性 | God-Graph | petgraph |
|------|-----------|----------|
| CSR 格式 | ✓ (桶式) | ✓ (传统) |
| 增量更新 | ✓ O(1) | ✗ 需重建 |
| Generation 验证 | ✓ | ✗ |
| 并行算法 | ✓ (5 个) | ✗ |
| 64 字节对齐 | ✓ | ✗ |
| 软件预取 | ✓ (nightly) | ✗ |
| 社区成熟度 | 发展中 | 成熟 |

**God-Graph 优势**:
1. Generation 索引稳定性，防止 ABA 问题
2. 桶式邻接表支持 O(1) 增量更新
3. 并行算法套件（PageRank、BFS、DFS、连通分量、度中心性）
4. 缓存优化（64 字节对齐、软件预取）

**petgraph 优势**:
1. 社区成熟度高，生产环境验证
2. 文档完整性
3. 更多图算法变体

**架构说明**:
- God-Graph 使用桶式邻接表（代码中命名为 CsrStorage），非传统 CSR 格式
- 并行算法使用细粒度锁（Mutex/RwLock），非无锁设计
- par_dijkstra 在 v0.3.0-beta 中标记为 experimental，存在已知问题

## 待完成工作

### 性能优化

- [ ] SIMD 向量化优化（PageRank 分数批量计算）
- [ ] par_dijkstra 重构（修复桶索引计算错误和死锁风险）
- [ ] 细粒度锁优化（减少 Mutex 竞争开销）
- [ ] 更完整的加速比数据（特别是大图）

### 文档

- [ ] petgraph 迁移指南（已创建框架）
- [ ] 生产环境案例收集
- [ ] 算法可视化示例

### 测试

- [ ] 大规模图测试（100K+ 节点）
- [ ] 内存使用基准测试
- [ ] 并发安全性测试

## 结论

God-Graph 核心功能已达到 **v0.3.0-beta** 标准，但存在已知问题需要修复：

✅ 所有 clippy 警告已修复（0 个）
✅ Doctest 100% 通过（23/23）
✅ 单元测试、集成测试、属性测试全部通过（121/121）
✅ 并行算法加速比数据支撑（PageRank 80x）
✅ 核心架构正确实现（桶式邻接表、generation 验证、缓存优化）

⚠️ **已知问题**:
- par_dijkstra 存在桶索引计算错误和死锁风险，标记为 experimental
- 并行算法使用细粒度锁（Mutex/RwLock），非无锁设计
- 桶式邻接表（代码中命名为 CsrStorage）非传统 CSR 格式

**建议**: 优先修复 P0 问题（文档诚信和 par_dijkstra bug），然后发布 v0.3.0-beta 收集用户反馈。后续优化重点放在 SIMD 向量化和更大规模基准测试上。

---

*最后更新：2026-03-26*
