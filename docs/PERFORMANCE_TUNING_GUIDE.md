# God-Graph 性能调优指南

本指南提供 god-graph 分布式图处理库的性能优化建议和最佳实践。

## 目录

- [内存优化](#内存优化)
- [并行计算优化](#并行计算优化)
- [算法特定优化](#算法特定优化)
- [分布式系统优化](#分布式系统优化)
- [基准测试与性能分析](#基准测试与性能分析)

---

## 内存优化

### 1. 使用合适的图表示

```rust
// 对于稀疏图，使用 adjacency_list 更节省内存
let graph = Graph::new(GraphType::Directed);

// 对于密集图 (边数 > 顶点数²/2)，考虑使用邻接矩阵
// 但 god-graph 主要优化稀疏图场景
```

**建议**：
- 稀疏图（边/顶点 < 4）：使用默认邻接表
- 中等密度（4 ≤ 边/顶点 < 10）：考虑压缩稀疏行 (CSR) 格式
- 密集图（边/顶点 ≥ 10）：评估内存使用，可能需要分片处理

### 2. 批量添加边

```rust
// ❌ 低效：逐个添加边
for edge in edges {
    graph.add_edge(edge.0, edge.1, edge.2);
}

// ✅ 高效：批量添加
graph.add_edges_batch(edges);
```

**性能提升**：批量操作减少内存重新分配，提升 30-50% 性能。

### 3. 预分配顶点容量

```rust
// 如果知道顶点数量，预分配容量
let mut graph = Graph::new(GraphType::Undirected);
graph.reserve_vertices(100_000);
```

### 4. 使用轻量级权重类型

```rust
// ✅ 推荐：使用 f32 或 i32
let graph: Graph<f32> = Graph::new(GraphType::Directed);

// ⚠️ 谨慎：复杂权重增加内存占用
let graph: Graph<ComplexWeight> = Graph::new(GraphType::Directed);
```

---

## 并行计算优化

### 1. 调整 Rayon 线程池

```rust
use rayon::ThreadPoolBuilder;

// 根据工作负载调整线程数
ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .build_global()
    .unwrap();
```

**建议**：
- CPU 密集型任务：使用物理核心数
- I/O 密集型任务：使用 2× 物理核心数
- 混合负载：使用 `num_cpus::get() * 2`

### 2. 选择合适的并行度

```rust
use god_graph::distributed::DistributedConfig;

// 小图 (< 10K 顶点)：减少并行开销
let config = DistributedConfig {
    num_partitions: 4,
    ..Default::default()
};

// 大图 (> 1M 顶点)：增加并行度
let config = DistributedConfig {
    num_partitions: 64,
    ..Default::default()
};
```

### 3. PageRank 并行优化

```rust
use god_graph::distributed::algorithms::pagerank::PageRankConfig;

// 调整收敛阈值和最大迭代次数
let config = PageRankConfig {
    damping_factor: 0.85,
    max_iterations: 20,      // 减少迭代次数
    convergence_threshold: 1e-4, // 放宽收敛条件
    use_parallel: true,
};

let result = pagerank::compute(&graph, Some(config));
```

**性能提示**：
- 放宽收敛阈值可显著减少迭代次数
- 对于近似计算，`max_iterations=10` 通常足够

### 4. BFS 层级并行

```rust
use god_graph::distributed::algorithms::bfs::BfsConfig;

let config = BfsConfig {
    parallel_level_processing: true, // 启用层级并行
    max_parallelism: 8,              // 限制最大并行度
};
```

---

## 算法特定优化

### DFS 优化

```rust
use god_graph::distributed::algorithms::dfs::{DfsConfig, DfsMode};

// 对于深度优先遍历，使用迭代模式避免栈溢出
let config = DfsConfig {
    mode: DfsMode::Iterative,
    max_stack_size: 10_000,
};

// 对于 SCC 计算，使用 Tarjan 算法
let scc_result = tarjan_scc(&graph);
```

**建议**：
- 深度图：使用迭代模式
- 需要调用栈信息：使用递归模式
- SCC 计算：Tarjan 算法最优

### Connected Components 优化

```rust
use god_graph::distributed::algorithms::connected_components::{CCConfig, CCAlgorithm};

// 大图使用并查集算法
let config = CCConfig {
    algorithm: CCAlgorithm::UnionFind,
    use_path_compression: true,
    use_union_by_rank: true,
};

// 动态连通性查询
let mut cc = ConnectedComponents::new(&graph, config);
cc.compute();

// 快速查询
if cc.in_same_component(u, v) {
    println!("u and v are connected");
}
```

**性能对比**：
| 算法 | 时间复杂度 | 适用场景 |
|------|-----------|---------|
| UnionFind | O(E α(V)) | 静态图 |
| LabelPropagation | O(E log V) | 动态图 |
| BFS-based | O(V + E) | 小图 |

### Dijkstra 优化

```rust
use god_graph::distributed::algorithms::dijkstra::{DijkstraConfig, DijkstraAlgorithm};

// 单源最短路径
let config = DijkstraConfig {
    algorithm: DijkstraAlgorithm::Standard,
    use_bidirectional: false,
};

let result = dijkstra::single_source(&graph, source, Some(config));

// 单对最短路径使用双向搜索
let config = DijkstraConfig {
    algorithm: DijkstraAlgorithm::Bidirectional,
    ..Default::default()
};

let path = dijkstra::single_pair(&graph, source, target, Some(config));
```

**优化技巧**：
- 单对查询：使用双向 Dijkstra，减少 50% 搜索空间
- 多源查询：使用 `multi_source_dijkstra`
- 负权重：考虑使用 Bellman-Ford（未实现）

### PageRank 优化

```rust
// 使用稀疏矩阵优化
let config = PageRankConfig {
    use_sparse_matrix: true,
    compression_threshold: 1e-6, // 压缩小值
    ..Default::default()
};
```

---

## 分布式系统优化

### 1. 分区策略选择

```rust
use god_graph::distributed::{PartitionStrategy, DistributedConfig};

// 顶点切割：适合高顶点度图
let config = DistributedConfig {
    partition_strategy: PartitionStrategy::VertexCut,
    num_partitions: 16,
};

// 边切割：适合高边密度图
let config = DistributedConfig {
    partition_strategy: PartitionStrategy::EdgeCut,
    num_partitions: 16,
};
```

**选择指南**：
| 图特征 | 推荐策略 | 理由 |
|--------|---------|------|
| 高顶点度 | VertexCut | 减少边复制 |
| 高边密度 | EdgeCut | 平衡负载 |
| 社交网络 | VertexCut | 处理幂律分布 |
| 路网图 | EdgeCut | 度数分布均匀 |

### 2. 消息传递优化

```rust
use god_graph::distributed::MessageConfig;

// 批量消息传递
let msg_config = MessageConfig {
    batch_size: 1024,           // 批量大小
    aggregation_threshold: 100, // 聚合阈值
    use_compression: true,      // 启用压缩
};
```

### 3. 容错性能权衡

```rust
use god_graph::distributed::fault_tolerance::{RetryPolicy, CircuitBreakerConfig};

// 调整重试策略
let retry_policy = RetryPolicy::exponential_with_jitter(
    Duration::from_millis(100),  // 初始延迟
    Duration::from_secs(5),      // 最大延迟
    3,                            // 最大重试次数
);

// 配置熔断器
let circuit_breaker = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    recovery_timeout: Duration::from_secs(30),
    half_open_max_calls: 3,
});
```

---

## 基准测试与性能分析

### 运行基准测试

```bash
# 运行所有分布式基准测试
cargo bench --bench distributed

# 运行特定算法基准测试
cargo bench --bench distributed -- --filter pagerank

# 生成性能报告
cargo bench --bench distributed -- --output-format bencher | tee results.txt
```

### 使用 perf 进行性能分析

```bash
# 安装 perf
sudo apt install linux-tools-common linux-tools-generic

# 生成火焰图
cargo flamegraph --bench distributed

# 或使用 perf
perf record --call-graph dwarf cargo bench --bench distributed
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### 内存分析

```bash
# 使用 valgrind
valgrind --tool=massif cargo bench --bench distributed

# 查看内存使用
ms_print massif.out.<pid>
```

### 性能监控指标

```rust
use god_graph::distributed::metrics::PerformanceMetrics;

let metrics = PerformanceMetrics::new();

// 关键指标
println!("处理时间：{:?}", metrics.processing_time());
println!("内存使用：{} MB", metrics.memory_usage_mb());
println!("吞吐量：{} 顶点/秒", metrics.throughput());
println!("并行效率：{:.2}%", metrics.parallel_efficiency());
```

---

## 最佳实践总结

### ✅ 推荐做法

1. **预分配**：知道大小时预分配容量
2. **批量操作**：使用批量 API 减少开销
3. **选择合适算法**：根据图特征选择算法
4. **调整并行度**：匹配硬件资源
5. **监控性能**：定期运行基准测试

### ❌ 避免做法

1. **频繁重新分配**：避免动态增长大图
2. **过度并行**：小图使用过多线程
3. **忽略内存布局**：使用复杂权重类型
4. **不释放资源**：及时释放不用的图数据
5. **盲目优化**：先测量再优化

---

## 性能检查清单

在性能关键应用中，检查以下项目：

- [ ] 使用 `reserve_vertices()` 预分配
- [ ] 使用批量 API (`add_edges_batch`)
- [ ] 选择合适的分区策略
- [ ] 调整 Rayon 线程池大小
- [ ] 配置算法参数（迭代次数、收敛阈值）
- [ ] 启用并行处理
- [ ] 监控内存使用
- [ ] 运行基准测试对比
- [ ] 使用性能分析工具定位瓶颈
- [ ] 考虑使用近似算法（如需要）

---

## 故障排除

### 常见问题

**Q: 内存使用过高？**
- 检查是否使用复杂权重类型
- 考虑分片处理大图
- 使用 `shrink_to_fit()` 释放多余容量

**Q: 并行性能不佳？**
- 检查线程数是否过多
- 确认工作负载足够并行
- 查看是否有锁竞争

**Q: 算法收敛慢？**
- 调整收敛阈值
- 减少最大迭代次数
- 考虑使用近似算法

---

## 参考资料

- [Rayon 文档](https://docs.rs/rayon)
- [Rust 性能优化指南](https://doc.rust-lang.org/book/ch13-00-functional-features.html)
- [分布式图处理论文](https://github.com/god-graph/papers)

---

*最后更新：2026-03-31*
*版本：god-graph v0.6.0-alpha*
