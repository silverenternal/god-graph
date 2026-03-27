# God-Graph

[![Crates.io](https://img.shields.io/crates/v/god-gragh.svg)](https://crates.io/crates/god-gragh)
[![Documentation](https://docs.rs/god-gragh/badge.svg)](https://docs.rs/god-gragh)
[![License](https://img.shields.io/crates/l/god-gragh.svg)](LICENSE)
[![Build Status](https://github.com/silverenternal/god-graph/workflows/CI/badge.svg)](https://github.com/silverenternal/god-graph/actions)
[![Coverage Status](https://codecov.io/gh/silverenternal/god-graph/branch/main/graph/badge.svg)](https://codecov.io/gh/silverenternal/god-graph)

**God-Graph** 是一个高性能的 Rust 图数据结构和算法库，采用桶式邻接表内存布局、Arena 分配器和并行计算优化。

## 特性

### 🚀 高性能
- **桶式邻接表内存布局**: 结合 Arena 分配器和桶式增量更新设计，实现 O(1) 节点访问和边插入
  - *注：传统 CSR（压缩稀疏行）格式不支持增量更新，本库采用桶式变体以支持动态图操作*
- **缓存友好设计**: 64 字节对齐、软件预取，优化 CPU 缓存命中率
- **稳定索引**: Generation 计数防止 ABA 问题，支持安全的节点/边删除
- **并行算法**: 基于 rayon 的并行 BFS、PageRank 等算法，8 核 CPU 上显著加速（PageRank 实测 80x，详见 [性能报告](docs/performance.md)）

### 📦 功能丰富
- **完整算法套件**: 遍历、最短路径、最小生成树、中心性、社区检测等
- **随机图生成**: Erdős-Rényi、Barabási-Albert、Watts-Strogatz 模型
- **多种导出格式**: DOT/Graphviz、SVG、邻接表、边列表
- **可选特性**: Serde 序列化、并行计算、矩阵表示

### 🛡️ 类型安全
- **泛型设计**: 节点和边支持任意数据类型
- **编译时检查**: 利用 Rust 类型系统确保图操作正确性
- **零成本抽象**: 无运行时开销的高层次抽象

## 快速开始

### 安装

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
god-gragh = "0.3.0-beta"
```

### 基本使用

```rust
use god_gragh::graph::Graph;
use god_gragh::graph::traits::{GraphOps, GraphQuery};

// 创建有向图
let mut graph = Graph::<String, f64>::directed();

// 添加节点
let a = graph.add_node("A".to_string()).unwrap();
let b = graph.add_node("B".to_string()).unwrap();
let c = graph.add_node("C".to_string()).unwrap();

// 添加边
graph.add_edge(a, b, 1.0).unwrap();
graph.add_edge(b, c, 2.0).unwrap();
graph.add_edge(a, c, 3.0).unwrap();

// 查询
println!("节点数：{}", graph.node_count());
println!("边数：{}", graph.edge_count());

// 遍历邻居
for neighbor in graph.neighbors(a) {
    println!("邻居：{}", graph[neighbor]);
}
```

### 使用图构建器

```rust
use god_gragh::graph::builders::GraphBuilder;

let graph = GraphBuilder::directed()
    .with_nodes(vec!["A", "B", "C", "D"])
    .with_edges(vec![
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 3, 3.0),
        (2, 3, 4.0),
    ])
    .build()
    .unwrap();
```

## 算法

### 遍历算法
```rust
use god_gragh::algorithms::traversal::{dfs, bfs, topological_sort, tarjan_scc};

// 深度优先搜索
dfs(&graph, start_node, |node| {
    println!("访问：{}", node.data());
    true // 继续遍历
});

// 广度优先搜索
bfs(&graph, start_node, |node| {
    println!("访问：{}", node.data());
    true
});

// 拓扑排序（DAG）
let order = topological_sort(&graph);

// Tarjan 强连通分量
let sccs = tarjan_scc(&graph);
```

### 最短路径算法
```rust
use god_gragh::algorithms::shortest_path::{dijkstra, bellman_ford, floyd_warshall, astar};

// Dijkstra 算法（非负权重）
let (path, distance) = dijkstra(&graph, start, Some(end)).unwrap();

// A* 搜索
let heuristic = |node: NodeIndex| -> f64 { /* 启发式函数 */ 0.0 };
let (path, distance) = astar(&graph, start, end, heuristic).unwrap();

// Bellman-Ford（可处理负权重）
let distances = bellman_ford(&graph, start);

// Floyd-Warshall（所有点对最短路径）
let distances = floyd_warshall(&graph);
```

### 最小生成树
```rust
use god_gragh::algorithms::mst::{kruskal, prim};

// Kruskal 算法
let mst = kruskal(&graph);

// Prim 算法
let mst = prim(&graph, start_node);
```

### 中心性算法
```rust
use god_gragh::algorithms::centrality::{
    degree_centrality, betweenness_centrality, closeness_centrality, pagerank
};

// 度中心性
let centrality = degree_centrality(&graph);

// 介数中心性
let centrality = betweenness_centrality(&graph);

// 接近中心性
let centrality = closeness_centrality(&graph);

// PageRank
let ranks = pagerank(&graph, 0.85, 20);
```

### 社区检测
```rust
use god_gragh::algorithms::community::{connected_components, label_propagation};

// 连通分量
let components = connected_components(&graph);

// 标签传播算法
let communities = label_propagation(&graph);
```

## 并行算法

启用 `parallel` 特性以使用并行算法：

```toml
[dependencies]
god-gragh = { version = "0.3.0-beta", features = ["parallel"] }
```

```rust
use god_gragh::algorithms::parallel;

// 并行 BFS
let layers = parallel::bfs_parallel(&graph, start_node);

// 并行 PageRank
let ranks = parallel::pagerank_parallel(&graph, 0.85, 20);

// 并行连通分量
let components = parallel::connected_components_parallel(&graph);
```

## 随机图生成

```rust
use god_gragh::generators::{
    erdos_renyi_graph, barabasi_albert_graph, watts_strogatz_graph,
    complete_graph, grid_graph, tree_graph
};

// Erdős-Rényi 随机图 G(n, p)
let graph = erdos_renyi_graph::<String>(100, 0.1, true, 42);

// Barabási-Albert 优先连接模型
let graph = barabasi_albert_graph::<String>(100, 3);

// Watts-Strogatz 小世界网络
let graph = watts_strogatz_graph::<String>(100, 4, 0.1);

// 完全图
let graph = complete_graph::<String, f64>(10);

// 网格图
let graph = grid_graph::<String, f64>(5, 5);

// 树
let graph = tree_graph::<String, f64>(3, 100);
```

## 图导出

```rust
use god_gragh::export::{to_dot, to_svg, to_adjacency_list, to_edge_list};

// 导出为 DOT 格式（Graphviz）
let dot = to_dot(&graph);
std::fs::write("graph.dot", dot)?;

// 导出为 SVG 格式（Web 可视化）
use god_gragh::export::svg::{SvgOptions, LayoutAlgorithm};
let options = SvgOptions::new()
    .with_size(800, 600)
    .with_node_radius(25.0)
    .with_layout(LayoutAlgorithm::ForceDirected);
let svg = to_svg(&graph, &options);
std::fs::write("graph.svg", svg)?;

// 导出为邻接表
let adj_list = to_adjacency_list(&graph);

// 导出为边列表
let edge_list = to_edge_list(&graph);
```

生成可视化：
```bash
# 使用 Graphviz
dot -Tpng graph.dot -o graph.png

# 或使用 Web 查看器（推荐）
# 在浏览器中打开 examples/graph_viewer.html，然后加载 graph.svg
```

## 特性标志

| 特性 | 描述 | 依赖 |
|------|------|------|
| `std` | 标准库支持（默认启用） | - |
| `parallel` | 并行算法 | rayon |
| `serde` | 序列化支持 | serde |
| `dot` | DOT 格式导出 | - |
| `simd` | SIMD 向量化（实验性） | wide (stable Rust 兼容) |
| `matrix` | 矩阵表示 | nalgebra |
| `rand` | 随机图生成 | rand, rand_chacha |
| `unstable` | 夜间 Rust 特性 | - |

### SIMD 特性说明

✅ **注意**: `simd` 特性使用 [`wide`](https://crates.io/crates/wide) crate 实现 SIMD 向量化计算，**支持 stable Rust**。
在 PageRank、中心性计算等算法中，SIMD 可提供 2-4 倍性能提升。

```bash
# stable Rust 用户直接使用
cargo build --features "simd"
```

**实现细节**: 使用 `wide::f64x4` 类型进行 4 路并行浮点运算，自动利用 CPU SIMD 指令集（SSE/AVX/AVX-512）。

## 与 petgraph 对比

| 特性 | God-Graph | petgraph |
|------|-----------|----------|
| 内存布局 | 桶式邻接表 + Arena | 邻接表 |
| 稳定索引 | ✅ Generation 计数 | ✅ Stable Graph |
| 并行算法 | ✅ 内置 | ❌ |
| 缓存优化 | ✅ 64 字节对齐 | ❌ |
| API 设计 | 泛型 trait | 具体类型 |
| 文档完整性 | 🎯 100% | ✅ 完整 |

## 性能基准

详细性能数据请参阅 [**性能报告**](docs/performance.md)。

在 8 核 CPU 上的基准测试结果：

| 算法 | 规模 | 串行时间 | 并行时间 | 加速比 |
|------|------|----------|----------|--------|
| PageRank | 1000 节点 | 53.9ms | 668µs | **80.7x** |
| DFS | 50K 节点 | 9.7ms | 1.3ms | **7.5x** |
| 连通分量 | 2000 节点 | - | 357.8µs | - |
| 度中心性 | 5000 节点 | - | 146µs | - |

运行基准测试：
```bash
cargo bench --all-features
```

## 测试覆盖率

本项目使用 `cargo-tarpaulin` 进行覆盖率统计，目标覆盖率 **80%+**。

### 生成覆盖率报告

```bash
# 安装 cargo-tarpaulin
cargo install cargo-tarpaulin

# 生成 HTML 覆盖率报告
cargo tarpaulin --all-features --out Html --output-dir coverage

# 查看报告
open coverage/tarpaulin-report.html  # macOS
xdg-open coverage/tarpaulin-report.html  # Linux
```

### 当前覆盖率

- **总体覆盖率**: 66.64% (1560/2341 lines)
- **单元测试**: 79 passed
- **集成测试**: 18 passed
- **属性测试**: 15 passed
- **文档测试**: 27 passed
- **总计**: 139 tests, 100% passing

详见 [coverage/tarpaulin-report.html](coverage/tarpaulin-report.html)

## 开发路线图

详见 [ROADMAP.json](ROADMAP.json)

- [x] v0.1.0-alpha: 核心图结构、基本 CRUD、DFS/BFS
- [x] v0.2.0-alpha: 完整算法套件、随机图生成器
- [ ] v0.3.0-beta: 完整文档、基准测试套件
- [ ] v0.4.0-beta: 性能优化、并行算法验证
- [ ] v0.5.0-rc: Serde 支持、API 稳定化
- [ ] v1.0.0-stable: 生产就绪

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

请确保：
- 代码通过 `cargo clippy` 和 `cargo fmt` 检查
- 添加适当的测试
- 更新文档

## 许可证

本项目采用双许可：MIT 或 Apache-2.0（由你选择）。

详见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。

## 致谢

- [petgraph](https://github.com/petgraph/petgraph) - Rust 图库的先驱
- [rayon](https://github.com/rayon-rs/rayon) - 数据并行库
- [Graphviz](https://graphviz.org/) - 图可视化工具

## 联系方式

- 问题反馈：[GitHub Issues](https://github.com/silverenternal/god-graph/issues)
- 讨论：[GitHub Discussions](https://github.com/silverenternal/god-graph/discussions)
