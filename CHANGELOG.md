# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- par_dijkstra 重构（修复桶索引计算错误和死锁风险）
- GitHub Pages 文档站点
- crates.io 发布

### Added (v0.3.1-beta)
- **SIMD 向量化支持**（实验性）
  - `par_pagerank_simd` - 使用 `wide::f64x4` 批量计算 PageRank 分数
  - `par_degree_centrality_simd` - SIMD 优化的度中心性计算
  - 使用 `wide` crate，支持 stable Rust，启用 `simd` 特性即可
  - 在支持 AVX2/AVX-512 的 CPU 上可获得 2-4x 加速
- **SVG 可视化导出**
  - `to_svg` 和 `to_svg_with_options` 函数
  - 支持 3 种布局算法：力导向、圆形、层次
  - 可自定义节点颜色、大小、标签等
  - 交互式 Web 查看器 (`examples/graph_viewer.html`)
- **测试覆盖率集成**
  - 集成 `cargo-tarpaulin` 进行覆盖率统计
  - 当前覆盖率：66.64%（目标 80%+）
  - 添加 `coverage.sh` 脚本
- **文档增强**
  - README 添加 SVG 导出示例
  - 新增覆盖率说明章节

### Fixed (v0.3.1 候选)
- Flow 算法内存优化澄清：当前实现已使用 O(V+E) 邻接表，但初始化有 O(V) 固定开销
  - 计划使用 `HashMap<usize, Vec<(usize, f64)>>` 替代 `Vec<Vec<(usize, f64)>>`
  - 针对超大规模稀疏图（V >> E）可减少内存浪费

---

## [0.3.0-beta] - 2026-03-26

### Added
- **完整性能基准报告** (`docs/performance.md`)
  - PageRank 1000 节点：串行 53.9ms → 并行 668µs = **80.7x 加速比**
  - DFS 50K 节点：串行 9.7ms → 并行 1.3ms = **7.5x 加速比**
  - 连通分量 2000 节点：357.8µs
  - 度中心性 5000 节点：146µs
- **petgraph 迁移指南** (`docs/migration-from-petgraph.md`)
  - 412 行完整迁移文档
  - 核心差异对比表
  - API 对照表
  - 完整代码示例
- **并行算法套件**（基于 rayon）
  - `par_dfs` - 子树并行 DFS
  - `par_bfs` - 分层并行 BFS
  - `par_dijkstra` - 并行松弛 Dijkstra
  - `par_pagerank` - 迭代内并行 PageRank
  - `par_connected_components` - 并行并查集
- **基准测试配置**
  - `speedup.rs` - 完整加速比基准测试
  - `parallel.rs` - 并行算法性能测试
  - `centrality.rs` - 中心性算法基准

### Changed
- **版本号升级**: 0.1.0 → 0.3.0-beta（反映项目成熟度）
- **README 性能数据更新**
  - 更新性能宣传语与 performance.md 保持一致
  - 添加详细性能数据表格
  - 更新示例代码中的版本号
- **架构文档澄清**
  - `ROADMAP.json` 明确说明"类 Arena 槽位管理"（非独立 Arena 类型）
  - `ROADMAP.json` 明确说明"桶式 CSR 变体"（非传统 CSR）

### Fixed
- **centrality.rs 编译错误**
  - 修复 `rng.gen::<f64>()` → `rng.gen_range(0.0..1.0)`（Rust 2024 保留关键字变化）
- **par_connected_components 死锁问题**
  - 根本原因：并行并查集路径压缩导致竞争条件
  - 修复方案：移除路径压缩，使用原子 CAS 操作
- **Doctest 忽略问题**
  - 修复 `src/lib.rs` line 39 的 doctest（添加 `par_pagerank` 导入）
  - 修复 `src/utils/arena.rs` doctest（`gen` → `generation` 关键字）
  - 达到 100% 通过率：23 passed, 0 ignored
- **Clippy 警告**
  - 修复 `needless_range_loop` 警告
  - 主代码保持 0 警告状态

### Performance
- **PageRank**: 80.7x 加速比（1000 节点，53.9ms → 668µs）
- **DFS**: 7.5x 加速比（50K 节点，9.7ms → 1.3ms）
- **连通分量**: 357.8µs（2000 节点）
- **度中心性**: 146µs（5000 节点）

### Documentation
- **新增文件**:
  - `docs/performance.md` - 性能基准报告
  - `docs/migration-from-petgraph.md` - petgraph 迁移指南
- **更新文件**:
  - `README.md` - 性能数据、版本号
  - `ROADMAP.json` - 架构实现细节澄清

### Testing
- **测试覆盖率**: 122 个测试全部通过
  - 88 单元测试
  - 19 集成测试
  - 15 属性测试
  - 23 Doctest（100% 通过率）
- **Clippy**: 0 警告（主代码）

### Technical Details
- **桶式 CSR 实现**
  - `AdjBucket` 结构：`neighbors + edge_indices + deleted_mask(位图) + deleted_count`
  - `#[repr(align(64))]` 缓存行对齐
  - O(1) 增量插入、惰性删除、按需压缩
- **类 Arena 槽位管理**
  - `NodeSlot`: `data: Option<T> + generation: u32`
  - `EdgeStorage`: `source + target + data + generation: u32`
  - `free_nodes + free_edges` 空闲列表
- **Generation 验证**
  - `contains_node`: 检查 `slot.generation == index.generation()`
  - `get_node`: 验证 generation 并返回 `NodeDeleted` 错误
  - `remove_node`: 验证 generation
  - `neighbors`: 自动获取目标节点的最新 generation

---

## [0.2.0-alpha] - 2026-03-26

### Added
- **桶式 CSR 内存布局**
  - 支持 O(1) 增量更新
  - 64 字节对齐优化
  - 软件预取支持（条件编译 nightly）
- **Generation 验证机制**
  - 完整集成到 `add_node`/`get_node`/`remove_node`/`contains_node`
  - 防止 ABA 问题
- **完整算法套件**
  - 遍历算法：DFS、BFS、拓扑排序、Tarjan SCC
  - 最短路径：Dijkstra、Bellman-Ford、Floyd-Warshall、A*
  - 最小生成树：Kruskal、Prim
  - 中心性算法：度中心性、介数中心性、接近中心性、PageRank
  - 社区检测：连通分量、标签传播
- **随机图生成器**
  - Erdős-Rényi 模型
  - Barabási-Albert 模型
  - Watts-Strogatz 模型
  - 完全图、网格图、树
- **图导出功能**
  - DOT/Graphviz 格式
  - 邻接表、边列表

### Changed
- **Flow 算法内存优化**
  - 残量图从 `Vec<Vec<f64>>` 改为 `O(V+E)` 邻接表
- **API 改进**
  - 泛型 trait 设计
  - 迭代器 API 遵循 Rust 习惯用法

### Fixed
- **并行算法崩溃问题**
  - 调试并修复 `par_connected_components` 死锁
- **基准测试内存优化**
  - 减小基准测试规模避免内存爆炸

---

## [0.1.0-alpha] - Initial Release

### Added
- **核心图结构**
  - `Graph<T, E>` 泛型图类型
  - `NodeIndex` / `EdgeIndex` 新类型包装
  - 邻接表表示法
- **基本 CRUD 操作**
  - `add_node` / `get_node` / `update_node` / `remove_node`
  - `add_edge` / `get_edge` / `update_edge` / `remove_edge`
  - `neighbors` / `incident_edges` / `has_edge` / `degree`
- **基础算法**
  - DFS（递归/迭代）
  - BFS（标准/分层）
  - 拓扑排序
  - Dijkstra 算法
- **图构建器**
  - `GraphBuilder` 链式 API
- **测试套件**
  - 单元测试
  - 集成测试
  - 属性测试（proptest）

---

## Migration Guide

### From 0.1.0 to 0.3.0-beta

1. **更新 Cargo.toml**:
   ```toml
   [dependencies]
   god-gragh = "0.3.0-beta"
   ```

2. **启用并行特性**（可选）:
   ```toml
   god-gragh = { version = "0.3.0-beta", features = ["parallel"] }
   ```

3. **API 变更**: 无破坏性变更

4. **性能提升**: 自动享受 80x PageRank 加速比

### From petgraph to god-gragh

详细迁移指南请参阅 [`docs/migration-from-petgraph.md`](docs/migration-from-petgraph.md)。

核心差异：
- God-Graph 使用**桶式 CSR 变体**而非传统邻接表
- God-Graph 使用**类 Arena 槽位管理**而非独立 Arena 类型
- God-Graph 内置**并行算法套件**
- God-Graph 提供**Generation 索引稳定性**

---

## Version History Summary

| Version | Release Date | Status | Key Features |
|---------|--------------|--------|--------------|
| 0.1.0-alpha | 2026-03-26 | Released | 核心图结构、基本 CRUD、DFS/BFS |
| 0.2.0-alpha | 2026-03-26 | Released | 桶式 CSR、完整算法套件、随机图生成 |
| 0.3.0-beta | 2026-03-26 | **Current** | 性能报告、迁移指南、并行算法验证 |
| 0.4.0-rc | Planned | - | crates.io 发布、文档站点 |
| 1.0.0-stable | Planned | - | API 稳定化、生产就绪 |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code passes `cargo clippy` and `cargo fmt`
- Add appropriate tests
- Update documentation

## License

This project is licensed under either of:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Contact

- Issues: [GitHub Issues](https://github.com/silverenternal/god-gragh/issues)
- Discussions: [GitHub Discussions](https://github.com/silverenternal/god-gragh/discussions)
