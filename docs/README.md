# God-Graph 文档导航

**版本**: v0.7.0-alpha
**更新日期**: 2026-04-08
**测试状态**: 512/512 测试通过 ✅
**Clippy 状态**: 0 警告 ✅

---

## 📚 文档分类

### 🚀 用户指南 (User Guide)

适合想要快速上手的开发者。

| 文档 | 说明 |
|------|------|
| [快速开始](user-guide/getting-started.md) | **5 分钟上手 God-Graph** ⭐ 推荐从这里开始 |
| [DifferentiableGraph 教程](user-guide/differentiable-graph.md) | 可微图结构完整教程 |
| [从 Petgraph 迁移](user-guide/migration-from-petgraph.md) | API 对照和迁移指南 |
| [Transformer 支持](user-guide/transformer-guide.md) | LLM 模型加载和优化 |

---

### 🏛️ VGI 架构文档

Virtual Graph Interface - 统一图后端接口。

| 文档 | 说明 |
|------|------|
| [VGI 架构总结](VGI_ARCHITECTURE_SUMMARY.md) | **VGI 架构完整总结** ⭐ 推荐从这里开始 |
| [VGI 指南](VGI_GUIDE.md) | Virtual Graph Interface 详细使用指南 |
| [VGI 实施计划](VGI_IMPLEMENTATION_PLAN.md) | VGI 架构详细设计 |
| [插件开发指南](PLUGIN_DEVELOPMENT_GUIDE.md) | 第三方算法插件开发 |
| [分布式处理指南](DISTRIBUTED_GUIDE.md) | 分布式图处理 |
| [容错指南](FAULT_TOLERANCE_GUIDE.md) | 分布式系统容错 |

---

### 📖 API 参考 (API Reference)

详细的模块和函数文档。

| 模块 | 说明 |
|------|------|
| `graph::Graph` | 核心图数据结构（桶式邻接表） |
| `vgi::VirtualGraph` | 虚拟图接口 trait |
| `tensor::differentiable::DifferentiableGraph` | 可微图结构 |
| `transformer::optimization::ModelSwitch` | Safetensors 双向转换 |
| `transformer::optimization::CadStyleEditor` | 拓扑缺陷检测 |
| `tensor::decomposition` | 李群/张量环分解 |

> 📌 **提示**: 完整 API 文档请参考 [docs.rs](https://docs.rs/god-graph)

---

### 🔧 内部实现 (Internals)

适合想要深入了解实现细节的开发者。

| 文档 | 说明 |
|------|------|
| [架构设计](internals/architecture.md) | 模块职责和工作流 |
| [CAD-LLM 设计哲学](internals/cad-design.md) | 为什么需要 CAD-LLM 范式迁移 |
| [桶式邻接表实现](internals/bucket-adjacency.md) | O(1) 增量更新原理 |
| [STE 估计器](internals/ste-estimator.md) | Straight-Through Estimator 实现 |

---

### 📊 报告 (Reports)

性能数据、验证报告、代码评审。

| 报告 | 说明 |
|------|------|
| [实现状态](reports/implementation-status.md) | 功能完成度和路线图 |
| [性能基准测试](reports/performance.md) | 并行算法加速比数据 |
| [分布式性能报告](reports/distributed-benchmarks.md) | 分布式处理性能数据 |
| [TinyLlama 验证报告](reports/validation.md) | 真实模型端到端验证 |
| [性能优化总结](../OPTIMIZATION_SUMMARY.md) | 全面性能优化技术清单 |
| [优化实施报告](../OPTIMIZATION_REPORT_2026_04_08.md) | 2026-04-08 优化详情 |

---

## 🎯 快速导航

### 我想...

- **快速上手** → [快速开始](user-guide/getting-started.md)
- **学习 DifferentiableGraph** → [教程](user-guide/differentiable-graph.md)
- **了解 VGI 架构** → [VGI 指南](VGI_GUIDE.md)
- **开发插件** → [插件开发指南](PLUGIN_DEVELOPMENT_GUIDE.md)
- **了解性能数据** → [性能报告](reports/performance.md)
- **理解设计哲学** → [CAD-LLM 设计](internals/cad-design.md)
- **查看 API 文档** → [docs.rs](https://docs.rs/god-graph)
- **贡献代码** → [架构设计](internals/architecture.md)

---

## 📁 目录结构

```
docs/
├── README.md                           # 本文档（导航页）
│
├── user-guide/                         # 用户指南
│   ├── getting-started.md             # 快速开始
│   ├── differentiable-graph.md        # DifferentiableGraph 教程
│   ├── migration-from-petgraph.md     # 从 Petgraph 迁移
│   ├── transformer-guide.md           # Transformer 支持
│   └── visualization.md               # 可视化指南
│
├── VGI_ARCHITECTURE_SUMMARY.md         # VGI 架构总结（新）
├── VGI_GUIDE.md                        # VGI 使用指南
├── VGI_IMPLEMENTATION_PLAN.md          # VGI 实施计划
├── PLUGIN_DEVELOPMENT_GUIDE.md         # 插件开发指南
├── DISTRIBUTED_GUIDE.md                # 分布式处理指南
├── FAULT_TOLERANCE_GUIDE.md            # 容错指南
├── MIGRATION_GUIDE_v0.6.md             # v0.5→v0.6 迁移指南
├── MIGRATION_GUIDE_v0.7_VGI_LAYERED_API.md  # v0.6→v0.7 迁移指南
│
├── api-reference/                      # API 参考
│   └── README.md                       # 指向 docs.rs
│
├── internals/                          # 内部实现文档
│   ├── architecture.md                 # 架构设计
│   └── cad-design.md                   # CAD-LLM 设计哲学
│
└── reports/                            # 报告
    ├── implementation-status.md        # 实现状态
    ├── performance.md                  # 性能基准
    ├── distributed-benchmarks.md       # 分布式性能
    ├── validation.md                   # TinyLlama 验证
    └── ... (其他历史报告)
```

---

## 🔗 外部链接

- [Crates.io](https://crates.io/crates/god-graph)
- [GitHub](https://github.com/silverenternal/god-graph)
- [docs.rs API 文档](https://docs.rs/god-graph)
- [GitHub Issues](https://github.com/silverenternal/god-graph/issues)

---

## 📝 文档更新日志

### v0.7.0-alpha (2026-04-08) - 文档整理

**删除冗余文档**:
- ✅ 删除临时任务文档 (TASK*.md) - 内容已合并到 CHANGELOG
- ✅ 删除重复性能优化文档 (PERFORMANCE_OPTIMIZATIONS.md)
- ✅ 删除 docs/reports 下重复性能报告 (6 个文件)
- ✅ 合并 VGI 阶段性报告 → VGI_ARCHITECTURE_SUMMARY.md

**新增文档**:
- ✅ VGI_ARCHITECTURE_SUMMARY.md - VGI 架构完整总结

**更新文档**:
- ✅ docs/README.md - 更新导航结构

### v0.6.0-alpha (2026-04-01)

- ✅ 创建全面的快速开始指南 (getting-started.md)
- ✅ 更新 README.md 版本号和特性标志
- ✅ 修复所有 `god-gragh` → `god-graph` 拼写错误
- ✅ 更新 VGI 架构文档链接
- ✅ 添加分布式处理文档导航
- ✅ 更新快速开始示例代码

---

## ❓ 常见问题

### Q: God-Graph 和 petgraph 有什么区别？

**A**: petgraph 是通用图算法库，适合静态图分析；God-Graph 专注于**LLM 白盒分析**和**可微图结构**，支持：
- Safetensors 双向转换
- DifferentiableGraph 梯度优化
- 李群正交化、张量环压缩
- VGI 架构支持多后端

### Q: DifferentiableGraph 可以用于训练吗？

**A**: DifferentiableGraph 目前主要用于**架构搜索**和**结构优化**，不是完整的训练框架。如需训练 GNN 模型，建议集成外部 autograd 库（如 dfdx、Candle）。

### Q: VGI 架构的性能开销是多少？

**A**: VGI trait 对象开销约 5-10%，但换来的是**后端可插拔**和**算法复用**的架构优势。对于性能敏感场景，可直接使用 `Graph<T, E>`。

### Q: 如何贡献代码？

**A**: 欢迎贡献！请确保：
1. 代码通过 `cargo clippy` 和 `cargo fmt`
2. 添加适当的测试
3. 更新相关文档
4. 提交 PR 到 GitHub
