# God-Graph 文档导航

**版本**: v0.5.0-alpha  
**更新日期**: 2026-03-29

---

## 📚 文档分类

### 🚀 用户指南 (User Guide)

适合想要快速上手的开发者。

| 文档 | 说明 |
|------|------|
| [快速开始](user-guide/getting-started.md) | 5 分钟上手 God-Graph |
| [DifferentiableGraph 教程](user-guide/differentiable-graph.md) | 可微图结构完整教程 |
| [从 Petgraph 迁移](user-guide/migration-from-petgraph.md) | API 对照和迁移指南 |
| [Transformer 支持](user-guide/transformer-guide.md) | LLM 模型加载和优化 |
| [内存池优化](user-guide/memory-pool.md) | 减少分配开销 98%+ |

---

### 📖 API 参考 (API Reference)

详细的模块和函数文档。

| 模块 | 说明 |
|------|------|
| `graph::Graph` | 核心图数据结构 |
| `tensor::differentiable::DifferentiableGraph` | 可微图结构 |
| `transformer::optimization::ModelSwitch` | Safetensors 双向转换 |
| `transformer::optimization::CadStyleEditor` | 拓扑缺陷检测 |
| `tensor::decomposition` | 李群/张量环分解 |

> 📌 **提示**: 完整 API 文档请参考 [docs.rs](https://docs.rs/god-gragh)

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
| [TinyLlama 验证报告](reports/validation.md) | 真实模型端到端验证 |
| [性能基准测试](reports/performance.md) | 并行算法加速比数据 |
| [P11 代码评审](reports/critical-review.md) | 第三方代码评审报告 |
| [实现状态](reports/implementation-status.md) | 功能完成度和路线图 |

---

## 🎯 快速导航

### 我想...

- **快速上手** → [快速开始](user-guide/getting-started.md)
- **学习 DifferentiableGraph** → [教程](user-guide/differentiable-graph.md)
- **了解性能数据** → [性能报告](reports/performance.md)
- **理解设计哲学** → [CAD-LLM 设计](internals/cad-design.md)
- **查看 API 文档** → [docs.rs](https://docs.rs/god-gragh)
- **贡献代码** → [架构设计](internals/architecture.md)

---

## 📁 目录结构

```
docs/
├── README.md                     # 本文档（导航页）
├── user-guide/                   # 用户指南
│   ├── getting-started.md
│   ├── differentiable-graph.md
│   ├── transformer-guide.md
│   ├── memory-pool.md
│   └── migration-from-petgraph.md
├── api-reference/                # API 参考（占位，指向 docs.rs）
│   └── README.md
├── internals/                    # 内部实现文档
│   ├── architecture.md
│   ├── cad-design.md
│   ├── bucket-adjacency.md
│   └── ste-estimator.md
└── reports/                      # 报告
    ├── validation.md
    ├── performance.md
    ├── critical-review.md
    └── implementation-status.md
```

---

## 🔗 外部链接

- [Crates.io](https://crates.io/crates/god-gragh)
- [GitHub](https://github.com/silverenternal/god-graph)
- [docs.rs API 文档](https://docs.rs/god-gragh)
