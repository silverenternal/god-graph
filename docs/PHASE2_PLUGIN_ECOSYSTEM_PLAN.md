# Phase 2: 插件生态建设实施计划

**版本**: v0.6.0-alpha
**日期**: 2026-03-31
**状态**: 进行中 🔄

---

## 📋 目标

完成 VGI 架构的 Phase 2：插件生态建设，包括：
1. 完善算法插件接口
2. 创建插件开发模板
3. 添加 10+ 示例插件
4. 编写插件开发文档

---

## ✅ Phase 2 完成标准

- [ ] 算法插件接口完善
- [ ] 插件开发模板发布
- [ ] 至少 10 个示例插件
- [ ] 文档：`PLUGIN_DEVELOPMENT.md` 完成
- [ ] 插件热加载机制（可选）

---

## 📦 示例插件清单（目标 10+）

### 已有插件 (4 个) ✅
1. **PageRank** - 页面排名算法
2. **BFS** - 广度优先搜索
3. **DFS** - 深度优先搜索
4. **Connected Components** - 连通分量

### 待添加插件 (6+ 个)

#### 最短路径算法 (2 个)
5. **Dijkstra** - 单源最短路径（加权图）
6. **Bellman-Ford** - 单源最短路径（支持负权边）

#### 中心性算法 (2 个)
7. **Betweenness Centrality** - 介数中心性
8. **Closeness Centrality** - 接近中心性

#### 社区检测 (1 个)
9. **Louvain** - Louvain 社区检测算法

#### 图遍历应用 (1 个)
10. **Topological Sort** - 拓扑排序（DAG）

#### 其他算法 (可选)
11. **K-Core Decomposition** - K 核分解
12. **Label Propagation** - 标签传播社区检测
13. **Graph Coloring** - 图着色算法

---

## 📁 文件结构

```
src/plugins/
├── mod.rs
├── algorithm.rs              # 已有 ✅
├── registry.rs               # 已有 ✅
├── algorithms/
│   ├── mod.rs                # 已有 ✅
│   ├── pagerank.rs           # 已有 ✅
│   ├── bfs.rs                # 已有 ✅
│   ├── dfs.rs                # 已有 ✅
│   ├── connected_components.rs # 已有 ✅
│   ├── dijkstra.rs           # 待添加
│   ├── bellman_ford.rs       # 待添加
│   ├── betweenness_centrality.rs # 待添加
│   ├── closeness_centrality.rs   # 待添加
│   ├── louvain.rs            # 待添加
│   └── topological_sort.rs   # 待添加
└── templates/                # 插件开发模板
    └── README.md             # 模板说明
```

---

## 🔧 实施步骤

### 步骤 1: 完善插件接口 (已完成 ✅)
- [x] `GraphAlgorithm` trait
- [x] `PluginInfo` 结构
- [x] `PluginContext` 上下文
- [x] `AlgorithmResult` 结果类型

### 步骤 2: 添加示例插件
1. **Dijkstra 算法** - 最短路径
2. **Bellman-Ford 算法** - 最短路径（负权）
3. **Betweenness Centrality** - 介数中心性
4. **Closeness Centrality** - 接近中心性
5. **Louvain 算法** - 社区检测
6. **Topological Sort** - 拓扑排序

### 步骤 3: 创建插件开发模板
- 插件模板代码
- Cargo.toml 配置示例
- 测试示例

### 步骤 4: 编写开发文档
- `PLUGIN_DEVELOPMENT.md` 完整指南
- API 参考文档
- 最佳实践

---

## 📝 插件开发模板

### 基础模板

```rust
//! [插件名称]
//!
//! [插件描述]

use crate::plugins::algorithm::{
    GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData,
};
use crate::vgi::{VirtualGraph, Capability, GraphType, VgiResult};
use std::any::Any;
use std::collections::HashMap;

/// [插件名称]
pub struct [PluginName]Plugin {
    // 配置参数
}

impl [PluginName]Plugin {
    pub fn new(/* 参数 */) -> Self {
        Self {
            // 初始化
        }
    }

    /// 核心算法实现
    pub fn compute<G>(&self, graph: &G) -> VgiResult</* 结果类型 */>
    where
        G: VirtualGraph + ?Sized,
    {
        // 实现算法逻辑
        Ok(result)
    }
}

impl GraphAlgorithm for [PluginName]Plugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("[plugin-name]", "1.0.0", "[插件描述]")
            .with_author("[作者]")
            .with_required_capabilities(&[/* 所需能力 */])
            .with_supported_graph_types(&[/* 支持的图类型 */])
            .with_tags(&[/* 标签 */])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        // 从配置中读取参数
        // 调用 compute() 执行算法
        // 返回 AlgorithmResult
        
        Ok(AlgorithmResult::new("[name]", AlgorithmData::/* 类型 */(result)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_basic() {
        // 编写测试
    }
}
```

---

## 📊 进度追踪

| 插件名称 | 状态 | 难度 | 预计时间 |
|---------|------|------|---------|
| PageRank | ✅ 完成 | 简单 | - |
| BFS | ✅ 完成 | 简单 | - |
| DFS | ✅ 完成 | 简单 | - |
| Connected Components | ✅ 完成 | 简单 | - |
| Dijkstra | 🔲 待开始 | 中等 | 2h |
| Bellman-Ford | 🔲 待开始 | 中等 | 2h |
| Betweenness Centrality | 🔲 待开始 | 困难 | 3h |
| Closeness Centrality | 🔲 待开始 | 中等 | 2h |
| Louvain | 🔲 待开始 | 困难 | 4h |
| Topological Sort | 🔲 待开始 | 简单 | 1h |

**预计总时间**: ~14 小时

---

## 🎯 下一步行动

1. 实现 Dijkstra 算法插件
2. 实现 Bellman-Ford 算法插件
3. 实现 Topological Sort 插件
4. 实现中心性算法插件
5. 实现 Louvain 社区检测插件
6. 创建插件开发模板文档
7. 编写 PLUGIN_DEVELOPMENT.md

---

## 📚 参考资源

- [Graph Algorithms](https://en.wikipedia.org/wiki/Graph_algorithm)
- [NetworkX Algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html)
- [Apache Spark GraphX](https://spark.apache.org/graphx/)
- [Boost Graph Library](https://www.boost.org/doc/libs/release/libs/graph/)
