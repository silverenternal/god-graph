# VGI 用户指南

## Virtual Graph Interface 完全指南

**版本**: v0.6.0-alpha
**日期**: 2026-03-31
**状态**: 已完成 ✅

---

## 📖 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [快速开始](#快速开始)
4. [核心概念](#核心概念)
5. [使用示例](#使用示例)
6. [插件系统](#插件系统)
7. [分布式处理](#分布式处理)
8. [最佳实践](#最佳实践)
9. [故障排查](#故障排查)

---

## 概述

### 什么是 VGI？

**Virtual Graph Interface (VGI)** 是 God-Graph 库的核心抽象层，它提供了一套统一的图操作接口，类似于 Linux 的 VFS (Virtual File System)。

VGI 的核心设计目标：
- **统一接口**: 所有图后端实现相同的接口
- **可插拔后端**: 支持单机、分布式、GPU 等多种后端
- **插件系统**: 第三方算法可以动态加载
- **向后兼容**: 现有的 `Graph<T, E>` 无缝集成到 VGI 体系

### 为什么需要 VGI？

在 VGI 之前，God-Graph 只支持单机内存图结构。VGI 的引入带来了：

1. **后端抽象**: 算法代码不需要关心底层存储细节
2. **扩展性**: 可以轻松添加新的后端（分布式、GPU、持久化）
3. **生态建设**: 第三方开发者可以编写插件算法
4. **性能优化**: 不同后端可以针对特定场景优化

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              (你的图处理应用程序)                            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plugin System Layer                       │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  Algorithm  │  │  Analyzer   │  │  Visualizer │  ...   │
│   │   Plugin    │  │   Plugin    │  │   Plugin    │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Virtual Graph Interface (VGI)                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              VirtualGraph Trait                      │   │
│   │  - nodes() / edges()                                │   │
│   │  - add_node() / add_edge()                          │   │
│   │  - neighbors() / incident_edges()                   │   │
│   │  - metadata() / capabilities()                      │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                            │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │   Single    │  │  Distributed│  │   External  │        │
│   │   Machine   │  │   Cluster   │  │   Database  │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 位置 | 描述 |
|------|------|------|
| `VirtualGraph` | `src/vgi/traits.rs` | 核心 trait，定义图的基本操作 |
| `GraphMetadata` | `src/vgi/metadata.rs` | 图的元数据和能力描述 |
| `Backend` | `src/backend/traits.rs` | 后端抽象 trait |
| `GraphAlgorithm` | `src/plugins/algorithm.rs` | 算法插件接口 |
| `PluginRegistry` | `src/plugins/registry.rs` | 插件注册表 |

---

## 快速开始

### 基础用法

```rust
use god_graph::prelude::*;
use god_graph::vgi::VirtualGraph;

fn main() -> god_graph::errors::GraphResult<()> {
    // 创建图（Graph<T, E> 自动实现 VirtualGraph）
    let mut graph = Graph::<String, f64>::directed();
    
    // 添加节点
    let a = graph.add_node("节点 A".to_string())?;
    let b = graph.add_node("节点 B".to_string())?;
    let c = graph.add_node("节点 C".to_string())?;
    
    // 添加边
    graph.add_edge(a, b, 1.0)?;
    graph.add_edge(b, c, 2.0)?;
    graph.add_edge(a, c, 4.0)?;
    
    // 查询操作
    println!("节点数：{}", graph.node_count());
    println!("边数：{}", graph.edge_count());
    
    // 遍历邻居
    println!("节点 A 的邻居:");
    for neighbor in graph.neighbors(a) {
        println!("  - {}", graph.get_node(neighbor)?);
    }
    
    Ok(())
}
```

### 使用后端构建器

```rust
use god_graph::backend::{BackendConfig, BackendBuilder};
use god_graph::backend::single_machine::SingleMachineBuilder;
use god_graph::vgi::metadata::GraphType;
use god_graph::vgi::VirtualGraph;

fn main() -> god_graph::vgi::VgiResult<()> {
    // 配置后端
    let config = BackendConfig::new(GraphType::Directed)
        .with_node_capacity(1000)
        .with_edge_capacity(5000)
        .with_parallel(Some(4));
    
    // 构建后端
    let mut backend: SingleMachineBackend<String, f64> = 
        SingleMachineBuilder::new()
            .with_config(config)
            .build()?;
    
    // 使用后端（和 Graph 一样的接口）
    let n1 = backend.add_node("节点 1".to_string())?;
    let n2 = backend.add_node("节点 2".to_string())?;
    backend.add_edge(n1, n2, 1.5)?;
    
    println!("后端名称：{}", backend.name());
    println!("节点数：{}", backend.node_count());
    
    Ok(())
}
```

---

## 核心概念

### 1. VirtualGraph Trait

`VirtualGraph` 是所有图后端的统一接口，定义了以下核心操作：

#### 元数据查询
```rust
fn metadata(&self) -> GraphMetadata;
fn has_capability(&self, capability: Capability) -> bool;
fn node_count(&self) -> usize;
fn edge_count(&self) -> usize;
```

#### 节点操作
```rust
fn add_node(&mut self, data: Self::NodeData) -> VgiResult<NodeIndex>;
fn get_node(&self, index: NodeIndex) -> VgiResult<&Self::NodeData>;
fn remove_node(&mut self, index: NodeIndex) -> VgiResult<Self::NodeData>;
fn contains_node(&self, index: NodeIndex) -> bool;
fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, Self::NodeData>>;
```

#### 边操作
```rust
fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, data: Self::EdgeData) 
    -> VgiResult<EdgeIndex>;
fn get_edge(&self, index: EdgeIndex) -> VgiResult<&Self::EdgeData>;
fn remove_edge(&mut self, index: EdgeIndex) -> VgiResult<Self::EdgeData>;
fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool;
fn edges(&self) -> impl Iterator<Item = EdgeRef<'_, Self::EdgeData>>;
```

#### 邻接查询
```rust
fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex>;
fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex>;
fn out_degree(&self, node: NodeIndex) -> VgiResult<usize>;
fn in_degree(&self, node: NodeIndex) -> VgiResult<usize>;
```

### 2. 图元数据 (GraphMetadata)

元数据描述图的基本信息和能力：

```rust
use god_graph::vgi::metadata::{GraphMetadata, GraphType, Capability};

let metadata = GraphMetadata::new("my_graph", GraphType::Directed)
    .with_name("我的图")
    .with_node_count(100)
    .with_edge_count(500)
    .with_capability(Capability::Parallel)
    .with_capability(Capability::IncrementalUpdate);

// 检查能力
assert!(metadata.supports(Capability::Parallel));
assert!(!metadata.supports(Capability::Distributed));

// 检查是否支持所有能力
assert!(metadata.supports_all(&[
    Capability::Parallel,
    Capability::IncrementalUpdate
]));
```

### 3. 能力标识 (Capability)

能力标识用于声明后端支持的功能：

| 能力 | 描述 |
|------|------|
| `Parallel` | 支持并行执行 |
| `Distributed` | 支持分布式执行 |
| `IncrementalUpdate` | 支持增量更新 |
| `Transactions` | 支持事务操作 |
| `Persistence` | 支持持久化存储 |
| `Partitioning` | 支持图分区 |
| `DynamicMode` | 支持动态模式 |
| `StaticMode` | 支持静态模式 |
| `WeightedEdges` | 支持加权边 |
| `SelfLoops` | 支持自环 |
| `MultiEdges` | 支持多重边 |

---

## 使用示例

### 示例 1: 泛型算法函数

编写一个适用于任何后端的通用算法：

```rust
use god_graph::vgi::VirtualGraph;
use god_graph::vgi::VgiResult;

/// 计算图的平均度数（适用于任何后端）
fn average_degree<G>(graph: &G) -> VgiResult<f64>
where
    G: VirtualGraph<NodeData = impl Clone, EdgeData = impl Clone>,
{
    if graph.node_count() == 0 {
        return Ok(0.0);
    }
    
    let mut total_degree = 0;
    for node_ref in graph.nodes() {
        total_degree += graph.out_degree(node_ref.index())?;
    }
    
    Ok(total_degree as f64 / graph.node_count() as f64)
}

// 使用示例
fn main() -> VgiResult<()> {
    use god_graph::graph::Graph;
    
    let mut graph = Graph::<i32, f64>::undirected();
    for i in 0..10 {
        graph.add_node(i)?;
    }
    
    let avg = average_degree(&graph)?;
    println!("平均度数：{}", avg);
    
    Ok(())
}
```

### 示例 2: 能力检查

在运行时检查后端能力：

```rust
use god_graph::vgi::{VirtualGraph, Capability};
use god_graph::backend::single_machine::SingleMachineBackend;

fn process_graph<N, E>(backend: &SingleMachineBackend<N, E>) 
where
    N: Clone + Send + Sync,
    E: Clone + Send + Sync,
{
    // 检查是否支持并行
    if backend.has_capability(Capability::Parallel) {
        println!("支持并行，使用并行算法");
        // 运行并行算法...
    } else {
        println!("使用串行算法");
        // 运行串行算法...
    }
    
    // 检查是否支持增量更新
    if backend.has_capability(Capability::IncrementalUpdate) {
        println!("支持增量更新");
    }
}
```

### 示例 3: 图构建器

使用构建器模式创建图：

```rust
use god_graph::vgi::builder::GraphBuilder;
use god_graph::graph::Graph;
use god_graph::vgi::metadata::GraphType;

fn main() {
    // 链式调用构建图
    let graph: Graph<String, f64> = Graph::builder()
        .directed()
        .with_capacity(100, 500)
        .with_config("custom_key", "custom_value")
        .build();
    
    // 或者使用辅助函数
    let undirected_graph = Graph::<i32, f64>::undirected();
    let directed_graph = Graph::<i32, f64>::directed();
}
```

---

## 插件系统

### 注册算法插件

```rust
use god_graph::plugins::{PluginRegistry, GraphAlgorithm, PluginContext, AlgorithmResult, AlgorithmData};
use god_graph::vgi::{VirtualGraph, VgiResult, Capability, GraphType};
use god_graph::plugins::algorithm::{PluginInfo, PluginPriority};
use std::collections::HashMap;

// 定义自定义算法
struct MyAlgorithm;

impl GraphAlgorithm for MyAlgorithm {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("my_algorithm", "1.0.0", "我的自定义算法")
            .with_author("你的名字")
            .with_required_capabilities(&[Capability::Parallel])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["custom", "demo"])
            .with_priority(PluginPriority::Normal)
    }
    
    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        // 获取配置参数
        let damping = ctx.get_config_as("damping", 0.85);
        let max_iter = ctx.get_config_as("max_iter", 20);
        
        // 实现算法逻辑
        let mut result = HashMap::new();
        for node_ref in ctx.graph.nodes() {
            result.insert(node_ref.index(), 1.0 / ctx.graph.node_count() as f64);
        }
        
        Ok(AlgorithmResult::node_values(result)
            .with_metadata("iterations", max_iter.to_string()))
    }
    
    fn as_any(&self) -> &std::any::Any {
        self
    }
}

fn main() -> VgiResult<()> {
    use god_graph::graph::Graph;
    
    // 创建注册表并注册插件
    let mut registry = PluginRegistry::new();
    registry.register_algorithm("my_algo", MyAlgorithm)?;
    
    // 创建图
    let mut graph = Graph::<String, f64>::directed();
    let n1 = graph.add_node("A".to_string())?;
    let n2 = graph.add_node("B".to_string())?;
    graph.add_edge(n1, n2, 1.0)?;
    
    // 执行算法
    let mut ctx = PluginContext::new(&graph)
        .with_config("damping", "0.85")
        .with_config("max_iter", "100");
    
    let result = registry.execute::<Graph<String, f64>, MyAlgorithm>(
        "my_algo",
        &graph,
        &mut ctx,
    )?;
    
    println!("算法结果：{:?}", result.data);
    
    Ok(())
}
```

### 使用内置算法插件

God-Graph 提供了多个内置算法插件：

```rust
use god_graph::plugins::{PluginRegistry, PageRankPlugin, BfsPlugin, DfsPlugin};
use god_graph::graph::Graph;

fn main() -> god_graph::vgi::VgiResult<()> {
    let mut registry = PluginRegistry::new();
    
    // 注册内置算法
    registry.register_algorithm("pagerank", PageRankPlugin)?;
    registry.register_algorithm("bfs", BfsPlugin)?;
    registry.register_algorithm("dfs", DfsPlugin)?;
    
    // 创建测试图
    let mut graph = Graph::<i32, f64>::directed();
    for i in 0..10 {
        graph.add_node(i)?;
    }
    // 添加边...
    
    // 列出所有插件
    println!("已注册插件：{:?}", registry.list_plugins());
    
    // 按标签查找
    let centrality_plugins = registry.find_by_tag("centrality");
    println!("中心性算法：{:?}", centrality_plugins);
    
    Ok(())
}
```

---

## 分布式处理

### 图分区

```rust
use god_graph::distributed::{
    HashPartitioner, RangePartitioner, Partitioner, PartitionerConfig
};
use god_graph::graph::Graph;
use god_graph::vgi::metadata::GraphType;

fn main() -> god_graph::vgi::VgiResult<()> {
    // 创建图
    let mut graph = Graph::<String, f64>::directed();
    for i in 0..100 {
        graph.add_node(format!("节点{}", i))?;
    }
    
    // 配置分区器
    let config = PartitionerConfig::new(4); // 4 个分区
    
    // 使用 Hash 分区
    let hash_partitioner = HashPartitioner::new(&config);
    let partitions = hash_partitioner.partition(&graph)?;
    
    println!("Hash 分区结果:");
    for (i, partition) in partitions.iter().enumerate() {
        println!("  分区 {}: {} 个节点", i, partition.nodes.len());
    }
    
    // 使用 Range 分区
    let range_partitioner = RangePartitioner::new(&config);
    let range_partitions = range_partitioner.partition(&graph)?;
    
    Ok(())
}
```

### 分布式执行器

```rust
use god_graph::distributed::{
    DistributedExecutor, ExecutorConfig, WorkerInfo
};

fn main() -> god_graph::vgi::VgiResult<()> {
    // 配置分布式执行器
    let config = ExecutorConfig {
        num_workers: 4,
        communication_timeout: Some(std::time::Duration::from_secs(30)),
        ..Default::default()
    };
    
    let executor = DistributedExecutor::new(&config);
    
    // 获取 worker 信息
    let workers = executor.get_workers();
    println!("Worker 数量：{}", workers.len());
    
    // 分布式 PageRank
    use god_graph::distributed::algorithms::{DistributedPageRank, PageRankConfig};
    
    let pr_config = PageRankConfig {
        damping: 0.85,
        max_iterations: 20,
        tolerance: 1e-6,
    };
    
    // let result = executor.run_pagerank(&graph, &pr_config)?;
    
    Ok(())
}
```

---

## 最佳实践

### 1. 选择合适的后端

根据应用场景选择后端：

```rust
// 单机小图：使用默认的 Graph<T, E>
let graph = Graph::<String, f64>::directed();

// 单机大图：使用带容量的 SingleMachineBackend
let config = BackendConfig::new(GraphType::Directed)
    .with_node_capacity(1_000_000)
    .with_edge_capacity(10_000_000);

// 分布式超大图：使用 DistributedBackend
// (未来实现)
```

### 2. 能力检查优先

在运行算法前检查后端能力：

```rust
fn run_algorithm<G>(graph: &G) -> VgiResult<()>
where
    G: VirtualGraph,
{
    if !graph.has_capability(Capability::Parallel) {
        // 降级到串行算法
        println!("警告：后端不支持并行，使用串行版本");
    }
    
    // 继续执行...
    Ok(())
}
```

### 3. 合理使用插件配置

```rust
// 好的做法：明确指定所有参数
let ctx = PluginContext::new(&graph)
    .with_config("damping", "0.85")
    .with_config("max_iter", "100")
    .with_config("tolerance", "1e-6");

// 避免：依赖默认值，可能导致不可预测的行为
let ctx = PluginContext::new(&graph); // 没有配置
```

### 4. 错误处理

```rust
use god_graph::vgi::{VgiError, VgiResult};

fn safe_operation<G>(graph: &mut G) -> VgiResult<()>
where
    G: VirtualGraph,
{
    // 检查节点是否存在
    let node_index = 42;
    if !graph.contains_node(node_index) {
        return Err(VgiError::Internal {
            message: format!("Node {} not found", node_index),
        });
    }
    
    // 安全操作
    let data = graph.get_node(node_index)?;
    
    Ok(())
}
```

### 5. 内存管理

```rust
// 预分配容量，避免频繁重新分配
let mut graph = Graph::<String, f64>::with_capacity(10000, 50000);

// 或者使用 reserve
graph.reserve(10000, 50000);

// 定期清理不需要的图
graph.clear();
```

---

## 故障排查

### 常见问题

#### 1. "Backend does not support capability" 错误

**原因**: 尝试在不支持的后端运行需要特定能力的算法。

**解决方案**:
```rust
// 检查能力
if !backend.has_capability(Capability::Parallel) {
    // 使用串行版本或升级后端
    println!("请使用支持并行的后端");
}
```

#### 2. 插件注册失败

**原因**: 插件名称冲突或信息不完整。

**解决方案**:
```rust
// 确保插件名称唯一
registry.register_algorithm("unique_name_v1", my_plugin)?;

// 提供完整的插件信息
let info = PluginInfo::new("name", "1.0.0", "Description")
    .with_author("Author Name");
```

#### 3. 分布式执行超时

**原因**: 网络延迟或 worker 故障。

**解决方案**:
```rust
let config = ExecutorConfig {
    communication_timeout: Some(std::time::Duration::from_secs(60)),
    retry_count: 3,
    ..Default::default()
};
```

### 调试技巧

1. **启用日志**:
```rust
env_logger::init();
log::info!("VGI initialized");
```

2. **检查元数据**:
```rust
let metadata = graph.metadata();
log::debug!("Graph metadata: {:?}", metadata);
```

3. **验证插件**:
```rust
let info = plugin.info();
log::debug!("Plugin info: {:?}", info);
```

---

## 附录

### A. 版本兼容性

| God-Graph 版本 | VGI 版本 | Rust 版本 |
|---------------|---------|----------|
| 0.5.0 | 1.0 (初始) | 1.85+ |
| 0.6.0 | 1.0 (稳定) | 1.85+ |

### B. 性能基准

| 操作 | Graph<T,E> | SingleMachine | 说明 |
|------|-----------|---------------|------|
| add_node | ~100ns | ~100ns | 相同实现 |
| add_edge | ~150ns | ~150ns | 相同实现 |
| neighbors | ~50ns/iter | ~50ns/iter | 零开销抽象 |

### C. 相关文档

- [VGI_IMPLEMENTATION_PLAN.md](VGI_IMPLEMENTATION_PLAN.md) - 实施计划
- [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) - 插件开发指南
- [DISTRIBUTED_GUIDE.md](DISTRIBUTED_GUIDE.md) - 分布式处理指南

---

**最后更新**: 2026-03-31
**维护者**: God-Graph Team
