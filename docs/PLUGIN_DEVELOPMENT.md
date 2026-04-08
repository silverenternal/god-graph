# 插件开发指南

**版本**: v0.6.0
**日期**: 2026-03-31

---

## 📖 概述

God-Graph 插件系统允许开发者轻松扩展图算法功能。本指南将帮助您快速上手开发自己的插件。

---

## 🏗️ 插件架构

God-Graph 的插件系统采用类似 Linux 内核模块的设计：

```
┌─────────────────────────────────────┐
│         插件注册表 (Registry)        │
├─────────────────────────────────────┤
│  后端插件  │  算法插件  │  序列化器  │
├─────────────────────────────────────┤
│         VirtualGraph 接口           │
├─────────────────────────────────────┤
│         图后端 (Backend)            │
└─────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 创建插件结构

```rust
use god_graph::plugins::algorithm::{
    GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData,
};
use god_graph::vgi::{VirtualGraph, Capability, GraphType, VgiResult};
use std::any::Any;
use std::collections::HashMap;

/// 您的插件名称
pub struct MyPlugin {
    // 配置参数
    threshold: f64,
}

impl MyPlugin {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}
```

### 2. 实现 GraphAlgorithm trait

```rust
impl GraphAlgorithm for MyPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("my-plugin", "1.0.0", "我的插件描述")
            .with_author("您的名字")
            .with_required_capabilities(&[Capability::Parallel])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["my-tag", "category"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        // 1. 从配置中读取参数
        let threshold = ctx.get_config_as("threshold", self.threshold);
        
        // 2. 执行算法逻辑
        let result = self.compute(ctx.graph, threshold)?;
        
        // 3. 返回结果
        Ok(AlgorithmResult::new("my_result", AlgorithmData::NodeValues(result)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

### 3. 注册插件

```rust
use god_graph::plugins::{PluginRegistry, register_algorithm};

fn main() {
    let plugin = MyPlugin::new(0.5);
    register_algorithm(Box::new(plugin));
    
    // 或者使用全局注册表
    let mut registry = PluginRegistry::new();
    registry.register_algorithm(Box::new(plugin));
}
```

---

## 📋 核心 API 参考

### GraphAlgorithm trait

所有算法插件必须实现的核心 trait：

```rust
pub trait GraphAlgorithm: Send + Sync {
    /// 获取插件信息
    fn info(&self) -> PluginInfo;

    /// 验证图是否满足算法要求
    fn validate<G>(&self, ctx: &PluginContext<G>) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized;

    /// 执行算法
    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized;

    /// 执行前回调
    fn before_execute<G>(&self, ctx: &PluginContext<G>) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized;

    /// 执行后回调
    fn after_execute<G>(&self, ctx: &PluginContext<G>, result: &AlgorithmResult) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized;

    /// 清理资源
    fn cleanup(&self);

    /// 获取 Any 引用，用于向下转型
    fn as_any(&self) -> &dyn Any;
}
```

### PluginInfo 结构

插件元数据：

```rust
pub struct PluginInfo {
    pub name: String,              // 插件名称
    pub version: String,           // 插件版本
    pub description: String,       // 插件描述
    pub author: Option<String>,    // 插件作者
    pub required_capabilities: Vec<Capability>,  // 所需能力
    pub supported_graph_types: Vec<GraphType>,   // 支持的图类型
    pub tags: Vec<String>,         // 插件标签
    pub priority: PluginPriority,  // 优先级
    pub config_schema: HashMap<String, ConfigField>, // 配置模式
}
```

### PluginContext 上下文

插件执行时的上下文环境：

```rust
pub struct PluginContext<'a, G>
where
    G: VirtualGraph + ?Sized,
{
    pub graph: &'a G,                                    // 图引用
    pub config: HashMap<String, String>,                 // 配置
    pub cancelled: bool,                                 // 取消标志
    pub progress_callback: Option<Box<dyn Fn(f32) + Send + 'a>>, // 进度回调
    pub timeout: Option<Duration>,                       // 超时时间
    pub start_time: Option<std::time::Instant>,          // 开始时间
    pub execution_id: Option<String>,                    // 执行 ID
}
```

### AlgorithmResult 结果

算法执行结果：

```rust
pub struct AlgorithmResult {
    pub name: String,                              // 结果名称
    pub data: AlgorithmData,                       // 结果数据
    pub metadata: HashMap<String, String>,         // 元数据
}

pub enum AlgorithmData {
    NodeValues(HashMap<usize, f64>),    // 节点值映射
    NodeList(Vec<usize>),               // 节点列表
    EdgeList(Vec<usize>),               // 边列表
    Communities(Vec<usize>),            // 社区划分
    Scalar(f64),                        // 标量值
    Boolean(bool),                      // 布尔值
    String(String),                     // 字符串
    Custom(String),                     // 自定义数据
}
```

---

## 📝 完整示例

### 示例 1: 简单统计算法

```rust
//! 图密度计算插件

use god_graph::plugins::algorithm::{
    GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData,
};
use god_graph::vgi::{VirtualGraph, Capability, GraphType, VgiResult};
use std::any::Any;

pub struct GraphDensityPlugin;

impl GraphAlgorithm for GraphDensityPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("graph-density", "1.0.0", "计算图的密度")
            .with_author("Your Name")
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["statistics", "density"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = ctx.graph.node_count();
        let m = ctx.graph.edge_count();

        let density = if n > 1 {
            let max_edges = if ctx.graph.is_directed() {
                n * (n - 1)
            } else {
                n * (n - 1) / 2
            };
            m as f64 / max_edges as f64
        } else {
            0.0
        };

        Ok(AlgorithmResult::new("density", AlgorithmData::Scalar(density))
            .with_metadata("nodes", n.to_string())
            .with_metadata("edges", m.to_string()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

### 示例 2: 带配置的算法

```rust
//! 带阈值过滤的 PageRank

use god_graph::plugins::algorithm::{
    GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData,
    ConfigField, ConfigFieldType,
};
use god_graph::vgi::{VirtualGraph, Capability, GraphType, VgiResult};
use std::any::Any;
use std::collections::HashMap;

pub struct FilteredPageRankPlugin {
    damping: f64,
    max_iterations: usize,
}

impl FilteredPageRankPlugin {
    pub fn new(damping: f64, max_iterations: usize) -> Self {
        Self { damping, max_iterations }
    }
}

impl GraphAlgorithm for FilteredPageRankPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("filtered-pagerank", "1.0.0", "带阈值过滤的 PageRank")
            .with_author("Your Name")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["centrality", "ranking", "filtered"])
            .with_config_fields(vec![
                ConfigField::new("threshold", ConfigFieldType::Float)
                    .default_value("0.01")
                    .description("过滤阈值"),
                ConfigField::new("damping", ConfigFieldType::Float)
                    .default_value("0.85")
                    .description("阻尼系数"),
            ])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let threshold = ctx.get_config_as("threshold", 0.01);
        let damping = ctx.get_config_as("damping", self.damping);
        let max_iter = ctx.get_config_as("max_iter", self.max_iterations);

        // 计算 PageRank
        let scores = self.compute_pagerank(ctx.graph, damping, max_iter)?;

        // 过滤低分节点
        let filtered: HashMap<usize, f64> = scores
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect();

        Ok(AlgorithmResult::new("filtered_pagerank", AlgorithmData::NodeValues(filtered))
            .with_metadata("threshold", threshold.to_string())
            .with_metadata("damping", damping.to_string()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl FilteredPageRankPlugin {
    fn compute_pagerank<G>(&self, graph: &G, damping: f64, max_iter: usize) -> VgiResult<HashMap<usize, f64>>
    where
        G: VirtualGraph + ?Sized,
    {
        // 实现 PageRank 算法
        // ...
        Ok(HashMap::new())
    }
}
```

### 示例 3: 支持进度报告的算法

```rust
use god_graph::plugins::algorithm::{
    GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData,
};
use god_graph::vgi::{VirtualGraph, Capability, GraphType, VgiResult};
use std::any::Any;

pub struct ProgressiveBfsPlugin {
    start_node: usize,
}

impl ProgressiveBfsPlugin {
    pub fn new(start_node: usize) -> Self {
        Self { start_node }
    }
}

impl GraphAlgorithm for ProgressiveBfsPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("progressive-bfs", "1.0.0", "支持进度报告的 BFS")
            .with_author("Your Name")
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["traversal", "bfs", "progressive"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = ctx.graph.node_count();
        let mut visited_count = 0;

        // BFS 实现
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back(self.start_node);
        visited.insert(self.start_node);

        while let Some(node) = queue.pop_front() {
            visited_count += 1;

            // 报告进度
            let progress = visited_count as f32 / n as f32;
            ctx.report_progress(progress);

            // 检查是否取消或超时
            if !ctx.can_continue() {
                break;
            }

            // 遍历邻居
            // ...
        }

        Ok(AlgorithmResult::new("bfs_visited", AlgorithmData::Scalar(visited_count as f64))
            .with_metadata("completed", ctx.can_continue().to_string()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

---

## 🧪 测试插件

### 单元测试示例

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use god_graph::graph::Graph;

    #[test]
    fn test_plugin_info() {
        let plugin = MyPlugin::new(0.5);
        let info = plugin.info();

        assert_eq!(info.name, "my-plugin");
        assert_eq!(info.version, "1.0.0");
    }

    #[test]
    fn test_plugin_execute() {
        let mut graph = Graph::<String, f64>::directed();
        graph.add_node("A".to_string()).unwrap();
        graph.add_node("B".to_string()).unwrap();
        graph.add_edge(0, 1, 1.0).unwrap();

        let plugin = MyPlugin::new(0.5);
        let mut ctx = PluginContext::new(&graph);
        let result = plugin.execute(&mut ctx);

        assert!(result.is_ok());
    }

    #[test]
    fn test_plugin_with_empty_graph() {
        let graph = Graph::<String, f64>::directed();
        let plugin = MyPlugin::new(0.5);
        let mut ctx = PluginContext::new(&graph);
        let result = plugin.execute(&mut ctx);

        // 测试空图处理
        assert!(result.is_ok());
    }
}
```

---

## 🎯 最佳实践

### 1. 配置验证

```rust
fn validate_config(ctx: &PluginContext<G>) -> VgiResult<()> {
    let info = self.info();
    ctx.validate_config(&info.config_schema)?;
    Ok(())
}
```

### 2. 取消和超时处理

```rust
fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
where
    G: VirtualGraph + ?Sized,
{
    for item in large_iterable {
        // 定期检查是否可以继续
        if !ctx.can_continue() {
            return Ok(AlgorithmResult::new("partial", AlgorithmData::String(
                "Execution cancelled or timed out".to_string()
            )));
        }
        
        // 处理 item
    }
    
    Ok(final_result)
}
```

### 3. 进度报告

```rust
let total = work_items.len();
for (i, item) in work_items.iter().enumerate() {
    process(item);
    
    let progress = (i + 1) as f32 / total as f32;
    ctx.report_progress(progress);
}
```

### 4. 错误处理

```rust
fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
where
    G: VirtualGraph + ?Sized,
{
    // 验证输入
    if ctx.graph.node_count() == 0 {
        return Err(VgiError::NotFound {
            message: "Graph is empty".to_string(),
        });
    }

    // 执行算法
    match self.compute(ctx.graph) {
        Ok(result) => Ok(AlgorithmResult::new("success", result)),
        Err(e) => Err(e),
    }
}
```

---

## 📦 发布插件

### 1. 创建独立 crate

```toml
# Cargo.toml
[package]
name = "god-graph-my-plugin"
version = "1.0.0"
edition = "2021"

[dependencies]
god-graph = { version = "0.6", features = ["plugins"] }
```

### 2. 编写文档

```rust
//! # My God-Graph Plugin
//!
//! 这是一个 God-Graph 插件，提供 XXX 功能。
//!
//! ## 使用示例
//!
//! ```rust
//! use god_graph_my_plugin::MyPlugin;
//! use god_graph::plugins::GraphAlgorithm;
//!
//! let plugin = MyPlugin::new(0.5);
//! println!("Plugin: {}", plugin.info().name);
//! ```
```

### 3. 发布到 crates.io

```bash
cargo publish
```

---

## 🔧 调试技巧

### 1. 日志记录

```rust
use log::{info, debug, error};

fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult> {
    info!("Executing plugin: {}", self.info().name);
    debug!("Graph has {} nodes", ctx.graph.node_count());
    
    // ...
    
    error!("Something went wrong");
    Ok(result)
}
```

### 2. 性能分析

```rust
use std::time::Instant;

fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult> {
    let start = Instant::now();
    
    let result = self.compute(ctx.graph)?;
    
    let duration = start.elapsed();
    println!("Execution time: {:?}", duration);
    
    Ok(result)
}
```

---

## ❓ 常见问题

### Q: 如何选择合适的算法数据类型？

A: 根据结果类型选择：
- 节点分数/排名 → `AlgorithmData::NodeValues`
- 路径/序列 → `AlgorithmData::NodeList`
- 标量结果 → `AlgorithmData::Scalar`
- 社区划分 → `AlgorithmData::Communities`

### Q: 如何处理大型图？

A: 
1. 使用增量更新
2. 实现进度报告
3. 支持取消和超时
4. 考虑并行化

### Q: 插件可以访问图的可变引用吗？

A: 不可以。插件只能通过 `VirtualGraph` trait 的只读方法访问图。

---

## 📚 参考资源

- [God-Graph 源码](https://github.com/your-repo/god-graph)
- [VirtualGraph trait 文档](https://docs.rs/god-graph)
- [示例插件集合](https://github.com/your-repo/god-graph-plugins)

---

## 🤝 贡献

欢迎贡献您的插件！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

---

**最后更新**: 2026-03-31
