# 插件开发指南

**版本**: v0.6.0-alpha  
**日期**: 2026-03-31  
**状态**: 已完成

---

## 📖 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [插件架构](#插件架构)
4. [开发步骤](#开发步骤)
5. [最佳实践](#最佳实践)
6. [示例](#示例)
7. [调试与测试](#调试与测试)
8. [发布插件](#发布插件)

---

## 概述

God-Graph 插件系统允许开发者扩展图算法，无需修改核心库代码。插件系统采用类型擦除技术，支持动态注册和执行。

### 核心特性

- **热插拔**: 运行时注册和注销插件
- **类型安全**: 编译时检查，运行时验证
- **能力发现**: 自动检查图是否满足算法要求
- **生命周期管理**: 完整的执行前后回调

### 适用场景

- 实现新的图算法
- 添加自定义分析功能
- 集成第三方算法库
- 实验性算法原型

---

## 快速开始

### 1. 添加依赖

```toml
[dependencies]
god-graph = "0.6"
```

### 2. 创建第一个插件

```rust
use god_graph::plugins::{
    GraphAlgorithm, PluginInfo, PluginContext, 
    AlgorithmResult, AlgorithmData,
};
use god_graph::vgi::{VirtualGraph, VgiResult};
use std::any::Any;
use std::collections::HashMap;

/// 计算图的节点数量
struct NodeCountPlugin;

impl GraphAlgorithm for NodeCountPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("node_count", "1.0.0", "计算图的节点数量")
            .with_author("Your Name")
            .with_tags(&["basic", "statistics"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let count = ctx.graph.node_count();
        Ok(AlgorithmResult::scalar(count as f64))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

### 3. 注册和执行插件

```rust
use god_graph::plugins::PluginRegistry;
use god_graph::graph::Graph;

fn main() {
    // 创建注册表
    let mut registry = PluginRegistry::new();
    
    // 注册插件
    registry.register_algorithm("node_count", NodeCountPlugin).unwrap();
    
    // 创建图
    let mut graph = Graph::<String, f64>::directed();
    graph.add_node("A".to_string()).unwrap();
    graph.add_node("B".to_string()).unwrap();
    
    // 执行插件
    let mut ctx = PluginContext::new(&graph);
    let result = registry.execute::<Graph<String, f64>, NodeCountPlugin>(
        "node_count",
        &graph,
        &mut ctx,
    ).unwrap();
    
    println!("Node count: {:?}", result.data.as_scalar());
}
```

---

## 插件架构

### Trait 层次结构

```text
GraphAlgorithm (用户实现)
    │
    ├── info() → PluginInfo          # 插件元数据
    ├── validate() → VgiResult<()>   # 验证（可选覆盖）
    ├── before_execute() → VgiResult<()>  # 执行前回调（可选）
    ├── execute() → AlgorithmResult  # 核心执行逻辑
    ├── after_execute() → VgiResult<()>   # 执行后回调（可选）
    └── as_any() → &dyn Any          # 类型转换
```

### 数据流

```text
用户调用
    │
    ▼
PluginRegistry::execute()
    │
    ├── 1. 查找插件
    ├── 2. 类型验证
    ├── 3. validate()
    ├── 4. before_execute()
    ├── 5. execute()  ← 用户实现
    ├── 6. after_execute()
    │
    ▼
返回 AlgorithmResult
```

---

## 开发步骤

### 步骤 1: 定义插件结构

```rust
pub struct MyAlgorithmPlugin {
    /// 算法参数
    param1: f64,
    param2: usize,
}

impl MyAlgorithmPlugin {
    pub fn new(param1: f64, param2: usize) -> Self {
        Self { param1, param2 }
    }
}
```

### 步骤 2: 实现 PluginInfo

```rust
impl GraphAlgorithm for MyAlgorithmPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("my_algorithm", "1.0.0", "我的算法描述")
            .with_author("Your Name <your@email.com>")
            .with_required_capabilities(&[
                Capability::Parallel,
                Capability::WeightedEdges,
            ])
            .with_supported_graph_types(&[
                GraphType::Directed,
                GraphType::Undirected,
            ])
            .with_tags(&["category", "subcategory"])
    }
    
    // ... 其他方法
}
```

### 步骤 3: 实现 execute 方法

```rust
fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
where
    G: VirtualGraph + ?Sized,
{
    // 1. 读取配置参数
    let param1 = ctx.get_config_as("param1", self.param1);
    let max_iter = ctx.get_config_as("max_iter", 100);
    
    // 2. 报告进度
    ctx.report_progress(0.1);
    
    // 3. 实现算法逻辑
    let mut results = HashMap::new();
    
    for node_ref in ctx.graph.nodes() {
        // 检查取消
        if ctx.is_cancelled() {
            return Err(VgiError::Internal {
                message: "Execution cancelled".to_string(),
            });
        }
        
        // 计算...
        results.insert(node_ref.index(), 0.0);
    }
    
    // 4. 报告完成
    ctx.report_progress(1.0);
    
    // 5. 返回结果
    Ok(AlgorithmResult::new(
        "my_algorithm",
        AlgorithmData::NodeValues(results),
    ))
}
```

### 步骤 4: 实现 as_any

```rust
fn as_any(&self) -> &dyn Any {
    self
}
```

---

## 最佳实践

### 1. 错误处理

```rust
// ✅ 好的做法：提供有意义的错误信息
if !ctx.graph.contains_node(start) {
    return Err(VgiError::Internal {
        message: format!("Start node {} not found", start),
    });
}

// ❌ 避免：模糊的错误
if error {
    return Err(VgiError::Internal {
        message: "Error occurred".to_string(),
    });
}
```

### 2. 进度报告

```rust
// 在关键步骤报告进度
ctx.report_progress(0.0);  // 开始
ctx.report_progress(0.25); // 25% 完成
ctx.report_progress(0.5);  // 50% 完成
ctx.report_progress(1.0);  // 完成
```

### 3. 取消检查

```rust
// 在循环中定期检查取消
for node_ref in ctx.graph.nodes() {
    if ctx.is_cancelled() {
        return Err(VgiError::Internal {
            message: "Execution cancelled by user".to_string(),
        });
    }
    // ... 处理节点
}
```

### 4. 日志记录

```rust
// 使用 log crate 记录关键信息
log::info!("Starting algorithm execution");
log::debug!("Processing node: {}", node_idx);
log::warn!("Large graph detected: {} nodes", node_count);
log::error!("Algorithm failed: {}", error_message);
```

### 5. 结果元数据

```rust
// 添加丰富的元数据便于调试
AlgorithmResult::new("pagerank", AlgorithmData::NodeValues(scores))
    .with_metadata("iterations", iterations.to_string())
    .with_metadata("converged", converged.to_string())
    .with_metadata("damping", damping.to_string())
```

---

## 示例

### 示例 1: 统计插件

```rust
/// 计算图的平均度
struct AverageDegreePlugin;

impl GraphAlgorithm for AverageDegreePlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("average_degree", "1.0.0", "计算图的平均度")
            .with_tags(&["statistics", "basic"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = ctx.graph.node_count();
        if n == 0 {
            return Ok(AlgorithmResult::scalar(0.0));
        }
        
        let mut total_degree = 0;
        for node_ref in ctx.graph.nodes() {
            total_degree += ctx.graph.out_degree(node_ref.index()).unwrap_or(0);
        }
        
        let avg = total_degree as f64 / n as f64;
        Ok(AlgorithmResult::scalar(avg))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

### 示例 2: 路径查找插件

```rust
/// 查找两点之间的所有路径
struct AllPathsPlugin {
    max_paths: usize,
}

impl GraphAlgorithm for AllPathsPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("all_paths", "1.0.0", "查找所有路径")
            .with_tags(&["pathfinding", "traversal"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let start = ctx.get_config_as("start", 0);
        let end = ctx.get_config_as("end", 0);
        
        let mut paths = Vec::new();
        let mut current_path = vec![start];
        let mut visited = std::collections::HashSet::new();
        
        self.dfs_all_paths(ctx.graph, start, end, &mut current_path, 
                          &mut visited, &mut paths, self.max_paths);
        
        Ok(AlgorithmResult::new(
            "all_paths",
            AlgorithmData::Custom(format!("{:?}", paths),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

### 示例 3: 社区检测插件

```rust
/// 简单的基于度的社区检测
struct DegreeCommunityPlugin;

impl GraphAlgorithm for DegreeCommunityPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("degree_community", "1.0.0", "基于度的社区检测")
            .with_tags(&["community", "clustering"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let mut communities = Vec::new();
        let mut node_community = HashMap::new();
        
        // 按度分组
        let mut degree_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        
        for node_ref in ctx.graph.nodes() {
            let degree = ctx.graph.out_degree(node_ref.index()).unwrap_or(0);
            degree_groups.entry(degree).or_default().push(node_ref.index());
        }
        
        // 每个度级别作为一个社区
        for (degree, nodes) in degree_groups {
            for node in nodes {
                node_community.insert(node, communities.len());
            }
            communities.push(nodes);
        }
        
        Ok(AlgorithmResult::new(
            "communities",
            AlgorithmData::Communities(
                (0..ctx.graph.node_count())
                    .map(|i| *node_community.get(&i).unwrap_or(&0))
                    .collect(),
            ),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

---

## 调试与测试

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use god_graph::graph::Graph;

    #[test]
    fn test_my_algorithm() {
        let mut graph = Graph::<String, f64>::directed();
        graph.add_node("A".to_string()).unwrap();
        graph.add_node("B".to_string()).unwrap();
        graph.add_edge(0, 1, 1.0).unwrap();

        let plugin = MyAlgorithmPlugin::new(0.5, 10);
        let mut ctx = PluginContext::new(&graph);
        
        let result = plugin.execute(&mut ctx);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.name, "my_algorithm");
    }

    #[test]
    fn test_plugin_info() {
        let plugin = MyAlgorithmPlugin::new(0.5, 10);
        let info = plugin.info();
        
        assert_eq!(info.name, "my_algorithm");
        assert_eq!(info.version, "1.0.0");
    }
}
```

### 集成测试

```rust
use god_graph::plugins::{PluginRegistry, MyAlgorithmPlugin};

#[test]
fn test_plugin_registry_integration() {
    let mut registry = PluginRegistry::new();
    registry.register_algorithm("my_algo", MyAlgorithmPlugin::new(0.5, 10)).unwrap();
    
    assert!(registry.is_registered("my_algo"));
    
    let info = registry.get_algorithm_info("my_algo");
    assert!(info.is_some());
    assert_eq!(info.unwrap().name, "my_algorithm");
}
```

### 调试技巧

1. **启用日志**: 在 `Cargo.toml` 中添加 `log` 和 `env_logger`
2. **使用断点**: 在 `execute` 方法中设置断点
3. **打印中间结果**: 使用 `eprintln!` 输出调试信息
4. **性能分析**: 使用 `cargo bench` 进行性能测试

---

## 发布插件

### 1. 创建独立 Crate

```toml
# Cargo.toml
[package]
name = "god-graph-my-algorithm"
version = "0.1.0"
edition = "2021"

[dependencies]
god-graph = "0.6"
```

### 2. 编写文档

```rust
//! My Algorithm Plugin for God-Graph
//!
//! # Example
//!
//! ```rust
//! use god_graph_my_algorithm::MyAlgorithmPlugin;
//!
//! let plugin = MyAlgorithmPlugin::default();
//! ```
```

### 3. 发布到 crates.io

```bash
cargo publish
```

### 4. 添加到社区列表

在 God-Graph 仓库的 `docs/COMMUNITY_PLUGINS.md` 中添加你的插件信息。

---

## 常见问题

### Q: 如何处理大型图？

A: 使用迭代器而非收集所有节点到向量，定期检查取消标志。

### Q: 插件可以访问外部资源吗？

A: 可以，但建议在 `PluginContext` 中添加资源句柄。

### Q: 如何实现并行算法？

A: 检查 `Capability::Parallel`，使用 `rayon` 等并行库。

### Q: 插件支持热加载吗？

A: 当前版本不支持，计划中。

---

## 参考资源

- [API 文档](https://docs.rs/god-graph)
- [示例代码](https://github.com/god-graph/examples)
- [问题追踪](https://github.com/god-graph/issues)

---

**最后更新**: 2026-03-31
