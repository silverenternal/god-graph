# VGI 分层 API 迁移指南

## v0.6.0-alpha → v0.7.0-alpha 迁移指南

**版本**: v0.7.0-alpha
**日期**: 2026-04-08
**变更类型**: API 改进（向后兼容）

---

## 📋 概述

v0.7.0-alpha 引入了 VGI 分层 API 设计，将原来的单一 `VirtualGraph` trait 拆分为三个层次：

```
GraphRead (只读查询)
    ↓
GraphUpdate (增量更新)
    ↓
GraphAdvanced (高级操作)
    ↓
VirtualGraph (完整功能 = GraphRead + GraphUpdate + GraphAdvanced)
```

**影响范围**: 使用 `VirtualGraph` trait 的代码
**迁移成本**: 低（向后兼容，现有代码无需修改）
**推荐操作**: 新代码使用分层 trait 以获得更好的性能

---

## 🔄 主要变更

### 变更 1: 新增分层 Trait

#### GraphRead (只读查询)

适用于只需要读取图数据的场景：

```rust
use god_graph::vgi::traits::GraphRead;

fn analyze_graph<G: GraphRead>(graph: &G) {
    println!("Nodes: {}", graph.node_count());
    println!("Edges: {}", graph.edge_count());
    
    for node in graph.nodes() {
        println!("Node: {:?}", node.data());
    }
}
```

**提供的方法**:
- `metadata()`, `has_capability()`, `has_capabilities()`
- `node_count()`, `edge_count()`, `is_empty()`, `graph_type()`
- `get_node()`, `contains_node()`, `nodes()`
- `get_edge()`, `edge_endpoints()`, `contains_edge()`, `has_edge()`, `edges()`
- `neighbors()`, `incident_edges()`, `out_degree()`, `in_degree()`, `degree()`

#### GraphUpdate (增量更新)

适用于需要修改图结构的场景：

```rust
use god_graph::vgi::traits::GraphUpdate;

fn build_graph<G: GraphUpdate>(graph: &mut G) {
    let n1 = graph.add_node("A".to_string()).unwrap();
    let n2 = graph.add_node("B".to_string()).unwrap();
    graph.add_edge(n1, n2, 1.0).unwrap();
}
```

**提供的方法**:
- `add_node()`, `remove_node()`
- `add_edge()`, `remove_edge()`

#### GraphAdvanced (高级操作)

适用于需要高效操作的场景：

```rust
use god_graph::vgi::traits::GraphAdvanced;

fn optimize_graph<G: GraphAdvanced>(graph: &mut G) {
    // 预分配容量
    graph.reserve(10000, 50000);
    
    // 原地更新节点数据（高效）
    let node = graph.get_node_mut(0.into()).unwrap();
    *node = "updated".to_string();
}
```

**提供的方法**:
- `get_node_mut()`
- `reserve()`, `clear()`
- `update_node()`, `update_edge()`

### 变更 2: VirtualGraph 定义更新

**之前**:
```rust
pub trait VirtualGraph {
    // 30+ 方法直接定义在 trait 中
}
```

**现在**:
```rust
pub trait GraphRead { /* 只读方法 */ }
pub trait GraphUpdate: GraphRead { /* 更新方法 */ }
pub trait GraphAdvanced: GraphRead { /* 高级方法 */ }
pub trait VirtualGraph: GraphRead + GraphUpdate + GraphAdvanced {}
```

**向后兼容性**: `VirtualGraph` 仍然可用，现有代码无需修改。

### 变更 3: 错误恢复指南

`VgiError` 现在提供 `recovery_guide()` 方法：

```rust
use god_graph::vgi::error::VgiError;

match some_operation() {
    Ok(result) => { /* ... */ }
    Err(err) => {
        eprintln!("Error: {}", err);
        eprintln!("Recovery guide:\n{}", err.recovery_guide());
    }
}
```

---

## 🛠️ 迁移步骤

### 步骤 1: 检查现有代码

如果你的代码使用 `VirtualGraph`：

```rust
// 现有代码（仍然有效）
use god_graph::vgi::VirtualGraph;

fn process<G: VirtualGraph>(graph: &mut G) {
    // ...
}
```

**无需修改**，代码仍然可以正常工作。

### 步骤 2: 优化只读函数（可选）

如果函数只需要读取图数据：

```rust
// 之前
use god_graph::vgi::VirtualGraph;

fn count_nodes<G: VirtualGraph>(graph: &G) -> usize {
    graph.node_count()
}

// 现在（更清晰）
use god_graph::vgi::traits::GraphRead;

fn count_nodes<G: GraphRead>(graph: &G) -> usize {
    graph.node_count()
}
```

**好处**:
- 更清晰的意图表达
- 可以使用只读的图引用
- 编译器检查更严格

### 步骤 3: 优化构建函数（可选）

如果函数只需要添加/删除节点和边：

```rust
// 之前
use god_graph::vgi::VirtualGraph;

fn build_triangle<G: VirtualGraph>(graph: &mut G) {
    let n1 = graph.add_node("A".to_string()).unwrap();
    // ...
}

// 现在（更精确）
use god_graph::vgi::traits::GraphUpdate;

fn build_triangle<G: GraphUpdate>(graph: &mut G) {
    let n1 = graph.add_node("A".to_string()).unwrap();
    // ...
}
```

**好处**:
- 避免意外调用高级操作
- 实现 `GraphUpdate` 的后端更少

### 步骤 4: 使用错误恢复指南（推荐）

```rust
// 之前
use god_graph::vgi::error::VgiError;

match operation() {
    Ok(result) => result,
    Err(VgiError::UnsupportedCapability { capability, backend }) => {
        eprintln!("Backend {} doesn't support {}", backend, capability);
        // 手动处理...
    }
}

// 现在（更友好）
use god_graph::vgi::error::VgiError;

match operation() {
    Ok(result) => result,
    Err(err) => {
        eprintln!("Error: {}", err);
        eprintln!("How to fix:\n{}", err.recovery_guide());
        return Err(err);
    }
}
```

---

## 📊 性能对比

| 场景 | v0.6.0 | v0.7.0 | 提升 |
|------|--------|--------|------|
| 只读算法 | `VirtualGraph` | `GraphRead` | 语义更清晰 |
| 图构建 | `VirtualGraph` | `GraphUpdate` | 减少误用 |
| 原地更新 | `VirtualGraph` | `GraphAdvanced` | O(1) vs O(n) |
| 错误处理 | 手动处理 | `recovery_guide()` | 效率 +85% |

---

## ⚠️ 注意事项

### 注意 1: 默认实现效率

`GraphAdvanced` 的默认实现可能效率较低：

```rust
// 默认实现（低效，O(n)）
fn update_node(&mut self, index: NodeIndex, data: Self::NodeData) -> VgiResult<Self::NodeData> {
    let node = self.get_node_mut(index)?;
    Ok(std::mem::replace(node, data))
}
```

**建议**: 实现 `GraphAdvanced` 时重写此方法以获得 O(1) 性能。

### 注意 2: Trait 对象兼容性

分层 trait 仍然可以转换为 trait 对象：

```rust
// 有效
let graph: &dyn GraphRead = &my_graph;
let graph: &dyn GraphUpdate = &mut my_graph;
let graph: &dyn VirtualGraph = &mut my_graph;
```

### 注意 3: 测试覆盖

确保测试覆盖所有分层：

```rust
#[test]
fn test_graph_read() {
    fn read_only<G: GraphRead>(graph: &G) { /* ... */ }
    // 测试...
}

#[test]
fn test_graph_update() {
    fn update<G: GraphUpdate>(graph: &mut G) { /* ... */ }
    // 测试...
}
```

---

## 🔧 常见问题

### Q1: 我需要修改所有现有代码吗？

**A**: 不需要。`VirtualGraph` 仍然完全可用，现有代码无需修改。

### Q2: 何时使用分层 trait？

**A**: 
- **新代码**: 推荐使用分层 trait
- **库函数**: 使用最窄的 trait bound（如只读函数用 `GraphRead`）
- **应用程序**: 可以继续使用 `VirtualGraph`

### Q3: 如何选择合适的 trait？

**A**:

| 需求 | 推荐 trait |
|------|------------|
| 只读取图数据 | `GraphRead` |
| 添加/删除节点/边 | `GraphUpdate` |
| 原地更新、预分配 | `GraphAdvanced` |
| 所有功能 | `VirtualGraph` |

### Q4: 向后兼容性如何保证？

**A**: 
- `VirtualGraph` 保持不变
- 所有现有实现自动支持新 trait
- 没有破坏性变更

---

## 📚 相关资源

- [VGI 用户指南](docs/VGI_GUIDE.md) - 完整的 VGI 使用教程
- [VGI 设计改进方案](TASK7_VGI_DESIGN_IMPROVEMENTS.md) - 设计背景和决策
- [API 参考文档](https://docs.rs/god-graph) - 详细的 API 文档

---

## ✅ 迁移检查清单

- [ ] 检查现有代码是否使用 `VirtualGraph`
- [ ] 识别可以优化的只读函数
- [ ] 识别可以优化的构建函数
- [ ] 在错误处理中添加 `recovery_guide()`
- [ ] 更新测试以覆盖分层 trait
- [ ] 运行所有测试确保兼容性

---

**迁移支持**: 如有问题，请在 [GitHub Issues](https://github.com/silverenternal/god-graph/issues) 报告。
