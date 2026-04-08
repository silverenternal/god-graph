# Task 3: API Return Type Unification

## Summary
统一并改进了 god-graph 项目中的 API 返回类型，提升了错误处理的健壮性和一致性。

## Changes Made

### 1. 修复 Tensor 矩阵操作的错误类型 (`src/tensor/ops.rs`)

**问题:** `inverse()` 和 `determinant()` 函数使用 `&'static str` 作为错误类型，与其他 Tensor API 不一致。

**修复:**
- 导入 `TensorError` 类型
- 将 `inverse()` 返回类型改为 `Result<DenseTensor, TensorError>`
- 将 `determinant()` 返回类型改为 `Result<f64, TensorError>`
- 添加新的 `TensorError::MatrixError` 变体用于矩阵操作错误

**代码变更:**
```rust
// Before
pub fn inverse(tensor: &DenseTensor) -> Result<DenseTensor, &'static str>

// After
pub fn inverse(tensor: &DenseTensor) -> Result<DenseTensor, TensorError>
```

### 2. 统一边数据访问 API (`src/tensor/unified_graph.rs`)

**问题:** `get_edge_data()` 和 `get_edge_data_mut()` 返回 `Option` 而非 `Result`，无法区分"边不存在"和其他错误。

**修复:**
- `get_edge_data()` 改为返回 `Result<&EdgeData, GraphError>`
- `get_edge_data_mut()` 改为返回 `Result<&mut EdgeData, GraphError>`
- 更新所有调用点使用 `?` 操作符或 `.ok()` 进行错误处理

**代码变更:**
```rust
// Before
pub fn get_edge_data(&self, edge_idx: usize) -> Option<&EdgeData>

// After
pub fn get_edge_data(&self, edge_idx: usize) -> Result<&EdgeData, GraphError>
```

### 3. 改进 Tensor 内存池 API (`src/tensor/pool.rs`)

**问题:** `get()` 和 `take()` 返回 `Option`，调用者无法知道为什么张量不存在。

**修复:**
- `get()` 改为返回 `Result<&DenseTensor, TensorError>`
- `take()` 改为返回 `Result<DenseTensor, TensorError>`
- 使用 `TensorError::MatrixError` 提供详细的错误信息

**代码变更:**
```rust
// Before
pub fn get(&self, id: usize) -> Option<&DenseTensor>

// After
pub fn get(&self, id: usize) -> Result<&DenseTensor, TensorError>
```

### 4. 改进算法结果 API (`src/parallel/algorithms/`)

**问题:** BFS、Dijkstra、PageRank 的距离/排名查询返回 `Option`，无法区分"节点不存在"和"节点不可达"。

**修复:**

#### BFS (`src/parallel/algorithms/bfs.rs`)
```rust
// Before
pub fn distance(&self, node: NodeIndex) -> Option<usize>

// After
pub fn distance(&self, node: NodeIndex) -> Result<usize, GraphError>
```

#### Dijkstra (`src/parallel/algorithms/dijkstra.rs`)
```rust
// Before
pub fn distance(&self, node: NodeIndex) -> Option<f64>

// After
pub fn distance(&self, node: NodeIndex) -> Result<f64, GraphError>
```

#### PageRank (`src/parallel/algorithms/pagerank.rs`)
```rust
// Before
pub fn rank(&self, node: NodeIndex) -> Option<f64>

// After
pub fn rank(&self, node: NodeIndex) -> Result<f64, GraphError>
```

### 5. 文档化简化 API 的 Panic 行为 (`src/transformer/perf.rs`)

**问题:** `TransformerMemoryPool` 的 buffer 访问方法使用 `.unwrap()`，但未文档化为何可以接受。

**修复:**
- 添加 `# Panics` 文档章节解释为何 panic 是可接受的
- 添加 `#[must_use]` 属性
- 说明这些方法在内存分配失败前都会成功

**代码变更:**
```rust
/// Get or allocate attention score buffer
///
/// # Panics
///
/// This method should never panic as it allocates the buffer if needed.
/// Panic would only occur if memory allocation fails.
#[must_use]
pub fn get_attn_score_buffer(&mut self) -> &mut Vec<f64> {
    // ...
}
```

### 6. 文档化简化 API 的设计决策 (`src/vgi/simple.rs`)

**问题:** `SimpleGraph` 的 API 设计选择（panic 或返回 `Option`）未充分文档化。

**修复:**
- `add_node()`: 添加 `# Panics` 和 `# Design Note` 解释为何使用 `.expect()`
- `add_edge()`: 改为返回 `Option<EdgeIndex>` 并添加文档
- `get_node()` / `get_node_mut()`: 添加 `# Note` 解释为何返回 `Option`

**代码变更:**
```rust
/// 添加节点
///
/// # Panics
///
/// 如果节点索引溢出（极少见，需要添加超过 2^32 个节点）
///
/// # Design Note
///
/// 此方法在溢出时 panic 而非返回 `Result`，是因为：
/// 1. 节点索引溢出在实际应用中几乎不可能发生（需要 2^32 个节点）
/// 2. 简化 API 调用，避免不必要的错误处理
/// 3. 如果确实需要处理溢出，请使用底层的 `Graph` 类型
pub fn add_node(&mut self, data: T) -> NodeIndex {
    // ...
}
```

### 7. 添加新错误类型 (`src/tensor/error.rs`)

**新增:** `TensorError::MatrixError` 变体

```rust
/// 矩阵操作错误
MatrixError {
    /// 错误描述
    message: String,
},
```

用于矩阵求逆、行列式等操作的错误。

### 8. 恢复缺失的 Feature Flags (`Cargo.toml`)

**问题:** 在 feature flag 精简过程中，`tensor-sparse` 和 `tensor-autograd` 被移除但代码中仍有引用。

**修复:**
```toml
# Sparse Tensor 支持（COO/CSR 格式，需要 tensor）
tensor-sparse = ["tensor"]

# 自动微分支持（需要 tensor）
tensor-autograd = ["tensor"]
```

## Files Modified

1. `src/tensor/error.rs` - 添加 `MatrixError` 变体
2. `src/tensor/ops.rs` - 修复 `inverse()` 和 `determinant()` 错误类型
3. `src/tensor/unified_graph.rs` - 统一 `get_edge_data*()` 返回类型
4. `src/tensor/pool.rs` - 改进 `get()` 和 `take()` 返回类型
5. `src/parallel/algorithms/bfs.rs` - 改进 `distance()` 返回类型
6. `src/parallel/algorithms/dijkstra.rs` - 改进 `distance()` 返回类型
7. `src/parallel/algorithms/pagerank.rs` - 改进 `rank()` 返回类型
8. `src/transformer/perf.rs` - 文档化 buffer 访问方法
9. `src/vgi/simple.rs` - 文档化简化 API 设计
10. `Cargo.toml` - 恢复缺失的 feature flags

## Verification

### Tests
✅ **295 tests pass** (增加了 9 个测试)
```
test result: ok. 295 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Error Handling Improvements

**Before:**
- Mix of `&'static str`, `Option<T>`, and `Result<T, E>` for similar operations
- No way to distinguish "not found" from other errors
- Inconsistent error types across modules

**After:**
- Consistent use of type-specific error types (`GraphError`, `TensorError`)
- `Result<T, E>` for operations that can fail with meaningful errors
- `Option<T>` only for truly optional values or simplified APIs
- Proper documentation for design trade-offs

## Impact

### Breaking Changes
⚠️ **Minor Breaking Changes** in the following APIs:

1. `TensorPool::get()` - Now returns `Result` instead of `Option`
2. `TensorPool::take()` - Now returns `Result` instead of `Option`
3. `BFSResult::distance()` - Now returns `Result` instead of `Option`
4. `DijkstraResult::distance()` - Now returns `Result` instead of `Option`
5. `PageRankResult::rank()` - Now returns `Result` instead of `Option`
6. `UnifiedGraph::get_edge_data()` - Now returns `Result` instead of `Option`
7. `UnifiedGraph::get_edge_data_mut()` - Now returns `Result` instead of `Option`

### Migration Guide

Users need to update call sites:

```rust
// Old code
if let Some(dist) = bfs_result.distance(node) {
    println!("Distance: {}", dist);
}

// New code
if let Ok(dist) = bfs_result.distance(node) {
    println!("Distance: {}", dist);
}

// Or using ? operator
let dist = bfs_result.distance(node)?;
```

## Benefits

1. **Better Error Messages:** Users get detailed error information instead of `None`
2. **Consistent API:** All modules use the same error handling patterns
3. **Easier Debugging:** Error variants provide context (e.g., which node was not found)
4. **Type Safety:** Compiler enforces error handling at compile time
5. **Documented Trade-offs:** Simplified APIs clearly explain design decisions

## Related Issues

- Task 4: ✅ Unsafe Code Documentation (completed)
- Task 7: ⏳ VGI Design Improvements (pending)
- Task 10: ✅ Error Handling Unification (completed)
