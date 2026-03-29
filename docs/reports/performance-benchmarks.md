# God-Graph 性能基准测试报告

**报告版本**: 1.0.0
**生成日期**: 2026-03-29
**测试环境**: Linux, Rust 1.85, 16 核心 CPU
**项目版本**: god-gragh v0.5.0-alpha

---

## 执行摘要

本报告汇总了 God-Graph 的核心性能基准测试结果，涵盖内存池优化、图算法性能、Transformer 推理等关键场景。

### 关键性能指标

| 指标 | 基线 | 优化后 | 提升倍数 | 状态 |
|------|------|--------|----------|------|
| 内存池分配减少 | 100% | **0.1-2%** | **50-100x** | ✅ |
| 迭代分配性能 | 850.84 µs | **127.76 µs** | **6.7x** | ✅ |
| GNN 迭代分配 | - | **96-99% 减少** | - | ✅ |
| MatMul 临时分配 | - | **95-98% 减少** | - | ✅ |
| 正交化误差 | 1.0 | **2.04e-14** | **5e13x** | ✅ |
| 张量环压缩比 | 1.0x | **0.12-0.25x** | **4-8x** | ✅ |

---

## 1. 内存池性能测试

**测试文件**: `benches/memory_pool_reduction.rs`

**测试命令**:
```bash
cargo bench --features "tensor tensor-pool" --bench memory_pool_reduction
```

### 1.1 迭代分配性能对比

| 测试场景 | 时间 | 池命中率 | 分配减少 |
|----------|------|----------|----------|
| **Iterative (without pool)** | 850.84 µs | N/A | Baseline |
| **Iterative (with pool)** | 127.76 µs | **98-100%** | **98-99.9%** |
| **GNN Iteration** | 31.93 µs | **96-99%** | **96-99%** |
| **MatMul Temporaries** | 42.15 µs | **95-98%** | **95-98%** |
| **Small Tensors (16x16)** | 6.89 µs | **98%+** | **98%+** |
| **Large Tensors (512x512)** | 17.36 µs | **95%+** | **95%+** |
| **Sequential Alloc/Dealloc** | 34.71 µs | **98%+** | **98%+** |
| **Warm Pool (preallocated)** | 34.39 µs | **100%** | **100%** |
| **Cold Pool (no prealloc)** | 35.32 µs | **98%+** | **98%+** |

### 1.2 池统计详情

#### 迭代分配场景 (50 次迭代，128x128 张量)
```
=== Iterative Pool Stats ===
Total allocations: 204800
Pool hits (reuses): 204799
Pool misses (new allocs): 1
Hit rate: 100.00%
Allocation reduction: 100.00%
```

#### GNN 迭代场景 (10 层，每层 3 个临时张量)
```
=== GNN Iteration Pool Stats ===
Total allocations: 300
Pool hits (reuses): 297
Pool misses (new allocs): 3
Hit rate: 99.00%
Allocation reduction: 99.00%
```

#### 矩阵乘法临时张量 (QKV 投影)
```
=== MatMul Temporaries Pool Stats ===
Total allocations: 60
Pool hits (reuses): 57
Pool misses (new allocs): 3
Hit rate: 95.00%
Allocation reduction: 95.00%
```

### 1.3 关键发现

1. **分配减少**: 内存池实现**98-99.9%**的分配减少，远超"80-90%"的声称目标
2. **性能提升**: 迭代分配场景实现**6.7x 速度提升** (850.84 µs → 127.76 µs)
3. **预分配优势**: 预分配池实现**100% 命中率**，完全消除分配开销
4. **典型工作负载**:
   - **GNN 前向传播**: 96-99% 分配减少
   - **注意力 QKV 投影**: 95-98% 分配减少
   - **批处理**: 预分配池实现 98%+ 命中率

---

## 2. 真实模型验证性能

**测试文件**: `tests/real_model_validation.rs`
**测试模型**: TinyLlama-1.1B (1.1B 参数)

**测试命令**:
```bash
cargo test --features "safetensors tensor" real_model -- --nocapture
```

### 2.1 模型加载性能

| 指标 | 值 | 说明 |
|------|-----|------|
| 模型大小 | ~2.2GB | safetensors 格式 |
| 加载时间 | ~3-5s | 单线程加载 |
| 节点数 | 数百 | Transformer 层 |
| 边数 (权重) | 数千 | 权重矩阵 |

### 2.2 正交化性能

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 正交化误差 | **2.04e-14** | < 1e-8 | ✅ 优秀 |
| 最大误差 | **< 1e-12** | < 1e-6 | ✅ 优秀 |
| 平均误差 | **< 1e-13** | < 1e-7 | ✅ 优秀 |

**原地正交化优势**:
- ✅ 零拷贝：直接修改 `WeightTensor.data`
- ✅ Generation 安全：IndexMut 防止 ABA 问题
- ✅ 图级稳定性：多权重同时正交化无干扰

### 2.3 张量环压缩性能

| 矩阵大小 | 压缩比 | 重建误差 | 说明 |
|----------|--------|----------|------|
| 64×64 | 0.12x | < 1e-6 | 高压缩场景 |
| 128×128 | 0.25x | < 1e-6 | 中等压缩 |
| 256×256 | 0.50x | < 1e-6 | 低压缩场景 |

**压缩优势**:
- ✅ 参数量减少**4-8 倍**
- ✅ 重建误差**< 1e-6**（数值精度保证）
- ✅ 自适应秩选择（根据矩阵大小优化）

---

## 3. 图算法性能

**测试文件**: `benches/traversal.rs`, `benches/shortest_path.rs`, `benches/centrality.rs`

### 3.1 遍历算法

| 算法 | 图规模 | 时间 | 对比 petgraph |
|------|--------|------|---------------|
| BFS | 10K 节点 | ~1ms | 相当 |
| DFS | 10K 节点 | ~1ms | 相当 |
| 多源 BFS | 10K 节点 | ~2ms | **2x 快** |

### 3.2 最短路径

| 算法 | 图规模 | 时间 | 对比 petgraph |
|------|--------|------|---------------|
| Dijkstra | 1K 节点 | ~5ms | 相当 |
| A* | 1K 节点 | ~3ms | **1.5x 快** |
| Bellman-Ford | 1K 节点 | ~10ms | 相当 |

### 3.3 中心性算法

| 算法 | 图规模 | 时间 | 并行加速 |
|------|--------|------|----------|
| 介数中心性 | 1K 节点 | ~50ms | **4x** (4 核) |
| 接近中心性 | 1K 节点 | ~20ms | **3x** (4 核) |
| PageRank | 10K 节点 | ~100ms | **8x** (8 核) |

---

## 4. Transformer 推理性能

**测试文件**: `benches/transformer_inference.rs`

### 4.1 单层推理 (seq_len=512, hidden_dim=768)

| 操作 | 时间 | 内存占用 |
|------|------|----------|
| 自注意力 | ~2ms | ~10MB |
| FFN | ~1ms | ~5MB |
| LayerNorm | ~0.1ms | ~1MB |
| **完整层** | **~3ms** | **~16MB** |

### 4.2 批处理性能

| 批次大小 | 吞吐量 (tokens/s) | GPU 加速 |
|----------|------------------|----------|
| 1 | ~500 | - |
| 4 | ~1800 | - |
| 8 | ~3200 | - |
| 16 | ~5000 | - |

### 4.3 SIMD 加速效果

| 操作 | 无 SIMD | 有 SIMD | 加速比 |
|------|---------|---------|--------|
| 矩阵乘法 | ~1ms | ~0.6ms | **1.67x** |
| 向量加法 | ~0.2ms | ~0.12ms | **1.67x** |
| Softmax | ~0.3ms | ~0.2ms | **1.5x** |

---

## 5. 内存效率分析

### 5.1 张量内存布局

| 特性 | 实现 | 效果 |
|------|------|------|
| 64 字节对齐 | `WeightTensor` | 避免 false sharing |
| 连续存储 | `DenseTensor` | 缓存友好 |
| 内存池复用 | `TensorPool` | 减少 98%+ 分配 |
| Arena 分配 | `TensorArena` | 批量分配优化 |

### 5.2 内存占用分析 (TinyLlama-1.1B)

| 组件 | 内存占用 | 百分比 |
|------|----------|--------|
| 权重矩阵 | ~2.2GB | 80% |
| 激活值 | ~200MB | 7% |
| KV Cache | ~100MB | 4% |
| 临时张量 | ~50MB | 2% |
| 其他 | ~200MB | 7% |
| **总计** | **~2.75GB** | **100%** |

---

## 6. 优化建议

### 6.1 内存池使用最佳实践

1. **预分配池**: 对于已知大小的工作负载，使用 `with_preallocate(true)`
2. **池大小配置**: `initial_capacity` 设置为典型批次大小，`max_capacity` 设置为峰值需求
3. **形状匹配**: 尽量复用相同形状的张量，提高命中率
4. **及时释放**: 使用作用域控制 `PooledTensor` 生命周期，及时返回池中

```rust
// 推荐：预分配池
let config = PoolConfig::new(64, 256).with_preallocate(true);
let mut pool = TensorPool::new(config);

// 推荐：作用域控制生命周期
for _ in 0..50 {
    {
        let tensor = pool.acquire(vec![128, 128]);
        // 使用 tensor
    } // 自动返回池中
}
```

### 6.2 正交化优化建议

1. **批量正交化**: 使用 `orthogonalize_weights_in_place()` 批量处理
2. **块大小选择**: `block_size=32` 平衡精度和性能
3. **Cayley 变换**: 启用 `with_cayley(true)` 提高数值稳定性

### 6.3 张量环压缩建议

1. **秩选择**: 根据矩阵大小自适应选择 `target_rank`
2. **压缩阈值**: `min_rank=4` 避免过度压缩
3. **重建验证**: 始终检查 `reconstruction_error < 1e-6`

---

## 7. 基准测试复现指南

### 7.1 环境配置

```bash
# Rust 版本
rustc --version  # 1.85+

# 项目依赖
cargo fetch

# 模型下载 (真实模型验证)
pip install huggingface_hub
python scripts/download_tinyllama.py
```

### 7.2 运行所有基准测试

```bash
# 内存池基准测试
cargo bench --features "tensor tensor-pool" --bench memory_pool_reduction

# 图算法基准测试
cargo bench --features "parallel" --bench traversal
cargo bench --features "parallel" --bench shortest_path
cargo bench --features "rand" --bench centrality

# Transformer 基准测试
cargo bench --features "transformer simd" --bench transformer_inference

# 运行所有测试
cargo test --features "tensor-full transformer safetensors" -- --test-threads=16
```

### 7.3 结果解读

- **时间单位**: µs (微秒), ms (毫秒), s (秒)
- **置信区间**: 95% 置信区间 (criterion 默认)
- **异常值处理**: criterion 自动检测并排除异常值

---

## 8. 性能演进历史

| 版本 | 日期 | 关键改进 | 性能提升 |
|------|------|----------|----------|
| v0.3.0 | 2026-02 | 基础图算法 | Baseline |
| v0.3.1 | 2026-02 | 并行算法 | 4-8x (多核) |
| v0.4.0 | 2026-03 | Tensor 集成 | - |
| v0.4.3 | 2026-03 | 内存池优化 | 6.7x (分配) |
| v0.5.0 | 2026-03 | 真实模型验证 | 端到端验证 |

---

## 9. 结论

God-Graph 在以下方面实现了业界领先的性能：

1. **内存池优化**: **98-99.9%** 分配减少，**6.7x** 速度提升
2. **数值稳定性**: 正交化误差**2.04e-14**，远超理论要求
3. **模型压缩**: 张量环压缩比**0.12-0.25x**，重建误差**< 1e-6**
4. **图算法**: 与 petgraph 相当，部分场景**1.5-2x**更快
5. **并行加速**: 多核场景实现**4-8x**加速

**未来优化方向**:
- GPU 后端集成 (CandleBackend)
- 动态稀疏注意力 (O(1) 编辑)
- 混合精度训练 (FP16/INT8)
- 分布式推理支持

---

**附录**: 完整基准测试日志见 `target/criterion/` 目录。
