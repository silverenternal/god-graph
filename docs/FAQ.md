# God-Graph 常见问题解答 (FAQ)

**版本**: v0.6.0-alpha
**更新日期**: 2026-03-31

---

## 📌 项目定位

### Q1: God-Graph 是什么？

**A**: God-Graph 是一个 **LLM 白盒分析工具**，核心功能包括：
- **模型拓扑分析**: 检查 LLM 的注意力连接、残差连接、层间依赖
- **可微图结构**: 用梯度下降优化神经网络架构（DifferentiableGraph）
- **数学优化**: 李群正交化、张量环压缩、拓扑约束求解
- **VGI 架构**: 类似 Linux VFS 的统一图接口，支持多后端

### Q2: God-Graph 和 petgraph 有什么区别？

| 特性 | petgraph | God-Graph |
|------|----------|-----------|
| **定位** | 通用图算法库 | LLM 白盒分析 + 可微图结构 |
| **数据结构** | CSR/邻接表 | 桶式邻接表 + Generation 索引 |
| **LLM 支持** | ❌ | ✅ Safetensors 双向转换 |
| **可微结构** | ❌ | ✅ DifferentiableGraph |
| **VGI 架构** | ❌ | ✅ 支持多后端 |
| **数学优化** | ❌ | ✅ 李群/张量环 |

**选择建议**:
- 通用图分析 → petgraph
- LLM 拓扑优化 → God-Graph
- 需要动态编辑 → God-Graph（桶式邻接表 O(1) 插入）

### Q3: God-Graph 和 llama.cpp 有什么区别？

**A**: 定位完全不同：
- **llama.cpp**: LLM **推理引擎**，目标是快速生成文本
- **God-Graph**: LLM **分析工具**，目标是检查和修改模型拓扑

**类比**:
- llama.cpp = 汽车引擎（驱动运行）
- God-Graph = 汽车诊断仪（检查/优化设计）

### Q4: God-Graph 和 PyTorch Geometric (PyG) 有什么区别？

**A**: 
- **PyG**: GNN **训练框架**，基于 PyTorch autograd
- **God-Graph**: GNN **推理 + 架构分析**，Rust 实现

**选择建议**:
- 训练 GNN → PyG / DGL
- 分析 GNN 结构 → God-Graph
- 生产部署 → God-Graph（Rust 性能优势）

---

## 🔧 技术使用

### Q5: DifferentiableGraph 可以用于训练吗？

**A**: **不推荐**。DifferentiableGraph 的设计目标是：
- ✅ **架构搜索**: 用梯度下降找到最优网络结构
- ✅ **结构剪枝**: 移除冗余的注意力连接
- ✅ **拓扑优化**: 自动添加残差连接

**不是为训练设计的原因**:
1. 没有完整的 autograd 支持
2. 不支持高阶导数
3. 缺少优化器（Adam、SGD 等）

**训练建议**:
- 使用 PyTorch / JAX / dfdx 训练
- 用 God-Graph 分析/优化训练好的模型

### Q6: VGI 架构的性能开销是多少？

**A**: 实测数据：
- **trait 对象开销**: 约 5-10%
- **动态分发开销**: 约 3-5%
- **总开销**: 约 8-15%

**优化建议**:
- 性能敏感场景 → 直接用 `Graph<T, E>`
- 需要后端切换 → 用 VGI trait

### Q7: 如何选择合适的 feature？

**A**: 根据使用场景选择：

| 场景 | 推荐 features |
|------|---------------|
| 基础图算法 | `default` (std + parallel) |
| LLM 分析 | `transformer`, `safetensors` |
| GNN 推理 | `tensor`, `tensor-gnn` |
| 内存优化 | `tensor-pool` |
| 完整功能 | `llm`, `tensor-full` |

**最小依赖示例**:
```toml
[dependencies]
god-graph = { version = "0.6", default-features = false, features = ["std"] }
```

### Q8: 如何加载 HuggingFace 模型？

**A**: 使用 `ModelSwitch`:

```rust
use god_graph::transformer::optimization::ModelSwitch;

// 1. 从 Safetensors 加载
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 2. 验证拓扑
let report = ModelSwitch::validate_topology(&graph)?;
println!("有效：{}", report.is_valid);

// 3. 优化后导出
ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

**支持的数据类型**:
- ✅ F32 (float32)
- ✅ F16 (float16)
- ✅ BF16 (bfloat16)

### Q9: 内存池能减少多少分配开销？

**A**: 实测数据：

| 场景 | 无池 | 有池 | 减少 |
|------|------|------|------|
| 迭代算法 (50 次) | 850µs | 128µs | **85%** |
| GNN 迭代 | - | - | **96-99%** |
| 注意力 QKV | - | - | **95-98%** |

**使用示例**:
```rust
use god_graph::tensor::{TensorPool, PoolConfig};

let config = PoolConfig::new(16, 128).with_preallocate(true);
let mut pool = TensorPool::new(config);

// 从池获取张量（自动归零）
let tensor = pool.acquire(vec![100, 100]);
// tensor 释放时自动回池
```

---

## 🏛️ VGI 架构

### Q10: VGI 和 Linux VFS 有什么相似之处？

**A**: 设计哲学完全一致：

| Linux VFS | God-Graph VGI |
|-----------|---------------|
| `file_operations` | `VirtualGraph` trait |
| ext4/NTFS 驱动 | SingleMachine/Distributed Backend |
| 挂载点 | `BackendRegistry` |
| 文件系统探针 | 能力发现 (`has_capability`) |

**代码类比**:
```rust
// Linux VFS
struct file_operations {
    read: fn() -> ...,
    write: fn() -> ...,
};

// God-Graph VGI
trait VirtualGraph {
    fn add_node(&mut self, data: T) -> VgiResult<NodeIndex>;
    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex>;
}
```

### Q11: 如何开发 VGI 插件？

**A**: 实现 `GraphAlgorithm` trait:

```rust
use god_graph::plugins::{GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult};

struct MyAlgorithm;

impl GraphAlgorithm for MyAlgorithm {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("my_algo", "1.0.0", "My Algorithm")
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        // 实现算法逻辑
        Ok(AlgorithmResult::scalar(42.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

**注册插件**:
```rust
let mut registry = PluginRegistry::new();
registry.register_algorithm("my_algo", MyAlgorithm)?;
```

### Q12: VGI 支持哪些后端？

**A**: 当前实现：

| 后端 | 状态 | 说明 |
|------|------|------|
| `SingleMachineBackend` | ✅ 完成 | 基于 `Graph<T, E>` |
| `DistributedBackend` | 🔲 开发中 | 基于消息传递 |
| `GPUBackend` | 📅 计划 | 基于 CUDA |

**第三方后端**:
- 可实现 `Backend` trait 贡献新后端
- 无需修改 God-Graph 核心代码

---

## 📊 性能相关

### Q13: God-Graph 的性能数据在哪里？

**A**: 详见：
- [性能基准测试报告](reports/performance.md)
- [分布式性能报告](reports/distributed-benchmarks.md)

**关键数据**:
- **并行 PageRank**: 80.7x 加速比（1000 节点）
- **并行 DFS**: 7.5x 加速比（50K 节点）
- **SIMD 优化**: 14x 提升（100 节点）
- **内存池**: 85-99% 分配减少

### Q14: 为什么不用 CSR 格式？

**A**: 详见 [架构决策记录](internals/architecture.md#为什么使用桶式邻接表而非-csr)

**核心原因**:
1. **O(1) 增量更新**: CSR 需要 O(V+E) 重建
2. **动态图场景**: LLM 拓扑优化需要频繁编辑
3. **Generation 索引**: 防止 ABA 问题

**权衡**:
- 空间效率略低（每节点一个 Vec 头）
- 缓存局部性稍差

### Q15: 如何优化大规模图的性能？

**A**: 建议：

1. **启用并行**:
```toml
god-graph = { version = "0.6", features = ["parallel"] }
```

2. **使用内存池**:
```toml
god-graph = { version = "0.6", features = ["tensor-pool"] }
```

3. **预分配容量**:
```rust
let mut graph = Graph::<String, f64>::directed();
graph.reserve(10000, 50000); // 预分配 1 万节点，5 万边
```

4. **使用 VGI 单机后端**（避免 trait 对象开销）:
```rust
use god_graph::backend::single_machine::SingleMachineBackend;
```

---

## 🐛 故障排查

### Q16: 编译失败：`cudarc` 找不到 CUDA

**A**: `tensor-gpu` feature 需要 CUDA。如果不需要 GPU：

```toml
god-graph = { version = "0.6", default-features = false, features = ["std", "parallel"] }
```

### Q17: 运行时 panic：`节点索引无效`

**A**: 可能原因：
1. 访问已删除的节点
2. Generation 不匹配

**解决方案**:
```rust
// 错误示例（可能 panic）
graph[node] = data;

// 正确示例（安全）
match graph.get_node_mut(node) {
    Ok(data) => *data = new_data,
    Err(_) => println!("节点不存在"),
}
```

### Q18: DifferentiableGraph 梯度不收敛

**A**: 检查：
1. **学习率**: 尝试降低到 0.001-0.01
2. **温度参数**: 调整 Gumbel-Softmax 的 τ
3. **稀疏正则**: 增加 `sparsity_weight`

**调试代码**:
```rust
let config = GradientConfig::default()
    .with_sparsity(0.1)  // 增加稀疏性
    .with_edge_learning_rate(0.001);  // 降低学习率
```

---

## 🤝 贡献相关

### Q19: 如何贡献代码？

**A**: 流程：
1. Fork 项目
2. 创建分支 (`git checkout -b feature/my-feature`)
3. 提交更改 (`git commit -am 'Add feature'`)
4. 推送分支 (`git push origin feature/my-feature`)
5. 提交 Pull Request

**要求**:
- ✅ 通过 `cargo clippy`
- ✅ 通过 `cargo fmt`
- ✅ 添加测试
- ✅ 更新文档

### Q20: 如何报告 Bug？

**A**: 在 GitHub Issues 提交，包含：
- 最小复现代码
- 错误信息
- 环境信息（Rust 版本、OS）

---

## 📚 学习资源

### Q21: 从哪里开始学习 God-Graph？

**A**: 推荐路径：

1. **快速上手** → [快速开始](user-guide/getting-started.md)
2. **理解架构** → [架构指南](internals/architecture.md)
3. **学习 VGI** → [VGI 指南](VGI_GUIDE.md)
4. **深入原理** → [CAD-LLM 设计](internals/cad-design.md)

### Q22: 有示例代码吗？

**A**: 详见 `examples/` 目录：

| 示例 | 说明 | 命令 |
|------|------|------|
| `differentiable_graph.rs` | 可微图结构 | `cargo run --example differentiable_graph` |
| `cad_llm_switch.rs` | ModelSwitch 转换 | `cargo run --example cad_llm_switch` |
| `cad_llm_validate_1b.rs` | 拓扑检测 | `cargo run --example cad_llm_validate_1b` |

---

## 🔗 外部资源

- [GitHub](https://github.com/silverenternal/god-graph)
- [Crates.io](https://crates.io/crates/god-graph)
- [docs.rs API 文档](https://docs.rs/god-graph)
- [GitHub Issues](https://github.com/silverenternal/god-graph/issues)
