# CAD-LLM 设计哲学

**版本**: 0.5.0-alpha (Updated)
**日期**: 2026-03-29

---

## 📖 概述

god-gragh 不是另一个 LLM 推理引擎，也不是另一个图算法库。

它是一个**基于图结构的 LLM 白盒优化工具箱**——把 LLM 从"黑盒"变成"白盒"，用图论和微分几何的工具检查和优化模型。

**核心创新**：
1. **DifferentiableGraph（可微图结构）**——将图结构从"静态容器"变为"可微分的计算本身"
2. **ModelSwitch 双向转换**——Safetensors ↔ GodGraph 无损工作流

---

## 🎯 核心定位

### 我们不是

❌ **LLM 推理引擎** —— 打不过 `llama.cpp` 和 `candle`
❌ **GNN 训练框架** —— 打不过 DGL 和 PyG
❌ **通用图算法库** —— `petgraph` 更成熟

### 我们是

✅ **LLM 白盒优化工具** —— 可以检查/修改模型拓扑
✅ **图 - 张量双向转换器** —— Safetensors ↔ GodGraph ↔ Safetensors
✅ **拓扑缺陷检测器** —— 发现梯度阻断、孤立节点、缺失残差连接
✅ **数学层面优化器** —— 李群正交化、张量环压缩
✅ **可微图结构引擎** —— 梯度下降优化神经网络架构（DifferentiableGraph）

---

## 🧠 CAD 范式迁移

我们把机械 CAD 软件的设计哲学迁移到 LLM 优化：

| CAD 概念 | LLM 等价物 | GodGraph 实现 |
|----------|-----------|--------------|
| **表面断裂检查** | 孤立注意力头检测 | `connected_components` |
| **非流形几何检查** | 梯度阻断检测 | `topological_sort + path_analysis` |
| **尺寸约束** | 注意力头权重平衡 | `AttentionHeadBalance` 约束 |
| **平行约束** | 残差连接强制 | `ResidualConnection` 约束 |
| **装配约束** | 模块接口匹配 | `validate_assembly` |
| **零件替换** | 模块提取/替换 | `extract_module` / `replace_module` |
| **公差分析** | 数值稳定性验证 | `lie_exponential` 正交化 |
| **轻量化设计** | 模型压缩 | `compress_tensor_ring` |
| **参数化设计** | **可微图结构** | **`DifferentiableGraph`** ⭐ |
| **约束求解** | **拓扑约束 + 梯度优化** | **`CadStyleEditor + DifferentiableGraph`** |

### 新增：DifferentiableGraph 与 CAD 的映射

**DifferentiableGraph 是 CAD 参数化设计思想在 LLM 架构优化中的体现**：

| CAD 参数化设计 | DifferentiableGraph 实现 |
|----------------|-------------------------|
| 参数化特征（尺寸、角度） | 边存在概率（连续松弛） |
| 参数更新驱动几何变形 | 梯度下降驱动图结构演化 |
| 设计空间探索 | 神经架构搜索 |
| 约束驱动的造型优化 | 拓扑约束 + 损失函数优化 |
| 设计变量灵敏度分析 | 结构梯度计算 |

**核心思想**：就像 CAD 软件用参数控制几何形状，DifferentiableGraph 用连续参数控制图结构，实现"可微分的架构设计"。

---

## 🔬 数学基础

### 1. 李群理论 (Lie Group Theory)

**为什么需要李群？**

LLM 权重矩阵的正交性影响：
- 训练稳定性（正交初始化）
- 推理数值鲁棒性（避免梯度消失/爆炸）
- 模型压缩效率（正交矩阵更容易压缩）

**GodGraph 的实现**：

```rust
// so(n) 李代数 → SO(n) 李群
use god_gragh::tensor::decomposition::{lie_exponential, lie_logarithm};

// 创建 so(2) 生成元（旋转）
let algebra = DenseTensor::from_vec(
    vec![0.0, -theta, theta, 0.0],
    vec![2, 2],
);

// 指数映射：得到旋转矩阵
let rotation = lie_exponential(&algebra)?;
// rotation = [cos(θ), -sin(θ)]
//            [sin(θ),  cos(θ)]

// 验证正交性：R^T * R = I
assert!(is_orthogonal(&rotation, 1e-5));
```

**数学原理**：
- 李代数 `so(n)` = 斜对称矩阵空间 (A^T = -A)
- 李群 `SO(n)` = 正交矩阵空间 (R^T * R = I, det(R) = 1)
- 指数映射 `exp: so(n) → SO(n)` 用 Padé 近似实现

---

### 2. 张量环分解 (Tensor Ring Decomposition)

**为什么需要张量环？**

传统矩阵分解（SVD）只能处理 2D 矩阵，而 LLM 权重本质是高维张量：
- Attention 权重：`(num_heads, seq_len, seq_len)`
- FFN 权重：`(num_layers, hidden_dim, intermediate_dim)`

张量环分解把高维张量表示为 3D 核心张量的环：

```
W(i₁, i₂, ..., iₙ) = Tr[G₁(i₁) × G₂(i₂) × ... × Gₙ(iₙ)]
```

**GodGraph 的实现**：

```rust
use god_gragh::transformer::optimization::{
    TensorRingCompressor, CompressionConfig
};

// 配置压缩参数
let config = CompressionConfig::new()
    .with_target_ranks(vec![4, 8, 4])  // TR 秩
    .with_min_rank(2)
    .with_max_rank(16);

let compressor = TensorRingCompressor::new(config);

// 压缩 64×64 权重矩阵
let weight = DenseTensor::from_vec(
    vec![1.0; 64 * 64],
    vec![64, 64],
);

let ring = compressor.decompose(&weight)?;

// 压缩比 > 1.0 表示成功压缩
assert!(ring.compression_ratio() > 1.0);
// 原始：64 * 64 = 4096 参数
// 压缩：3 * 4 * 64 * 4 = 3072 参数（秩=4 时）
```

**压缩比公式**：
```
原始参数：m × n
TR 参数：r₀×m×r₁ + r₁×n×r₂
压缩比：(m × n) / (r₀×m×r₁ + r₁×n×r₂)
```

---

### 3. 图论拓扑分析

**为什么需要拓扑分析？**

LLM 计算图的拓扑结构决定：
- 梯度流是否畅通（反向传播）
- 信息是否充分混合（注意力机制）
- 残差连接是否完整（避免退化）

**GodGraph 的实现**：

```rust
use god_gragh::transformer::optimization::{
    CadStyleEditor, TopologyConstraint, TopologyDefect
};

let mut editor = CadStyleEditor::new(&mut graph);

// 1. 检测拓扑缺陷
let defects: Vec<TopologyDefect> = editor.detect_defects()?;

for defect in &defects {
    match defect.defect_type {
        DefectType::IsolatedNode => {
            // 发现孤立节点（无连接）
            println!("节点 {} 无连接", defect.location);
        }
        DefectType::DisconnectedComponent => {
            // 发现 disconnected 组件
            println!("发现 {} 个 disconnected 组件", defect.location);
        }
        DefectType::BlockedGradientFlow => {
            // 发现梯度阻断
            println!("梯度流在 {} 处阻断", defect.location);
        }
        _ => {}
    }
}

// 2. 添加约束
editor.add_constraint(TopologyConstraint::ResidualConnection {
    from_layer: "attention".to_string(),
    to_layer: "output".to_string(),
})?;

// 3. 求解约束（自动修复）
editor.solve_constraints()?;
```

**拓扑缺陷类型**：
- `IsolatedNode` - 孤立节点（无连接）
- `DisconnectedComponent` - disconnected 组件
- `UnexpectedCycle` - 意外环（前向图不应有环）
- `MissingResidual` - 缺失残差连接
- `BlockedGradientFlow` - 梯度流阻断

---

## 🏗️ 架构设计

### 核心工作流

```
┌─────────────────┐
│ Safetensors     │ HuggingFace 标准格式
│ (model.safetensors) │
└────────┬────────┘
         │ ModelSwitch::load_from_safetensors()
         ▼
┌─────────────────┐
│ GodGraph        │ 图结构中间表示
│ - 节点：OperatorType │
│ - 边：WeightTensor  │
└────────┬────────┘
         │ 优化
         ▼
    ┌────────────┬────────────┬──────────────┐
    │            │            │              │
    ▼            ▼            ▼              ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐
│ 李群   │  │ 张量环 │  │ 拓扑约束 │  │ 动态剪枝 │
│ 正交化 │  │ 压缩   │  │ 求解     │  │ 注意力   │
└────┬───┘  └────┬───┘  └────┬─────┘  └────┬─────┘
     │           │            │             │
     └───────────┴────────────┴─────────────┘
                     │
                     │ ModelSwitch::save_to_safetensors()
                     ▼
            ┌─────────────────┐
            │ 优化后的模型    │
            │ (optimized.safetensors) │
            └─────────────────┘
```

### 模块职责

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| **Model Switch** | Safetensors ↔ GodGraph 双向转换 | `src/transformer/optimization/switch.rs` |
| **CAD Editor** | 拓扑缺陷检测 + 约束求解 | `src/transformer/optimization/cad_editor.rs` |
| **Lie Group** | 李群正交化 + SO(k) 分解 | `src/tensor/decomposition/lie_algebra.rs` |
| **Tensor Ring** | 张量环压缩 | `src/transformer/optimization/tensor_ring.rs` |
| **Constraints** | 拓扑约束定义 | `src/transformer/optimization/constraints.rs` |

---

## 📊 使用场景

### 场景 0: ModelSwitch 双向转换 ⭐ 新增

**ModelSwitch** 提供 HuggingFace Safetensors 和 GodGraph 之间的双向无损转换：

```rust
use god_gragh::transformer::optimization::ModelSwitch;

// 1. 加载：Safetensors → GodGraph
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 2. 验证拓扑完整性
let report = ModelSwitch::validate_topology(&graph)?;
println!("拓扑有效：{}", report.is_valid);
println!("连通分量：{}", report.connected_components);
println!("是 DAG: {}", report.is_dag);

// 3. 验证权重精度（比较两个图的权重差异）
let diff = ModelSwitch::verify_weights(&original_graph, &modified_graph)?;
println!("最大 L2 差异：{:.6e}", diff.max_l2_diff);
println!("平均 L2 差异：{:.6e}", diff.avg_l2_diff);

// 4. 导出：GodGraph → Safetensors
ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

**功能特性**：
- ✅ **双向转换**：Safetensors ↔ GodGraph 无损转换
- ✅ **数据类型支持**：F32、F64、F16 自动转换
- ✅ **拓扑验证**：检查连通性、环、孤立节点
- ✅ **权重验证**：L2 范数比较，精度损失 < 1e-5
- ✅ **算子推断**：根据权重名称自动推断算子类型（Attention、MLP、Norm 等）

**运行示例**：
```bash
cargo run --example cad_llm_switch --features safetensors
```

**示例输出**：
```
=== CAD-LLM Model Switch Example ===

Creating demo GodGraph...
Created graph with 4 nodes and 4 edges

Step 1: Validating topology...
  Topology valid: false
  Connected components: 4

Step 2: Exporting to Safetensors...
  Exported to: demo_export.safetensors
  File size: 1781.38 KB

Step 3: Loading back from Safetensors...
  Loaded graph with 4 nodes and 4 edges

Step 4: Verifying weights...
  Max L2 difference: 0.000000e0
  Avg L2 difference: 0.000000e0

=== Example Complete ===
```

---

### 场景 1: 模型拓扑检查

```rust
use god_gragh::transformer::optimization::ModelSwitch;

// 加载模型
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 验证拓扑
let report = ModelSwitch::validate_topology(&graph)?;

if !report.is_valid {
    println!("模型拓扑问题:");
    for issue in &report.issues {
        println!("  - {}", issue);
    }
}
```

**典型输出**：
```
模型拓扑问题:
  - Graph has 2 disconnected components
  - Graph has 3 isolated nodes
  - Graph contains cycles (may be valid for recurrent models)
```

---

### 场景 2: 李群正交化优化

```rust
use god_gragh::transformer::optimization::{
    LieGroupOptimizer, LieGroupConfig
};

let config = LieGroupConfig::new()
    .with_cayley_transform(true)
    .with_layers(vec!["q_proj".to_string(), "k_proj".to_string()]);

let optimizer = LieGroupOptimizer::new(config);

// 正交化权重
optimizer.orthogonalize_weights(&mut graph)?;

// 验证正交性
for edge in graph.edges() {
    let weight = DenseTensor::new(edge.data().data.clone(), edge.data().shape.clone());
    assert!(is_orthogonal(&weight, 1e-5));
}
```

---

### 场景 3: 张量环压缩

```rust
use god_gragh::transformer::optimization::{
    TensorRingCompressor, CompressionConfig
};

let config = CompressionConfig::new()
    .with_target_ranks(vec![4, 8, 4])
    .with_target_ratio(2.0);  // 目标 2 倍压缩

let compressor = TensorRingCompressor::new(config);

// 压缩并获取报告
let report = compressor.compress_graph(&graph)?;

println!("压缩报告:");
println!("  原始参数：{}", report.original_params);
println!("  压缩后参数：{}", report.compressed_params);
println!("  压缩比：{:.2}x", report.compression_ratio);

for layer in &report.layers {
    println!("  {}: {:.2}x", layer.layer_name, layer.compression_ratio);
}
```

---

### 场景 4: 动态注意力剪枝

```rust
use god_gragh::transformer::graph_transformer::GraphTransformer;

let mut transformer = GraphTransformer::new(
    12,  // num_layers
    12,  // num_heads
    768, // hidden_dim
);

// 构建计算图
transformer.build_graph(&input_ids);

// 剪枝弱注意力边（阈值 0.01）
let pruned_count = transformer.prune_weak_edges(0.01);

println!("剪枝 {} 条弱注意力边", pruned_count);

// 导出可视化
std::fs::write("graph.dot", transformer.to_dot())?;
// 用 Graphviz 查看：dot -Tpng graph.dot -o graph.png
```

---

## 🎓 目标用户

### 适合使用 god-gragh 的人

✅ **LLM 研究人员** - 想检查和修改模型拓扑结构
✅ **模型压缩工程师** - 想用张量环/正交化压缩模型
✅ **QA 团队** - 想验证模型完整性（梯度流、残差连接）
✅ **算法探索者** - 想实验动态注意力剪枝、稀疏注意力

### 不适合使用 god-gragh 的人

❌ **应用开发者** - 只想用 LLM 做推理（用 `llama.cpp`）
❌ **训练工程师** - 想训练新模型（用 PyTorch/JAX）
❌ **GPU 加速需求** - 需要 CUDA 加速（用 `candle` 或 `vllm`）

---

## 🔮 路线图

### v0.5.0-alpha (当前)
- ✅ 李群正交化
- ✅ 张量环压缩
- ✅ 拓扑约束求解
- ✅ 图结构 Transformer
- ✅ **ModelSwitch 双向转换** (Safetensors ↔ GodGraph)
- ✅ **真实模型验证** (TinyLlama-1.1B)

### v0.6.0-beta
- [ ] 稀疏注意力模式（SlidingWindow, BlockSparse）
- [ ] 性能基准测试
- [ ] GraphTransformer 执行引擎优化

### v0.7.0-rc
- [ ] 白盒优化案例研究
- [ ] 生产环境测试
- [ ] crates.io 发布

---

## 📚 参考资源

### 李群理论
- "Lie Groups, Lie Algebras, and Representations" - Brian C. Hall
- "Matrix Groups for Undergraduates" - Kristopher Tapp

### 张量分解
- "Tensor Decompositions and Applications" - Kolda & Bader (2009)
- "Tensor Ring Decomposition" - Zhao et al. (2016)

### 图论与 LLM
- "Graph Neural Networks" - Wu et al. (2020)
- "Transformers as Graph Neural Networks" - Ying et al. (2021)

---

## 🤝 贡献指南

我们欢迎以下类型的贡献：

1. **数学验证** - 验证李群/张量环实现的正确性
2. **用例分享** - 用 god-gragh 发现/优化真实模型的问题
3. **性能优化** - 提升大规模图的处理速度
4. **文档改进** - 让复杂概念更容易理解

---

**最后的话**

god-gragh 的愿景不是成为"最快的 LLM 推理引擎"，而是成为"最懂 LLM 内部结构的工具"。

我们相信：**理解模型比使用模型更重要**。

---

**联系方式**: silverenternal <3147264070@qq.com>
**项目地址**: https://github.com/silverenternal/god-graph
