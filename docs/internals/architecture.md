# God-Graph 架构指南

**版本**: 0.6.0-alpha
**日期**: 2026-04-01

---

## 📐 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      God-Graph Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Graph      │     │   Tensor     │     │ Transformer  │   │
│  │  Core        │     │   Core       │     │   Core       │   │
│  │              │     │              │     │              │   │
│  │ - Graph<T,E> │     │ - DenseTensor│     │ - LlamaModel │   │
│  │ - BucketAdj  │     │ - SparseTensor│    │ - Layers     │   │
│  │ - Generation │     │ - COO/CSR    │     │ - KVCache    │   │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘   │
│         │                    │                    │           │
│         └────────────────────┼────────────────────┘           │
│                              │                                 │
│                     ┌────────▼────────┐                       │
│                     │  Optimization   │                       │
│                     │     Layer       │                       │
│                     │                 │                       │
│                     │ - ModelSwitch   │                       │
│                     │ - CadEditor     │                       │
│                     │ - LieGroup      │                       │
│                     │ - TensorRing    │                       │
│                     │ - Constraints   │                       │
│                     └─────────────────┘                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VGI (Virtual Graph Interface)                │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │ Single      │  │ Distributed │  │ GPU         │      │  │
│  │  │ Machine     │  │ Cluster     │  │ Accelerator │      │  │
│  │  │ Backend     │  │ Backend     │  │ Backend     │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 核心模块职责

### 1. Graph Core (`src/graph/`)

**职责**: 高性能图数据结构和算法

| 文件 | 职责 | 关键类型 |
|------|------|----------|
| `graph_impl.rs` | Graph 核心实现 | `Graph<T,E>`, `AdjBucket` |
| `traits.rs` | Graph trait 定义 | `GraphBase`, `GraphOps`, `GraphQuery` |
| `iterators.rs` | 迭代器实现 | `NeighborsIter`, `NodeIter`, `EdgeIter` |
| `builders.rs` | 构建器 | `GraphBuilder` |

**关键设计**：
- **桶式邻接表**: 支持 O(1) 增量边插入
- **Generation 索引**: 防止 ABA 问题
- **64 字节对齐**: 避免 CPU 缓存伪共享

---

### 2. Tensor Core (`src/tensor/`)

**职责**: 张量基础设施

| 文件 | 职责 | 关键类型 |
|------|------|----------|
| `dense.rs` | 密集张量 | `DenseTensor` |
| `sparse.rs` | 稀疏张量 | `SparseTensor`, `COOTensor`, `CSRTensor` |
| `graph_tensor.rs` | 图 - 张量转换 | `GraphAdjacencyMatrix`, `GraphFeatureExtractor` |
| `gnn.rs` | GNN 层 | `GCNConv`, `GATConv`, `GraphSAGE` |
| `pool.rs` | 内存池 | `TensorPool`, `PooledTensor` |
| `decomposition/` | 张量分解 | `qr.rs`, `svd.rs`, `lie_algebra.rs`, `tensor_ring.rs` |

**关键设计**：
- **ndarray 后端**: 基于成熟的 ndarray crate
- **多格式支持**: Dense/COO/CSR/BSR
- **内存池优化**: 减少迭代算法的分配开销

---

### 3. Transformer Core (`src/transformer/`)

**职责**: Transformer 架构组件

| 文件 | 职责 | 关键类型 |
|------|------|----------|
| `model.rs` | LLaMA 模型 | `LlamaModel`, `LlamaDecoderLayer` |
| `layers/` | 层实现 | `MultiHeadAttention`, `RMSNorm`, `FeedForward`, `RoPE` |
| `loader/` | 模型加载 | `SafetensorsLoader`, `LlamaConfig` |
| `kv_cache/` | KV 缓存 | `KVCache` |
| `generation.rs` | 文本生成 | `TextGenerator`, `GenerationConfig` |

**关键设计**：
- **Pre-Norm 架构**: LLaMA/Mistral 专用
- **GQA 支持**: Grouped-Query Attention
- **RoPE 位置编码**: 旋转位置编码

---

### 4. Optimization Layer (`src/transformer/optimization/`)

**职责**: LLM 白盒优化核心

| 文件 | 职责 | 关键类型 |
|------|------|----------|
| `switch.rs` | Safetensors ↔ GodGraph | `ModelSwitch`, `OperatorType`, `WeightTensor` |
| `cad_editor.rs` | CAD 风格编辑器 | `CadStyleEditor`, `SubGraph`, `EditOperation` |
| `lie_group.rs` | 李群优化 | `LieGroupOptimizer`, `SO kBlock` |
| `tensor_ring.rs` | 张量环压缩 | `TensorRingCompressor`, `CompressionConfig` |
| `constraints.rs` | 拓扑约束 | `TopologyValidator`, `TopologyConstraint` |
| `switch.rs` | 模型转换 | `ModelSwitch` |

**关键设计**：
- **CAD 范式迁移**: 把 LLM 当机械零件设计
- **双向转换**: Safetensors → GodGraph → Safetensors
- **数学优化**: 李群、张量环、图论

---

## 🔄 核心工作流

### 工作流 1: 模型拓扑优化

```
1. 加载 Safetensors
   ↓
2. 转换为 GodGraph
   ↓
3. 检测拓扑缺陷
   ↓
4. 求解约束（自动修复）
   ↓
5. 导出优化后的 Safetensors
```

**代码示例**:
```rust
use god_graph::transformer::optimization::*;

// 加载
let mut graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 编辑
let mut editor = CadStyleEditor::new(&mut graph);
editor.solve_constraints()?;

// 导出
// ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

---

### 工作流 2: 李群正交化

```
1. 提取权重矩阵
   ↓
2. 投影到 so(n) 李代数
   ↓
3. 指数映射到 SO(n) 李群
   ↓
4. 验证正交性
   ↓
5. 替换原权重
```

**代码示例**:
```rust
use god_graph::transformer::optimization::LieGroupOptimizer;

let config = LieGroupConfig::new().with_cayley_transform(true);
let optimizer = LieGroupOptimizer::new(config);

optimizer.orthogonalize_weights(&mut graph)?;
```

---

### 工作流 3: 张量环压缩

```
1. 分析权重矩阵维度
   ↓
2. 选择最优 TR 秩
   ↓
3. 分解为 Tensor Ring
   ↓
4. 计算压缩比
   ↓
5. 存储压缩后的核心张量
```

**代码示例**:
```rust
use god_graph::transformer::optimization::TensorRingCompressor;

let compressor = TensorRingCompressor::default();
let report = compressor.compress_graph(&graph)?;

println!("压缩比：{:.2}x", report.compression_ratio);
```

---

## 📦 数据流

### Safetensors → GodGraph

```
Safetensors 文件
   │
   ├─→ 读取 Header (8 bytes + JSON)
   │   └─→ 解析 tensor 元数据 (name, dtype, shape, offsets)
   │
   ├─→ 读取 Tensor 数据
   │   └─→ 转换为 DenseTensor (F32/F16/BF16 → F64)
   │
   └─→ 推断 OperatorType
       └─→ 创建 Graph 节点和边
```

### GodGraph → Safetensors

```
GodGraph
   │
   ├─→ 提取 WeightTensor
   │   └─→ 获取 (name, data, shape)
   │
   ├─→ 转换为原始 dtype
   │   └─→ F64 → F32/F16/BF16
   │
   └─→ 序列化到 Safetensors
       └─→ 写入 Header + Data
```

---

## 🧩 模块依赖关系

```
Graph Core
    │
    ├─→ 被 Tensor Core 使用 (GraphTensorExt)
    │
    └─→ 被 Optimization Layer 使用 (Graph<OperatorType, WeightTensor>)

Tensor Core
    │
    ├─→ 被 Transformer Core 使用 (DenseTensor)
    │
    └─→ 被 Optimization Layer 使用 (lie_algebra, tensor_ring)

Transformer Core
    │
    └─→ 被 Optimization Layer 使用 (ModelSwitch 加载权重)

Optimization Layer
    │
    └─→ 顶层应用，依赖所有下层模块
```

---

## 🎯 设计决策

### 为什么用图结构表示 LLM？

**传统表示**（Safetensors）：
```
model.layers.0.self_attn.q_proj.weight: [4096, 4096]
model.layers.0.self_attn.k_proj.weight: [4096, 4096]
...
```
- ✅ 紧凑，适合存储
- ❌ 无法表达拓扑关系

**GodGraph 表示**：
```
Node(Attention) ──edge(q_proj)──> Node(Attention)
         │
         └──edge(residual)──> Node(FFN)
```
- ✅ 显式表达数据流和依赖
- ✅ 支持拓扑分析和优化
- ❌ 占用更多内存

**结论**：GodGraph 是**优化中间表示**，不是推理格式。

---

### 为什么用李群理论？

**问题**：LLM 权重矩阵的正交性影响数值稳定性。

**传统方法**：
- 正交初始化（训练前）
- 正交正则化（训练中）

**GodGraph 方法**：
- 李群指数映射（推理后优化）
- `exp: so(n) → SO(n)` 保证正交性

**优势**：
- 数学上严格的正交性保证
- 可以应用于预训练模型（无需重新训练）

---

### 为什么用张量环分解？

**传统压缩**：
- SVD：只适用于 2D 矩阵
- 量化：精度损失

**张量环分解**：
- 适用于高维张量
- 可控的精度 - 压缩权衡

**公式**：
```
W(i₁,...,iₙ) = Tr[G₁(i₁) × ... × Gₙ(iₙ)]
```

---

## 🔧 扩展指南

### 添加新的优化器

1. 在 `src/transformer/optimization/` 创建新模块
2. 实现 `Optimizer` trait：
```rust
pub trait Optimizer {
    fn optimize(&self, graph: &mut Graph<OperatorType, WeightTensor>) -> GraphResult<OptimizationReport>;
}
```
3. 在 `mod.rs` 中导出
4. 添加测试

---

### 添加新的拓扑约束

1. 在 `constraints.rs` 添加 `TopologyConstraint` 变体：
```rust
pub enum TopologyConstraint {
    // ... 现有约束 ...
    MyCustomConstraint { /* 参数 */ },
}
```
2. 实现验证逻辑：
```rust
fn validate_my_constraint(&self, graph: &Graph<...>) -> GraphResult<(bool, String, Option<String>)> {
    // 验证逻辑
}
```
3. 添加到 `TopologyValidator::validate()`

---

### 添加新的张量分解

1. 在 `src/tensor/decomposition/` 创建新模块
2. 实现分解函数：
```rust
pub fn my_decomposition(tensor: &DenseTensor, rank: usize) -> Result<MyDecomposition, TensorError> {
    // 分解逻辑
}
```
3. 实现重构函数：
```rust
impl MyDecomposition {
    pub fn reconstruct(&self) -> Result<DenseTensor, TensorError> {
        // 重构逻辑
    }
}
```
4. 添加测试验证数值稳定性

---

## 📊 性能特征

### 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|------------|------|
| 节点访问 | O(1) | 直接索引 |
| 边插入 | O(1) | 桶式邻接表 |
| 边删除 | O(1) | 位图标记 |
| 邻居迭代 | O(degree) | 线性扫描 |
| 拓扑排序 | O(V+E) | DFS |
| 连通分量 | O(V+E) | BFS |

### 空间复杂度

| 结构 | 空间复杂度 | 说明 |
|------|------------|------|
| Graph<V,E> | O(V + E) | 顶点 + 边 |
| DenseTensor | O(∏shape) | 元素数量 |
| CSRTensor | O(nnz) | 非零元素 |
| TensorRing | O(n·r²·d) | n 核心，r 秩，d 维度 |

---

## 🧪 测试策略

### 单元测试

- 每个核心函数都有单元测试
- 数值计算测试用容差验证（1e-5）

### 集成测试

- `cad_llm_integration.rs`: CAD-LLM 完整工作流
- `transformer_integration.rs`: Transformer 层集成

### 属性测试

- `property_tests.rs`: 图算法不变量验证

---

## 🔮 未来架构演进

### v0.6.0-alpha (已发布)
- ✅ VGI 架构完整实现
- ✅ 分布式处理框架
- ✅ 插件生态系统
- ✅ 文档体系完善（快速开始、教程、API 参考）
- ✅ 真实模型端到端验证（TinyLlama-1.1B）

### v0.7.0-rc (计划中)
- 🔲 GPU 后端原型（CUDA 支持）
- 🔲 动态稀疏注意力模式
- 🔲 Model Switch 导出功能完善
- 🔲 性能基准套件
- 🔲 API 稳定化

### v1.0.0-stable (愿景)
- 🔲 生产环境验证
- 🔲 完整文档和示例
- 🔲 crates.io 正式发布
- 🔲 社区贡献生态

---

## 📝 架构决策记录

### 为什么使用桶式邻接表而非 CSR？

**决策**: 采用桶式邻接表（Bucket Adjacency List）而非传统 CSR 格式

**原因**:
1. **O(1) 增量更新**: CSR 需要 O(V+E) 重建，桶式邻接表支持 O(1) 边插入
2. **动态图场景**: LLM 拓扑优化需要频繁编辑边结构
3. **Generation 索引**: 防止 ABA 问题，CSR 难以支持

**权衡**:
- 空间效率略低于 CSR（每节点一个 Vec 头）
- 缓存局部性稍差

### 为什么设计 VGI 架构？

**决策**: 引入 Virtual Graph Interface 统一图后端接口

**原因**:
1. **后端可插拔**: 类似 Linux VFS，支持单机/分布式/GPU 后端
2. **算法复用**: 一套算法代码，多个后端复用
3. **第三方扩展**: 第三方可贡献新后端，无需修改核心代码

**权衡**:
- trait 对象开销约 5-10%
- 增加架构复杂度

### 为什么 DifferentiableGraph 不用 autograd？

**决策**: DifferentiableGraph 使用手搓梯度计算，而非集成 autograd

**原因**:
1. **专注架构搜索**: 目标是优化图结构，不是训练权重
2. **依赖精简**: 避免强制依赖 dfdx/Candle 等重型框架
3. **灵活性**: 用户可自选 autograd 后端集成

**权衡**:
- 需要手动实现梯度计算
- 不支持高阶导数

**联系方式**: silverenternal <3147264070@qq.com>
**项目地址**: https://github.com/silverenternal/god-graph
