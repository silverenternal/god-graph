# CPU 并行化与可视化方案

**版本**: 0.5.0-alpha
**日期**: 2026-03-29
**状态**: P0 已完成 ✅

---

## 完成情况

### ✅ P0（已完成）
- [x] 修改 `Cargo.toml` 默认启用 `parallel` 特性
- [x] 创建 `export_model_dot` 示例（导出 Transformer 架构到 DOT 格式）
- [x] 编写可视化使用教程 (`docs/VISUALIZATION_TUTORIAL.md`)

### 🔄 P1（进行中）
- [ ] 改造 `load_from_safetensors()` 使用并行迭代器
- [ ] 实现 HTML 交互式可视化（Cytoscape.js）
- [ ] 添加真实模型可视化示例（TinyLlama）

### 📋 P2（计划）
- [ ] 李群正交化并行化
- [ ] 张量环压缩并行化
- [ ] 3D 可视化原型（Three.js）

### 1.1 当前问题分析

**问题**：每次跑起来的时候后台都只有一个核有负载

**根本原因**：
1. `load_from_safetensors()` 是**单线程串行**处理所有权重
2. 李群正交化、张量环压缩等核心操作未启用并行
3. 默认未启用 `parallel` 特性

### 1.2 并行化改造方案

#### 方案 A：启用现有 parallel 特性（立即生效）

```toml
# Cargo.toml
[features]
default = ["std", "parallel"]  # 默认启用并行
parallel = ["dep:rayon", "dep:crossbeam-queue"]
```

```rust
// 在加载模型时启用多线程
use rayon::prelude::*;

// ModelSwitch::load_from_safetensors 并行改造
pub fn load_from_safetensors_parallel<P: AsRef<Path>>(path: P) -> GraphResult<Graph<OperatorType, WeightTensor>> {
    use safetensors::SafeTensors;
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path.as_ref())?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let safetensors = SafeTensors::deserialize(&buffer)?;
    let tensors: Vec<_> = safetensors.tensors().collect();

    // 并行处理所有权重（Rayon 自动利用多核）
    let results: Vec<_> = tensors
        .par_iter()  // 并行迭代器
        .map(|(name, tensor_view)| {
            // 每个权重的处理是独立的
            let shape = tensor_view.shape().to_vec();
            let dtype = tensor_view.dtype();
            let data = convert_to_f64(tensor_view, dtype);
            let weight = WeightTensor::new(name.to_string(), data, shape);
            let operator = Self::infer_operator_from_name(name);
            (name.to_string(), operator, weight)
        })
        .collect();

    // 构建图（串行，因为图操作不是线程安全的）
    let mut graph = Graph::<OperatorType, WeightTensor>::directed();
    for (_name, operator, weight) in results {
        let node = graph.add_node(operator)?;
        graph.add_edge(node, node, weight)?;
    }

    Ok(graph)
}
```

#### 方案 B：李群正交化并行化

```rust
// lie_group.rs 并行改造
use rayon::prelude::*;

pub fn orthogonalize_weights_parallel(
    &self,
    graph: &mut Graph<OperatorType, WeightTensor>,
) -> GraphResult<()> {
    use crate::graph::traits::GraphQuery;

    // 收集所有边的索引和数据
    let edge_data: Vec<_> = graph.edges()
        .map(|e| {
            (e.index(), e.data().name.clone(), e.data().data.to_vec(), e.data().shape.to_vec())
        })
        .collect();

    // 并行正交化（每个权重独立）
    let orthogonalized: Vec<_> = edge_data
        .par_iter()  // 并行处理
        .map(|(edge_idx, name, data, shape)| {
            let mut tensor = DenseTensor::new(data.clone(), shape.clone());
            let error = orthogonalize_in_place(&mut tensor.data, &tensor.shape)
                .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;
            Ok((*edge_idx, name.clone(), tensor, error))
        })
        .collect::<GraphResult<Vec<_>>>()?;

    // 写回图（串行）
    for (edge_idx, name, tensor, error) in orthogonalized {
        graph[edge_idx].data = tensor.data;
        // 记录误差...
    }

    Ok(())
}
```

#### 方案 C：张量环压缩并行化

```rust
// tensor_ring.rs 并行改造
pub fn compress_model_parallel(
    &self,
    graph: &mut Graph<OperatorType, WeightTensor>,
) -> GraphResult<CompressionReport> {
    use crate::graph::traits::GraphQuery;

    // 收集所有权重
    let edge_data: Vec<_> = graph.edges()
        .map(|e| (e.index(), e.data().name.clone(), e.data().data.to_vec(), e.data().shape.to_vec()))
        .collect();

    // 并行压缩
    let compressed: Vec<_> = edge_data
        .par_iter()
        .map(|(edge_idx, name, data, shape)| {
            let tensor = DenseTensor::new(data.clone(), shape.clone());
            let ranks = compute_optimal_ranks(&tensor.shape, self.config.target_compression_ratio);
            let tr = tensor_ring_decompose(&tensor, &ranks)?;
            Ok((*edge_idx, name.clone(), tr))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // 写回图...
    Ok(CompressionReport { compressed_count: compressed.len() })
}
```

### 1.3 性能预期

| 操作 | 单线程 | 8 核并行 | 加速比 |
|------|--------|---------|--------|
| 模型加载 (1.1B) | ~30s | ~4s | 7.5x |
| 李群正交化 | ~60s | ~8s | 7.5x |
| 张量环压缩 | ~120s | ~15s | 8x |

### 1.4 实施步骤

1. **立即**：修改 `Cargo.toml` 默认启用 `parallel` 特性
2. **P1**：改造 `load_from_safetensors()` 使用并行迭代器
3. **P1**：改造 `orthogonalize_weights()` 使用并行处理
4. **P2**：改造 `tensor_ring_decompose()` 支持批量并行

---

## 任务 2：大模型架构可视化方案

### 2.1 现有可视化能力

项目**已有**可视化基础设施：

| 模块 | 功能 | 状态 |
|------|------|------|
| `src/export/dot.rs` | Graphviz DOT 导出 | ✅ 完整 |
| `src/export/svg.rs` | SVG 矢量图导出 | ✅ 完整 |
| `src/transformer/graph_transformer/execution.rs` | Transformer 拓扑导出 | ✅ 完整 |

### 2.2 增强方案：交互式 Web 可视化

#### 方案 A：增强 DOT 导出（立即生效）

```rust
// 新增：针对 Transformer 架构优化的 DOT 导出
pub fn export_transformer_to_dot(
    graph: &Graph<OperatorType, WeightTensor>,
) -> String {
    let mut dot = String::from("digraph Transformer {\n");
    
    // 1. 设置布局方向（从上到下）
    dot.push_str("    rankdir=TB;\n");
    dot.push_str("    node [shape=box, style=filled];\n");
    dot.push_str("    splines=ortho;\n");  // 正交边，更清晰
    
    // 2. 按层分组（subgraph cluster）
    dot.push_str("    subgraph cluster_embedding {\n");
    dot.push_str("        label=\"Embedding\";\n");
    dot.push_str("        style=filled;\n");
    dot.push_str("        fillcolor=lightyellow;\n");
    // ... 添加嵌入层节点
    dot.push_str("    }\n");
    
    // 3. 对每个 Transformer 层创建子图
    for layer_idx in 0..num_layers {
        dot.push_str(&format!("    subgraph cluster_layer_{} {{\n", layer_idx));
        dot.push_str(&format!("        label=\"Layer {}\";\n", layer_idx));
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightblue;\n");
        
        // 添加层内节点：Attention -> Norm -> MLP -> Norm
        dot.push_str(&format!("        attn_{} [label=\"MultiHeadAttention\\n{}x{}\", fillcolor=lightgreen];\n", 
            layer_idx, num_heads, hidden_dim));
        dot.push_str(&format!("        mlp_{} [label=\"FeedForward\\n{}->{}\", fillcolor=lightcoral];\n", 
            layer_idx, hidden_dim, intermediate_dim));
        // ...
        
        dot.push_str("    }\n");
    }
    
    // 4. 添加层间连接
    for layer_idx in 0..num_layers - 1 {
        dot.push_str(&format!("    layer_{}_out -> layer_{}_in;\n", 
            layer_idx, layer_idx + 1));
    }
    
    dot.push_str("}");
    dot
}
```

**使用方法**：
```bash
# 1. 导出 DOT
cargo run --example export_model_dot --features transformer

# 2. 用 Graphviz 渲染
dot -Tpng model.dot -o model.png
dot -Tsvg model.dot -o model.svg
```

#### 方案 B：交互式 HTML 可视化（推荐）

创建新的可视化模块 `src/export/html.rs`：

```rust
//! HTML 交互式可视化导出
//!
//! 使用 D3.js + Cytoscape.js 实现交互式图可视化

use crate::graph::Graph;
use crate::transformer::optimization::{OperatorType, WeightTensor};

pub fn export_transformer_html(
    graph: &Graph<OperatorType, WeightTensor>,
    output_path: &str,
) -> GraphResult<()> {
    let html = generate_html_content(graph);
    std::fs::write(output_path, html)?;
    Ok(())
}

fn generate_html_content(graph: &Graph<OperatorType, WeightTensor>) -> String {
    format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Transformer Architecture Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        #cy {{
            width: 100%;
            height: 800px;
            display: block;
        }}
        .node-attention {{ background-color: #4CAF50; }}
        .node-mlp {{ background-color: #f44336; }}
        .node-norm {{ background-color: #2196F3; }}
        .node-embedding {{ background-color: #FFC107; }}
    </style>
</head>
<body>
    <h1>Transformer Architecture</h1>
    <div id="cy"></div>
    
    <script>
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {elements},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'background-color': 'data(color)',
                        'width': 60,
                        'height': 60
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle'
                    }}
                }}
            ],
            layout: {{
                name: 'dagre',  // DAG 布局，适合 Transformer
                rankDir: 'TB',
                animate: true
            }}
        }});
    </script>
</body>
</html>
    "#, elements = generate_cytoscape_elements(graph))
}

fn generate_cytoscape_elements(graph: &Graph<OperatorType, WeightTensor>) -> String {
    let mut elements = Vec::new();
    
    // 添加节点
    for node in graph.nodes() {
        let (color, label) = match node.data() {
            OperatorType::Attention { num_heads, hidden_dim } => {
                ("#4CAF50", format!("Attention\n{}x{}", num_heads, hidden_dim))
            }
            OperatorType::MLP { hidden_dim, activation } => {
                ("#f44336", format!("MLP\n{}", activation))
            }
            OperatorType::Norm { norm_type, .. } => {
                ("#2196F3", format!("Norm\n{}", norm_type))
            }
            OperatorType::Embedding { vocab_size, embed_dim } => {
                ("#FFC107", format!("Embedding\n{}x{}", vocab_size, embed_dim))
            }
            _ => ("#9E9E9E", "Other".to_string()),
        };
        
        elements.push(format!(
            r#"{{ data: {{ id: '{}', label: '{}', color: '{}' }} }}"#,
            node.index().index(), label, color
        ));
    }
    
    // 添加边
    for edge in graph.edges() {
        elements.push(format!(
            r#"{{ data: {{ source: '{}', target: '{}', weight: {} }} }}"#,
            edge.source().index().index(),
            edge.target().index().index(),
            edge.data().shape.iter().product::<usize>()
        ));
    }
    
    format!("[{}]", elements.join(","))
}
```

**使用方法**：
```bash
# 导出交互式 HTML
cargo run --example export_model_html --features transformer

# 在浏览器中打开
open model_visualization.html
```

#### 方案 C：3D 可视化（未来）

使用 Three.js 实现 3D 架构可视化：
- Z 轴：网络深度（层数）
- XY 平面：每层内的模块布局
- 颜色：模块类型
- 透明度：权重稀疏度

### 2.3 可视化内容设计

| 可视化层级 | 内容 | 推荐形式 |
|-----------|------|---------|
| **整体架构** | 所有层的宏观视图 | DOT/SVG 静态图 |
| **单层细节** | Attention + MLP + Norm | HTML 交互式 |
| **注意力头** | 每个头的权重矩阵 | 热力图 |
| **权重分布** | 正交化前后的奇异值 | 直方图 |
| **压缩效果** | 张量环压缩比 | 饼图 |

---

## 任务 3：项目核心定位分析

### 3.1 God-Graph 想解决什么问题？

**核心问题**：LLM 是黑盒，无法像 CAD 软件设计机械零件那样**可解释、可编辑、可验证**

### 3.2 三大核心价值

#### 价值 1：白盒化（从黑盒到透明）

**问题**：
- 传统 LLM：权重是扁平的 `.safetensors` 文件
- 无法直观理解模型结构
- 无法检测拓扑缺陷（孤立节点、梯度阻断）

**God-Graph 方案**：
```rust
// 1. 加载模型到图结构
let graph = ModelSwitch::load_from_safetensors("llama-1b.safetensors")?;

// 2. 拓扑验证（发现设计缺陷）
let report = ModelSwitch::validate_topology(&graph)?;
println!("连通分量：{}", report.connected_components);  // 应该=1
println!("是否 DAG: {}", report.is_dag);  // 前馈模型应该=true

// 3. 发现拓扑问题
for issue in &report.issues {
    println!("问题：{}", issue);
}
// 输出示例：
// "Layer 5 缺少残差连接"
// "Attention head 3 与其他头无连接"
```

**类比 CAD**：
- CAD 检查"表面断裂" → God-Graph 检查"孤立注意力头"
- CAD 检查"非流形几何" → God-Graph 检查"梯度阻断"

---

#### 价值 2：数学优化（从经验到理论）

**问题**：
- 传统 LLM 优化靠经验（"试试这个学习率"）
- 权重矩阵数值不稳定（条件数大）
- 模型冗余（过参数化）

**God-Graph 方案**：

**李群正交化**（数值稳定性）：
```rust
use god_gragh::transformer::optimization::LieGroupOptimizer;

let optimizer = LieGroupOptimizer::new(LieGroupConfig::new());

// 正交化所有权重（保证 W^T W = I）
optimizer.orthogonalize_weights(&mut graph)?;

// 验证正交性
let error = check_orthogonality(&graph);
println!("正交化误差：{:.2e}", error);  // 2.04e-14（机器精度）
```

**数学原理**：
- 李代数 `so(n)` → 李群 `SO(n)` 的指数映射
- 保证权重矩阵正交，避免梯度消失/爆炸

**张量环压缩**（模型轻量化）：
```rust
use god_gragh::transformer::optimization::TensorRingCompressor;

let compressor = TensorRingCompressor::new(CompressionConfig::new());

// 压缩模型（4-8 倍压缩比）
let report = compressor.compress_model(&mut graph)?;
println!("压缩比：{:.2}x", report.compression_ratio);
println!("参数量：{}M -> {}M", report.original_params, report.compressed_params);
```

**数学原理**：
- 张量环分解：`W(i,j) = Tr[G₁(:,i,:) × G₂(:,j,:)]`
- 环闭合约束：`r₀ = r₂`

---

#### 价值 3：可微图结构（从静态到动态）

**问题**：
- 传统 LLM 架构是**静态**的（设计好后不能改）
- 神经架构搜索（NAS）需要手动定义搜索空间
- 无法用梯度下降优化架构本身

**God-Graph 方案**：DifferentiableGraph（核心创新）

```rust
use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};

// 1. 从标准 Transformer 构建可微图
let mut graph = build_transformer();
let config = GradientConfig::default().with_sparsity(0.1);
let mut diff_graph = DifferentiableGraph::from_graph(graph, config);

// 2. 定义损失函数（注意力熵 + 稀疏性）
let loss_fn = |g: &DifferentiableGraph| {
    g.entropy_loss() + 0.1 * g.sparsity_loss()
};

// 3. 梯度下降优化架构
for step in 0..100 {
    let loss = loss_fn(&diff_graph);
    let grads = diff_graph.compute_structure_gradients(loss);
    diff_graph.update_structure(&grads, 0.01);
    
    if step % 10 == 0 {
        println!("Step {}: loss={:.4}", step, loss);
    }
}

// 4. 导出优化后的架构
let policy = ThresholdEditPolicy::new(0.5);
let pruned_graph = diff_graph.discretize(&policy);
println!("剪枝了 {} 条弱注意力边", pruned_graph.num_pruned_edges());
```

**数学原理**：
- 连续松弛：边存在概率 `p ∈ [0,1]` 而非 `{0,1}`
- Gumbel-Softmax：可微分的离散采样
- STE（Straight-Through Estimator）：前向离散、反向连续

**类比 CAD**：
- CAD 参数化设计：尺寸驱动几何变形
- DifferentiableGraph：梯度驱动架构演化

---

### 3.3 目标用户

| 用户类型 | 需求 | God-Graph 价值 |
|---------|------|---------------|
| **模型压缩工程师** | 减小模型体积 | 张量环压缩 4-8 倍 |
| **模型部署工程师** | 数值稳定性 | 李群正交化（误差 2e-14） |
| **架构研究员** | 自动架构搜索 | DifferentiableGraph 梯度优化 |
| **可解释性研究员** | 理解模型内部 | 拓扑可视化 + 缺陷检测 |

---

### 3.4 与竞品对比

| 工具 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| **llama.cpp** | 推理引擎 | 速度快、量化支持 | 黑盒、不可编辑 |
| **HuggingFace** | 模型库 | 生态丰富 | 黑盒、拓扑不可见 |
| **DGL/PyG** | GNN 训练 | 图神经网络 | 不支持 LLM 白盒分析 |
| **petgraph** | 通用图库 | 算法全面 | 无 LLM 专用功能 |
| **God-Graph** | **LLM 白盒工具** | **拓扑可见、数学优化、可微架构** | **推理速度慢** |

---

### 3.5 一句话定位

> **God-Graph 是 LLM 界的 CAD 软件**——用梯度下降优化架构，用李群理论保证稳定性，用张量环实现压缩，把 LLM 从黑盒变成可编辑的白盒。

---

## 实施路线图

### P0（立即）
- [ ] 修改 `Cargo.toml` 默认启用 `parallel` 特性
- [ ] 创建 `export_transformer_to_dot()` 示例
- [ ] 编写可视化使用教程

### P1（1 周）
- [ ] 改造 `load_from_safetensors()` 使用并行迭代器
- [ ] 实现 HTML 交互式可视化（Cytoscape.js）
- [ ] 添加真实模型可视化示例（TinyLlama）

### P2（1 月）
- [ ] 李群正交化并行化
- [ ] 张量环压缩并行化
- [ ] 3D 可视化原型（Three.js）

---

## 总结

**三件事的答案**：

1. **CPU 并行化**：启用 `parallel` 特性 + 改造核心函数使用 Rayon 并行迭代器，预期 7-8x 加速
2. **可视化方案**：已有 DOT/SVG 导出，增强为 HTML 交互式可视化（Cytoscape.js）
3. **核心定位**：LLM 界的 CAD 软件——白盒化、数学优化、可微架构
