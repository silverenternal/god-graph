# 模型可视化教程

**版本**: 0.5.0-alpha
**日期**: 2026-03-29

---

## 📋 概述

God-Graph 支持多种可视化方式，帮助你直观理解 LLM 架构：

| 可视化类型 | 格式 | 适用场景 | 工具 |
|-----------|------|---------|------|
| **DOT/Graphviz** | 静态图 | 整体架构、论文插图 | Graphviz |
| **SVG 矢量图** | 交互式 SVG | Web 展示、缩放查看 | 浏览器 |
| **HTML 交互式** | D3.js/Cytoscape | 动态探索、调试 | 浏览器 |

---

## 🚀 快速开始

### 方法 1：DOT 格式（推荐）

#### 步骤 1：导出模型

```bash
# 使用示例模型（无需真实模型文件）
cargo run --example export_model_dot --features transformer

# 使用真实模型
cargo run --example export_model_dot --features transformer -- model.safetensors model.dot
```

#### 步骤 2：渲染为图片

```bash
# 安装 Graphviz
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# 渲染为 PNG
dot -Tpng model.dot -o model.png

# 渲染为 SVG（推荐，可缩放）
dot -Tsvg model.dot -o model.svg

# 渲染为 PDF（高质量打印）
dot -Tpdf model.dot -o model.pdf
```

#### 步骤 3：查看结果

```bash
# Linux
xdg-open model.png

# macOS
open model.png

# Windows
start model.png
```

---

### 方法 2：SVG 直接导出

```rust
use god_graph::export::svg::{to_svg, SvgOptions};
use god_graph::transformer::optimization::ModelSwitch;

// 加载模型
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 导出为 SVG
let options = SvgOptions::new()
    .with_size(1200, 800)
    .with_node_radius(25.0)
    .with_layout(god_graph::export::svg::LayoutAlgorithm::Hierarchical);

let svg = to_svg(&graph, options);
std::fs::write("model.svg", svg)?;
```

---

## 📊 可视化示例

### 示例 1：完整 Transformer 架构

```bash
cargo run --example export_model_dot --features transformer
dot -Tsvg model.dot -o transformer.svg
```

**输出示例**：

```
digraph Transformer {
    rankdir=TB;
    splines=ortho;
    
    subgraph cluster_embedding {
        label="Embedding";
        fillcolor=lightyellow;
        n0 [label="Embedding\n32000x512" fillcolor="#FFC107"];
    }
    
    subgraph cluster_attention {
        label="Attention Layers";
        fillcolor=lightgreen;
        n1 [label="Attention\n8 heads\n512 dim" fillcolor="#4CAF50"];
        n4 [label="Attention\n8 heads\n512 dim" fillcolor="#4CAF50"];
    }
    
    subgraph cluster_mlp {
        label="MLP Layers";
        fillcolor=lightcoral;
        n3 [label="MLP\n512\nSiLU" fillcolor="#f44336"];
    }
    
    n0 -> n1;
    n1 -> n2;
    n2 -> n3;
}
```

---

### 示例 2：单层细节可视化

```rust
use god_graph::graph::traits::GraphQuery;
use god_graph::export::dot::{to_dot_with_options, DotOptions};

// 提取单层子图
let layer_graph = extract_layer(&graph, layer_idx=5);

// 自定义样式
let options = DotOptions::new()
    .with_name("Layer_5")
    .with_graph_attribute("rankdir", "LR")  // 从左到右
    .with_node_attribute("shape", "ellipse")
    .hide_edge_labels();

let dot = to_dot_with_options(&layer_graph, &options);
std::fs::write("layer_5.dot", dot)?;
```

---

### 示例 3：注意力头可视化

```rust
// 导出注意力权重矩阵
use god_graph::tensor::DenseTensor;

for edge in graph.edges() {
    let weight = edge.data();
    if weight.name.contains("attn") {
        // 导出为 CSV（可用 Python matplotlib 可视化）
        let csv = tensor_to_csv(&weight.data);
        std::fs::write(format!("{}.csv", weight.name), csv)?;
    }
}
```

**Python 可视化**：
```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取权重
df = pd.read_csv('layer_5_attn_weight.csv', header=None)

# 热力图
plt.figure(figsize=(10, 8))
plt.imshow(df.values, cmap='viridis')
plt.colorbar(label='Weight Value')
plt.title('Attention Weight Matrix - Layer 5')
plt.xlabel('Output Dimension')
plt.ylabel('Input Dimension')
plt.savefig('attention_heatmap.png', dpi=300)
plt.show()
```

---

## 🎨 自定义样式

### 颜色方案

| 模块类型 | 颜色代码 | 说明 |
|---------|---------|------|
| Embedding | `#FFC107` | 琥珀色 |
| Attention | `#4CAF50` | 绿色 |
| MLP | `#f44336` | 红色 |
| Norm | `#2196F3` | 蓝色 |
| Linear | `#9E9E9E` | 灰色 |

### 布局方向

```rust
// 从上到下（适合整体架构）
dot.push_str("    rankdir=TB;\n");

// 从左到右（适合单层细节）
dot.push_str("    rankdir=LR;\n");
```

### 边样式

```rust
// 正交边（直角转弯，更清晰）
dot.push_str("    splines=ortho;\n");

// 曲线边（更美观）
dot.push_str("    splines=curved;\n");

// 直线边（最简单）
dot.push_str("    splines=line;\n");
```

---

## 🔧 高级用法

### 1. 批量导出多个模型

```bash
#!/bin/bash
# export_all_models.sh

for model in models/*.safetensors; do
    name=$(basename "$model" .safetensors)
    echo "Exporting $name..."
    
    cargo run --example export_model_dot --features transformer -- \
        "$model" "output/${name}.dot"
    
    dot -Tsvg "output/${name}.dot" -o "output/${name}.svg"
done
```

---

### 2. 比较优化前后架构

```rust
// 导出优化前
export_transformer_to_dot(&original_graph, "before_optimization.dot")?;

// 应用李群正交化
optimizer.orthogonalize_weights(&mut graph)?;

// 导出优化后
export_transformer_to_dot(&graph, "after_optimization.dot")?;

// 并排比较（使用 diff 工具）
// dot -Tpng before_optimization.dot -o before.png
// dot -Tpng after_optimization.dot -o after.png
```

---

### 3. 生成报告

```rust
use god_graph::transformer::optimization::ModelSwitch;

let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 生成拓扑报告
let topology_report = ModelSwitch::validate_topology(&graph)?;
println!("拓扑验证报告:");
println!("  节点数：{}", topology_report.node_count);
println!("  边数：{}", topology_report.edge_count);
println!("  连通分量：{}", topology_report.connected_components);
println!("  是否 DAG: {}", topology_report.is_dag);
println!("  问题：{:?}", topology_report.issues);

// 导出可视化
export_transformer_to_dot(&graph, "model_with_report.dot")?;
```

---

## 📱 Web 交互式可视化（即将推出）

### HTML 交互式导出

```rust
use god_graph::export::html::export_transformer_html;

let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
export_transformer_html(&graph, "model_visualization.html")?;

// 在浏览器中打开
// open model_visualization.html  (macOS)
// xdg-open model_visualization.html  (Linux)
```

**功能**：
- ✅ 拖拽缩放
- ✅ 节点搜索
- ✅ 点击查看详情
- ✅ 权重分布直方图
- ✅ 注意力热力图

---

## 🐛 常见问题

### Q1: Graphviz 安装失败

**问题**：`dot: command not found`

**解决**：
```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# macOS
brew install graphviz

# Windows
# 下载 https://graphviz.org/download/#windows
# 安装后添加到 PATH
```

---

### Q2: 导出的图太大/太小

**问题**：节点太多看不清

**解决**：
```rust
// 方法 1：只导出部分层
let subgraph = extract_layers(&graph, &[0, 1, 2]);  // 只导出前 3 层
export_transformer_to_dot(&subgraph, "first_3_layers.dot")?;

// 方法 2：增大画布
dot.push_str("    size=\"20,30\";\n");  // 20x30 英寸
dot.push_str("    ratio=expand;\n");
```

---

### Q3: 中文标签乱码

**问题**：中文显示为方框

**解决**：
```rust
// 指定中文字体
dot.push_str("    node [fontname=\"Microsoft YaHei\"];\n");
dot.push_str("    edge [fontname=\"Microsoft YaHei\"];\n");

// 或使用英文标签
let label = format!("Attention\\n{} heads", num_heads);  // 避免中文
```

---

### Q4: 边交叉太多，看不清

**问题**：图太混乱

**解决**：
```rust
// 使用正交边
dot.push_str("    splines=ortho;\n");

// 启用层级布局
dot.push_str("    rankdir=TB;\n");

// 添加边的权重（粗边表示重要连接）
dot.push_str(&format!("    n{} -> n{} [penwidth={}];\n", 
    source, target, weight));
```

---

## 📚 相关资源

- [Graphviz 官方文档](https://graphviz.org/documentation/)
- [DOT 语言指南](https://graphviz.org/doc/info/lang.html)
- [Cytoscape.js 交互式可视化](https://js.cytoscape.org/)
- [God-Graph DifferentiableGraph 教程](differentiable_graph.md)

---

## 🎯 下一步

1. **尝试示例**：`cargo run --example export_model_dot`
2. **可视化你的模型**：导出为 SVG 查看
3. **探索 DifferentiableGraph**：用梯度优化架构

---

**最后更新**: 2026-03-29
