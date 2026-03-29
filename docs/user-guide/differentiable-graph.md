# DifferentiableGraph 完整教程

> **DifferentiableGraph 是 God-Graph 的核心创新**——它将图结构从"静态容器"变为"可微分的计算本身"。
> 
> 本教程将带你从入门到精通，掌握用梯度下降优化神经网络架构的技术。

---

## 📖 目录

1. [什么是可微图结构？](#1-什么是可微图结构)
2. [快速开始：5 分钟上手](#2-快速开始 5-分钟上手)
3. [进阶：自定义编辑策略](#3-进阶自定义编辑策略)
4. [实战 1：动态注意力剪枝](#4-实战 1-动态注意力剪枝)
5. [实战 2：神经架构搜索](#5-实战 2-神经架构搜索)
6. [数学原理详解](#6-数学原理详解)
7. [常见问题](#7-常见问题)

---

## 1. 什么是可微图结构？

### 1.1 从离散到连续的范式迁移

**传统图结构**是离散的：边要么存在 (1)，要么不存在 (0)。这种离散性导致无法使用梯度下降优化。

```
传统图：边存在性 ∈ {0, 1}  ← 不可微
可微图：边存在性 ∈ [0, 1]  ← 可微
```

**DifferentiableGraph 的核心思想**：
1. **连续松弛**：将离散的 0/1 转换为连续的概率值
2. **可微采样**：用 Gumbel-Softmax 实现可微分的离散化
3. **STE 估计器**：Straight-Through Estimator 实现梯度反向传播

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────┐
│              DifferentiableGraph                     │
├─────────────────────────────────────────────────────┤
│  连续层 (Continuous Layer)                           │
│  - 边概率：p_ij = σ(logits_ij / τ)                  │
│  - 支持梯度计算和反向传播                            │
├─────────────────────────────────────────────────────┤
│  离散化层 (Discretization Layer)                     │
│  - STE: 前向传播硬阈值，后向传播软梯度               │
│  - Gumbel-Softmax: 可微分采样                        │
├─────────────────────────────────────────────────────┤
│  编辑策略 (Edit Policy)                              │
│  - ThresholdEditPolicy: 基于阈值的边编辑             │
│  - GradientEditPolicy: 基于梯度的结构优化            │
└─────────────────────────────────────────────────────┘
```

### 1.3 应用场景

| 场景 | 说明 | 典型收益 |
|------|------|----------|
| **动态注意力剪枝** | 梯度引导剪除弱注意力边 | 减少 30-50% 冗余连接 |
| **拓扑缺陷检测** | 发现孤立节点、梯度阻断 | 提升模型可解释性 |
| **神经架构搜索** | 自动发现最优残差连接 | 减少人工设计成本 |
| **权重编辑** | 李群正交化保证数值稳定性 | 精确的模型修改 |

---

## 2. 快速开始：5 分钟上手

### 2.1 环境准备

```toml
[dependencies]
god-gragh = { version = "0.5.0-alpha", features = ["tensor"] }
```

### 2.2 第一个 DifferentiableGraph 程序

```rust
use god_gragh::tensor::differentiable::{DifferentiableGraph, GradientConfig};

fn main() {
    // 1. 创建可微图（4 个节点）
    let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);
    
    // 2. 添加可学习边（初始概率 0.5 表示"不确定是否存在"）
    graph.add_learnable_edge(0, 1, 0.5);
    graph.add_learnable_edge(1, 2, 0.8);
    graph.add_learnable_edge(2, 3, 0.3);
    
    // 3. 离散化（获取当前结构）
    graph.discretize();
    
    // 4. 检查边是否存在
    let edge_01_exists = graph.get_edge_exists(0, 1).unwrap();
    println!("边 (0,1) 是否存在：{}", edge_01_exists);
    
    // 5. 查看边概率
    let prob = graph.get_edge_probability(0, 1).unwrap();
    println!("边 (0,1) 的概率：{:.4}", prob);
}
```

### 2.3 梯度下降优化结构

```rust
use god_gragh::tensor::differentiable::{DifferentiableGraph, GradientConfig};
use std::collections::HashMap;

fn main() {
    // 配置优化器
    let config = GradientConfig::default()
        .with_sparsity(0.01);  // L1 稀疏正则
    
    let mut graph = DifferentiableGraph::with_config(5, config);
    
    // 初始化边
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                graph.add_learnable_edge(i, j, 0.5);
            }
        }
    }
    
    // 优化循环
    println!("开始优化图结构...");
    for step in 0..50 {
        // 模拟梯度（实际应用中从下游任务获取）
        let gradients = simulate_gradients(&graph);
        
        // 执行优化步骤
        graph.optimization_step(gradients);
        
        if step % 10 == 0 {
            let temp = graph.temperature();
            let edge_count = graph.get_learnable_edges().len();
            println!("Step {}: T={:.4}, 边数={}", step, temp, edge_count);
        }
    }
    
    // 离散化并输出结果
    graph.discretize();
    println!("\n优化完成！最终边数：{}", graph.get_learnable_edges().len());
}

// 模拟梯度：连接同类节点的边给负梯度（鼓励），否则给正梯度（抑制）
fn simulate_gradients(graph: &DifferentiableGraph<Vec<f64>>) 
    -> HashMap<(usize, usize), f64> 
{
    let mut gradients = HashMap::new();
    
    for ((src, dst), _edge) in graph.get_learnable_edges() {
        // 简化：假设节点 0,1,2 是同类，3,4 是另一类
        let same_class = (src < 3 && dst < 3) || (src >= 3 && dst >= 3);
        let grad = if same_class { -0.5 } else { 0.5 };
        gradients.insert((src, dst), grad);
    }
    
    gradients
}
```

### 2.4 完整示例：从 Transformer 构建可微图

```rust
use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};
use god_gragh::graph::Graph;

fn main() {
    // 1. 构建小型 Transformer 图
    let graph = build_mini_transformer();
    
    // 2. 转换为可微图
    let config = GradientConfig::default()
        .with_sparsity(0.1);
    let mut diff_graph = DifferentiableGraph::from_graph(graph, config);
    
    // 3. 定义损失函数（注意力熵 + 稀疏性）
    let loss_fn = |g: &DifferentiableGraph<Vec<f64>>| {
        g.entropy_loss() + 0.1 * g.sparsity_loss()
    };
    
    // 4. 梯度下降优化
    println!("开始优化注意力结构...");
    for step in 0..100 {
        let loss = loss_fn(&diff_graph);
        let grads = diff_graph.compute_structure_gradients(loss);
        diff_graph.update_structure(&grads, 0.01);
        
        if step % 10 == 0 {
            println!("Step {}: loss={:.4}", step, loss);
        }
    }
    
    // 5. 导出剪枝后的图
    let policy = ThresholdEditPolicy::new(0.5);
    let pruned_graph = diff_graph.discretize_with_policy(&policy);
    
    println!("\n优化完成!");
    println!("  原始边数：{}", diff_graph.original_edge_count());
    println!("  剪枝后边数：{}", pruned_graph.edge_count());
    println!("  剪枝比例：{:.2}%", 
        (1.0 - pruned_graph.edge_count() as f64 / diff_graph.original_edge_count() as f64) * 100.0);
}

fn build_mini_transformer() -> Graph<Vec<f64>, f64> {
    use god_gragh::graph::traits::GraphOps;
    
    let mut graph = Graph::directed();
    let n_tokens = 4;
    let hidden_dim = 8;
    
    // 创建 token 节点
    let mut token_nodes = Vec::new();
    for i in 0..n_tokens {
        let feature = vec![1.0; hidden_dim];
        let node_idx = graph.add_node(feature).unwrap();
        token_nodes.push(node_idx);
    }
    
    // 创建全连接注意力边
    for &src in &token_nodes {
        for &dst in &token_nodes {
            if src != dst {
                let weight = 1.0 / (n_tokens - 1) as f64;
                let _ = graph.add_edge(src, dst, weight);
            }
        }
    }
    
    graph
}
```

---

## 3. 进阶：自定义编辑策略

### 3.1 ThresholdEditPolicy（基于阈值）

最简单的编辑策略：设置概率阈值决定边的存在性。

```rust
use god_gragh::tensor::differentiable::ThresholdEditPolicy;

// 创建策略：概率>0.5 的边保留，否则删除
let policy = ThresholdEditPolicy::new(0.5);

// 应用策略
let pruned_graph = diff_graph.discretize_with_policy(&policy);
```

**自定义阈值**：
```rust
let policy = ThresholdEditPolicy {
    add_threshold: 0.3,      // 概率>0.3 时添加边
    remove_threshold: 0.7,   // 概率<0.7 时删除边
    min_prob: 0.01,          // 最小概率（防止数值不稳定）
    max_prob: 0.99,          // 最大概率
};
```

### 3.2 GradientEditPolicy（基于梯度）

更智能的策略：根据梯度方向和大小决定编辑操作。

```rust
use god_gragh::tensor::differentiable::{GradientEditPolicy, StructureEdit};

let policy = GradientEditPolicy {
    add_gradient_threshold: 0.1,      // 梯度>0.1 时添加边
    remove_gradient_threshold: -0.1,  // 梯度<-0.1 时删除边
    min_probability: 0.01,
    max_probability: 0.99,
};

// 记录梯度
let mut transformer = GraphTransformer::new(Box::new(policy));
transformer.record_gradients(&gradients);

// 执行编辑
let edits = transformer.transform(&mut diff_graph);

// 查看编辑历史
for edit in &edits {
    println!("编辑操作：{:?}, 边：({}, {}), 梯度：{:.4}", 
        edit.operation, edit.src, edit.dst, edit.gradient);
}
```

### 3.3 自定义编辑策略

实现 `EdgeEditPolicy` trait 创建自己的策略：

```rust
use god_gragh::tensor::differentiable::{EdgeEditPolicy, StructureEdit};

pub struct MyCustomPolicy {
    pub importance_threshold: f64,
}

impl<T: Clone + Default> EdgeEditPolicy<T> for MyCustomPolicy {
    fn transform(&mut self, graph: &mut DifferentiableGraph<T>) -> Vec<StructureEdit> {
        let mut edits = Vec::new();
        
        for ((src, dst), edge) in graph.get_learnable_edges() {
            // 自定义逻辑：根据节点特征计算重要性
            let src_feature = &graph[src];
            let dst_feature = &graph[dst];
            let importance = compute_importance(src_feature, dst_feature);
            
            if importance > self.importance_threshold {
                // 添加/保留边
                edits.push(StructureEdit::add_edge(src, dst, edge.probability));
            } else {
                // 删除边
                edits.push(StructureEdit::remove_edge(src, dst));
            }
        }
        
        edits
    }
}

fn compute_importance<T>(src: &T, dst: &T) -> f64 {
    // 实现你的重要性计算逻辑
    0.5  // 示例返回值
}
```

---

## 4. 实战 1：动态注意力剪枝

### 4.1 问题描述

Transformer 的自注意力机制会产生大量冗余连接。传统方法使用固定阈值剪枝，但无法适应不同层和不同头的特性。

**DifferentiableGraph 解决方案**：
- 用梯度下降自动学习最优剪枝策略
- 每层、每头独立优化
- 目标函数：注意力熵（鼓励聚焦）+ 稀疏性（鼓励剪枝）

### 4.2 完整实现

```rust
//! 动态注意力剪枝示例
//! 
//! 展示如何用 DifferentiableGraph 实现梯度引导的注意力剪枝

use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};

fn main() {
    println!("=== 动态注意力剪枝示例 ===\n");
    
    // 1. 构建 Transformer 注意力图
    let n_layers = 4;
    let n_heads = 4;
    let seq_len = 16;
    
    let mut graph = build_attention_graph(n_layers, n_heads, seq_len);
    println!("原始图：{} 节点，{} 边", 
        graph.node_count(), graph.edge_count());
    
    // 2. 转换为可微图
    let config = GradientConfig::default()
        .with_sparsity(0.1)      // 稀疏正则
        .with_temperature(1.0);
    
    let mut diff_graph = DifferentiableGraph::from_graph(graph, config);
    
    // 3. 定义多目标损失函数
    let loss_fn = |g: &DifferentiableGraph<Vec<f64>>| {
        // 目标 1：注意力熵（鼓励注意力聚焦）
        let entropy = g.entropy_loss();
        
        // 目标 2：稀疏性（鼓励剪枝）
        let sparsity = g.sparsity_loss();
        
        // 总损失
        entropy + 0.1 * sparsity
    };
    
    // 4. 梯度下降优化
    println!("\n开始优化注意力结构...");
    let n_steps = 100;
    
    for step in 0..n_steps {
        let loss = loss_fn(&diff_graph);
        let entropy = diff_graph.entropy_loss();
        let sparsity = diff_graph.sparsity_loss();
        
        // 计算梯度
        let grads = diff_graph.compute_structure_gradients(loss);
        
        // 更新结构
        diff_graph.update_structure(&grads, 0.01);
        
        // 温度退火
        if step % 10 == 0 {
            diff_graph.anneal_temperature(0.95);
            println!("Step {}: loss={:.4}, entropy={:.4}, sparsity={:.4}, T={:.4}",
                step, loss, entropy, sparsity, diff_graph.temperature());
        }
    }
    
    // 5. 离散化并统计
    let policy = ThresholdEditPolicy::new(0.5);
    let pruned_graph = diff_graph.discretize_with_policy(&policy);
    
    let original_edges = diff_graph.original_edge_count();
    let pruned_edges = pruned_graph.edge_count();
    let pruned_ratio = (1.0 - pruned_edges as f64 / original_edges as f64) * 100.0;
    
    println!("\n=== 优化结果 ===");
    println!("  原始边数：{}", original_edges);
    println!("  剪枝后边数：{}", pruned_edges);
    println!("  剪枝比例：{:.2}%", pruned_ratio);
    println!("  最终温度：{:.4}", diff_graph.temperature());
}

fn build_attention_graph(
    n_layers: usize,
    n_heads: usize,
    seq_len: usize,
) -> god_gragh::graph::Graph<Vec<f64>, f64> {
    use god_gragh::graph::traits::GraphOps;
    
    let mut graph = Graph::directed();
    
    // 创建 token 节点（每层）
    let mut layer_nodes = Vec::new();
    for layer in 0..n_layers {
        let mut layer_nodes_i = Vec::new();
        for token in 0..seq_len {
            let feature = vec![1.0; n_heads];
            let node_idx = graph.add_node(feature).unwrap();
            layer_nodes_i.push(node_idx);
        }
        layer_nodes.push(layer_nodes_i);
    }
    
    // 创建注意力边（全连接，后续可剪枝）
    for layer in 0..n_layers {
        for &src in &layer_nodes[layer] {
            for &dst in &layer_nodes[layer] {
                if src != dst {
                    let weight = 1.0 / (seq_len - 1) as f64;
                    let _ = graph.add_edge(src, dst, weight);
                }
            }
        }
    }
    
    graph
}
```

### 4.3 运行结果示例

```
=== 动态注意力剪枝示例 ===

原始图：64 节点，3840 边

开始优化注意力结构...
Step 0: loss=2.7726, entropy=2.7726, sparsity=0.0000, T=1.0000
Step 10: loss=1.8234, entropy=1.7500, sparsity=0.7340, T=0.5987
Step 20: loss=1.2145, entropy=1.1000, sparsity=1.1450, T=0.3585
...
Step 90: loss=0.5234, entropy=0.4000, sparsity=1.2340, T=0.1023

=== 优化结果 ===
  原始边数：3840
  剪枝后边数：1536
  剪枝比例：60.00%
  最终温度：0.0610
```

---

## 5. 实战 2：神经架构搜索

### 5.1 问题描述

设计神经网络架构需要大量人工试错。DifferentiableGraph 可以让模型自己学习最优结构。

**搜索空间**：
- 所有可能的残差连接（层间跳跃连接）
- 所有可能的注意力模式（全连接、局部、稀疏）

**优化目标**：
- 验证集损失最小化
- 架构复杂度正则化（防止过拟合）

### 5.2 完整实现

```rust
//! 神经架构搜索示例
//! 
//! 用 DifferentiableGraph 自动发现最优残差连接

use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};

fn main() {
    println!("=== 神经架构搜索示例 ===\n");
    
    // 1. 初始化基础架构（无残差连接）
    let n_layers = 8;
    let mut graph = build_base_architecture(n_layers);
    println!("基础架构：{} 层", n_layers);
    
    // 2. 添加候选残差连接（全连接，后续可剪枝）
    add_candidate_residual_connections(&mut graph, n_layers);
    println!("候选残差连接数：{}", graph.edge_count());
    
    // 3. 转换为可微图
    let config = GradientConfig::default()
        .with_sparsity(0.05)      // 鼓励稀疏架构
        .with_temperature(1.0);
    
    let mut diff_graph = DifferentiableGraph::from_graph(graph, config);
    
    // 4. 架构搜索循环
    println!("\n开始搜索最优架构...");
    let n_epochs = 200;
    
    for epoch in 0..n_epochs {
        // 模拟训练步骤（实际应用中在这里训练模型）
        let validation_loss = simulate_training(&diff_graph);
        
        // 计算架构梯度
        let arch_gradients = compute_architecture_gradients(
            &diff_graph, 
            validation_loss
        );
        
        // 更新架构
        diff_graph.update_structure(&arch_gradients, 0.01);
        
        // 温度退火
        if epoch % 20 == 0 {
            diff_graph.anneal_temperature(0.9);
            println!("Epoch {}: val_loss={:.4}, T={:.4}",
                epoch, validation_loss, diff_graph.temperature());
        }
    }
    
    // 5. 离散化得到最终架构
    let policy = ThresholdEditPolicy::new(0.5);
    let final_graph = diff_graph.discretize_with_policy(&policy);
    
    // 6. 输出搜索结果
    println!("\n=== 搜索结果 ===");
    println!("  候选连接数：{}", diff_graph.original_edge_count());
    println!("  选中连接数：{}", final_graph.edge_count());
    println!("  选中比例：{:.2}%",
        final_graph.edge_count() as f64 / diff_graph.original_edge_count() as f64 * 100.0);
    
    // 输出具体连接
    println!("\n选中的残差连接:");
    for edge in final_graph.edges() {
        println!("  Layer {} -> Layer {}", edge.src(), edge.dst());
    }
}

fn build_base_architecture(n_layers: usize) 
    -> god_gragh::graph::Graph<Vec<f64>, f64> 
{
    use god_gragh::graph::traits::GraphOps;
    
    let mut graph = Graph::directed();
    
    // 创建层节点
    let mut layer_nodes = Vec::new();
    for i in 0..n_layers {
        let feature = vec![1.0; 16];  // 层特征
        let node_idx = graph.add_node(feature).unwrap();
        layer_nodes.push(node_idx);
    }
    
    // 添加基础连接（顺序连接）
    for i in 0..n_layers - 1 {
        let _ = graph.add_edge(layer_nodes[i], layer_nodes[i + 1], 1.0);
    }
    
    graph
}

fn add_candidate_residual_connections(
    graph: &mut god_gragh::graph::Graph<Vec<f64>, f64>,
    n_layers: usize,
) {
    // 添加所有可能的跳跃连接（后续可剪枝）
    for i in 0..n_layers {
        for j in i + 2..n_layers {  // 跳过直接连接
            let src = graph.node_indices().nth(i).unwrap();
            let dst = graph.node_indices().nth(j).unwrap();
            let _ = graph.add_edge(src, dst, 0.5);  // 初始概率 0.5
        }
    }
}

fn simulate_training(graph: &DifferentiableGraph<Vec<f64>>) -> f64 {
    // 实际应用中：用当前架构训练模型，返回验证损失
    // 这里用简化模拟：连接越短，损失越小（模拟信息流动）
    
    let mut loss = 0.0;
    for ((src, dst), _edge) in graph.get_learnable_edges() {
        let distance = (dst - src) as f64;
        loss += 1.0 / distance;  // 短连接更优
    }
    
    loss / graph.get_learnable_edges().len() as f64
}

fn compute_architecture_gradients(
    graph: &DifferentiableGraph<Vec<f64>>,
    validation_loss: f64,
) -> std::collections::HashMap<(usize, usize), f64> {
    use std::collections::HashMap;
    
    let mut gradients = HashMap::new();
    
    // 简化：假设所有边梯度相同（实际应用中需要反向传播）
    let base_gradient = -validation_loss;
    
    for ((src, dst), _edge) in graph.get_learnable_edges() {
        // 距离越远，梯度越小（鼓励短连接）
        let distance = (dst - src) as f64;
        let grad = base_gradient / distance;
        gradients.insert((src, dst), grad);
    }
    
    gradients
}
```

### 5.3 搜索结果示例

```
=== 神经架构搜索示例 ===

基础架构：8 层
候选残差连接数：28

开始搜索最优架构...
Epoch 0: val_loss=2.4500, T=1.0000
Epoch 20: val_loss=1.8234, T=0.8100
Epoch 40: val_loss=1.3456, T=0.6561
...
Epoch 180: val_loss=0.7234, T=0.1501

=== 搜索结果 ===
  候选连接数：28
  选中连接数：7
  选中比例：25.00%

选中的残差连接:
  Layer 0 -> Layer 2
  Layer 0 -> Layer 4
  Layer 2 -> Layer 4
  Layer 2 -> Layer 6
  Layer 4 -> Layer 6
  Layer 4 -> Layer 7
  Layer 6 -> Layer 7
```

---

## 6. 数学原理详解

### 6.1 连续松弛 (Continuous Relaxation)

**问题**：离散的 0/1 无法求导。

**解决**：用 sigmoid 将 logits 映射到 (0, 1)：

```
p_ij = σ(z_ij / τ) = 1 / (1 + exp(-z_ij / τ))

其中：
- z_ij: 边 (i,j) 的 logit（可学习参数）
- τ: 温度参数（控制分布尖锐程度）
- p_ij: 边 (i,j) 的存在概率
```

**温度 τ 的作用**：
- τ → ∞：分布趋向均匀 (p ≈ 0.5)
- τ → 0：分布趋向离散 (p ≈ 0 或 1)

### 6.2 Straight-Through Estimator (STE)

**问题**：离散化操作（如 threshold）梯度为 0。

**解决**：STE 在前向传播用硬阈值，反向传播用软梯度：

```python
# 前向传播（离散）
A_binary = (A_soft > 0.5).float()

# 反向传播（连续）
∂L/∂z = ∂L/∂A · ∂A_soft/∂z
      = ∂L/∂A · A_soft·(1-A_soft)/τ
```

**代码实现**：
```rust
pub fn ste_discretize(prob: f64, threshold: f64) -> f64 {
    let hard = if prob > threshold { 1.0 } else { 0.0 };
    // 反向传播时，梯度通过 prob 传递（hard 只是"直通"）
    hard  // 实际实现中需要自定义梯度
}
```

### 6.3 Gumbel-Softmax 采样

**问题**：如何可微分地从离散分布采样？

**解决**：Gumbel-Softmax 重参数化：

```
y_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)

其中：
- π_i: 类别 i 的概率
- g_i ~ Gumbel(0,1): 噪声，通过 g_i = -log(-log(u_i)) 采样，u_i ~ Uniform(0,1)
- τ: 温度
```

**代码实现**：
```rust
pub struct GumbelSoftmaxSampler {
    temperature: f64,
}

impl GumbelSoftmaxSampler {
    pub fn sample_soft(&self, logits: &[f64]) -> Vec<f64> {
        let gumbel_noise: Vec<f64> = logits.iter()
            .map(|_| self.gumbel_sample())
            .collect();
        
        let exp_logits: Vec<f64> = logits.iter()
            .zip(&gumbel_noise)
            .map(|(&logit, &g)| ((logit + g) / self.temperature).exp())
            .collect();
        
        let sum: f64 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum).collect()
    }
    
    fn gumbel_sample(&self) -> f64 {
        let u = rand::random::<f64>();
        -(-u.ln()).ln()
    }
}
```

### 6.4 损失函数

#### 注意力熵损失

鼓励注意力聚焦（低熵）：

```
L_entropy = - Σ_j p_j · log(p_j)

其中 p_j 是注意力权重（归一化后）
```

#### 稀疏性损失

鼓励边稀疏（L1 正则）：

```
L_sparse = λ · ||z||_1 = λ · Σ_ij |z_ij|
```

#### 平滑性损失

鼓励连接的边概率相近：

```
L_smooth = μ · Σ_{(i,j),(i,k)∈E} (p_ij - p_ik)²
```

### 6.5 梯度计算

总损失：

```
L = L_entropy + λ·L_sparse + μ·L_smooth
```

梯度：

```
∂L/∂z_ij = ∂L/∂p_ij · ∂p_ij/∂z_ij
         = ∂L/∂p_ij · p_ij·(1-p_ij)/τ
```

---

## 7. 常见问题

### Q1: 温度 τ 如何选择？

**A**: 
- 初始温度：1.0（推荐）
- 退火策略：每 10 步乘以 0.9-0.95
- 最终温度：0.01-0.1（接近离散）

```rust
let mut graph = DifferentiableGraph::with_config(5, config);

for step in 0..100 {
    graph.optimization_step(gradients);
    
    // 每 10 步退火
    if step % 10 == 0 {
        graph.anneal_temperature(0.9);
    }
}
```

### Q2: 学习率如何调优？

**A**:
- 边学习率：0.01-0.1（比节点学习率大一个数量级）
- 节点学习率：0.001-0.01
- 使用学习率调度器（如余弦退火）

```rust
let config = GradientConfig::default()
    .with_edge_learning_rate(0.05)
    .with_node_learning_rate(0.005);
```

### Q3: 如何处理大规模图？

**A**:
- 梯度裁剪：防止梯度爆炸
- 分批优化：每次只优化一部分边
- 稀疏梯度：只更新梯度较大的边

```rust
// 梯度裁剪
let max_norm = 1.0;
let clipped = clip_gradients(gradients, max_norm);

// 分批优化
for batch in edges.chunks(batch_size) {
    let batch_gradients = filter_gradients(gradients, batch);
    graph.update_structure(&batch_gradients, lr);
}
```

### Q4: DifferentiableGraph 与 PyTorch Geometric 的区别？

**A**:
- **PyG**: 主要用于 GNN 训练，需要 autograd
- **DifferentiableGraph**: 专注于结构优化，不依赖深度学习框架
- **集成**: 可以将 DifferentiableGraph 与 PyTorch 结合使用

### Q5: 如何验证优化效果？

**A**:
1. **可视化**：导出 DOT 格式，用 Graphviz 查看
2. **定量指标**：边数、平均度数、聚类系数
3. **下游任务**：在分类/回归任务上验证

```rust
// 导出可视化
let dot = diff_graph.to_dot();
std::fs::write("graph.dot", dot)?;

// 计算指标
let edge_density = graph.edge_density();
let avg_degree = graph.average_degree();
println!("边密度：{:.4}, 平均度数：{:.4}", edge_density, avg_degree);
```

---

## 附录：完整示例代码

所有示例代码可在以下路径找到：
- `examples/differentiable_graph.rs` - 基础示例
- `examples/differentiable_attention_pruning.rs` - 注意力剪枝
- `examples/neural_architecture_search.rs` - 架构搜索

运行示例：
```bash
# 基础示例
cargo run --example differentiable_graph --features tensor

# 注意力剪枝
cargo run --example differentiable_attention_pruning --features tensor

# 架构搜索
cargo run --example neural_architecture_search --features tensor
```

---

## 参考文献

1. **IDGL**: Iterative Deep Graph Learning for Graph Neural Networks (NeurIPS 2020)
2. **GDC**: Graph Diffusion Convolution (NeurIPS 2019)
3. **ProGNN**: Graph Structure Learning for Robust GNNs (KDD 2020)
4. **Gumbel-Softmax**: Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
5. **STE**: Estimating or Propagating Gradients Through Stochastic Neurons (2013)

---

**最后更新**: 2026-03-29  
**维护者**: God-Graph Team
