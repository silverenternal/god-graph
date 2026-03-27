# 可微图结构变换 (Differentiable Graph Transformation)

## 概述

本模块实现了**图结构变换操作的梯度计算**，使得可以通过梯度下降优化图的结构。这是图神经网络、图生成和图优化领域的前沿技术。

## 核心概念

### 1. 连续松弛 (Continuous Relaxation)

传统图结构是离散的：边要么存在 (1) 要么不存在 (0)。为了支持梯度计算，我们使用**连续松弛**：

```
A_soft = σ(logits / τ)

其中：
- logits: 边的对数几率（可学习参数）
- τ: 温度参数（控制离散程度）
- σ: sigmoid 函数
```

当 τ → 0 时，软松弛趋近于离散值。

### 2. Straight-Through Estimator (STE)

对于需要离散输出的场景，使用 STE：
- **前向传播**：硬阈值（0 或 1）
- **反向传播**：软梯度（通过 sigmoid）

```rust
// 离散化
exists = probability > 0.5

// 梯度传播
∂L/∂logits = ∂L/∂A · A·(1-A)/τ
```

### 3. Gumbel-Softmax 采样

用于可微的离散采样：

```
y_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)

其中：
- π: 类别概率
- g_i ~ Gumbel(0,1): 噪声
- τ: 温度
```

## 使用方法

### 基本示例

```rust
use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig
};

// 创建可微图
let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);

// 添加可学习边（初始概率表示边的存在可能性）
graph.add_learnable_edge(0, 1, 0.5);
graph.add_learnable_edge(1, 2, 0.8);
graph.add_learnable_edge(2, 3, 0.3);

// 离散化（获取当前结构）
graph.discretize();

// 获取边是否存在
let edge_01_exists = graph.get_edge_exists(0, 1).unwrap();
```

### 梯度计算与优化

```rust
use std::collections::HashMap;

// 假设从下游任务（如 GNN 分类）获得梯度
// 这些梯度表示：如果某条边存在，损失会增加/减少多少
let mut loss_gradients = HashMap::new();
loss_gradients.insert((0, 1), 0.5);   // 正梯度：鼓励删除
loss_gradients.insert((1, 2), -0.8);  // 负梯度：鼓励保留

// 计算结构梯度
let structure_gradients = graph.compute_structure_gradients(&loss_gradients);

// 基于梯度更新结构
graph.update_structure(&structure_gradients);
```

### 完整的优化循环

```rust
// 配置优化器
let config = GradientConfig::new(
    1.0,    // 初始温度
    true,   // 使用 STE
    0.05,   // 边学习率
    0.01,   // 节点学习率
)
.with_sparsity(0.001)      // L1 稀疏正则化
.with_smoothness(0.0001);  // 平滑正则化

let mut graph = DifferentiableGraph::with_config(5, config);

// 初始化边
graph.add_learnable_edge(0, 1, 0.5);
graph.add_learnable_edge(1, 2, 0.5);
// ...

// 优化循环
for step in 0..100 {
    // 1. 从下游任务获取梯度
    let loss_gradients = compute_loss_gradients(&graph);
    
    // 2. 一步优化（离散化 → 计算梯度 → 更新 → 退火）
    graph.optimization_step(loss_gradients);
    
    // 3. 定期检查
    if step % 10 == 0 {
        println!("Step {}: T={:.4}", step, graph.temperature());
    }
}
```

### 使用编辑策略

```rust
use god_gragh::tensor::differentiable::{
    GraphTransformer, ThresholdEditPolicy
};

// 定义编辑策略
let policy = Box::new(ThresholdEditPolicy {
    add_threshold: 0.1,      // 梯度>0.1 时添加边
    remove_threshold: -0.1,  // 梯度<-0.1 时删除边
    min_prob: 0.01,
    max_prob: 0.99,
});

let mut transformer = GraphTransformer::new(policy);

// 记录梯度
transformer.record_gradients(&gradients);

// 执行结构变换
let edits = transformer.transform(&mut graph);

// 查看编辑历史
for edit in &edits {
    println!("编辑操作：{:?}, 梯度：{:.4}", edit.operation, edit.gradient);
}
```

### Gumbel-Softmax 采样

```rust
use god_gragh::tensor::differentiable::GumbelSoftmaxSampler;

let sampler = GumbelSoftmaxSampler::new(1.0);
let logits = vec![0.5, 1.0, -0.5, 2.0];

// 软采样（可微，用于训练）
let soft = sampler.sample_soft(&logits);

// 硬采样（不可微，用于推理）
let hard = sampler.sample_hard(&logits);

// STE 采样（前向硬，反向软）
let (hard_ste, soft_ste) = sampler.sample_ste(&logits);
```

## 配置选项

### GradientConfig

```rust
pub struct GradientConfig {
    pub temperature: f64,           // 温度参数（默认 1.0）
    pub use_ste: bool,              // 是否使用 STE（默认 true）
    pub edge_learning_rate: f64,    // 边学习率（默认 0.01）
    pub node_learning_rate: f64,    // 节点学习率（默认 0.001）
    pub sparsity_weight: f64,       // L1 稀疏权重（默认 0.0）
    pub smoothness_weight: f64,     // 平滑权重（默认 0.0）
}
```

### 正则化

**稀疏正则化 (L1)**:
```rust
let config = GradientConfig::default().with_sparsity(0.01);
```
鼓励边概率趋向 0，产生更稀疏的图结构。

**平滑正则化**:
```rust
let config = GradientConfig::default().with_smoothness(0.001);
```
鼓励相连的边有相似的概率。

### 温度退火

```rust
let mut graph = DifferentiableGraph::with_config(5, config)
    .with_temperature_annealing(100);  // 100 步退火

// 每步优化自动退火
for _ in 0..100 {
    graph.optimization_step(gradients);
    // 温度从 1.0 指数衰减到 ~0.1
}
```

## 数学原理

### 梯度计算

对于边 (i, j)，损失 L 对 logits 的梯度：

```
∂L/∂logits_ij = ∂L/∂A_ij · ∂A_ij/∂logits_ij
              = ∂L/∂A_ij · A_ij·(1-A_ij)/τ
```

其中：
- `∂L/∂A_ij`: 从下游任务传来的梯度
- `A_ij·(1-A_ij)/τ`: sigmoid 的导数

### 稀疏正则化梯度

```
L_sparse = λ · ||logits||_1
∂L_sparse/∂logits = λ · sign(logits)
```

### 平滑正则化梯度

```
L_smooth = μ · Σ_{(i,j),(i,k)∈E} (A_ij - A_ik)²
∂L_smooth/∂A_ij = 2μ · Σ_k (A_ij - A_ik)
```

## 应用场景

### 1. 图结构学习

从数据中学习最优的图结构：
```rust
// 初始图可能不完整或有噪声
let mut graph = create_initial_graph(data);

// 通过优化任务损失来学习结构
for epoch in 0..num_epochs {
    // 前向：GNN 在当前结构上训练
    let predictions = gnn.forward(&graph, features);
    let loss = compute_loss(predictions, labels);
    
    // 反向：计算结构梯度
    let structure_gradients = compute_structure_gradients(loss);
    
    // 更新图结构
    graph.optimization_step(structure_gradients);
}
```

### 2. 图优化

优化图以满足特定目标：
```rust
// 目标：学习一个有利于分类的图结构
fn compute_loss_gradients(graph: &DifferentiableGraph) -> HashMap<(usize, usize), f64> {
    let mut gradients = HashMap::new();
    
    for ((src, dst), edge) in graph.get_learnable_edges() {
        // 模拟梯度：如果边连接同类节点，给负梯度（鼓励）
        // 如果连接异类节点，给正梯度（抑制）
        let same_class = check_same_class(*src, *dst);
        let grad = if same_class { -0.5 } else { 0.5 };
        gradients.insert((*src, *dst), grad);
    }
    
    gradients
}
```

### 3. 图生成

生成符合特定分布的图：
```rust
// 学习边的概率分布
let mut graph = DifferentiableGraph::new(num_nodes);

// 通过对抗训练或最大似然学习
for step in 0..num_steps {
    // 采样当前结构
    graph.discretize();
    
    // 计算生成分布与目标分布的差异
    let gradients = compute_distribution_gradients(&graph);
    
    // 更新概率参数
    graph.optimization_step(gradients);
}
```

## 运行示例

```bash
# 运行完整示例
cargo run --example differentiable_graph --features "tensor,tensor-sparse,rand"

# 运行测试
cargo test --features "tensor,tensor-sparse,rand" --lib tensor::differentiable
```

## 参考资料

- **IDGL**: Iterative Deep Graph Learning (NeurIPS 2020)
- **GDC**: Graph Diffusion Convolution (NeurIPS 2019)
- **ProGNN**: Graph Structure Learning for Robust GNNs (KDD 2020)
- **Gumbel-Softmax**: Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)

## 注意事项

1. **温度选择**: 较高的温度（τ>1）使分布更平滑，较低的温度（τ<0.5）使分布更离散
2. **学习率调优**: 边学习率通常比节点学习率大一个数量级
3. **正则化平衡**: 稀疏和平滑正则化需要根据具体任务调整权重
4. **梯度截断**: 对于大规模图，建议对梯度进行截断以防止数值不稳定
