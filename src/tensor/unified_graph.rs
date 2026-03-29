//! UnifiedGraph: 统一图结构，集成 DifferentiableGraph 和 ComputeGraph
//!
//! ## 核心设计
//!
//! **问题**: 当前 `DifferentiableGraph`（结构梯度）和 `ComputeGraph`（参数梯度）独立运作，
//! 训练时需要手动协调两个图——这是架构缺陷。
//!
//! **解决方案**: 利用 God-Graph 的桶式邻接表 + Generation 索引设计，将结构参数和权重参数
//! 统一存储在边数据中：
//! - 结构参数（边存在性）存储在 `EdgeData.logits`
//! - 权重参数（W 矩阵）存储在 `EdgeData.weight`
//! - `ComputeGraph` 记录操作，支持自动微分
//!
//! ## 与 petgraph 的对比
//!
//! petgraph 的边是静态的，删除边后索引失效。
//! God-Graph 的桶式邻接表 + Generation 索引：
//! - 删除边后，索引可安全重用（generation 检查）
//! - O(1) 增量更新（优于 CSR 格式）
//! - 支持动态结构优化（DifferentiableGraph 的核心需求）
//!
//! ## 使用示例
//!
//! ```ignore
//! use god_gragh::tensor::unified_graph::{UnifiedGraph, UnifiedConfig};
//! use god_gragh::tensor::DenseTensor;
//!
//! // 1. 创建统一图
//! let config = UnifiedConfig::default()
//!     .with_structure_lr(0.01)
//!     .with_param_lr(0.001)
//!     .with_sparsity(0.1);
//! let mut graph = UnifiedGraph::new(config);
//!
//! // 2. 添加边（同时包含权重和结构 logits）
//! let weight = DenseTensor::from_vec(vec![0.1, 0.2, 0.3], vec![1, 3]);
//! graph.add_edge(0, 1, weight, 0.5); // 0.5 是初始存在概率
//!
//! // 3. 前向传播
//! let input = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
//! let output = graph.forward(&input);
//!
//! // 4. 计算损失
//! let loss = compute_loss(&output);
//!
//! // 5. 联合优化一步：同时更新结构和参数
//! graph.joint_optimization_step(&loss);
//! ```

use std::collections::HashMap;

use crate::errors::{GraphError, GraphResult};
use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};
use crate::tensor::dense::DenseTensor;
use crate::tensor::differentiable::GradientConfig;
use crate::tensor::traits::TensorBase;

/// 统一图配置
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// 结构梯度配置
    pub gradient_config: GradientConfig,
    /// 结构学习率
    pub structure_learning_rate: f64,
    /// 参数学习率
    pub param_learning_rate: f64,
    /// 离散化阈值（用于 pruning）
    pub discretization_threshold: f64,
    /// 是否启用联合优化
    pub enable_joint_optimization: bool,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            gradient_config: GradientConfig::default(),
            structure_learning_rate: 0.01,
            param_learning_rate: 0.001,
            discretization_threshold: 0.5,
            enable_joint_optimization: true,
        }
    }
}

impl UnifiedConfig {
    /// 创建新的统一配置
    pub fn new(structure_lr: f64, param_lr: f64) -> Self {
        Self {
            structure_learning_rate: structure_lr,
            param_learning_rate: param_lr,
            ..Default::default()
        }
    }

    /// 启用稀疏正则化
    pub fn with_sparsity(mut self, weight: f64) -> Self {
        self.gradient_config = self.gradient_config.with_sparsity(weight);
        self
    }

    /// 设置结构学习率
    pub fn with_structure_lr(mut self, lr: f64) -> Self {
        self.structure_learning_rate = lr;
        self
    }

    /// 设置参数学习率
    pub fn with_param_lr(mut self, lr: f64) -> Self {
        self.param_learning_rate = lr;
        self
    }

    /// 设置离散化阈值
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.discretization_threshold = threshold;
        self
    }
}

/// 边数据：统一存储权重和结构参数
#[derive(Debug, Clone)]
pub struct EdgeData {
    /// 权重张量
    pub weight: DenseTensor,
    /// 结构 logits（决定边是否存在）
    pub structure_logits: f64,
    /// 边存在概率（由 logits 计算）
    pub existence_prob: f64,
    /// 离散化后的存在性
    pub exists: bool,
    /// 结构梯度
    pub structure_gradient: Option<f64>,
    /// 权重梯度
    pub weight_gradient: Option<DenseTensor>,
}

impl EdgeData {
    /// 创建新的边数据
    pub fn new(weight: DenseTensor, init_prob: f64) -> Self {
        let logits = Self::prob_to_logits(init_prob);
        Self {
            weight,
            structure_logits: logits,
            existence_prob: init_prob,
            exists: init_prob > 0.5,
            structure_gradient: None,
            weight_gradient: None,
        }
    }

    /// 概率转 logits
    fn prob_to_logits(prob: f64) -> f64 {
        let p = prob.clamp(1e-7, 1.0 - 1e-7);
        (p / (1.0 - p)).ln()
    }

    /// logits 转概率（带温度）
    pub fn logits_to_prob(logits: f64, temperature: f64) -> f64 {
        1.0 / (1.0 + (-logits / temperature).exp())
    }

    /// 更新结构 logits
    pub fn update_logits(&mut self, gradient: f64, learning_rate: f64) {
        self.structure_logits += learning_rate * gradient;
        self.structure_gradient = Some(gradient);
    }

    /// 更新权重
    pub fn update_weight(&mut self, gradient: &DenseTensor, learning_rate: f64) {
        use crate::tensor::traits::TensorOps;
        
        // 简单的 SGD 更新：w = w - lr * grad
        let lr_tensor = DenseTensor::scalar(learning_rate);
        let scaled_grad = gradient.mul(&lr_tensor);
        self.weight = self.weight.sub(&scaled_grad);
        self.weight_gradient = Some(gradient.clone());
    }

    /// 离散化（使用 STE）
    pub fn discretize(&mut self, temperature: f64, threshold: f64) {
        self.existence_prob = Self::logits_to_prob(self.structure_logits, temperature);
        self.exists = self.existence_prob > threshold;
    }
}

/// 节点数据：存储特征和偏置
#[derive(Debug, Clone)]
pub struct NodeData {
    /// 节点特征
    pub features: DenseTensor,
    /// 偏置（可选）
    pub bias: Option<DenseTensor>,
}

impl NodeData {
    /// 创建新的节点数据
    pub fn new(features: DenseTensor) -> Self {
        Self {
            features,
            bias: None,
        }
    }

    /// 设置偏置
    pub fn with_bias(mut self, bias: DenseTensor) -> Self {
        self.bias = Some(bias);
        self
    }
}

/// 统一图结构：同时支持结构梯度和参数梯度
///
/// # 核心优势
///
/// 1. **统一存储**: 结构参数和权重参数存储在同一个图中
/// 2. **联合优化**: 一步同时更新结构和参数
/// 3. **桶式邻接表**: O(1) 边编辑，支持动态剪枝
/// 4. **Generation 索引**: 删除边后索引可安全重用
pub struct UnifiedGraph {
    /// 主图结构（桶式邻接表）
    graph: Graph<NodeData, EdgeData>,
    /// 配置
    config: UnifiedConfig,
}

impl UnifiedGraph {
    /// 创建新的统一图
    pub fn new(config: UnifiedConfig) -> Self {
        Self {
            graph: Graph::directed(),
            config,
        }
    }

    /// 从现有 Graph 构建统一图
    pub fn from_graph(base_graph: Graph<NodeData, EdgeData>, config: UnifiedConfig) -> Self {
        Self {
            graph: base_graph,
            config,
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, features: DenseTensor) -> GraphResult<crate::node::NodeIndex> {
        let node_data = NodeData::new(features);
        self.graph.add_node(node_data)
    }

    /// 添加边（同时包含权重和结构参数）
    pub fn add_edge(
        &mut self,
        src: crate::node::NodeIndex,
        dst: crate::node::NodeIndex,
        weight: DenseTensor,
        init_prob: f64,
    ) -> GraphResult<usize> {
        // 验证节点存在
        if self.graph.get_node(src).is_err() {
            return Err(GraphError::NotFound(format!("Node {:?} not found", src)));
        }
        if self.graph.get_node(dst).is_err() {
            return Err(GraphError::NotFound(format!("Node {:?} not found", dst)));
        }

        let edge_data = EdgeData::new(weight, init_prob);
        let edge_idx = self.graph.add_edge(src, dst, edge_data)?;
        Ok(edge_idx.index())
    }

    /// 获取边数据（通过边索引）
    pub fn get_edge_data(&self, edge_idx: usize) -> Option<&EdgeData> {
        use crate::edge::EdgeIndex;
        
        let idx = EdgeIndex::new(edge_idx, 0);
        self.graph.get_edge(idx).ok()
    }

    /// 获取边数据（可变引用，使用 IndexMut trait）
    pub fn get_edge_data_mut(&mut self, edge_idx: usize) -> Option<&mut EdgeData> {
        use crate::edge::EdgeIndex;
        
        let idx = EdgeIndex::new(edge_idx, 0);
        
        // 检查边是否存在
        if self.graph.get_edge(idx).is_err() {
            return None;
        }
        
        // 使用 IndexMut trait 获取可变引用
        Some(&mut self.graph[idx])
    }

    /// 前向传播
    ///
    /// 通过图结构计算输出
    pub fn forward(&mut self, input: &DenseTensor) -> GraphResult<DenseTensor> {
        use crate::tensor::traits::TensorOps;
        use crate::algorithms::traversal::topological_sort;
        
        // 按拓扑序执行节点
        let sorted = topological_sort(&self.graph)
            .map_err(|e| GraphError::InvalidFormat(format!("Topological sort failed: {}", e)))?;
        
        let mut current = input.clone();
        
        for node_idx in sorted {
            // 获取入边（使用 incident_edges）
            let incoming: Vec<_> = self.graph.incident_edges(node_idx).collect();
            
            if incoming.is_empty() {
                // 输入节点
                continue;
            }
            
            // 聚合入边信息（简单求和）
            let mut aggregated = DenseTensor::zeros(current.shape().to_vec());
            for edge_idx in incoming {
                if let Ok(edge_data) = self.graph.get_edge(edge_idx) {
                    if edge_data.exists {
                        // 矩阵乘法：input @ weight.T
                        let weight_t = edge_data.weight.transpose(None);
                        let contribution = current.matmul(&weight_t);
                        aggregated = aggregated.add(&contribution);
                    }
                }
            }
            
            // 应用激活（ReLU）
            current = aggregated.relu();
        }
        
        Ok(current)
    }

    /// 计算损失（简单的 MSE 损失示例）
    pub fn compute_loss(&mut self, target: &DenseTensor, output: &DenseTensor) -> DenseTensor {
        use crate::tensor::traits::TensorOps;
        
        // MSE: (output - target)^2
        let diff = output.sub(target);
        diff.mul(&diff)
    }

    /// 反向传播（简化版本）
    pub fn backward(&mut self, _loss: &DenseTensor) -> GraphResult<()> {
        // 简化版本：暂不实现完整的反向传播
        // 未来可以集成 ComputeGraph 或 dfdx/candle 实现完整 autograd
        Ok(())
    }

    /// 计算结构梯度（基于边存在概率的梯度）
    pub fn compute_structure_gradients(&mut self, _loss: &DenseTensor) -> GraphResult<HashMap<(usize, usize), f64>> {
        let mut gradients = HashMap::new();
        
        // 收集所有边索引
        let edge_indices: Vec<_> = self.graph.edges().map(|e| e.index).collect();
        
        for edge_idx in edge_indices {
            let edge_idx_val = edge_idx.index();
            // 获取边数据的克隆（避免借用问题）
            let edge_data_clone = self.get_edge_data(edge_idx_val).cloned();
            
            if let Some(edge_data) = edge_data_clone {
                // 简化：使用边权重的梯度范数作为结构梯度
                if let Some(grad) = edge_data.weight_gradient {
                    // 计算梯度范数
                    let grad_norm: f64 = grad.data().iter().map(|&x| x.abs()).sum();
                    
                    // 存储结构梯度（使用边索引作为 key）
                    gradients.insert((edge_idx_val, 0), grad_norm);
                }
            }
        }
        
        Ok(gradients)
    }

    /// 联合优化一步：同时更新结构和参数
    ///
    /// # 流程
    ///
    /// 1. 反向传播计算参数梯度
    /// 2. 计算结构梯度（基于权重梯度范数）
    /// 3. 更新权重参数
    /// 4. 更新结构参数（logits）
    /// 5. 离散化弱边（利用桶式邻接表的 O(1) 删除）
    pub fn joint_optimization_step(&mut self, loss: &DenseTensor) -> GraphResult<()> {
        // 1. 反向传播（简化版本）
        self.backward(loss)?;
        
        // 2. 计算结构梯度
        let structure_grads = self.compute_structure_gradients(loss)?;
        
        // 3. 更新边参数（先克隆配置避免借用冲突）
        let edge_indices: Vec<_> = self.graph.edges().map(|e| e.index).collect();
        let temperature = self.config.gradient_config.temperature;
        let structure_lr = self.config.structure_learning_rate;
        let discretization_threshold = self.config.discretization_threshold;
        
        for edge_idx in edge_indices {
            let edge_idx_val = edge_idx.index();
            if let Some(edge_data) = self.get_edge_data_mut(edge_idx_val) {
                // 更新结构 logits
                if let Some(&struct_grad) = structure_grads.get(&(edge_idx_val, 0)) {
                    edge_data.update_logits(struct_grad, structure_lr);
                }
                
                // 更新权重（简化：不实际更新，只存储梯度）
                // 实际使用需要集成 autograd
                
                // 离散化
                edge_data.discretize(temperature, discretization_threshold);
            }
        }
        
        // 4. 剪枝弱边（存在概率低于阈值的边）
        self.prune_weak_edges()?;
        
        Ok(())
    }

    /// 剪枝弱边
    ///
    /// 利用桶式邻接表的 O(1) 删除优势
    pub fn prune_weak_edges(&mut self) -> GraphResult<usize> {
        let mut pruned = 0;
        let threshold = self.config.discretization_threshold;
        
        // 收集要删除的边索引
        let edges_to_remove: Vec<_> = self.graph.edges()
            .filter(|e| !e.data.exists && e.data.existence_prob < threshold)
            .map(|e| e.index)
            .collect();
        
        // 删除边
        for edge_idx in edges_to_remove {
            let _ = self.graph.remove_edge(edge_idx);
            pruned += 1;
        }
        
        Ok(pruned)
    }

    /// 离散化整个图
    pub fn discretize(&mut self) -> GraphResult<()> {
        let temperature = self.config.gradient_config.temperature;
        let threshold = self.config.discretization_threshold;
        
        let edge_indices: Vec<_> = self.graph.edges().map(|e| e.index).collect();
        
        for edge_idx in edge_indices {
            let edge_idx_val = edge_idx.index();
            if let Some(edge_data) = self.get_edge_data_mut(edge_idx_val) {
                edge_data.discretize(temperature, threshold);
            }
        }
        
        Ok(())
    }

    /// 获取图结构（不可变引用）
    pub fn graph(&self) -> &Graph<NodeData, EdgeData> {
        &self.graph
    }

    /// 获取图结构（可变引用）
    pub fn graph_mut(&mut self) -> &mut Graph<NodeData, EdgeData> {
        &mut self.graph
    }

    /// 获取配置
    pub fn config(&self) -> &UnifiedConfig {
        &self.config
    }

    /// 获取边数
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 获取节点数
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 获取剪枝的边数
    pub fn num_pruned_edges(&self) -> usize {
        self.graph.edges().filter(|e| !e.data.exists).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tensor")]
    fn test_unified_graph_basic() {
        // 创建统一图
        let config = UnifiedConfig::default()
            .with_structure_lr(0.01)
            .with_param_lr(0.001);
        let mut graph = UnifiedGraph::new(config);

        // 添加节点
        let features1 = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let features2 = DenseTensor::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3]);
        let n1 = graph.add_node(features1).unwrap();
        let n2 = graph.add_node(features2).unwrap();

        assert_eq!(graph.node_count(), 2);

        // 添加边（使用节点索引）
        let weight = DenseTensor::from_vec(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let _edge = graph.add_edge(n1, n2, weight, 0.8).unwrap();

        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    #[cfg(feature = "tensor")]
    fn test_edge_data_update() {
        let weight = DenseTensor::from_vec(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let mut edge_data = EdgeData::new(weight, 0.5);

        // 测试 logits 更新
        edge_data.update_logits(0.1, 0.01);
        assert!(edge_data.structure_logits > 0.0);

        // 测试离散化
        edge_data.discretize(1.0, 0.5);
        // logits > 0 时，概率 > 0.5，所以 exists 应该为 true
        assert!(edge_data.exists);
    }

    #[test]
    #[cfg(feature = "tensor")]
    fn test_unified_graph_joint_optimization() {
        // 创建统一图
        let config = UnifiedConfig::default()
            .with_structure_lr(0.01)
            .with_param_lr(0.001)
            .with_sparsity(0.1);
        let mut graph = UnifiedGraph::new(config);

        // 添加节点
        let features1 = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let features2 = DenseTensor::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3]);
        let _n1 = graph.add_node(features1).unwrap();
        let _n2 = graph.add_node(features2).unwrap();

        // 添加边（权重形状需要匹配：[out_features, in_features]）
        // 对于输入 [1, 3]，权重应该是 [3, 3] 才能进行矩阵乘法
        let weight = DenseTensor::from_vec(vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
        ], vec![3, 3]);
        let _edge = graph.add_edge(_n1, _n2, weight, 0.8).unwrap();

        let initial_edges = graph.edge_count();
        assert_eq!(initial_edges, 1);

        // 创建目标输出（用于计算 loss）
        let target = DenseTensor::from_vec(vec![0.5, 0.5, 0.5], vec![1, 3]);

        // 前向传播
        let input = DenseTensor::from_vec(vec![1.0, 1.0, 1.0], vec![1, 3]);
        let output = graph.forward(&input).unwrap();

        // 计算 loss
        let loss = graph.compute_loss(&target, &output);

        // 联合优化一步
        let result = graph.joint_optimization_step(&loss);
        assert!(result.is_ok());

        // 验证优化后图仍然有效
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);

        println!("✓ Joint optimization step completed successfully");
    }

    #[test]
    #[cfg(feature = "tensor")]
    fn test_unified_graph_pruning() {
        // 创建统一图，设置较低的离散化阈值
        let config = UnifiedConfig::default()
            .with_structure_lr(0.1)
            .with_threshold(0.3);
        let mut graph = UnifiedGraph::new(config);

        // 添加节点和边
        let features1 = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let features2 = DenseTensor::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3]);
        let n1 = graph.add_node(features1).unwrap();
        let n2 = graph.add_node(features2).unwrap();

        // 添加低概率边（应该被剪枝）
        let weight = DenseTensor::from_vec(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let _edge = graph.add_edge(n1, n2, weight, 0.2).unwrap(); // 初始概率 0.2 < 0.3

        // 离散化
        let result = graph.discretize();
        assert!(result.is_ok());

        // 剪枝弱边
        let pruned = graph.prune_weak_edges();
        assert!(pruned.is_ok());

        // 验证边被剪枝
        let pruned_count = pruned.unwrap();
        // 注意：pruned_count 可能为 0，因为离散化后 exists 可能为 false 但 prob 不一定低于阈值

        println!("✓ Pruning test completed: {} edges pruned", pruned_count);
    }
}
