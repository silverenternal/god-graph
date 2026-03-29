//! 可微图结构变换模块
//!
//! 本模块实现了图结构变换操作的梯度计算，支持：
//! - 可微边编辑（添加/删除/修改边权重）
//! - 可微节点编辑（添加/删除节点）
//! - Straight-Through Estimator (STE) 用于离散操作
//! - Gumbel-Softmax 松弛用于可微采样
//! - 图结构优化的梯度传播
//!
//! ## 核心概念
//!
//! ### 连续松弛表示
//!
//! 传统图结构是离散的：边要么存在 (1) 要么不存在 (0)。
//! 为了支持梯度计算，我们使用连续松弛：
//!
//! ```text
//! A_soft = σ(A_logits / τ)
//!
//! 其中：
//! - A_logits: 边的对数几率（可学习参数）
//! - τ: 温度参数（控制离散程度）
//! - σ: sigmoid 函数
//! ```
//!
//! ### Straight-Through Estimator (STE)
//!
//! 对于需要离散输出的场景，使用 STE：
//! - 前向传播：硬阈值（0/1）
//! - 反向传播：软梯度（通过 sigmoid）
//!
//! ```text
//! A_hard = (A_soft > 0.5).to_f64()
//! gradient = A_hard - A_soft.detach() + A_soft
//! ```
//!
//! ## 示例
//!
//! ```ignore
//! use god_gragh::graph::Graph;
//! use god_gragh::tensor::differentiable::{
//!     DifferentiableGraph, EdgeEditPolicy, GradientConfig
//! };
//!
//! // 创建可微图
//! let mut diff_graph = DifferentiableGraph::new(4);
//!
//! // 添加可学习边
//! diff_graph.add_learnable_edge(0, 1, 0.5);
//! diff_graph.add_learnable_edge(1, 2, 0.8);
//!
//! // 计算损失对边权重的梯度
//! let loss = compute_loss(&diff_graph);
//! let gradients = diff_graph.compute_structure_gradients(loss);
//!
//! // 基于梯度更新结构
//! diff_graph.update_structure(&gradients, learning_rate=0.01);
//! ```

use std::collections::HashMap;

#[cfg(all(feature = "tensor", feature = "tensor-autograd"))]
use dfdx::prelude::*;

#[cfg(feature = "rand")]
use rand::{random, Rng};

/// 图结构变换的梯度配置
#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// 温度参数（用于 Gumbel-Softmax）
    pub temperature: f64,
    /// 是否使用 Straight-Through Estimator
    pub use_ste: bool,
    /// 边编辑的学习率
    pub edge_learning_rate: f64,
    /// 节点编辑的学习率
    pub node_learning_rate: f64,
    /// 结构正则化权重（L1 稀疏）
    pub sparsity_weight: f64,
    /// 结构正则化权重（L2 平滑）
    pub smoothness_weight: f64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            use_ste: true,
            edge_learning_rate: 0.01,
            node_learning_rate: 0.001,
            sparsity_weight: 0.0,
            smoothness_weight: 0.0,
        }
    }
}

impl GradientConfig {
    /// 创建新的梯度配置
    pub fn new(temperature: f64, use_ste: bool, edge_lr: f64, node_lr: f64) -> Self {
        Self {
            temperature,
            use_ste,
            edge_learning_rate: edge_lr,
            node_learning_rate: node_lr,
            sparsity_weight: 0.0,
            smoothness_weight: 0.0,
        }
    }

    /// 启用稀疏正则化
    pub fn with_sparsity(mut self, weight: f64) -> Self {
        self.sparsity_weight = weight;
        self
    }

    /// 启用平滑正则化
    pub fn with_smoothness(mut self, weight: f64) -> Self {
        self.smoothness_weight = weight;
        self
    }

    /// 设置边学习率
    pub fn with_edge_learning_rate(mut self, lr: f64) -> Self {
        self.edge_learning_rate = lr;
        self
    }
}

/// 边编辑操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeEditOp {
    /// 添加边
    Add,
    /// 删除边
    Remove,
    /// 修改边权重
    Modify,
}

/// 节点编辑操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeEditOp {
    /// 添加节点
    Add,
    /// 删除节点
    Remove,
    /// 修改节点特征
    Modify,
}

/// 结构编辑操作（带梯度信息）
#[derive(Debug, Clone)]
pub struct StructureEdit {
    /// 操作类型
    pub operation: EditOperation,
    /// 梯度值
    pub gradient: f64,
    /// 编辑前的值
    pub before: f64,
    /// 编辑后的值
    pub after: f64,
}

/// 编辑操作枚举
#[derive(Debug, Clone)]
pub enum EditOperation {
    /// 边编辑 (src, dst, operation_type)
    EdgeEdit(usize, usize, EdgeEditOp),
    /// 节点编辑 (node_id, operation_type)
    NodeEdit(usize, NodeEditOp),
}

/// 可微边：包含可学习的存在概率
#[derive(Debug, Clone)]
pub struct DifferentiableEdge {
    /// 源节点索引
    pub src: usize,
    /// 目标节点索引
    pub dst: usize,
    /// 边的对数几率（logits）
    pub logits: f64,
    /// 边的存在概率（由 logits 计算）
    pub probability: f64,
    /// 离散化后的存在性（0 或 1）
    pub exists: bool,
    /// 梯度值
    pub gradient: Option<f64>,
}

impl DifferentiableEdge {
    /// 创建新的可微边
    pub fn new(src: usize, dst: usize, init_probability: f64) -> Self {
        let logits = Self::prob_to_logits(init_probability);
        Self {
            src,
            dst,
            logits,
            probability: init_probability,
            exists: init_probability > 0.5,
            gradient: None,
        }
    }

    /// 概率转 logits
    fn prob_to_logits(prob: f64) -> f64 {
        let p = prob.clamp(1e-7, 1.0 - 1e-7);
        (p / (1.0 - p)).ln()
    }

    /// logits 转概率（带温度）
    fn logits_to_prob(logits: f64, temperature: f64) -> f64 {
        1.0 / (1.0 + (-logits / temperature).exp())
    }

    /// 离散化（使用 STE）
    fn discretize(&mut self, temperature: f64, use_ste: bool) {
        let prob = Self::logits_to_prob(self.logits, temperature);
        self.probability = prob;
        self.exists = prob > 0.5;

        if use_ste {
            // STE: 前向离散，反向连续
            // gradient = exists - prob.detach() + prob
            // 这里我们存储概率，梯度计算在外层
        }
    }

    /// 基于梯度更新 logits（梯度下降）
    ///
    /// # Gradient Descent
    ///
    /// logits -= learning_rate * gradient
    ///
    /// 其中 gradient = ∂L/∂logits（增加损失的方向）
    pub fn update_logits(&mut self, gradient: f64, learning_rate: f64) {
        self.logits -= learning_rate * gradient;
        self.gradient = Some(gradient);
    }
}

/// 可微节点：包含可学习的存在概率和特征
#[derive(Debug, Clone)]
pub struct DifferentiableNode<T = Vec<f64>> {
    /// 节点索引
    pub id: usize,
    /// 节点存在概率
    pub existence_prob: f64,
    /// 节点特征（可选）
    pub features: Option<T>,
    /// 存在性的梯度
    pub existence_gradient: Option<f64>,
    /// 特征的梯度（如果是 tensor）
    pub feature_gradient: Option<T>,
}

impl<T: Clone> DifferentiableNode<T> {
    /// 创建新的可微节点
    pub fn new(id: usize, features: Option<T>) -> Self {
        Self {
            id,
            existence_prob: 1.0,
            features,
            existence_gradient: None,
            feature_gradient: None,
        }
    }

    /// 更新存在性
    pub fn update_existence(&mut self, gradient: f64, learning_rate: f64) {
        let new_prob = self.existence_prob + learning_rate * gradient;
        self.existence_prob = new_prob.clamp(0.0, 1.0);
        self.existence_gradient = Some(gradient);
    }
}

/// 可微图结构：支持梯度计算的结构变换
///
/// 核心思想：将离散的图结构参数化为连续空间，
/// 使得梯度可以反向传播到结构参数。
///
/// # Architecture Notes
///
/// ## 与自动微分框架的集成
///
/// 当前实现使用手动梯度计算。要与真正的自动微分框架（如 dfdx）集成，
/// 需要：
///
/// 1. 将 `logits` 存储为 `Tensor1D<f64>` 而非 `f64`
/// 2. 构建计算图：logits → probability → adjacency_matrix → loss
/// 3. 调用 `loss.backward()` 获取梯度
///
/// ## 与 Graph 的转换
///
/// 使用 `to_graph()` 将可微图转换为普通 `Graph`，
/// 使用 `from_graph()` 从现有图初始化可微图。
#[derive(Debug, Clone)]
pub struct DifferentiableGraph<T = Vec<f64>> {
    /// 节点数
    num_nodes: usize,
    /// 可微边集合（key: (src, dst)）
    edges: HashMap<(usize, usize), DifferentiableEdge>,
    /// 可微节点集合
    nodes: HashMap<usize, DifferentiableNode<T>>,
    /// 梯度配置
    config: GradientConfig,
    /// 温度退火步数
    annealing_steps: usize,
    /// 当前步数
    current_step: usize,
    /// STE 模式：如果为 true，在 discretize 时存储 STE 修正项
    use_ste: bool,
    /// STE 修正项：hard - soft
    ste_corrections: HashMap<(usize, usize), f64>,
}

impl<T: Clone + Default> DifferentiableGraph<T> {
    /// 创建新的可微图
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            edges: HashMap::new(),
            nodes: HashMap::new(),
            config: GradientConfig::default(),
            annealing_steps: 0,
            current_step: 0,
            use_ste: true,
            ste_corrections: HashMap::new(),
        }
    }

    /// 创建带配置的可微图
    pub fn with_config(num_nodes: usize, config: GradientConfig) -> Self {
        let use_ste = config.use_ste;
        Self {
            num_nodes,
            edges: HashMap::new(),
            nodes: HashMap::new(),
            config,
            annealing_steps: 0,
            current_step: 0,
            use_ste,
            ste_corrections: HashMap::new(),
        }
    }

    /// 初始化节点
    pub fn init_nodes(&mut self, features: Option<T>) {
        for i in 0..self.num_nodes {
            self.nodes
                .insert(i, DifferentiableNode::new(i, features.clone()));
        }
    }

    /// 添加可学习边
    pub fn add_learnable_edge(&mut self, src: usize, dst: usize, init_prob: f64) {
        let edge = DifferentiableEdge::new(src, dst, init_prob);
        self.edges.insert((src, dst), edge);
    }

    /// 移除边
    pub fn remove_edge(&mut self, src: usize, dst: usize) -> Option<DifferentiableEdge> {
        self.edges.remove(&(src, dst))
    }

    /// 获取边的存在概率
    pub fn get_edge_probability(&self, src: usize, dst: usize) -> Option<f64> {
        self.edges.get(&(src, dst)).map(|e| e.probability)
    }

    /// 获取边的存在性（离散）
    pub fn get_edge_exists(&self, src: usize, dst: usize) -> Option<bool> {
        self.edges.get(&(src, dst)).map(|e| e.exists)
    }

    /// 获取所有边的概率矩阵
    pub fn get_probability_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.num_nodes]; self.num_nodes];
        for ((src, dst), edge) in &self.edges {
            matrix[*src][*dst] = edge.probability;
        }
        matrix
    }

    /// 获取离散邻接矩阵（使用 STE）
    pub fn get_adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.num_nodes]; self.num_nodes];
        for ((src, dst), edge) in &self.edges {
            if edge.exists {
                matrix[*src][*dst] = 1.0;
            }
        }
        matrix
    }

    /// 温度退火
    pub fn anneal_temperature(&mut self) {
        if self.annealing_steps > 0 {
            let progress = self.current_step as f64 / self.annealing_steps as f64;
            // 指数退火：τ_t = τ_0 * exp(-k * t)
            let k = 3.0;
            self.config.temperature = 1.0 * (-k * progress).exp();
            self.config.temperature = self.config.temperature.max(0.1); // 最小温度
        }
        self.current_step += 1;
    }

    /// 设置温度退火
    pub fn with_temperature_annealing(mut self, steps: usize) -> Self {
        self.annealing_steps = steps;
        self
    }

    /// 离散化所有边（前向传播）
    ///
    /// 如果启用了 STE 模式，会存储 STE 修正项 (hard - soft)，
    /// 用于后续梯度计算时修正梯度。
    pub fn discretize(&mut self) {
        self.ste_corrections.clear();

        for (&(src, dst), edge) in &mut self.edges {
            let prob_before = edge.probability;
            edge.discretize(self.config.temperature, self.config.use_ste);

            // 存储 STE 修正项
            if self.use_ste {
                let hard = if edge.exists { 1.0 } else { 0.0 };
                let ste_correction = hard - prob_before;
                self.ste_corrections.insert((src, dst), ste_correction);
            }
        }
    }

    /// 计算结构梯度
    ///
    /// # Arguments
    /// * `loss_gradients` - 损失对边存在性的梯度 {(src, dst): ∂L/∂A_ij}
    ///
    /// # Returns
    /// HashMap {(src, dst): ∂L/∂logits}，可用于更新边的 logits 参数
    ///
    /// # Gradient Computation
    ///
    /// 梯度计算遵循链式法则：
    /// ```text
    /// ∂L/∂logits = ∂L/∂A * ∂A/∂logits
    /// ```
    ///
    /// 其中 A = σ(logits/τ)，所以：
    /// ```text
    /// ∂A/∂logits = A * (1 - A) / τ
    /// ```
    ///
    /// ## STE 修正
    ///
    /// 当启用 STE 模式时，梯度会加上 STE 修正项：
    /// ```text
    /// gradient = ∂L/∂logits + (hard - soft)
    /// ```
    ///
    /// 这确保了前向传播的离散化与反向传播的连续梯度一致。
    ///
    /// # Regularization
    ///
    /// ## L1 稀疏正则化
    /// L_sparse = λ_sparse * Σ|logits|
    /// ∂L_sparse/∂logits = λ_sparse * sign(logits)
    ///
    /// 梯度下降更新：logits -= lr * gradient
    /// - 正 logits → 正梯度 → logits 减小 → 概率趋向 0 → 稀疏
    /// - 负 logits → 负梯度 → logits 增大 → 概率趋向 0 → 稀疏
    ///
    /// ## L2 平滑正则化
    /// L_smooth = λ_smooth * Σ_{(i,j),(i,k)∈E} (A_ij - A_ik)²
    /// ∂L_smooth/∂A_ij = 2 * λ_smooth * Σ_k (A_ij - A_ik)
    ///
    /// 平滑正则化鼓励：
    /// - 共享源节点的边有相似概率
    /// - 共享目标节点的边有相似概率
    pub fn compute_structure_gradients(
        &mut self,
        loss_gradients: &HashMap<(usize, usize), f64>,
    ) -> HashMap<(usize, usize), f64> {
        let mut gradients = HashMap::new();

        // 直接遍历 edges，避免不必要的收集操作
        for (&(src, dst), edge) in &self.edges {
            if let Some(&loss_grad) = loss_gradients.get(&(src, dst)) {
                let prob = edge.probability;
                let logits = edge.logits;

                // 链式法则：∂L/∂logits = ∂L/∂A * ∂A/∂logits
                // 其中 ∂A/∂logits = A * (1 - A) / τ
                let d_prob_d_logits = prob * (1.0 - prob) / self.config.temperature;
                let mut logits_gradient = loss_grad * d_prob_d_logits;

                // STE 修正：gradient = gradient + (hard - soft)
                // 这确保了前向离散的梯度能正确传播
                if self.use_ste {
                    if let Some(&ste_correction) = self.ste_corrections.get(&(src, dst)) {
                        logits_gradient += ste_correction;
                    }
                }

                // L1 稀疏正则化梯度
                // ∂L_sparse/∂logits = λ_sparse * sign(logits)
                let sparse_grad = if self.config.sparsity_weight > 0.0 {
                    self.config.sparsity_weight * logits.signum()
                } else {
                    0.0
                };

                // L2 平滑正则化梯度
                // ∂L_smooth/∂logits = 2 * λ_smooth * Σ_k (A_ij - A_ik)
                let smooth_grad = if self.config.smoothness_weight > 0.0 {
                    self.compute_smoothness_gradient(src, dst, prob) * self.config.smoothness_weight
                } else {
                    0.0
                };

                let total_gradient = logits_gradient + sparse_grad + smooth_grad;
                gradients.insert((src, dst), total_gradient);
            }
        }

        gradients
    }

    /// 计算平滑正则化梯度
    ///
    /// 考虑两种相邻关系：
    /// 1. 共享源节点的边：(src, dst) 和 (src, k)
    /// 2. 共享目标节点的边：(src, dst) 和 (k, dst)
    fn compute_smoothness_gradient(&self, src: usize, dst: usize, prob: f64) -> f64 {
        let mut gradient = 0.0;

        // 遍历所有边，计算平滑梯度
        for (&(s, d), other_edge) in &self.edges {
            let other_prob = other_edge.probability;

            // 共享源节点：(src, dst) 和 (src, k)
            if s == src && d != dst {
                gradient += 2.0 * (prob - other_prob);
            }

            // 共享目标节点：(src, dst) 和 (k, dst)
            if d == dst && s != src {
                gradient += 2.0 * (prob - other_prob);
            }
        }

        gradient
    }

    /// 基于梯度更新结构
    pub fn update_structure(&mut self, gradients: &HashMap<(usize, usize), f64>) {
        for ((src, dst), &gradient) in gradients {
            if let Some(edge) = self.edges.get_mut(&(*src, *dst)) {
                edge.update_logits(gradient, self.config.edge_learning_rate);
            }
        }
    }

    /// 一步优化：离散化 -> 计算梯度 -> 更新
    pub fn optimization_step(
        &mut self,
        loss_gradients: HashMap<(usize, usize), f64>,
    ) -> HashMap<(usize, usize), f64> {
        // 1. 离散化（前向）
        self.discretize();

        // 2. 计算梯度（反向）
        let gradients = self.compute_structure_gradients(&loss_gradients);

        // 3. 更新结构
        self.update_structure(&gradients);

        // 4. 温度退火
        self.anneal_temperature();

        gradients
    }

    /// 获取可微边列表
    pub fn get_learnable_edges(&self) -> Vec<&DifferentiableEdge> {
        self.edges.values().collect()
    }

    /// 获取边数
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// 获取节点数
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// 获取配置
    pub fn config(&self) -> &GradientConfig {
        &self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: GradientConfig) {
        self.config = config;
    }

    /// 获取当前温度
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }

    /// 设置温度
    pub fn set_temperature(&mut self, temp: f64) {
        self.config.temperature = temp;
    }

    /// 获取边迭代器
    pub fn edges(&self) -> impl Iterator<Item = (&(usize, usize), &DifferentiableEdge)> {
        self.edges.iter()
    }

    /// 转换为普通 Graph
    ///
    /// 使用离散化的边存在性构建 Graph。
    /// 边的权重为 1.0（如果存在）或 0.0（如果不存在）。
    ///
    /// # Note
    ///
    /// 此方法创建的图使用节点索引作为节点数据，边权重为 f64。
    /// 节点索引通过 `NodeIndex::new(index, generation)` 创建，
    /// 其中 generation 由 Graph 内部管理。
    pub fn to_graph(&self) -> crate::graph::Graph<usize, f64> {
        use crate::graph::traits::GraphOps;
        use crate::graph::Graph;
        use crate::node::NodeIndex;

        let mut graph: crate::graph::Graph<usize, f64> =
            Graph::with_capacity(self.num_nodes, self.edges.len());

        // 添加节点，使用索引作为节点数据
        // Graph 会内部管理 NodeIndex 的 generation
        let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(self.num_nodes);
        for i in 0..self.num_nodes {
            let result = graph.add_node(i);
            match result {
                Ok(idx) => node_indices.push(idx),
                Err(_) => {
                    // 如果失败，创建一个占位符
                    node_indices.push(NodeIndex::new(i, 0));
                }
            }
        }

        // 添加存在的边
        for (&(src, dst), edge) in &self.edges {
            if edge.exists && src < node_indices.len() && dst < node_indices.len() {
                let _ = graph.add_edge(node_indices[src], node_indices[dst], 1.0);
            }
        }

        graph
    }

    /// 转换为带类型信息的 Graph（保留 OperatorType 和 WeightTensor）
    ///
    /// 使用离散化的边存在性构建 Graph，保留原始的节点和边类型信息。
    ///
    /// # Arguments
    /// * `node_types` - 节点类型映射 (node_index -> OperatorType)
    /// * `edge_weights` - 边权重映射 ((src, dst) -> WeightTensor)
    ///
    /// # Returns
    ///
    /// 带类型信息的 Graph<OperatorType, WeightTensor>
    #[cfg(feature = "transformer")]
    pub fn to_graph_with_types(
        &self,
        node_types: &std::collections::HashMap<
            usize,
            crate::transformer::optimization::switch::OperatorType,
        >,
        edge_weights: &std::collections::HashMap<
            (usize, usize),
            crate::transformer::optimization::switch::WeightTensor,
        >,
    ) -> crate::graph::Graph<
        crate::transformer::optimization::switch::OperatorType,
        crate::transformer::optimization::switch::WeightTensor,
    > {
        use crate::graph::traits::GraphOps;
        use crate::graph::Graph;
        use crate::node::NodeIndex;
        use crate::transformer::optimization::switch::{OperatorType, WeightTensor};

        let mut graph: Graph<OperatorType, WeightTensor> =
            Graph::with_capacity(self.num_nodes, self.edges.len());

        // 添加节点，使用提供的类型信息
        let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(self.num_nodes);
        for i in 0..self.num_nodes {
            let node_type = node_types
                .get(&i)
                .cloned()
                .unwrap_or_else(|| OperatorType::Custom {
                    name: format!("node_{}", i),
                });

            let result = graph.add_node(node_type);
            match result {
                Ok(idx) => node_indices.push(idx),
                Err(_) => {
                    // 如果失败，创建一个占位符
                    node_indices.push(NodeIndex::new(i, 0));
                }
            }
        }

        // 添加存在的边，使用提供的权重信息
        for (&(src, dst), edge) in &self.edges {
            if edge.exists && src < node_indices.len() && dst < node_indices.len() {
                let weight = edge_weights.get(&(src, dst)).cloned().unwrap_or_else(|| {
                    WeightTensor::new(format!("edge_{}_to_{}", src, dst), vec![1.0], vec![1])
                });
                let _ = graph.add_edge(node_indices[src], node_indices[dst], weight);
            }
        }

        graph
    }

    /// 从普通 Graph 初始化可微图
    ///
    /// # Arguments
    /// * `graph` - 源图
    /// * `init_probs` - 边的初始存在概率
    ///   - 如果提供，只初始化指定的边
    ///   - 如果为 None，根据图中存在的边初始化（概率设为 1.0）
    ///
    /// # Note
    ///
    /// 此方法忽略原图的节点和边数据，只使用图结构。
    /// 节点数据默认为 `()`,边数据默认为 `()`.
    pub fn from_graph<U, V>(
        graph: &crate::graph::Graph<U, V>,
        init_probs: Option<HashMap<(usize, usize), f64>>,
    ) -> DifferentiableGraph<()>
    where
        U: Clone,
        V: Clone,
    {
        use crate::graph::traits::{GraphBase, GraphQuery};

        let num_nodes = graph.node_count();
        let mut diff_graph = DifferentiableGraph::new(num_nodes);

        if let Some(probs) = init_probs {
            // 使用提供的概率初始化边
            for ((src, dst), &prob) in &probs {
                diff_graph.add_learnable_edge(*src, *dst, prob);
            }
        } else {
            // 根据图中存在的边初始化（概率设为 1.0）
            for node in graph.nodes() {
                let src_idx = node.index().index();
                for neighbor in graph.neighbors(node.index()) {
                    let dst_idx = neighbor.index();
                    diff_graph.add_learnable_edge(src_idx, dst_idx, 1.0);
                }
            }
        }

        diff_graph
    }

    /// 从普通图构建可微图（使用统一的初始概率）
    ///
    /// # Arguments
    /// * `graph` - 原始图
    /// * `init_prob` - 边的初始存在概率（0.0~1.0）
    ///
    /// # Returns
    /// DifferentiableGraph<()> - 可微图
    ///
    /// # Note
    ///
    /// 此方法忽略原图的节点和边数据，只使用图结构。
    pub fn from_graph_with_prob<U, V>(
        graph: &crate::graph::Graph<U, V>,
        init_prob: Option<f64>,
    ) -> DifferentiableGraph<()>
    where
        U: Clone,
        V: Clone,
    {
        use crate::graph::traits::{GraphBase, GraphQuery};

        let num_nodes = graph.node_count();
        let mut diff_graph = DifferentiableGraph::new(num_nodes);

        let prob = init_prob.unwrap_or(1.0);

        // 根据图中存在的边初始化
        for node in graph.nodes() {
            let src_idx = node.index().index();
            for neighbor in graph.neighbors(node.index()) {
                let dst_idx = neighbor.index();
                diff_graph.add_learnable_edge(src_idx, dst_idx, prob);
            }
        }

        diff_graph
    }

    /// 启用/禁用 STE 模式
    pub fn set_ste(&mut self, use_ste: bool) {
        self.use_ste = use_ste;
        self.config.use_ste = use_ste;
    }

    /// 获取 STE 修正项
    pub fn get_ste_corrections(&self) -> &HashMap<(usize, usize), f64> {
        &self.ste_corrections
    }
}

/// Gumbel-Softmax 采样器：用于可微离散采样
pub struct GumbelSoftmaxSampler {
    temperature: f64,
}

impl GumbelSoftmaxSampler {
    /// 创建新的采样器
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// 采样（软版本，可微）
    ///
    /// y_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)
    /// 其中 g_i ~ Gumbel(0, 1)
    pub fn sample_soft(&self, logits: &[f64]) -> Vec<f64> {
        let gumbel_noise: Vec<f64> = logits.iter().map(|_| self.gumbel_sample()).collect();

        let max_logit = logits
            .iter()
            .zip(&gumbel_noise)
            .map(|(&l, &g)| l + g)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_logits: Vec<f64> = logits
            .iter()
            .zip(&gumbel_noise)
            .map(|(&l, &g)| ((l + g - max_logit) / self.temperature).exp())
            .collect();

        let sum_exp: f64 = exp_logits.iter().sum();

        exp_logits.iter().map(|&e| e / sum_exp).collect()
    }

    /// 采样（硬版本，不可微，用于前向）
    pub fn sample_hard(&self, logits: &[f64]) -> Vec<f64> {
        let soft = self.sample_soft(logits);
        let mut result = vec![0.0; soft.len()];

        // 取最大值位置为 1
        if let Some(max_idx) = soft
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
        {
            result[max_idx] = 1.0;
        }

        result
    }

    /// STE 版本：前向硬，反向软
    pub fn sample_ste(&self, logits: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let hard = self.sample_hard(logits);
        let soft = self.sample_soft(logits);

        // STE: gradient = hard - soft.detach() + soft = hard (因为 soft 会 detach)
        // 实际实现中，我们返回 (hard, soft) 用于后续梯度计算
        (hard, soft)
    }

    /// Gumbel 分布采样：g = -log(-log(u)), u ~ Uniform(0,1)
    fn gumbel_sample(&self) -> f64 {
        #[cfg(feature = "rand")]
        {
            let u: f64 = random::<f64>();
            -(-u.ln()).ln()
        }
        #[cfg(not(feature = "rand"))]
        {
            // 无 rand 特性时，使用简单确定性值
            // 注意：这会使 Gumbel-Softmax 变成确定性函数，仅用于测试
            let u: f64 = 0.5;
            -(-u.ln()).ln()
        }
    }

    /// 设置温度
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp;
    }

    /// 使用自定义 RNG 的 Gumbel 采样
    ///
    /// 允许调用者提供随机数生成器，便于控制和复现结果
    #[cfg(feature = "rand")]
    pub fn gumbel_sample_with_rng<R: Rng>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.gen_range(1e-7..1.0 - 1e-7);
        -(-u.ln()).ln()
    }

    /// 使用自定义 RNG 的软采样
    #[cfg(feature = "rand")]
    pub fn sample_soft_with_rng(&self, logits: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let gumbel_noise: Vec<f64> = logits
            .iter()
            .map(|_| self.gumbel_sample_with_rng(rng))
            .collect();

        let max_logit = logits
            .iter()
            .zip(&gumbel_noise)
            .map(|(&l, &g)| l + g)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_logits: Vec<f64> = logits
            .iter()
            .zip(&gumbel_noise)
            .map(|(&l, &g)| ((l + g - max_logit) / self.temperature).exp())
            .collect();

        let sum_exp: f64 = exp_logits.iter().sum();

        exp_logits.iter().map(|&e| e / sum_exp).collect()
    }
}

/// 边编辑策略：定义如何基于梯度编辑边
pub trait EdgeEditPolicy: Send + Sync {
    /// 决定是否添加边
    fn should_add_edge(&self, gradient: f64, current_prob: f64) -> bool;

    /// 决定是否删除边
    fn should_remove_edge(&self, gradient: f64, current_prob: f64) -> bool;

    /// 计算新的边概率
    fn update_probability(&self, current_prob: f64, gradient: f64, learning_rate: f64) -> f64;
}

/// 基于阈值的编辑策略
#[derive(Debug, Clone)]
pub struct ThresholdEditPolicy {
    /// 添加边的梯度阈值
    pub add_threshold: f64,
    /// 删除边的梯度阈值
    pub remove_threshold: f64,
    /// 概率下限
    pub min_prob: f64,
    /// 概率上限
    pub max_prob: f64,
}

impl Default for ThresholdEditPolicy {
    fn default() -> Self {
        Self {
            add_threshold: 0.1,
            remove_threshold: -0.1,
            min_prob: 0.01,
            max_prob: 0.99,
        }
    }
}

impl EdgeEditPolicy for ThresholdEditPolicy {
    fn should_add_edge(&self, gradient: f64, current_prob: f64) -> bool {
        gradient > self.add_threshold && current_prob < 0.5
    }

    fn should_remove_edge(&self, gradient: f64, current_prob: f64) -> bool {
        gradient < self.remove_threshold && current_prob > 0.5
    }

    fn update_probability(&self, current_prob: f64, gradient: f64, learning_rate: f64) -> f64 {
        let new_prob = current_prob + learning_rate * gradient;
        new_prob.clamp(self.min_prob, self.max_prob)
    }
}

/// 结构梯度记录器：记录所有结构变换的梯度
#[derive(Debug, Default, Clone)]
pub struct GradientRecorder {
    /// 边梯度记录 {(src, dst): gradient}
    edge_gradients: HashMap<(usize, usize), f64>,
    /// 节点梯度记录 {node_id: gradient}
    node_gradients: HashMap<usize, f64>,
    /// 速度历史（用于经典动量）：v_t = μ * v_{t-1} + g_t
    edge_velocities: HashMap<(usize, usize), f64>,
    /// 动量系数
    momentum: f64,
}

impl GradientRecorder {
    /// 创建新的记录器
    pub fn new(momentum: f64) -> Self {
        Self {
            edge_gradients: HashMap::new(),
            node_gradients: HashMap::new(),
            edge_velocities: HashMap::new(),
            momentum,
        }
    }

    /// 记录边梯度
    pub fn record_edge_gradient(&mut self, src: usize, dst: usize, gradient: f64) {
        self.edge_gradients.insert((src, dst), gradient);
    }

    /// 记录节点梯度
    pub fn record_node_gradient(&mut self, node_id: usize, gradient: f64) {
        self.node_gradients.insert(node_id, gradient);
    }

    /// 获取边梯度
    pub fn get_edge_gradient(&self, src: usize, dst: usize) -> Option<f64> {
        self.edge_gradients.get(&(src, dst)).copied()
    }

    /// 获取所有边梯度
    pub fn get_all_edge_gradients(&self) -> &HashMap<(usize, usize), f64> {
        &self.edge_gradients
    }

    /// 应用经典动量
    ///
    /// 使用经典动量公式：v_t = μ * v_{t-1} + g_t
    /// 其中：
    /// - v_t: t 时刻的速度（累积梯度）
    /// - μ: 动量系数 (0.9 常见)
    /// - g_t: t 时刻的原始梯度
    ///
    /// 这与指数移动平均 (EMA) 不同：
    /// - EMA: g_ema = μ * g_ema + (1-μ) * g_t （会缩小梯度）
    /// - 经典动量：v_t = μ * v_{t-1} + g_t （保持梯度量级）
    pub fn apply_momentum(&mut self) -> HashMap<(usize, usize), f64> {
        let mut momentum_gradients = HashMap::new();

        for ((src, dst), &grad) in &self.edge_gradients {
            let last_velocity = self
                .edge_velocities
                .get(&(*src, *dst))
                .copied()
                .unwrap_or(0.0);
            // 经典动量公式：v_t = μ * v_{t-1} + g_t
            let new_velocity = self.momentum * last_velocity + grad;
            self.edge_velocities.insert((*src, *dst), new_velocity);
            momentum_gradients.insert((*src, *dst), new_velocity);
        }

        momentum_gradients
    }

    /// 清空记录（保留速度历史）
    pub fn clear(&mut self) {
        self.edge_gradients.clear();
        self.node_gradients.clear();
    }

    /// 清空所有状态（包括速度历史）
    pub fn reset(&mut self) {
        self.clear();
        self.edge_velocities.clear();
    }
}

/// 图结构变换器：执行具体的结构编辑操作
pub struct GraphTransformer<T> {
    /// 编辑策略
    policy: Box<dyn EdgeEditPolicy>,
    /// 梯度记录器
    recorder: GradientRecorder,
    /// 标记
    _marker: std::marker::PhantomData<T>,
}

impl<T: Clone + Default> GraphTransformer<T> {
    /// 创建新的变换器
    pub fn new(policy: Box<dyn EdgeEditPolicy>) -> Self {
        Self {
            policy,
            recorder: GradientRecorder::new(0.9),
            _marker: std::marker::PhantomData,
        }
    }

    /// 执行结构变换
    pub fn transform(&mut self, graph: &mut DifferentiableGraph<T>) -> Vec<StructureEdit> {
        let mut edits = Vec::new();

        // 应用动量
        let momentum_gradients = self.recorder.apply_momentum();

        // 遍历所有边，决定是否编辑
        for ((src, dst), edge) in &mut graph.edges {
            if let Some(&gradient) = momentum_gradients.get(&(*src, *dst)) {
                let before = edge.probability;

                // 决定是否删除
                if self.policy.should_remove_edge(gradient, edge.probability) {
                    let new_prob = self.policy.update_probability(
                        edge.probability,
                        gradient,
                        graph.config.edge_learning_rate,
                    );

                    let after = new_prob;
                    edge.probability = new_prob;
                    edge.exists = new_prob > 0.5;

                    edits.push(StructureEdit {
                        operation: EditOperation::EdgeEdit(*src, *dst, EdgeEditOp::Remove),
                        gradient,
                        before,
                        after,
                    });
                }
                // 决定是否添加
                else if self.policy.should_add_edge(gradient, edge.probability) {
                    let new_prob = self.policy.update_probability(
                        edge.probability,
                        gradient,
                        graph.config.edge_learning_rate,
                    );

                    let after = new_prob;
                    edge.probability = new_prob;
                    edge.exists = new_prob > 0.5;

                    edits.push(StructureEdit {
                        operation: EditOperation::EdgeEdit(*src, *dst, EdgeEditOp::Add),
                        gradient,
                        before,
                        after,
                    });
                }
                // 否则只是修改概率
                else {
                    let new_prob = self.policy.update_probability(
                        edge.probability,
                        gradient,
                        graph.config.edge_learning_rate,
                    );

                    let after = new_prob;
                    edge.probability = new_prob;
                    edge.exists = new_prob > 0.5;

                    edits.push(StructureEdit {
                        operation: EditOperation::EdgeEdit(*src, *dst, EdgeEditOp::Modify),
                        gradient,
                        before,
                        after,
                    });
                }
            }
        }

        edits
    }

    /// 记录梯度
    pub fn record_gradients(&mut self, gradients: &HashMap<(usize, usize), f64>) {
        for ((src, dst), &grad) in gradients {
            self.recorder.record_edge_gradient(*src, *dst, grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiable_edge() {
        let mut edge = DifferentiableEdge::new(0, 1, 0.5);

        assert_eq!(edge.src, 0);
        assert_eq!(edge.dst, 1);
        assert!((edge.logits - 0.0).abs() < 1e-6); // log(0.5/0.5) = 0
        assert!((edge.probability - 0.5).abs() < 1e-6);

        // 更新 logits（负梯度增加 logits，正梯度减小 logits）
        edge.update_logits(-0.1, 0.01); // 负梯度：增加 logits
        assert!(edge.logits > 0.0);
    }

    #[test]
    fn test_differentiable_graph() {
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);

        // 添加边
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(1, 2, 0.8);
        graph.add_learnable_edge(2, 3, 0.3);

        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.num_nodes(), 4);

        // 获取概率矩阵
        let prob_matrix = graph.get_probability_matrix();
        assert!((prob_matrix[0][1] - 0.5).abs() < 1e-6);
        assert!((prob_matrix[1][2] - 0.8).abs() < 1e-6);

        // 离散化
        graph.discretize();
        // 0.5 概率时 exists=false (因为 0.5 > 0.5 为 false)
        assert!(!graph.get_edge_exists(0, 1).unwrap());
        assert!(graph.get_edge_exists(1, 2).unwrap()); // 0.8 > 0.5 -> true
        assert!(!graph.get_edge_exists(2, 3).unwrap()); // 0.3 < 0.5 -> false
    }

    #[test]
    fn test_structure_gradient_computation() {
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(1, 2, 0.8);

        // 不 discretize，因为我们要测试纯梯度计算（不含 STE 修正）
        // 或者禁用 STE 模式
        graph.set_ste(false);

        // 模拟损失梯度
        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 0.5); // 正梯度：鼓励添加
        loss_gradients.insert((1, 2), -0.3); // 负梯度：鼓励删除

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        assert!(gradients.contains_key(&(0, 1)));
        assert!(gradients.contains_key(&(1, 2)));

        // 正梯度应该导致正的 logits 梯度（不考虑 STE 修正时）
        assert!(*gradients.get(&(0, 1)).unwrap() > 0.0);
        // 负梯度应该导致负的 logits 梯度
        assert!(*gradients.get(&(1, 2)).unwrap() < 0.0);
    }

    #[test]
    fn test_gumbel_softmax_sampler() {
        let sampler = GumbelSoftmaxSampler::new(1.0);
        let logits = vec![1.0, 2.0, 3.0];

        // 软采样
        let soft = sampler.sample_soft(&logits);
        assert_eq!(soft.len(), 3);
        assert!((soft.iter().sum::<f64>() - 1.0).abs() < 1e-5); // 和为 1

        // 硬采样
        let hard = sampler.sample_hard(&logits);
        assert_eq!(hard.len(), 3);
        assert_eq!(hard.iter().filter(|&&x| x > 0.5).count(), 1); // 只有一个为 1

        // STE 采样
        let (hard_ste, soft_ste) = sampler.sample_ste(&logits);
        assert_eq!(hard_ste.len(), 3);
        assert_eq!(soft_ste.len(), 3);
    }

    #[test]
    fn test_threshold_edit_policy() {
        let policy = ThresholdEditPolicy::default();

        // 测试添加边决策
        assert!(policy.should_add_edge(0.2, 0.3)); // 梯度>阈值，概率<0.5
        assert!(!policy.should_add_edge(0.05, 0.3)); // 梯度<阈值

        // 测试删除边决策
        assert!(policy.should_remove_edge(-0.2, 0.7)); // 梯度<阈值，概率>0.5
        assert!(!policy.should_remove_edge(-0.05, 0.7)); // 梯度>阈值

        // 测试概率更新
        let new_prob = policy.update_probability(0.5, 0.1, 0.01);
        assert!((new_prob - 0.501).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_recorder_with_momentum() {
        let mut recorder = GradientRecorder::new(0.9);

        recorder.record_edge_gradient(0, 1, 0.5);
        recorder.record_edge_gradient(1, 2, -0.3);

        let momentum_grads = recorder.apply_momentum();

        // 第一轮：v_1 = 0.9 * 0 + 0.5 = 0.5
        assert!((momentum_grads.get(&(0, 1)).unwrap() - 0.5).abs() < 1e-6);
        assert!((momentum_grads.get(&(1, 2)).unwrap() + 0.3).abs() < 1e-6);

        // 第二轮
        recorder.clear();
        recorder.record_edge_gradient(0, 1, 0.6);
        recorder.record_edge_gradient(1, 2, -0.2);

        let momentum_grads2 = recorder.apply_momentum();

        // 经典动量：v_2 = 0.9 * v_1 + g_2
        // v_2(0,1) = 0.9 * 0.5 + 0.6 = 1.05
        // v_2(1,2) = 0.9 * (-0.3) + (-0.2) = -0.47
        let expected_01 = 0.9 * 0.5 + 0.6;
        let expected_12 = 0.9 * (-0.3) + (-0.2);

        assert!((momentum_grads2.get(&(0, 1)).unwrap() - expected_01).abs() < 1e-6);
        assert!((momentum_grads2.get(&(1, 2)).unwrap() - expected_12).abs() < 1e-6);
    }

    #[test]
    fn test_optimization_step() {
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(1, 2, 0.8);

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 0.5);
        loss_gradients.insert((1, 2), -0.3);

        let gradients = graph.optimization_step(loss_gradients);

        assert!(gradients.contains_key(&(0, 1)));
        assert!(gradients.contains_key(&(1, 2)));

        // 温度应该退火
        assert!(graph.temperature() <= 1.0);
    }

    #[test]
    fn test_gradient_computation_with_low_temperature() {
        // 测试：低温下梯度计算不应产生 NaN 或 inf
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 0.5);
        graph.config.temperature = 0.1; // 最小温度

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 1.0);

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        // 梯度应该是有限的
        for &grad in gradients.values() {
            assert!(grad.is_finite(), "Gradient should be finite, got {}", grad);
        }
    }

    #[test]
    fn test_gradient_computation_with_zero_probability() {
        // 测试：概率接近 0 时的梯度计算
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 1e-7); // 接近 0 的概率

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 1.0);

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        // 梯度应该是有限的
        for &grad in gradients.values() {
            assert!(grad.is_finite(), "Gradient should be finite, got {}", grad);
        }
    }

    #[test]
    fn test_gradient_computation_with_one_probability() {
        // 测试：概率接近 1 时的梯度计算
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 1.0 - 1e-7); // 接近 1 的概率

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 1.0);

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        // 梯度应该是有限的
        for &grad in gradients.values() {
            assert!(grad.is_finite(), "Gradient should be finite, got {}", grad);
        }
    }

    #[test]
    fn test_smoothness_gradient_computation() {
        // 测试：平滑正则化梯度计算
        let mut graph = DifferentiableGraph::<Vec<f64>>::with_config(
            4,
            GradientConfig::new(1.0, true, 0.01, 0.01).with_smoothness(0.1),
        );

        // 添加共享源节点的边
        graph.add_learnable_edge(0, 1, 0.8);
        graph.add_learnable_edge(0, 2, 0.2);
        graph.add_learnable_edge(0, 3, 0.5);

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), -0.5);
        loss_gradients.insert((0, 2), -0.5);
        loss_gradients.insert((0, 3), -0.5);

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        // 平滑正则化应该使梯度趋向于平均
        // (0, 1) 的概率最高，平滑梯度应该为负（降低概率）
        // (0, 2) 的概率最低，平滑梯度应该为正（提高概率）
        assert!(gradients.contains_key(&(0, 1)));
        assert!(gradients.contains_key(&(0, 2)));
        assert!(gradients.contains_key(&(0, 3)));
    }

    #[test]
    fn test_sparsity_gradient_computation() {
        // 测试：稀疏正则化梯度计算
        let mut graph = DifferentiableGraph::<Vec<f64>>::with_config(
            3,
            GradientConfig::new(1.0, true, 0.01, 0.01).with_sparsity(0.1),
        );

        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(1, 2, 0.5);

        // 设置 logits 为正
        if let Some(edge) = graph.edges.get_mut(&(0, 1)) {
            edge.logits = 2.0; // 正 logits
        }
        if let Some(edge) = graph.edges.get_mut(&(1, 2)) {
            edge.logits = -2.0; // 负 logits
        }

        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), 0.0); // 无损失梯度，只有正则化梯度
        loss_gradients.insert((1, 2), 0.0);

        let gradients = graph.compute_structure_gradients(&loss_gradients);

        // 正 logits 应该得到正梯度（推向 0）
        assert!(*gradients.get(&(0, 1)).unwrap() > 0.0);
        // 负 logits 应该得到负梯度（推向 0）
        assert!(*gradients.get(&(1, 2)).unwrap() < 0.0);
    }

    #[test]
    fn test_ste_correction() {
        // 测试：STE 修正项计算
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(3);
        graph.add_learnable_edge(0, 1, 0.6); // 概率 > 0.5，离散化后为 1
        graph.add_learnable_edge(1, 2, 0.4); // 概率 < 0.5，离散化后为 0

        graph.discretize();

        let corrections = graph.get_ste_corrections();

        // (0, 1): hard=1, soft=0.6, correction=0.4
        assert!((corrections.get(&(0, 1)).unwrap() - 0.4).abs() < 0.01);
        // (1, 2): hard=0, soft=0.4, correction=-0.4
        assert!((corrections.get(&(1, 2)).unwrap() + 0.4).abs() < 0.01);
    }

    #[test]
    fn test_momentum_classical() {
        // 测试：经典动量公式
        let mut recorder = GradientRecorder::new(0.9);

        // 第一轮
        recorder.record_edge_gradient(0, 1, 1.0);
        let momentum_grads_1 = recorder.apply_momentum();
        // v_1 = 0.9 * 0 + 1.0 = 1.0
        assert!((momentum_grads_1.get(&(0, 1)).unwrap() - 1.0).abs() < 1e-6);

        // 第二轮
        recorder.clear();
        recorder.record_edge_gradient(0, 1, 1.0);
        let momentum_grads_2 = recorder.apply_momentum();
        // v_2 = 0.9 * 1.0 + 1.0 = 1.9
        assert!((momentum_grads_2.get(&(0, 1)).unwrap() - 1.9).abs() < 1e-6);

        // 第三轮
        recorder.clear();
        recorder.record_edge_gradient(0, 1, 1.0);
        let momentum_grads_3 = recorder.apply_momentum();
        // v_3 = 0.9 * 1.9 + 1.0 = 2.71
        assert!((momentum_grads_3.get(&(0, 1)).unwrap() - 2.71).abs() < 1e-6);
    }

    #[test]
    fn test_graph_conversion() {
        // 测试：DifferentiableGraph 与 Graph 的转换
        use crate::graph::traits::{GraphBase, GraphQuery};

        let mut diff_graph = DifferentiableGraph::<()>::new(4);
        diff_graph.add_learnable_edge(0, 1, 0.8);
        diff_graph.add_learnable_edge(1, 2, 0.3);
        diff_graph.add_learnable_edge(2, 3, 0.9);

        // 离散化
        diff_graph.discretize();

        // 转换为普通 Graph
        let graph = diff_graph.to_graph();

        // 验证节点数
        assert_eq!(graph.node_count(), 4);

        // 验证边：只有概率 > 0.5 的边应该存在
        // 使用 graph.nodes() 获取正确的 NodeIndex
        let nodes: Vec<_> = graph.nodes().collect();
        assert_eq!(nodes.len(), 4);

        // nodes 按索引排序，所以 nodes[0] 对应索引 0，等等
        let n0 = nodes[0].index();
        let n1 = nodes[1].index();
        let n2 = nodes[2].index();
        let n3 = nodes[3].index();

        // 检查边是否存在
        assert!(graph.has_edge(n0, n1)); // 0.8 > 0.5
        assert!(!graph.has_edge(n1, n2)); // 0.3 < 0.5
        assert!(graph.has_edge(n2, n3)); // 0.9 > 0.5
    }

    #[test]
    fn test_from_graph() {
        // 测试：从普通 Graph 初始化
        use crate::graph::builders::GraphBuilder;

        let graph = GraphBuilder::directed()
            .with_nodes(vec![(0, ()), (1, ()), (2, ()), (3, ())])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        // 使用 turbofish 语法明确指定类型
        let diff_graph = DifferentiableGraph::<()>::from_graph(&graph, None);

        assert_eq!(diff_graph.num_nodes(), 4);
        assert_eq!(diff_graph.num_edges(), 3);
        assert!(diff_graph.get_edge_probability(0, 1).is_some());
        assert!(diff_graph.get_edge_probability(1, 2).is_some());
        assert!(diff_graph.get_edge_probability(2, 3).is_some());
    }
}
