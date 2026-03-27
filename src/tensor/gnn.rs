//! GNN（图神经网络）原语模块
//!
//! 提供图神经网络的核心构建块：
//! - 消息传递框架
//! - 图卷积层（GCN, GAT, GraphSAGE）
//! - 图 pooling 和 normalization
//!
//! ## 示例
//!
//! ```ignore
//! # #[cfg(feature = "tensor-gnn")]
//! # {
//! use god_gragh::tensor::gnn::{MessagePassingLayer, GCNConv, SumAggregator};
//! use god_gragh::tensor::DenseTensor;
//!
//! // 创建 GCN 层
//! let gcn = GCNConv::new(64, 64);
//!
//! // 前向传播
//! let output = gcn.forward(&node_features, &adjacency);
//! # }
//! ```

#[cfg(feature = "tensor-gnn")]
use rand_distr::{Distribution, StandardNormal};

#[cfg(feature = "tensor-gnn")]
use crate::tensor::traits::{TensorBase, TensorOps};

#[cfg(feature = "tensor-gnn")]
use crate::tensor::dense::DenseTensor;

#[cfg(feature = "tensor-gnn")]
use crate::tensor::sparse::SparseTensor;

#[cfg(all(feature = "tensor-gnn", not(feature = "std")))]
use rand::{rngs::StdRng, SeedableRng};

#[cfg(all(feature = "tensor-gnn", feature = "std"))]
use rand::thread_rng;

/// 消息函数 trait：定义边上的消息计算
pub trait MessageFunction<H: TensorBase>: Send + Sync {
    /// 计算消息
    ///
    /// # Arguments
    /// * `src_features` - 源节点特征
    /// * `edge_features` - 边特征（可选）
    /// * `dst_features` - 目标节点特征
    ///
    /// # Returns
    /// 返回计算得到的消息张量
    fn message(&self, src_features: &H, edge_features: Option<&H>, dst_features: &H) -> H;
}

/// 聚合器 trait：定义邻居消息的聚合方式
pub trait Aggregator<H: TensorBase>: Send + Sync {
    /// 聚合消息
    ///
    /// # Arguments
    /// * `messages` - 消息切片
    ///
    /// # Returns
    /// 返回聚合后的张量
    fn aggregate(&self, messages: &[H]) -> H;
}

/// 更新函数 trait：定义节点状态更新
pub trait UpdateFunction<H: TensorBase>: Send + Sync {
    /// 更新节点状态
    ///
    /// # Arguments
    /// * `old_state` - 旧的节点状态
    /// * `new_message` - 新聚合的消息
    ///
    /// # Returns
    /// 返回更新后的状态
    fn update(&self, old_state: &H, new_message: &H) -> H;
}

/// 求和聚合器
#[derive(Debug, Clone, Default)]
pub struct SumAggregator;

#[cfg(feature = "tensor-gnn")]
impl Aggregator<DenseTensor> for SumAggregator {
    fn aggregate(&self, messages: &[DenseTensor]) -> DenseTensor {
        if messages.is_empty() {
            return DenseTensor::zeros(vec![1]);
        }

        let mut result = messages[0].clone();
        for msg in &messages[1..] {
            result = result.add(msg);
        }
        result
    }
}

/// 均值聚合器
#[derive(Debug, Clone, Default)]
pub struct MeanAggregator;

#[cfg(feature = "tensor-gnn")]
impl Aggregator<DenseTensor> for MeanAggregator {
    fn aggregate(&self, messages: &[DenseTensor]) -> DenseTensor {
        if messages.is_empty() {
            return DenseTensor::zeros(vec![1]);
        }

        let sum = SumAggregator.aggregate(messages);
        sum.mul_scalar(1.0 / messages.len() as f64)
    }
}

/// 最大值聚合器
#[derive(Debug, Clone, Default)]
pub struct MaxAggregator;

#[cfg(feature = "tensor-gnn")]
impl Aggregator<DenseTensor> for MaxAggregator {
    fn aggregate(&self, messages: &[DenseTensor]) -> DenseTensor {
        if messages.is_empty() {
            return DenseTensor::zeros(vec![1]);
        }

        let mut result = messages[0].clone();
        for msg in &messages[1..] {
            // 逐元素取最大值
            let data = result.data().to_vec();
            let msg_data = msg.data();
            let max_data: Vec<f64> = data
                .iter()
                .zip(msg_data.iter())
                .map(|(&a, &b)| a.max(b))
                .collect();
            result = DenseTensor::new(max_data, result.shape().to_vec());
        }
        result
    }
}

/// 恒等消息函数：直接传递源节点特征
#[derive(Debug, Clone, Default)]
pub struct IdentityMessage;

#[cfg(feature = "tensor-gnn")]
impl MessageFunction<DenseTensor> for IdentityMessage {
    fn message(
        &self,
        src_features: &DenseTensor,
        _edge_features: Option<&DenseTensor>,
        _dst_features: &DenseTensor,
    ) -> DenseTensor {
        src_features.clone()
    }
}

/// 线性消息函数：应用线性变换
#[derive(Debug, Clone)]
pub struct LinearMessage {
    /// 权重矩阵
    weight: DenseTensor,
}

#[cfg(feature = "tensor-gnn")]
impl LinearMessage {
    /// 创建新的线性消息函数
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier 初始化
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = thread_rng();
        let weight_data: Vec<f64> = (0..in_features * out_features)
            .map(|_| {
                let x: f64 = StandardNormal.sample(&mut rng);
                x * std
            })
            .collect();

        Self {
            weight: DenseTensor::new(weight_data, vec![in_features, out_features]),
        }
    }
}

#[cfg(feature = "tensor-gnn")]
impl MessageFunction<DenseTensor> for LinearMessage {
    fn message(
        &self,
        src_features: &DenseTensor,
        _edge_features: Option<&DenseTensor>,
        _dst_features: &DenseTensor,
    ) -> DenseTensor {
        // src_features @ weight.T
        src_features.matmul(&self.weight.transpose(None))
    }
}

/// 消息传递层：GNN 的核心构建块
pub struct MessagePassingLayer<M, A, U> {
    /// 消息函数
    message_fn: M,
    /// 聚合器
    aggregator: A,
    /// 更新函数
    update_fn: U,
}

impl<M, A, U> MessagePassingLayer<M, A, U>
where
    M: MessageFunction<DenseTensor>,
    A: Aggregator<DenseTensor>,
    U: UpdateFunction<DenseTensor>,
{
    /// 创建新的消息传递层
    pub fn new(message_fn: M, aggregator: A, update_fn: U) -> Self {
        Self {
            message_fn,
            aggregator,
            update_fn,
        }
    }

    /// 前向传播
    ///
    /// # Arguments
    /// * `node_features` - 节点特征 [num_nodes, hidden_size]
    /// * `edge_index` - 边索引 [(src, dst), ...]
    /// * `edge_features` - 边特征（可选）
    ///
    /// # Returns
    /// 返回更新后的节点特征
    pub fn forward(
        &self,
        node_features: &DenseTensor,
        edge_index: &[(usize, usize)],
        edge_features: Option<&DenseTensor>,
    ) -> DenseTensor {
        // 为每个节点收集消息
        let mut messages: Vec<Vec<DenseTensor>> = vec![Vec::new(); node_features.shape()[0]];

        for (src, dst) in edge_index {
            let src_feat = self.extract_node(node_features, *src);
            let dst_feat = self.extract_node(node_features, *dst);
            let edge_feat = edge_features.map(|_| DenseTensor::scalar(1.0)); // 简化

            let msg = self
                .message_fn
                .message(&src_feat, edge_feat.as_ref(), &dst_feat);
            messages[*dst].push(msg);
        }

        // 聚合消息并更新
        let mut updated_features = Vec::new();
        for (node_idx, node_msgs) in messages.iter().enumerate() {
            let old_state = self.extract_node(node_features, node_idx);

            if node_msgs.is_empty() {
                updated_features.extend_from_slice(old_state.data());
            } else {
                let aggregated = self.aggregator.aggregate(node_msgs);
                let updated = self.update_fn.update(&old_state, &aggregated);
                updated_features.extend_from_slice(updated.data());
            }
        }

        DenseTensor::new(updated_features, node_features.shape().to_vec())
    }

    /// 提取节点特征
    fn extract_node(&self, features: &DenseTensor, node_idx: usize) -> DenseTensor {
        let num_features = features.shape()[1];
        let start = node_idx * num_features;
        let _end = start + num_features;
        features.slice(&[0, 1], &[node_idx..node_idx + 1, 0..num_features])
    }
}

/// GCN（图卷积网络）层
#[allow(dead_code)]
pub struct GCNConv {
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
    /// 权重矩阵
    weight: DenseTensor,
    /// 偏置
    bias: DenseTensor,
}

#[cfg(feature = "tensor-gnn")]
impl GCNConv {
    /// 创建新的 GCN 层
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier 初始化
        let std = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = thread_rng();
        let weight_data: Vec<f64> = (0..in_features * out_features)
            .map(|_| {
                let x: f64 = StandardNormal.sample(&mut rng);
                x * std
            })
            .collect();

        let bias_data = vec![0.0; out_features];

        Self {
            in_features,
            out_features,
            weight: DenseTensor::new(weight_data, vec![in_features, out_features]),
            bias: DenseTensor::new(bias_data, vec![out_features]),
        }
    }

    /// 前向传播
    ///
    /// # Arguments
    /// * `node_features` - 节点特征 [num_nodes, in_features]
    /// * `adjacency` - 邻接矩阵（稀疏格式）
    ///
    /// # Returns
    /// 返回更新后的节点特征 [num_nodes, out_features]
    pub fn forward(&self, node_features: &DenseTensor, adjacency: &SparseTensor) -> DenseTensor {
        // 1. 线性变换：H @ W
        let h_transformed = node_features.matmul(&self.weight);

        // 2. 度归一化：D^(-1/2) A D^(-1/2)
        let normalized = self.normalize_adjacency(adjacency);

        // 3. 图卷积：normalized_adj @ H_transformed
        normalized.spmv(&h_transformed).unwrap()
    }

    /// 归一化邻接矩阵
    fn normalize_adjacency(&self, adjacency: &SparseTensor) -> SparseTensor {
        // 计算度
        let degrees = self.compute_degrees(adjacency);

        // 计算 D^(-1/2)
        let _inv_sqrt_degrees = degrees.map(|d: f64| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 });

        // 归一化：D^(-1/2) A D^(-1/2)
        // 简化实现：实际需要对每个边权重乘以对应的度归一化因子
        adjacency.clone() // TODO: 实现完整的归一化
    }

    /// 计算节点度
    fn compute_degrees(&self, adjacency: &SparseTensor) -> DenseTensor {
        let num_nodes = adjacency.shape()[0];
        let mut degrees = vec![0.0; num_nodes];

        let coo = adjacency.to_coo();
        for &row in coo.row_indices() {
            degrees[row] += 1.0;
        }

        DenseTensor::new(degrees, vec![num_nodes])
    }
}

/// GAT（图注意力网络）层
#[allow(dead_code)]
pub struct GATConv {
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
    /// 注意力头数
    num_heads: usize,
    /// 注意力权重向量
    attention_vec: DenseTensor,
}

#[cfg(feature = "tensor-gnn")]
impl GATConv {
    /// 创建新的 GAT 层
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let std = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = thread_rng();
        let attention_data: Vec<f64> = (0..out_features * 2)
            .map(|_| {
                let x: f64 = StandardNormal.sample(&mut rng);
                x * std
            })
            .collect();

        Self {
            in_features,
            out_features,
            num_heads,
            attention_vec: DenseTensor::new(attention_data, vec![out_features * 2]),
        }
    }

    /// 前向传播
    pub fn forward(
        &self,
        node_features: &DenseTensor,
        edge_index: &[(usize, usize)],
    ) -> DenseTensor {
        // 1. 线性变换
        let h_transformed = node_features.matmul(&self.weight());

        // 2. 计算注意力分数
        let attention_scores = self.compute_attention(node_features, edge_index);

        // 3. Softmax 归一化
        let normalized_attention = self.softmax(&attention_scores, edge_index);

        // 4. 加权聚合
        self.aggregate_with_attention(&h_transformed, &normalized_attention, edge_index)
    }

    /// 获取权重矩阵
    fn weight(&self) -> DenseTensor {
        // 简化实现
        DenseTensor::eye(self.in_features)
    }

    /// 计算注意力分数
    fn compute_attention(
        &self,
        node_features: &DenseTensor,
        edge_index: &[(usize, usize)],
    ) -> Vec<f64> {
        edge_index
            .iter()
            .map(|(src, dst)| {
                let src_feat = node_features.data()
                    [src * self.in_features..(src + 1) * self.in_features]
                    .to_vec();
                let dst_feat = node_features.data()
                    [dst * self.in_features..(dst + 1) * self.in_features]
                    .to_vec();

                // 拼接并计算注意力
                let mut concatenated = src_feat;
                concatenated.extend_from_slice(&dst_feat);

                // LeakyReLU(attention_vec @ concatenated)
                let score: f64 = concatenated
                    .iter()
                    .zip(self.attention_vec.data().iter().cycle())
                    .map(|(&a, &b)| a * b)
                    .sum();

                score.max(0.0) // LeakyReLU with alpha=0
            })
            .collect()
    }

    /// Softmax 归一化
    fn softmax(&self, scores: &[f64], edge_index: &[(usize, usize)]) -> Vec<f64> {
        // 按目标节点分组
        let mut dst_scores: std::collections::HashMap<usize, Vec<(usize, f64)>> =
            std::collections::HashMap::new();

        for ((src, dst), score) in edge_index.iter().zip(scores.iter()) {
            dst_scores.entry(*dst).or_default().push((*src, *score));
        }

        // 对每个目标节点的注意力分数进行 softmax
        let mut normalized = vec![0.0; scores.len()];
        for (dst, scores) in dst_scores {
            let max_score = scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<(usize, f64)> = scores
                .iter()
                .map(|(src, s)| (*src, (*s - max_score).exp()))
                .collect();

            let sum_exp: f64 = exp_scores.iter().map(|(_, e)| *e).sum();

            for (src, exp_val) in exp_scores {
                // 找到对应的索引
                if let Some(idx) = edge_index.iter().position(|(s, d)| *s == src && *d == dst) {
                    normalized[idx] = exp_val / sum_exp;
                }
            }
        }

        normalized
    }

    /// 带注意力的聚合
    fn aggregate_with_attention(
        &self,
        node_features: &DenseTensor,
        attention: &[f64],
        edge_index: &[(usize, usize)],
    ) -> DenseTensor {
        let num_nodes = node_features.shape()[0];
        let mut result = vec![0.0; num_nodes * self.out_features];

        for ((src, dst), &attn) in edge_index.iter().zip(attention.iter()) {
            for i in 0..self.out_features {
                result[dst * self.out_features + i] +=
                    attn * node_features.data()[src * self.in_features + i];
            }
        }

        DenseTensor::new(result, vec![num_nodes, self.out_features])
    }
}

/// GraphSAGE 层
pub struct GraphSAGE {
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
    /// 邻居采样数
    num_samples: usize,
}

#[cfg(feature = "tensor-gnn")]
impl GraphSAGE {
    /// 创建新的 GraphSAGE 层
    pub fn new(in_features: usize, out_features: usize, num_samples: usize) -> Self {
        Self {
            in_features,
            out_features,
            num_samples,
        }
    }

    /// 前向传播
    pub fn forward(
        &self,
        node_features: &DenseTensor,
        edge_index: &[(usize, usize)],
    ) -> DenseTensor {
        let num_nodes = node_features.shape()[0];
        let mut result = Vec::new();

        for node_idx in 0..num_nodes {
            // 1. 采样邻居
            let neighbors: Vec<usize> = edge_index
                .iter()
                .filter(|(src, _)| *src == node_idx)
                .take(self.num_samples)
                .map(|(_, dst)| *dst)
                .collect();

            // 2. 聚合邻居特征（均值）
            let neighbor_features = if neighbors.is_empty() {
                DenseTensor::zeros(vec![self.in_features])
            } else {
                let features: Vec<DenseTensor> = neighbors
                    .iter()
                    .map(|&n| {
                        let start = n * self.in_features;
                        let end = start + self.in_features;
                        DenseTensor::new(
                            node_features.data()[start..end].to_vec(),
                            vec![self.in_features],
                        )
                    })
                    .collect();
                MeanAggregator.aggregate(&features)
            };

            // 3. 拼接自身特征和邻居特征
            let self_features = node_features.data()
                [node_idx * self.in_features..(node_idx + 1) * self.in_features]
                .to_vec();
            let mut concatenated = self_features;
            concatenated.extend_from_slice(neighbor_features.data());

            // 4. 线性变换（简化：直接取前 out_features 个）
            let transformed: Vec<f64> = concatenated
                .iter()
                .take(self.out_features)
                .copied()
                .collect();

            result.extend_from_slice(&transformed);
        }

        DenseTensor::new(result, vec![num_nodes, self.out_features])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_aggregator() {
        let aggregator = SumAggregator;
        let messages = vec![
            DenseTensor::new(vec![1.0, 2.0], vec![2]),
            DenseTensor::new(vec![3.0, 4.0], vec![2]),
            DenseTensor::new(vec![5.0, 6.0], vec![2]),
        ];

        let result = aggregator.aggregate(&messages);
        assert_eq!(result.data(), &[9.0, 12.0]);
    }

    #[test]
    fn test_mean_aggregator() {
        let aggregator = MeanAggregator;
        let messages = vec![
            DenseTensor::new(vec![1.0, 2.0], vec![2]),
            DenseTensor::new(vec![3.0, 4.0], vec![2]),
            DenseTensor::new(vec![5.0, 6.0], vec![2]),
        ];

        let result = aggregator.aggregate(&messages);
        assert_eq!(result.data(), &[3.0, 4.0]);
    }

    #[test]
    fn test_identity_message() {
        let message_fn = IdentityMessage;
        let src = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let dst = DenseTensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let result = message_fn.message(&src, None, &dst);
        assert_eq!(result.data(), src.data());
    }
}
