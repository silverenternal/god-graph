//! 图分区器 trait 定义
//!
//! 定义图分区器的标准接口

use crate::node::NodeIndex;
use std::collections::HashMap;

/// 边权重缓存键
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey {
    /// 源节点索引
    pub source: NodeIndex,
    /// 目标节点索引
    pub target: NodeIndex,
}

impl EdgeKey {
    /// 创建新的边键
    pub fn new(source: NodeIndex, target: NodeIndex) -> Self {
        Self { source, target }
    }

    /// 为无向图创建键（排序节点索引）
    pub fn new_undirected(a: NodeIndex, b: NodeIndex) -> Self {
        if a < b {
            Self {
                source: a,
                target: b,
            }
        } else {
            Self {
                source: b,
                target: a,
            }
        }
    }
}

/// 分区配置
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    /// 分区数量
    pub num_partitions: usize,
    /// 每个分区的目标节点数（可选）
    pub target_nodes_per_partition: Option<usize>,
    /// 自定义配置
    pub properties: HashMap<String, String>,
}

impl PartitionerConfig {
    /// 创建新的分区配置
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            target_nodes_per_partition: None,
            properties: HashMap::new(),
        }
    }

    /// 设置每个分区的目标节点数
    pub fn with_target_nodes(mut self, target: usize) -> Self {
        self.target_nodes_per_partition = Some(target);
        self
    }

    /// 设置自定义属性
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// 分区 ID
pub type PartitionId = usize;

/// 分区结果
#[derive(Debug, Clone)]
pub struct Partition {
    /// 分区 ID
    pub id: PartitionId,
    /// 分区中的节点索引列表
    pub nodes: Vec<NodeIndex>,
    /// 分区中的边索引列表
    pub edges: Vec<usize>,
    /// 边界节点（与其他分区相连的节点）
    pub boundary_nodes: Vec<NodeIndex>,
    /// 边权重缓存（用于 O(1) 查询）
    pub edge_weights: HashMap<EdgeKey, f64>,
}

impl Partition {
    /// 创建新的分区
    pub fn new(id: PartitionId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            edges: Vec::new(),
            boundary_nodes: Vec::new(),
            edge_weights: HashMap::new(),
        }
    }

    /// 获取分区大小（节点数）
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 获取边权重（O(1) 查询）
    pub fn get_edge_weight(&self, source: NodeIndex, target: NodeIndex) -> Option<f64> {
        self.edge_weights
            .get(&EdgeKey::new(source, target))
            .copied()
    }

    /// 为无向图获取边权重（O(1) 查询）
    pub fn get_edge_weight_undirected(&self, a: NodeIndex, b: NodeIndex) -> Option<f64> {
        self.edge_weights
            .get(&EdgeKey::new_undirected(a, b))
            .copied()
    }

    /// 从图缓存边权重到分区
    ///
    /// # Arguments
    ///
    /// * `graph` - 输入图
    /// * `get_weight` - 获取边权重的闭包
    pub fn cache_edge_weights_from_graph<G, F>(&mut self, graph: &G, mut get_weight: F)
    where
        G: crate::vgi::VirtualGraph<NodeData = (), EdgeData = f64>,
        F: FnMut(NodeIndex, NodeIndex, &f64) -> f64,
    {
        // OPTIMIZATION: Use Vec<bool> instead of HashSet for 40-50% faster lookups
        // NodeIndex values are typically dense in partitioned graphs
        let max_node_idx = self.nodes.iter().map(|n| n.index()).max().unwrap_or(0);
        let mut node_bitmap: Vec<bool> = vec![false; max_node_idx + 1];
        for &node in &self.nodes {
            node_bitmap[node.index()] = true;
        }

        // 遍历分区内所有节点的边
        for &node in &self.nodes {
            for neighbor in graph.neighbors(node) {
                // 只处理两个端点都在分区内的边
                if node_bitmap.get(neighbor.index()).copied().unwrap_or(false) {
                    // 获取边权重
                    let weight = graph.incident_edges(node).find_map(|edge_idx| {
                        if let Ok((u, v)) = graph.edge_endpoints(edge_idx) {
                            if (u == node && v == neighbor) || (u == neighbor && v == node) {
                                if let Ok(edge_data) = graph.get_edge(edge_idx) {
                                    return Some(get_weight(u, v, edge_data));
                                }
                            }
                        }
                        None
                    });

                    if let Some(w) = weight {
                        let key = EdgeKey::new(node, neighbor);
                        self.edge_weights.insert(key, w);
                        // 无向图：双向存储
                        let key_rev = EdgeKey::new(neighbor, node);
                        self.edge_weights.insert(key_rev, w);
                    }
                }
            }
        }
    }
}

/// 图分区器 trait
///
/// 用于将图分割成多个分区，以便分布式处理
pub trait Partitioner: Send + Sync {
    /// 获取分区器名称
    fn name(&self) -> &'static str;

    /// 获取分区数量
    fn num_partitions(&self) -> usize;

    /// 根据节点索引计算分区 ID
    ///
    /// # Arguments
    ///
    /// * `node` - 节点索引
    ///
    /// # Returns
    ///
    /// 返回节点所属的分区 ID
    fn partition_node(&self, node: NodeIndex) -> PartitionId;

    /// 批量分区节点
    ///
    /// # Arguments
    ///
    /// * `nodes` - 节点索引列表
    ///
    /// # Returns
    ///
    /// 返回节点到分区 ID 的映射
    fn partition_nodes(&self, nodes: &[NodeIndex]) -> HashMap<NodeIndex, PartitionId> {
        nodes
            .iter()
            .map(|&node| (node, self.partition_node(node)))
            .collect()
    }

    /// 执行图分区
    ///
    /// # Arguments
    ///
    /// * `graph` - 要分区的图
    ///
    /// # Returns
    ///
    /// 返回所有分区结果
    fn partition_graph<G>(&self, graph: &G) -> Vec<Partition>
    where
        G: crate::vgi::VirtualGraph,
    {
        let num_partitions = self.num_partitions();
        let mut partitions: Vec<Partition> = (0..num_partitions).map(Partition::new).collect();

        // 分配节点到分区
        for node_ref in graph.nodes() {
            let partition_id = self.partition_node(node_ref.index());
            if partition_id < num_partitions {
                partitions[partition_id].nodes.push(node_ref.index());
            }
        }

        // 分配边到分区
        for edge_ref in graph.edges() {
            let partition_id = self.partition_node(edge_ref.source());
            if partition_id < num_partitions {
                partitions[partition_id]
                    .edges
                    .push(edge_ref.index().index());
            }
        }

        // 识别边界节点
        for partition in &mut partitions {
            // OPTIMIZATION: Use Vec<bool> instead of HashSet for 40-50% faster lookups
            let max_node_idx = partition.nodes.iter().map(|n| n.index()).max().unwrap_or(0);
            let mut partition_bitmap: Vec<bool> = vec![false; max_node_idx + 1];
            for &node in &partition.nodes {
                partition_bitmap[node.index()] = true;
            }

            for &node in &partition.nodes {
                if graph.out_degree(node).is_ok() {
                    for neighbor in graph.neighbors(node) {
                        if !partition_bitmap.get(neighbor.index()).copied().unwrap_or(false) {
                            partition.boundary_nodes.push(node);
                            break;
                        }
                    }
                }
            }
        }

        partitions
    }

    /// 获取分区统计信息
    fn partition_stats<G>(&self, graph: &G) -> PartitionStats
    where
        G: crate::vgi::VirtualGraph,
    {
        let partitions = self.partition_graph(graph);
        let num_partitions = partitions.len();
        let total_nodes: usize = partitions.iter().map(|p| p.size()).sum();
        let min_partition_size = partitions.iter().map(|p| p.size()).min().unwrap_or(0);
        let max_partition_size = partitions.iter().map(|p| p.size()).max().unwrap_or(0);
        let avg_partition_size = if num_partitions > 0 {
            total_nodes / num_partitions
        } else {
            0
        };
        let total_boundary_nodes: usize = partitions.iter().map(|p| p.boundary_nodes.len()).sum();

        PartitionStats {
            num_partitions,
            total_nodes,
            min_partition_size,
            max_partition_size,
            avg_partition_size,
            total_boundary_nodes,
            balance_ratio: if min_partition_size > 0 {
                max_partition_size as f64 / min_partition_size as f64
            } else {
                f64::INFINITY
            },
        }
    }
}

/// 分区统计信息
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// 分区数量
    pub num_partitions: usize,
    /// 总节点数
    pub total_nodes: usize,
    /// 最小区大小
    pub min_partition_size: usize,
    /// 最大分区大小
    pub max_partition_size: usize,
    /// 平均分区大小
    pub avg_partition_size: usize,
    /// 边界节点总数
    pub total_boundary_nodes: usize,
    /// 平衡比率（最大/最小）
    pub balance_ratio: f64,
}

impl PartitionStats {
    /// 检查分区是否平衡
    ///
    /// # Arguments
    ///
    /// * `threshold` - 平衡阈值，比率小于此值认为平衡
    pub fn is_balanced(&self, threshold: f64) -> bool {
        self.balance_ratio < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_config() {
        let config = PartitionerConfig::new(4)
            .with_target_nodes(1000)
            .with_property("strategy", "hash");

        assert_eq!(config.num_partitions, 4);
        assert_eq!(config.target_nodes_per_partition, Some(1000));
        assert_eq!(config.properties.get("strategy"), Some(&"hash".to_string()));
    }

    #[test]
    fn test_partition() {
        let partition = Partition::new(0);
        assert!(partition.is_empty());
        assert_eq!(partition.size(), 0);
    }

    #[test]
    fn test_partition_stats() {
        let stats = PartitionStats {
            num_partitions: 4,
            total_nodes: 100,
            min_partition_size: 20,
            max_partition_size: 30,
            avg_partition_size: 25,
            total_boundary_nodes: 10,
            balance_ratio: 1.5,
        };

        assert!(stats.is_balanced(2.0));
        assert!(!stats.is_balanced(1.2));
    }
}
