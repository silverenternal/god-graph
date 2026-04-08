//! Range 分区器实现
//!
//! 基于节点索引的范围进行分区

use super::traits::{PartitionId, Partitioner, PartitionerConfig};
use crate::node::NodeIndex;

/// Range 分区器
///
/// 将节点索引按范围划分到不同分区
///
/// # 示例
///
/// ```
/// use god_graph::parallel::partitioner::RangePartitioner;
///
/// let partitioner = RangePartitioner::new(4);
/// // 节点 0-2 在分区 0，节点 3-5 在分区 1，等等
/// ```
pub struct RangePartitioner {
    num_partitions: usize,
    nodes_per_partition: usize,
}

impl RangePartitioner {
    /// 创建新的 Range 分区器
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - 分区数量
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            nodes_per_partition: usize::MAX, // 默认无限制
        }
    }

    /// 创建带每分区节点数的 Range 分区器
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - 分区数量
    /// * `nodes_per_partition` - 每个分区的节点数
    pub fn with_nodes_per_partition(num_partitions: usize, nodes_per_partition: usize) -> Self {
        Self {
            num_partitions,
            nodes_per_partition,
        }
    }

    /// 从配置创建分区器
    pub fn from_config(config: &PartitionerConfig) -> Self {
        let nodes_per_partition = config.target_nodes_per_partition.unwrap_or(usize::MAX);
        Self::with_nodes_per_partition(config.num_partitions, nodes_per_partition)
    }

    /// 设置最大节点索引（用于动态调整）
    pub fn set_max_node_index(&mut self, max_index: usize) {
        if self.nodes_per_partition == usize::MAX && max_index > 0 {
            self.nodes_per_partition = (max_index + self.num_partitions) / self.num_partitions;
        }
    }
}

impl Partitioner for RangePartitioner {
    fn name(&self) -> &'static str {
        "range"
    }

    fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition_node(&self, node: NodeIndex) -> PartitionId {
        let index = node.index();
        if self.nodes_per_partition == usize::MAX {
            // 如果没有设置每分区节点数，使用简单除法
            index % self.num_partitions
        } else {
            index / self.nodes_per_partition
        }
    }
}

impl Default for RangePartitioner {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_partitioner_basic() {
        let partitioner = RangePartitioner::new(4);
        assert_eq!(partitioner.num_partitions(), 4);
        assert_eq!(partitioner.name(), "range");

        // 测试简单除法分布
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(0)), 0);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(1)), 1);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(2)), 2);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(3)), 3);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(4)), 0);
    }

    #[test]
    fn test_range_partitioner_with_nodes_per_partition() {
        let partitioner = RangePartitioner::with_nodes_per_partition(4, 10);

        // 0-9 在分区 0
        for i in 0..10 {
            assert_eq!(partitioner.partition_node(NodeIndex::new_public(i)), 0);
        }

        // 10-19 在分区 1
        for i in 10..20 {
            assert_eq!(partitioner.partition_node(NodeIndex::new_public(i)), 1);
        }

        // 20-29 在分区 2
        for i in 20..30 {
            assert_eq!(partitioner.partition_node(NodeIndex::new_public(i)), 2);
        }
    }

    #[test]
    fn test_range_partitioner_from_config() {
        let config = PartitionerConfig::new(4).with_target_nodes(100);
        let partitioner = RangePartitioner::from_config(&config);

        assert_eq!(partitioner.num_partitions(), 4);
    }

    #[test]
    fn test_range_partitioner_boundary() {
        let partitioner = RangePartitioner::with_nodes_per_partition(3, 5);

        // 边界测试
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(4)), 0);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(5)), 1);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(9)), 1);
        assert_eq!(partitioner.partition_node(NodeIndex::new_public(10)), 2);
    }
}
