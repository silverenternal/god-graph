//! Hash 分区器实现
//!
//! 基于节点索引的哈希值进行分区

use super::traits::{PartitionId, Partitioner, PartitionerConfig};
use crate::node::NodeIndex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash 分区器
///
/// 使用哈希函数将节点均匀分布到各个分区
///
/// # 示例
///
/// ```
/// use god_graph::parallel::partitioner::HashPartitioner;
///
/// let partitioner = HashPartitioner::new(4);
/// assert_eq!(partitioner.num_partitions(), 4);
/// ```
pub struct HashPartitioner {
    num_partitions: usize,
    seed: u64,
}

impl HashPartitioner {
    /// 创建新的 Hash 分区器
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - 分区数量
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            seed: 0,
        }
    }

    /// 创建带种子的 Hash 分区器
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - 分区数量
    /// * `seed` - 哈希种子，用于产生不同的分区结果
    pub fn with_seed(num_partitions: usize, seed: u64) -> Self {
        Self {
            num_partitions,
            seed,
        }
    }

    /// 从配置创建分区器
    pub fn from_config(config: &PartitionerConfig) -> Self {
        let seed = config
            .properties
            .get("seed")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        Self::with_seed(config.num_partitions, seed)
    }

    /// 计算节点的哈希值
    fn hash_node(&self, node: NodeIndex) -> u64 {
        let mut hasher = DefaultHasher::new();
        node.index().hash(&mut hasher);
        self.seed.hash(&mut hasher);
        hasher.finish()
    }
}

impl Partitioner for HashPartitioner {
    fn name(&self) -> &'static str {
        "hash"
    }

    fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition_node(&self, node: NodeIndex) -> PartitionId {
        let hash = self.hash_node(node);
        (hash % self.num_partitions as u64) as PartitionId
    }
}

impl Default for HashPartitioner {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_hash_partitioner_basic() {
        let partitioner = HashPartitioner::new(4);
        assert_eq!(partitioner.num_partitions(), 4);
        assert_eq!(partitioner.name(), "hash");

        // 同一节点应该总是映射到同一分区
        let node = NodeIndex::new_public(5);
        let partition1 = partitioner.partition_node(node);
        let partition2 = partitioner.partition_node(node);
        assert_eq!(partition1, partition2);
    }

    #[test]
    fn test_hash_partitioner_with_seed() {
        let partitioner1 = HashPartitioner::with_seed(4, 0);
        let partitioner2 = HashPartitioner::with_seed(4, 42);

        let node = NodeIndex::new_public(5);
        let _p1 = partitioner1.partition_node(node);
        let _p2 = partitioner2.partition_node(node);

        // 不同种子可能产生不同分区
        // 注意：由于哈希碰撞，这可能偶尔相等
        let node2 = NodeIndex::new_public(100);
        let p1_2 = partitioner1.partition_node(node2);
        let p2_2 = partitioner2.partition_node(node2);

        // 验证分区在有效范围内
        assert!(p1_2 < 4);
        assert!(p2_2 < 4);
    }

    #[test]
    fn test_hash_partitioner_distribution() {
        let partitioner = HashPartitioner::new(4);
        let mut partition_counts = vec![0; 4];

        // 测试 1000 个节点的分布
        for i in 0..1000 {
            let node = NodeIndex::new_public(i);
            let partition = partitioner.partition_node(node);
            partition_counts[partition] += 1;
        }

        // 验证分布相对均匀（允许 50% 的偏差）
        let avg = 1000 / 4;
        for count in &partition_counts {
            assert!(
                *count >= avg / 2 && *count <= avg * 2,
                "Partition count {} is not balanced (avg: {})",
                count,
                avg
            );
        }
    }

    #[test]
    fn test_hash_partitioner_from_config() {
        let config = PartitionerConfig::new(8).with_property("seed", "12345");
        let partitioner = HashPartitioner::from_config(&config);

        assert_eq!(partitioner.num_partitions(), 8);
    }

    #[test]
    fn test_hash_partition_graph() {
        use crate::vgi::VirtualGraph;

        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..10 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(3);
        let partitions = partitioner.partition_graph(&graph);

        assert_eq!(partitions.len(), 3);

        // 验证所有节点都被分配
        let total_nodes: usize = partitions.iter().map(|p| p.size()).sum();
        assert_eq!(total_nodes, 10);
    }
}
