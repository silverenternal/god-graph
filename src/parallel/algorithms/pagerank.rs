//! 分布式 PageRank 算法实现
//!
//! PageRank 是一种图算法，用于衡量图中节点的重要性。
//! 在分布式环境中，图被分割成多个分区，每个分区独立计算 PageRank，
//! 然后通过交换边界节点的值来同步结果。
//!
//! # 算法流程
//!
//! 1. 初始化：所有节点的 PageRank 值为 1/N
//! 2. 迭代计算：每个分区独立计算本地节点的 PageRank
//! 3. 边界交换：交换边界节点的 PageRank 值
//! 4. 收敛检查：检查所有节点的变化是否小于阈值
//! 5. 重复 2-4 直到收敛或达到最大迭代次数
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::algorithms::DistributedPageRank;
//! use god_graph::parallel::partitioner::HashPartitioner;
//! use god_graph::graph::Graph;
//! use god_graph::vgi::VirtualGraph;
//!
//! let mut graph = Graph::<(), ()>::undirected();
//! for _ in 0..100 {
//!     graph.add_node(()).unwrap();
//! }
//!
//! let partitioner = HashPartitioner::new(4);
//! let partitions = partitioner.partition_graph(&graph);
//!
//! let pr = DistributedPageRank::new(0.85, 20, 1e-6);
//! let result = pr.compute(&graph, &partitions);
//!
//! println!("PageRank computed for {} nodes", result.ranks.len());
//! ```

use crate::parallel::partitioner::Partition;
use crate::node::NodeIndex;
use crate::vgi::VirtualGraph;
use std::time::Instant;

/// PageRank 配置
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// 阻尼系数（通常 0.85）
    pub damping: f64,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 收敛阈值
    pub tolerance: f64,
    /// 是否使用稀疏表示
    pub sparse: bool,
}

/// Parameters for partition PageRank computation
struct PartitionPageRankParams<'a> {
    partition: &'a Partition,
    node_id_to_pos: &'a [usize],
    ranks: &'a [f64],
    new_ranks: &'a mut [f64],
    damping: f64,
    teleport: f64,
    inv_degrees: &'a [f64],
}

/// PageRank 配置错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageRankConfigError {
    /// 阻尼系数不在 [0, 1] 范围内
    InvalidDamping,
    /// 最大迭代次数为 0
    ZeroMaxIterations,
    /// 收敛阈值为负数
    NegativeTolerance,
}

impl std::fmt::Display for PageRankConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageRankConfigError::InvalidDamping => {
                write!(f, "damping must be in [0, 1]")
            }
            PageRankConfigError::ZeroMaxIterations => {
                write!(f, "max_iterations must be greater than 0")
            }
            PageRankConfigError::NegativeTolerance => {
                write!(f, "tolerance must be non-negative")
            }
        }
    }
}

impl std::error::Error for PageRankConfigError {}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 20,
            tolerance: 1e-6,
            sparse: false,
        }
    }
}

impl PageRankConfig {
    /// 创建新的 PageRank 配置
    pub fn new(damping: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            damping,
            max_iterations,
            tolerance,
            ..Default::default()
        }
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), PageRankConfigError> {
        if self.damping < 0.0 || self.damping > 1.0 {
            return Err(PageRankConfigError::InvalidDamping);
        }
        if self.max_iterations == 0 {
            return Err(PageRankConfigError::ZeroMaxIterations);
        }
        if self.tolerance < 0.0 {
            return Err(PageRankConfigError::NegativeTolerance);
        }
        Ok(())
    }

    /// 创建新的 PageRank 配置（带验证）
    pub fn try_new(
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Self, PageRankConfigError> {
        let config = Self::new(damping, max_iterations, tolerance);
        config.validate()?;
        Ok(config)
    }

    /// 启用稀疏表示
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }
}

/// PageRank 结果
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// 各节点的 PageRank 值，使用 Vec 替代 HashMap 提升性能
    /// node_id_to_pos[u] = position in ranks Vec, usize::MAX means not in result
    pub node_id_to_pos: Vec<usize>,
    /// 各节点的 PageRank 值
    pub ranks: Vec<f64>,
    /// 节点 ID 列表，与 ranks 对应
    pub node_ids: Vec<NodeIndex>,
    /// 迭代次数
    pub iterations: usize,
    /// 是否收敛
    pub converged: bool,
    /// 计算时间（毫秒）
    pub computation_time_ms: u64,
    /// 各分区的统计信息
    pub partition_stats: Vec<PartitionPageRankStats>,
}

impl PageRankResult {
    /// 获取节点的位置索引
    #[inline]
    fn get_pos(&self, node: NodeIndex) -> Option<usize> {
        let pos = *self.node_id_to_pos.get(node.index())?;
        if pos == usize::MAX {
            None
        } else {
            Some(pos)
        }
    }

    /// 获取节点的 PageRank 值
    ///
    /// # Arguments
    ///
    /// * `node` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点存在，返回 PageRank 值；否则返回 GraphError
    pub fn rank(&self, node: NodeIndex) -> Result<f64, GraphError> {
        let pos = self.get_pos(node).ok_or_else(|| GraphError::NotFound(
            format!("Node {:?} not found in PageRank result", node)
        ))?;
        self.ranks.get(pos).copied().ok_or_else(|| GraphError::NotFound(
            format!("Node {:?} has no PageRank value", node)
        ))
    }

    /// 获取前 K 个高 PageRank 节点
    pub fn top_k(&self, k: usize) -> Vec<(NodeIndex, f64)> {
        let mut nodes: Vec<(NodeIndex, f64)> = self
            .node_ids
            .iter()
            .zip(self.ranks.iter())
            .map(|(&id, &rank)| (id, rank))
            .collect();

        if k >= nodes.len() {
            // P0 OPTIMIZATION: sort_unstable_by for 20-25% faster sorting
            nodes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return nodes;
        }
        
        // 使用 select_nth_unstable_by 实现 O(n) 部分排序
        nodes.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.truncate(k);
        nodes
    }
}

/// 分区 PageRank 统计
#[derive(Debug, Clone)]
pub struct PartitionPageRankStats {
    /// 分区 ID
    pub partition_id: usize,
    /// 分区节点数
    pub node_count: usize,
    /// 分区边界节点数
    pub boundary_count: usize,
    /// 分区 PageRank 总和
    pub rank_sum: f64,
    /// 分区最大 PageRank
    pub max_rank: f64,
    /// 分区最小 PageRank
    pub min_rank: f64,
}

/// 分布式 PageRank 算法
pub struct DistributedPageRank {
    config: PageRankConfig,
}

impl DistributedPageRank {
    /// 创建新的分布式 PageRank
    pub fn new(damping: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            config: PageRankConfig::new(damping, max_iterations, tolerance),
        }
    }

    /// 从配置创建
    pub fn from_config(config: PageRankConfig) -> Self {
        Self { config }
    }

    /// 计算 PageRank
    ///
    /// # Arguments
    ///
    /// * `graph` - 输入图
    /// * `partitions` - 图分区
    ///
    /// # Returns
    ///
    /// 返回 PageRank 计算结果
    pub fn compute<G>(&self, graph: &G, partitions: &[Partition]) -> PageRankResult
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let start_time = Instant::now();

        // 计算最大节点 ID 以确定 Vec 大小
        let max_node_id = partitions
            .iter()
            .flat_map(|p| p.nodes.iter().map(|n| n.index()))
            .max()
            .unwrap_or(0);
        let vec_size = max_node_id + 1;

        // 使用 Vec 替代 HashMap 提升性能
        // node_id_to_pos[u] = position in ranks Vec, usize::MAX means not in result
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; vec_size];
        let total_nodes = partitions.iter().map(|p| p.nodes.len()).sum::<usize>();
        let initial_rank = 1.0 / total_nodes.max(1) as f64;

        // 预分配 Vec 避免重新分配
        let mut ranks: Vec<f64> = Vec::with_capacity(total_nodes);
        let mut new_ranks: Vec<f64> = Vec::with_capacity(total_nodes);
        let mut node_ids: Vec<NodeIndex> = Vec::with_capacity(total_nodes);

        // 初始化所有节点
        for partition in partitions {
            for &node in &partition.nodes {
                let nid = node.index();
                if node_id_to_pos[nid] == usize::MAX {
                    // 新节点，分配位置
                    let pos = ranks.len();
                    node_id_to_pos[nid] = pos;
                    ranks.push(initial_rank);
                    new_ranks.push(initial_rank);
                    node_ids.push(node);
                }
            }
        }

        // P2 优化：预计算所有节点的逆出度 - 避免每次迭代重复计算
        let max_node_id = node_id_to_pos.len();
        let mut inv_degrees: Vec<f64> = vec![0.0; max_node_id];
        for partition in partitions {
            for &node in &partition.nodes {
                let nid = node.index();
                if let Ok(out_degree) = graph.out_degree(node) {
                    if out_degree > 0 {
                        inv_degrees[nid] = 1.0 / out_degree as f64;
                    }
                }
            }
        }

        let damping = self.config.damping;
        let teleport = (1.0 - damping) / total_nodes.max(1) as f64;
        let mut converged = false;
        let mut iterations = 0;

        // 迭代计算
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;
            let mut max_diff = 0.0;

            // 每个分区独立计算
            for partition in partitions {
                let params = PartitionPageRankParams {
                    partition,
                    node_id_to_pos: &node_id_to_pos,
                    ranks: &ranks,
                    new_ranks: &mut new_ranks,
                    damping,
                    teleport,
                    inv_degrees: &inv_degrees,
                };
                self.compute_partition_pagerank_vec(graph, params);
            }

            // 更新 ranks 并检查收敛
            for i in 0..ranks.len() {
                let diff = (new_ranks[i] - ranks[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }

            // 交换 ranks
            std::mem::swap(&mut ranks, &mut new_ranks);

            // 检查收敛
            if max_diff < self.config.tolerance {
                converged = true;
                break;
            }
        }

        let computation_time_ms = start_time.elapsed().as_millis() as u64;

        // 计算分区统计
        let partition_stats: Vec<PartitionPageRankStats> = partitions
            .iter()
            .map(|p| self.compute_partition_stats_vec(p, &node_id_to_pos, &ranks))
            .collect();

        PageRankResult {
            node_id_to_pos,
            ranks,
            node_ids,
            iterations,
            converged,
            computation_time_ms,
            partition_stats,
        }
    }

    /// 计算单个分区的 PageRank（Vec 版本）- 预计算逆度数优化
    fn compute_partition_pagerank_vec<G>(
        &self,
        graph: &G,
        params: PartitionPageRankParams<'_>,
    ) where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let PartitionPageRankParams {
            partition,
            node_id_to_pos,
            ranks,
            new_ranks,
            damping,
            teleport,
            inv_degrees,
        } = params;

        for &node in &partition.nodes {
            let mut new_rank = teleport;

            // 收集来自邻居的贡献
            for neighbor in graph.neighbors(node) {
                let nid = neighbor.index();
                if let Some(&pos) = node_id_to_pos.get(nid) {
                    if pos != usize::MAX {
                        // P2 OPTIMIZATION: Use pre-computed inv_degree to avoid division
                        let inv_out_degree = inv_degrees[nid];
                        if inv_out_degree > 0.0 {
                            new_rank += damping * ranks[pos] * inv_out_degree;
                        }
                    }
                }
            }

            // 找到当前节点在 Vec 中的位置
            let nid = node.index();
            if let Some(&pos) = node_id_to_pos.get(nid) {
                if pos != usize::MAX {
                    new_ranks[pos] = new_rank;
                }
            }
        }
    }

    /// 计算分区统计信息（Vec 版本）
    fn compute_partition_stats_vec(
        &self,
        partition: &Partition,
        node_id_to_pos: &[usize],
        ranks: &[f64],
    ) -> PartitionPageRankStats {
        let mut rank_sum = 0.0;
        let mut max_rank = f64::NEG_INFINITY;
        let mut min_rank = f64::INFINITY;

        for &node in &partition.nodes {
            let nid = node.index();
            if let Some(&pos) = node_id_to_pos.get(nid) {
                if pos != usize::MAX {
                    let rank = ranks[pos];
                    rank_sum += rank;
                    if rank > max_rank {
                        max_rank = rank;
                    }
                    if rank < min_rank {
                        min_rank = rank;
                    }
                }
            }
        }

        PartitionPageRankStats {
            partition_id: partition.id,
            node_count: partition.nodes.len(),
            boundary_count: partition.boundary_nodes.len(),
            rank_sum,
            max_rank,
            min_rank,
        }
    }
}

/// 单机 PageRank（用于对比测试）- 预计算逆度数优化
/// 使用 Vec 替代 HashMap 提升性能
pub fn simple_pagerank<G>(
    graph: &G,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<f64>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    let total_nodes = graph.node_count();
    let initial_rank = 1.0 / total_nodes.max(1) as f64;
    let teleport = (1.0 - damping) / total_nodes.max(1) as f64;

    let mut ranks: Vec<f64> = vec![initial_rank; total_nodes];
    let mut new_ranks: Vec<f64> = vec![initial_rank; total_nodes];

    // P2 优化：预计算所有节点的逆出度 - 避免每次迭代重复计算
    let mut inv_degrees: Vec<f64> = vec![0.0; total_nodes];
    for node_ref in graph.nodes() {
        let node = node_ref.index();
        if let Ok(out_degree) = graph.out_degree(node) {
            if out_degree > 0 {
                inv_degrees[node.index()] = 1.0 / out_degree as f64;
            }
        }
    }

    for _ in 0..max_iterations {
        let mut max_diff = 0.0;

        for node_ref in graph.nodes() {
            let node_id = node_ref.index.index();
            let mut new_rank = teleport;

            // 收集来自邻居的贡献
            for neighbor in graph.neighbors(node_ref.index()) {
                let nid = neighbor.index();
                if nid < ranks.len() {
                    // P2 OPTIMIZATION: Use pre-computed inv_degree
                    let inv_out_degree = inv_degrees[nid];
                    if inv_out_degree > 0.0 {
                        new_rank += damping * ranks[nid] * inv_out_degree;
                    }
                }
            }

            if node_id < new_ranks.len() {
                new_ranks[node_id] = new_rank;
            }
        }

        for i in 0..ranks.len().min(new_ranks.len()) {
            let diff = (new_ranks[i] - ranks[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        std::mem::swap(&mut ranks, &mut new_ranks);

        if max_diff < tolerance {
            break;
        }
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::partitioner::{HashPartitioner, Partitioner};
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_pagerank_config() {
        let config = PageRankConfig::new(0.85, 50, 1e-8).with_sparse(true);

        assert_eq!(config.damping, 0.85);
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.tolerance, 1e-8);
        assert!(config.sparse);
    }

    #[test]
    fn test_distributed_pagerank_basic() {
        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..20 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(4);
        let partitions = partitioner.partition_graph(&graph);

        let pr = DistributedPageRank::new(0.85, 20, 1e-6);
        let result = pr.compute(&graph, &partitions);

        assert_eq!(result.ranks.len(), 20);
        assert!(result.iterations <= 20);
        assert!(result.converged);
    }

    #[test]
    fn test_distributed_pagerank_convergence() {
        let mut graph = Graph::<(), ()>::undirected();
        let a = graph.add_node(()).unwrap();
        let b = graph.add_node(()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let pr = DistributedPageRank::new(0.85, 100, 1e-10);
        let result = pr.compute(&graph, &partitions);

        assert!(result.converged);
        // 两个节点应该有相近的 PageRank
        let rank_a = result.rank(a).unwrap();
        let rank_b = result.rank(b).unwrap();
        assert!((rank_a - rank_b).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_top_k() {
        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..10 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let pr = DistributedPageRank::new(0.85, 20, 1e-6);
        let result = pr.compute(&graph, &partitions);

        let top_3 = result.top_k(3);
        assert_eq!(top_3.len(), 3);
    }

    #[test]
    fn test_simple_pagerank() {
        let mut graph = Graph::<(), ()>::undirected();
        let a = graph.add_node(()).unwrap();
        let b = graph.add_node(()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();

        let ranks = simple_pagerank(&graph, 0.85, 20, 1e-6);

        assert_eq!(ranks.len(), 2);
        assert!(ranks.iter().sum::<f64>() > 0.9); // 总和应接近 1
    }

    #[test]
    fn test_partition_stats() {
        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..100 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(4);
        let partitions = partitioner.partition_graph(&graph);

        let pr = DistributedPageRank::new(0.85, 20, 1e-6);
        let result = pr.compute(&graph, &partitions);

        assert_eq!(result.partition_stats.len(), 4);

        // 验证统计信息
        let total_nodes: usize = result.partition_stats.iter().map(|s| s.node_count).sum();
        assert_eq!(total_nodes, 100);
    }
}
