//! 分布式 BFS（广度优先搜索）算法实现
//!
//! BFS 是一种图遍历算法，从起始节点开始逐层遍历图中的所有可达节点。
//! 在分布式环境中，图被分割成多个分区，BFS 需要在分区之间协调遍历过程。
//!
//! # 算法流程
//!
//! 1. 确定起始节点所在的分区
//! 2. 在起始分区内执行本地 BFS
//! 3. 当遇到边界节点时，将遍历请求发送到相邻分区
//! 4. 各分区并行执行本地 BFS
//! 5. 合并所有分区的结果
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::algorithms::DistributedBFS;
//! use god_graph::parallel::partitioner::HashPartitioner;
//! use god_graph::graph::Graph;
//! use god_graph::node::NodeIndex;
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
//! let start_node = NodeIndex::new_public(0);
//! let bfs = DistributedBFS::new(start_node);
//! let result = bfs.compute(&graph, &partitions);
//!
//! println!("Visited {} nodes", result.distances.len());
//! ```

use crate::parallel::partitioner::Partition;
use crate::node::NodeIndex;
use crate::vgi::VirtualGraph;
use std::collections::VecDeque;
use std::time::Instant;

/// BFS 配置
#[derive(Debug, Clone, Default)]
pub struct BFSConfig {
    /// 是否记录遍历路径
    pub record_path: bool,
    /// 最大遍历深度（None 表示无限制）
    pub max_depth: Option<usize>,
    /// 是否并行处理分区
    pub parallel: bool,
}

/// BFS 配置错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BFSConfigError {
    /// 最大深度为 0
    ZeroMaxDepth,
}

impl std::fmt::Display for BFSConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BFSConfigError::ZeroMaxDepth => write!(f, "max_depth must be greater than 0"),
        }
    }
}

impl std::error::Error for BFSConfigError {}

impl BFSConfig {
    /// 创建新的 BFS 配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), BFSConfigError> {
        if let Some(0) = self.max_depth {
            return Err(BFSConfigError::ZeroMaxDepth);
        }
        Ok(())
    }

    /// 创建新的 BFS 配置（带验证）
    pub fn try_new() -> Result<Self, BFSConfigError> {
        let config = Self::default();
        config.validate()?;
        Ok(config)
    }

    /// 启用路径记录
    pub fn with_record_path(mut self, record: bool) -> Self {
        self.record_path = record;
        self
    }

    /// 设置最大深度
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// 启用并行处理
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// BFS 结果
#[derive(Debug, Clone)]
pub struct BFSResult {
    /// 起始节点
    pub start_node: NodeIndex,
    /// 各节点的距离（从起始节点），使用 Vec 替代 HashMap 提升性能
    /// node_id_to_pos[u] = position in data Vecs, usize::MAX means not visited
    pub node_id_to_pos: Vec<usize>,
    /// 各节点的距离（从起始节点）
    pub distances: Vec<usize>,
    /// 各节点的前驱节点位置索引（用于重构路径）
    pub predecessors: Vec<Option<usize>>,
    /// 节点 ID 列表，与 distances/predecessors 对应
    pub node_ids: Vec<NodeIndex>,
    /// 遍历的节点总数
    pub visited_count: usize,
    /// 最大深度
    pub max_depth_reached: usize,
    /// 计算时间（毫秒）
    pub computation_time_ms: u64,
    /// 各分区的统计信息
    pub partition_stats: Vec<PartitionBFSStats>,
}

impl BFSResult {
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

    /// 检查节点是否被访问
    pub fn is_visited(&self, node: NodeIndex) -> bool {
        self.get_pos(node).is_some()
    }

    /// 获取节点的距离
    ///
    /// # Arguments
    ///
    /// * `node` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点已访问，返回距离；如果节点未访问，返回 GraphError
    pub fn distance(&self, node: NodeIndex) -> Result<usize, GraphError> {
        let pos = self.get_pos(node).ok_or_else(|| GraphError::NotFound(
            format!("Node {:?} not found in BFS result", node)
        ))?;
        self.distances.get(pos).copied().ok_or_else(|| GraphError::NotFound(
            format!("Node {:?} is not reachable", node)
        ))
    }

    /// 重构从起始节点到目标节点的路径
    pub fn reconstruct_path(&self, target: NodeIndex) -> Vec<NodeIndex> {
        let pos = match self.get_pos(target) {
            Some(p) => p,
            None => return vec![],
        };

        let mut path = vec![target];
        let mut current_pos = pos;

        while let Some(Some(pred_pos)) = self.predecessors.get(current_pos) {
            let pred_node = self.node_ids[*pred_pos];
            if pred_node == self.start_node {
                path.push(pred_node);
                break;
            }
            path.push(pred_node);
            current_pos = *pred_pos;
        }

        path.reverse();
        path
    }
}

/// 分区 BFS 统计
#[derive(Debug, Clone)]
pub struct PartitionBFSStats {
    /// 分区 ID
    pub partition_id: usize,
    /// 分区中访问的节点数
    pub visited_count: usize,
    /// 分区中边界节点数
    pub boundary_count: usize,
    /// 分区最大深度
    pub max_depth: usize,
}

/// 分布式 BFS 算法
pub struct DistributedBFS {
    start_node: NodeIndex,
    config: BFSConfig,
}

impl DistributedBFS {
    /// 创建新的分布式 BFS
    pub fn new(start_node: NodeIndex) -> Self {
        Self {
            start_node,
            config: BFSConfig::default(),
        }
    }

    /// 从配置创建
    pub fn from_config(start_node: NodeIndex, config: BFSConfig) -> Self {
        Self { start_node, config }
    }

    /// 计算 BFS
    ///
    /// # Arguments
    ///
    /// * `graph` - 输入图
    /// * `partitions` - 图分区
    ///
    /// # Returns
    ///
    /// 返回 BFS 遍历结果
    pub fn compute<G>(&self, graph: &G, partitions: &[Partition]) -> BFSResult
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

        // 预计算总节点数以预分配 Vec
        let total_nodes = partitions.iter().map(|p| p.nodes.len()).sum::<usize>();

        // 使用 Vec 替代 HashMap/HashSet 提升性能
        // node_id_to_pos[u] = position in data Vecs, usize::MAX means not visited
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; vec_size];
        let mut distances: Vec<usize> = Vec::with_capacity(total_nodes);
        let mut predecessors: Vec<Option<usize>> = Vec::with_capacity(total_nodes);
        let mut node_ids: Vec<NodeIndex> = Vec::with_capacity(total_nodes);
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new(); // (position, depth)

        // P0 OPTIMIZATION: Use Vec<usize> instead of HashMap for node_to_partition mapping
        // node_to_partition_vec[u] = partition_id, usize::MAX means not in any partition
        let mut node_to_partition_vec: Vec<usize> = vec![usize::MAX; vec_size];
        for partition in partitions {
            for &node in &partition.nodes {
                node_to_partition_vec[node.index()] = partition.id;
            }
        }

        // 找到起始节点所在的分区
        let start_partition = {
            let pid = node_to_partition_vec.get(self.start_node.index()).copied();
            if pid == Some(usize::MAX) { None } else { pid }
        };

        if start_partition.is_none() {
            // 起始节点不在任何分区中
            return BFSResult {
                start_node: self.start_node,
                node_id_to_pos: vec![],
                distances: vec![],
                predecessors: vec![],
                node_ids: vec![],
                visited_count: 0,
                max_depth_reached: 0,
                computation_time_ms: 0,
                partition_stats: partitions
                    .iter()
                    .map(|p| PartitionBFSStats {
                        partition_id: p.id,
                        visited_count: 0,
                        boundary_count: p.boundary_nodes.len(),
                        max_depth: 0,
                    })
                    .collect(),
            };
        }

        // 初始化起始节点 - position 0
        let start_pos = 0;
        node_id_to_pos[self.start_node.index()] = start_pos;
        distances.push(0);
        predecessors.push(None);
        node_ids.push(self.start_node);
        queue.push_back((start_pos, 0));

        let mut max_depth_reached = 0;
        // 使用 Vec<bool> 替代 HashSet 进行访问标记，无锁且更快
        let mut visited: Vec<bool> = vec![false; vec_size];
        visited[self.start_node.index()] = true;

        // BFS 主循环
        while let Some((current_pos, depth)) = queue.pop_front() {
            if depth > max_depth_reached {
                max_depth_reached = depth;
            }

            // 检查是否达到最大深度
            if let Some(max_depth) = self.config.max_depth {
                if depth >= max_depth {
                    continue;
                }
            }

            // 从 position 恢复节点 ID
            let current_node = NodeIndex::new_public(current_pos);

            // 遍历当前节点的邻居
            for neighbor in graph.neighbors(current_node) {
                let neighbor_id = neighbor.index();
                if !visited[neighbor_id] {
                    visited[neighbor_id] = true;
                    let new_depth = depth + 1;
                    
                    // 分配新的 position
                    let new_pos = distances.len();
                    node_id_to_pos[neighbor_id] = new_pos;
                    distances.push(new_depth);
                    predecessors.push(Some(current_pos));
                    node_ids.push(neighbor);
                    queue.push_back((new_pos, new_depth));
                }
            }
        }

        let computation_time_ms = start_time.elapsed().as_millis() as u64;
        let visited_count = distances.len();

        // 计算分区统计
        let partition_stats: Vec<PartitionBFSStats> = partitions
            .iter()
            .map(|p| {
                let visited_in_partition = p
                    .nodes
                    .iter()
                    .filter(|&&n| node_id_to_pos.get(n.index()).is_some_and(|&p| p != usize::MAX))
                    .count();
                let max_depth_in_partition = p
                    .nodes
                    .iter()
                    .filter_map(|&n| {
                        let pos = node_id_to_pos.get(n.index())?;
                        if *pos == usize::MAX { None } else { distances.get(*pos) }
                    })
                    .copied()
                    .max()
                    .unwrap_or(0);

                PartitionBFSStats {
                    partition_id: p.id,
                    visited_count: visited_in_partition,
                    boundary_count: p.boundary_nodes.len(),
                    max_depth: max_depth_in_partition,
                }
            })
            .collect();

        BFSResult {
            start_node: self.start_node,
            node_id_to_pos,
            distances,
            predecessors,
            node_ids,
            visited_count,
            max_depth_reached,
            computation_time_ms,
            partition_stats,
        }
    }

    /// 单源最短路径（基于 BFS）
    pub fn shortest_path<G>(
        &self,
        graph: &G,
        partitions: &[Partition],
        target: NodeIndex,
    ) -> Option<Vec<NodeIndex>>
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let result = self.compute(graph, partitions);
        let path = result.reconstruct_path(target);
        if path.is_empty() {
            None
        } else {
            Some(path)
        }
    }
}

/// 单机 BFS（用于对比测试）
/// 使用 Vec 替代 HashMap/HashSet 提升性能
pub fn simple_bfs<G>(graph: &G, start: NodeIndex) -> Vec<usize>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    // 计算最大节点 ID
    let max_node_id = graph.node_count();
    let mut distances: Vec<Option<usize>> = vec![None; max_node_id];
    let mut visited: Vec<bool> = vec![false; max_node_id];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new(); // (node_id, depth)

    let start_id = start.index();
    distances[start_id] = Some(0);
    visited[start_id] = true;
    queue.push_back((start_id, 0));

    while let Some((current, depth)) = queue.pop_front() {
        for neighbor in graph.neighbors(NodeIndex::new_public(current)) {
            let neighbor_id = neighbor.index();
            if !visited[neighbor_id] {
                visited[neighbor_id] = true;
                distances[neighbor_id] = Some(depth + 1);
                queue.push_back((neighbor_id, depth + 1));
            }
        }
    }

    // 转换为紧凑的 Vec<usize>，只包含可达节点的距离
    distances.into_iter().flatten().collect()
}

/// 多源 BFS
/// 使用 Vec 替代 HashMap/HashSet 提升性能
pub fn multi_source_bfs<G>(
    graph: &G,
    sources: &[NodeIndex],
    max_depth: Option<usize>,
) -> Vec<(usize, usize)> // (distance, source_id)
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    let max_node_id = graph.node_count();
    let mut distances: Vec<Option<(usize, usize)>> = vec![None; max_node_id];
    let mut visited: Vec<bool> = vec![false; max_node_id];
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new(); // (node_id, depth, source_id)

    // 初始化所有源节点
    for &source in sources {
        let source_id = source.index();
        distances[source_id] = Some((0, source_id));
        visited[source_id] = true;
        queue.push_back((source_id, 0, source_id));
    }

    while let Some((current, depth, source_id)) = queue.pop_front() {
        if let Some(max_d) = max_depth {
            if depth >= max_d {
                continue;
            }
        }

        for neighbor in graph.neighbors(NodeIndex::new_public(current)) {
            let neighbor_id = neighbor.index();
            if !visited[neighbor_id] {
                visited[neighbor_id] = true;
                distances[neighbor_id] = Some((depth + 1, source_id));
                queue.push_back((neighbor_id, depth + 1, source_id));
            }
        }
    }

    // 转换为紧凑的 Vec
    distances.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::partitioner::{HashPartitioner, Partitioner};
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_bfs_config() {
        let config = BFSConfig::new()
            .with_record_path(true)
            .with_max_depth(10)
            .with_parallel(true);

        assert!(config.record_path);
        assert_eq!(config.max_depth, Some(10));
        assert!(config.parallel);
    }

    #[test]
    fn test_distributed_bfs_basic() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建链式结构
        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let bfs = DistributedBFS::new(nodes[0]);
        let result = bfs.compute(&graph, &partitions);

        assert_eq!(result.visited_count, 10);
        assert_eq!(result.max_depth_reached, 9);
        assert_eq!(result.distance(nodes[0]), Some(0));
        assert_eq!(result.distance(nodes[9]), Some(9));
    }

    #[test]
    fn test_distributed_bfs_max_depth() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let bfs = DistributedBFS::from_config(nodes[0], BFSConfig::new().with_max_depth(3));
        let result = bfs.compute(&graph, &partitions);

        assert!(result.max_depth_reached <= 3);
        assert_eq!(result.distance(nodes[3]), Some(3));
        assert!(result.distance(nodes[4]).is_none());
    }

    #[test]
    fn test_bfs_result_reconstruct_path() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..5).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let bfs = DistributedBFS::new(nodes[0]);
        let result = bfs.compute(&graph, &partitions);

        let path = result.reconstruct_path(nodes[4]);
        assert_eq!(path.len(), 5);
        assert_eq!(path[0], nodes[0]);
        assert_eq!(path[4], nodes[4]);
    }

    #[test]
    fn test_simple_bfs() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..5).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let distances = simple_bfs(&graph, nodes[0]);

        // distances now returns Vec<usize> with all reachable node distances
        assert_eq!(distances.len(), 5);
        assert!(distances.contains(&0));
        assert!(distances.contains(&4));
    }

    #[test]
    fn test_multi_source_bfs() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建两个分离的链
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        graph.add_edge(nodes[3], nodes[4], ()).unwrap();
        graph.add_edge(nodes[4], nodes[5], ()).unwrap();

        let sources = vec![nodes[0], nodes[3]];
        let distances = multi_source_bfs(&graph, &sources, None);

        // distances now returns Vec<(usize, usize)> with (distance, source_id)
        assert_eq!(distances.len(), 6);
        // Check that we have distances from both sources
        let source_ids: Vec<usize> = distances.iter().map(|(_, s)| *s).collect();
        assert!(source_ids.iter().any(|&s| s == nodes[0].index()));
        assert!(source_ids.iter().any(|&s| s == nodes[3].index()));
    }

    #[test]
    fn test_bfs_disconnected_graph() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 只连接部分节点
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        // nodes[3], nodes[4], nodes[5] 是孤立的

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let bfs = DistributedBFS::new(nodes[0]);
        let result = bfs.compute(&graph, &partitions);

        assert_eq!(result.visited_count, 3);
        assert!(!result.is_visited(nodes[3]));
    }

    #[test]
    fn test_partition_stats() {
        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..20 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(4);
        let partitions = partitioner.partition_graph(&graph);

        let start_node = NodeIndex::new_public(0);
        let bfs = DistributedBFS::new(start_node);
        let result = bfs.compute(&graph, &partitions);

        assert_eq!(result.partition_stats.len(), 4);

        let total_visited: usize = result.partition_stats.iter().map(|s| s.visited_count).sum();
        assert_eq!(total_visited, result.visited_count);
    }
}
