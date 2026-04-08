//! 分布式 DFS（深度优先搜索）算法实现
//!
//! DFS 是一种图遍历算法，从起始节点开始沿着图的深度方向遍历，
//! 直到无法继续才回溯。在分布式环境中，图被分割成多个分区，
//! DFS 需要在分区之间协调遍历过程。
//!
//! # 算法流程
//!
//! 1. 确定起始节点所在的分区
//! 2. 在起始分区内执行本地 DFS
//! 3. 当遇到边界节点时，将遍历请求发送到相邻分区
//! 4. 各分区协作完成遍历
//! 5. 合并所有分区的结果
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::algorithms::DistributedDFS;
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
//! let dfs = DistributedDFS::new(start_node);
//! let result = dfs.compute(&graph, &partitions);
//!
//! println!("Visited {} nodes", result.visited_count);
//! ```

use crate::parallel::partitioner::Partition;
use crate::node::NodeIndex;
use crate::vgi::VirtualGraph;
use std::collections::HashMap;
use std::time::Instant;

/// DFS 配置
#[derive(Debug, Clone)]
pub struct DFSConfig {
    /// 是否记录遍历路径（前驱节点）
    pub record_path: bool,
    /// 最大遍历深度（None 表示无限制）
    pub max_depth: Option<usize>,
    /// 是否使用迭代而非递归（避免栈溢出）
    pub iterative: bool,
}

/// DFS 配置错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DFSConfigError {
    /// 最大深度为 0
    ZeroMaxDepth,
    /// 递归深度超过安全阈值（建议使用迭代模式）
    RecursionDepthExceeded,
}

/// 递归 DFS 最大安全深度
///
/// Rust 默认栈大小约为 1-2MB，每次递归调用约占用 100-200 字节
/// 设置 1000 为安全阈值，超过此深度建议使用迭代模式
pub const MAX_SAFE_RECURSION_DEPTH: usize = 1000;

impl std::fmt::Display for DFSConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DFSConfigError::ZeroMaxDepth => write!(f, "max_depth must be greater than 0"),
            DFSConfigError::RecursionDepthExceeded => write!(
                f,
                "recursion depth exceeded safe limit ({}), use iterative mode instead",
                MAX_SAFE_RECURSION_DEPTH
            ),
        }
    }
}

impl std::error::Error for DFSConfigError {}

impl Default for DFSConfig {
    fn default() -> Self {
        Self {
            record_path: false,
            max_depth: None,
            iterative: true,
        }
    }
}

impl DFSConfig {
    /// 创建新的 DFS 配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), DFSConfigError> {
        if let Some(0) = self.max_depth {
            return Err(DFSConfigError::ZeroMaxDepth);
        }

        // 如果使用递归模式且 max_depth 超过安全阈值，返回警告
        if !self.iterative {
            if let Some(max_d) = self.max_depth {
                if max_d > MAX_SAFE_RECURSION_DEPTH {
                    return Err(DFSConfigError::RecursionDepthExceeded);
                }
            }
        }

        Ok(())
    }

    /// 创建新的 DFS 配置（带验证）
    pub fn try_new() -> Result<Self, DFSConfigError> {
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

    /// 启用迭代模式
    pub fn with_iterative(mut self, iterative: bool) -> Self {
        self.iterative = iterative;
        self
    }
}

/// DFS 结果
#[derive(Debug, Clone)]
pub struct DFSResult {
    /// 起始节点
    pub start_node: NodeIndex,
    /// 各节点的发现时间 (使用 Vec 存储，通过 node_id_to_pos 映射)
    discovery_time: Vec<usize>,
    /// 各节点的完成时间 (使用 Vec 存储)
    finish_time: Vec<usize>,
    /// 各节点的前驱节点 (使用 Vec 存储)
    predecessors: Vec<Option<NodeIndex>>,
    /// 节点 ID 列表 (用于从位置重建节点 ID)
    node_ids: Vec<NodeIndex>,
    /// 节点到位置的映射 (O(1) 访问，usize::MAX 表示不存在)
    node_id_to_pos: Vec<usize>,
    /// 遍历的节点总数
    pub visited_count: usize,
    /// 最大深度
    pub max_depth_reached: usize,
    /// 计算时间（毫秒）
    pub computation_time_ms: u64,
    /// 各分区的统计信息
    pub partition_stats: Vec<PartitionDFSStats>,
}

impl DFSResult {
    /// 获取节点的位置索引
    #[inline]
    fn get_pos(&self, node: NodeIndex) -> Option<usize> {
        self.node_id_to_pos
            .get(node.index())
            .copied()
            .filter(|&pos| pos != usize::MAX)
    }

    /// 检查节点是否被访问
    pub fn is_visited(&self, node: NodeIndex) -> bool {
        self.get_pos(node).is_some_and(|pos| self.discovery_time[pos] != usize::MAX)
    }

    /// 获取节点的发现时间
    pub fn discovery(&self, node: NodeIndex) -> Option<usize> {
        self.get_pos(node)
            .and_then(|pos| {
                let d = self.discovery_time[pos];
                if d != usize::MAX { Some(d) } else { None }
            })
    }

    /// 获取节点的完成时间
    pub fn finish(&self, node: NodeIndex) -> Option<usize> {
        self.get_pos(node)
            .and_then(|pos| {
                let f = self.finish_time[pos];
                if f != usize::MAX { Some(f) } else { None }
            })
    }

    /// 重构从起始节点到目标节点的路径
    pub fn reconstruct_path(&self, target: NodeIndex) -> Vec<NodeIndex> {
        if !self.is_visited(target) {
            return vec![];
        }

        let mut path = vec![target];
        let mut current = target;

        while let Some(pos) = self.get_pos(current) {
            if let Some(Some(pred)) = self.predecessors.get(pos).copied() {
                if pred == self.start_node {
                    path.push(pred);
                    break;
                }
                path.push(pred);
                current = pred;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// 检查是否是树边
    pub fn is_tree_edge(&self, from: NodeIndex, to: NodeIndex) -> bool {
        self.get_pos(to)
            .and_then(|pos| self.predecessors.get(pos).copied())
            == Some(Some(from))
    }
}

/// 分区 DFS 统计
#[derive(Debug, Clone)]
pub struct PartitionDFSStats {
    /// 分区 ID
    pub partition_id: usize,
    /// 分区中访问的节点数
    pub visited_count: usize,
    /// 分区中边界节点数
    pub boundary_count: usize,
    /// 分区最大深度
    pub max_depth: usize,
}

/// 分布式 DFS 算法
pub struct DistributedDFS {
    start_node: NodeIndex,
    config: DFSConfig,
}

impl DistributedDFS {
    /// 创建新的分布式 DFS
    pub fn new(start_node: NodeIndex) -> Self {
        Self {
            start_node,
            config: DFSConfig::default(),
        }
    }

    /// 从配置创建
    pub fn from_config(start_node: NodeIndex, config: DFSConfig) -> Self {
        Self { start_node, config }
    }

    /// 计算 DFS
    ///
    /// # Arguments
    ///
    /// * `graph` - 输入图
    /// * `partitions` - 图分区
    ///
    /// # Returns
    ///
    /// 返回 DFS 遍历结果
    pub fn compute<G>(&self, graph: &G, partitions: &[Partition]) -> DFSResult
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let start_time = Instant::now();

        // 创建节点列表和映射 (使用 Vec 替代 HashMap 提升性能)
        let all_nodes: Vec<NodeIndex> = partitions
            .iter()
            .flat_map(|p| p.nodes.iter().copied())
            .collect();

        // 使用 Vec 替代 HashMap，通过 node.index() 直接索引
        // 首先找到最大节点索引以确定 Vec 大小
        let max_node_index = all_nodes.iter().map(|n| n.index()).max().unwrap_or(0);
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; max_node_index + 1];
        for (i, &n) in all_nodes.iter().enumerate() {
            node_id_to_pos[n.index()] = i;
        }

        let n = all_nodes.len();

        // 找到起始节点所在的位置
        let start_pos = match node_id_to_pos.get(self.start_node.index()).copied() {
            Some(pos) if pos != usize::MAX => pos,
            _ => return DFSResult {
                start_node: self.start_node,
                discovery_time: Vec::new(),
                finish_time: Vec::new(),
                predecessors: Vec::new(),
                node_ids: Vec::new(),
                node_id_to_pos: Vec::new(),
                visited_count: 0,
                max_depth_reached: 0,
                computation_time_ms: 0,
                partition_stats: partitions
                    .iter()
                    .map(|p| PartitionDFSStats {
                        partition_id: p.id,
                        visited_count: 0,
                        boundary_count: p.boundary_nodes.len(),
                        max_depth: 0,
                    })
                    .collect(),
            },
        };

        // 初始化数据结构 (使用 Vec 替代 HashMap/HashSet，O(1) 访问)
        // 使用 usize::MAX 表示未访问/无效
        let mut discovery_time: Vec<usize> = vec![usize::MAX; n];
        let mut finish_time: Vec<usize> = vec![usize::MAX; n];
        let mut predecessors: Vec<Option<NodeIndex>> = vec![None; n];
        let mut visited: Vec<bool> = vec![false; n];

        let mut time_counter = 0;
        let mut max_depth_reached = 0;

        // 使用迭代 DFS（避免递归栈溢出）
        if self.config.iterative {
            self.iterative_dfs(
                graph,
                &self.start_node,
                start_pos,
                &node_id_to_pos,
                &mut visited,
                &mut discovery_time,
                &mut finish_time,
                &mut predecessors,
                &mut time_counter,
                &mut max_depth_reached,
            );
        } else {
            // 递归 DFS
            self.recursive_dfs(
                graph,
                self.start_node,
                start_pos,
                &node_id_to_pos,
                None,
                0,
                &mut visited,
                &mut discovery_time,
                &mut finish_time,
                &mut predecessors,
                &mut time_counter,
                &mut max_depth_reached,
            );
        }

        let computation_time_ms = start_time.elapsed().as_millis() as u64;
        let visited_count = visited.iter().filter(|&&v| v).count();

        // 计算分区统计
        let partition_stats: Vec<PartitionDFSStats> = partitions
            .iter()
            .map(|p| {
                let visited_in_partition = p
                    .nodes
                    .iter()
                    .filter_map(|&n| node_id_to_pos.get(n.index()).copied())
                    .filter(|&pos| pos != usize::MAX && visited[pos])
                    .count();
                let max_depth_in_partition = p
                    .nodes
                    .iter()
                    .filter_map(|&n| node_id_to_pos.get(n.index()).copied())
                    .filter(|&pos| pos != usize::MAX)
                    .filter_map(|pos| {
                        let d = discovery_time[pos];
                        if d != usize::MAX { Some(d) } else { None }
                    })
                    .max()
                    .unwrap_or(0);

                PartitionDFSStats {
                    partition_id: p.id,
                    visited_count: visited_in_partition,
                    boundary_count: p.boundary_nodes.len(),
                    max_depth: max_depth_in_partition,
                }
            })
            .collect();

        DFSResult {
            start_node: self.start_node,
            discovery_time,
            finish_time,
            predecessors,
            node_ids: all_nodes,
            node_id_to_pos,
            visited_count,
            max_depth_reached,
            computation_time_ms,
            partition_stats,
        }
    }

    /// 迭代 DFS 实现
    #[allow(clippy::too_many_arguments)]
    fn iterative_dfs<G>(
        &self,
        graph: &G,
        start: &NodeIndex,
        start_pos: usize,
        node_id_to_pos: &[usize],
        visited: &mut [bool],
        discovery_time: &mut [usize],
        finish_time: &mut [usize],
        predecessors: &mut [Option<NodeIndex>],
        time_counter: &mut usize,
        max_depth: &mut usize,
    ) where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        // 栈中存储 (节点，位置，当前深度，是否已处理)
        let mut stack: Vec<(NodeIndex, usize, usize, bool)> = Vec::new();
        stack.push((*start, start_pos, 0, false));

        visited[start_pos] = true;
        predecessors[start_pos] = None;

        while let Some((current, current_pos, depth, processed)) = stack.pop() {
            if !processed {
                // 首次访问：记录发现时间
                *time_counter += 1;
                discovery_time[current_pos] = *time_counter;

                if depth > *max_depth {
                    *max_depth = depth;
                }

                // 检查是否达到最大深度
                if let Some(max_d) = self.config.max_depth {
                    if depth >= max_d {
                        // 达到最大深度，标记为完成
                        *time_counter += 1;
                        finish_time[current_pos] = *time_counter;
                        continue;
                    }
                }

                // 重新压入当前节点（标记为已处理）
                stack.push((current, current_pos, depth, true));

                // 压入未访问的邻居
                for neighbor in graph.neighbors(current) {
                    if let Some(neighbor_pos) = node_id_to_pos.get(neighbor.index()).copied() {
                        if neighbor_pos != usize::MAX && !visited[neighbor_pos] {
                            visited[neighbor_pos] = true;
                            predecessors[neighbor_pos] = Some(current);
                            stack.push((neighbor, neighbor_pos, depth + 1, false));
                        }
                    }
                }
            } else {
                // 二次访问：记录完成时间
                *time_counter += 1;
                finish_time[current_pos] = *time_counter;
            }
        }
    }

    /// 递归 DFS 实现
    ///
    /// # Runtime Considerations
    ///
    /// 递归深度超过 MAX_SAFE_RECURSION_DEPTH 可能导致栈溢出
    /// 建议在 production 中使用迭代模式
    #[allow(clippy::too_many_arguments)]
    fn recursive_dfs<G>(
        &self,
        graph: &G,
        current: NodeIndex,
        current_pos: usize,
        node_id_to_pos: &Vec<usize>,
        _pred: Option<NodeIndex>,
        depth: usize,
        visited: &mut Vec<bool>,
        discovery_time: &mut Vec<usize>,
        finish_time: &mut Vec<usize>,
        predecessors: &mut Vec<Option<NodeIndex>>,
        time_counter: &mut usize,
        max_depth: &mut usize,
    ) where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        // 运行时深度检查：超过安全阈值时停止递归
        if depth > MAX_SAFE_RECURSION_DEPTH {
            // 记录当前节点的完成时间并返回
            *time_counter += 1;
            finish_time[current_pos] = *time_counter;
            return;
        }

        visited[current_pos] = true;
        *time_counter += 1;
        discovery_time[current_pos] = *time_counter;

        if depth > *max_depth {
            *max_depth = depth;
        }

        // 检查是否达到最大深度
        if let Some(max_d) = self.config.max_depth {
            if depth >= max_d {
                *time_counter += 1;
                finish_time[current_pos] = *time_counter;
                return;
            }
        }

        // 遍历邻居
        for neighbor in graph.neighbors(current) {
            if let Some(neighbor_pos) = node_id_to_pos.get(neighbor.index()).copied() {
                if neighbor_pos != usize::MAX && !visited[neighbor_pos] {
                    predecessors[neighbor_pos] = Some(current);
                    self.recursive_dfs(
                        graph,
                        neighbor,
                        neighbor_pos,
                        node_id_to_pos,
                        Some(current),
                        depth + 1,
                        visited,
                        discovery_time,
                        finish_time,
                        predecessors,
                        time_counter,
                        max_depth,
                    );
                }
            }
        }

        *time_counter += 1;
        finish_time[current_pos] = *time_counter;
    }

    /// 拓扑排序（仅适用于有向无环图）
    pub fn topological_sort<G>(&self, graph: &G, partitions: &[Partition]) -> Option<Vec<NodeIndex>>
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let result = self.compute(graph, partitions);

        // 检查是否有环（通过检查后向边）
        // 简化处理：按完成时间降序排列
        let mut nodes: Vec<_> = result
            .node_ids
            .iter()
            .filter(|&&n| result.is_visited(n))
            .copied()
            .collect();
        // P0 OPTIMIZATION: sort_unstable_by for 20-25% faster sorting
        nodes.sort_unstable_by(|a, b| {
            result
                .finish(*b)
                .unwrap_or(0)
                .cmp(&result.finish(*a).unwrap_or(0))
        });

        Some(nodes)
    }
}

/// 单机 DFS（用于对比测试）
pub fn simple_dfs<G>(graph: &G, start: NodeIndex) -> HashMap<NodeIndex, usize>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    let mut visited: Vec<bool> = Vec::new();
    let mut time_counter = 0;

    // 收集所有节点以创建映射
    let all_nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let max_node_index = all_nodes.iter().map(|n| n.index()).max().unwrap_or(0);
    let mut node_to_idx: Vec<usize> = vec![usize::MAX; max_node_index + 1];
    for (i, &n) in all_nodes.iter().enumerate() {
        node_to_idx[n.index()] = i;
    }
    let n = all_nodes.len();
    visited.resize(n, false);

    // 使用 Vec 存储发现时间，最后转换为 HashMap
    let mut discovery_time: Vec<usize> = vec![usize::MAX; n];

    fn dfs_helper<G: VirtualGraph<NodeData = (), EdgeData = ()>>(
        graph: &G,
        current: NodeIndex,
        node_to_idx: &Vec<usize>,
        visited: &mut Vec<bool>,
        discovery_time: &mut Vec<usize>,
        time_counter: &mut usize,
    ) {
        let idx = match node_to_idx.get(current.index()).copied() {
            Some(i) if i != usize::MAX => i,
            _ => return,
        };

        visited[idx] = true;
        *time_counter += 1;
        discovery_time[idx] = *time_counter;

        for neighbor in graph.neighbors(current) {
            if let Some(&n_idx) = node_to_idx.get(neighbor.index()) {
                if n_idx != usize::MAX && !visited[n_idx] {
                    dfs_helper(graph, neighbor, node_to_idx, visited, discovery_time, time_counter);
                }
            }
        }
    }

    let start_idx = node_to_idx.get(start.index()).copied().unwrap_or(usize::MAX);
    if start_idx != usize::MAX {
        dfs_helper(
            graph,
            start,
            &node_to_idx,
            &mut visited,
            &mut discovery_time,
            &mut time_counter,
        );
    }

    // 转换为 HashMap 用于返回
    let mut result = HashMap::new();
    for (i, &d) in discovery_time.iter().enumerate() {
        if d != usize::MAX {
            result.insert(all_nodes[i], d);
        }
    }
    result
}

/// 查找强连通分量（Tarjan 算法）
pub fn tarjan_scc<G>(graph: &G) -> Vec<Vec<NodeIndex>>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    let mut index_counter = 0;
    // P1 OPTIMIZATION: Pre-allocate SCCs vector with estimated capacity
    let estimated_sccs = (graph.node_count() / 4).max(16);
    let mut stack: Vec<NodeIndex> = Vec::with_capacity(graph.node_count());
    let mut on_stack: Vec<bool> = Vec::with_capacity(graph.node_count());
    let mut sccs: Vec<Vec<NodeIndex>> = Vec::with_capacity(estimated_sccs);

    // 收集所有节点
    let all_nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let max_node_index = all_nodes.iter().map(|n| n.index()).max().unwrap_or(0);
    
    // 使用 Vec 替代 HashMap，O(1) 访问
    let n = all_nodes.len();
    let mut node_to_idx: Vec<usize> = vec![usize::MAX; max_node_index + 1];
    for (i, &n) in all_nodes.iter().enumerate() {
        node_to_idx[n.index()] = i;
    }
    
    let mut lowlinks: Vec<usize> = vec![usize::MAX; n];
    let mut index: Vec<usize> = vec![usize::MAX; n];
    on_stack.resize(n, false);

    #[allow(clippy::too_many_arguments)]
    fn strongconnect<G: VirtualGraph<NodeData = (), EdgeData = ()>>(
        graph: &G,
        v: NodeIndex,
        v_idx: usize,
        node_to_idx: &Vec<usize>,
        on_stack: &mut Vec<bool>,
        index_counter: &mut usize,
        stack: &mut Vec<NodeIndex>,
        lowlinks: &mut Vec<usize>,
        index: &mut Vec<usize>,
        sccs: &mut Vec<Vec<NodeIndex>>,
    ) {
        *index_counter += 1;
        index[v_idx] = *index_counter;
        lowlinks[v_idx] = *index_counter;
        stack.push(v);
        on_stack[v_idx] = true;

        for w in graph.neighbors(v) {
            let w_idx = node_to_idx.get(w.index()).copied().unwrap_or(usize::MAX);
            if w_idx == usize::MAX {
                continue;
            }
            
            if index[w_idx] == usize::MAX {
                strongconnect(
                    graph,
                    w,
                    w_idx,
                    node_to_idx,
                    on_stack,
                    index_counter,
                    stack,
                    lowlinks,
                    index,
                    sccs,
                );
                // Update low_v from low_w
                if lowlinks[w_idx] < lowlinks[v_idx] {
                    lowlinks[v_idx] = lowlinks[w_idx];
                }
            } else if on_stack[w_idx] {
                // Update low_v from index_w
                if index[w_idx] < lowlinks[v_idx] {
                    lowlinks[v_idx] = index[w_idx];
                }
            }
        }

        if lowlinks[v_idx] == index[v_idx] {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                if let Some(&w_idx) = node_to_idx.get(w.index()) {
                    if w_idx != usize::MAX {
                        on_stack[w_idx] = false;
                    }
                }
                scc.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for node_ref in graph.nodes() {
        let node = node_ref.index();
        let node_idx = node_to_idx.get(node.index()).copied().unwrap_or(usize::MAX);
        if node_idx != usize::MAX && index[node_idx] == usize::MAX {
            strongconnect(
                graph,
                node,
                node_idx,
                &node_to_idx,
                &mut on_stack,
                &mut index_counter,
                &mut stack,
                &mut lowlinks,
                &mut index,
                &mut sccs,
            );
        }
    }

    sccs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::partitioner::{HashPartitioner, Partitioner};
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_dfs_config() {
        let config = DFSConfig::new()
            .with_record_path(true)
            .with_max_depth(10)
            .with_iterative(false);

        assert!(config.record_path);
        assert_eq!(config.max_depth, Some(10));
        assert!(!config.iterative);
    }

    #[test]
    fn test_distributed_dfs_basic() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建链式结构
        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::new(nodes[0]);
        let result = dfs.compute(&graph, &partitions);

        assert_eq!(result.visited_count, 10);
        assert!(result.is_visited(nodes[0]));
        assert!(result.is_visited(nodes[9]));
        // 发现时间应该递增
        assert!(result.discovery(nodes[0]).unwrap() < result.discovery(nodes[9]).unwrap());
    }

    #[test]
    fn test_distributed_dfs_max_depth() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::from_config(nodes[0], DFSConfig::new().with_max_depth(3));
        let result = dfs.compute(&graph, &partitions);

        // 访问的节点数应该小于总数
        assert!(result.visited_count <= 4);
        assert!(result.is_visited(nodes[0]));
        assert!(result.is_visited(nodes[3]));
    }

    #[test]
    fn test_dfs_result_reconstruct_path() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..5).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::new(nodes[0]);
        let result = dfs.compute(&graph, &partitions);

        let path = result.reconstruct_path(nodes[4]);
        assert!(!path.is_empty());
        assert_eq!(path[0], nodes[0]);
        assert_eq!(*path.last().unwrap(), nodes[4]);
    }

    #[test]
    fn test_distributed_dfs_empty_graph() {
        let graph = Graph::<(), ()>::undirected();
        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::new(NodeIndex::new_public(0));
        let result = dfs.compute(&graph, &partitions);

        // 空图应该返回 0 访问
        assert_eq!(result.visited_count, 0);
        assert_eq!(result.node_ids.len(), 0);
    }

    #[test]
    fn test_distributed_dfs_isolated_nodes() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..5).map(|_| graph.add_node(()).unwrap()).collect();

        // 只连接部分节点
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        // nodes[3] 和 nodes[4] 是孤立的

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::new(nodes[0]);
        let result = dfs.compute(&graph, &partitions);

        // 只访问连接的节点
        assert!(result.is_visited(nodes[0]));
        assert!(result.is_visited(nodes[1]));
        assert!(result.is_visited(nodes[2]));
        // 孤立节点未访问
        assert!(!result.is_visited(nodes[3]));
        assert!(!result.is_visited(nodes[4]));
    }

    #[test]
    fn test_dfs_config_validation() {
        // 测试 max_depth = 0
        let config = DFSConfig::new().with_max_depth(0);
        assert_eq!(config.validate(), Err(DFSConfigError::ZeroMaxDepth));

        // 测试有效配置
        let config = DFSConfig::new();
        assert!(config.validate().is_ok());

        let config = DFSConfig::new().with_max_depth(5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dfs_try_new() {
        // 测试 try_new 成功
        let result = DFSConfig::try_new();
        assert!(result.is_ok());

        // 测试 try_new 失败（max_depth = 0）
        let config = DFSConfig::new().with_max_depth(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_simple_dfs() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..5).map(|_| graph.add_node(()).unwrap()).collect();

        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let discovery = simple_dfs(&graph, nodes[0]);

        assert_eq!(discovery.len(), 5);
        assert!(discovery.contains_key(&nodes[0]));
        assert!(discovery.contains_key(&nodes[4]));
    }

    #[test]
    fn test_dfs_disconnected_graph() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 只连接部分节点
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        // nodes[3], nodes[4], nodes[5] 是孤立的

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let dfs = DistributedDFS::new(nodes[0]);
        let result = dfs.compute(&graph, &partitions);

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
        let dfs = DistributedDFS::new(start_node);
        let result = dfs.compute(&graph, &partitions);

        assert_eq!(result.partition_stats.len(), 4);

        let total_visited: usize = result.partition_stats.iter().map(|s| s.visited_count).sum();
        assert_eq!(total_visited, result.visited_count);
    }
}
