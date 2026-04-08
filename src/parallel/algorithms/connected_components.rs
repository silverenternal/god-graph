//! 分布式连通分量（Connected Components）算法实现
//!
//! 连通分量算法用于找出无向图中的所有连通分量。
//! 在分布式环境中，图被分割成多个分区，算法需要在分区之间协调以识别全局连通分量。
//!
//! # 算法流程
//!
//! 1. 每个分区独立计算本地连通分量
//! 2. 交换边界节点的分量信息
//! 3. 合并跨分区的连通分量
//! 4. 迭代直到分量不再变化
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::algorithms::DistributedConnectedComponents;
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
//! let cc = DistributedConnectedComponents::new();
//! let result = cc.compute(&graph, &partitions);
//!
//! println!("Found {} connected components", result.component_count);
//! ```

use crate::parallel::partitioner::Partition;
use crate::node::NodeIndex;
use crate::vgi::VirtualGraph;
use std::collections::HashMap;
use std::time::Instant;

/// 连通分量配置
#[derive(Debug, Clone)]
pub struct ConnectedComponentsConfig {
    /// 是否并行处理分区
    pub parallel: bool,
    /// 最大迭代次数
    pub max_iterations: usize,
}

/// 连通分量配置错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectedComponentsConfigError {
    /// 最大迭代次数为 0
    ZeroMaxIterations,
}

impl std::fmt::Display for ConnectedComponentsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectedComponentsConfigError::ZeroMaxIterations => {
                write!(f, "max_iterations must be greater than 0")
            }
        }
    }
}

impl std::error::Error for ConnectedComponentsConfigError {}

impl Default for ConnectedComponentsConfig {
    fn default() -> Self {
        Self {
            parallel: false,
            max_iterations: 100,
        }
    }
}

impl ConnectedComponentsConfig {
    /// 创建新的配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), ConnectedComponentsConfigError> {
        if self.max_iterations == 0 {
            return Err(ConnectedComponentsConfigError::ZeroMaxIterations);
        }
        Ok(())
    }

    /// 创建新的配置（带验证）
    pub fn try_new() -> Result<Self, ConnectedComponentsConfigError> {
        let config = Self::default();
        config.validate()?;
        Ok(config)
    }

    /// 启用并行处理
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

/// 连通分量结果
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    /// 各节点所属的连通分量 ID (使用 Vec 存储)
    components: Vec<usize>,
    /// 节点 ID 列表
    node_ids: Vec<NodeIndex>,
    /// 节点到位置的映射
    node_id_to_pos: HashMap<NodeIndex, usize>,
    /// 连通分量总数
    pub component_count: usize,
    /// 各分量的节点列表 (延迟构建)
    component_nodes: Option<HashMap<usize, Vec<NodeIndex>>>,
    /// 计算时间（毫秒）
    pub computation_time_ms: u64,
    /// 各分区的统计信息
    pub partition_stats: Vec<PartitionCCStats>,
}

impl ConnectedComponentsResult {
    /// 获取节点的位置索引
    #[inline]
    fn get_pos(&self, node: NodeIndex) -> Option<usize> {
        self.node_id_to_pos.get(&node).copied()
    }

    /// 获取节点所属的分量 ID
    pub fn component_id(&self, node: NodeIndex) -> Option<usize> {
        self.get_pos(node).and_then(|pos| {
            let c = self.components[pos];
            if c != usize::MAX { Some(c) } else { None }
        })
    }

    /// 检查两个节点是否在同一连通分量中
    pub fn is_connected(&self, a: NodeIndex, b: NodeIndex) -> bool {
        match (self.component_id(a), self.component_id(b)) {
            (Some(id_a), Some(id_b)) => id_a == id_b,
            _ => false,
        }
    }

    /// 获取指定分量的所有节点
    pub fn get_component_nodes(&mut self, component_id: usize) -> Option<&Vec<NodeIndex>> {
        // 延迟构建 component_nodes
        if self.component_nodes.is_none() {
            let mut map: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
            for (pos, &comp_id) in self.components.iter().enumerate() {
                if comp_id != usize::MAX {
                    map.entry(comp_id).or_default().push(self.node_ids[pos]);
                }
            }
            self.component_nodes = Some(map);
        }
        self.component_nodes.as_ref().unwrap().get(&component_id)
    }

    /// 获取最大的连通分量
    pub fn largest_component(&mut self) -> Option<(usize, usize)> {
        self.get_component_nodes(0); // 确保已构建
        self.component_nodes
            .as_ref()
            .unwrap()
            .iter()
            .max_by_key(|(_, nodes)| nodes.len())
            .map(|(&id, nodes)| (id, nodes.len()))
    }
}

/// 分区连通分量统计
#[derive(Debug, Clone)]
pub struct PartitionCCStats {
    /// 分区 ID
    pub partition_id: usize,
    /// 分区节点数
    pub node_count: usize,
    /// 分区中本地连通分量数
    pub local_component_count: usize,
    /// 分区边界节点数
    pub boundary_count: usize,
}

/// 分布式连通分量算法
pub struct DistributedConnectedComponents {
    config: ConnectedComponentsConfig,
}

impl Default for DistributedConnectedComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedConnectedComponents {
    /// 创建新的分布式连通分量算法
    pub fn new() -> Self {
        Self {
            config: ConnectedComponentsConfig::default(),
        }
    }

    /// 从配置创建
    pub fn from_config(config: ConnectedComponentsConfig) -> Self {
        Self { config }
    }

    /// 计算连通分量
    ///
    /// # Arguments
    ///
    /// * `graph` - 输入图
    /// * `partitions` - 图分区
    ///
    /// # Returns
    ///
    /// 返回连通分量计算结果
    pub fn compute<G>(&self, graph: &G, partitions: &[Partition]) -> ConnectedComponentsResult
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        let start_time = Instant::now();

        // 创建节点列表和映射 (使用 Vec 替代 HashMap 提升性能)
        let all_nodes: Vec<NodeIndex> = partitions
            .iter()
            .flat_map(|p| p.nodes.iter().copied())
            .collect();

        let n = all_nodes.len();

        // P0 OPTIMIZATION: Vec-based node indexing instead of HashMap
        // NodeIndex is a newtype over usize, enabling direct-mapped Vec
        // Uses usize::MAX sentinel for non-existent nodes (same as parallel PageRank)
        let max_index = all_nodes.iter().map(|ni| ni.index()).max().unwrap_or(0);
        let mut node_id_to_pos_vec = vec![usize::MAX; max_index + 1];
        for (pos, &node_idx) in all_nodes.iter().enumerate() {
            node_id_to_pos_vec[node_idx.index()] = pos;
        }

        // Helper closure for O(1) lookup with sentinel check
        let get_node_pos = |node: NodeIndex| -> Option<usize> {
            let pos = node_id_to_pos_vec.get(node.index()).copied();
            if pos == Some(usize::MAX) { None } else { pos }
        };

        // 初始化：每个节点一个分量 (使用 Vec 存储)
        let mut components: Vec<usize> = (0..n).collect();

        // 迭代合并连通分量
        let mut changed = true;
        let mut iteration = 0;

        while changed && iteration < self.config.max_iterations {
            changed = false;
            iteration += 1;

            // 每个分区内合并
            for partition in partitions {
                let local_changed =
                    self.merge_partition_components(graph, partition, &get_node_pos, &mut components);
                if local_changed {
                    changed = true;
                }
            }

            // 合并跨分区的分量（通过边界节点）
            let cross_changed = self.merge_cross_partition_components(
                graph,
                partitions,
                &get_node_pos,
                &mut components,
            );
            if cross_changed {
                changed = true;
            }

            // 重新编号分量（压缩到连续范围）
            if changed {
                self.renumber_components(&mut components);
            }
        }

        let computation_time_ms = start_time.elapsed().as_millis() as u64;

        // 计算分量总数
        let component_count = components.iter().filter(|&&c| c != usize::MAX).collect::<std::collections::HashSet<_>>().len();

        // 计算分区统计
        let partition_stats: Vec<PartitionCCStats> = partitions
            .iter()
            .map(|p| {
                let local_components: std::collections::HashSet<_> = p
                    .nodes
                    .iter()
                    .filter_map(|&n| get_node_pos(n))
                    .map(|pos| components[pos])
                    .filter(|&c| c != usize::MAX)
                    .collect();

                PartitionCCStats {
                    partition_id: p.id,
                    node_count: p.nodes.len(),
                    local_component_count: local_components.len(),
                    boundary_count: p.boundary_nodes.len(),
                }
            })
            .collect();

        // Build node_id_to_pos HashMap for API compatibility (stored in result)
        let node_id_to_pos: HashMap<NodeIndex, usize> = all_nodes
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();

        ConnectedComponentsResult {
            components,
            node_ids: all_nodes,
            node_id_to_pos,
            component_count,
            component_nodes: None, // 延迟构建
            computation_time_ms,
            partition_stats,
        }
    }

    /// 合并分区内的连通分量 - 使用 Union-Find with union-by-rank 优化
    fn merge_partition_components<G, F>(
        &self,
        graph: &G,
        partition: &Partition,
        get_node_pos: &F,
        components: &mut [usize],
    ) -> bool
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
        F: Fn(NodeIndex) -> Option<usize>,
    {
        let n = components.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<u8> = vec![0; n]; // Union by rank
        let mut changed = false;

        // Union-Find 查找（带路径压缩）
        #[inline]
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            let root = {
                let mut cur = x;
                while parent[cur] != cur {
                    cur = parent[cur];
                }
                cur
            };
            // 路径压缩
            while parent[x] != root {
                let next = parent[x];
                parent[x] = root;
                x = next;
            }
            root
        }

        // Union-Find 合并（union by rank - 将较小的树连接到较大的树）
        // 时间复杂度：O(α(n)) amortized，其中α是反阿克曼函数
        #[inline]
        fn union(parent: &mut [usize], rank: &mut [u8], x: usize, y: usize) -> bool {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                // Union by rank: attach smaller tree to larger tree
                if rank[root_x] < rank[root_y] {
                    parent[root_x] = root_y;
                } else if rank[root_x] > rank[root_y] {
                    parent[root_y] = root_x;
                } else {
                    // Same rank: arbitrarily choose one as root and increment rank
                    parent[root_y] = root_x;
                    rank[root_x] += 1;
                }
                true
            } else {
                false
            }
        }

        // 第一阶段：使用 Union-Find 收集所有需要合并的分量
        for &node in &partition.nodes {
            let node_pos = match get_node_pos(node) {
                Some(pos) => pos,
                None => continue,
            };
            let node_comp = components[node_pos];

            for neighbor in graph.neighbors(node) {
                if let Some(neighbor_pos) = get_node_pos(neighbor) {
                    let neighbor_comp = components[neighbor_pos];
                    if node_comp != neighbor_comp {
                        changed |= union(&mut parent, &mut rank, node_comp, neighbor_comp);
                    }
                }
            }
        }

        // 第二阶段：应用所有合并（单次遍历）
        if changed {
            for comp in components.iter_mut() {
                *comp = find(&mut parent, *comp);
            }
        }

        changed
    }

    /// 合并跨分区的连通分量 - 使用 Union-Find 优化 (O(V) → O(α(n)) per merge)
    fn merge_cross_partition_components<G, F>(
        &self,
        graph: &G,
        partitions: &[Partition],
        get_node_pos: &F,
        components: &mut [usize],
    ) -> bool
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
        F: Fn(NodeIndex) -> Option<usize>,
    {
        let n = components.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<u8> = vec![0; n];
        let mut changed = false;

        // Union-Find 查找（带路径压缩）
        #[inline]
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            let root = {
                let mut cur = x;
                while parent[cur] != cur {
                    cur = parent[cur];
                }
                cur
            };
            // 路径压缩
            while parent[x] != root {
                let next = parent[x];
                parent[x] = root;
                x = next;
            }
            root
        }

        // Union-Find 合并（union by rank）
        // 时间复杂度：O(α(n)) amortized vs O(V) for naive scan
        #[inline]
        fn union(parent: &mut [usize], rank: &mut [u8], x: usize, y: usize) -> bool {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                if rank[root_x] < rank[root_y] {
                    parent[root_x] = root_y;
                } else if rank[root_x] > rank[root_y] {
                    parent[root_y] = root_x;
                } else {
                    parent[root_y] = root_x;
                    rank[root_x] += 1;
                }
                true
            } else {
                false
            }
        }

        // 收集所有边界节点的连接信息，使用 Union-Find 合并
        for partition in partitions {
            for &boundary_node in &partition.boundary_nodes {
                let boundary_pos = match get_node_pos(boundary_node) {
                    Some(pos) => pos,
                    None => continue,
                };
                let boundary_comp = components[boundary_pos];

                for neighbor in graph.neighbors(boundary_node) {
                    if let Some(neighbor_pos) = get_node_pos(neighbor) {
                        let neighbor_comp = components[neighbor_pos];
                        if boundary_comp != neighbor_comp {
                            // 使用 Union-Find 合并，O(α(n)) vs O(V)
                            changed |= union(&mut parent, &mut rank, boundary_comp, neighbor_comp);
                        }
                    }
                }
            }
        }

        // 应用所有合并（单次遍历）
        if changed {
            for comp in components.iter_mut() {
                *comp = find(&mut parent, *comp);
            }
        }

        changed
    }

    /// 重新编号分量（压缩到连续范围）
    fn renumber_components(&self, components: &mut [usize]) {
        let mut id_mapping: Vec<usize> = vec![usize::MAX; components.len()];
        let mut next_id = 0;

        for comp_id in components.iter() {
            if *comp_id != usize::MAX && id_mapping[*comp_id] == usize::MAX {
                id_mapping[*comp_id] = next_id;
                next_id += 1;
            }
        }

        for comp_id in components.iter_mut() {
            if *comp_id != usize::MAX {
                *comp_id = id_mapping[*comp_id];
            }
        }
    }

    /// 计算单个节点的连通分量（从指定节点开始的 BFS/DFS）
    pub fn component_from_node<G>(&self, graph: &G, start: NodeIndex) -> Vec<NodeIndex>
    where
        G: VirtualGraph<NodeData = (), EdgeData = ()>,
    {
        // 使用 Vec<bool> 替代 HashSet
        let all_nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let node_to_idx: HashMap<NodeIndex, usize> = all_nodes
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();
        let n = all_nodes.len();

        let mut visited: Vec<bool> = vec![false; n];
        let mut component = Vec::with_capacity(n / 2); // Estimate: assume 2 components on average
        let mut stack = vec![start];

        if let Some(&start_idx) = node_to_idx.get(&start) {
            visited[start_idx] = true;
        } else {
            return component;
        }

        while let Some(node) = stack.pop() {
            component.push(node);
            for neighbor in graph.neighbors(node) {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    if !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
                        stack.push(neighbor);
                    }
                }
            }
        }

        component
    }
}

/// 单机连通分量算法（用于对比）
pub fn simple_connected_components<G>(graph: &G) -> Vec<Vec<NodeIndex>>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    // 使用 Vec<bool> 替代 HashSet
    let all_nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let node_to_idx: HashMap<NodeIndex, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let n = all_nodes.len();

    let mut visited: Vec<bool> = vec![false; n];
    let mut components = Vec::with_capacity(n / 4); // Estimate: assume 4 components on average

    for node_ref in graph.nodes() {
        let node = node_ref.index();
        let node_idx = node_to_idx[&node];
        if !visited[node_idx] {
            let mut component = Vec::with_capacity(n / 4); // Estimate component size
            let mut stack = vec![node];

            visited[node_idx] = true;

            while let Some(current) = stack.pop() {
                component.push(current);
                for neighbor in graph.neighbors(current) {
                    if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                        if !visited[neighbor_idx] {
                            visited[neighbor_idx] = true;
                            stack.push(neighbor);
                        }
                    }
                }
            }

            components.push(component);
        }
    }

    components
}

/// 弱连通分量（针对有向图）
pub fn weakly_connected_components<G>(graph: &G) -> Vec<Vec<NodeIndex>>
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    // 将有向图视为无向图处理
    simple_connected_components(graph)
}

/// 强连通分量数量（使用 Tarjan 算法）- Vec 优化版
pub fn count_strongly_connected_components<G>(graph: &G) -> usize
where
    G: VirtualGraph<NodeData = (), EdgeData = ()>,
{
    let all_nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = all_nodes.len();
    
    // 使用 Vec 替代 HashMap - O(1) 访问无哈希开销
    let mut node_to_idx: Vec<usize> = vec![usize::MAX; n];
    for (i, &node) in all_nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    let mut index_counter = 0;
    let mut stack: Vec<NodeIndex> = Vec::with_capacity(n);
    let mut lowlinks: Vec<usize> = vec![0; n];
    let mut index: Vec<Option<usize>> = vec![None; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut scc_count = 0;

    #[allow(clippy::too_many_arguments)]
    fn strongconnect<G: VirtualGraph<NodeData = (), EdgeData = ()>>(
        graph: &G,
        v: NodeIndex,
        v_idx: usize,
        node_to_idx: &[usize],
        on_stack: &mut Vec<bool>,
        index_counter: &mut usize,
        stack: &mut Vec<NodeIndex>,
        lowlinks: &mut Vec<usize>,
        index: &mut Vec<Option<usize>>,
        scc_count: &mut usize,
    ) {
        *index_counter += 1;
        index[v_idx] = Some(*index_counter);
        lowlinks[v_idx] = *index_counter;
        stack.push(v);
        on_stack[v_idx] = true;

        for w in graph.neighbors(v) {
            let w_idx = node_to_idx[w.index()];
            if w_idx == usize::MAX {
                continue;
            }
            
            if index[w_idx].is_none() {
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
                    scc_count,
                );
                if lowlinks[w_idx] < lowlinks[v_idx] {
                    lowlinks[v_idx] = lowlinks[w_idx];
                }
            } else if on_stack[w_idx] {
                if let Some(idx_w) = index[w_idx] {
                    if idx_w < lowlinks[v_idx] {
                        lowlinks[v_idx] = idx_w;
                    }
                }
            }
        }

        if lowlinks[v_idx] == index[v_idx].unwrap() {
            *scc_count += 1;
            loop {
                let w = stack.pop().unwrap();
                let w_idx = node_to_idx[w.index()];
                if w_idx != usize::MAX {
                    on_stack[w_idx] = false;
                }
                if w == v {
                    break;
                }
            }
        }
    }

    for node_ref in graph.nodes() {
        let node = node_ref.index();
        let node_idx = node_to_idx[node.index()];
        if node_idx != usize::MAX && index[node_idx].is_none() {
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
                &mut scc_count,
            );
        }
    }

    scc_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::partitioner::{HashPartitioner, Partitioner};
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_cc_config() {
        let config = ConnectedComponentsConfig::new()
            .with_parallel(true)
            .with_max_iterations(50);

        assert!(config.parallel);
        assert_eq!(config.max_iterations, 50);
    }

    #[test]
    fn test_distributed_cc_basic() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建两个连通分量
        // 分量 1: 0-1-2-3-4
        for i in 0..4 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }
        // 分量 2: 5-6-7-8-9
        for i in 5..9 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let cc = DistributedConnectedComponents::new();
        let result = cc.compute(&graph, &partitions);

        assert_eq!(result.component_count, 2);
        assert!(result.is_connected(nodes[0], nodes[4]));
        assert!(result.is_connected(nodes[5], nodes[9]));
        assert!(!result.is_connected(nodes[0], nodes[5]));
    }

    #[test]
    fn test_distributed_cc_single_component() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建单个连通分量（所有节点相连）
        for i in 0..nodes.len() - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(4);
        let partitions = partitioner.partition_graph(&graph);

        let cc = DistributedConnectedComponents::new();
        let result = cc.compute(&graph, &partitions);

        assert_eq!(result.component_count, 1);
    }

    #[test]
    fn test_distributed_cc_disconnected() {
        let mut graph = Graph::<(), ()>::undirected();
        let _nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 所有节点都是孤立的
        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let cc = DistributedConnectedComponents::new();
        let result = cc.compute(&graph, &partitions);

        assert_eq!(result.component_count, 6);
    }

    #[test]
    fn test_simple_connected_components() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建两个连通分量
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        graph.add_edge(nodes[3], nodes[4], ()).unwrap();
        graph.add_edge(nodes[4], nodes[5], ()).unwrap();

        let components = simple_connected_components(&graph);

        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_component_from_node() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        graph.add_edge(nodes[3], nodes[4], ()).unwrap();

        let cc = DistributedConnectedComponents::new();
        let component = cc.component_from_node(&graph, nodes[0]);

        assert_eq!(component.len(), 3);
        assert!(component.contains(&nodes[0]));
        assert!(component.contains(&nodes[2]));
        assert!(!component.contains(&nodes[3]));
    }

    #[test]
    fn test_largest_component() {
        let mut graph = Graph::<(), ()>::undirected();
        let nodes: Vec<NodeIndex> = (0..10).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建大小不同的分量
        for i in 0..6 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }
        for i in 7..9 {
            graph.add_edge(nodes[i], nodes[i + 1], ()).unwrap();
        }

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition_graph(&graph);

        let cc = DistributedConnectedComponents::new();
        let mut result = cc.compute(&graph, &partitions);

        let largest = result.largest_component().unwrap();
        assert_eq!(largest.1, 7); // 节点 0-6
    }

    #[test]
    fn test_partition_stats() {
        let mut graph = Graph::<(), ()>::undirected();
        for _ in 0..20 {
            graph.add_node(()).unwrap();
        }

        let partitioner = HashPartitioner::new(4);
        let partitions = partitioner.partition_graph(&graph);

        let cc = DistributedConnectedComponents::new();
        let result = cc.compute(&graph, &partitions);

        assert_eq!(result.partition_stats.len(), 4);

        let total_nodes: usize = result.partition_stats.iter().map(|s| s.node_count).sum();
        assert_eq!(total_nodes, 20);
    }

    #[test]
    fn test_count_strongly_connected_components() {
        let mut graph = Graph::<(), ()>::directed();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(()).unwrap()).collect();

        // 创建两个强连通分量
        graph.add_edge(nodes[0], nodes[1], ()).unwrap();
        graph.add_edge(nodes[1], nodes[2], ()).unwrap();
        graph.add_edge(nodes[2], nodes[0], ()).unwrap();

        graph.add_edge(nodes[3], nodes[4], ()).unwrap();
        graph.add_edge(nodes[4], nodes[5], ()).unwrap();
        graph.add_edge(nodes[5], nodes[3], ()).unwrap();

        let count = count_strongly_connected_components(&graph);
        assert_eq!(count, 2);
    }
}
