//! BFS (广度优先搜索) 算法插件实现
//!
//! BFS 是一种图遍历算法，从起始节点开始逐层访问所有可达节点。
//! 广泛应用于最短路径（无权图）、连通性检查、层级分析等场景。
//!
//! # 算法原理
//!
//! BFS 使用队列按层次遍历：
//! 1. 从起始节点开始，标记为已访问
//! 2. 访问当前节点的所有未访问邻居
//! 3. 将邻居加入队列
//! 4. 重复直到队列为空
//!
//! # 使用示例
//!
//! ```ignore
//! use god_graph::plugins::BfsPlugin;
//! use god_graph::graph::Graph;
//!
//! let mut graph = Graph::<String, f64>::directed();
//! // ... 添加节点和边
//!
//! let plugin = BfsPlugin::new(0); // 从节点 0 开始
//! let result = plugin.traverse(&graph, 0)?;
//! println!("访问了 {} 个节点", result.visited_count);
//! ```

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;
use std::collections::{HashMap, VecDeque};

/// 快速访问宏：将 NodeIndex 转换为 usize 索引
#[inline]
fn node_to_idx(node: NodeIndex) -> usize {
    node.index()
}

/// BFS 遍历结果
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// 访问的节点数
    pub visited_count: usize,
    /// 访问顺序（节点索引列表）
    pub order: Vec<usize>,
    /// 从起始节点到各节点的距离（key 为节点索引的整数值）
    pub distances: HashMap<usize, usize>,
    /// 每个节点的父节点（用于重构路径，key/value 为节点索引的整数值）
    pub parent: HashMap<usize, usize>,
}

/// BFS (广度优先搜索) 算法插件
///
/// 从指定起始节点开始遍历图，记录访问顺序和距离信息。
///
/// # 参数
///
/// * `start_node`: 起始节点索引
/// * `record_order`: 是否记录访问顺序
///
/// # 示例
///
/// ```ignore
/// let plugin = BfsPlugin::new(0);
/// let result = plugin.traverse(&graph, 0)?;
/// ```
pub struct BfsPlugin {
    start_node: usize,
    record_order: bool,
}

impl BfsPlugin {
    /// 创建新的 BFS 插件
    ///
    /// # 参数
    ///
    /// * `start_node`: 起始节点索引
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::BfsPlugin;
    /// let plugin = BfsPlugin::new(0);
    /// ```
    pub fn new(start_node: usize) -> Self {
        Self {
            start_node,
            record_order: true,
        }
    }

    /// 创建未指定起始节点的 BFS 插件（默认从 0 开始）
    pub fn new_unspecified() -> Self {
        Self {
            start_node: 0,
            record_order: true,
        }
    }

    /// 设置是否记录访问顺序
    pub fn with_record_order(mut self, record: bool) -> Self {
        self.record_order = record;
        self
    }

    /// 执行 BFS 遍历
    ///
    /// # 参数
    ///
    /// * `graph`: 要遍历的图
    /// * `start`: 起始节点索引
    ///
    /// # 返回
    ///
    /// 返回 BFS 遍历结果，包含访问顺序、距离等信息
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let plugin = BfsPlugin::new(0);
    /// let result = plugin.traverse(&graph, 0)?;
    /// println!("访问顺序：{:?}", result.order);
    /// ```
    pub fn traverse<G>(&self, graph: &G, start: usize) -> VgiResult<BfsResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(BfsResult {
                visited_count: 0,
                order: vec![],
                distances: HashMap::new(),
                parent: HashMap::new(),
            });
        }

        // 收集所有节点索引
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();

        if start >= node_indices.len() {
            return Err(crate::vgi::VgiError::Internal {
                message: format!("Start node {} out of range", start),
            });
        }

        let start_idx = node_indices[start];
        
        // 使用 Vec 替代 HashMap 提升性能：O(1) 直接索引 vs O(1) 哈希（高常数）
        // visited: 使用 Option<usize> 同时记录访问状态和距离（Some=已访问 + 距离，None=未访问）
        let mut visited = vec![None; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut parent: Vec<Option<usize>> = vec![None; n];

        let mut queue = VecDeque::with_capacity(n);
        queue.push_back(start_idx);
        visited[start] = Some(0); // 距离为 0

        while let Some(node_idx) = queue.pop_front() {
            let node_usize = node_to_idx(node_idx);
            
            // 记录访问顺序（使用节点的索引值）
            if self.record_order {
                order.push(node_usize);
            }

            let current_dist = visited[node_usize].unwrap();

            for neighbor_idx in graph.neighbors(node_idx) {
                let neighbor_usize = node_to_idx(neighbor_idx);
                if visited[neighbor_usize].is_none() {
                    visited[neighbor_usize] = Some(current_dist + 1);
                    parent[neighbor_usize] = Some(node_usize);
                    queue.push_back(neighbor_idx);
                }
            }
        }

        // 构建结果 HashMap（仅用于外部 API 兼容）
        let visited_count = visited.iter().filter(|x| x.is_some()).count();
        let mut distances = HashMap::with_capacity(visited_count);
        for (i, dist) in visited.iter().enumerate() {
            if let Some(d) = dist {
                distances.insert(i, *d);
            }
        }

        let mut parent_map = HashMap::with_capacity(visited_count);
        for (i, p) in parent.iter().enumerate() {
            if let Some(parent_node) = p {
                parent_map.insert(i, *parent_node);
            }
        }

        Ok(BfsResult {
            visited_count: distances.len(),
            order,
            distances,
            parent: parent_map,
        })
    }

    /// 计算从起始节点到目标节点的最短路径
    ///
    /// 使用 BFS 遍历找出无权图中的最短路径（边数最少）。
    ///
    /// # 参数
    ///
    /// * `graph` - 要计算的图
    /// * `target` - 目标节点索引
    ///
    /// # 返回
    ///
    /// 返回从起始节点到目标节点的最短路径（节点索引列表）。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::bfs::BfsPlugin;
    /// use god_graph::graph::Graph;
    ///
    /// let mut graph = Graph::<i32, f64>::directed();
    /// graph.add_node(1).unwrap();
    /// graph.add_node(2).unwrap();
    /// graph.add_node(3).unwrap();
    /// graph.add_edge(0, 1, 1.0).unwrap();
    /// graph.add_edge(1, 2, 1.0).unwrap();
    ///
    /// let plugin = BfsPlugin::new(0);
    /// let path = plugin.shortest_path(&graph, 2).unwrap();
    ///
    /// assert_eq!(path, vec![0, 1, 2]);
    /// ```
    pub fn shortest_path<G>(&self, graph: &G, target: usize) -> VgiResult<Vec<usize>>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        
        if self.start_node >= node_indices.len() || target >= node_indices.len() {
            return Err(crate::vgi::VgiError::Internal {
                message: "Node index out of range".to_string(),
            });
        }

        let start_idx = node_indices[self.start_node];
        
        // 优化的 Vec 版本 BFS
        let mut visited = vec![false; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut queue = VecDeque::with_capacity(n);
        
        queue.push_back(start_idx);
        visited[self.start_node] = true;

        while let Some(node_idx) = queue.pop_front() {
            let node_usize = node_to_idx(node_idx);
            
            if node_usize == target {
                break;
            }

            for neighbor_idx in graph.neighbors(node_idx) {
                let neighbor_usize = node_to_idx(neighbor_idx);
                if !visited[neighbor_usize] {
                    visited[neighbor_usize] = true;
                    parent[neighbor_usize] = Some(node_usize);
                    queue.push_back(neighbor_idx);
                }
            }
        }

        if !visited[target] {
            return Err(crate::vgi::VgiError::Internal {
                message: format!("No path from {} to {}", self.start_node, target),
            });
        }

        // 重构路径
        let mut path = Vec::new();
        let mut current = target;

        loop {
            path.push(current);
            if current == self.start_node {
                break;
            }
            current = parent[current].unwrap_or(self.start_node);
        }

        path.reverse();
        Ok(path)
    }
}

impl GraphAlgorithm for BfsPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("bfs", "1.0.0", "广度优先搜索算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["traversal", "search", "shortest-path"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let start = self.start_node;

        if start >= ctx.graph.node_count() {
            return Err(crate::vgi::VgiError::Internal {
                message: format!("Start node {} out of range", start),
            });
        }

        ctx.report_progress(0.1);

        let result = self.traverse(ctx.graph, start)?;

        ctx.report_progress(1.0);

        Ok(
            AlgorithmResult::new("bfs", AlgorithmData::NodeList(result.order))
                .with_metadata("start_node", start.to_string())
                .with_metadata("visited_count", result.visited_count.to_string()),
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_bfs_basic() {
        let mut graph = Graph::<String, f64>::directed();

        let n0 = graph.add_node("node_0".to_string()).unwrap();
        let n1 = graph.add_node("node_1".to_string()).unwrap();
        let n2 = graph.add_node("node_2".to_string()).unwrap();
        let n3 = graph.add_node("node_3".to_string()).unwrap();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n0, n2, 1.0).unwrap();
        graph.add_edge(n1, n3, 1.0).unwrap();

        let plugin = BfsPlugin::new(0);
        let result = plugin.traverse(&graph, 0).unwrap();

        assert_eq!(result.visited_count, 4);
        assert_eq!(result.order.len(), 4);
        assert_eq!(result.order[0], 0);

        assert_eq!(result.distances[&0], 0);
        assert_eq!(result.distances[&1], 1);
        assert_eq!(result.distances[&2], 1);
        assert_eq!(result.distances[&3], 2);
    }

    #[test]
    fn test_bfs_plugin_info() {
        let plugin = BfsPlugin::new(0);
        let info = plugin.info();

        assert_eq!(info.name, "bfs");
        assert!(info.tags.contains(&"traversal".to_string()));
    }
}
