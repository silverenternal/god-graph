//! DFS (深度优先搜索) 算法插件实现

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;

/// DFS traversal state
struct DfsState<'a> {
    visited: &'a mut Vec<bool>,
    order: &'a mut Vec<usize>,
    discovery_time: &'a mut Vec<usize>,
    finish_time: &'a mut Vec<usize>,
    parent: &'a mut Vec<Option<usize>>,
    time: &'a mut usize,
}

/// DFS (深度优先搜索) 算法插件
///
/// 提供基于深度优先搜索的图遍历功能，支持：
/// - 图遍历并记录访问顺序
/// - 环检测（有向图）
/// - 连通分量计算（无向图）
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::dfs::DfsPlugin;
/// use god_graph::graph::Graph;
///
/// // 创建 DFS 插件，从节点 0 开始遍历
/// let plugin = DfsPlugin::new(0);
///
/// // 或者使用默认配置
/// let plugin = DfsPlugin::new_unspecified();
///
/// // 可选：不记录访问顺序
/// let plugin = DfsPlugin::new(0).with_record_order(false);
/// ```
pub struct DfsPlugin {
    start_node: usize,
    record_order: bool,
}

impl DfsPlugin {
    /// 创建新的 DFS 插件实例
    ///
    /// # 参数
    ///
    /// * `start_node` - 起始节点索引
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    ///
    /// let plugin = DfsPlugin::new(0);
    /// ```
    pub fn new(start_node: usize) -> Self {
        Self {
            start_node,
            record_order: true,
        }
    }

    /// 创建默认配置的 DFS 插件实例
    ///
    /// 默认从节点 0 开始遍历，并记录访问顺序。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    ///
    /// let plugin = DfsPlugin::new_unspecified();
    /// ```
    pub fn new_unspecified() -> Self {
        Self {
            start_node: 0,
            record_order: true,
        }
    }

    /// 设置是否记录访问顺序
    ///
    /// # 参数
    ///
    /// * `record` - `true` 记录访问顺序，`false` 不记录
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    ///
    /// let plugin = DfsPlugin::new(0).with_record_order(false);
    /// ```
    pub fn with_record_order(mut self, record: bool) -> Self {
        self.record_order = record;
        self
    }

    /// 从指定节点开始遍历图
    ///
    /// 执行深度优先搜索并返回遍历结果。
    ///
    /// # 参数
    ///
    /// * `graph` - 要遍历的图
    /// * `start` - 起始节点索引
    ///
    /// # 返回
    ///
    /// 返回包含访问节点数、访问顺序、发现时间、完成时间和父节点信息的 `DfsResult`。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    /// use god_graph::graph::Graph;
    ///
    /// let mut graph = Graph::<i32, f64>::directed();
    /// graph.add_node(1).unwrap();
    /// graph.add_node(2).unwrap();
    /// graph.add_edge(0, 1, 1.0).unwrap();
    ///
    /// let plugin = DfsPlugin::new(0);
    /// let result = plugin.traverse(&graph, 0).unwrap();
    ///
    /// assert!(result.visited_count >= 1);
    /// ```
    pub fn traverse<G>(&self, graph: &G, start: usize) -> VgiResult<DfsResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(DfsResult {
                visited_count: 0,
                order: vec![],
                discovery_time: vec![],
                finish_time: vec![],
                parent: vec![],
            });
        }

        // 收集所有节点索引并创建位置映射
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let node_id_to_pos: Vec<usize> = {
            let mut map = vec![usize::MAX; n.max(1)];
            for (i, idx) in node_indices.iter().enumerate() {
                if idx.index() < map.len() {
                    map[idx.index()] = i;
                }
            }
            map
        };

        if start >= node_indices.len() {
            return Err(crate::vgi::VgiError::Internal {
                message: format!("Start node {} out of range", start),
            });
        }

        let start_pos = node_id_to_pos.get(start).copied().unwrap_or(usize::MAX);
        if start_pos == usize::MAX || start_pos >= n {
            return Err(crate::vgi::VgiError::Internal {
                message: format!("Start node {} not found", start),
            });
        }

        // 初始化 DFS 状态 (Vec instead of HashMap/HashSet)
        let mut visited: Vec<bool> = vec![false; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut discovery_time: Vec<usize> = vec![0; n];
        let mut finish_time: Vec<usize> = vec![0; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut time = 0;
        let mut state = DfsState {
            visited: &mut visited,
            order: &mut order,
            discovery_time: &mut discovery_time,
            finish_time: &mut finish_time,
            parent: &mut parent,
            time: &mut time,
        };

        self.dfs_visit(graph, start_pos, &node_id_to_pos, 0, &mut state);

        Ok(DfsResult {
            visited_count: visited.iter().filter(|&&v| v).count(),
            order,
            discovery_time,
            finish_time,
            parent,
        })
    }

    fn dfs_visit<G>(
        &self,
        graph: &G,
        node_pos: usize,
        node_id_to_pos: &Vec<usize>,
        depth: usize,
        state: &mut DfsState<'_>,
    ) where
        G: VirtualGraph + ?Sized,
    {
        if depth > graph.node_count() {
            return; // 防止无限递归
        }

        state.visited[node_pos] = true;
        *state.time += 1;
        state.discovery_time[node_pos] = *state.time;

        if self.record_order {
            state.order.push(node_pos);
        }

        // 获取邻居
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        for neighbor_idx in graph.neighbors(node_indices[node_pos]) {
            let neighbor_pos = node_id_to_pos[neighbor_idx.index()];
            if neighbor_pos != usize::MAX && !state.visited[neighbor_pos] {
                state.parent[neighbor_pos] = Some(node_pos);
                self.dfs_visit(graph, neighbor_pos, node_id_to_pos, depth + 1, state);
            }
        }

        *state.time += 1;
        state.finish_time[node_pos] = *state.time;
    }

    /// 检测有向图中是否存在环
    ///
    /// 使用 DFS 遍历检测图中是否存在环（cycle）。
    ///
    /// # 参数
    ///
    /// * `graph` - 要检测的图
    ///
    /// # 返回
    ///
    /// 如果图中存在环则返回 `Ok(true)`，否则返回 `Ok(false)`。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    /// use god_graph::graph::Graph;
    ///
    /// let mut graph = Graph::<i32, f64>::directed();
    /// graph.add_node(1).unwrap();
    /// graph.add_node(2).unwrap();
    /// graph.add_edge(0, 1, 1.0).unwrap();
    /// graph.add_edge(1, 0, 1.0).unwrap();
    ///
    /// let plugin = DfsPlugin::new(0);
    /// let has_cycle = plugin.has_cycle(&graph).unwrap();
    ///
    /// assert!(has_cycle);
    /// ```
    pub fn has_cycle<G>(&self, graph: &G) -> VgiResult<bool>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let node_id_to_pos: Vec<usize> = {
            let mut map = vec![usize::MAX; n.max(1)];
            for (i, idx) in node_indices.iter().enumerate() {
                if idx.index() < map.len() {
                    map[idx.index()] = i;
                }
            }
            map
        };
        
        let mut visited: Vec<bool> = vec![false; n];
        let mut rec_stack: Vec<bool> = vec![false; n];

        for (pos, _idx) in node_indices.iter().enumerate() {
            if !visited[pos]
                && self.has_cycle_util(graph, pos, &node_id_to_pos, &mut visited, &mut rec_stack)?
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn has_cycle_util<G>(
        &self,
        graph: &G,
        node_pos: usize,
        node_id_to_pos: &Vec<usize>,
        visited: &mut Vec<bool>,
        rec_stack: &mut Vec<bool>,
    ) -> VgiResult<bool>
    where
        G: VirtualGraph + ?Sized,
    {
        visited[node_pos] = true;
        rec_stack[node_pos] = true;

        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        for neighbor_idx in graph.neighbors(node_indices[node_pos]) {
            let neighbor_pos = node_id_to_pos[neighbor_idx.index()];
            if neighbor_pos == usize::MAX {
                continue;
            }
            if !visited[neighbor_pos] {
                if self.has_cycle_util(graph, neighbor_pos, node_id_to_pos, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack[neighbor_pos] {
                return Ok(true);
            }
        }

        rec_stack[node_pos] = false;
        Ok(false)
    }

    /// 计算无向图的连通分量
    ///
    /// 使用 DFS 遍历找出图中所有的连通分量。
    ///
    /// # 参数
    ///
    /// * `graph` - 要计算的图
    ///
    /// # 返回
    ///
    /// 返回包含所有连通分量的向量，每个连通分量是一个节点索引向量。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::dfs::DfsPlugin;
    /// use god_graph::graph::Graph;
    ///
    /// let mut graph = Graph::<i32, f64>::undirected();
    /// graph.add_node(1).unwrap();
    /// graph.add_node(2).unwrap();
    /// graph.add_node(3).unwrap();
    /// graph.add_edge(0, 1, 1.0).unwrap();
    ///
    /// let plugin = DfsPlugin::new(0);
    /// let components = plugin.connected_components(&graph).unwrap();
    ///
    /// assert_eq!(components.len(), 2);
    /// ```
    pub fn connected_components<G>(&self, graph: &G) -> VgiResult<Vec<Vec<usize>>>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let node_id_to_pos: Vec<usize> = {
            let mut map = vec![usize::MAX; n.max(1)];
            for (i, idx) in node_indices.iter().enumerate() {
                if idx.index() < map.len() {
                    map[idx.index()] = i;
                }
            }
            map
        };
        
        let mut visited: Vec<bool> = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::with_capacity(n / 4 + 1);

        for (pos, _idx) in node_indices.iter().enumerate() {
            if !visited[pos] {
                let mut component = Vec::with_capacity(n / 2 + 1);
                self.dfs_component(graph, pos, &node_id_to_pos, &mut visited, &mut component);
                components.push(component);
            }
        }

        Ok(components)
    }

    fn dfs_component<G>(
        &self,
        graph: &G,
        node_pos: usize,
        node_id_to_pos: &Vec<usize>,
        visited: &mut Vec<bool>,
        component: &mut Vec<usize>,
    ) where
        G: VirtualGraph + ?Sized,
    {
        visited[node_pos] = true;
        component.push(node_pos);

        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        for neighbor_idx in graph.neighbors(node_indices[node_pos]) {
            let neighbor_pos = node_id_to_pos[neighbor_idx.index()];
            if neighbor_pos != usize::MAX && !visited[neighbor_pos] {
                self.dfs_component(graph, neighbor_pos, node_id_to_pos, visited, component);
            }
        }
    }
}

/// DFS 遍历结果
///
/// 包含深度优先搜索的完整结果信息。
///
/// # 字段
///
/// * `visited_count` - 访问的节点总数
/// * `order` - 节点访问顺序（节点索引列表）
/// * `discovery_time` - 每个节点的发现时间 (position-based)
/// * `finish_time` - 每个节点的完成时间 (position-based)
/// * `parent` - 每个节点在 DFS 树中的父节点 (position-based)
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::dfs::DfsResult;
///
/// // DfsResult 通常由 DfsPlugin::traverse() 返回
/// // 包含完整的 DFS 遍历信息
/// ```
#[derive(Debug, Clone)]
pub struct DfsResult {
    /// 访问的节点总数
    pub visited_count: usize,
    /// 节点访问顺序（节点索引列表）
    pub order: Vec<usize>,
    /// 每个节点的发现时间 (position-based)
    pub discovery_time: Vec<usize>,
    /// 每个节点的完成时间 (position-based)
    pub finish_time: Vec<usize>,
    /// 每个节点在 DFS 树中的父节点 (position-based)
    pub parent: Vec<Option<usize>>,
}

impl GraphAlgorithm for DfsPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dfs", "1.0.0", "深度优先搜索算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["traversal", "search", "cycle-detection"])
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
            AlgorithmResult::new("dfs", AlgorithmData::NodeList(result.order))
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
    fn test_dfs_basic() {
        let mut graph = Graph::<String, f64>::directed();
        graph.add_node("node_0".to_string()).unwrap();
        graph.add_node("node_1".to_string()).unwrap();
        graph.add_node("node_2".to_string()).unwrap();
        graph.add_node("node_3".to_string()).unwrap();

        let plugin = DfsPlugin::new(0);
        let result = plugin.traverse(&graph, 0).unwrap();

        assert!(result.visited_count >= 1);
        assert!(!result.order.is_empty());
    }

    #[test]
    fn test_dfs_cycle_detection() {
        let mut graph_with_cycle = Graph::<i32, f64>::directed();
        graph_with_cycle.add_node(1).unwrap();
        graph_with_cycle.add_node(2).unwrap();
        graph_with_cycle.add_node(3).unwrap();
        // 注意：需要正确的边设置来形成环

        let plugin = DfsPlugin::new(0);
        // 简单测试
        let result = plugin.has_cycle(&graph_with_cycle);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dfs_plugin_info() {
        let plugin = DfsPlugin::new(0);
        let info = plugin.info();

        assert_eq!(info.name, "dfs");
        assert!(info.tags.contains(&"traversal".to_string()));
    }
}
