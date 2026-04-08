//! 拓扑排序算法插件实现
//!
//! 适用于有向无环图 (DAG)，返回节点的线性排序

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;
use std::collections::VecDeque;

/// 拓扑排序结果
#[derive(Debug, Clone)]
pub enum TopologicalSortResult {
    /// 排序成功，返回节点顺序
    Sorted(Vec<usize>),
    /// 图中存在环，无法进行拓扑排序
    HasCycle,
}

/// 拓扑排序算法插件
///
/// 适用于有向无环图 (DAG)，返回节点的线性排序。
/// 如果图中存在环，则返回 HasCycle 错误。
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::topological_sort::TopologicalSortPlugin;
/// use god_graph::graph::Graph;
///
/// let plugin = TopologicalSortPlugin::new();
///
/// let mut graph = Graph::<i32, f64>::directed();
/// graph.add_node(1).unwrap();
/// graph.add_node(2).unwrap();
/// graph.add_edge(0, 1, 1.0).unwrap();
///
/// let result = plugin.sort(&graph).unwrap();
/// ```
pub struct TopologicalSortPlugin;

impl TopologicalSortPlugin {
    /// 创建新的拓扑排序算法插件实例
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::topological_sort::TopologicalSortPlugin;
    ///
    /// let plugin = TopologicalSortPlugin::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// 拓扑排序核心实现（Kahn 算法）
    ///
    /// 使用入度 BFS 方法进行拓扑排序
    pub fn sort<G>(&self, graph: &G) -> VgiResult<TopologicalSortResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(TopologicalSortResult::Sorted(Vec::new()));
        }

        // 收集所有节点索引
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();

        // 构建节点 ID 到位置的映射：node_id_to_pos[id] = position in node_indices
        // 使用 usize::MAX 表示无效条目
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; n];
        for (pos, idx) in node_indices.iter().enumerate() {
            node_id_to_pos[idx.index()] = pos;
        }

        // 使用 Vec 代替 HashMap 计算入度，O(1) 访问
        let mut in_degree: Vec<usize> = vec![0; n];

        // 遍历所有边，计算入度
        for node_ref in graph.nodes() {
            let from_idx = node_ref.index();
            let from_node_idx = NodeIndex::new_public(from_idx.index());

            for neighbor_idx in graph.neighbors(from_node_idx) {
                let to_idx = neighbor_idx.index();
                let pos = node_id_to_pos[to_idx];
                if pos != usize::MAX {
                    in_degree[pos] += 1;
                }
            }
        }

        // 初始化队列：将所有入度为 0 的节点加入队列
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (pos, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(pos);
            }
        }

        // BFS 拓扑排序
        let mut result: Vec<usize> = Vec::with_capacity(n);

        while let Some(pos) = queue.pop_front() {
            let node_id = node_indices[pos].index();
            result.push(node_id);

            // 遍历该节点的所有邻居
            let node_idx = NodeIndex::new_public(node_id);
            for neighbor_idx in graph.neighbors(node_idx) {
                let neighbor_id = neighbor_idx.index();
                let neighbor_pos = node_id_to_pos[neighbor_id];
                if neighbor_pos != usize::MAX {
                    in_degree[neighbor_pos] -= 1;

                    // 如果邻居入度变为 0，加入队列
                    if in_degree[neighbor_pos] == 0 {
                        queue.push_back(neighbor_pos);
                    }
                }
            }
        }

        // 如果结果中的节点数小于总节点数，说明图中有环
        if result.len() < n {
            Ok(TopologicalSortResult::HasCycle)
        } else {
            Ok(TopologicalSortResult::Sorted(result))
        }
    }

    /// 使用 DFS 方法进行拓扑排序
    pub fn sort_dfs<G>(&self, graph: &G) -> VgiResult<TopologicalSortResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(TopologicalSortResult::Sorted(Vec::new()));
        }

        // 收集所有节点索引
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();

        // 构建节点 ID 到位置的映射
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; n];
        for (pos, idx) in node_indices.iter().enumerate() {
            node_id_to_pos[idx.index()] = pos;
        }

        // 状态：0=未访问，1=访问中，2=已访问（使用 Vec 代替 HashMap）
        let mut state: Vec<u8> = vec![0; n];

        let mut result: Vec<usize> = Vec::with_capacity(n);

        // DFS 访问
        for pos in 0..n {
            if state[pos] == 0 && !self.dfs_visit(graph, pos, &mut state, &mut result, &node_id_to_pos)? {
                return Ok(TopologicalSortResult::HasCycle);
            }
        }

        result.reverse(); // DFS 后序遍历的逆序即为拓扑排序
        Ok(TopologicalSortResult::Sorted(result))
    }

    /// DFS 辅助函数
    fn dfs_visit<G>(
        &self,
        graph: &G,
        pos: usize,
        state: &mut Vec<u8>,
        result: &mut Vec<usize>,
        node_id_to_pos: &[usize],
    ) -> VgiResult<bool>
    where
        G: VirtualGraph + ?Sized,
    {
        state[pos] = 1; // 标记为访问中

        let node_id = node_id_to_pos[pos];
        let node_idx = NodeIndex::new_public(node_id);
        for neighbor_idx in graph.neighbors(node_idx) {
            let neighbor_id = neighbor_idx.index();
            let neighbor_pos = node_id_to_pos[neighbor_id];

            if neighbor_pos != usize::MAX {
                let neighbor_state = state[neighbor_pos];

                if neighbor_state == 1 {
                    // 发现后向边，存在环
                    return Ok(false);
                } else if neighbor_state == 0
                    && !self.dfs_visit(graph, neighbor_pos, state, result, node_id_to_pos)?
                {
                    return Ok(false);
                }
            }
        }

        state[pos] = 2; // 标记为已访问
        let node_id = node_id_to_pos[pos];
        result.push(node_id);
        Ok(true)
    }

    /// 检查图是否为 DAG（有向无环图）
    pub fn is_dag<G>(&self, graph: &G) -> VgiResult<bool>
    where
        G: VirtualGraph + ?Sized,
    {
        match self.sort(graph)? {
            TopologicalSortResult::Sorted(_) => Ok(true),
            TopologicalSortResult::HasCycle => Ok(false),
        }
    }
}

impl Default for TopologicalSortPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAlgorithm for TopologicalSortPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(
            "topological-sort",
            "1.0.0",
            "有向无环图 (DAG) 的拓扑排序算法",
        )
        .with_author("God-Graph Team")
        .with_required_capabilities(&[Capability::IncrementalUpdate])
        .with_supported_graph_types(&[GraphType::Directed])
        .with_tags(&["topological-sort", "dag", "ordering", "scheduling"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let use_dfs = ctx.get_config_or("use_dfs", "false") == "true";

        ctx.report_progress(0.1);

        let result = if use_dfs {
            self.sort_dfs(ctx.graph)?
        } else {
            self.sort(ctx.graph)?
        };

        ctx.report_progress(0.8);

        let algorithm_result = match result {
            TopologicalSortResult::Sorted(order) => {
                AlgorithmResult::new("topological_sort", AlgorithmData::NodeList(order.clone()))
                    .with_metadata("is_dag", "true")
                    .with_metadata("node_count", order.len().to_string())
            }
            TopologicalSortResult::HasCycle => AlgorithmResult::new(
                "topological_sort_error",
                AlgorithmData::String(
                    "Graph contains a cycle, cannot perform topological sort".to_string(),
                ),
            )
            .with_metadata("is_dag", "false")
            .with_metadata("error", "has_cycle"),
        }
        .with_metadata("algorithm", "topological-sort")
        .with_metadata("method", if use_dfs { "dfs" } else { "kahn" });

        ctx.report_progress(1.0);
        Ok(algorithm_result)
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

    fn create_dag() -> Graph<String, ()> {
        let mut graph = Graph::<String, ()>::directed();

        // 创建 DAG: 任务依赖关系
        // A -> B -> D
        // A -> C -> D
        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        let c = graph.add_node("C".to_string()).unwrap();
        let d = graph.add_node("D".to_string()).unwrap();

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(a, c, ()).unwrap();
        graph.add_edge(b, d, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();

        graph
    }

    #[test]
    fn test_topological_sort_dag() {
        let graph = create_dag();
        let plugin = TopologicalSortPlugin::new();

        let result = plugin.sort(&graph).unwrap();

        match result {
            TopologicalSortResult::Sorted(order) => {
                assert_eq!(order.len(), 4);
                // A 应该在 B 和 C 之前
                let a_pos = order.iter().position(|&x| x == 0).unwrap();
                let b_pos = order.iter().position(|&x| x == 1).unwrap();
                let c_pos = order.iter().position(|&x| x == 2).unwrap();
                let d_pos = order.iter().position(|&x| x == 3).unwrap();

                assert!(a_pos < b_pos);
                assert!(a_pos < c_pos);
                assert!(b_pos < d_pos);
                assert!(c_pos < d_pos);
            }
            TopologicalSortResult::HasCycle => panic!("DAG should not have cycle"),
        }
    }

    #[test]
    fn test_topological_sort_dfs() {
        let graph = create_dag();
        let plugin = TopologicalSortPlugin::new();

        let result = plugin.sort_dfs(&graph).unwrap();

        match result {
            TopologicalSortResult::Sorted(order) => {
                assert_eq!(order.len(), 4);
            }
            TopologicalSortResult::HasCycle => panic!("DAG should not have cycle"),
        }
    }

    #[test]
    fn test_topological_sort_with_cycle() {
        let mut graph = Graph::<String, ()>::directed();

        // 创建有环图：A -> B -> C -> A
        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        let c = graph.add_node("C".to_string()).unwrap();

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, a, ()).unwrap();

        let plugin = TopologicalSortPlugin::new();
        let result = plugin.sort(&graph).unwrap();

        match result {
            TopologicalSortResult::HasCycle => {} // 预期结果
            TopologicalSortResult::Sorted(_) => panic!("Graph with cycle should not be sortable"),
        }
    }

    #[test]
    fn test_topological_sort_empty_graph() {
        let graph = Graph::<String, ()>::directed();
        let plugin = TopologicalSortPlugin::new();

        let result = plugin.sort(&graph).unwrap();

        match result {
            TopologicalSortResult::Sorted(order) => {
                assert!(order.is_empty());
            }
            TopologicalSortResult::HasCycle => panic!("Empty graph should not have cycle"),
        }
    }

    #[test]
    fn test_topological_sort_is_dag() {
        let dag = create_dag();
        let plugin = TopologicalSortPlugin::new();

        assert!(plugin.is_dag(&dag).unwrap());

        let mut cyclic = Graph::<String, ()>::directed();
        let a = cyclic.add_node("A".to_string()).unwrap();
        let b = cyclic.add_node("B".to_string()).unwrap();
        cyclic.add_edge(a, b, ()).unwrap();
        cyclic.add_edge(b, a, ()).unwrap();

        assert!(!plugin.is_dag(&cyclic).unwrap());
    }

    #[test]
    fn test_topological_sort_plugin_info() {
        let plugin = TopologicalSortPlugin::new();
        let info = plugin.info();

        assert_eq!(info.name, "topological-sort");
        assert_eq!(info.version, "1.0.0");
        assert!(info.tags.contains(&"topological-sort".to_string()));
    }
}
