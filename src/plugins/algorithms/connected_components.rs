//! Connected Components (连通分量) 算法插件实现

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, ConfigField, ConfigFieldType, GraphAlgorithm, PluginContext,
    PluginInfo, PluginPriority,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;

/// 连通分量算法插件
///
/// 用于检测无向图中的连通分量
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::connected_components::ConnectedComponentsPlugin;
/// use god_graph::graph::Graph;
///
/// let plugin = ConnectedComponentsPlugin::new();
/// ```
pub struct ConnectedComponentsPlugin;

impl ConnectedComponentsPlugin {
    /// 创建新的连通分量算法插件实例
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::connected_components::ConnectedComponentsPlugin;
    ///
    /// let plugin = ConnectedComponentsPlugin::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// 计算连通分量
    pub fn compute<G>(&self, graph: &G) -> VgiResult<ConnectedComponentsResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(ConnectedComponentsResult {
                component_count: 0,
                node_to_component: Vec::new(),
                components: Vec::new(),
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

        // 使用 Vec<bool> instead of HashSet
        let mut visited: Vec<bool> = vec![false; n];
        let mut node_to_component: Vec<usize> = vec![usize::MAX; n];
        let mut components: Vec<Vec<usize>> = Vec::with_capacity(n / 4 + 1);
        let mut component_id = 0;

        for (start_pos, _start_idx) in node_indices.iter().enumerate() {
            if !visited[start_pos] {
                let mut component = Vec::with_capacity(n / 2 + 1);
                self.bfs_component(graph, start_pos, &node_id_to_pos, &mut visited, &mut component);

                // 记录节点到分量的映射
                for &pos in &component {
                    node_to_component[pos] = component_id;
                }

                components.push(component);
                component_id += 1;
            }
        }

        Ok(ConnectedComponentsResult {
            component_count: components.len(),
            node_to_component,
            components,
        })
    }

    /// 使用 BFS 查找连通分量
    fn bfs_component<G>(
        &self,
        graph: &G,
        start_pos: usize,
        node_id_to_pos: &[usize],
        visited: &mut [bool],
        component: &mut Vec<usize>,
    ) where
        G: VirtualGraph + ?Sized,
    {
        let mut queue: Vec<usize> = Vec::new();
        queue.push(start_pos);
        visited[start_pos] = true;

        while let Some(pos) = queue.pop() {
            component.push(pos);

            let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
            for neighbor_idx in graph.neighbors(node_indices[pos]) {
                let neighbor_pos = node_id_to_pos[neighbor_idx.index()];
                if neighbor_pos != usize::MAX && !visited[neighbor_pos] {
                    visited[neighbor_pos] = true;
                    queue.push(neighbor_pos);
                }
            }
        }
    }
}

impl Default for ConnectedComponentsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// 连通分量计算结果
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    /// 连通分量数量
    pub component_count: usize,
    /// 节点到连通分量的映射 (position-based)
    pub node_to_component: Vec<usize>,
    /// 每个连通分量包含的节点列表
    pub components: Vec<Vec<usize>>,
}

impl GraphAlgorithm for ConnectedComponentsPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("connected_components", "1.0.0", "连通分量检测算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Undirected])
            .with_tags(&["connectivity", "component", "clustering"])
            .with_priority(PluginPriority::Normal)
            .with_config_field(
                ConfigField::new("min_component_size", ConfigFieldType::Integer)
                    .description("最小连通分量大小，小于此值的分量将被忽略")
                    .default_value("1"),
            )
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        ctx.report_progress(0.1);

        let result = self.compute(ctx.graph)?;

        ctx.report_progress(1.0);

        // 获取最小分量大小配置
        let min_size = ctx.get_config_as("min_component_size", 1usize);

        // 过滤小的分量
        let filtered_components: Vec<Vec<usize>> = result
            .components
            .into_iter()
            .filter(|c| c.len() >= min_size)
            .collect();

        let flat_components: Vec<usize> = filtered_components.iter().flatten().copied().collect();

        Ok(AlgorithmResult::new(
            "connected_components",
            AlgorithmData::Communities(flat_components),
        )
        .with_metadata("component_count", filtered_components.len().to_string())
        .with_metadata("total_nodes", ctx.graph.node_count().to_string())
        .with_metadata("min_component_size", min_size.to_string()))
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
    fn test_connected_components_basic() {
        let mut graph = Graph::<String, f64>::undirected();

        // 创建两个连通分量
        // 分量 1: 0-1-2
        let n0 = graph.add_node("node_0".to_string()).unwrap();
        let n1 = graph.add_node("node_1".to_string()).unwrap();
        let n2 = graph.add_node("node_2".to_string()).unwrap();
        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();

        // 分量 2: 3-4
        let n3 = graph.add_node("node_3".to_string()).unwrap();
        let n4 = graph.add_node("node_4".to_string()).unwrap();
        graph.add_edge(n3, n4, 1.0).unwrap();

        let plugin = ConnectedComponentsPlugin::new();
        let result = plugin.compute(&graph).unwrap();

        assert_eq!(result.component_count, 2);
        assert_eq!(result.components.len(), 2);
    }

    #[test]
    fn test_connected_components_empty_graph() {
        let graph = Graph::<String, f64>::undirected();
        let plugin = ConnectedComponentsPlugin::new();
        let result = plugin.compute(&graph).unwrap();

        assert_eq!(result.component_count, 0);
        assert!(result.components.is_empty());
    }

    #[test]
    fn test_connected_components_plugin_info() {
        let plugin = ConnectedComponentsPlugin::new();
        let info = plugin.info();

        assert_eq!(info.name, "connected_components");
        assert!(info.tags.contains(&"connectivity".to_string()));
        assert_eq!(info.priority, PluginPriority::Normal);
    }
}
