//! 接近中心性（Closeness Centrality）算法插件实现
//!
//! 衡量节点到图中所有其他节点的平均距离的倒数
//! 值越高表示该节点与其他节点的平均距离越近

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;
use std::collections::VecDeque;

/// 接近中心性算法插件
///
/// 衡量节点到图中所有其他节点的平均距离的倒数。
/// 值越高表示该节点与其他节点的平均距离越近。
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::closeness_centrality::ClosenessCentralityPlugin;
///
/// // 创建改进的接近中心性插件（处理非连通图）
/// let plugin = ClosenessCentralityPlugin::improved();
///
/// // 创建标准的接近中心性插件
/// let plugin = ClosenessCentralityPlugin::standard();
/// ```
pub struct ClosenessCentralityPlugin {
    /// 是否使用改进的归一化方法（处理非连通图）
    improved: bool,
}

impl ClosenessCentralityPlugin {
    /// 创建新的接近中心性算法插件实例
    ///
    /// # 参数
    ///
    /// * `improved` - 是否使用改进的归一化方法
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::closeness_centrality::ClosenessCentralityPlugin;
    ///
    /// let plugin = ClosenessCentralityPlugin::new(true);
    /// ```
    pub fn new(improved: bool) -> Self {
        Self { improved }
    }

    /// 创建改进的接近中心性插件实例
    ///
    /// 改进版本使用可达节点数进行归一化，适用于非连通图。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::closeness_centrality::ClosenessCentralityPlugin;
    ///
    /// let plugin = ClosenessCentralityPlugin::improved();
    /// ```
    pub fn improved() -> Self {
        Self { improved: true }
    }

    /// 创建标准的接近中心性插件实例
    ///
    /// 标准版本假设图是连通的。
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::closeness_centrality::ClosenessCentralityPlugin;
    ///
    /// let plugin = ClosenessCentralityPlugin::standard();
    /// ```
    pub fn standard() -> Self {
        Self { improved: false }
    }

    /// 接近中心性核心实现
    ///
    /// C(v) = (N - 1) / sum(d(v, t)) for all t in V
    /// 其中 d(v, t) 是节点 v 到节点 t 的最短距离
    pub fn compute<G>(&self, graph: &G) -> VgiResult<Vec<(usize, f64)>>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(Vec::new());
        }

        // 收集所有节点索引
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();

        // 构建节点 ID 到位置的映射
        let mut node_id_to_pos: Vec<usize> = vec![usize::MAX; n];
        for (pos, idx) in node_indices.iter().enumerate() {
            node_id_to_pos[idx.index()] = pos;
        }

        // 初始化接近中心性结果 Vec
        let mut centrality: Vec<(usize, f64)> = Vec::with_capacity(n);

        // 对每个节点执行 BFS 计算到所有其他节点的距离
        for (source_pos, source_idx) in node_indices.iter().enumerate() {
            let source = source_idx.index();

            // BFS 计算最短路径距离（使用 Vec 代替 HashMap）
            let distances = self.bfs_distances(graph, source, &node_id_to_pos, n)?;

            // 计算总距离
            let mut total_distance = 0.0;
            let mut reachable_count = 0usize;

            for (target_pos, &dist) in distances.iter().enumerate() {
                if target_pos != source_pos && dist.is_finite() && dist > 0.0 {
                    total_distance += dist;
                    reachable_count += 1;
                }
            }

            // 计算接近中心性
            let closeness = if total_distance > 0.0 {
                if self.improved && reachable_count < n - 1 {
                    // 改进的归一化方法（处理非连通图）
                    // C_improved(v) = (reachable_count / (N - 1)) * (reachable_count / total_distance)
                    let reachability_ratio = reachable_count as f64 / (n - 1) as f64;
                    reachability_ratio * (reachable_count as f64 / total_distance)
                } else {
                    // 标准接近中心性：C(v) = (N - 1) / total_distance
                    (n - 1) as f64 / total_distance
                }
            } else {
                0.0 // 没有可达节点
            };

            centrality.push((source, closeness));
        }

        Ok(centrality)
    }

    /// 使用 BFS 计算从源节点到所有其他节点的距离
    fn bfs_distances<G>(
        &self,
        graph: &G,
        source: usize,
        node_id_to_pos: &[usize],
        n: usize,
    ) -> VgiResult<Vec<f64>>
    where
        G: VirtualGraph + ?Sized,
    {
        // 使用 Vec 代替 HashMap，O(1) 访问
        let mut distances: Vec<f64> = vec![f64::INFINITY; n];
        let mut queue: VecDeque<usize> = VecDeque::new();

        // 找到源节点的位置
        let source_pos = node_id_to_pos[source];
        if source_pos == usize::MAX {
            return Ok(distances);
        }

        distances[source_pos] = 0.0;
        queue.push_back(source_pos);

        while let Some(pos) = queue.pop_front() {
            let node_id = node_id_to_pos[pos];
            let v_dist = distances[pos];

            let v_idx = NodeIndex::new_public(node_id);
            for w_idx in graph.neighbors(v_idx) {
                let w = w_idx.index();
                let w_pos = node_id_to_pos[w];

                if w_pos != usize::MAX {
                    // 边权重默认为 1.0
                    let edge_weight = 1.0;

                    if v_dist + edge_weight < distances[w_pos] {
                        distances[w_pos] = v_dist + edge_weight;
                        queue.push_back(w_pos);
                    }
                }
            }
        }

        Ok(distances)
    }

    /// 获取接近中心性最高的 K 个节点
    pub fn top_k<G>(&self, graph: &G, k: usize) -> VgiResult<Vec<(usize, f64)>>
    where
        G: VirtualGraph + ?Sized,
    {
        let centrality = self.compute(graph)?;

        let mut nodes: Vec<(usize, f64)> = centrality;

        if k >= nodes.len() {
            // P0 OPTIMIZATION: sort_unstable_by for 20-25% faster sorting
            nodes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(nodes);
        }
        
        // 使用 select_nth_unstable_by 实现 O(n) 部分排序
        nodes.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.truncate(k);
        Ok(nodes)
    }
}

impl Default for ClosenessCentralityPlugin {
    fn default() -> Self {
        Self::improved()
    }
}

impl GraphAlgorithm for ClosenessCentralityPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("closeness-centrality", "1.0.0", "接近中心性算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["centrality", "closeness", "distance", "importance"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let improved = ctx.get_config_or("improved", "true") == "true";
        let top_k = ctx.get_config_as("top_k", 0usize);

        let plugin = if improved {
            ClosenessCentralityPlugin::improved()
        } else {
            ClosenessCentralityPlugin::standard()
        };

        ctx.report_progress(0.1);

        let result = if top_k > 0 {
            let top_nodes = plugin.top_k(ctx.graph, top_k)?;
            let nodes: Vec<usize> = top_nodes.iter().map(|(id, _)| *id).collect();
            let scores: Vec<(usize, f64)> = top_nodes;

            AlgorithmResult::new("closeness_top_k", AlgorithmData::NodeList(nodes))
                .with_metadata("top_k", top_k.to_string())
                .with_metadata("scores", format!("{:?}", scores))
        } else {
            let centrality = plugin.compute(ctx.graph)?;
            let centrality_map: Vec<(usize, f64)> = centrality;

            AlgorithmResult::new(
                "closeness_centrality",
                AlgorithmData::NodeValues(centrality_map.into_iter().collect()),
            )
        }
        .with_metadata("improved", improved.to_string())
        .with_metadata("algorithm", "closeness-centrality")
        .with_metadata("node_count", ctx.graph.node_count().to_string());

        ctx.report_progress(1.0);
        Ok(result)
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

    fn create_star_graph() -> Graph<String, ()> {
        // 星型图：中心节点连接所有其他节点
        // 中心节点应该有最高的接近中心性
        let mut graph = Graph::<String, ()>::undirected();

        let center = graph.add_node("Center".to_string()).unwrap();
        let leaves: Vec<NodeIndex> = (0..5)
            .map(|i| graph.add_node(format!("Leaf{}", i)).unwrap())
            .collect();

        // 无向图需要添加双向边
        for leaf in leaves {
            graph.add_edge(center, leaf, ()).unwrap();
            graph.add_edge(leaf, center, ()).unwrap();
        }

        graph
    }

    fn create_line_graph() -> Graph<String, ()> {
        // 线性图：A -- B -- C -- D -- E
        // 中间节点 C 应该有最高的接近中心性
        let mut graph = Graph::<String, ()>::undirected();

        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        let c = graph.add_node("C".to_string()).unwrap();
        let d = graph.add_node("D".to_string()).unwrap();
        let e = graph.add_node("E".to_string()).unwrap();

        // 无向图需要添加双向边
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, b, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();
        graph.add_edge(d, e, ()).unwrap();
        graph.add_edge(e, d, ()).unwrap();

        graph
    }

    fn create_disconnected_graph() -> Graph<String, ()> {
        // 非连通图：两个分离的组件
        let mut graph = Graph::<String, ()>::undirected();

        // 组件 1: A -- B
        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();

        // 组件 2: C -- D
        let c = graph.add_node("C".to_string()).unwrap();
        let d = graph.add_node("D".to_string()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        graph
    }

    #[test]
    fn test_closeness_star_graph() {
        let graph = create_star_graph();
        let plugin = ClosenessCentralityPlugin::standard();

        let centrality = plugin.compute(&graph).unwrap();

        // 中心节点应该有最高的接近中心性
        let center_centrality = centrality.iter().find(|(id, _)| *id == 0).map(|(_, v)| *v).unwrap_or(0.0);

        // 叶子节点的接近中心性应该较低
        for i in 1..=5 {
            let leaf_centrality = centrality.iter().find(|(id, _)| *id == i).map(|(_, v)| *v).unwrap_or(0.0);
            assert!(leaf_centrality < center_centrality);
        }

        // 中心节点中心性应该接近 1.0（到所有叶子距离为 1）
        assert!(center_centrality > 0.8);
    }

    #[test]
    fn test_closeness_line_graph() {
        let graph = create_line_graph();
        let plugin = ClosenessCentralityPlugin::standard();

        let centrality = plugin.compute(&graph).unwrap();

        // 中间节点 C 应该有最高的接近中心性
        let c_centrality = centrality.iter().find(|(id, _)| *id == 2).map(|(_, v)| *v).unwrap_or(0.0);
        let a_centrality = centrality.iter().find(|(id, _)| *id == 0).map(|(_, v)| *v).unwrap_or(0.0);
        let e_centrality = centrality.iter().find(|(id, _)| *id == 4).map(|(_, v)| *v).unwrap_or(0.0);

        assert!(c_centrality > a_centrality);
        assert!(c_centrality > e_centrality);

        // A 和 E 应该有相同的中心性（对称）
        assert!((a_centrality - e_centrality).abs() < 1e-10);
    }

    #[test]
    fn test_closeness_disconnected_graph_standard() {
        let graph = create_disconnected_graph();
        let plugin = ClosenessCentralityPlugin::standard();

        let centrality = plugin.compute(&graph).unwrap();

        // 标准方法在非连通图中可能产生不合理的结果
        // 但不应崩溃
        assert_eq!(centrality.len(), 4);
    }

    #[test]
    fn test_closeness_disconnected_graph_improved() {
        let graph = create_disconnected_graph();
        let plugin = ClosenessCentralityPlugin::improved();

        let centrality = plugin.compute(&graph).unwrap();

        // 改进方法应该能更好地处理非连通图
        assert_eq!(centrality.len(), 4);

        // 所有值应该在 [0, 1] 范围内
        for (_, value) in &centrality {
            assert!((0.0..=1.0).contains(value));
        }
    }

    #[test]
    fn test_closeness_empty_graph() {
        let graph = Graph::<String, ()>::undirected();
        let plugin = ClosenessCentralityPlugin::default();

        let centrality = plugin.compute(&graph).unwrap();
        assert!(centrality.is_empty());
    }

    #[test]
    fn test_closeness_top_k() {
        let graph = create_line_graph();
        let plugin = ClosenessCentralityPlugin::standard();

        let top_nodes = plugin.top_k(&graph, 1).unwrap();

        assert_eq!(top_nodes.len(), 1);

        // 最高的应该是中间节点 C
        let (top_id, _) = top_nodes[0];
        assert_eq!(top_id, 2);
    }

    #[test]
    fn test_closeness_plugin_info() {
        let plugin = ClosenessCentralityPlugin::default();
        let info = plugin.info();

        assert_eq!(info.name, "closeness-centrality");
        assert_eq!(info.version, "1.0.0");
        assert!(info.tags.contains(&"centrality".to_string()));
    }
}
