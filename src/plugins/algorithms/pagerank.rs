//! PageRank 算法插件实现
//!
//! PageRank 是一种图算法，用于衡量图中节点的重要性或影响力。
//! 广泛应用于网页排名、社交网络分析、推荐系统等领域。
//!
//! # 算法原理
//!
//! PageRank 通过迭代计算每个节点的分数：
//! ```text
//! PR(i) = (1 - d) / N + d * Σ(PR(j) / out_degree(j))
//! ```
//! 其中：
//! - `d`: 阻尼系数（通常 0.85）
//! - `N`: 节点总数
//! - `j`: 所有指向节点 `i` 的节点
//!
//! # 使用示例
//!
//! ```ignore
//! use god_graph::plugins::PageRankPlugin;
//! use god_graph::graph::Graph;
//!
//! let mut graph = Graph::<String, f64>::directed();
//! // ... 添加节点和边
//!
//! let plugin = PageRankPlugin::new(0.85, 20, 1e-6);
//! let scores = plugin.compute(&graph)?;
//! ```

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;

/// PageRank 算法插件
///
/// 计算图中每个节点的 PageRank 分数，表示节点的重要性。
///
/// # 参数
///
/// * `damping`: 阻尼系数（通常 0.85），表示用户继续点击链接的概率
/// * `max_iterations`: 最大迭代次数
/// * `tolerance`: 收敛阈值，当分数变化小于此值时停止迭代
///
/// # 示例
///
/// ```ignore
/// let plugin = PageRankPlugin::new(0.85, 20, 1e-6);
/// let scores = plugin.compute(&graph)?;
/// ```
pub struct PageRankPlugin {
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl PageRankPlugin {
    /// 创建新的 PageRank 插件
    ///
    /// # 参数
    ///
    /// * `damping`: 阻尼系数（推荐 0.85）
    /// * `max_iterations`: 最大迭代次数
    /// * `tolerance`: 收敛阈值
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::PageRankPlugin;
    /// let plugin = PageRankPlugin::new(0.85, 20, 1e-6);
    /// ```
    pub fn new(damping: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            damping: damping.clamp(0.0, 1.0),
            max_iterations,
            tolerance,
        }
    }

    /// 计算图的 PageRank 分数
    ///
    /// # 参数
    ///
    /// * `graph`: 要计算的图
    ///
    /// # 返回
    ///
    /// 返回 HashMap，key 为节点索引，value 为 PageRank 分数
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let plugin = PageRankPlugin::new(0.85, 20, 1e-6);
    /// let scores = plugin.compute(&graph)?;
    /// for (node_idx, score) in &scores {
    ///     println!("Node {}: {:.6}", node_idx, score);
    /// }
    /// ```
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

        // P0 OPTIMIZATION: O(E) edge iteration instead of O(V²) pair checking
        // Build incoming edge adjacency list by iterating edges once
        // incoming_edges[pos] = Vec of (source_pos, inverse_out_degree)
        let mut incoming_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        // Single pass over all edges - O(E) instead of O(V²)
        for edge in graph.edges() {
            let src = edge.source();
            let tgt = edge.target();
            
            let src_pos = node_id_to_pos[src.index()];
            let tgt_pos = node_id_to_pos[tgt.index()];
            
            if src_pos != usize::MAX && tgt_pos != usize::MAX {
                if let Ok(out_degree) = graph.out_degree(src) {
                    if out_degree > 0 {
                        incoming_edges[tgt_pos].push((src_pos, 1.0 / out_degree as f64));
                    }
                }
            }
        }

        // 使用 Vec 代替 HashMap 存储分数
        let mut scores: Vec<f64> = vec![1.0 / n as f64; n];
        let teleport = (1.0 - self.damping) / n as f64;

        for _iteration in 0..self.max_iterations {
            let mut new_scores: Vec<f64> = vec![0.0; n];
            let mut max_diff = 0.0;

            for (pos, _idx) in node_indices.iter().enumerate() {
                // Start with teleportation component
                let mut new_score = teleport;

                // P0 OPTIMIZATION: Only iterate actual incoming edges (O(E) total)
                // Instead of checking all V² pairs
                for &(src_pos, inv_degree) in &incoming_edges[pos] {
                    new_score += self.damping * scores[src_pos] * inv_degree;
                }

                let diff = (new_score - scores[pos]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }

                new_scores[pos] = new_score;
            }

            scores = new_scores;

            if max_diff < self.tolerance {
                break;
            }
        }

        // 返回 (节点 ID, 分数) 对
        Ok(node_indices
            .iter()
            .zip(scores.iter())
            .map(|(idx, &score)| (idx.index(), score))
            .collect())
    }
}

impl GraphAlgorithm for PageRankPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("pagerank", "1.0.0", "PageRank 中心性算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
            .with_tags(&["centrality", "ranking", "iterative"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let damping = ctx.get_config_as("damping", self.damping);
        let max_iter = ctx.get_config_as("max_iter", self.max_iterations);
        let tolerance = ctx.get_config_as("tolerance", self.tolerance);

        let plugin = PageRankPlugin::new(damping, max_iter, tolerance);

        ctx.report_progress(0.1);
        let scores = plugin.compute(ctx.graph)?;
        ctx.report_progress(1.0);

        let result = AlgorithmResult::new("pagerank", AlgorithmData::NodeValues(scores.into_iter().collect()))
            .with_metadata("damping", damping.to_string())
            .with_metadata("iterations", max_iter.to_string());

        Ok(result)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Default for PageRankPlugin {
    fn default() -> Self {
        Self::new(0.85, 100, 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_pagerank_basic() {
        let mut graph = Graph::<String, f64>::directed();
        graph.add_node("A".to_string()).unwrap();
        graph.add_node("B".to_string()).unwrap();
        graph.add_node("C".to_string()).unwrap();

        let plugin = PageRankPlugin::default();
        let scores = plugin.compute(&graph).unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|(_, v)| *v > 0.0));
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = Graph::<String, f64>::directed();
        let plugin = PageRankPlugin::default();
        let scores = plugin.compute(&graph).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_pagerank_plugin_info() {
        let plugin = PageRankPlugin::default();
        let info = plugin.info();

        assert_eq!(info.name, "pagerank");
        assert_eq!(info.version, "1.0.0");
        assert!(info.tags.contains(&"centrality".to_string()));
    }
}
