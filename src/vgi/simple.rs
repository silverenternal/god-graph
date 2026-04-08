//! SimpleGraph - 简化版 VGI 封装
//!
//! 提供高级 API，降低使用门槛，特别适合 AI Agent 快速生成代码
//!
//! # 设计理念
//!
//! - **简化 API**: 只暴露 10 个核心方法，其他通过 trait 默认实现
//! - **零学习成本**: API 设计直观，符合直觉
//! - **算法集成**: 常用算法直接作为方法提供
//!
//! # 使用示例
//!
//! ```rust
//! use god_graph::vgi::simple::SimpleGraph;
//!
//! // 创建图
//! let mut graph = SimpleGraph::<String, f64>::directed();
//!
//! // 添加节点
//! let a = graph.add_node("Alice".to_string());
//! let b = graph.add_node("Bob".to_string());
//!
//! // 添加边
//! graph.add_edge(a, b, 1.0);
//!
//! // 运行算法
//! let ranks = graph.pagerank(0.85, 20);
//! ```

use crate::algorithms::centrality::{pagerank, degree_centrality};
use crate::algorithms::community::connected_components;
use crate::algorithms::properties::{is_connected, has_cycle, density, is_dag};
use crate::algorithms::shortest_path::dijkstra;
use crate::algorithms::traversal::bfs;
use crate::export::to_dot;
use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};
use crate::node::{NodeIndex, NodeRef};
use crate::vgi::traits::VirtualGraph;
use std::collections::HashMap;
use std::hash::Hash;

/// 简化版图封装
///
/// 封装 `Graph<T, E>`，提供简化的高级 API
pub struct SimpleGraph<T, E> {
    inner: Graph<T, E>,
}

impl<T, E> SimpleGraph<T, E>
where
    T: Clone,
    E: Clone,
{
    /// 创建有向图
    pub fn directed() -> Self {
        Self {
            inner: Graph::<T, E>::directed(),
        }
    }

    /// 创建无向图
    pub fn undirected() -> Self {
        Self {
            inner: Graph::<T, E>::undirected(),
        }
    }

    /// 添加节点
    ///
    /// # Arguments
    ///
    /// * `data` - 节点数据
    ///
    /// # Returns
    ///
    /// 返回新添加节点的索引
    ///
    /// # Panics
    ///
    /// 如果节点索引溢出（极少见，需要添加超过 2^32 个节点）
    ///
    /// # Design Note
    ///
    /// 此方法在溢出时 panic 而非返回 `Result`，是因为：
    /// 1. 节点索引溢出在实际应用中几乎不可能发生（需要 2^32 个节点）
    /// 2. 简化 API 调用，避免不必要的错误处理
    /// 3. 如果确实需要处理溢出，请使用底层的 `Graph` 类型
    pub fn add_node(&mut self, data: T) -> NodeIndex {
        // SAFETY: GraphOps::add_node 只在索引溢出时返回 Err，
        // 实际使用中几乎不可能触发（需要 2^32 个节点）
        GraphOps::add_node(&mut self.inner, data)
            .expect("节点索引溢出：无法添加超过 2^32 个节点")
    }

    /// 添加边
    ///
    /// # Arguments
    ///
    /// * `from` - 源节点索引
    /// * `to` - 目标节点索引
    /// * `data` - 边数据
    ///
    /// # Returns
    ///
    /// 如果边添加成功，返回边的索引；如果边已存在或无效，返回 `None`
    ///
    /// # Note
    ///
    /// 此方法返回 `Option` 而非 `Result` 是为了简化 API。
    /// 如果需要详细的错误信息，请使用底层的 `Graph` 类型。
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, data: E) -> Option<EdgeIndex> {
        GraphOps::add_edge(&mut self.inner, from, to, data).ok()
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        GraphBase::node_count(&self.inner)
    }

    /// 获取边数量
    pub fn edge_count(&self) -> usize {
        GraphBase::edge_count(&self.inner)
    }

    /// 获取节点数据
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点存在，返回节点数据引用；否则返回 `None`
    ///
    /// # Note
    ///
    /// 此方法返回 `Option` 而非 `Result` 是为了简化 API。
    pub fn get_node(&self, index: NodeIndex) -> Option<&T> {
        GraphQuery::get_node(&self.inner, index)
    }

    /// 获取节点数据的可变引用
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点存在，返回节点数据可变引用；否则返回 `None`
    ///
    /// # Note
    ///
    /// 此方法返回 `Option` 而非 `Result` 是为了简化 API。
    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut T> {
        VirtualGraph::get_node_mut(&mut self.inner, index).ok()
    }

    /// 检查是否包含节点
    pub fn contains_node(&self, index: NodeIndex) -> bool {
        GraphQuery::get_node(&self.inner, index).is_some()
    }

    /// 获取所有节点索引
    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        GraphQuery::nodes(&self.inner).map(|n| n.index())
    }

    /// 获取邻居节点索引
    pub fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        GraphQuery::neighbors(&self.inner, node)
    }
}

// 算法实现需要 Send + Sync
impl<T, E> SimpleGraph<T, E>
where
    T: Clone + Send + Sync,
    E: Clone,
{

    // ==================== 集成算法 ====================

    /// PageRank 算法
    ///
    /// # 参数
    /// * `damping` - 阻尼系数（通常 0.85）
    /// * `iterations` - 迭代次数
    ///
    /// # 返回
    /// 每个节点的 PageRank 分数（节点索引 -> 分数）
    pub fn pagerank(&self, damping: f64, iterations: usize) -> HashMap<usize, f64> {
        let ranks = pagerank(&self.inner, damping, iterations);
        ranks
            .into_iter()
            .map(|(idx, rank)| (idx.index(), rank))
            .collect()
    }

    /// 度中心性
    ///
    /// # 返回
    /// 每个节点的度中心性（节点索引 -> 中心性）
    pub fn degree_centrality(&self) -> HashMap<usize, f64> {
        let centrality = degree_centrality(&self.inner);
        centrality
            .into_iter()
            .enumerate()
            .collect()
    }

    /// 连通分量
    ///
    /// # 返回
    /// 每个连通分量的节点索引列表
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let components = connected_components(&self.inner);
        components
            .into_iter()
            .map(|comp| comp.into_iter().map(|n| n.index()).collect())
            .collect()
    }

    /// 最短路径（Dijkstra）
    ///
    /// # 参数
    /// * `start` - 起始节点索引
    ///
    /// # 返回
    /// 从起点到所有节点的最短距离（节点索引 -> 距离）
    /// 如果边数据是 f64，直接使用权重；否则假设所有边权重为 1.0
    pub fn shortest_paths(&self, start: NodeIndex) -> HashMap<usize, f64>
    where
        E: Into<f64> + Copy,
    {
        let distances = dijkstra(&self.inner, start, |_, _, edge_data| {
            (*edge_data).into()
        }).unwrap_or_default();
        distances
            .iter()
            .map(|(&idx, &dist)| (idx.index(), dist))
            .collect()
    }

    /// BFS 遍历
    ///
    /// # 参数
    /// * `start` - 起始节点索引
    ///
    /// # 返回
    /// 按 BFS 顺序访问的节点索引列表
    pub fn bfs_order(&self, start: NodeIndex) -> Vec<usize> {
        let mut order = Vec::new();
        bfs(&self.inner, start, |node, _depth| {
            order.push(node.index());
            true
        });
        order
    }

    // ==================== 图属性检查 ====================

    /// 检查图是否连通
    pub fn is_connected(&self) -> bool {
        is_connected(&self.inner)
    }

    /// 检查图是否有环
    pub fn has_cycle(&self) -> bool {
        has_cycle(&self.inner)
    }

    /// 检查图是否是 DAG（有向无环图）
    pub fn is_dag(&self) -> bool {
        is_dag(&self.inner)
    }

    /// 计算图密度
    pub fn density(&self) -> f64 {
        density(&self.inner)
    }

    // ==================== 导出 ====================

    /// 导出为 DOT 格式（Graphviz）
    pub fn to_dot(&self) -> String
    where
        T: std::fmt::Display,
        E: std::fmt::Display,
    {
        to_dot(&self.inner)
    }
}

impl<T, E> Default for SimpleGraph<T, E>
where
    T: Clone,
    E: Clone,
{
    fn default() -> Self {
        Self::directed()
    }
}

impl<T, E> From<Graph<T, E>> for SimpleGraph<T, E> {
    fn from(graph: Graph<T, E>) -> Self {
        Self { inner: graph }
    }
}

impl<T, E> From<SimpleGraph<T, E>> for Graph<T, E> {
    fn from(simple: SimpleGraph<T, E>) -> Self {
        simple.inner
    }
}

// 为 Hash + Eq 约束提供额外方法
impl<T, E> SimpleGraph<T, E>
where
    T: Clone + Eq + Hash + Ord,
    E: Clone,
{
    /// 通过数据查找节点
    ///
    /// # 返回
    /// 第一个匹配数据的节点索引
    pub fn find_node_by_data(&self, data: &T) -> Option<NodeIndex> {
        GraphQuery::nodes(&self.inner)
            .find(|n: &NodeRef<'_, T>| n.data() == data)
            .map(|n: NodeRef<'_, T>| n.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_creation() {
        let mut graph = SimpleGraph::<String, f64>::directed();
        
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        
        graph.add_edge(a, b, 1.0);
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_simple_graph_pagerank() {
        let mut graph = SimpleGraph::<String, ()>::directed();

        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());

        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        let ranks = graph.pagerank(0.85, 20);

        assert_eq!(ranks.len(), 3);
        // All nodes should have similar ranks in a cycle
        let values: Vec<f64> = ranks.values().copied().collect();
        assert!((values[0] - values[1]).abs() < 0.1);
    }

    #[test]
    fn test_simple_graph_properties() {
        let mut graph = SimpleGraph::<String, ()>::undirected();

        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        
        assert!(graph.is_connected());
        assert!(!graph.has_cycle());
        assert!(graph.density() > 0.0);
    }

    #[test]
    fn test_simple_graph_shortest_path() {
        let mut graph = SimpleGraph::<String, f64>::directed();

        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());

        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 2.0);
        graph.add_edge(a, c, 5.0);

        let distances = graph.shortest_paths(a);

        assert_eq!(distances.get(&a.index()), Some(&0.0));
        assert_eq!(distances.get(&b.index()), Some(&1.0));
        assert_eq!(distances.get(&c.index()), Some(&3.0)); // via b
    }

    #[test]
    fn test_simple_graph_export() {
        let mut graph = SimpleGraph::<String, f64>::directed();
        
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        
        graph.add_edge(a, b, 1.0);
        
        let dot = graph.to_dot();
        
        assert!(dot.contains("digraph"));
        assert!(dot.contains("A"));
        assert!(dot.contains("B"));
    }
}
