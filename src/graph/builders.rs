//! 图构建器模块
//!
//! 提供流式 API 用于构建图

use crate::graph::Graph;
use crate::graph::traits::GraphOps;
use crate::errors::{GraphError, GraphResult};

/// 图构建器
///
/// 提供流式 API 用于方便地构建图
pub struct GraphBuilder<T, E> {
    graph: Graph<T, E>,
    nodes: Vec<T>,
    edges: Vec<(usize, usize, E)>,
}

impl<T, E> GraphBuilder<T, E>
where
    T: Clone,
    E: Clone,
{
    /// 创建新的有向图构建器
    pub fn directed() -> Self {
        Self {
            graph: Graph::directed(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// 创建新的无向图构建器
    pub fn undirected() -> Self {
        Self {
            graph: Graph::undirected(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// 添加节点数据
    pub fn with_node(mut self, data: T) -> Self {
        self.nodes.push(data);
        self
    }

    /// 批量添加节点
    pub fn with_nodes<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        self.nodes.extend(iter);
        self
    }

    /// 添加边
    pub fn with_edge(mut self, from: usize, to: usize, data: E) -> Self {
        self.edges.push((from, to, data));
        self
    }

    /// 批量添加边
    pub fn with_edges<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, E)>,
    {
        self.edges.extend(iter);
        self
    }

    /// 构建图
    pub fn build(mut self) -> GraphResult<Graph<T, E>> {
        // 先添加所有节点
        let mut node_indices = Vec::with_capacity(self.nodes.len());
        for data in self.nodes {
            let idx = self.graph.add_node(data)?;
            node_indices.push(idx);
        }

        // 再添加所有边
        for (from, to, data) in self.edges {
            if from >= node_indices.len() || to >= node_indices.len() {
                return Err(GraphError::IndexOutOfBounds {
                    index: from.max(to),
                    bound: node_indices.len(),
                });
            }
            self.graph.add_edge(node_indices[from], node_indices[to], data)?;
        }

        Ok(self.graph)
    }
}

impl<T, E> Default for GraphBuilder<T, E>
where
    T: Clone,
    E: Clone,
{
    fn default() -> Self {
        Self::directed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphBase;

    #[test]
    fn test_builder_basic() {
        let graph = GraphBuilder::directed()
            .with_node("A")
            .with_node("B")
            .with_node("C")
            .with_edge(0, 1, 1.0)
            .with_edge(1, 2, 2.0)
            .build()
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_builder_with_nodes() {
        let graph: Graph<i32, f64> = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3, 4])
            .build()
            .unwrap();

        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn test_builder_with_edges() {
        use crate::graph::traits::GraphBase;
        
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])
            .build()
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }
}
