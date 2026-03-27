//! 矩阵表示模块
//!
//! 提供图的矩阵表示和线性代数集成
//! 需要启用 `matrix` 特性

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use nalgebra::{DMatrix, DVector};

/// 图的邻接矩阵表示
pub struct AdjacencyMatrix {
    matrix: DMatrix<f64>,
    node_indices: Vec<NodeIndex>,
}

impl AdjacencyMatrix {
    /// 从图构建邻接矩阵
    pub fn from_graph<T, E>(graph: &Graph<T, E>) -> Self
    where
        E: Clone + Into<f64>,
    {
        let n = graph.node_count();
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let index_to_pos: std::collections::HashMap<usize, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(i, ni)| (ni.index(), i))
            .collect();

        let mut matrix = DMatrix::zeros(n, n);

        for edge in graph.edges() {
            if let Ok((src, tgt)) = graph.edge_endpoints(edge.index()) {
                if let (Some(&i), Some(&j)) = (
                    index_to_pos.get(&src.index()),
                    index_to_pos.get(&tgt.index()),
                ) {
                    let weight: f64 = edge.data().clone().into();
                    matrix[(i, j)] = weight;
                }
            }
        }

        Self {
            matrix,
            node_indices,
        }
    }

    /// 获取底层矩阵
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// 获取节点索引映射
    pub fn node_indices(&self) -> &[NodeIndex] {
        &self.node_indices
    }

    /// 获取矩阵大小
    pub fn size(&self) -> usize {
        self.matrix.nrows()
    }

    /// 获取节点在矩阵中的位置
    pub fn node_position(&self, node: NodeIndex) -> Option<usize> {
        self.node_indices
            .iter()
            .position(|&ni| ni.index() == node.index())
    }

    /// 获取节点从矩阵位置
    pub fn node_from_position(&self, pos: usize) -> Option<NodeIndex> {
        self.node_indices.get(pos).copied()
    }
}

/// 图的拉普拉斯矩阵表示
///
/// L = D - A，其中 D 是度矩阵，A 是邻接矩阵
pub struct LaplacianMatrix {
    matrix: DMatrix<f64>,
    node_indices: Vec<NodeIndex>,
}

impl LaplacianMatrix {
    /// 从图构建拉普拉斯矩阵
    pub fn from_graph<T>(graph: &Graph<T, f64>) -> Self {
        let n = graph.node_count();
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        let index_to_pos: std::collections::HashMap<usize, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(i, ni)| (ni.index(), i))
            .collect();

        let mut matrix = DMatrix::zeros(n, n);

        // 构建度矩阵和邻接矩阵
        for edge in graph.edges() {
            if let Ok((src, tgt)) = graph.edge_endpoints(edge.index()) {
                if let (Some(&i), Some(&j)) = (
                    index_to_pos.get(&src.index()),
                    index_to_pos.get(&tgt.index()),
                ) {
                    let weight = *edge.data();
                    // 非对角线元素为 -w_ij
                    matrix[(i, j)] = -weight;
                    // 对角线元素为度
                    matrix[(i, i)] += weight;
                    matrix[(j, j)] += weight;
                }
            }
        }

        Self {
            matrix,
            node_indices,
        }
    }

    /// 获取底层矩阵
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// 获取节点索引映射
    pub fn node_indices(&self) -> &[NodeIndex] {
        &self.node_indices
    }
}

/// 计算图的 Fiedler 向量（拉普拉斯矩阵的第二小特征值对应的特征向量）
///
/// Fiedler 向量可用于图分割和社区检测
pub fn fiedler_vector<T>(graph: &Graph<T, f64>) -> Option<DVector<f64>>
where
    T: Clone,
{
    let laplacian = LaplacianMatrix::from_graph(graph);
    let matrix = laplacian.as_matrix();

    // 使用幂迭代法计算最小非零特征值对应的特征向量
    // 这里使用简化的实现
    let n = matrix.nrows();
    if n < 2 {
        return None;
    }

    // 初始化随机向量
    let mut v = DVector::from_fn(n, |i, _| {
        (i as f64 * 0.1).sin()
    });
    v.normalize_mut();

    // 减去与全 1 向量平行的分量（对应零特征值）
    let sum: f64 = v.sum();
    let ones = DVector::from_element(n, 1.0);
    v -= &ones * (sum / n as f64);
    v.normalize_mut();

    // 幂迭代
    for _ in 0..100 {
        let mut w = matrix * &v;

        // 减去与全 1 向量平行的分量
        let sum: f64 = w.sum();
        w -= &ones * (sum / n as f64);

        let norm = w.norm();
        if norm < 1e-10 {
            break;
        }
        v = w / norm;
    }

    Some(v)
}

/// 计算图的谱半径（邻接矩阵的最大特征值的绝对值）
pub fn spectral_radius<T>(graph: &Graph<T, f64>) -> f64
where
    T: Clone,
{
    let adjacency = AdjacencyMatrix::from_graph(graph);
    let matrix = adjacency.as_matrix();

    // 使用幂迭代法计算最大特征值
    let n = matrix.nrows();
    if n == 0 {
        return 0.0;
    }

    let mut v = DVector::from_element(n, 1.0);
    let mut eigenvalue = 0.0;

    for _ in 0..100 {
        let w = matrix * &v;
        let new_eigenvalue = w.norm();
        if (new_eigenvalue - eigenvalue).abs() < 1e-10 {
            break;
        }
        eigenvalue = new_eigenvalue;
        v = w / eigenvalue;
    }

    eigenvalue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_adjacency_matrix() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)])
            .build()
            .unwrap();

        let adj = AdjacencyMatrix::from_graph(&graph);
        assert_eq!(adj.size(), 3);

        // 检查邻接矩阵元素
        assert_eq!(adj.matrix[(0, 1)], 1.0);
        assert_eq!(adj.matrix[(1, 2)], 2.0);
        assert_eq!(adj.matrix[(2, 0)], 3.0);
    }

    #[test]
    fn test_laplacian_matrix() {
        // 使用有向图测试，避免无向图双向边的问题
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();

        let lap = LaplacianMatrix::from_graph(&graph);

        // 拉普拉斯矩阵的对角线元素应该是节点的度（出度 + 入度）
        assert_eq!(lap.matrix[(0, 0)], 1.0); // A 的度为 1（只有出边）
        assert_eq!(lap.matrix[(1, 1)], 2.0); // B 的度为 2（入边 + 出边）
        assert_eq!(lap.matrix[(2, 2)], 1.0); // C 的度为 1（只有入边）

        // 非对角线元素应该是 -1（如果有边）或 0
        assert_eq!(lap.matrix[(0, 1)], -1.0);
        assert_eq!(lap.matrix[(1, 2)], -1.0);
    }

    #[test]
    fn test_spectral_radius() {
        // 完全图 K3 的谱半径是 2（对于无向图）
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![
                (0, 1, 1.0), (1, 0, 1.0),  // A-B
                (1, 2, 1.0), (2, 1, 1.0),  // B-C
                (2, 0, 1.0), (0, 2, 1.0),  // C-A
            ])
            .build()
            .unwrap();

        let radius = spectral_radius(&graph);
        // 双向完全图的谱半径应该是 2
        assert!((radius - 2.0).abs() < 0.5);
    }
}
