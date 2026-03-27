//! 最小生成树算法模块
//!
//! 包含 Kruskal 和 Prim 算法

use crate::edge::EdgeIndex;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::graph::Graph;
use crate::node::NodeIndex;
use std::cmp::Ordering;
use std::collections::HashMap;

/// 并查集数据结构，用于 Kruskal 算法
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        true
    }
}

/// Kruskal 算法
///
/// 计算无向加权图的最小生成树
/// 返回构成 MST 的边索引列表
///
/// # 复杂度
/// - 时间：O(E log E) - 主要来自边排序
/// - 空间：O(V) - 并查集存储
///
/// # 示例
/// ```rust,no_run
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphOps;
/// use god_gragh::algorithms::mst::kruskal;
///
/// let mut graph = Graph::<&str, f64>::undirected();
/// let a = graph.add_node("A").unwrap();
/// let b = graph.add_node("B").unwrap();
/// let c = graph.add_node("C").unwrap();
/// let d = graph.add_node("D").unwrap();
/// graph.add_edge(a, b, 1.0).unwrap();
/// graph.add_edge(a, c, 4.0).unwrap();
/// graph.add_edge(b, c, 2.0).unwrap();
/// graph.add_edge(b, d, 5.0).unwrap();
/// graph.add_edge(c, d, 3.0).unwrap();
///
/// let mst_edges = kruskal(&graph);
/// assert_eq!(mst_edges.len(), 3); // n-1 条边
/// ```
pub fn kruskal<T>(graph: &Graph<T, f64>) -> Vec<EdgeIndex> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // 收集所有边及其权重
    let mut edges: Vec<(EdgeIndex, f64)> = graph
        .edges()
        .map(|edge| (edge.index(), *edge.data()))
        .collect();

    // 按权重排序
    edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // 使用并查集构建 MST
    let mut uf = UnionFind::new(n);
    let mut mst = Vec::with_capacity(n - 1);

    for (edge_idx, _weight) in edges {
        if let Ok((source, target)) = graph.edge_endpoints(edge_idx) {
            if uf.union(source.index(), target.index()) {
                mst.push(edge_idx);
                if mst.len() == n - 1 {
                    break;
                }
            }
        }
    }

    mst
}

/// Prim 算法
///
/// 计算无向加权图的最小生成树
/// 返回构成 MST 的边索引列表
///
/// # 复杂度
/// - 时间：O((V + E) log V) - 使用二叉堆
/// - 空间：O(V) - 距离数组和堆
///
/// # 示例
/// ```rust,no_run
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphOps;
/// use god_gragh::algorithms::mst::prim;
///
/// let mut graph = Graph::<&str, f64>::undirected();
/// let a = graph.add_node("A").unwrap();
/// let b = graph.add_node("B").unwrap();
/// let c = graph.add_node("C").unwrap();
/// let d = graph.add_node("D").unwrap();
/// graph.add_edge(a, b, 1.0).unwrap();
/// graph.add_edge(a, c, 4.0).unwrap();
/// graph.add_edge(b, c, 2.0).unwrap();
/// graph.add_edge(b, d, 5.0).unwrap();
/// graph.add_edge(c, d, 3.0).unwrap();
///
/// let mst_edges = prim(&graph);
/// assert_eq!(mst_edges.len(), 3); // n-1 条边
/// ```
pub fn prim<T>(graph: &Graph<T, f64>) -> Vec<EdgeIndex> {
    use std::collections::BinaryHeap;

    #[derive(Debug)]
    struct EdgeCandidate {
        weight: f64,
        target_idx: usize,
        edge_idx: EdgeIndex,
    }

    impl PartialEq for EdgeCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.weight == other.weight
        }
    }

    impl Eq for EdgeCandidate {}

    impl PartialOrd for EdgeCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for EdgeCandidate {
        fn cmp(&self, other: &Self) -> Ordering {
            other.weight.total_cmp(&self.weight)
        }
    }

    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    let mut in_mst = vec![false; n];
    let mut mst = Vec::with_capacity(n - 1);
    let mut heap = BinaryHeap::new();

    // 收集所有有效节点及其索引映射
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let index_to_pos: HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    // 从第一个节点开始
    let start_pos = 0;
    in_mst[start_pos] = true;

    // 将起始节点的边加入堆
    let start_node = node_indices[start_pos];
    for edge in graph.incident_edges(start_node) {
        if let Ok((source, target)) = graph.edge_endpoints(edge) {
            let neighbor = if source == start_node { target } else { source };
            if let Some(&pos) = index_to_pos.get(&neighbor.index()) {
                if !in_mst[pos] {
                    if let Ok(weight) = graph.get_edge(edge) {
                        heap.push(EdgeCandidate {
                            weight: *weight,
                            target_idx: pos,
                            edge_idx: edge,
                        });
                    }
                }
            }
        }
    }

    while let Some(EdgeCandidate {
        target_idx,
        edge_idx,
        weight: _,
    }) = heap.pop()
    {
        if in_mst[target_idx] {
            continue;
        }

        in_mst[target_idx] = true;
        mst.push(edge_idx);

        if mst.len() == n - 1 {
            break;
        }

        // 将新加入节点的边加入堆
        let target_node = node_indices[target_idx];
        for edge in graph.incident_edges(target_node) {
            if let Ok((source, target)) = graph.edge_endpoints(edge) {
                let neighbor = if source == target_node {
                    target
                } else {
                    source
                };
                if let Some(&pos) = index_to_pos.get(&neighbor.index()) {
                    if !in_mst[pos] {
                        if let Ok(weight) = graph.get_edge(edge) {
                            heap.push(EdgeCandidate {
                                weight: *weight,
                                target_idx: pos,
                                edge_idx: edge,
                            });
                        }
                    }
                }
            }
        }
    }

    mst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_kruskal_basic() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 4.0),
                (1, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 3.0),
            ])
            .build()
            .unwrap();

        let mst = kruskal(&graph);
        assert_eq!(mst.len(), 3); // 4 个节点需要 3 条边
    }

    #[test]
    fn test_prim_basic() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 4.0),
                (1, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 3.0),
            ])
            .build()
            .unwrap();

        let mst = prim(&graph);
        assert_eq!(mst.len(), 3);
    }
}
