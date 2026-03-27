//! 图属性检测模块
//!
//! 包含连通性、二分性、环检测等图属性检查算法

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// 检查图是否连通
pub fn is_connected<T>(graph: &Graph<T, impl Clone>) -> bool {
    if graph.node_count() == 0 {
        return true;
    }

    let first_node = graph.nodes().next().map(|n| n.index()).unwrap();
    let mut visited = vec![false; graph.node_count()];
    let mut count = 0;

    let mut queue = VecDeque::new();
    queue.push_back(first_node);
    visited[first_node.index()] = true;

    while let Some(node) = queue.pop_front() {
        count += 1;

        for neighbor in graph.neighbors(node) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                queue.push_back(neighbor);
            }
        }
    }

    count == graph.node_count()
}

/// 检查图是否为二分图
pub fn is_bipartite<T>(graph: &Graph<T, impl Clone>) -> bool {
    let mut color: HashMap<usize, bool> = HashMap::new();

    for start in graph.nodes() {
        let start_idx = start.index();
        if color.contains_key(&start_idx.index()) {
            continue;
        }

        let mut queue = VecDeque::new();
        queue.push_back(start_idx.index());
        color.insert(start_idx.index(), true);

        while let Some(node_idx) = queue.pop_front() {
            let current_color = color[&node_idx];
            let node_ni = NodeIndex::new(node_idx, 0);

            for neighbor in graph.neighbors(node_ni) {
                let neighbor_idx = neighbor.index();
                match color.get(&neighbor_idx) {
                    Some(&c) if c == current_color => return false,
                    None => {
                        color.insert(neighbor_idx, !current_color);
                        queue.push_back(neighbor_idx);
                    }
                    _ => {}
                }
            }
        }
    }

    true
}

/// 检查图是否为 DAG（有向无环图）
pub fn is_dag<T>(graph: &Graph<T, impl Clone>) -> bool {
    // 使用拓扑排序检测环
    crate::algorithms::traversal::topological_sort(graph).is_ok()
}

/// 检测图中是否存在环
pub fn has_cycle<T>(graph: &Graph<T, impl Clone>) -> bool {
    !is_dag(graph)
}

/// 检查图是否为树
pub fn is_tree<T>(graph: &Graph<T, impl Clone>) -> bool {
    let n = graph.node_count();
    if n == 0 {
        return true;
    }

    // 树的定义：连通且边数 = 节点数 - 1
    is_connected(graph) && graph.edge_count() == n - 1
}

/// 计算图的直径（最长最短路径）
pub fn diameter<T>(graph: &Graph<T, impl Clone>) -> usize {
    if graph.node_count() == 0 {
        return 0;
    }

    let mut max_distance = 0;

    for start in graph.nodes() {
        let mut distances: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        let start_idx = start.index();
        distances.insert(start_idx.index(), 0);
        queue.push_back(start_idx);

        while let Some(node) = queue.pop_front() {
            let node_idx = node.index();
            let dist = distances[&node_idx];
            max_distance = max_distance.max(dist);

            let node_ni = NodeIndex::new(node_idx, 0);
            for neighbor in graph.neighbors(node_ni) {
                let neighbor_idx = neighbor.index();
                if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor_idx) {
                    e.insert(dist + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    max_distance
}

/// 计算图的密度
pub fn density<T>(graph: &Graph<T, impl Clone>) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 0.0;
    }

    let max_edges = n * (n - 1);
    graph.edge_count() as f64 / max_edges as f64
}

/// 计算图的 girth（最短环的长度）
///
/// Girth 是图中最短环的长度。如果图中没有环，返回 0。
///
/// # 复杂度
/// - 时间：O(V * (V + E)) - 对每个节点进行 BFS
/// - 空间：O(V)
///
/// # 示例
/// ```
/// use god_gragh::prelude::*;
/// use god_gragh::algorithms::properties::girth;
///
/// let graph = GraphBuilder::directed()
///     .with_nodes(vec!["A", "B", "C"])
///     .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
///     .build()
///     .unwrap();
///
/// let g = girth(&graph);
/// assert_eq!(g, 3); // 三角形环
/// ```
pub fn girth<T>(graph: &Graph<T, impl Clone>) -> usize {
    use std::collections::VecDeque;

    let n = graph.node_count();
    if n == 0 {
        return 0;
    }

    let node_indices: Vec<_> = graph.nodes().map(|n| n.index()).collect();
    let mut min_cycle = usize::MAX;

    // 对每个节点进行 BFS，寻找回到自身的最短路径
    for start in &node_indices {
        let mut dist = vec![usize::MAX; n];
        let mut queue = VecDeque::new();

        dist[start.index()] = 0;
        queue.push_back(*start);

        while let Some(u) = queue.pop_front() {
            if dist[u.index()] >= min_cycle {
                continue; // 剪枝：当前路径已经超过已知最短环
            }

            for v in graph.neighbors(u) {
                let new_dist = dist[u.index()] + 1;

                // 如果 v 已经访问过，且不是 u 的直接父节点，则找到环
                if dist[v.index()] != usize::MAX {
                    if dist[v.index()] != dist[u.index()] - 1 {
                        let cycle_len = new_dist + dist[v.index()];
                        min_cycle = min_cycle.min(cycle_len);
                    }
                } else {
                    dist[v.index()] = new_dist;
                    queue.push_back(v);
                }
            }
        }
    }

    if min_cycle == usize::MAX {
        0
    } else {
        min_cycle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_is_connected() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();
        assert!(is_connected(&graph));
    }

    #[test]
    fn test_is_bipartite() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();
        assert!(is_bipartite(&graph));
    }

    #[test]
    fn test_is_dag() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();
        assert!(is_dag(&graph));
    }

    #[test]
    fn test_density() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();
        let d = density(&graph);
        assert!(d > 0.0 && d < 1.0);
    }

    #[test]
    fn test_diameter() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3, 4, 5])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)])
            .build()
            .unwrap();
        
        // 线性图 A->B->C->D->E 的直径是 4
        let diam = diameter(&graph);
        assert_eq!(diam, 4);
    }

    #[test]
    fn test_girth_triangle() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
            .build()
            .unwrap();
        
        // 三角形环的长度是 3
        let g = girth(&graph);
        assert_eq!(g, 3);
    }

    #[test]
    fn test_girth_no_cycle() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();
        
        // 无环图的 girth 是 0
        let g = girth(&graph);
        assert_eq!(g, 0);
    }

    #[test]
    fn test_has_cycle() {
        let graph_with_cycle = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
            .build()
            .unwrap();
        assert!(has_cycle(&graph_with_cycle));

        let graph_without_cycle = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();
        assert!(!has_cycle(&graph_without_cycle));
    }

    #[test]
    fn test_is_tree() {
        let tree = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)])
            .build()
            .unwrap();
        assert!(is_tree(&tree));

        let not_tree = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
            .build()
            .unwrap();
        assert!(!is_tree(&not_tree));
    }
}
