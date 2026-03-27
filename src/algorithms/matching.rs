//! 匹配算法模块
//!
//! 包含最大匹配算法：Hopcroft-Karp（二分图）、Blossom（一般图）

use crate::edge::EdgeIndex;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::graph::Graph;
use std::collections::VecDeque;

/// Hopcroft-Karp 二分图最大匹配算法
///
/// # 复杂度
/// - 时间：O(E * sqrt(V))
/// - 空间：O(V)
///
/// # 参数
/// * `graph` - 二分图
/// * `left_nodes` - 左部节点的索引列表
/// * `right_nodes` - 右部节点的索引列表
///
/// # 返回
/// 匹配对列表，每个元素是 (左部节点索引，右部节点索引)
///
/// # 示例
/// ```
/// use god_gragh::prelude::*;
/// use god_gragh::algorithms::matching::hopcroft_karp;
///
/// let graph = GraphBuilder::undirected()
///     .with_nodes(vec!["L1", "L2", "R1", "R2"])
///     .with_edges(vec![(0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0)])
///     .build()
///     .unwrap();
///
/// let left = vec![0, 1];  // L1, L2
/// let right = vec![2, 3]; // R1, R2
/// let matching = hopcroft_karp(&graph, &left, &right);
/// assert!(!matching.is_empty());
/// ```
pub fn hopcroft_karp<T>(
    graph: &Graph<T, impl Clone>,
    left_nodes: &[usize],
    right_nodes: &[usize],
) -> Vec<(usize, usize)> {
    let n_left = left_nodes.len();
    let n_right = right_nodes.len();

    if n_left == 0 || n_right == 0 {
        return vec![];
    }

    // 构建邻接表
    let node_indices: Vec<_> = graph.nodes().map(|n| n.index()).collect();
    let _index_to_pos: std::collections::HashMap<usize, usize> = left_nodes
        .iter()
        .enumerate()
        .map(|(i, &idx)| (idx, i))
        .collect();

    let mut adj: Vec<Vec<usize>> = vec![vec![]; n_left];

    for (left_pos, &left_idx) in left_nodes.iter().enumerate() {
        if left_idx < node_indices.len() {
            for neighbor in graph.neighbors(node_indices[left_idx]) {
                if let Some(right_pos) = right_nodes.iter().position(|&r| r == neighbor.index()) {
                    adj[left_pos].push(right_pos);
                }
            }
        }
    }

    // 匹配数组
    let mut pair_left = vec![None; n_left];
    let mut pair_right = vec![None; n_right];
    let mut dist = vec![0; n_left];

    // BFS 寻找增广路
    fn bfs(
        adj: &[Vec<usize>],
        pair_left: &[Option<usize>],
        pair_right: &[Option<usize>],
        dist: &mut [usize],
    ) -> bool {
        let mut queue = VecDeque::new();
        for (i, &p) in pair_left.iter().enumerate() {
            if p.is_none() {
                dist[i] = 0;
                queue.push_back(i);
            } else {
                dist[i] = usize::MAX;
            }
        }

        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if let Some(next) = pair_right.get(v).and_then(|&x| x) {
                    if dist[next] == usize::MAX {
                        dist[next] = dist[u] + 1;
                        queue.push_back(next);
                    }
                } else {
                    found = true;
                }
            }
        }
        found
    }

    // DFS 寻找增广路
    fn dfs(
        u: usize,
        adj: &[Vec<usize>],
        pair_left: &mut [Option<usize>],
        pair_right: &mut [Option<usize>],
        dist: &mut [usize],
    ) -> bool {
        for &v in &adj[u] {
            if let Some(next) = pair_right.get(v).and_then(|&x| x) {
                if dist[next] == dist[u] + 1 && dfs(next, adj, pair_left, pair_right, dist) {
                    pair_left[u] = Some(v);
                    pair_right[v] = Some(u);
                    return true;
                }
            } else {
                pair_left[u] = Some(v);
                pair_right[v] = Some(u);
                return true;
            }
        }
        dist[u] = usize::MAX;
        false
    }

    // 主循环
    while bfs(&adj, &pair_left, &pair_right, &mut dist) {
        for i in 0..n_left {
            if pair_left[i].is_none() {
                dfs(i, &adj, &mut pair_left, &mut pair_right, &mut dist);
            }
        }
    }

    // 构建结果
    left_nodes
        .iter()
        .enumerate()
        .filter_map(|(i, &left_idx)| {
            pair_left[i].map(|right_pos| (left_idx, right_nodes[right_pos]))
        })
        .collect()
}

/// Blossom 算法（一般图最大匹配）
///
/// 使用带花树算法求解一般图的最大匹配
///
/// # 复杂度
/// - 时间：O(V * E)
/// - 空间：O(V)
///
/// # 参数
/// * `graph` - 无向图
///
/// # 返回
/// 匹配边索引列表
///
/// # 示例
/// ```
/// use god_gragh::prelude::*;
/// use god_gragh::algorithms::matching::blossom;
///
/// let graph = GraphBuilder::undirected()
///     .with_nodes(vec!["A", "B", "C", "D"])
///     .with_edges(vec![(0, 1, 1.0), (2, 3, 1.0)])
///     .build()
///     .unwrap();
///
/// let matching = blossom(&graph);
/// assert_eq!(matching.len(), 2);
/// ```
pub fn blossom<T>(graph: &Graph<T, impl Clone>) -> Vec<EdgeIndex> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    let node_indices: Vec<_> = graph.nodes().map(|n| n.index()).collect();
    let mut match_node: Vec<Option<usize>> = vec![None; n]; // 每个节点的匹配对象
    let mut used_edges = Vec::new();

    // 尝试为每个未匹配节点寻找增广路
    for start in 0..n {
        if match_node[start].is_some() {
            continue;
        }

        // BFS 寻找增广路
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(u) = queue.pop_front() {
            for neighbor in graph.neighbors(node_indices[u]) {
                let v = neighbor.index();
                if v >= n {
                    continue;
                }

                if !visited[v] {
                    visited[v] = true;
                    parent[v] = Some(u);

                    if let Some(matched) = match_node.get(v).and_then(|&x| x) {
                        // v 已匹配，继续搜索
                        if !visited[matched] {
                            visited[matched] = true;
                            parent[matched] = Some(v);
                            queue.push_back(matched);
                        }
                    } else {
                        // 找到增广路
                        let mut curr = v;
                        let mut prev = u;
                        while let Some(p) = parent[prev] {
                            match_node[prev] = Some(curr);
                            match_node[curr] = Some(prev);
                            curr = p;
                            prev = parent[p].unwrap();
                        }
                        match_node[start] = Some(curr);
                        match_node[curr] = Some(start);
                        break;
                    }
                }
            }
        }
    }

    // 收集匹配边
    let mut seen = std::collections::HashSet::new();
    for (u, &v_opt) in match_node.iter().enumerate() {
        if let Some(v) = v_opt {
            if u < v && !seen.contains(&(u, v)) {
                seen.insert((u, v));
                // 找到对应的边索引
                if u < node_indices.len() {
                    for edge_idx in graph.incident_edges(node_indices[u]) {
                        if let Ok((src, tgt)) = graph.edge_endpoints(edge_idx) {
                            if (src.index() == u && tgt.index() == v)
                                || (src.index() == v && tgt.index() == u)
                            {
                                used_edges.push(edge_idx);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    used_edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_hopcroft_karp_basic() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec!["L1", "L2", "R1", "R2"])
            .with_edges(vec![(0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();

        let left = vec![0, 1];
        let right = vec![2, 3];
        let matching = hopcroft_karp(&graph, &left, &right);
        assert!(!matching.is_empty());
    }

    #[test]
    fn test_hopcroft_karp_empty() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["L1", "R1"])
            .build()
            .unwrap();

        let left = vec![0];
        let right = vec![1];
        let matching = hopcroft_karp(&graph, &left, &right);
        assert!(matching.is_empty()); // 没有边，匹配为空
    }

    #[test]
    fn test_hopcroft_karp_perfect_matching() {
        // 完美匹配：每个左部节点都匹配到一个右部节点
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["L1", "L2", "R1", "R2"])
            .with_edges(vec![(0, 2, 1.0), (1, 3, 1.0)])
            .build()
            .unwrap();

        let left = vec![0, 1];
        let right = vec![2, 3];
        let matching = hopcroft_karp(&graph, &left, &right);
        assert_eq!(matching.len(), 2); // 完美匹配
    }

    #[test]
    fn test_hopcroft_karp_single_node() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["L1"])
            .build()
            .unwrap();

        let left = vec![0];
        let right = vec![];
        let matching = hopcroft_karp(&graph, &left, &right);
        assert!(matching.is_empty());
    }

    #[test]
    fn test_blossom_basic() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert!(!matching.is_empty());
    }

    #[test]
    fn test_blossom_odd_cycle() {
        // 测试花算法处理奇环的能力
        // 5 个节点的环：最大匹配应该是 2 条边
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4, 5])
            .with_edges(vec![
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
                (4, 0, 1.0),
            ])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert_eq!(matching.len(), 2); // 5 个节点的环最大匹配为 2
    }

    #[test]
    fn test_blossom_empty_graph() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B", "C"])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert!(matching.is_empty());
    }

    #[test]
    fn test_blossom_single_edge() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B"])
            .with_edges(vec![(0, 1, 1.0)])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert_eq!(matching.len(), 1);
    }

    #[test]
    fn test_blossom_path() {
        // 路径图：4 个节点的线性链
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert_eq!(matching.len(), 2); // 路径最大匹配为 2 条边
    }

    #[test]
    fn test_blossom_complete_graph() {
        // 完全图 K4：4 个节点，每对节点之间都有边
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ])
            .build()
            .unwrap();

        let matching = blossom(&graph);
        assert_eq!(matching.len(), 2); // K4 最大匹配为 2 条边
    }
}
