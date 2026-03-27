//! 社区发现算法模块
//!
//! 包含 Label Propagation、Louvain、连通分量等算法

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// 连通分量算法（基于 BFS）
///
/// 返回所有连通分量，每个分量是节点索引的向量
pub fn connected_components<T>(graph: &Graph<T, impl Clone>) -> Vec<Vec<NodeIndex>> {
    // 收集所有有效节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    
    // 创建索引到 NodeIndex 的映射
    let index_to_node: std::collections::HashMap<usize, NodeIndex> = 
        node_indices.iter().map(|&ni| (ni.index(), ni)).collect();
    
    let n = graph.node_count();
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for &node in &node_indices {
        if !visited[node.index()] {
            let mut component = Vec::new();
            bfs_component(graph, node, &mut visited, &mut component, &index_to_node);
            components.push(component);
        }
    }

    components
}

fn bfs_component<T>(
    graph: &Graph<T, impl Clone>,
    start: NodeIndex,
    visited: &mut [bool],
    component: &mut Vec<NodeIndex>,
    index_to_node: &std::collections::HashMap<usize, NodeIndex>,
) {
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start.index()] = true;

    while let Some(node) = queue.pop_front() {
        component.push(node);

        for neighbor in graph.neighbors(node) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                // 使用映射获取正确的 NodeIndex
                if let Some(&neighbor_ni) = index_to_node.get(&neighbor.index()) {
                    queue.push_back(neighbor_ni);
                }
            }
        }
    }
}

/// 强连通分量算法（基于 Kosaraju 算法）
///
/// 返回所有强连通分量（仅适用于有向图）
pub fn strongly_connected_components<T>(graph: &Graph<T, impl Clone>) -> Vec<Vec<NodeIndex>> {
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let index_to_node: std::collections::HashMap<usize, NodeIndex> = 
        node_indices.iter().map(|&ni| (ni.index(), ni)).collect();
    
    let n = graph.node_count();
    let mut visited = vec![false; n];
    let mut finish_order = Vec::with_capacity(n);

    // 第一次 DFS，记录完成顺序
    for &node in &node_indices {
        if !visited[node.index()] {
            dfs_finish_order(graph, node, &mut visited, &mut finish_order);
        }
    }

    // 构建反向图（隐式）
    // 在反向图上按完成顺序的逆序进行 DFS
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for &node in finish_order.iter().rev() {
        if !visited[node.index()] {
            let mut component = Vec::new();
            dfs_reverse(graph, node, &mut visited, &mut component, &index_to_node);
            components.push(component);
        }
    }

    components
}

fn dfs_finish_order<T>(
    graph: &Graph<T, impl Clone>,
    node: NodeIndex,
    visited: &mut [bool],
    finish_order: &mut Vec<NodeIndex>,
) {
    visited[node.index()] = true;

    for neighbor in graph.neighbors(node) {
        if !visited[neighbor.index()] {
            dfs_finish_order(graph, neighbor, visited, finish_order);
        }
    }

    finish_order.push(node);
}

fn dfs_reverse<T>(
    graph: &Graph<T, impl Clone>,
    node: NodeIndex,
    visited: &mut [bool],
    component: &mut Vec<NodeIndex>,
    index_to_node: &std::collections::HashMap<usize, NodeIndex>,
) {
    component.push(node);
    visited[node.index()] = true;

    // 在反向图中，我们需要找到所有指向当前节点的节点
    for potential_source in graph.nodes() {
        if graph.has_edge(potential_source.index(), node) {
            let source_idx = potential_source.index();
            if !visited[source_idx.index()] {
                if let Some(&source_ni) = index_to_node.get(&source_idx.index()) {
                    dfs_reverse(graph, source_ni, visited, component, index_to_node);
                }
            }
        }
    }
}

/// Label Propagation 社区发现算法
///
/// 基于标签传播的社区发现，每个节点最终被分配到一个社区标签
pub fn label_propagation<T>(graph: &Graph<T, impl Clone>, max_iterations: usize) -> HashMap<NodeIndex, usize> {
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = node_indices.len();

    if n == 0 {
        return HashMap::new();
    }

    // 初始化：每个节点一个唯一标签
    let mut labels: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ni)| (ni, i))
        .collect();

    for _ in 0..max_iterations {
        let mut changed = false;

        for &node in &node_indices {
            // 收集邻居的标签
            let mut label_counts: HashMap<usize, usize> = HashMap::new();

            // 无向图：考虑所有邻居（出边和入边）
            for neighbor in graph.neighbors(node) {
                if let Some(&label) = labels.get(&neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            // 找到出现频率最高的标签
            if let Some((&max_label, _)) = label_counts.iter().max_by_key(|&(_, count)| count) {
                let current_label = labels.get(&node).copied().unwrap_or(usize::MAX);
                if max_label != current_label {
                    labels.insert(node, max_label);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    labels
}

/// Louvain 社区发现算法（简化版）
///
/// 基于模块度优化的社区发现算法
pub fn louvain<T>(graph: &Graph<T, impl Clone>, resolution: f64) -> HashMap<NodeIndex, usize> 
where
    T: Clone,
{
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = node_indices.len();
    
    if n == 0 {
        return HashMap::new();
    }

    // 初始化：每个节点一个社区
    let mut communities: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ni)| (ni, i))
        .collect();

    // 计算总边数
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return communities;
    }

    // 计算每个节点的度数
    let mut degrees: HashMap<NodeIndex, usize> = HashMap::new();
    for &node in &node_indices {
        degrees.insert(node, graph.out_degree(node).unwrap_or(0));
    }

    let mut improved = true;
    while improved {
        improved = false;

        for &node in &node_indices {
            let current_comm = *communities.get(&node).unwrap();
            let node_degree = *degrees.get(&node).unwrap();

            // 计算移动到每个邻居社区的模块度增益
            let mut comm_delta_q: HashMap<usize, f64> = HashMap::new();

            for neighbor in graph.neighbors(node) {
                let neighbor_comm = *communities.get(&neighbor).unwrap();
                if neighbor_comm != current_comm {
                    // 简化：假设无权重边
                    let delta_q = 1.0 / (2.0 * total_edges)
                        - resolution * (node_degree as f64 * degrees.get(&neighbor).copied().unwrap_or(0) as f64)
                            / (4.0 * total_edges * total_edges);

                    *comm_delta_q.entry(neighbor_comm).or_insert(0.0) += delta_q;
                }
            }

            // 选择最大增益的社区（使用 partial_cmp 处理 f64）
            if let Some((&best_comm, &max_delta)) = comm_delta_q.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                if max_delta > 0.0 {
                    communities.insert(node, best_comm);
                    improved = true;
                }
            }
        }
    }

    // 重新编号社区，使标签从 0 开始连续
    let mut comm_remap: HashMap<usize, usize> = HashMap::new();
    let mut next_comm = 0usize;
    for comm in communities.values_mut() {
        if !comm_remap.contains_key(comm) {
            comm_remap.insert(*comm, next_comm);
            next_comm += 1;
        }
        *comm = comm_remap[comm];
    }

    communities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_connected_components() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4, 5, 6])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)])
            .build()
            .unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 3); // {0,1,2}, {3,4}, {5}
    }

    #[test]
    fn test_connected_components_empty_graph() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3])
            .build()
            .unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 3); // 每个节点独立一个分量
        assert!(components.iter().all(|c| c.len() == 1));
    }

    #[test]
    fn test_connected_components_single_node() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1])
            .build()
            .unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 1);
    }

    #[test]
    fn test_connected_components_fully_connected() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![
                (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
                (1, 2, 1.0), (1, 3, 1.0),
                (2, 3, 1.0),
            ])
            .build()
            .unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);
    }

    #[test]
    fn test_strongly_connected_components() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let components = strongly_connected_components(&graph);
        // {0,1,2} 形成一个强连通分量，{3} 单独一个
        assert!(!components.is_empty());
    }

    #[test]
    fn test_strongly_connected_single_node() {
        let graph: Graph<i32, f64> = GraphBuilder::directed()
            .with_nodes(vec![1])
            .build()
            .unwrap();

        let components = strongly_connected_components(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 1);
    }

    #[test]
    fn test_strongly_connected_dag() {
        // DAG 没有强连通分量（除了单个节点）
        let graph: Graph<i32, f64> = GraphBuilder::directed()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let components = strongly_connected_components(&graph);
        assert_eq!(components.len(), 4); // 每个节点独立
    }

    #[test]
    fn test_label_propagation() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let labels = label_propagation(&graph, 10);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_label_propagation_empty_graph() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3])
            .build()
            .unwrap();

        let labels = label_propagation(&graph, 10);
        // 空图中每个节点保持自己的标签
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_label_propagation_single_node() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1])
            .build()
            .unwrap();

        let labels = label_propagation(&graph, 10);
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_louvain_basic() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let communities = louvain(&graph, 1.0);
        assert!(!communities.is_empty());
        assert_eq!(communities.len(), 4);
    }

    #[test]
    fn test_louvain_empty_graph() {
        let graph: Graph<i32, f64> = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3])
            .build()
            .unwrap();

        let communities = louvain(&graph, 1.0);
        // 空图中每个节点独立一个社区
        assert_eq!(communities.len(), 3);
    }
}
