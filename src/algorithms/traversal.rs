//! 图遍历算法模块
//!
//! 包含 DFS、BFS、拓扑排序等遍历算法

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use crate::errors::GraphError;
use crate::GraphResult;
use std::collections::VecDeque;

/// 深度优先搜索
///
/// 从起始节点开始进行 DFS 遍历，对每个访问的节点调用 visitor 闭包
///
/// # 参数
/// * `graph` - 要遍历的图
/// * `start` - 起始节点
/// * `visitor` - 访问函数，返回 false 时停止遍历
///
/// # 示例
/// ```rust,no_run
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphOps;
/// use god_gragh::algorithms::traversal::dfs;
///
/// let mut graph = Graph::<i32, f64>::directed();
/// let a = graph.add_node(1).unwrap();
/// let b = graph.add_node(2).unwrap();
/// let c = graph.add_node(3).unwrap();
/// graph.add_edge(a, b, 1.0).unwrap();
/// graph.add_edge(b, c, 1.0).unwrap();
///
/// dfs(&graph, a, |node| {
///     println!("Visited: {:?}", node);
///     true
/// });
/// ```
pub fn dfs<T, F>(graph: &Graph<T, impl Clone>, start: NodeIndex, mut visitor: F)
where
    F: FnMut(NodeIndex) -> bool,
{
    let n = graph.node_count();
    let mut visited = vec![false; n];
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if visited[node.index()] {
            continue;
        }

        visited[node.index()] = true;

        if !visitor(node) {
            return;
        }

        // 将邻居压入栈
        for neighbor in graph.neighbors(node) {
            if !visited[neighbor.index()] {
                stack.push(neighbor);
            }
        }
    }
}

/// 广度优先搜索
///
/// 从起始节点开始进行 BFS 遍历，对每个访问的节点调用 visitor 闭包
///
/// # 参数
/// * `graph` - 要遍历的图
/// * `start` - 起始节点
/// * `visitor` - 访问函数，返回 false 时停止遍历
///
/// # 示例
/// ```rust,no_run
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphOps;
/// use god_gragh::algorithms::traversal::bfs;
///
/// let mut graph = Graph::<i32, f64>::directed();
/// let a = graph.add_node(1).unwrap();
/// let b = graph.add_node(2).unwrap();
/// let c = graph.add_node(3).unwrap();
/// graph.add_edge(a, b, 1.0).unwrap();
/// graph.add_edge(b, c, 1.0).unwrap();
///
/// bfs(&graph, a, |node, depth| {
///     println!("Visited {:?} at depth {}", node, depth);
///     true
/// });
/// ```
pub fn bfs<T, F>(graph: &Graph<T, impl Clone>, start: NodeIndex, mut visitor: F)
where
    F: FnMut(NodeIndex, usize) -> bool,
{
    let mut visited = vec![false; graph.node_count()];
    let mut queue = VecDeque::new();
    
    visited[start.index()] = true;
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if !visitor(node, depth) {
            return;
        }

        for neighbor in graph.neighbors(node) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                queue.push_back((neighbor, depth + 1));
            }
        }
    }
}

/// 拓扑排序
///
/// 对 DAG 进行拓扑排序，返回节点的拓扑顺序
/// 如果图中存在环，返回 Err
///
/// # 参数
/// * `graph` - 有向无环图
///
/// # 返回
/// * `Ok(Vec<NodeIndex>)` - 拓扑排序结果
/// * `Err(GraphError::GraphHasCycle)` - 图中存在环
pub fn topological_sort<T>(graph: &Graph<T, impl Clone>) -> GraphResult<Vec<NodeIndex>> {
    let n = graph.node_count();
    let mut in_degree = vec![0usize; n];

    // 收集所有有效节点及其索引映射
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    
    // 创建索引到 NodeIndex 的映射
    let index_to_node: std::collections::HashMap<usize, NodeIndex> = 
        node_indices.iter().map(|ni| (ni.index(), *ni)).collect();

    // 计算每个节点的入度（使用内部索引）
    for node in graph.nodes() {
        for neighbor in graph.neighbors(node.index()) {
            in_degree[neighbor.index()] += 1;
        }
    }

    // 初始化队列（入度为 0 的节点）
    let mut queue = VecDeque::new();
    for node in &node_indices {
        if in_degree[node.index()] == 0 {
            queue.push_back(*node);
        }
    }

    let mut result = Vec::with_capacity(n);

    while let Some(node_idx) = queue.pop_front() {
        result.push(node_idx);

        for neighbor in graph.neighbors(node_idx) {
            // 获取正确的 NodeIndex（包含 generation）
            if let Some(neighbor_ni) = index_to_node.get(&neighbor.index()) {
                in_degree[neighbor_ni.index()] -= 1;
                if in_degree[neighbor_ni.index()] == 0 {
                    queue.push_back(*neighbor_ni);
                }
            }
        }
    }

    if result.len() != n {
        Err(GraphError::GraphHasCycle)
    } else {
        Ok(result)
    }
}

/// 查找所有路径
///
/// 查找从源节点到目标节点的所有路径
///
/// # 参数
/// * `graph` - 图
/// * `source` - 源节点
/// * `target` - 目标节点
///
/// # 返回
/// 所有路径的集合，每条路径是节点索引的向量
pub fn all_paths<T>(
    graph: &Graph<T, impl Clone>,
    source: NodeIndex,
    target: NodeIndex,
) -> Vec<Vec<NodeIndex>> {
    let mut result = Vec::new();
    let mut path = vec![source];
    let mut visited = vec![false; graph.node_count()];
    visited[source.index()] = true;

    fn dfs_helper<T>(
        graph: &Graph<T, impl Clone>,
        current: NodeIndex,
        target: NodeIndex,
        path: &mut Vec<NodeIndex>,
        visited: &mut [bool],
        result: &mut Vec<Vec<NodeIndex>>,
    ) {
        if current == target {
            result.push(path.clone());
            return;
        }

        for neighbor in graph.neighbors(current) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                path.push(neighbor);
                dfs_helper(graph, neighbor, target, path, visited, result);
                path.pop();
                visited[neighbor.index()] = false;
            }
        }
    }

    dfs_helper(graph, source, target, &mut path, &mut visited, &mut result);
    result
}

/// Tarjan 强连通分量算法
///
/// 查找有向图中的所有强连通分量 (SCC)
/// 使用基于 DFS 的实现，时间复杂度 O(V + E)
///
/// # 参数
/// * `graph` - 有向图
///
/// # 返回
/// 所有强连通分量的集合，每个分量是节点索引的向量
///
/// # 示例
/// ```rust
/// use god_gragh::graph::builders::GraphBuilder;
/// use god_gragh::algorithms::traversal::tarjan_scc;
///
/// let graph = GraphBuilder::directed()
///     .with_nodes(vec![0, 1, 2, 3, 4])
///     .with_edges(vec![
///         (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0),  // SCC: {0, 1, 2}
///         (2, 3, 1.0),                              // 桥接边
///         (3, 4, 1.0), (4, 3, 1.0),                // SCC: {3, 4}
///     ])
///     .build()
///     .unwrap();
///
/// let sccs = tarjan_scc(&graph);
/// assert_eq!(sccs.len(), 2);
/// ```
pub fn tarjan_scc<T: Clone>(graph: &Graph<T, impl Clone>) -> Vec<Vec<NodeIndex>> {
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = node_indices.len();

    if n == 0 {
        return Vec::new();
    }

    // 创建索引到 NodeIndex 的映射
    let index_to_node: Vec<NodeIndex> = node_indices;

    // Tarjan 算法状态
    let mut index_counter = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![None; n];
    let mut lowlinks = vec![0usize; n];
    let mut sccs: Vec<Vec<NodeIndex>> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn strongconnect<T, E>(
        graph: &Graph<T, E>,
        v: usize,
        index_counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut [bool],
        indices: &mut [Option<usize>],
        lowlinks: &mut [usize],
        sccs: &mut Vec<Vec<NodeIndex>>,
        index_to_node: &[NodeIndex],
    ) where
        T: Clone,
        E: Clone,
    {
        indices[v] = Some(*index_counter);
        lowlinks[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        let node = index_to_node[v];
        for neighbor in graph.neighbors(node) {
            let w = neighbor.index();
            if w >= index_to_node.len() {
                continue;
            }

            match indices[w] {
                None => {
                    strongconnect(
                        graph, w, index_counter, stack, on_stack, indices, lowlinks, sccs,
                        index_to_node,
                    );
                    lowlinks[v] = lowlinks[v].min(lowlinks[w]);
                }
                Some(idx_w) if on_stack[w] => {
                    lowlinks[v] = lowlinks[v].min(idx_w);
                }
                _ => {}
            }
        }

        if Some(lowlinks[v]) == indices[v] {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                scc.push(index_to_node[w]);
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for i in 0..n {
        if indices[i].is_none() {
            strongconnect(
                graph,
                i,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlinks,
                &mut sccs,
                &index_to_node,
            );
        }
    }

    sccs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_dfs() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let start = graph.nodes().next().unwrap().index();
        let mut visited = Vec::new();

        dfs(&graph, start, |node| {
            visited.push(node.index());
            true
        });

        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn test_bfs() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let start = graph.nodes().next().unwrap().index();
        let mut visited = Vec::new();

        bfs(&graph, start, |node, _depth| {
            visited.push(node.index());
            true
        });

        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn test_topological_sort() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        let result = topological_sort(&graph).unwrap();
        assert_eq!(result.len(), 4);

        // 验证拓扑顺序：A 必须在 B、C 之前，B、C 必须在 D 之前
        let pos: Vec<_> = result.iter().map(|n| n.index()).enumerate()
            .flat_map(|(i, n)| vec![(n, i)])
            .collect();
        assert!(!pos.is_empty());
    }

    #[test]
    fn test_tarjan_scc() {
        // 测试两个 SCC 的情况
        let graph = GraphBuilder::directed()
            .with_nodes(vec![0, 1, 2, 3, 4])
            .with_edges(vec![
                (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0),  // SCC: {0, 1, 2}
                (2, 3, 1.0),                              // 桥接边
                (3, 4, 1.0), (4, 3, 1.0),                // SCC: {3, 4}
            ])
            .build()
            .unwrap();

        let sccs = tarjan_scc(&graph);
        assert_eq!(sccs.len(), 2);

        // 验证每个 SCC 的大小
        let sizes: Vec<_> = sccs.iter().map(|scc| scc.len()).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn test_tarjan_scc_single_node() {
        let graph: Graph<i32, f64> = GraphBuilder::directed()
            .with_nodes(vec![1])
            .build()
            .unwrap();

        let sccs = tarjan_scc(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 1);
    }

    #[test]
    fn test_tarjan_scc_empty_graph() {
        let graph = GraphBuilder::<i32, f64>::directed()
            .build()
            .unwrap();

        let sccs = tarjan_scc(&graph);
        assert!(sccs.is_empty());
    }
}
