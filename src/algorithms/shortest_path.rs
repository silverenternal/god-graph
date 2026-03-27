//! 最短路径算法模块
//!
//! 包含 Dijkstra、Bellman-Ford、Floyd-Warshall、A* 等算法

use crate::errors::{GraphError, GraphResult};
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::graph::Graph;
use crate::node::NodeIndex;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Dijkstra 最短路径算法
///
/// 计算从源节点到所有其他节点的最短路径距离
/// 适用于非负权重的图
///
/// # 参数
/// * `graph` - 图
/// * `source` - 源节点
/// * `get_weight` - 获取边权重的闭包
///
/// # 返回
/// HashMap，键为节点索引，值为最短距离
///
/// # 错误
/// * `GraphError::NegativeWeight` - 检测到负权重边，建议使用 Bellman-Ford 算法
///
/// # 注意
/// Dijkstra 算法不适用于负权重图。如果图可能包含负权重，请使用 `bellman_ford` 算法。
pub fn dijkstra<T, E, F>(
    graph: &Graph<T, E>,
    source: NodeIndex,
    mut get_weight: F,
) -> GraphResult<HashMap<NodeIndex, f64>>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    // 检测负权重边
    for edge in graph.edges() {
        let u = edge.source();
        let v = edge.target();
        let weight = get_weight(u, v, edge.data());
        if weight < 0.0 {
            return Err(GraphError::NegativeWeight {
                from: u.index(),
                to: v.index(),
                weight,
            });
        }
    }

    // 优先队列项：(节点，距离)，使用 Reverse 实现最小堆
    struct State {
        node: NodeIndex,
        distance: f64,
    }

    impl PartialEq for State {
        fn eq(&self, other: &Self) -> bool {
            self.distance == other.distance
        }
    }

    impl Eq for State {}

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.distance.total_cmp(&self.distance)
        }
    }

    let mut distances: HashMap<NodeIndex, f64> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(State {
        node: source,
        distance: 0.0,
    });

    while let Some(State { node, distance }) = heap.pop() {
        // 跳过过期的条目
        if distance > *distances.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }

        for neighbor in graph.neighbors(node) {
            let edge_data = graph.get_edge_by_nodes(node, neighbor)?;
            let weight = get_weight(node, neighbor, edge_data);
            let new_distance = distance + weight;

            if new_distance < *distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                distances.insert(neighbor, new_distance);
                heap.push(State {
                    node: neighbor,
                    distance: new_distance,
                });
            }
        }
    }

    Ok(distances)
}

/// Bellman-Ford 算法
///
/// 计算从源节点到所有其他节点的最短路径
/// 可以处理负权重边，并能检测负权环
///
/// # 返回
/// * `Ok(HashMap)` - 最短距离
/// * `Err(GraphError::NegativeCycle)` - 检测到负权环
pub fn bellman_ford<T, E, F>(
    graph: &Graph<T, E>,
    source: NodeIndex,
    mut get_weight: F,
) -> Result<HashMap<NodeIndex, f64>, GraphError>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    let mut distances: HashMap<NodeIndex, f64> = HashMap::new();

    // 初始化距离
    for node in graph.nodes() {
        distances.insert(node.index(), f64::INFINITY);
    }
    distances.insert(source, 0.0);

    let n = graph.node_count();

    // 松弛操作，执行 n-1 轮
    for _ in 0..n - 1 {
        for edge in graph.edges() {
            let u = edge.source();
            let v = edge.target();
            let w = get_weight(u, v, edge.data());

            if distances.get(&u) != Some(&f64::INFINITY) {
                let new_dist = distances[&u] + w;
                if new_dist < *distances.get(&v).unwrap_or(&f64::INFINITY) {
                    distances.insert(v, new_dist);
                }
            }
        }
    }

    // 检测负权环
    for edge in graph.edges() {
        let u = edge.source();
        let v = edge.target();
        let w = get_weight(u, v, edge.data());

        if distances.get(&u) != Some(&f64::INFINITY)
            && distances[&u] + w < *distances.get(&v).unwrap_or(&f64::INFINITY)
        {
            return Err(GraphError::NegativeCycle);
        }
    }

    Ok(distances)
}

/// A* 搜索算法
///
/// 使用启发式函数找到从起点到终点的最短路径
///
/// # 参数
/// * `graph` - 图
/// * `start` - 起始节点
/// * `goal` - 目标节点
/// * `get_weight` - 获取边权重的闭包
/// * `heuristic` - 启发式函数，估计节点到目标的距离
pub fn astar<T, E, F, H>(
    graph: &Graph<T, E>,
    start: NodeIndex,
    goal: NodeIndex,
    mut get_weight: F,
    mut heuristic: H,
) -> GraphResult<(f64, Vec<NodeIndex>)>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
    H: FnMut(NodeIndex) -> f64,
{
    #[derive(Debug)]
    struct State {
        node: NodeIndex,
        f_score: f64,
    }

    impl PartialEq for State {
        fn eq(&self, other: &Self) -> bool {
            self.f_score == other.f_score
        }
    }

    impl Eq for State {}

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.f_score.total_cmp(&self.f_score)
        }
    }

    let mut g_scores: HashMap<NodeIndex, f64> = HashMap::new();
    let mut came_from: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut heap = BinaryHeap::new();

    g_scores.insert(start, 0.0);
    heap.push(State {
        node: start,
        f_score: heuristic(start),
    });

    while let Some(State { node, .. }) = heap.pop() {
        if node == goal {
            // 重构路径
            let mut path = vec![goal];
            let mut current = goal;
            while let Some(&prev) = came_from.get(&current) {
                path.push(prev);
                current = prev;
            }
            path.reverse();
            return Ok((*g_scores.get(&goal).unwrap_or(&0.0), path));
        }

        let current_g = *g_scores.get(&node).unwrap_or(&f64::INFINITY);

        for neighbor in graph.neighbors(node) {
            let edge_data = graph.get_edge_by_nodes(node, neighbor)?;
            let weight = get_weight(node, neighbor, edge_data);
            let tentative_g = current_g + weight;

            if tentative_g < *g_scores.get(&neighbor).unwrap_or(&f64::INFINITY) {
                came_from.insert(neighbor, node);
                g_scores.insert(neighbor, tentative_g);
                let f_score = tentative_g + heuristic(neighbor);
                heap.push(State {
                    node: neighbor,
                    f_score,
                });
            }
        }
    }

    Err(GraphError::NodeNotFound {
        index: goal.index(),
    })
}

/// Floyd-Warshall 算法
///
/// 计算所有节点对之间的最短路径距离
/// 适用于任意权重的图（可处理负权重）
/// 时间复杂度：O(V³)，空间复杂度：O(V²)
///
/// # 参数
/// * `graph` - 图
/// * `get_weight` - 获取边权重的闭包
///
/// # 返回
/// * `Ok(HashMap<(NodeIndex, NodeIndex), f64>)` - 所有节点对的最短距离
/// * `Err(GraphError::NegativeCycle)` - 检测到负权环
///
/// # 示例
/// ```rust
/// use god_gragh::graph::builders::GraphBuilder;
/// use god_gragh::algorithms::shortest_path::floyd_warshall;
///
/// let graph = GraphBuilder::directed()
///     .with_nodes(vec!["A", "B", "C"])
///     .with_edges(vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)])
///     .build()
///     .unwrap();
///
/// let distances = floyd_warshall(&graph, |_, _, w| *w).unwrap();
/// ```
pub fn floyd_warshall<T, E, F>(
    graph: &Graph<T, E>,
    mut get_weight: F,
) -> Result<HashMap<(NodeIndex, NodeIndex), f64>, GraphError>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = node_indices.len();

    if n == 0 {
        return Ok(HashMap::new());
    }

    // 创建索引到 NodeIndex 的映射
    let index_to_node = &node_indices;
    let node_to_index: std::collections::HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    // 初始化距离矩阵
    const INF: f64 = f64::INFINITY;
    let mut dist = vec![vec![INF; n]; n];

    // 设置对角线为 0
    for (i, row) in dist.iter_mut().enumerate().take(n) {
        row[i] = 0.0;
    }

    // 设置直接边的权重
    for edge in graph.edges() {
        let u = edge.source();
        let v = edge.target();
        if let (Some(&i), Some(&j)) = (node_to_index.get(&u.index()), node_to_index.get(&v.index()))
        {
            let weight = get_weight(u, v, edge.data());
            dist[i][j] = dist[i][j].min(weight);
        }
    }

    // Floyd-Warshall 主循环
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] != INF && dist[k][j] != INF {
                    let new_dist = dist[i][k] + dist[k][j];
                    if new_dist < dist[i][j] {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }
    }

    // 检测负权环：检查对角线是否有负值
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if dist[i][i] < 0.0 {
            return Err(GraphError::NegativeCycle);
        }
    }

    // 构建结果 HashMap
    let mut result = HashMap::with_capacity(n * n);
    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &value) in row.iter().enumerate().take(n) {
            if value != INF {
                result.insert((index_to_node[i], index_to_node[j]), value);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_dijkstra_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 4.0),
                (1, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 1.0),
            ])
            .build()
            .unwrap();

        let start = NodeIndex::new(0, 1);
        let distances = dijkstra(&graph, start, |_, _, _| 1.0).unwrap();

        assert!(distances.contains_key(&start));
    }

    #[test]
    fn test_dijkstra_negative_weights() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, -2.0), (0, 2, 3.0)])
            .build()
            .unwrap();

        let start = NodeIndex::new(0, 1);
        let result = dijkstra(&graph, start, |_, _, w| *w);

        assert!(matches!(result, Err(GraphError::NegativeWeight { .. })));
    }

    #[test]
    fn test_bellman_ford_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 4.0),
                (0, 2, 2.0),
                (1, 2, 1.0),
                (1, 3, 5.0),
                (2, 3, 3.0),
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let distances = bellman_ford(&graph, source, |_, _, w| *w).unwrap();

        assert_eq!(distances.get(&source), Some(&0.0));
        assert!(distances.len() == 4);
    }

    #[test]
    fn test_bellman_ford_negative_weights() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, -2.0), (0, 2, 3.0)])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let distances = bellman_ford(&graph, source, |_, _, w| *w).unwrap();

        // A->B->C = 1 + (-2) = -1, which is shorter than A->C = 3
        assert_eq!(distances.get(&NodeIndex::new(2, 1)), Some(&-1.0));
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, -2.0), (2, 0, -3.0)])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let result = bellman_ford(&graph, source, |_, _, w| *w);

        assert!(matches!(result, Err(GraphError::NegativeCycle)));
    }

    #[test]
    fn test_astar_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 4.0),
                (1, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 1.0),
            ])
            .build()
            .unwrap();

        let start = NodeIndex::new(0, 1);
        let goal = NodeIndex::new(3, 1);

        // 使用简单的启发式函数（始终返回 0，退化为 Dijkstra）
        let (distance, path) = astar(&graph, start, goal, |_, _, _| 1.0, |_| 0.0).unwrap();

        assert!(distance > 0.0);
        assert!(!path.is_empty());
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
    }

    #[test]
    fn test_floyd_warshall_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0),
                (0, 2, 4.0),
                (1, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 1.0),
            ])
            .build()
            .unwrap();

        let distances = floyd_warshall(&graph, |_, _, w| *w).unwrap();

        // 验证节点对之间的距离
        let nodes: Vec<_> = graph.nodes().collect();
        assert_eq!(
            distances.get(&(nodes[0].index(), nodes[3].index())),
            Some(&4.0)
        ); // A->B->C->D = 1+2+1 = 4
    }

    #[test]
    fn test_floyd_warshall_negative_weights() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, -2.0), (0, 2, 3.0)])
            .build()
            .unwrap();

        let distances = floyd_warshall(&graph, |_, _, w| *w).unwrap();

        let nodes: Vec<_> = graph.nodes().collect();
        // A->B->C = 1 + (-2) = -1
        assert_eq!(
            distances.get(&(nodes[0].index(), nodes[2].index())),
            Some(&-1.0)
        );
    }

    #[test]
    fn test_floyd_warshall_negative_cycle() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, -2.0), (2, 0, -3.0)])
            .build()
            .unwrap();

        let result = floyd_warshall(&graph, |_, _, w| *w);
        assert!(matches!(result, Err(GraphError::NegativeCycle)));
    }

    #[test]
    fn test_floyd_warshall_empty_graph() {
        let graph: Graph<i32, f64> = GraphBuilder::directed().build().unwrap();
        let distances = floyd_warshall(&graph, |_, _, _: &f64| 1.0).unwrap();
        assert!(distances.is_empty());
    }
}
