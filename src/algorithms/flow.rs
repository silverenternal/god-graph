//! 最大流算法模块
//!
//! 包含 Edmonds-Karp、Dinic 等算法
//!
//! ## 内存优化
//!
//! 使用稀疏邻接表存储残量图，空间复杂度 O(V+E)：
//! - `edmonds_karp`: 使用 `HashMap<usize, Vec<(usize, f64)>>` 存储残量图
//! - `dinic`: 使用 `HashMap<usize, HashMap<usize, f64>>` 存储残量
//! - `push_relabel`: 使用 `HashMap<usize, HashMap<usize, f64>>` 存储残量
//!
//! 对于稠密图（E ≈ V²），可考虑使用邻接矩阵实现（未提供）。

use crate::graph::traits::{GraphBase, GraphQuery};
use crate::graph::Graph;
use crate::node::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// Edmonds-Karp 最大流算法
///
/// 使用 BFS 寻找增广路的 Ford-Fulkerson 方法变体
///
/// # 参数
/// * `graph` - 有向图
/// * `source` - 源点
/// * `sink` - 汇点
/// * `capacity` - 获取边容量的闭包
///
/// # 返回
/// 最大流值
///
/// # 复杂度
/// - 时间：O(V * E²)
/// - 空间：O(V + E)
///
/// # 示例
/// ```
/// use god_gragh::prelude::*;
/// use god_gragh::algorithms::flow::edmonds_karp;
///
/// let mut graph = GraphBuilder::directed()
///     .with_nodes(vec!["S", "A", "B", "T"])
///     .with_edges(vec![
///         (0, 1, 10.0), // S->A
///         (0, 2, 5.0),  // S->B
///         (1, 2, 15.0), // A->B
///         (1, 3, 10.0), // A->T
///         (2, 3, 10.0), // B->T
///     ])
///     .build()
///     .unwrap();
///
/// let source = graph.nodes().next().unwrap().index();
/// let sink = graph.nodes().nth(3).unwrap().index();
/// let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);
/// assert_eq!(max_flow, 15.0);
/// ```
pub fn edmonds_karp<T, E, F>(
    graph: &mut Graph<T, E>,
    source: NodeIndex,
    sink: NodeIndex,
    mut capacity: F,
) -> f64
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    let n = graph.node_count();
    if n == 0 {
        return 0.0;
    }

    // 收集所有节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let index_to_pos: HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    // 构建残量邻接表：使用 HashMap 优化稀疏图内存使用
    // adj[u] = [(v, capacity), ...]
    // 空间复杂度：O(V + E)，而非邻接矩阵的 O(V²)
    let mut adj: HashMap<usize, Vec<(usize, f64)>> = HashMap::with_capacity(n);

    // 初始化残量图
    for edge in graph.edges() {
        if let Ok((src, tgt)) = graph.edge_endpoints(edge.index()) {
            if let Some(&u) = index_to_pos.get(&src.index()) {
                if let Some(&v) = index_to_pos.get(&tgt.index()) {
                    let cap = capacity(src, tgt, edge.data());
                    if cap > 0.0 {
                        adj.entry(u).or_default().push((v, cap));
                        // 添加反向边（初始容量为 0）
                        adj.entry(v).or_default().push((u, 0.0));
                    }
                }
            }
        }
    }

    let mut max_flow = 0.0;

    // BFS 寻找增广路
    fn bfs(
        adj: &HashMap<usize, Vec<(usize, f64)>>,
        source_pos: usize,
        sink_pos: usize,
        n: usize,
        parent: &mut [(isize, usize)], // (parent, edge_index)
    ) -> bool {
        parent.fill((-1, 0));
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();

        queue.push_back(source_pos);
        visited[source_pos] = true;

        while let Some(u) = queue.pop_front() {
            if let Some(neighbors) = adj.get(&u) {
                for (idx, (v, cap)) in neighbors.iter().enumerate() {
                    if !visited[*v] && *cap > 1e-9 {
                        visited[*v] = true;
                        parent[*v] = (u as isize, idx);
                        if *v == sink_pos {
                            return true;
                        }
                        queue.push_back(*v);
                    }
                }
            }
        }
        false
    }

    let source_pos = *index_to_pos.get(&source.index()).unwrap_or(&0);
    let sink_pos = *index_to_pos.get(&sink.index()).unwrap_or(&0);
    let mut parent = vec![(-1, 0); n];

    while bfs(&adj, source_pos, sink_pos, n, &mut parent) {
        // 找到路径上的最小残量
        let mut path_flow = f64::INFINITY;
        let mut v = sink_pos;
        while v != source_pos {
            let (u, edge_idx) = parent[v];
            let u = u as usize;
            if let Some(neighbors) = adj.get(&u) {
                if let Some((_, cap)) = neighbors.get(edge_idx) {
                    path_flow = path_flow.min(*cap);
                }
            }
            v = u;
        }

        // 更新残量
        let mut v = sink_pos;
        while v != source_pos {
            let (u, edge_idx) = parent[v];
            let u = u as usize;

            // 更新正向边
            if let Some(neighbors) = adj.get_mut(&u) {
                if let Some((_target, cap)) = neighbors.get_mut(edge_idx) {
                    *cap -= path_flow;
                    let target_val = *_target;
                    // 找到反向边并更新
                    if let Some(reverse_neighbors) = adj.get_mut(&v) {
                        if let Some((_, rev_cap)) =
                            reverse_neighbors.iter_mut().find(|(t, _)| *t == target_val)
                        {
                            *rev_cap += path_flow;
                        }
                    }
                }
            }
            v = u;
        }

        max_flow += path_flow;
    }

    max_flow
}

/// Dinic 最大流算法
///
/// 使用分层图和阻塞流的高效最大流算法
///
/// ## 内存优化
///
/// 使用 `HashMap<usize, HashMap<usize, f64>>` 存储残量，空间复杂度 O(V+E)
/// 对于稀疏图（E << V²），比邻接矩阵 O(V²) 节省 10-100x 内存
///
/// # 复杂度
/// - 时间：O(V² * E)
/// - 空间：O(V + E)
pub fn dinic<T, E, F>(
    graph: &mut Graph<T, E>,
    source: NodeIndex,
    sink: NodeIndex,
    mut capacity: F,
) -> f64
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    let n = graph.node_count();
    if n == 0 {
        return 0.0;
    }

    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let index_to_pos: HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    // 构建邻接表和残量图：使用 HashMap 优化稀疏图内存
    // adj: 邻接表结构，residual: 残量 HashMap
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut residual: HashMap<usize, HashMap<usize, f64>> = HashMap::with_capacity(n);

    for edge in graph.edges() {
        if let Ok((src, tgt)) = graph.edge_endpoints(edge.index()) {
            if let (Some(&u), Some(&v)) = (
                index_to_pos.get(&src.index()),
                index_to_pos.get(&tgt.index()),
            ) {
                let cap = capacity(src, tgt, edge.data());
                if !residual
                    .get(&u)
                    .map(|m| m.contains_key(&v))
                    .unwrap_or(false)
                {
                    adj[u].push(v);
                }
                residual.entry(u).or_default().insert(v, cap);
            }
        }
    }

    let source_pos = *index_to_pos.get(&source.index()).unwrap_or(&0);
    let sink_pos = *index_to_pos.get(&sink.index()).unwrap_or(&0);
    let mut max_flow = 0.0;

    // BFS 构建分层图
    fn bfs_level(
        adj: &[Vec<usize>],
        residual: &HashMap<usize, HashMap<usize, f64>>,
        source_pos: usize,
        sink_pos: usize,
        level: &mut [isize],
    ) -> bool {
        let _n = adj.len();
        level.fill(-1);
        level[source_pos] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source_pos);

        while let Some(u) = queue.pop_front() {
            if let Some(neighbors) = adj.get(u) {
                for &v in neighbors {
                    let cap = residual
                        .get(&u)
                        .and_then(|m| m.get(&v))
                        .copied()
                        .unwrap_or(0.0);
                    if level[v] == -1 && cap > 1e-9 {
                        level[v] = level[u] + 1;
                        if v == sink_pos {
                            return true;
                        }
                        queue.push_back(v);
                    }
                }
            }
        }
        false
    }

    // DFS 寻找阻塞流
    fn dfs_block(
        u: usize,
        flow: f64,
        sink_pos: usize,
        adj: &[Vec<usize>],
        residual: &mut HashMap<usize, HashMap<usize, f64>>,
        level: &mut [isize],
        ptr: &mut [usize],
    ) -> f64 {
        if u == sink_pos || flow < 1e-9 {
            return flow;
        }

        let start = ptr[u];
        for i in start..adj[u].len() {
            ptr[u] = i;
            let v = adj[u][i];
            let cap = residual
                .get(&u)
                .and_then(|m| m.get(&v))
                .copied()
                .unwrap_or(0.0);
            if level[v] == level[u] + 1 && cap > 1e-9 {
                let pushed = dfs_block(v, flow.min(cap), sink_pos, adj, residual, level, ptr);
                if pushed > 1e-9 {
                    *residual.entry(u).or_default().get_mut(&v).unwrap() -= pushed;
                    *residual.entry(v).or_default().entry(u).or_insert(0.0) += pushed;
                    return pushed;
                }
            }
        }
        0.0
    }

    let mut level = vec![-1; n];
    let mut ptr = vec![0; n];

    while bfs_level(&adj, &residual, source_pos, sink_pos, &mut level) {
        ptr.fill(0);
        while let Some(push) = {
            let pushed = dfs_block(
                source_pos,
                f64::INFINITY,
                sink_pos,
                &adj,
                &mut residual,
                &mut level,
                &mut ptr,
            );
            if pushed > 1e-9 {
                Some(pushed)
            } else {
                None
            }
        } {
            max_flow += push;
        }
    }

    max_flow
}

/// Push-Relabel 最大流算法（FIFO 队列优化）
///
/// 使用预流推进策略，通过高度标签和 FIFO 队列优化
///
/// ## 内存优化
///
/// 使用 `HashMap<usize, HashMap<usize, f64>>` 存储残量，空间复杂度 O(V+E)
///
/// # 参数
/// * `graph` - 有向图
/// * `source` - 源点
/// * `sink` - 汇点
/// * `capacity` - 获取边容量的闭包
///
/// # 返回
/// 最大流值
///
/// # 复杂度
/// - 时间：O(V²E)（使用 FIFO 队列优化）
/// - 空间：O(V + E)
///
/// # 示例
/// ```
/// use god_gragh::prelude::*;
/// use god_gragh::algorithms::flow::push_relabel;
///
/// let mut graph = GraphBuilder::directed()
///     .with_nodes(vec!["S", "A", "B", "T"])
///     .with_edges(vec![
///         (0, 1, 10.0), // S->A
///         (0, 2, 5.0),  // S->B
///         (1, 2, 15.0), // A->B
///         (1, 3, 10.0), // A->T
///         (2, 3, 10.0), // B->T
///     ])
///     .build()
///     .unwrap();
///
/// let source = graph.nodes().next().unwrap().index();
/// let sink = graph.nodes().nth(3).unwrap().index();
/// let max_flow = push_relabel(&mut graph, source, sink, |_, _, cap| *cap);
/// assert_eq!(max_flow, 15.0);
/// ```
pub fn push_relabel<T, E, F>(
    graph: &mut Graph<T, E>,
    source: NodeIndex,
    sink: NodeIndex,
    mut capacity: F,
) -> f64
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> f64,
{
    let n = graph.node_count();
    if n == 0 {
        return 0.0;
    }

    // 收集所有节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let index_to_pos: HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    let source_pos = index_to_pos[&source.index()];
    let sink_pos = index_to_pos[&sink.index()];

    // 构建邻接表和残量图：使用 HashMap 优化稀疏图内存
    // 空间复杂度：O(V + E)，而非邻接矩阵的 O(V²)
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut residual: HashMap<usize, HashMap<usize, f64>> = HashMap::with_capacity(n);

    // 初始化残量图
    for edge in graph.edges() {
        if let Ok((src, tgt)) = graph.edge_endpoints(edge.index()) {
            if let Some(&u) = index_to_pos.get(&src.index()) {
                if let Some(&v) = index_to_pos.get(&tgt.index()) {
                    let cap = capacity(src, tgt, edge.data());
                    residual.entry(u).or_default().insert(v, cap);
                    adj[u].push(v);
                    adj[v].push(u); // 反向边
                }
            }
        }
    }

    // 高度标签和超额流
    let mut height = vec![0; n];
    let mut excess = vec![0.0; n];

    // 初始化：源点高度为 n，推送初始流
    height[source_pos] = n;

    for &v in &adj[source_pos] {
        if let Some(cap) = residual.get(&source_pos).and_then(|m| m.get(&v)).copied() {
            if cap > 1e-9 {
                *residual
                    .entry(source_pos)
                    .or_default()
                    .entry(v)
                    .or_insert(0.0) = 0.0;
                *residual
                    .entry(v)
                    .or_default()
                    .entry(source_pos)
                    .or_insert(0.0) = cap;
                excess[v] = cap;
                excess[source_pos] -= cap;
            }
        }
    }

    // FIFO 队列存储活跃节点（超额流 > 0 且非源汇点）
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut in_queue = vec![false; n];

    for i in 0..n {
        if i != source_pos && i != sink_pos && excess[i] > 1e-9 {
            queue.push_back(i);
            in_queue[i] = true;
        }
    }

    // Push-Relabel 主循环
    while let Some(u) = queue.pop_front() {
        in_queue[u] = false;

        if excess[u] <= 1e-9 {
            continue;
        }

        // 尝试推送
        let mut pushed = false;
        if let Some(neighbors) = adj.get(u) {
            for &v in neighbors {
                let cap = residual
                    .get(&u)
                    .and_then(|m| m.get(&v))
                    .copied()
                    .unwrap_or(0.0);
                if cap > 1e-9 && height[u] == height[v] + 1 {
                    let delta = excess[u].min(cap);
                    *residual.entry(u).or_default().entry(v).or_insert(0.0) -= delta;
                    *residual.entry(v).or_default().entry(u).or_insert(0.0) += delta;
                    excess[u] -= delta;
                    excess[v] += delta;

                    if v != source_pos && v != sink_pos && !in_queue[v] && excess[v] > 1e-9 {
                        queue.push_back(v);
                        in_queue[v] = true;
                    }

                    if excess[u] <= 1e-9 {
                        pushed = true;
                        break;
                    }
                }
            }
        }

        // 如果无法推送，进行 Relabel
        if !pushed && excess[u] > 1e-9 {
            let mut min_height = usize::MAX;
            if let Some(neighbors) = adj.get(u) {
                for &v in neighbors {
                    let cap = residual
                        .get(&u)
                        .and_then(|m| m.get(&v))
                        .copied()
                        .unwrap_or(0.0);
                    if cap > 1e-9 && height[v] < min_height {
                        min_height = height[v];
                    }
                }
            }

            if min_height < usize::MAX {
                height[u] = min_height + 1;
            }

            // 重新加入队列
            if !in_queue[u] {
                queue.push_back(u);
                in_queue[u] = true;
            }
        }
    }

    // 计算最大流（从源点流出的总流量）
    let mut max_flow = 0.0;
    if let Some(neighbors) = adj.get(source_pos) {
        for &v in neighbors {
            let cap = residual
                .get(&v)
                .and_then(|m| m.get(&source_pos))
                .copied()
                .unwrap_or(0.0);
            if cap > 1e-9 {
                max_flow += cap;
            }
        }
    }

    max_flow
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_edmonds_karp_basic() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "C", "D", "T"])
            .with_edges(vec![
                (0, 1, 16.0), // S->A
                (0, 2, 13.0), // S->B
                (1, 2, 10.0), // A->B
                (1, 3, 12.0), // A->C
                (2, 1, 4.0),  // B->A
                (2, 4, 14.0), // B->D
                (3, 1, 9.0),  // C->A
                (3, 5, 20.0), // C->T
                (4, 3, 7.0),  // D->C
                (4, 5, 4.0),  // D->T
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(5, 1);
        let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 23.0);
    }

    #[test]
    fn test_dinic_basic() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "T"])
            .with_edges(vec![
                (0, 1, 10.0), // S->A
                (0, 2, 5.0),  // S->B
                (1, 2, 15.0), // A->B
                (1, 3, 10.0), // A->T
                (2, 3, 10.0), // B->T
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(3, 1);
        let max_flow = dinic(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 15.0);
    }

    #[test]
    fn test_edmonds_karp_simple() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "T"])
            .with_edges(vec![
                (0, 1, 5.0), // S->A
                (1, 2, 3.0), // A->T
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(2, 1);
        let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 3.0); // 瓶颈在 A->T
    }

    #[test]
    fn test_push_relabel_basic() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "T"])
            .with_edges(vec![
                (0, 1, 10.0), // S->A
                (0, 2, 5.0),  // S->B
                (1, 2, 15.0), // A->B
                (1, 3, 10.0), // A->T
                (2, 3, 10.0), // B->T
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(3, 1);
        let max_flow = push_relabel(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 15.0);
    }

    #[test]
    fn test_edmonds_karp_no_path() {
        // 源点和汇点不连通
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "T"])
            .with_edges(vec![
                (0, 1, 10.0), // S->A
                (2, 3, 5.0),  // B->T (不连通)
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(3, 1);
        let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 0.0);
    }

    #[test]
    fn test_dinic_no_path() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "T"])
            .with_edges(vec![
                (0, 1, 10.0), // S->A
                (2, 3, 5.0),  // B->T (不连通)
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(3, 1);
        let max_flow = dinic(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 0.0);
    }

    #[test]
    fn test_edmonds_karp_single_edge() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "T"])
            .with_edges(vec![(0, 1, 42.0)])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(1, 1);
        let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 42.0);
    }

    #[test]
    fn test_dinic_single_edge() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "T"])
            .with_edges(vec![(0, 1, 42.0)])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(1, 1);
        let max_flow = dinic(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 42.0);
    }

    #[test]
    fn test_push_relabel_no_path() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "B", "T"])
            .with_edges(vec![
                (0, 1, 10.0), // S->A
                (2, 3, 5.0),  // B->T (不连通)
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(3, 1);
        let max_flow = push_relabel(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 0.0);
    }

    #[test]
    fn test_edmonds_karp_zero_capacity() {
        let mut graph = GraphBuilder::directed()
            .with_nodes(vec!["S", "A", "T"])
            .with_edges(vec![
                (0, 1, 0.0), // S->A (零容量)
                (1, 2, 5.0), // A->T
            ])
            .build()
            .unwrap();

        let source = NodeIndex::new(0, 1);
        let sink = NodeIndex::new(2, 1);
        let max_flow = edmonds_karp(&mut graph, source, sink, |_, _, cap| *cap);

        assert_eq!(max_flow, 0.0);
    }
}
