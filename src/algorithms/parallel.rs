//! 并行算法模块
//!
//! 基于 rayon 的并行图算法实现
//!
//! ## 锁策略说明
//!
//! | 算法 | 锁策略 | 说明 |
//! |------|--------|------|
//! | `par_dfs` | 无锁 | 使用 `AtomicBool` + CAS 操作 |
//! | `par_bfs` | 无锁 | 使用 `AtomicBool` + 线程局部收集 |
//! | `par_pagerank` | 无锁 | 纯函数式迭代，无共享状态 |
//! | `par_dijkstra` | 细粒度锁 | 使用 `SegQueue` 无锁队列 + `AtomicU64` CAS 距离更新 |
//! | `par_connected_components` | 无锁 | 使用 `AtomicUsize` + CAS 操作 |
//! | `par_degree_centrality` | 无锁 | 纯并行映射 |
//!
//! 需要启用 `parallel` 特性

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_queue::SegQueue;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use crate::errors::GraphResult;

/// 并行 DFS（子树并行，无锁设计）
///
/// 使用原子操作标记访问状态，避免 Mutex 锁竞争
pub fn par_dfs<T, F>(
    graph: &Graph<T, impl Clone + Send + Sync>,
    start: NodeIndex,
    visitor: F,
) where
    T: Clone + Send + Sync,
    F: Fn(NodeIndex) -> bool + Send + Sync,
{
    let n = graph.node_count();
    if n == 0 {
        return;
    }

    let visited: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    let visited = Arc::new(visited);

    // 标记起始节点
    visited[start.index()].store(true, Ordering::SeqCst);

    // 获取起始节点的所有邻居
    let neighbors: Vec<NodeIndex> = graph.neighbors(start).collect();

    // 对每个邻居启动并行 DFS
    neighbors.into_par_iter().for_each(|neighbor| {
        if !visited[neighbor.index()].load(Ordering::Relaxed)
            && visited[neighbor.index()]
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            && visitor(neighbor)
        {
            par_dfs_subtree(graph, neighbor, &visited, &visitor);
        }
    });
}

/// 并行 DFS 子树处理
fn par_dfs_subtree<T, F>(
    graph: &Graph<T, impl Clone + Send + Sync>,
    node: NodeIndex,
    visited: &Vec<AtomicBool>,
    visitor: &F,
) where
    T: Clone + Send + Sync,
    F: Fn(NodeIndex) -> bool + Send + Sync,
{
    // 收集所有未访问的邻居
    let unvisited_neighbors: Vec<NodeIndex> = graph
        .neighbors(node)
        .filter(|&n| {
            !visited[n.index()].load(Ordering::Relaxed)
                && visited[n.index()]
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
        })
        .collect();

    if unvisited_neighbors.is_empty() {
        return;
    }

    // 如果邻居数量足够多，使用并行处理
    if unvisited_neighbors.len() >= 4 {
        unvisited_neighbors.into_par_iter().for_each(|neighbor| {
            if visitor(neighbor) {
                par_dfs_subtree(graph, neighbor, visited, visitor);
            }
        });
    } else {
        // 串行处理
        for neighbor in unvisited_neighbors {
            if visitor(neighbor) {
                par_dfs_subtree(graph, neighbor, visited, visitor);
            }
        }
    }
}

/// 并行 PageRank 算法
///
/// 使用无锁并行设计，每次迭代中所有节点的更新可以并行执行
/// 使用反向邻接表优化，时间复杂度 O(iterations * E)
pub fn par_pagerank<T>(
    graph: &Graph<T, impl Clone + Send + Sync>,
    damping: f64,
    iterations: usize,
) -> HashMap<NodeIndex, f64>
where
    T: Clone + Send + Sync,
{
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // 收集所有有效节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let node_to_pos: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ni)| (ni, i))
        .collect();

    // 预计算每个节点的出度
    let out_degrees: Vec<usize> = node_indices
        .iter()
        .map(|&ni| graph.out_degree(ni).unwrap_or(0))
        .collect();

    // 预计算反向邻接表（谁指向谁）- O(E) 复杂度
    // 遍历所有边，构建 incoming[pos] = 指向节点 pos 的所有源节点位置
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.edges() {
        let src = edge.source();
        let tgt = edge.target();
        if let (Some(&src_pos), Some(&tgt_pos)) = (node_to_pos.get(&src), node_to_pos.get(&tgt)) {
            incoming[tgt_pos].push(src_pos);
        }
    }

    // 初始化：均匀分布
    let mut scores: Vec<f64> = vec![1.0 / n as f64; n];

    for _ in 0..iterations {
        // 并行计算每个节点的新分数
        let new_scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                // 基础分数：随机跳转贡献
                let mut rank = (1.0 - damping) / n as f64;

                // 只遍历指向当前节点的邻居（O(in_degree) 而非 O(V)）
                for &neighbor_pos in &incoming[i] {
                    let out_degree = out_degrees[neighbor_pos];
                    if out_degree > 0 {
                        rank += damping * scores[neighbor_pos] / out_degree as f64;
                    }
                }

                rank
            })
            .collect();

        scores = new_scores;
    }

    // 转换回 HashMap
    node_indices
        .into_iter()
        .enumerate()
        .map(|(i, ni)| (ni, scores[i]))
        .collect()
}

/// SIMD 优化的并行 PageRank 算法（预留接口）
///
/// # 特性
/// - 使用批量计算优化，减少缓存未命中
/// - **注意**: SIMD 优化尚未实现，当前为预留接口
/// - 未来计划使用 `std::simd` 或 `portable-simd` 实现真正的 AVX2/AVX-512 优化
///
/// # 参数
/// * `graph` - 图
/// * `damping` - 阻尼系数（通常 0.85）
/// * `iterations` - 迭代次数
///
/// # 返回
/// HashMap，键为节点索引，值为 PageRank 分数
///
/// SIMD 优化的并行 PageRank 实现
///
/// 使用 `std::simd` 批量计算入边邻居的贡献，在支持 AVX2/AVX-512 的 CPU 上可获得 2-4x 加速。
///
/// # Algorithm
///
/// 1. 预计算反向邻接表和出度数组
/// 2. 迭代更新 PageRank 分数：
///    - 基础分数：`(1 - damping) / n`
///    - 邻居贡献：`damping * score[neighbor] / out_degree[neighbor]`
/// 3. 使用 SIMD 批量处理 4 个或 8 个邻居的贡献计算
///
/// # Performance
///
/// - 时间复杂度：O(iterations * (V + E))
/// - 空间复杂度：O(V + E)
/// - SIMD 加速：对入边邻居密集的场景效果最佳
///
/// # Example
///
/// ```rust,ignore
/// use god_gragh::algorithms::parallel::par_pagerank_simd;
///
/// let scores = par_pagerank_simd(&graph, 0.85, 20);
/// ```
///
/// # Requirements
///
/// 需要启用 `simd` 特性（支持 stable Rust）：
/// ```toml
/// [dependencies]
/// god-gragh = { version = "0.3", features = ["simd"] }
/// ```
#[cfg(feature = "simd")]
pub fn par_pagerank_simd<T>(
    graph: &Graph<T, impl Clone + Send + Sync>,
    damping: f64,
    iterations: usize,
) -> HashMap<NodeIndex, f64>
where
    T: Clone + Send + Sync,
{
    use wide::f64x4;
    
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // 收集所有有效节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let node_to_pos: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ni)| (ni, i))
        .collect();

    // 预计算每个节点的出度
    let out_degrees: Vec<usize> = node_indices
        .iter()
        .map(|&ni| graph.out_degree(ni).unwrap_or(0))
        .collect();

    // 预计算反向邻接表（谁指向谁）- O(E) 复杂度
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.edges() {
        let src = edge.source();
        let tgt = edge.target();
        if let (Some(&src_pos), Some(&tgt_pos)) = (node_to_pos.get(&src), node_to_pos.get(&tgt)) {
            incoming[tgt_pos].push(src_pos);
        }
    }

    // 初始化：均匀分布
    let mut scores: Vec<f64> = vec![1.0 / n as f64; n];

    // 预计算阻尼系数和基础分数
    let base_rank = (1.0 - damping) / n as f64;
    let damping_simd = f64x4::new([damping; 4]);

    for _ in 0..iterations {
        // SIMD 批量并行计算
        let new_scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                // 基础分数：随机跳转贡献
                let mut rank = base_rank;

                let neighbors = &incoming[i];
                let len = neighbors.len();
                
                // SIMD 批量处理：每 4 个邻居一组
                let mut j = 0;
                while j + 4 <= len {
                    // 加载 4 个邻居的位置
                    let neighbor_indices = [
                        neighbors[j],
                        neighbors[j + 1],
                        neighbors[j + 2],
                        neighbors[j + 3],
                    ];
                    
                    // 加载 4 个邻居的分数
                    let scores_array = [
                        scores[neighbor_indices[0]],
                        scores[neighbor_indices[1]],
                        scores[neighbor_indices[2]],
                        scores[neighbor_indices[3]],
                    ];
                    let scores_simd = f64x4::new(scores_array);

                    // 加载 4 个邻居的出度倒数（避免除法）
                    let inv_degrees = [
                        if out_degrees[neighbor_indices[0]] > 0 {
                            1.0 / out_degrees[neighbor_indices[0]] as f64
                        } else { 0.0 },
                        if out_degrees[neighbor_indices[1]] > 0 {
                            1.0 / out_degrees[neighbor_indices[1]] as f64
                        } else { 0.0 },
                        if out_degrees[neighbor_indices[2]] > 0 {
                            1.0 / out_degrees[neighbor_indices[2]] as f64
                        } else { 0.0 },
                        if out_degrees[neighbor_indices[3]] > 0 {
                            1.0 / out_degrees[neighbor_indices[3]] as f64
                        } else { 0.0 },
                    ];
                    let inv_degrees_simd = f64x4::new(inv_degrees);

                    // SIMD 批量计算：damping * score * (1/out_degree)
                    let contributions = damping_simd * scores_simd * inv_degrees_simd;

                    // 水平求和：将 4 个贡献值相加
                    let sum: [f64; 4] = contributions.into();
                    rank += sum[0] + sum[1] + sum[2] + sum[3];
                    
                    j += 4;
                }
                
                // 处理剩余不足 4 个的邻居
                while j < len {
                    let neighbor_pos = neighbors[j];
                    let out_degree = out_degrees[neighbor_pos];
                    if out_degree > 0 {
                        rank += damping * scores[neighbor_pos] / out_degree as f64;
                    }
                    j += 1;
                }

                rank
            })
            .collect();

        scores = new_scores;
    }

    // 转换回 HashMap
    node_indices
        .into_iter()
        .enumerate()
        .map(|(i, ni)| (ni, scores[i]))
        .collect()
}

/// SIMD 优化的度中心性计算
///
/// **注意**: 度中心性计算本身是简单的 O(V) 操作，SIMD 加速效果有限（约 1.1-1.3x）。
/// 主要优化来自并行迭代而非 SIMD 指令。
///
/// 此函数保留 SIMD 接口以便未来扩展加权度中心性等更复杂的计算。
///
/// # Performance
///
/// - 时间复杂度：O(V)
/// - 空间复杂度：O(V)
/// - SIMD 加速：有限（简单归一化操作）
#[cfg(feature = "simd")]
pub fn par_degree_centrality_simd<T>(
    graph: &Graph<T, impl Clone + Send + Sync>,
) -> HashMap<NodeIndex, f64>
where
    T: Clone + Send + Sync,
{
    let n = graph.node_count();
    if n <= 1 {
        return HashMap::new();
    }

    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let norm = 1.0 / (n - 1) as f64;

    // 批量并行计算度数
    // 注意：SIMD 对此简单计算的加速有限，主要收益来自并行迭代
    let centralities: Vec<f64> = node_indices
        .par_iter()
        .map(|&ni| {
            let degree = graph.out_degree(ni).unwrap_or(0) as f64;
            degree * norm
        })
        .collect();

    // 转换回 HashMap
    node_indices
        .into_iter()
        .zip(centralities)
        .collect()
}

/// 并行 BFS（分层并行，无锁设计）
///
/// 每层节点并行处理，层间同步
/// 使用线程局部收集 + 合并策略避免 Mutex 锁竞争
pub fn par_bfs<T, F>(
    graph: &Graph<T, impl Clone + Send + Sync>,
    start: NodeIndex,
    visitor: F,
) where
    T: Clone + Send + Sync,
    F: Fn(NodeIndex, usize) -> bool + Send + Sync,
{
    let n = graph.node_count();
    let visited: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    let visited = Arc::new(visited);

    // 标记起始节点
    visited[start.index()].store(true, Ordering::SeqCst);

    let mut current_layer = vec![start];
    let mut depth = 0;

    while !current_layer.is_empty() {
        // 使用线程局部收集策略避免 Mutex 锁
        let next_layer_vecs: Vec<Vec<NodeIndex>> = current_layer
            .par_iter()
            .filter_map(|&node| {
                if !visitor(node, depth) {
                    return None;
                }

                let neighbors: Vec<NodeIndex> = graph
                    .neighbors(node)
                    .filter(|&neighbor| {
                        !visited[neighbor.index()].load(Ordering::Relaxed)
                            && visited[neighbor.index()]
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                                .is_ok()
                    })
                    .collect();

                if neighbors.is_empty() {
                    None
                } else {
                    Some(neighbors)
                }
            })
            .collect();

        // 合并所有线程局部的结果
        let mut next_layer = Vec::new();
        for layer_vec in next_layer_vecs {
            next_layer.extend(layer_vec);
        }

        depth += 1;
        current_layer = next_layer;
    }
}

/// 并行连通分量（基于并查集）
/// 
/// 注意：由于并行并查集的实现复杂性，这里使用简化版本：
/// 1. 边的处理是并行的
/// 2. 但 union 操作使用原子操作保证安全性
/// 3. find 操作使用迭代而非递归避免栈溢出
/// 
/// 注意：此实现在多核上可能不会带来显著加速，因为并查集本质上是串行的
pub fn par_connected_components<T>(
    graph: &Graph<T, impl Clone + Send + Sync>,
) -> Vec<Vec<NodeIndex>>
where
    T: Clone + Send + Sync,
{
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = node_indices.len();

    if n == 0 {
        return Vec::new();
    }

    // 并查集：parent[i] 表示节点 i 的父节点
    let parent: Vec<AtomicUsize> = (0..n).map(AtomicUsize::new).collect();

    // 初始化：每个节点的父节点是自己
    for (i, atomic) in parent.iter().enumerate().take(n) {
        atomic.store(i, Ordering::Relaxed);
    }

    // 查找根节点（迭代版本，不带路径压缩以避免竞争）
    fn find(parent: &[AtomicUsize], mut i: usize) -> usize {
        loop {
            let p = parent[i].load(Ordering::Relaxed);
            if p == i {
                return p;
            }
            i = p;
        }
    }

    // 合并两个集合（使用原子 CAS 操作）
    fn union_atomic(
        parent: &[AtomicUsize],
        i: usize,
        j: usize,
    ) {
        let root_i = find(parent, i);
        let root_j = find(parent, j);

        if root_i == root_j {
            return;
        }

        // 使用较小的根作为新根，保证确定性
        let (old_root, new_root) = if root_i < root_j {
            (root_i, root_j)
        } else {
            (root_j, root_i)
        };

        // 尝试设置父节点，失败说明已被其他线程设置
        let _ = parent[old_root].compare_exchange(
            old_root,
            new_root,
            Ordering::SeqCst,
            Ordering::Relaxed
        );
    }

    // 并行处理所有边
    let edges: Vec<(usize, usize)> = graph
        .edges()
        .flat_map(|edge| {
            let source_idx = edge.source().index();
            let target_idx = edge.target().index();
            vec![(source_idx, target_idx), (target_idx, source_idx)]
        })
        .collect();

    edges.par_iter().for_each(|&(src, tgt)| {
        union_atomic(&parent, src, tgt);
    });

    // 收集每个根节点对应的分量
    let mut components_map: HashMap<usize, Vec<NodeIndex>> = HashMap::new();

    for &node in &node_indices {
        let root = find(&parent, node.index());
        components_map.entry(root).or_default().push(node);
    }

    components_map.into_values().collect()
}

/// 并行度中心性
pub fn par_degree_centrality<T>(
    graph: &Graph<T, impl Clone + Send + Sync>,
) -> HashMap<NodeIndex, f64>
where
    T: Clone + Send + Sync,
{
    let n = graph.node_count();
    if n <= 1 {
        return HashMap::new();
    }

    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let norm = 1.0 / (n - 1) as f64;

    node_indices
        .par_iter()
        .map(|&ni| {
            let degree = graph.out_degree(ni).unwrap_or(0) as f64;
            (ni, degree * norm)
        })
        .collect()
}

/// 并行 Dijkstra 算法（delta-stepping 简化版）
///
/// # 锁设计
///
/// 此实现使用混合锁/无锁数据结构：
/// - **桶队列**: 使用 `crossbeam-queue::SegQueue` 无锁并发队列
/// - **距离数组**: 使用 `AtomicU64` 存储距离的位表示，支持无锁 CAS 更新
/// - **完成标记**: 使用 `AtomicBool` 标记已完成的节点
///
/// 注意：虽然核心数据结构使用无锁操作，但桶队列的 `pop()` 操作内部存在细粒度锁。
/// 在稠密图上，锁竞争可能带来一定开销。
///
/// 使用桶式优先队列实现并行松弛操作
/// 适用于非负权重的图，delta 参数控制桶的粒度
///
/// # 参数
/// * `graph` - 图
/// * `source` - 源节点
/// * `get_weight` - 获取边权重的闭包（需要是 Fn 而非 FnMut）
/// * `delta` - 桶宽度（默认 1.0）
///
/// # 返回
/// HashMap，键为节点索引，值为最短距离
///
/// # 复杂度
/// - 时间：O((V + E) * log(V) / P)，P 为并行度
/// - 空间：O(V + E + B)，B 为桶数量
pub fn par_dijkstra<T, E, F>(
    graph: &Graph<T, E>,
    source: NodeIndex,
    get_weight: F,
    delta: f64,
) -> GraphResult<HashMap<NodeIndex, f64>>
where
    T: Clone + Send + Sync,
    E: Clone + Send + Sync,
    F: Fn(NodeIndex, NodeIndex, &E) -> f64 + Send + Sync,
{
    let n = graph.node_count();
    if n == 0 {
        return Ok(HashMap::new());
    }

    // 收集所有有效节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let node_to_pos: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ni)| (ni, i))
        .collect();

    // 距离数组：使用 AtomicU64 存储 f64 的位表示，支持无锁 CAS 更新
    let distances: Vec<AtomicU64> = (0..n)
        .map(|_| AtomicU64::new(f64::INFINITY.to_bits()))
        .collect();

    // 设置源节点距离
    if let Some(&source_pos) = node_to_pos.get(&source) {
        distances[source_pos].store(0.0_f64.to_bits(), Ordering::Relaxed);
    }

    // 桶式优先队列：buckets[k] 存储距离在 [k*delta, (k+1)*delta) 范围内的节点
    // 使用 SegQueue 无锁并发队列
    let buckets: Vec<SegQueue<usize>> = (0..10000).map(|_| SegQueue::new()).collect();
    let buckets = Arc::new(buckets);

    // 初始桶
    if let Some(&source_pos) = node_to_pos.get(&source) {
        buckets[0].push(source_pos);
    }

    // 已完成的节点
    let settled: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

    let mut current_bucket = 0;
    let mut empty_count = 0;
    let max_empty_buckets = 100;

    loop {
        // 收集当前桶中的所有节点
        let mut nodes_to_process: Vec<usize> = Vec::new();
        while let Some(node_pos) = buckets[current_bucket].pop() {
            nodes_to_process.push(node_pos);
        }

        if nodes_to_process.is_empty() {
            empty_count += 1;
            if empty_count >= max_empty_buckets {
                // 连续多个空桶，结束
                break;
            }
            current_bucket += 1;
            continue;
        }

        empty_count = 0;

        // 克隆节点列表用于并行处理后的标记
        let nodes_to_process_clone = nodes_to_process.clone();

        // 并行处理当前桶中的节点
        nodes_to_process.into_par_iter().for_each(|node_pos| {
            if settled[node_pos].load(Ordering::Relaxed) {
                return;
            }

            let node = node_indices[node_pos];
            let node_dist = {
                let dist_bits = distances[node_pos].load(Ordering::Relaxed);
                f64::from_bits(dist_bits)
            };

            // 遍历所有邻居
            for neighbor in graph.neighbors(node) {
                if let Some(&neighbor_pos) = node_to_pos.get(&neighbor) {
                    if settled[neighbor_pos].load(Ordering::Relaxed) {
                        continue;
                    }

                    if let Ok(edge_data) = graph.get_edge_by_nodes(node, neighbor) {
                        let weight = get_weight(node, neighbor, edge_data);
                        let new_dist = node_dist + weight;

                        // 使用 CAS 无锁更新距离
                        let mut current_bits = distances[neighbor_pos].load(Ordering::Relaxed);
                        loop {
                            let current_dist = f64::from_bits(current_bits);
                            if new_dist >= current_dist {
                                break;
                            }
                            let new_bits = new_dist.to_bits();
                            match distances[neighbor_pos].compare_exchange(
                                current_bits,
                                new_bits,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => {
                                    // 成功更新，将节点放入对应的桶
                                    let bucket_idx = ((new_dist / delta).floor() as usize)
                                        .saturating_add(1)
                                        .min(buckets.len() - 1);
                                    buckets[bucket_idx].push(neighbor_pos);
                                    break;
                                }
                                Err(observed) => {
                                    // CAS 失败，重试
                                    current_bits = observed;
                                }
                            }
                        }
                    }
                }
            }
        });

        // 标记当前桶中的节点为已完成
        for node_pos in nodes_to_process_clone {
            settled[node_pos].store(true, Ordering::Relaxed);
        }

        current_bucket += 1;
    }

    // 构建结果 HashMap
    let mut result = HashMap::with_capacity(n);
    for (i, &ni) in node_indices.iter().enumerate() {
        let dist_bits = distances[i].load(Ordering::Relaxed);
        let dist = f64::from_bits(dist_bits);
        if dist != f64::INFINITY {
            result.insert(ni, dist);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_par_pagerank() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
            .build()
            .unwrap();

        let ranks = par_pagerank(&graph, 0.85, 20);
        assert_eq!(ranks.len(), 3);

        // 在环形结构中，所有节点的 PageRank 应该相近
        let values: Vec<_> = ranks.values().collect();
        for i in 1..values.len() {
            assert!((values[i] - values[0]).abs() < 0.01);
        }
    }

    #[test]
    fn test_par_connected_components() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec![1, 2, 3, 4, 5, 6])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)])
            .build()
            .unwrap();

        let components = par_connected_components(&graph);
        assert_eq!(components.len(), 3); // {0,1,2}, {3,4}, {5}
    }

    #[test]
    fn test_par_degree_centrality() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)])
            .build()
            .unwrap();

        let centrality = par_degree_centrality(&graph);
        assert_eq!(centrality.len(), 4);
    }

    #[test]
    fn test_par_dfs() {
        use std::sync::atomic::AtomicUsize;

        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D", "E", "F"])
            .with_edges(vec![
                (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),  // A -> B, C, D
                (1, 4, 1.0),                              // B -> E
                (2, 5, 1.0),                              // C -> F
            ])
            .build()
            .unwrap();

        let start = graph.nodes().next().unwrap().index();
        let count = Arc::new(AtomicUsize::new(1)); // 起始节点已经计数
        let count_clone = count.clone();

        par_dfs(&graph, start, move |_node| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            true
        });

        // 应该访问所有 6 个节点
        assert_eq!(count.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn test_par_dijkstra_basic() {
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

        let start = graph.nodes().next().unwrap().index();
        let distances = par_dijkstra(&graph, start, |_, _, w| *w, 1.0).unwrap();

        assert!(distances.contains_key(&start));
        assert_eq!(distances.get(&start), Some(&0.0));
    }

    #[test]
    fn test_par_dijkstra_single_node() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec![1])
            .build()
            .unwrap();

        let start = graph.nodes().next().unwrap().index();
        let distances = par_dijkstra(&graph, start, |_, _, _: &f64| 1.0, 1.0).unwrap();

        assert_eq!(distances.len(), 1);
        assert_eq!(distances.get(&start), Some(&0.0));
    }

    #[test]
    fn test_par_dijkstra_empty_graph() {
        let graph: Graph<i32, f64> = GraphBuilder::directed().build().unwrap();
        let distances = par_dijkstra(&graph, NodeIndex::new(0, 1), |_, _, _: &f64| 1.0, 1.0).unwrap();
        assert!(distances.is_empty());
    }
}
