//! 中心性算法模块
//!
//! 包含度中心性、介数中心性、接近中心性、PageRank 等算法

use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::node::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// 度中心性
///
/// 计算每个节点的度中心性（归一化的度数）
pub fn degree_centrality<T>(graph: &Graph<T, impl Clone>) -> HashMap<NodeIndex, f64> {
    let n = graph.node_count();
    if n <= 1 {
        return HashMap::new();
    }

    let mut centrality = HashMap::new();
    let norm = 1.0 / (n - 1) as f64;

    for node in graph.nodes() {
        let degree = graph.out_degree(node.index()).unwrap_or(0) as f64;
        centrality.insert(node.index(), degree * norm);
    }

    centrality
}

/// PageRank 算法
///
/// 计算每个节点的 PageRank 分数
///
/// # 参数
/// * `graph` - 图
/// * `damping` - 阻尼系数（通常 0.85）
/// * `iterations` - 迭代次数
///
/// # 返回
/// HashMap，键为节点索引，值为 PageRank 分数
pub fn pagerank<T>(
    graph: &Graph<T, impl Clone>,
    damping: f64,
    iterations: usize,
) -> HashMap<NodeIndex, f64> {
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // 收集所有有效节点
    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    
    // 初始化：均匀分布
    let mut scores: HashMap<NodeIndex, f64> = node_indices
        .iter()
        .map(|&ni| (ni, 1.0 / n as f64))
        .collect();

    for _ in 0..iterations {
        let mut new_scores: HashMap<NodeIndex, f64> = node_indices
            .iter()
            .map(|&ni| (ni, 0.0))
            .collect();

        // 计算每个节点的 PageRank
        for &node in &node_indices {
            // 基础分数：随机跳转贡献
            let mut rank = (1.0 - damping) / n as f64;

            // 收集指向当前节点的邻居贡献
            for neighbor in graph.nodes() {
                // 检查 neighbor 是否指向 node
                if graph.has_edge(neighbor.index(), node) {
                    let out_degree = graph.out_degree(neighbor.index()).unwrap_or(1);
                    if out_degree > 0 {
                        let contribution = scores.get(&neighbor.index()).copied().unwrap_or(0.0);
                        rank += damping * contribution / out_degree as f64;
                    }
                }
            }

            new_scores.insert(node, rank);
        }

        scores = new_scores;
    }

    scores
}

/// 介数中心性（基于 Brandes 算法）
///
/// 计算每个节点的介数中心性
/// 介数中心性衡量节点在所有最短路径中出现的频率
pub fn betweenness_centrality<T>(graph: &Graph<T, impl Clone>) -> HashMap<NodeIndex, f64> {
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let mut centrality: HashMap<NodeIndex, f64> = node_indices
        .iter()
        .map(|&ni| (ni, 0.0))
        .collect();

    for s in &node_indices {
        // 单源最短路径
        let mut dist: HashMap<NodeIndex, i32> = node_indices.iter().map(|&ni| (ni, -1)).collect();
        let mut sigma: HashMap<NodeIndex, usize> = node_indices.iter().map(|&ni| (ni, 0)).collect();
        let mut predecessors: HashMap<NodeIndex, Vec<NodeIndex>> = 
            node_indices.iter().map(|&ni| (ni, Vec::new())).collect();
        let mut stack = Vec::new();

        dist.insert(*s, 0);
        sigma.insert(*s, 1);
        let mut queue = VecDeque::new();
        queue.push_back(*s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);

            for w in graph.neighbors(v) {
                // w 第一次访问
                if dist.get(&w).copied().unwrap_or(-1) < 0 {
                    dist.insert(w, dist.get(&v).copied().unwrap_or(0) + 1);
                    queue.push_back(w);
                }

                // 最短路径经过 v 到 w
                if dist.get(&w).copied().unwrap_or(0) == dist.get(&v).copied().unwrap_or(0) + 1 {
                    *sigma.get_mut(&w).unwrap() += sigma.get(&v).copied().unwrap_or(0);
                    predecessors.get_mut(&w).unwrap().push(v);
                }
            }
        }

        // 反向累加依赖值
        let mut delta: HashMap<NodeIndex, f64> = node_indices.iter().map(|&ni| (ni, 0.0)).collect();
        
        while let Some(w) = stack.pop() {
            for &v in predecessors.get(&w).unwrap_or(&Vec::new()) {
                let delta_w = delta.get(&w).copied().unwrap_or(0.0);
                let sigma_w = sigma.get(&w).copied().unwrap_or(1);
                let sigma_v = sigma.get(&v).copied().unwrap_or(1);
                
                if sigma_w > 0 {
                    let contrib = (sigma_v as f64 / sigma_w as f64) * (1.0 + delta_w);
                    *delta.get_mut(&v).unwrap() += contrib;
                }
            }
            
            if w != *s {
                let centrality_w = centrality.get_mut(&w).unwrap();
                *centrality_w += delta.get(&w).copied().unwrap_or(0.0);
            }
        }
    }

    // 归一化（有向图）
    if n > 2 {
        let norm = 1.0 / ((n - 1) * (n - 2)) as f64;
        for val in centrality.values_mut() {
            *val *= norm;
        }
    }

    centrality
}

/// 接近中心性（基于 BFS 计算平均最短距离）
///
/// 计算每个节点的接近中心性
/// 接近中心性衡量节点到所有其他节点的平均最短距离的倒数
pub fn closeness_centrality<T>(graph: &Graph<T, impl Clone>) -> HashMap<NodeIndex, f64> {
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let mut centrality: HashMap<NodeIndex, f64> = HashMap::new();

    for &source in &node_indices {
        // BFS 计算从 source 到所有节点的最短距离
        let mut dist: HashMap<NodeIndex, usize> = node_indices.iter().map(|&ni| (ni, usize::MAX)).collect();
        let mut queue = VecDeque::new();

        dist.insert(source, 0);
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            let d = dist.get(&v).copied().unwrap_or(usize::MAX);

            for w in graph.neighbors(v) {
                if dist.get(&w).copied().unwrap_or(usize::MAX) == usize::MAX {
                    dist.insert(w, d + 1);
                    queue.push_back(w);
                }
            }
        }

        // 计算可达节点的总距离
        let mut total_dist = 0usize;
        let mut reachable = 0usize;
        
        for &node in &node_indices {
            if node != source {
                let d = dist.get(&node).copied().unwrap_or(usize::MAX);
                if d != usize::MAX {
                    total_dist += d;
                    reachable += 1;
                }
            }
        }

        // 接近中心性 = 可达节点数 / 总距离
        let closeness = if total_dist > 0 {
            reachable as f64 / total_dist as f64
        } else if reachable == 0 {
            0.0
        } else {
            1.0 // 所有节点距离都为 0（只有孤立点）
        };

        centrality.insert(source, closeness);
    }

    centrality
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_pagerank_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
            .build()
            .unwrap();

        let ranks = pagerank(&graph, 0.85, 20);
        assert_eq!(ranks.len(), 3);

        // 在环形结构中，所有节点的 PageRank 应该相近
        let values: Vec<_> = ranks.values().collect();
        for i in 1..values.len() {
            assert!((values[i] - values[0]).abs() < 0.01);
        }
    }

    #[test]
    fn test_degree_centrality() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)])
            .build()
            .unwrap();

        let centrality = degree_centrality(&graph);
        
        // A 的出度为 3，中心性最高
        let a_idx = graph.nodes().next().unwrap().index();
        assert!(centrality.contains_key(&a_idx));
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();

        let centrality = betweenness_centrality(&graph);
        assert_eq!(centrality.len(), 3);
        
        // B 在 A->C 的路径上，介数中心性应该最高
        let b_idx = graph.nodes().nth(1).unwrap().index();
        let b_centrality = centrality.get(&b_idx).copied().unwrap_or(0.0);
        assert!(b_centrality > 0.0);
    }

    #[test]
    fn test_closeness_centrality() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 1.0)])
            .build()
            .unwrap();

        let centrality = closeness_centrality(&graph);
        assert_eq!(centrality.len(), 3);
    }

    #[test]
    fn test_betweenness_centrality_star() {
        // 星型图：中心节点介数应该最高
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["center", "A", "B", "C"])
            .with_edges(vec![
                (1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0),  // A, B, C -> center
                (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),  // center -> A, B, C
            ])
            .build()
            .unwrap();

        let centrality = betweenness_centrality(&graph);
        
        // 中心节点应该是介数最高的
        let center_idx = graph.nodes().next().unwrap().index();
        let center_centrality = centrality.get(&center_idx).copied().unwrap_or(0.0);
        
        // 验证中心节点的介数大于 0
        assert!(center_centrality > 0.0);
    }

    #[test]
    fn test_closeness_centrality_connected() {
        // 完全连通图的接近中心性应该都接近 1
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C", "D"])
            .with_edges(vec![
                (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
                (1, 0, 1.0), (1, 2, 1.0), (1, 3, 1.0),
                (2, 0, 1.0), (2, 1, 1.0), (2, 3, 1.0),
                (3, 0, 1.0), (3, 1, 1.0), (3, 2, 1.0),
            ])
            .build()
            .unwrap();

        let centrality = closeness_centrality(&graph);
        
        // 所有节点的接近中心性应该都大于 0
        for (&node, &cent) in &centrality {
            assert!(cent > 0.0, "Node {:?} has zero closeness", node);
        }
    }
}
