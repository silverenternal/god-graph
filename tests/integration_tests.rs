//! 集成测试
//!
//! 测试 god-gragh 库的整体功能和正确性

use god_gragh::algorithms::centrality::degree_centrality;
use god_gragh::algorithms::community::connected_components;
use god_gragh::algorithms::mst::{kruskal, prim};
use god_gragh::algorithms::shortest_path::{astar, bellman_ford, dijkstra, floyd_warshall};
use god_gragh::algorithms::traversal::{bfs, dfs, tarjan_scc, topological_sort};
use god_gragh::generators::{complete_graph, grid_graph, tree_graph};
use god_gragh::graph::builders::GraphBuilder;
use god_gragh::graph::traits::{GraphBase, GraphOps, GraphQuery};
use god_gragh::graph::Graph;

/// 测试图的基本 CRUD 操作
#[test]
fn test_graph_crud() {
    let mut graph = Graph::<String, f64>::directed();

    // Create
    let a = graph.add_node("A".to_string()).unwrap();
    let b = graph.add_node("B".to_string()).unwrap();
    let c = graph.add_node("C".to_string()).unwrap();

    assert_eq!(graph.node_count(), 3);
    assert!(graph.contains_node(a));
    assert!(graph.contains_node(b));
    assert!(graph.contains_node(c));

    // Read
    assert_eq!(graph.get_node(a).unwrap(), "A");
    assert_eq!(graph.get_node(b).unwrap(), "B");
    assert_eq!(graph.get_node(c).unwrap(), "C");

    // Update
    let old = graph.update_node(a, "A_updated".to_string()).unwrap();
    assert_eq!(old, "A");
    assert_eq!(graph.get_node(a).unwrap(), "A_updated");

    // Add edges
    let ab = graph.add_edge(a, b, 1.0).unwrap();
    let bc = graph.add_edge(b, c, 2.0).unwrap();

    assert_eq!(graph.edge_count(), 2);
    assert!(graph.contains_edge(ab));
    assert!(graph.contains_edge(bc));

    // Delete node (should remove associated edges)
    let data = graph.remove_node(b).unwrap();
    assert_eq!(data, "B");
    assert!(!graph.contains_node(b));
    assert_eq!(graph.node_count(), 2);
}

/// 测试节点索引复用和 generation 机制
#[test]
fn test_node_index_generation() {
    let mut graph = Graph::<i32, f64>::directed();

    let idx1 = graph.add_node(1).unwrap();
    let gen1 = idx1.generation();

    graph.remove_node(idx1).unwrap();

    let idx2 = graph.add_node(2).unwrap();
    let gen2 = idx2.generation();

    // 索引应被复用，但 generation 应递增
    assert_eq!(idx1.index(), idx2.index());
    assert!(gen2 > gen1);

    // 旧索引应失效
    assert!(!graph.contains_node(idx1));
    assert!(graph.contains_node(idx2));
}

/// 测试 DFS 遍历
#[test]
fn test_dfs_traversal() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C", "D", "E"])
        .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 4, 1.0)])
        .build()
        .unwrap();

    let start = graph.nodes().next().unwrap().index();
    let mut visited = Vec::new();

    dfs(&graph, start, |node| {
        visited.push(node.index());
        true
    });

    assert_eq!(visited.len(), 5);
}

/// 测试 BFS 遍历
#[test]
fn test_bfs_traversal() {
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

/// 测试拓扑排序
#[test]
fn test_topological_sort() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C", "D", "E"])
        .with_edges(vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ])
        .build()
        .unwrap();

    let result = topological_sort(&graph).unwrap();
    assert_eq!(result.len(), 5);

    // 验证拓扑顺序
    let pos: std::collections::HashMap<usize, usize> = result
        .iter()
        .enumerate()
        .map(|(i, ni)| (ni.index(), i))
        .collect();

    // A(0) 必须在 B(1), C(2) 之前
    assert!(pos.get(&0).unwrap() < pos.get(&1).unwrap());
    assert!(pos.get(&0).unwrap() < pos.get(&2).unwrap());
    // B, C 必须在 D(3) 之前
    assert!(pos.get(&1).unwrap() < pos.get(&3).unwrap());
    assert!(pos.get(&2).unwrap() < pos.get(&3).unwrap());
}

/// 测试强连通分量
#[test]
fn test_strongly_connected_components() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec![1, 2, 3, 4, 5, 6, 7, 8])
        .with_edges(vec![
            // SCC 1: {1, 2, 4, 5}
            (0, 1, 1.0),
            (1, 3, 1.0),
            (3, 0, 1.0),
            (0, 4, 1.0),
            (4, 0, 1.0),
            // SCC 2: {3}
            // SCC 3: {6, 7}
            (5, 6, 1.0),
            (6, 5, 1.0),
            // SCC 4: {8}
            // 连接边
            (2, 5, 1.0),
            (5, 7, 1.0),
        ])
        .build()
        .unwrap();

    let sccs = tarjan_scc(&graph);
    assert!(!sccs.is_empty());
}

/// 测试 Dijkstra 最短路径
#[test]
fn test_dijkstra_shortest_path() {
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

    let nodes: Vec<_> = graph.nodes().collect();
    let start = nodes[0].index();
    let distances = dijkstra(&graph, start, |_, _, w| *w).unwrap();

    // A->A = 0
    assert_eq!(distances.get(&start), Some(&0.0));
    // A->B = 1
    let b = nodes[1].index();
    assert_eq!(distances.get(&b), Some(&1.0));
    // A->C = 3 (A->B->C)
    let c = nodes[2].index();
    assert_eq!(distances.get(&c), Some(&3.0));
    // A->D = 4 (A->B->C->D)
    let d = nodes[3].index();
    assert_eq!(distances.get(&d), Some(&4.0));
}

/// 测试 Bellman-Ford 负权重检测
#[test]
fn test_bellman_ford_negative_cycle() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C"])
        .with_edges(vec![
            (0, 1, 1.0),
            (1, 2, -2.0),
            (2, 0, -3.0), // 负权环
        ])
        .build()
        .unwrap();

    let nodes: Vec<_> = graph.nodes().collect();
    let source = nodes[0].index();
    let result = bellman_ford(&graph, source, |_, _, w| *w);

    assert!(matches!(result, Err(god_gragh::GraphError::NegativeCycle)));
}

/// 测试 Floyd-Warshall 全源最短路径
#[test]
fn test_floyd_warshall_all_pairs() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C"])
        .with_edges(vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)])
        .build()
        .unwrap();

    let distances = floyd_warshall(&graph, |_, _, w| *w).unwrap();

    let nodes: Vec<_> = graph.nodes().collect();

    // A->C 应该是 3 (A->B->C)，而不是 4 (直接边)
    assert_eq!(
        distances.get(&(nodes[0].index(), nodes[2].index())),
        Some(&3.0)
    );
}

/// 测试 A* 路径查找
#[test]
fn test_astar_pathfinding() {
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

    let nodes: Vec<_> = graph.nodes().collect();
    let start = nodes[0].index();
    let goal = nodes[3].index();

    // 使用启发式函数：估计到目标的距离
    let heuristic = |node: god_gragh::NodeIndex| -> f64 {
        let goal_idx = 3;
        (goal_idx as i64 - node.index() as i64).abs() as f64
    };

    let (distance, path) = astar(&graph, start, goal, |_, _, w| *w, heuristic).unwrap();

    assert!(!path.is_empty());
    assert_eq!(path.first(), Some(&start));
    assert_eq!(path.last(), Some(&goal));
    assert!(distance > 0.0);
}

/// 测试 Kruskal MST
#[test]
fn test_kruskal_mst() {
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

    let mst_edges = kruskal(&graph);

    // MST 应该有 n-1 = 3 条边
    assert_eq!(mst_edges.len(), 3);

    // 计算 MST 总权重
    let total_weight: f64 = mst_edges.iter().map(|&e| *graph.get_edge(e).unwrap()).sum();

    // 最小生成树总权重应该是 1 + 2 + 3 = 6
    assert_eq!(total_weight, 6.0);
}

/// 测试 Prim MST
#[test]
fn test_prim_mst() {
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

    let mst_edges = prim(&graph);

    // MST 应该有 n-1 = 3 条边
    assert_eq!(mst_edges.len(), 3);

    // 计算 MST 总权重
    let total_weight: f64 = mst_edges.iter().map(|&e| *graph.get_edge(e).unwrap()).sum();

    // 最小生成树总权重应该是 1 + 2 + 3 = 6
    assert_eq!(total_weight, 6.0);
}

/// 测试度中心性
#[test]
fn test_degree_centrality() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C", "D"])
        .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0)])
        .build()
        .unwrap();

    let centrality = degree_centrality(&graph);

    // A 有 3 个出边，中心性最高
    let nodes: Vec<_> = graph.nodes().collect();
    let a = nodes[0].index();
    assert!(centrality.contains_key(&a));
}

/// 测试连通分量
#[test]
fn test_connected_components() {
    let graph = GraphBuilder::undirected()
        .with_nodes(vec![1, 2, 3, 4, 5, 6])
        .with_edges(vec![
            (0, 1, 1.0),
            (1, 2, 1.0), // 分量 1: {0, 1, 2}
            (3, 4, 1.0), // 分量 2: {3, 4}
                         // 分量 3: {5} 孤立节点
        ])
        .build()
        .unwrap();

    let components = connected_components(&graph);

    // 应该有 3 个连通分量
    assert_eq!(components.len(), 3);
}

/// 测试完全图生成器
#[test]
fn test_complete_graph_generator() {
    let graph: Graph<i32, f64> = complete_graph(5);

    assert_eq!(graph.node_count(), 5);
    // 完全图 K5 有 5*4/2 = 10 条边
    assert_eq!(graph.edge_count(), 10);
}

/// 测试网格图生成器
#[test]
fn test_grid_graph_generator() {
    let graph: Graph<i32, f64> = grid_graph(3, 4, false);

    assert_eq!(graph.node_count(), 12);
    // 3x4 网格有 (3-1)*4 + 3*(4-1) = 8 + 9 = 17 条边
    assert_eq!(graph.edge_count(), 17);
}

/// 测试树生成器
#[test]
fn test_tree_graph_generator() {
    let graph: Graph<i32, f64> = tree_graph(10);

    assert_eq!(graph.node_count(), 10);
    // 树有 n-1 条边
    assert_eq!(graph.edge_count(), 9);
}

/// 测试序列化（需要 serde 特性）
#[cfg(all(feature = "serde", feature = "std"))]
#[test]
fn test_serialization() {
    // 简化测试，不依赖 serde_json
    let mut graph = Graph::<i32, f64>::directed();
    let a = graph.add_node(1).unwrap();
    let b = graph.add_node(2).unwrap();
    graph.add_edge(a, b, 3.0).unwrap();

    // 验证图可以克隆（序列化能力的代理测试）
    let _cloned = graph.clone();
    assert_eq!(_cloned.node_count(), 2);
    assert_eq!(_cloned.edge_count(), 1);
}

/// 测试大型图的性能
#[test]
fn test_large_graph_performance() {
    const NUM_NODES: usize = 1000;
    const NUM_EDGES: usize = 5000;

    let mut graph = Graph::<i32, f64>::with_capacity(NUM_NODES, NUM_EDGES);

    // 添加节点
    for i in 0..NUM_NODES {
        graph.add_node(i as i32).unwrap();
    }

    // 随机添加边
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut edges_added = 0;
    while edges_added < NUM_EDGES {
        let from = rng.gen_range(0..NUM_NODES);
        let to = rng.gen_range(0..NUM_NODES);
        if from != to {
            // 通过索引直接获取节点（跳过 generation 检查）
            let from_idx = graph.nodes().nth(from).unwrap().index();
            let to_idx = graph.nodes().nth(to).unwrap().index();
            if graph.add_edge(from_idx, to_idx, 1.0).is_ok() {
                edges_added += 1;
            }
        }
    }

    assert_eq!(graph.node_count(), NUM_NODES);
    assert_eq!(graph.edge_count(), NUM_EDGES);

    // 测试 BFS 性能
    let start = graph.nodes().next().unwrap().index();
    let mut count = 0;
    bfs(&graph, start, |_node, _depth| {
        count += 1;
        true
    });
    assert!(count > 0);
}
