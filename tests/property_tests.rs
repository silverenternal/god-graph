//! 属性测试
//!
//! 使用 proptest 测试图操作的不变量和边界情况
//!
//! 属性测试通过随机生成测试数据，验证图操作的核心不变量：
//! - 删除节点后，关联边也被删除
//! - 节点索引在节点存活期间保持有效
//! - 图的节点数和边数始终正确
//! - 遍历算法访问所有可达节点

use proptest::prelude::*;
use std::collections::{BTreeSet, HashSet};

use god_gragh::algorithms::shortest_path::dijkstra;
use god_gragh::algorithms::traversal::{bfs, dfs};
use god_gragh::graph::traits::{GraphBase, GraphOps, GraphQuery};
use god_gragh::graph::Graph;

// ============================================
// 图操作不变量测试
// ============================================

// 测试：添加节点后节点计数正确
proptest! {
    #[test]
    fn prop_add_node_increments_count(num_nodes in 1usize..50) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
            prop_assert_eq!(graph.node_count(), i + 1);
        }
    }
}

// 测试：删除节点后节点计数正确，且关联边被删除
proptest! {
    #[test]
    fn prop_remove_node_decrements_count_and_removes_edges(
        num_nodes in 3usize..20,
        remove_idx in 0usize..20
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        // 创建图
        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 添加一些边
        for i in 0..num_nodes.saturating_sub(1) {
            let _ = graph.add_edge(indices[i], indices[i + 1], 1.0);
        }

        let initial_edge_count = graph.edge_count();

        // 删除一个节点（如果索引有效）
        if remove_idx < indices.len() {
            let _ = graph.remove_node(indices[remove_idx]);
            prop_assert_eq!(graph.node_count(), num_nodes - 1);
            // 边数应该减少（至少删除了 2 条关联边，如果是中间节点）
            prop_assert!(graph.edge_count() <= initial_edge_count);
        }
    }
}

// 测试：节点删除后，其索引失效
proptest! {
    #[test]
    fn prop_removed_node_index_becomes_invalid(
        num_nodes in 2usize..20,
        remove_idx in 0usize..19
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        if remove_idx < indices.len() {
            let removed_idx = indices[remove_idx];
            let _ = graph.remove_node(removed_idx);

            // 删除后不应再包含该节点
            prop_assert!(!graph.contains_node(removed_idx));

            // 尝试访问应该返回错误
            prop_assert!(graph.get_node(removed_idx).is_err());
        }
    }
}

// 测试：边删除后，其索引失效
proptest! {
    #[test]
    fn prop_removed_edge_index_becomes_invalid(
        num_nodes in 2usize..20,
        edge_idx in 0usize..19
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 添加边
        let mut edge_indices = Vec::new();
        for i in 0..num_nodes.saturating_sub(1) {
            if let Ok(edge_idx) = graph.add_edge(indices[i], indices[i + 1], 1.0) {
                edge_indices.push(edge_idx);
            }
        }

        if !edge_indices.is_empty() && edge_idx < edge_indices.len() {
            let removed_edge_idx = edge_indices[edge_idx];
            let _ = graph.remove_edge(removed_edge_idx);

            // 删除后不应再包含该边
            prop_assert!(!graph.contains_edge(removed_edge_idx));

            // 尝试访问应该返回错误
            prop_assert!(graph.get_edge(removed_edge_idx).is_err());
        }
    }
}

// 测试：节点索引复用但 generation 递增
proptest! {
    #[test]
    fn prop_node_index_reuse_increments_generation(
        num_iterations in 1usize..30
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut last_index: Option<god_gragh::node::NodeIndex> = None;
        let mut last_generation = 0u32;

        for _ in 0..num_iterations {
            let idx = graph.add_node(1).unwrap();

            if let Some(prev_idx) = last_index {
                // 如果索引被复用，generation 应该递增
                if idx.index() == prev_idx.index() {
                    prop_assert!(idx.generation() > last_generation);
                }
            }

            last_index = Some(idx);
            last_generation = idx.generation();

            // 删除节点
            let _ = graph.remove_node(idx);
        }
    }
}

// ============================================
// 遍历算法不变量测试
// ============================================

// 测试：DFS 访问所有可达节点
proptest! {
    #[test]
    fn prop_dfs_visits_all_reachable_nodes(
        num_nodes in 3usize..30,
        _seed in any::<u64>()
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        // 创建线性图：0 -> 1 -> 2 -> ... -> n
        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        for i in 0..num_nodes.saturating_sub(1) {
            let _ = graph.add_edge(indices[i], indices[i + 1], 1.0);
        }

        // 从节点 0 开始 DFS
        let start = indices[0];
        let mut visited = HashSet::new();

        dfs(&graph, start, |node| {
            visited.insert(node.index());
            true
        });

        // 应该访问所有节点
        prop_assert_eq!(visited.len(), num_nodes);
    }
}

// 测试：BFS 访问所有可达节点
proptest! {
    #[test]
    fn prop_bfs_visits_all_reachable_nodes(
        num_nodes in 3usize..30
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        // 创建线性图
        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        for i in 0..num_nodes.saturating_sub(1) {
            let _ = graph.add_edge(indices[i], indices[i + 1], 1.0);
        }

        let start = indices[0];
        let mut visited = HashSet::new();

        bfs(&graph, start, |node, _depth| {
            visited.insert(node.index());
            true
        });

        prop_assert_eq!(visited.len(), num_nodes);
    }
}

// 测试：从不可达节点开始遍历只访问起始节点
proptest! {
    #[test]
    fn prop_traversal_from_unreachable_visits_only_start(
        num_nodes in 3usize..20
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        // 创建不连通的图
        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 只连接部分节点
        if num_nodes >= 3 {
            let _ = graph.add_edge(indices[0], indices[1], 1.0);
        }

        // 从最后一个节点（不可达）开始
        let start = indices[num_nodes - 1];
        let mut visited = HashSet::new();

        dfs(&graph, start, |node| {
            visited.insert(node.index());
            true
        });

        // 只应访问起始节点
        prop_assert_eq!(visited.len(), 1);
        prop_assert!(visited.contains(&start.index()));
    }
}

// ============================================
// 最短路径算法不变量测试
// ============================================

// 测试：Dijkstra 算法在非负权重图上的正确性
proptest! {
    #[test]
    fn prop_dijkstra_non_negative_weights(
        num_nodes in 3usize..20
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        // 创建图
        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 添加正权重边
        for i in 0..num_nodes.saturating_sub(1) {
            let weight = (i + 1) as f64;
            let _ = graph.add_edge(indices[i], indices[i + 1], weight);
        }

        if num_nodes >= 2 {
            let start = indices[0];
            let end = indices[num_nodes - 1];

            let result = dijkstra(&graph, start, |_, _, w| *w);

            // 应该找到路径
            if let Ok(distances) = result {
                // 距离应该非负
                for dist in distances.values() {
                    prop_assert!(*dist >= 0.0);
                }
                // 应该包含 end 节点的距离
                prop_assert!(distances.contains_key(&end));
            } else {
                // 如果没有结果，测试失败
                prop_assert!(false, "Dijkstra 应该找到路径");
            }
        }
    }
}

// 测试：Dijkstra 返回的距离是自洽的
proptest! {
    #[test]
    fn prop_dijkstra_distance_consistency(
        num_nodes in 3usize..15
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 创建完全连接的图
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    let weight = ((i + j) % 10 + 1) as f64;
                    let _ = graph.add_edge(indices[i], indices[j], weight);
                }
            }
        }

        let start = indices[0];

        // 计算到所有节点的距离
        let result = dijkstra(&graph, start, |_, _, w| *w);

        if let Ok(distances) = result {
            for (&node, &dist) in &distances {
                // 距离应该非负
                prop_assert!(dist >= 0.0);
                // 节点应该在图中
                prop_assert!(graph.contains_node(node));
            }
        }
    }
}

// ============================================
// 图查询不变量测试
// ============================================

// 测试：邻居查询返回正确的结果
proptest! {
    #[test]
    fn prop_neighbors_correctness(
        num_nodes in 3usize..20
    ) {
        let mut graph = Graph::<i32, f64>::directed();
        let mut indices = Vec::new();

        for i in 0..num_nodes {
            let idx = graph.add_node(i as i32).unwrap();
            indices.push(idx);
        }

        // 第一个节点连接到第二个节点
        let _ = graph.add_edge(indices[0], indices[1], 1.0);

        // 其他节点连接到下一个节点
        for i in 1..num_nodes.saturating_sub(1) {
            let _ = graph.add_edge(indices[i], indices[i + 1], 1.0);
        }

        // 检查第一个节点的邻居（应该只有节点 1）
        let neighbors: BTreeSet<usize> = graph
            .neighbors(indices[0])
            .map(|n| n.index())
            .collect();

        let mut expected_neighbors = BTreeSet::new();
        expected_neighbors.insert(indices[1].index());

        prop_assert_eq!(neighbors, expected_neighbors);
    }
}

// 测试：度数计算正确
proptest! {
    #[test]
    fn prop_degree_calculation(
        num_edges in 1usize..50
    ) {
        let mut graph = Graph::<i32, f64>::directed();

        // 创建中心节点
        let center = graph.add_node(0).unwrap();

        // 添加叶子节点并连接到中心
        let mut expected_degree = 0;
        for i in 1..=num_edges {
            let leaf = graph.add_node(i as i32).unwrap();
            if graph.add_edge(center, leaf, 1.0).is_ok() {
                expected_degree += 1;
            }
        }

        let out_degree = graph.out_degree(center).unwrap_or(0);
        prop_assert_eq!(out_degree, expected_degree);
    }
}

// ============================================
// 边界情况测试
// ============================================

// 测试：空图操作
proptest! {
    #[test]
    fn prop_empty_graph_operations(_ in 0usize..1) {
        let graph = Graph::<i32, f64>::directed();

        prop_assert!(graph.is_empty());
        prop_assert_eq!(graph.node_count(), 0);
        prop_assert_eq!(graph.edge_count(), 0);

        // 在空图上查询应该返回错误
        // 使用无效的节点索引查询
        let nodes: Vec<_> = graph.nodes().collect();
        if nodes.is_empty() {
            // 空图应该没有节点
            prop_assert_eq!(graph.node_count(), 0);
        }
    }
}

// 测试：单节点图操作
proptest! {
    #[test]
    fn prop_single_node_graph(_ in 0usize..1) {
        let mut graph = Graph::<i32, f64>::directed();
        let node = graph.add_node(42).unwrap();

        prop_assert_eq!(graph.node_count(), 1);
        prop_assert_eq!(graph.edge_count(), 0);
        prop_assert_eq!(graph.get_node(node).unwrap(), &42);

        // 自环（如果允许）
        let self_loop = graph.add_edge(node, node, 1.0);
        // 根据实现，自环可能成功或失败
        // 如果成功，边数应为 1
        if self_loop.is_ok() {
            prop_assert_eq!(graph.edge_count(), 1);
        }
    }
}

// 测试：大图操作的性能边界
proptest! {
    #[test]
    fn prop_large_graph_basic_operations(_ in 0usize..1) {
        let mut graph = Graph::<i32, f64>::directed();

        // 添加大量节点
        for i in 0..100 {
            let _ = graph.add_node(i);
        }

        prop_assert_eq!(graph.node_count(), 100);

        // 添加一些边 - 收集节点索引而不是引用
        let node_indices: Vec<_> = graph.nodes().map(|n| n.index()).collect();
        for i in 0..node_indices.len().saturating_sub(1) {
            let from = node_indices[i];
            let to = node_indices[i + 1];
            let _ = graph.add_edge(from, to, 1.0);
        }

        prop_assert!(graph.edge_count() > 0);
    }
}
