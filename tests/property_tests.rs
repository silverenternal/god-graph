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

use god_graph::algorithms::shortest_path::dijkstra;
use god_graph::algorithms::traversal::{bfs, dfs};
use god_graph::graph::traits::{GraphBase, GraphOps, GraphQuery};
use god_graph::graph::Graph;

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
        let mut last_index: Option<god_graph::node::NodeIndex> = None;
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

// ============================================
// Tensor 操作不变量测试
// ============================================

#[cfg(feature = "tensor")]
mod tensor_properties {
    use super::*;
    use god_graph::tensor::DenseTensor;
    use god_graph::tensor::traits::{TensorBase, TensorOps};

    // 测试：矩阵乘法结合律
    proptest! {
        #[test]
        fn prop_matrix_mult_assoc(
            a_data in prop::collection::vec(-10.0..10.0, 9),
            b_data in prop::collection::vec(-10.0..10.0, 9),
            c_data in prop::collection::vec(-10.0..10.0, 9),
        ) {
            let a = DenseTensor::new(a_data, vec![3, 3]);
            let b = DenseTensor::new(b_data, vec![3, 3]);
            let c = DenseTensor::new(c_data, vec![3, 3]);

            // (A × B) × C
            let ab = a.matmul(&b);
            let ab_c = ab.matmul(&c);

            // A × (B × C)
            let bc = b.matmul(&c);
            let a_bc = a.matmul(&bc);

            // 结果应该相等（允许浮点误差）
            for (v1, v2) in ab_c.data().iter().zip(a_bc.data().iter()) {
                let diff = (v1 - v2).abs();
                prop_assert!(diff < 1e-6, "Matrix multiplication not associative: diff={}", diff);
            }
        }
    }

    // 测试：转置两次等于原矩阵
    proptest! {
        #[test]
        fn prop_double_transpose_equals_original(
            data in prop::collection::vec(-100.0..100.0, 16),
        ) {
            let tensor = DenseTensor::new(data, vec![4, 4]);
            let transposed = tensor.transpose(None);
            let double_transposed = transposed.transpose(None);

            prop_assert_eq!(tensor.shape(), double_transposed.shape());
            for (orig, result) in tensor.data().iter().zip(double_transposed.data().iter()) {
                prop_assert!((orig - result).abs() < 1e-10, "Double transpose changed values");
            }
        }
    }

    // 测试：矩阵加法交换律
    proptest! {
        #[test]
        fn prop_matrix_add_commutative(
            a_data in prop::collection::vec(-1000.0..1000.0, 16),
            b_data in prop::collection::vec(-1000.0..1000.0, 16),
        ) {
            let a = DenseTensor::new(a_data, vec![4, 4]);
            let b = DenseTensor::new(b_data, vec![4, 4]);

            let a_plus_b = a.add(&b);
            let b_plus_a = b.add(&a);

            for (v1, v2) in a_plus_b.data().iter().zip(b_plus_a.data().iter()) {
                prop_assert!((v1 - v2).abs() < 1e-10, "Matrix addition not commutative");
            }
        }
    }

    // 测试：softmax 输出和为 1
    proptest! {
        #[test]
        fn prop_softmax_sums_to_one(
            data in prop::collection::vec(-100.0..100.0, 16),
        ) {
            let tensor = DenseTensor::new(data, vec![4, 4]);
            let softmax_result = god_graph::tensor::ops::activations::softmax(&tensor, 1);

            // 每行的和应该为 1
            for row in 0..4 {
                let row_sum: f64 = (0..4)
                    .map(|col| softmax_result.data()[row * 4 + col])
                    .sum();
                prop_assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "Softmax row {} sum is {}, expected 1.0",
                    row,
                    row_sum
                );
            }
        }
    }

    // 测试：ReLU 输出非负
    proptest! {
        #[test]
        fn prop_relu_output_non_negative(
            data in prop::collection::vec(-1000.0..1000.0, 64),
        ) {
            let tensor = DenseTensor::new(data, vec![8, 8]);
            let relu_result = god_graph::tensor::ops::activations::relu(&tensor);

            for &val in relu_result.data() {
                prop_assert!(val >= 0.0, "ReLU output should be non-negative");
            }
        }
    }

    // 测试：张量重塑保持元素不变
    proptest! {
        #[test]
        fn prop_reshape_preserves_elements(
            data in prop::collection::vec(-100.0..100.0, 16),
        ) {
            let tensor = DenseTensor::new(data.clone(), vec![4, 4]);
            let reshaped = tensor.reshape(&[2, 8]);

            prop_assert_eq!(reshaped.numel(), 16);
            prop_assert_eq!(reshaped.shape(), &[2, 8]);

            // 元素应该相同（顺序可能改变）
            let mut original_sorted = data.clone();
            let mut reshaped_data: Vec<_> = reshaped.data().to_vec();
            original_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            reshaped_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (orig, reshaped_val) in original_sorted.iter().zip(reshaped_data.iter()) {
                prop_assert!(
                    (orig - reshaped_val).abs() < 1e-10,
                    "Reshape changed element values"
                );
            }
        }
    }
}

// ============================================
// 图 + Tensor 联合不变量测试
// ============================================

#[cfg(all(feature = "tensor", feature = "tensor-gnn"))]
mod graph_tensor_properties {
    use super::*;
    use god_graph::tensor::DenseTensor;
    use god_graph::tensor::gnn::{GCNConv, MessagePassingLayer};
    use god_graph::tensor::traits::{TensorBase, TensorOps};

    // 测试：GCN 前向传播输出形状正确
    proptest! {
        #[test]
        fn prop_gcn_forward_output_shape(
            num_nodes in 3usize..20,
            in_features in 4usize..16,
            out_features in 4usize..16,
        ) {
            let gcn = GCNConv::new(in_features, out_features);

            // 创建随机特征
            let features_data: Vec<f64> = (0..num_nodes * in_features).map(|i| ((i % 100) as f64) / 100.0).collect();
            let features = DenseTensor::new(
                features_data,
                vec![num_nodes, in_features],
            );

            // 创建简单的邻接矩阵
            let adjacency = god_graph::tensor::SparseTensor::from_edges(
                &[(0, 1, 1.0), (1, 2, 1.0)],
                [num_nodes, num_nodes],
            );

            let output = gcn.forward(&features, &adjacency);

            prop_assert_eq!(output.shape(), &[num_nodes, out_features]);
        }
    }
}

// ============================================
// 李群正交化不变量测试
// ============================================

#[cfg(feature = "tensor")]
mod lie_group_properties {
    use super::*;
    use god_graph::tensor::DenseTensor;
    use god_graph::tensor::decomposition::qr::orthogonalize;
    use god_graph::tensor::traits::TensorOps;

    // 测试：正交化后 W^T * W = I
    proptest! {
        #[test]
        fn prop_orthogonalization_produces_orthogonal_matrix(
            data in prop::collection::vec(-10.0..10.0, 16),
        ) {
            let tensor = DenseTensor::new(data, vec![4, 4]);
            let ortho = orthogonalize(&tensor).expect("Orthogonalization failed");

            // 计算 W^T * W
            let transposed = ortho.transpose(None);
            let wt_w = transposed.matmul(&ortho);

            // 检查是否接近单位矩阵
            for i in 0..4 {
                for j in 0..4 {
                    let val = wt_w.data()[i * 4 + j];
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let error = (val - expected).abs();
                    prop_assert!(
                        error < 1e-6,
                        "W^T * W[{}][{}] = {}, expected {}",
                        i, j, val, expected
                    );
                }
            }
        }
    }

    // 测试：正交矩阵的行列式绝对值为 1
    proptest! {
        #[test]
        fn prop_orthogonal_matrix_determinant_abs_is_one(
            data in prop::collection::vec(-5.0..5.0, 9),
        ) {
            let tensor = DenseTensor::new(data, vec![3, 3]);
            let ortho = orthogonalize(&tensor).expect("Orthogonalization failed");

            // 计算行列式（3x3 矩阵）
            let d = ortho.data();
            let det = d[0] * (d[4] * d[8] - d[5] * d[7])
                    - d[1] * (d[3] * d[8] - d[5] * d[6])
                    + d[2] * (d[3] * d[7] - d[4] * d[6]);

            // 正交矩阵的行列式绝对值应该为 1
            let det_abs = det.abs();
            prop_assert!(
                (det_abs - 1.0).abs() < 1e-5,
                "Orthogonal matrix determinant |{}| should be 1",
                det_abs
            );
        }
    }
}
