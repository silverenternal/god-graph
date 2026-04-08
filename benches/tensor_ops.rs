//! Tensor 操作性能基准测试
//!
//! 测试 God-Graph v0.4.0 的 tensor 模块性能，包括：
//! - 稠密张量操作
//! - 稀疏张量操作
//! - GNN 原语
//! - 内存池性能
//! - Graph-Tensor 转换

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::graph::traits::GraphOps;
use god_graph::graph::Graph;
use god_graph::tensor::dense::DenseTensor;
use god_graph::tensor::gnn::Aggregator;
use god_graph::tensor::gnn::{GCNConv, MeanAggregator, SumAggregator};
use god_graph::tensor::graph_tensor::{GraphAdjacencyMatrix, GraphFeatureExtractor};
use god_graph::tensor::pool::{PoolConfig, TensorPool};
use god_graph::tensor::sparse::{COOTensor, SparseTensor};
use god_graph::tensor::traits::TensorOps;

/// 稠密张量基础操作性能测试
fn bench_dense_tensor_ops(c: &mut Criterion) {
    let sizes = vec![64, 128, 256, 512];

    let mut group = c.benchmark_group("dense_tensor_ops");
    for size in sizes {
        // 准备测试数据
        let data_a: Vec<f64> = (0..size * size).map(|i| (i as f64) * 0.01).collect();
        let data_b: Vec<f64> = (0..size * size).map(|i| (i as f64) * 0.02).collect();
        let tensor_a = DenseTensor::new(data_a, vec![size, size]);
        let tensor_b = DenseTensor::new(data_b, vec![size, size]);

        // 矩阵乘法
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &(&tensor_a, &tensor_b),
            |b, (ta, tb)| b.iter(|| black_box(ta.matmul(tb))),
        );

        // 转置（使用 matrix::transpose）
        group.bench_with_input(BenchmarkId::new("transpose", size), &tensor_a, |b, t| {
            b.iter(|| black_box(god_graph::tensor::ops::matrix::transpose(t)))
        });
    }
    group.finish();
}

/// 稀疏张量操作性能测试
fn bench_sparse_tensor_ops(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000];
    let sparsity = 0.05; // 5% 非零元素

    let mut group = c.benchmark_group("sparse_tensor_ops");
    for size in sizes {
        let nnz = ((size as f64) * (size as f64) * sparsity) as usize;

        // 生成随机稀疏数据
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            row_indices.push(rand::random::<usize>() % size);
            col_indices.push(rand::random::<usize>() % size);
            values.push(rand::random::<f64>());
        }

        let sparse = SparseTensor::COO(COOTensor::new(
            row_indices.clone(),
            col_indices.clone(),
            DenseTensor::new(values.clone(), vec![nnz]),
            [size, size],
        ));

        // COO 转 CSR
        group.bench_with_input(BenchmarkId::new("coo_to_csr", size), &sparse, |b, s| {
            b.iter(|| black_box(s.to_csr()))
        });

        // 稀疏 - 稠密矩阵向量乘法
        let dense_vec = DenseTensor::new(vec![1.0; size], vec![size]);
        group.bench_with_input(
            BenchmarkId::new("spmv", size),
            &(&sparse, &dense_vec),
            |b, (s, v)| b.iter(|| black_box(s.spmv(v).unwrap())),
        );
    }
    group.finish();
}

/// GNN 原语性能测试
fn bench_gnn_primitives(c: &mut Criterion) {
    let num_nodes = vec![50, 100, 200, 500];
    let in_features = 16;
    let out_features = 8;

    let mut group = c.benchmark_group("gnn_primitives");
    for n in num_nodes {
        // 准备节点特征
        let features_data: Vec<f64> = (0..n * in_features).map(|i| (i as f64) * 0.1).collect();
        let features = DenseTensor::new(features_data, vec![n, in_features]);

        // 准备边列表（随机图）
        let num_edges = n * 3; // 平均度数为 6
        let mut edges = Vec::with_capacity(num_edges);
        let mut edge_values = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            let src = rand::random::<usize>() % n;
            let dst = rand::random::<usize>() % n;
            edges.push((src, dst, 1.0));
            edge_values.push(1.0);
        }

        let adjacency = SparseTensor::COO(COOTensor::new(
            edges.iter().map(|(s, _, _)| *s).collect(),
            edges.iter().map(|(_, d, _)| *d).collect(),
            DenseTensor::new(edge_values, vec![num_edges]),
            [n, n],
        ));

        // GCN 前向传播
        let gcn = GCNConv::new(in_features, out_features);
        group.bench_with_input(
            BenchmarkId::new("gcn_forward", n),
            &(&features, &adjacency),
            |b, (feat, adj)| b.iter(|| black_box(gcn.forward(feat, adj))),
        );

        // 聚合器（模拟消息聚合）
        let num_messages = 10;
        let message_shape = vec![in_features];
        let messages: Vec<DenseTensor> = (0..num_messages)
            .map(|_| {
                let data: Vec<f64> = (0..in_features).map(|i| (i as f64) * 0.1).collect();
                DenseTensor::new(data, message_shape.clone())
            })
            .collect();

        let sum_agg = SumAggregator;
        let mean_agg = MeanAggregator;

        group.bench_with_input(
            BenchmarkId::new("sum_aggregator", n),
            &messages,
            |b, msgs| b.iter(|| black_box(sum_agg.aggregate(msgs))),
        );

        group.bench_with_input(
            BenchmarkId::new("mean_aggregator", n),
            &messages,
            |b, msgs| b.iter(|| black_box(mean_agg.aggregate(msgs))),
        );
    }
    group.finish();
}

/// 内存池性能测试
fn bench_memory_pool(c: &mut Criterion) {
    let sizes = vec![32, 64, 128, 256];

    let mut group = c.benchmark_group("memory_pool");
    for size in sizes {
        let shape = vec![size, size];

        // 传统分配
        group.bench_with_input(
            BenchmarkId::new("traditional_alloc", size),
            &shape,
            |b, s| b.iter(|| black_box(DenseTensor::zeros(s.clone()))),
        );

        // 内存池分配（预创建池，模拟复用场景）
        group.bench_with_input(BenchmarkId::new("pool_alloc", size), &shape, |b, s| {
            let config = PoolConfig::new(8, size * size * 2);
            let mut pool = TensorPool::new(config);
            b.iter(|| {
                let _tensor = black_box(pool.acquire(s.clone()));
                // 立即回收
            })
        });
    }
    group.finish();
}

/// 内存池分配减少验证测试
///
/// 验证"内存池减少 80-90% 分配开销"的声称
/// 通过模拟迭代算法场景（如 PageRank、GNN 训练）中的重复分配
fn bench_memory_pool_reduction(c: &mut Criterion) {
    let iterations = vec![10, 50, 100, 200];
    let tensor_size = 128; // 128x128 张量

    let mut group = c.benchmark_group("memory_pool_reduction");

    for &iters in &iterations {
        let shape = vec![tensor_size, tensor_size];

        // 场景 1: 传统分配 - 每次迭代都新分配
        group.bench_with_input(
            BenchmarkId::new("traditional_iterative", iters),
            &shape,
            |b, s| {
                b.iter(|| {
                    let mut allocations = 0usize;
                    for _ in 0..iters {
                        let _tensor = black_box(DenseTensor::zeros(s.clone()));
                        allocations += 1;
                    }
                    allocations // 防止优化
                })
            },
        );

        // 场景 2: 内存池分配 - 复用已分配内存
        group.bench_with_input(BenchmarkId::new("pool_iterative", iters), &shape, |b, s| {
            let config = PoolConfig::new(iters, iters * 2);
            let mut pool = TensorPool::new(config);
            b.iter(|| {
                for _ in 0..iters {
                    let _tensor = black_box(pool.acquire(s.clone()));
                    // 立即回收
                }
            });
        });
    }
    group.finish();
}

/// 内存池命中率基准测试
///
/// 测量不同池大小配置下的命中率
fn bench_memory_pool_hitrate(c: &mut Criterion) {
    let pool_configs = vec![
        (4, 16, "small"),
        (16, 64, "medium"),
        (64, 256, "large"),
        (256, 1024, "xlarge"),
    ];

    let tensor_size = 64;
    let shape = vec![tensor_size, tensor_size];
    let iterations = 100;

    let mut group = c.benchmark_group("memory_pool_hitrate");

    for (initial, max, name) in pool_configs {
        group.bench_with_input(
            BenchmarkId::new("hitrate", name),
            &(initial, max),
            |b, &(init, max)| {
                b.iter(|| {
                    let config = PoolConfig::new(init, max);
                    let mut pool = TensorPool::new(config);

                    // 模拟迭代工作负载
                    for _ in 0..iterations {
                        let tensor = pool.acquire(shape.clone());
                        // 模拟使用后立即释放
                        drop(tensor);
                    }

                    // 返回命中率统计
                    let stats = pool.stats();
                    if stats.total_allocations > 0 {
                        stats.pool_hits as f64 / stats.total_allocations as f64
                    } else {
                        0.0
                    }
                })
            },
        );
    }
    group.finish();
}

/// 内存池内存节省验证
///
/// 直接测量内存池相比传统分配减少的分配次数
fn bench_memory_allocation_savings(c: &mut Criterion) {
    let tensor_sizes = vec![32, 64, 128, 256];
    let iterations = 50;

    let mut group = c.benchmark_group("memory_allocation_savings");

    for &size in &tensor_sizes {
        let shape = vec![size, size];
        let _numel = size * size;

        // 传统方法：每次迭代新分配
        group.bench_with_input(
            BenchmarkId::new("traditional_total_allocs", size),
            &shape,
            |b, s| {
                b.iter(|| {
                    let mut total_bytes = 0usize;
                    for _ in 0..iterations {
                        let tensor = DenseTensor::zeros(s.clone());
                        total_bytes += tensor.nbytes();
                    }
                    total_bytes
                })
            },
        );

        // 内存池方法：首次分配后复用
        group.bench_with_input(
            BenchmarkId::new("pool_total_allocs", size),
            &shape,
            |b, s| {
                b.iter(|| {
                    let config = PoolConfig::new(iterations, iterations * 2);
                    let mut pool = TensorPool::new(config);
                    let mut total_bytes = 0usize;

                    for _ in 0..iterations {
                        let tensor = pool.acquire(s.clone());
                        total_bytes += tensor.nbytes();
                        // 模拟使用后立即释放以触发回收
                        drop(tensor);
                    }

                    // 计算实际新分配次数（pool_misses 表示未命中池，需要新分配）
                    let actual_allocations = pool.stats().pool_misses;
                    let theoretical_allocations = iterations;
                    let reduction =
                        1.0 - (actual_allocations as f64 / theoretical_allocations as f64);

                    (total_bytes, reduction, pool.stats().pool_hits)
                })
            },
        );
    }
    group.finish();
}

/// Graph-Tensor 转换性能测试
fn bench_graph_tensor_conversion(c: &mut Criterion) {
    let num_nodes = vec![50, 100, 200, 500];

    let mut group = c.benchmark_group("graph_tensor_conversion");
    for n in num_nodes {
        // 创建随机图
        let mut graph: Graph<usize, f64> = Graph::with_capacity(n, n * 3);
        let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i).unwrap()).collect();

        // 添加随机边
        let num_edges = n * 3;
        let mut edge_list = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            let src_idx = rand::random::<usize>() % n;
            let dst_idx = rand::random::<usize>() % n;
            if src_idx != dst_idx {
                let _ = graph.add_edge(nodes[src_idx], nodes[dst_idx], 1.0);
                edge_list.push((src_idx, dst_idx));
            }
        }

        // 邻接矩阵创建（使用 from_edge_list）
        group.bench_with_input(
            BenchmarkId::new("adjacency_matrix", n),
            &(edge_list.clone(), n),
            |b, (edges, num_nodes)| {
                b.iter(|| {
                    black_box(GraphAdjacencyMatrix::from_edge_list(
                        edges, *num_nodes, false,
                    ))
                })
            },
        );

        // 特征提取（使用 extract_node_features_scalar）
        group.bench_with_input(BenchmarkId::new("feature_extraction", n), &graph, |b, g| {
            b.iter(|| {
                let extractor = GraphFeatureExtractor::new(g);
                black_box(
                    extractor
                        .extract_node_features_scalar(|node_data| *node_data as f64)
                        .unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dense_tensor_ops,
    bench_sparse_tensor_ops,
    bench_gnn_primitives,
    bench_memory_pool,
    bench_memory_pool_reduction,
    bench_memory_pool_hitrate,
    bench_memory_allocation_savings,
    bench_graph_tensor_conversion,
);

criterion_main!(benches);
