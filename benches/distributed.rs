//! 分布式图处理性能基准测试
//!
//! 测试分布式图处理模块的性能，包括：
//! - 分区器性能（Hash vs Range）
//! - 分布式 PageRank 算法
//! - 分布式 BFS 算法
//! - 分区数量对性能的影响
//! - 边界节点比例对性能的影响

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::parallel::algorithms::bfs::DistributedBFS;
use god_graph::parallel::algorithms::connected_components::DistributedConnectedComponents;
use god_graph::parallel::algorithms::dfs::DistributedDFS;
use god_graph::parallel::algorithms::dijkstra::DistributedDijkstra;
use god_graph::parallel::algorithms::pagerank::DistributedPageRank;
use god_graph::parallel::partitioner::hash::HashPartitioner;
use god_graph::parallel::partitioner::range::RangePartitioner;
use god_graph::parallel::partitioner::traits::Partitioner;
use god_graph::graph::Graph;
use god_graph::vgi::VirtualGraph;

/// 创建测试用图（使用 () 作为节点和边数据，符合分布式算法要求）
fn create_test_graph(num_nodes: usize, avg_degree: usize) -> Graph<(), ()> {
    let mut graph: Graph<(), ()> = Graph::with_capacity(num_nodes, num_nodes * avg_degree);
    let nodes: Vec<_> = (0..num_nodes)
        .map(|_| VirtualGraph::add_node(&mut graph, ()).unwrap())
        .collect();

    // 添加随机边（确定性伪随机）
    let mut seed = 42u64;
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            // 简单的 LCG 伪随机
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed as usize) % num_nodes;
            if target != i {
                let _ = VirtualGraph::add_edge(&mut graph, nodes[i], nodes[target], ());
            }
        }
    }

    graph
}

/// 创建稀疏图（用于大规模测试）
fn create_sparse_graph(num_nodes: usize) -> Graph<(), ()> {
    create_test_graph(num_nodes, 3) // 平均度数 3
}

// ============================================================================
// 分区器性能测试
// ============================================================================

fn bench_hash_partitioner(c: &mut Criterion) {
    let params = vec![1000, 5000, 10000, 20000];
    let num_partitions = 4;

    let mut group = c.benchmark_group("hash_partitioner");
    for &num_nodes in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |b, &num_nodes| {
                let graph = create_sparse_graph(num_nodes);
                let partitioner = HashPartitioner::new(num_partitions);

                b.iter(|| {
                    let partitions = partitioner.partition_graph(&graph);
                    black_box(partitions);
                });
            },
        );
    }
    group.finish();
}

fn bench_range_partitioner(c: &mut Criterion) {
    let params = vec![1000, 5000, 10000, 20000];
    let num_partitions = 4;

    let mut group = c.benchmark_group("range_partitioner");
    for &num_nodes in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |b, &num_nodes| {
                let graph = create_sparse_graph(num_nodes);
                let partitioner = RangePartitioner::new(num_partitions);

                b.iter(|| {
                    let partitions = partitioner.partition_graph(&graph);
                    black_box(partitions);
                });
            },
        );
    }
    group.finish();
}

fn bench_partitioner_comparison(c: &mut Criterion) {
    let num_nodes = 10000;
    let partition_counts = vec![2, 4, 8, 16];

    let mut group = c.benchmark_group("partitioner_comparison");
    for &num_partitions in &partition_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_partitions),
            &num_partitions,
            |b, &num_partitions| {
                let graph = create_sparse_graph(num_nodes);
                let hash_partitioner = HashPartitioner::new(num_partitions);
                let range_partitioner = RangePartitioner::new(num_partitions);

                b.iter(|| {
                    let hash_partitions = hash_partitioner.partition_graph(&graph);
                    let range_partitions = range_partitioner.partition_graph(&graph);
                    black_box((hash_partitions, range_partitions));
                });
            },
        );
    }
    group.finish();
}

fn bench_partition_stats(c: &mut Criterion) {
    let num_nodes = 10000;
    let num_partitions = 4;

    let mut group = c.benchmark_group("partition_stats");
    group.bench_with_input(
        BenchmarkId::from_parameter(num_nodes),
        &num_nodes,
        |b, &num_nodes| {
            let graph = create_sparse_graph(num_nodes);
            let partitioner = HashPartitioner::new(num_partitions);

            b.iter(|| {
                let stats = partitioner.partition_stats(&graph);
                black_box(stats);
            });
        },
    );
    group.finish();
}

// ============================================================================
// 分布式 PageRank 性能测试
// ============================================================================

fn bench_distributed_pagerank(c: &mut Criterion) {
    let params = vec![(1000, 4), (5000, 4), (10000, 4)];
    let num_partitions = 4;

    let mut group = c.benchmark_group("distributed_pagerank");
    for &(num_nodes, avg_degree) in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &(num_nodes, avg_degree),
            |b, &(num_nodes, avg_degree)| {
                let graph = create_test_graph(num_nodes, avg_degree);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let pagerank = DistributedPageRank::new(0.85, 20, 1e-6);

                b.iter(|| {
                    let result = pagerank.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_pagerank_partition_comparison(c: &mut Criterion) {
    let num_nodes = 10000;
    let avg_degree = 4;
    let partition_counts = vec![2, 4, 8, 16];

    let mut group = c.benchmark_group("pagerank_partition_comparison");
    for &num_partitions in &partition_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_partitions),
            &num_partitions,
            |b, &num_partitions| {
                let graph = create_test_graph(num_nodes, avg_degree);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let pagerank = DistributedPageRank::new(0.85, 20, 1e-6);

                b.iter(|| {
                    let result = pagerank.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_pagerank_hash_vs_range(c: &mut Criterion) {
    let num_nodes = 10000;
    let avg_degree = 4;
    let num_partitions = 4;

    let mut group = c.benchmark_group("pagerank_hash_vs_range");
    group.bench_with_input(
        BenchmarkId::from_parameter("partitioner"),
        &"hash_vs_range",
        |b, _| {
            let graph = create_test_graph(num_nodes, avg_degree);
            let hash_partitioner = HashPartitioner::new(num_partitions);
            let range_partitioner = RangePartitioner::new(num_partitions);
            let hash_partitions = hash_partitioner.partition_graph(&graph);
            let range_partitions = range_partitioner.partition_graph(&graph);
            let pagerank = DistributedPageRank::new(0.85, 20, 1e-6);

            b.iter(|| {
                let hash_result = pagerank.compute(&graph, &hash_partitions);
                let range_result = pagerank.compute(&graph, &range_partitions);
                black_box((hash_result, range_result));
            });
        },
    );
    group.finish();
}

// ============================================================================
// 分布式 BFS 性能测试
// ============================================================================

fn bench_distributed_bfs(c: &mut Criterion) {
    let params = vec![(1000, 4), (5000, 4), (10000, 4)];
    let num_partitions = 4;

    let mut group = c.benchmark_group("distributed_bfs");
    for &(num_nodes, avg_degree) in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &(num_nodes, avg_degree),
            |b, &(num_nodes, avg_degree)| {
                let graph = create_test_graph(num_nodes, avg_degree);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let start_node = god_graph::node::NodeIndex::new_public(0);
                let bfs = DistributedBFS::new(start_node);

                b.iter(|| {
                    let result = bfs.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_bfs_partition_comparison(c: &mut Criterion) {
    let num_nodes = 10000;
    let avg_degree = 4;
    let partition_counts = vec![2, 4, 8, 16];

    let mut group = c.benchmark_group("bfs_partition_comparison");
    for &num_partitions in &partition_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_partitions),
            &num_partitions,
            |b, &num_partitions| {
                let graph = create_test_graph(num_nodes, avg_degree);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let start_node = god_graph::node::NodeIndex::new_public(0);
                let bfs = DistributedBFS::new(start_node);

                b.iter(|| {
                    let result = bfs.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_bfs_hash_vs_range(c: &mut Criterion) {
    let num_nodes = 10000;
    let avg_degree = 4;
    let num_partitions = 4;

    let mut group = c.benchmark_group("bfs_hash_vs_range");
    group.bench_with_input(
        BenchmarkId::from_parameter("partitioner"),
        &"hash_vs_range",
        |b, _| {
            let graph = create_test_graph(num_nodes, avg_degree);
            let hash_partitioner = HashPartitioner::new(num_partitions);
            let range_partitioner = RangePartitioner::new(num_partitions);
            let hash_partitions = hash_partitioner.partition_graph(&graph);
            let range_partitions = range_partitioner.partition_graph(&graph);
            let start_node = god_graph::node::NodeIndex::new_public(0);
            let bfs = DistributedBFS::new(start_node);

            b.iter(|| {
                let hash_result = bfs.compute(&graph, &hash_partitions);
                let range_result = bfs.compute(&graph, &range_partitions);
                black_box((hash_result, range_result));
            });
        },
    );
    group.finish();
}

// ============================================================================
// 分区平衡性分析
// ============================================================================

fn bench_partition_balance(c: &mut Criterion) {
    let num_nodes = 10000;
    let num_partitions = 4;

    let mut group = c.benchmark_group("partition_balance");
    group.bench_with_input(BenchmarkId::from_parameter("hash"), &"hash", |b, _| {
        let graph = create_sparse_graph(num_nodes);
        let partitioner = HashPartitioner::new(num_partitions);

        b.iter(|| {
            let stats = partitioner.partition_stats(&graph);
            let balance_ratio = stats.balance_ratio;
            let boundary_ratio = stats.total_boundary_nodes as f64 / stats.total_nodes as f64;
            black_box((balance_ratio, boundary_ratio));
        });
    });

    group.bench_with_input(BenchmarkId::from_parameter("range"), &"range", |b, _| {
        let graph = create_sparse_graph(num_nodes);
        let partitioner = RangePartitioner::new(num_partitions);

        b.iter(|| {
            let stats = partitioner.partition_stats(&graph);
            let balance_ratio = stats.balance_ratio;
            let boundary_ratio = stats.total_boundary_nodes as f64 / stats.total_nodes as f64;
            black_box((balance_ratio, boundary_ratio));
        });
    });

    group.finish();
}

// ============================================================================
// 分布式 DFS 基准测试
// ============================================================================

fn bench_distributed_dfs(c: &mut Criterion) {
    let params = vec![1000, 5000, 10000];
    let num_partitions = 4;

    let mut group = c.benchmark_group("distributed_dfs");
    for &num_nodes in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |b, &num_nodes| {
                let graph = create_sparse_graph(num_nodes);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let start_node = god_graph::node::NodeIndex::new_public(0);
                let dfs = DistributedDFS::new(start_node);

                b.iter(|| {
                    let result = dfs.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_dfs_partition_comparison(c: &mut Criterion) {
    let num_nodes = 5000;
    let params = vec![2, 4, 8, 16];

    let mut group = c.benchmark_group("dfs_partition_comparison");
    for &num_partitions in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_partitions),
            &num_partitions,
            |b, &num_partitions| {
                let graph = create_sparse_graph(num_nodes);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let start_node = god_graph::node::NodeIndex::new_public(0);
                let dfs = DistributedDFS::new(start_node);

                b.iter(|| {
                    let result = dfs.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 分布式连通分量基准测试
// ============================================================================

fn bench_distributed_cc(c: &mut Criterion) {
    let params = vec![1000, 5000, 10000];
    let num_partitions = 4;

    let mut group = c.benchmark_group("distributed_connected_components");
    for &num_nodes in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |b, &num_nodes| {
                let graph = create_sparse_graph(num_nodes);
                let partitioner = HashPartitioner::new(num_partitions);
                let partitions = partitioner.partition_graph(&graph);
                let cc = DistributedConnectedComponents::new();

                b.iter(|| {
                    let result = cc.compute(&graph, &partitions);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 分布式 Dijkstra 基准测试
// ============================================================================

fn create_weighted_graph(num_nodes: usize, avg_degree: usize) -> Graph<(), f64> {
    let mut graph: Graph<(), f64> = Graph::with_capacity(num_nodes, num_nodes * avg_degree);
    let nodes: Vec<_> = (0..num_nodes)
        .map(|_| VirtualGraph::add_node(&mut graph, ()).unwrap())
        .collect();

    // 添加随机边（确定性伪随机）
    let mut seed = 42u64;
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed as usize) % num_nodes;
            if target != i {
                let weight = ((seed % 100) as f64) / 10.0 + 0.1; // 权重 0.1 - 10.1
                let _ = VirtualGraph::add_edge(&mut graph, nodes[i], nodes[target], weight);
            }
        }
    }

    graph
}

fn bench_distributed_dijkstra(c: &mut Criterion) {
    let params = vec![500, 1000, 2000];
    let num_partitions = 4;

    let mut group = c.benchmark_group("distributed_dijkstra");
    for &num_nodes in &params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |b, &num_nodes| {
                let graph = create_weighted_graph(num_nodes, 4);
                let partitioner = HashPartitioner::new(num_partitions);
                let mut partitions = partitioner.partition_graph(&graph);

                // 缓存边权重到分区（O(1) 查询优化）
                for partition in &mut partitions {
                    partition.cache_edge_weights_from_graph(&graph, |_, _, w| *w);
                }

                let source = god_graph::node::NodeIndex::new_public(0);
                let dijkstra = DistributedDijkstra::new(source);

                b.iter(|| {
                    let result = dijkstra.compute(&graph, &partitions, |_, _, w| *w);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_dijkstra_hash_vs_range(c: &mut Criterion) {
    let num_nodes = 1000;
    let num_partitions = 4;

    let mut group = c.benchmark_group("dijkstra_hash_vs_range");
    group.bench_with_input(
        BenchmarkId::from_parameter("partitioner"),
        &"hash_vs_range",
        |b, _| {
            let graph = create_weighted_graph(num_nodes, 4);
            let hash_partitioner = HashPartitioner::new(num_partitions);
            let range_partitioner = RangePartitioner::new(num_partitions);
            let mut hash_partitions = hash_partitioner.partition_graph(&graph);
            let mut range_partitions = range_partitioner.partition_graph(&graph);

            // 缓存边权重到分区（O(1) 查询优化）
            for partition in &mut hash_partitions {
                partition.cache_edge_weights_from_graph(&graph, |_, _, w| *w);
            }
            for partition in &mut range_partitions {
                partition.cache_edge_weights_from_graph(&graph, |_, _, w| *w);
            }

            let source = god_graph::node::NodeIndex::new_public(0);
            let dijkstra = DistributedDijkstra::new(source);

            b.iter(|| {
                let hash_result = dijkstra.compute(&graph, &hash_partitions, |_, _, w| *w);
                let range_result = dijkstra.compute(&graph, &range_partitions, |_, _, w| *w);
                black_box((hash_result, range_result));
            });
        },
    );
    group.finish();
}

// ============================================================================
// Criterion 注册
// ============================================================================

criterion_group!(
    benches,
    bench_hash_partitioner,
    bench_range_partitioner,
    bench_partitioner_comparison,
    bench_partition_stats,
    bench_distributed_pagerank,
    bench_pagerank_partition_comparison,
    bench_pagerank_hash_vs_range,
    bench_distributed_bfs,
    bench_bfs_partition_comparison,
    bench_bfs_hash_vs_range,
    bench_partition_balance,
    bench_distributed_dfs,
    bench_dfs_partition_comparison,
    bench_distributed_cc,
    bench_distributed_dijkstra,
    bench_dijkstra_hash_vs_range,
);

criterion_main!(benches);
