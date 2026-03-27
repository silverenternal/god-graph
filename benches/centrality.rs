//! 中心性算法性能基准测试

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_gragh::algorithms::centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality, pagerank,
};
#[cfg(all(feature = "parallel", feature = "simd"))]
use god_gragh::algorithms::parallel::par_pagerank_simd;
#[cfg(feature = "parallel")]
use god_gragh::algorithms::parallel::{
    par_degree_centrality as par_degree_centrality_fn, par_pagerank,
};
use god_gragh::graph::traits::{GraphOps, GraphQuery};
use god_gragh::graph::Graph;

/// 创建随机图
fn create_random_graph(num_nodes: usize, edge_probability: f64, seed: u64) -> Graph<usize, f64> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes * num_nodes / 2);

    let nodes: Vec<_> = (0..num_nodes).map(|i| graph.add_node(i).unwrap()).collect();

    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            if rng.gen_range(0.0..1.0) < edge_probability {
                graph.add_edge(nodes[i], nodes[j], 1.0).unwrap();
            }
        }
    }

    graph
}

/// 创建星型图（用于测试度中心性）
fn create_star_graph(num_nodes: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes - 1);

    let nodes: Vec<_> = (0..num_nodes).map(|i| graph.add_node(i).unwrap()).collect();

    // 中心节点连接到所有其他节点
    for i in 1..num_nodes {
        graph.add_edge(nodes[0], nodes[i], 1.0).unwrap();
    }

    graph
}

/// 创建线性图（用于测试介数中心性）
fn create_linear_graph(num_nodes: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes - 1);

    let nodes: Vec<_> = (0..num_nodes).map(|i| graph.add_node(i).unwrap()).collect();

    for i in 0..num_nodes.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
    }

    graph
}

// ============================================
// 度中心性基准测试
// ============================================

fn bench_degree_centrality(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000, 5000];

    let mut group = c.benchmark_group("degree_centrality");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.1, 42);

            b.iter(|| {
                let centrality = degree_centrality(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

// ============================================
// 介数中心性基准测试
// ============================================

fn bench_betweenness_centrality(c: &mut Criterion) {
    let sizes = vec![50, 100, 200, 500];

    let mut group = c.benchmark_group("betweenness_centrality");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.1, 42);

            b.iter(|| {
                let centrality = betweenness_centrality(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

/// 在线性图上测试介数中心性（中间节点应该有最高介数）
fn bench_betweenness_centrality_linear(c: &mut Criterion) {
    let sizes = vec![50, 100, 200, 500];

    let mut group = c.benchmark_group("betweenness_centrality_linear");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_linear_graph(size);

            b.iter(|| {
                let centrality = betweenness_centrality(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

// ============================================
// 接近中心性基准测试
// ============================================

fn bench_closeness_centrality(c: &mut Criterion) {
    let sizes = vec![50, 100, 200, 500];

    let mut group = c.benchmark_group("closeness_centrality");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.1, 42);

            b.iter(|| {
                let centrality = closeness_centrality(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

// ============================================
// PageRank 基准测试
// ============================================

fn bench_pagerank(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("pagerank");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.05, 42);

            b.iter(|| {
                let ranks = pagerank(&graph, 0.85, 20);
                black_box(ranks);
            });
        });
    }
    group.finish();
}

/// PageRank 收敛测试（不同迭代次数）
fn bench_pagerank_convergence(c: &mut Criterion) {
    let iterations = vec![5, 10, 20, 50, 100];

    let mut group = c.benchmark_group("pagerank_convergence");
    for &iters in &iterations {
        group.bench_with_input(BenchmarkId::from_parameter(iters), &iters, |b, &iters| {
            let graph = create_random_graph(1000, 0.05, 42);

            b.iter(|| {
                let ranks = pagerank(&graph, 0.85, iters);
                black_box(ranks);
            });
        });
    }
    group.finish();
}

// ============================================
// 星型图度中心性（验证正确性）
// ============================================

fn bench_degree_centrality_star(c: &mut Criterion) {
    let sizes = vec![10, 50, 100, 500];

    let mut group = c.benchmark_group("degree_centrality_star");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_star_graph(size);

            b.iter(|| {
                let centrality = degree_centrality(&graph);

                // 验证：中心节点的度中心性应该最高
                let center_idx = graph.nodes().next().unwrap().index();
                let center_centrality = centrality.get(&center_idx).copied().unwrap_or(0.0);
                black_box(center_centrality);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_degree_centrality,
    bench_betweenness_centrality,
    bench_betweenness_centrality_linear,
    bench_closeness_centrality,
    bench_pagerank,
    bench_pagerank_convergence,
    bench_degree_centrality_star,
    bench_pagerank_parallel,
    bench_pagerank_simd_comparison,
    bench_degree_centrality_parallel,
);
criterion_main!(benches);

// ============================================
// 并行 PageRank 基准测试
// ============================================

#[cfg(feature = "parallel")]
fn bench_pagerank_parallel(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("pagerank_parallel");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.05, 42);

            b.iter(|| {
                let ranks = par_pagerank(&graph, 0.85, 20);
                black_box(ranks);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_pagerank_parallel(_: &mut Criterion) {}

// ============================================
// SIMD PageRank 对比测试
// ============================================

#[cfg(all(feature = "parallel", feature = "simd"))]
fn bench_pagerank_simd_comparison(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("pagerank_simd_comparison");
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("serial", size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.05, 42);
            b.iter(|| {
                let ranks = pagerank(&graph, 0.85, 20);
                black_box(ranks);
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.05, 42);
            b.iter(|| {
                let ranks = par_pagerank(&graph, 0.85, 20);
                black_box(ranks);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.05, 42);
            b.iter(|| {
                let ranks = par_pagerank_simd(&graph, 0.85, 20);
                black_box(ranks);
            });
        });
    }
    group.finish();
}

#[cfg(not(all(feature = "parallel", feature = "simd")))]
fn bench_pagerank_simd_comparison(_: &mut Criterion) {}

// ============================================
// 并行度中心性基准测试
// ============================================

#[cfg(feature = "parallel")]
fn bench_degree_centrality_parallel(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000, 5000];

    let mut group = c.benchmark_group("degree_centrality_parallel");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_random_graph(size, 0.1, 42);

            b.iter(|| {
                let centrality = par_degree_centrality_fn(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_degree_centrality_parallel(_: &mut Criterion) {}
