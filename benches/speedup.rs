//! 并行算法加速比基准测试
//!
//! 验证并行算法相对于串行版本的加速比
//! 目标：8 核 CPU 上达到 6-8x 加速比

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use god_gragh::graph::Graph;
use god_gragh::graph::traits::{GraphOps, GraphQuery};
use god_gragh::algorithms::traversal::{dfs, bfs};
use god_gragh::algorithms::centrality::pagerank;
use god_gragh::algorithms::parallel::{par_dfs, par_bfs, par_pagerank};
use std::sync::atomic::{AtomicUsize, Ordering};

/// 创建测试图
fn create_test_graph(num_nodes: usize, avg_degree: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes * avg_degree);
    let nodes: Vec<_> = (0..num_nodes)
        .map(|i| graph.add_node(i).unwrap())
        .collect();

    // 添加随机边（确定性伪随机）
    let mut seed = 42u64;
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed as usize) % num_nodes;
            if target != i {
                let _ = graph.add_edge(nodes[i], nodes[target], 1.0);
            }
        }
    }

    graph
}

/// 串行 DFS 包装函数
fn serial_dfs(graph: &Graph<usize, f64>, start: usize) {
    let start_idx = graph.nodes().nth(start).unwrap().index();
    let mut count = 0;
    dfs(graph, start_idx, |_node| {
        count += 1;
        true
    });
    black_box(count);
}

/// 并行 DFS 包装函数
fn parallel_dfs(graph: &Graph<usize, f64>, start: usize) {
    let start_idx = graph.nodes().nth(start).unwrap().index();
    let count = AtomicUsize::new(0);
    par_dfs(graph, start_idx, |_node| {
        count.fetch_add(1, Ordering::Relaxed);
        true
    });
    black_box(count.load(Ordering::Relaxed));
}

/// 串行 BFS 包装函数
fn serial_bfs(graph: &Graph<usize, f64>, start: usize) {
    let start_idx = graph.nodes().nth(start).unwrap().index();
    let mut count = 0;
    bfs(graph, start_idx, |_node, _depth| {
        count += 1;
        true
    });
    black_box(count);
}

/// 并行 BFS 包装函数
fn parallel_bfs(graph: &Graph<usize, f64>, start: usize) {
    let start_idx = graph.nodes().nth(start).unwrap().index();
    let count = AtomicUsize::new(0);
    par_bfs(graph, start_idx, |_node, _depth| {
        count.fetch_add(1, Ordering::Relaxed);
        true
    });
    black_box(count.load(Ordering::Relaxed));
}

/// 串行 PageRank 包装函数
fn serial_pagerank(graph: &Graph<usize, f64>, damping: f64, iterations: usize) {
    let ranks = pagerank(graph, damping, iterations);
    black_box(ranks);
}

/// 并行 PageRank 包装函数
fn parallel_pagerank(graph: &Graph<usize, f64>, damping: f64, iterations: usize) {
    let ranks = par_pagerank(graph, damping, iterations);
    black_box(ranks);
}

fn bench_dfs_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("dfs_speedup");
    
    let sizes = vec![
        (1_000, 5),
        (5_000, 5),
        (10_000, 5),
        (50_000, 5),
        (100_000, 5),
    ];

    for (num_nodes, avg_degree) in sizes {
        let graph = create_test_graph(num_nodes, avg_degree);
        let start = 0;
        
        group.throughput(Throughput::Elements(num_nodes as u64));
        
        group.bench_with_input(
            BenchmarkId::new("serial", num_nodes),
            &(&graph, start),
            |b, &(graph, start)| b.iter(|| serial_dfs(graph, start)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_nodes),
            &(&graph, start),
            |b, &(graph, start)| b.iter(|| parallel_dfs(graph, start)),
        );
    }
    group.finish();
}

fn bench_bfs_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_speedup");
    
    let sizes = vec![
        (1_000, 5),
        (5_000, 5),
        (10_000, 5),
        (50_000, 5),
        (100_000, 5),
    ];

    for (num_nodes, avg_degree) in sizes {
        let graph = create_test_graph(num_nodes, avg_degree);
        let start = 0;
        
        group.throughput(Throughput::Elements(num_nodes as u64));
        
        group.bench_with_input(
            BenchmarkId::new("serial", num_nodes),
            &(&graph, start),
            |b, &(graph, start)| b.iter(|| serial_bfs(graph, start)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_nodes),
            &(&graph, start),
            |b, &(graph, start)| b.iter(|| parallel_bfs(graph, start)),
        );
    }
    group.finish();
}

fn bench_pagerank_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank_speedup");
    
    let sizes = vec![
        (1_000, 5),
        (5_000, 5),
        (10_000, 5),
        (50_000, 5),
    ];

    let damping = 0.85;
    let iterations = 20;

    for (num_nodes, avg_degree) in sizes {
        let graph = create_test_graph(num_nodes, avg_degree);
        
        group.throughput(Throughput::Elements((num_nodes * iterations) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("serial", num_nodes),
            &(&graph, damping, iterations),
            |b, &(graph, damping, iterations)| b.iter(|| serial_pagerank(graph, damping, iterations)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_nodes),
            &(&graph, damping, iterations),
            |b, &(graph, damping, iterations)| b.iter(|| parallel_pagerank(graph, damping, iterations)),
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = 
        bench_dfs_speedup,
        bench_bfs_speedup,
        bench_pagerank_speedup,
);
criterion_main!(benches);
