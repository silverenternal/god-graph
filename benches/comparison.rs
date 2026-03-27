//! 与 petgraph 库的性能对比基准测试
//!
//! 这些基准测试用于对比 god-gragh 和 petgraph 在相同操作上的性能

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashSet;

// ============================================
// god-gragh 导入
// ============================================
use god_gragh::algorithms::shortest_path::dijkstra as god_dijkstra;
use god_gragh::algorithms::traversal::{bfs as god_bfs, dfs as god_dfs};
use god_gragh::graph::traits::{GraphOps as GodGraphOps, GraphQuery as GodGraphQuery};
use god_gragh::graph::Graph as GodGraph;

// ============================================
// petgraph 导入
// ============================================
use petgraph::algo::dijkstra as pet_dijkstra;
use petgraph::graph::{Graph as PetGraph, NodeIndex as PetNodeIndex};
use petgraph::visit::{Bfs, Dfs};

// ============================================
// 图构建辅助函数
// ============================================

/// 创建线性图（god-gragh）
fn create_god_linear_graph(size: usize) -> GodGraph<usize, f64> {
    let mut graph: GodGraph<usize, f64> = GodGraph::with_capacity(size, size);
    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    for i in 0..size.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
    }

    graph
}

/// 创建线性图（petgraph）
fn create_pet_linear_graph(size: usize) -> PetGraph<usize, f64> {
    let mut graph = PetGraph::<usize, f64>::new();
    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i)).collect();

    for i in 0..size.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], 1.0);
    }

    graph
}

/// 创建网格图（god-gragh）
fn create_god_grid_graph(rows: usize, cols: usize) -> GodGraph<usize, f64> {
    let size = rows * cols;
    let mut graph: GodGraph<usize, f64> = GodGraph::with_capacity(size, size * 4);

    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if col + 1 < cols {
                graph.add_edge(nodes[idx], nodes[idx + 1], 1.0).unwrap();
            }
            if row + 1 < rows {
                graph.add_edge(nodes[idx], nodes[idx + cols], 1.0).unwrap();
            }
        }
    }

    graph
}

/// 创建网格图（petgraph）
fn create_pet_grid_graph(rows: usize, cols: usize) -> PetGraph<usize, f64> {
    let size = rows * cols;
    let mut graph = PetGraph::<usize, f64>::new();
    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i)).collect();

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if col + 1 < cols {
                graph.add_edge(nodes[idx], nodes[idx + 1], 1.0);
            }
            if row + 1 < rows {
                graph.add_edge(nodes[idx], nodes[idx + cols], 1.0);
            }
        }
    }

    graph
}

// ============================================
// DFS 对比
// ============================================

fn bench_dfs_comparison(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("dfs_comparison");

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("god-gragh", size), &size, |b, &size| {
            let graph = create_god_linear_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let mut count = 0;
                god_dfs(&graph, start, |_node| {
                    count += 1;
                    true
                });
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("petgraph", size), &size, |b, &size| {
            let graph = create_pet_linear_graph(size);
            let start = PetNodeIndex::new(0);

            b.iter(|| {
                let mut dfs = Dfs::new(&graph, start);
                let mut count = 0;
                while dfs.next(&graph).is_some() {
                    count += 1;
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

// ============================================
// BFS 对比
// ============================================

fn bench_bfs_comparison(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("bfs_comparison");

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("god-gragh", size), &size, |b, &size| {
            let graph = create_god_linear_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let mut count = 0;
                god_bfs(&graph, start, |_node, _depth| {
                    count += 1;
                    true
                });
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("petgraph", size), &size, |b, &size| {
            let graph = create_pet_linear_graph(size);
            let start = PetNodeIndex::new(0);

            b.iter(|| {
                let mut bfs = Bfs::new(&graph, start);
                let mut count = 0;
                while bfs.next(&graph).is_some() {
                    count += 1;
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

// ============================================
// Dijkstra 对比
// ============================================

fn bench_dijkstra_comparison(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000, 5000];

    let mut group = c.benchmark_group("dijkstra_comparison");

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("god-gragh", size), &size, |b, &size| {
            let graph = create_god_linear_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let distances = god_dijkstra(&graph, start, |_, _, w| *w).unwrap();
                black_box(distances);
            });
        });

        group.bench_with_input(BenchmarkId::new("petgraph", size), &size, |b, &size| {
            let graph = create_pet_linear_graph(size);
            let start = PetNodeIndex::new(0);

            b.iter(|| {
                let distances = pet_dijkstra(
                    &graph,
                    start,
                    None,
                    |edge: petgraph::graph::EdgeReference<f64>| *edge.weight(),
                );
                black_box(distances);
            });
        });
    }
    group.finish();
}

// ============================================
// 网格图遍历对比
// ============================================

fn bench_grid_traversal_comparison(c: &mut Criterion) {
    let sizes = vec![(10, 10), (20, 20), (30, 30), (50, 50)];

    let mut group = c.benchmark_group("grid_traversal_comparison");

    for (rows, cols) in sizes {
        let total = rows * cols;

        group.bench_with_input(
            BenchmarkId::new("god-gragh", total),
            &(rows, cols),
            |b, &(rows, cols)| {
                let graph = create_god_grid_graph(rows, cols);
                let start = graph.nodes().next().unwrap().index();

                b.iter(|| {
                    let mut visited = HashSet::new();
                    god_dfs(&graph, start, |node| {
                        visited.insert(node.index());
                        true
                    });
                    black_box(visited.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("petgraph", total),
            &(rows, cols),
            |b, &(rows, cols)| {
                let graph = create_pet_grid_graph(rows, cols);
                let start = PetNodeIndex::new(0);

                b.iter(|| {
                    let mut dfs = Dfs::new(&graph, start);
                    let mut count = 0;
                    while dfs.next(&graph).is_some() {
                        count += 1;
                    }
                    black_box(count);
                });
            },
        );
    }
    group.finish();
}

// ============================================
// 图构建性能对比
// ============================================

fn bench_graph_construction_comparison(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("graph_construction_comparison");

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("god-gragh", size), &size, |b, &size| {
            b.iter(|| {
                black_box(create_god_linear_graph(size));
            });
        });

        group.bench_with_input(BenchmarkId::new("petgraph", size), &size, |b, &size| {
            b.iter(|| {
                black_box(create_pet_linear_graph(size));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dfs_comparison,
    bench_bfs_comparison,
    bench_dijkstra_comparison,
    bench_grid_traversal_comparison,
    bench_graph_construction_comparison,
);
criterion_main!(benches);
