//! 图遍历算法性能基准测试

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::algorithms::traversal::{bfs, dfs, tarjan_scc};
use god_graph::graph::traits::{GraphOps, GraphQuery};
use god_graph::graph::Graph;

fn create_linear_graph(size: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(size, size);
    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    for i in 0..size.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
    }

    graph
}

fn create_grid_graph(rows: usize, cols: usize) -> Graph<usize, f64> {
    let size = rows * cols;
    let mut graph: Graph<usize, f64> = Graph::with_capacity(size, size * 4);

    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            // 向右连接
            if col + 1 < cols {
                graph.add_edge(nodes[idx], nodes[idx + 1], 1.0).unwrap();
            }
            // 向下连接
            if row + 1 < rows {
                graph.add_edge(nodes[idx], nodes[idx + cols], 1.0).unwrap();
            }
        }
    }

    graph
}

fn create_scc_graph(num_sccs: usize, nodes_per_scc: usize) -> Graph<usize, f64> {
    let total_nodes = num_sccs * nodes_per_scc;
    let mut graph: Graph<usize, f64> = Graph::with_capacity(total_nodes, total_nodes * 2);

    let nodes: Vec<_> = (0..total_nodes)
        .map(|i| graph.add_node(i).unwrap())
        .collect();

    // 每个 SCC 内部完全连接
    for scc in 0..num_sccs {
        let start = scc * nodes_per_scc;
        for i in start..start + nodes_per_scc {
            for j in start..start + nodes_per_scc {
                if i != j {
                    graph.add_edge(nodes[i], nodes[j], 1.0).unwrap();
                }
            }
        }
        // 连接到下一个 SCC（形成链）
        if scc + 1 < num_sccs {
            let last_of_current = start + nodes_per_scc - 1;
            let first_of_next = (scc + 1) * nodes_per_scc;
            graph
                .add_edge(nodes[last_of_current], nodes[first_of_next], 1.0)
                .unwrap();
        }
    }

    graph
}

fn bench_dfs(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000];

    let mut group = c.benchmark_group("dfs");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_linear_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let mut count = 0;
                dfs(&graph, start, |_node| {
                    count += 1;
                    true
                });
                black_box(count);
            });
        });
    }
    group.finish();
}

fn bench_bfs(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000];

    let mut group = c.benchmark_group("bfs");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_linear_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let mut count = 0;
                bfs(&graph, start, |_node, _depth| {
                    count += 1;
                    true
                });
                black_box(count);
            });
        });
    }
    group.finish();
}

fn bench_tarjan_scc(c: &mut Criterion) {
    let params = vec![(5, 20), (10, 50), (20, 100), (50, 200)];

    let mut group = c.benchmark_group("tarjan_scc");
    for (num_sccs, nodes_per_scc) in params {
        let total = num_sccs * nodes_per_scc;
        group.bench_with_input(
            BenchmarkId::from_parameter(total),
            &(num_sccs, nodes_per_scc),
            |b, &(num_sccs, nodes_per_scc)| {
                let graph = create_scc_graph(num_sccs, nodes_per_scc);

                b.iter(|| {
                    let sccs = tarjan_scc(&graph);
                    black_box(sccs);
                });
            },
        );
    }
    group.finish();
}

fn bench_dfs_grid(c: &mut Criterion) {
    let sizes = vec![(10, 10), (20, 20), (30, 30), (50, 50)];

    let mut group = c.benchmark_group("dfs_grid");
    for (rows, cols) in sizes {
        let total = rows * cols;
        group.bench_with_input(
            BenchmarkId::from_parameter(total),
            &(rows, cols),
            |b, &(rows, cols)| {
                let graph = create_grid_graph(rows, cols);
                let start = graph.nodes().next().unwrap().index();

                b.iter(|| {
                    let mut count = 0;
                    dfs(&graph, start, |_node| {
                        count += 1;
                        true
                    });
                    black_box(count);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dfs,
    bench_bfs,
    bench_tarjan_scc,
    bench_dfs_grid,
);
criterion_main!(benches);
