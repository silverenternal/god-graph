//! 最短路径算法性能基准测试

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::algorithms::shortest_path::{astar, bellman_ford, dijkstra};
use god_graph::graph::traits::{GraphOps, GraphQuery};
use god_graph::graph::Graph;

fn create_weighted_graph(size: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(size, size * 3);
    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    // 添加线性边
    for i in 0..size.saturating_sub(1) {
        graph
            .add_edge(nodes[i], nodes[i + 1], (i + 1) as f64)
            .unwrap();
    }

    // 添加一些额外边
    for i in 0..size.saturating_sub(2) {
        graph
            .add_edge(nodes[i], nodes[i + 2], (i + 2) as f64 * 1.5)
            .unwrap();
    }

    graph
}

fn create_grid_graph_with_weights(rows: usize, cols: usize) -> Graph<usize, f64> {
    let size = rows * cols;
    let mut graph: Graph<usize, f64> = Graph::with_capacity(size, size * 4);

    let nodes: Vec<_> = (0..size).map(|i| graph.add_node(i).unwrap()).collect();

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            // 向右连接
            if col + 1 < cols {
                graph
                    .add_edge(nodes[idx], nodes[idx + 1], (idx % 10 + 1) as f64)
                    .unwrap();
            }
            // 向下连接
            if row + 1 < rows {
                graph
                    .add_edge(nodes[idx], nodes[idx + cols], ((idx * 3) % 10 + 1) as f64)
                    .unwrap();
            }
        }
    }

    graph
}

fn bench_dijkstra(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000];

    let mut group = c.benchmark_group("dijkstra");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_weighted_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let distances = dijkstra(&graph, start, |_, _, w| *w).unwrap();
                black_box(distances);
            });
        });
    }
    group.finish();
}

fn bench_bellman_ford(c: &mut Criterion) {
    let sizes = vec![50, 100, 200, 500];

    let mut group = c.benchmark_group("bellman_ford");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_weighted_graph(size);
            let start = graph.nodes().next().unwrap().index();

            b.iter(|| {
                let distances = bellman_ford(&graph, start, |_, _, w| *w).unwrap();
                black_box(distances);
            });
        });
    }
    group.finish();
}

fn bench_astar(c: &mut Criterion) {
    let sizes = vec![(10, 10), (20, 20), (30, 30), (50, 50)];

    let mut group = c.benchmark_group("astar");
    for (rows, cols) in sizes {
        let total = rows * cols;
        group.bench_with_input(
            BenchmarkId::from_parameter(total),
            &(rows, cols),
            |b, &(rows, cols)| {
                let graph = create_grid_graph_with_weights(rows, cols);
                let nodes: Vec<_> = graph.nodes().map(|n| n.index()).collect();
                let start = nodes[0];
                let goal = nodes[nodes.len() - 1];

                // 曼哈顿距离启发式
                let heuristic = move |node: god_graph::node::NodeIndex| -> f64 {
                    let idx = node.index();
                    let row = idx / cols;
                    let col = idx % cols;
                    let goal_row = (nodes.len() - 1) / cols;
                    let goal_col = (nodes.len() - 1) % cols;
                    ((goal_row as i32 - row as i32).abs() + (goal_col as i32 - col as i32).abs())
                        as f64
                };

                b.iter(|| {
                    let result = astar(&graph, start, goal, |_, _, w| *w, &heuristic).unwrap();
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_dijkstra, bench_bellman_ford, bench_astar,);
criterion_main!(benches);
