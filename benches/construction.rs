//! 图构建性能基准测试

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::graph::traits::GraphOps;
use god_graph::graph::Graph;

fn bench_add_nodes(c: &mut Criterion) {
    let sizes = vec![100, 1000, 5000, 10000];

    let mut group = c.benchmark_group("add_nodes");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut graph: Graph<i32, f64> = Graph::directed();
                for i in 0..size {
                    graph.add_node(i).unwrap();
                }
                black_box(graph);
            });
        });
    }
    group.finish();
}

fn bench_add_edges(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000];

    let mut group = c.benchmark_group("add_edges");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut graph: Graph<i32, f64> = Graph::with_capacity(size, size * 2);

                // 先添加节点
                let nodes: Vec<_> = (0..size)
                    .map(|i| graph.add_node(i as i32).unwrap())
                    .collect();

                // 添加线性边
                for i in 0..size.saturating_sub(1) {
                    graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
                }

                black_box(graph);
            });
        });
    }
    group.finish();
}

fn bench_remove_node(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000];

    let mut group = c.benchmark_group("remove_node");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut graph: Graph<i32, f64> = Graph::with_capacity(size, size * 2);

                let nodes: Vec<_> = (0..size)
                    .map(|i| graph.add_node(i as i32).unwrap())
                    .collect();

                // 添加边：每个节点连接到下一个节点
                for i in 0..size.saturating_sub(1) {
                    graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
                }

                // 删除中间节点
                if size > 2 {
                    graph.remove_node(nodes[size / 2]).unwrap();
                }

                black_box(graph);
            });
        });
    }
    group.finish();
}

fn bench_with_capacity(c: &mut Criterion) {
    let sizes = vec![1000, 5000, 10000, 50000];

    let mut group = c.benchmark_group("with_capacity");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let graph: Graph<i32, f64> = Graph::with_capacity(size, size * 2);
                black_box(graph);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add_nodes,
    bench_add_edges,
    bench_remove_node,
    bench_with_capacity,
);
criterion_main!(benches);
