//! 并行算法性能基准测试

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::algorithms::parallel::{
    par_connected_components, par_degree_centrality, par_pagerank,
};
use god_graph::graph::traits::GraphOps;
use god_graph::graph::Graph;

fn create_large_graph(num_nodes: usize, avg_degree: usize) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes * avg_degree);
    let nodes: Vec<_> = (0..num_nodes).map(|i| graph.add_node(i).unwrap()).collect();

    // 添加随机边（确定性伪随机）
    let mut seed = 42u64;
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            // 简单的 LCG 伪随机
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed as usize) % num_nodes;
            if target != i {
                let _ = graph.add_edge(nodes[i], nodes[target], 1.0);
            }
        }
    }

    graph
}

/// 创建稀疏的 disconnected 图（环状连接，避免内存爆炸）
fn create_disconnected_graph_sparse(
    num_components: usize,
    nodes_per_component: usize,
) -> Graph<usize, f64> {
    let total_nodes = num_components * nodes_per_component;
    let total_edges = total_nodes * 2; // 每个节点约 2 条边
    let mut graph: Graph<usize, f64> = Graph::with_capacity(total_nodes, total_edges);

    let nodes: Vec<_> = (0..total_nodes)
        .map(|i| graph.add_node(i).unwrap())
        .collect();

    // 每个连通分量内部形成环状连接（而非完全连接）
    for comp in 0..num_components {
        let start = comp * nodes_per_component;
        for i in 0..nodes_per_component {
            let current = start + i;
            let next = start + (i + 1) % nodes_per_component;
            let _ = graph.add_edge(nodes[current], nodes[next], 1.0);
            // 添加一些额外的边确保连通性
            if i + 2 < nodes_per_component {
                let _ = graph.add_edge(nodes[current], nodes[start + i + 2], 1.0);
            }
        }
    }

    graph
}

fn bench_par_pagerank(c: &mut Criterion) {
    let params = vec![(500, 5), (1000, 5), (2000, 5), (5000, 5)];

    let mut group = c.benchmark_group("par_pagerank");
    for (num_nodes, avg_degree) in params {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &(num_nodes, avg_degree),
            |b, &(num_nodes, avg_degree)| {
                let graph = create_large_graph(num_nodes, avg_degree);

                b.iter(|| {
                    let ranks = par_pagerank(&graph, 0.85, 20);
                    black_box(ranks);
                });
            },
        );
    }
    group.finish();
}

fn bench_par_connected_components(c: &mut Criterion) {
    // 使用稀疏图而非完全图，避免内存爆炸
    // 每个分量内部形成环状连接，而非完全连接
    let params = vec![(4, 50), (4, 100), (4, 250), (4, 500)];

    let mut group = c.benchmark_group("par_connected_components");
    for (num_components, nodes_per_component) in params {
        let total = num_components * nodes_per_component;
        group.bench_with_input(
            BenchmarkId::from_parameter(total),
            &(num_components, nodes_per_component),
            |b, &(num_components, nodes_per_component)| {
                let graph = create_disconnected_graph_sparse(num_components, nodes_per_component);

                b.iter(|| {
                    let components = par_connected_components(&graph);
                    black_box(components);
                });
            },
        );
    }
    group.finish();
}

fn bench_par_degree_centrality(c: &mut Criterion) {
    let sizes = vec![500, 1000, 2000, 5000];

    let mut group = c.benchmark_group("par_degree_centrality");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_large_graph(size, 10);

            b.iter(|| {
                let centrality = par_degree_centrality(&graph);
                black_box(centrality);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_par_pagerank,
    bench_par_connected_components,
    bench_par_degree_centrality,
);
criterion_main!(benches);
