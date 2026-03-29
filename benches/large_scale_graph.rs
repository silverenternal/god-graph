//! Large-Scale Graph Performance Benchmarks
//!
//! This benchmark suite tests god-gragh's performance on large-scale graphs
//! with 1K, 100K, and 1M nodes to validate scalability claims.
//!
//! ## Benchmarks
//!
//! - PageRank on graphs of varying sizes
//! - BFS traversal performance
//! - Connected components computation
//! - Strong scaling with increasing thread counts
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench large_scale_graph --features "parallel"
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use god_gragh::algorithms::centrality::pagerank;
use god_gragh::algorithms::community::connected_components;
use god_gragh::algorithms::traversal::bfs;
use god_gragh::graph::traits::{GraphOps, GraphQuery};
use god_gragh::graph::Graph;
use rand::Rng;

/// Generate random graph using Barabási-Albert model (preferential attachment)
///
/// This creates scale-free networks similar to real-world graphs.
///
/// # Arguments
///
/// * `num_nodes` - Number of nodes
/// * `m` - Number of edges to attach from a new node to existing nodes
fn generate_barabasi_albert_graph(num_nodes: usize, m: usize) -> Graph<usize, f64> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes * m);

    if num_nodes == 0 {
        return graph;
    }

    // Start with a small complete graph of m+1 nodes
    let mut nodes = Vec::new();
    let initial_size = (m + 1).min(num_nodes);

    for i in 0..initial_size {
        let node = graph.add_node(i).unwrap();
        nodes.push(node);
    }

    // Connect initial nodes into a complete graph
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let _ = graph.add_edge(nodes[i], nodes[j], 1.0);
        }
    }

    // Track degrees for preferential attachment
    let mut degrees = vec![0; num_nodes];
    for &node in &nodes {
        degrees[node.index()] = nodes.len() - 1;
    }

    // Add remaining nodes with preferential attachment
    let mut rng = rand::thread_rng();
    let total_degree_initial: usize = degrees.iter().sum();

    for new_idx in initial_size..num_nodes {
        let new_node = graph.add_node(new_idx).unwrap();

        // Select m existing nodes with probability proportional to degree
        let mut targets = Vec::new();
        let mut attempts = 0;

        while targets.len() < m && attempts < m * 10 {
            attempts += 1;

            // Select a random node weighted by degree
            let total_degree: usize = degrees[..new_idx].iter().sum();
            if total_degree == 0 {
                break;
            }

            let r = (rng.gen_range(0..total_degree)) % num_nodes;
            let mut cumsum = 0;
            let mut selected = 0;

            for (i, &deg) in degrees[..new_idx].iter().enumerate() {
                cumsum += deg;
                if cumsum > r {
                    selected = i;
                    break;
                }
            }

            if !targets.contains(&selected) {
                targets.push(selected);
            }
        }

        // Add edges to selected nodes
        for &target_idx in &targets {
            let target_node = nodes.get(target_idx).copied().unwrap_or_else(|| {
                // Node might not exist in nodes vec if we're adding many nodes
                // This is a simplification
                graph.add_node(target_idx).unwrap()
            });
            let _ = graph.add_edge(new_node, target_node, 1.0);
            degrees[target_idx] += 1;
        }

        degrees[new_idx] = targets.len();
        nodes.push(new_node);
    }

    graph
}

/// Generate random sparse graph
fn generate_sparse_graph(num_nodes: usize, avg_degree: usize, seed: u64) -> Graph<usize, f64> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut graph: Graph<usize, f64> = Graph::with_capacity(num_nodes, num_nodes * avg_degree);

    let nodes: Vec<_> = (0..num_nodes).map(|i| graph.add_node(i).unwrap()).collect();

    // Each node connects to avg_degree random neighbors
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            let j = rng.gen_range(0..num_nodes);
            if i != j {
                let _ = graph.add_edge(nodes[i], nodes[j], 1.0);
            }
        }
    }

    graph
}

/// Benchmark PageRank on large graphs
fn bench_pagerank_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank_large_scale");

    let scales = vec![
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    for (num_nodes, scale_name) in scales {
        let graph = generate_sparse_graph(num_nodes, 10, 42);
        group.throughput(Throughput::Elements(num_nodes as u64));

        group.bench_function(BenchmarkId::new("pagerank", scale_name), |b| {
            b.iter(|| {
                let scores = pagerank(black_box(&graph), 0.85, 20);
                black_box(scores);
            });
        });
    }

    group.finish();
}

/// Benchmark BFS on large graphs
fn bench_bfs_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_large_scale");

    let scales = vec![
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    for (num_nodes, scale_name) in scales {
        let graph = generate_sparse_graph(num_nodes, 10, 42);
        group.throughput(Throughput::Elements(num_nodes as u64));

        // Get start node from graph
        let start_node = graph.nodes().next().unwrap();

        group.bench_function(BenchmarkId::new("bfs", scale_name), |b| {
            b.iter(|| {
                let mut visited_count = 0;
                bfs(
                    black_box(&graph),
                    black_box(start_node.index()),
                    |_node, _depth| {
                        visited_count += 1;
                        true
                    },
                );
                black_box(visited_count);
            });
        });
    }

    group.finish();
}

/// Benchmark connected components on large graphs
fn bench_connected_components_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("cc_large_scale");

    let scales = vec![
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    for (num_nodes, scale_name) in scales {
        let graph = generate_sparse_graph(num_nodes, 10, 42);
        group.throughput(Throughput::Elements(num_nodes as u64));

        group.bench_function(BenchmarkId::new("connected_components", scale_name), |b| {
            b.iter(|| {
                let components = connected_components(black_box(&graph));
                black_box(components.len());
            });
        });
    }

    group.finish();
}

/// Strong scaling test: fixed problem size, increasing thread counts
#[cfg(feature = "parallel")]
fn bench_strong_scaling(c: &mut Criterion) {
    use god_gragh::algorithms::parallel::par_pagerank;

    let mut group = c.benchmark_group("strong_scaling_pagerank");

    let num_nodes = 100_000;
    let graph = generate_sparse_graph(num_nodes, 10, 42);

    let thread_counts = vec![1, 2, 4, 8];

    for &threads in &thread_counts {
        group.bench_function(BenchmarkId::new("threads", threads.to_string()), |b| {
            b.iter(|| {
                // Note: ThreadPool is built globally, thread count is a hint
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                    .ok();

                let scores = par_pagerank(black_box(&graph), 0.85, 20);
                black_box(scores);
            });
        });
    }

    group.finish();
}

/// Weak scaling test: problem size increases with thread count
#[cfg(feature = "parallel")]
fn bench_weak_scaling(c: &mut Criterion) {
    use god_gragh::algorithms::parallel::par_pagerank;

    let mut group = c.benchmark_group("weak_scaling_pagerank");

    // Problem size per thread
    let nodes_per_thread = 25_000;

    let thread_counts = vec![1, 2, 4, 8];

    for &threads in &thread_counts {
        let num_nodes = nodes_per_thread * threads;
        let graph = generate_sparse_graph(num_nodes, 10, 42);
        group.throughput(Throughput::Elements(num_nodes as u64));

        group.bench_function(BenchmarkId::new("weak", threads.to_string()), |b| {
            b.iter(|| {
                let scores = par_pagerank(black_box(&graph), 0.85, 20);
                black_box(scores);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_pagerank_large,
        bench_bfs_large,
        bench_connected_components_large,
);

#[cfg(feature = "parallel")]
criterion_group!(
    name = parallel_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_strong_scaling,
        bench_weak_scaling,
);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);
