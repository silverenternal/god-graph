//! Memory Pool Reduction Benchmark
//!
//! This benchmark validates the "80-90% allocation reduction" claim
//! by comparing allocation patterns with and without the tensor pool.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_graph::tensor::dense::DenseTensor;
use god_graph::tensor::pool::{PoolConfig, TensorPool};

/// Baseline: Iterative allocation without pool (simulates typical GNN/Transformer training)
fn bench_iterative_allocation_without_pool(c: &mut Criterion) {
    c.bench_function("pool_reduction_iterative_without_pool", |b| {
        b.iter(|| {
            let mut tensors = Vec::new();
            // Simulate 50 iterations of hidden state allocations (e.g., GNN layers)
            for _ in 0..50 {
                let tensor = DenseTensor::zeros(black_box(vec![128, 128]));
                tensors.push(tensor);
            }
            // All tensors dropped at end - simulates batch processing
            drop(tensors);
        });
    });
}

/// With Pool: Same pattern but with memory pool reuse
fn bench_iterative_allocation_with_pool(c: &mut Criterion) {
    c.bench_function("pool_reduction_iterative_with_pool", |b| {
        let config = PoolConfig::new(64, 256).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            // Acquire and release immediately (simulates temporary allocations)
            for _ in 0..50 {
                let tensor = pool.acquire(black_box(vec![128, 128]));
                drop(tensor); // Return to pool immediately
            }
        });

        // Report pool statistics
        let stats = pool.stats();
        eprintln!("=== Iterative Pool Stats ===");
        eprintln!("Total allocations: {}", stats.total_allocations);
        eprintln!("Pool hits (reuses): {}", stats.pool_hits);
        eprintln!("Pool misses (new allocs): {}", stats.pool_misses);
        eprintln!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("Allocation reduction: {:.2}%", stats.allocation_reduction());
    });
}

/// GNN Iteration: Typical GNN forward pass with hidden state temporaries
fn bench_gnn_iteration_with_pool(c: &mut Criterion) {
    c.bench_function("pool_reduction_gnn_iteration", |b| {
        let config = PoolConfig::new(128, 512).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            // Simulate 10 GNN layers, each with multiple temporaries
            for _ in 0..10 {
                // Hidden state
                {
                    let hidden = pool.acquire(black_box(vec![100, 64]));
                    drop(hidden);
                }

                // Message passing temporary
                {
                    let message = pool.acquire(black_box(vec![100, 64]));
                    drop(message);
                }

                // Update output
                {
                    let output = pool.acquire(black_box(vec![100, 64]));
                    drop(output);
                }
            }
        });

        let stats = pool.stats();
        eprintln!("=== GNN Iteration Pool Stats ===");
        eprintln!("Total allocations: {}", stats.total_allocations);
        eprintln!("Pool hits (reuses): {}", stats.pool_hits);
        eprintln!("Pool misses (new allocs): {}", stats.pool_misses);
        eprintln!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("Allocation reduction: {:.2}%", stats.allocation_reduction());
    });
}

/// Matrix Multiplication Temporaries: Sequential matmul pattern
fn bench_matmul_temporaries_with_pool(c: &mut Criterion) {
    c.bench_function("pool_reduction_matmul_temporaries", |b| {
        let config = PoolConfig::new(16, 64).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            // Simulate sequential matmul operations (e.g., Q, K, V projections)
            for _ in 0..20 {
                {
                    let _q = pool.acquire(black_box(vec![64, 64]));
                }
                {
                    let _k = pool.acquire(black_box(vec![64, 64]));
                }
                {
                    let _v = pool.acquire(black_box(vec![64, 64]));
                }
            }
        });

        let stats = pool.stats();
        eprintln!("=== MatMul Temporaries Pool Stats ===");
        eprintln!("Total allocations: {}", stats.total_allocations);
        eprintln!("Pool hits (reuses): {}", stats.pool_hits);
        eprintln!("Pool misses (new allocs): {}", stats.pool_misses);
        eprintln!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("Allocation reduction: {:.2}%", stats.allocation_reduction());
    });
}

/// Varying batch sizes to show pool effectiveness at different scales
fn bench_pool_varying_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_reduction_batch_sizes");

    for batch_size in [10, 25, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let config = PoolConfig::new(64, 256).with_preallocate(true);
                let mut pool = TensorPool::new(config);

                b.iter(|| {
                    for _ in 0..batch_size {
                        let tensor = pool.acquire(black_box(vec![64, 64]));
                        drop(tensor);
                    }
                });

                let stats = pool.stats();
                eprintln!(
                    "Batch {} - Hit rate: {:.2}%, Reduction: {:.2}%",
                    batch_size,
                    stats.hit_rate() * 100.0,
                    stats.allocation_reduction()
                );
            },
        );
    }
    group.finish();
}

/// Warm vs Cold Pool: Show preallocation benefit
fn bench_pool_warm_vs_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_reduction_warm_cold");

    // Cold pool (no preallocation)
    group.bench_function("cold_pool_no_prealloc", |b| {
        let config = PoolConfig::new(64, 256).with_preallocate(false);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            for _ in 0..50 {
                let tensor = pool.acquire(black_box(vec![64, 64]));
                drop(tensor);
            }
        });

        let stats = pool.stats();
        eprintln!("=== Cold Pool (no prealloc) ===");
        eprintln!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("Reduction: {:.2}%", stats.allocation_reduction());
    });

    // Warm pool (with preallocation)
    group.bench_function("warm_pool_with_prealloc", |b| {
        let config = PoolConfig::new(64, 256).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            for _ in 0..50 {
                let tensor = pool.acquire(black_box(vec![64, 64]));
                drop(tensor);
            }
        });

        let stats = pool.stats();
        eprintln!("=== Warm Pool (with prealloc) ===");
        eprintln!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("Reduction: {:.2}%", stats.allocation_reduction());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_iterative_allocation_without_pool,
    bench_iterative_allocation_with_pool,
    bench_gnn_iteration_with_pool,
    bench_matmul_temporaries_with_pool,
    bench_pool_varying_batch_sizes,
    bench_pool_warm_vs_cold,
);

criterion_main!(benches);
