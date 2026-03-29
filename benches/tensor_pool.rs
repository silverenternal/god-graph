//! Tensor Pool Benchmarks
//!
//! This module benchmarks the tensor memory pool performance,
//! measuring allocation reuse, hit rate, and throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_gragh::tensor::dense::DenseTensor;
use god_gragh::tensor::pool::{PoolConfig, TensorPool};

/// Benchmark: iterative allocation without pool
fn bench_iterative_allocation_without_pool(c: &mut Criterion) {
    c.bench_function("iterative_alloc_without_pool", |b| {
        b.iter(|| {
            let mut tensors = Vec::new();
            for _ in 0..50 {
                let tensor = DenseTensor::zeros(black_box(vec![128, 128]));
                tensors.push(tensor);
            }
            drop(tensors);
        });
    });
}

/// Benchmark: iterative allocation with pool (single tensor per iteration)
fn bench_iterative_allocation_with_pool(c: &mut Criterion) {
    c.bench_function("iterative_alloc_with_pool", |b| {
        let config = PoolConfig::new(16, 128).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            // Single tensor allocation/deallocation per iteration
            let tensor = pool.acquire(black_box(vec![128, 128]));
            drop(tensor);
        });

        let stats = pool.stats();
        eprintln!("Pool hit rate: {:.2}%", stats.hit_rate() * 100.0);
        eprintln!("New allocations (misses): {}", stats.pool_misses);
    });
}

/// Benchmark: GNN iteration (single hidden state per step)
fn bench_gnn_iteration_with_pool(c: &mut Criterion) {
    c.bench_function("gnn_iteration_with_pool", |b| {
        let config = PoolConfig::new(16, 128).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            for _ in 0..10 {
                let hidden = pool.acquire(black_box(vec![100, 64]));
                drop(hidden);
            }
        });

        let stats = pool.stats();
        eprintln!("GNN Pool hit rate: {:.2}%", stats.hit_rate() * 100.0);
    });
}

/// Benchmark: matrix multiplication temporaries (sequential)
fn bench_matmul_temporaries_with_pool(c: &mut Criterion) {
    c.bench_function("matmul_temporaries_with_pool", |b| {
        let config = PoolConfig::new(8, 64).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            for _ in 0..20 {
                // Sequential acquire/release
                {
                    let _a = pool.acquire(black_box(vec![64, 64]));
                }
                {
                    let _b = pool.acquire(black_box(vec![64, 64]));
                }
                {
                    let _c = pool.acquire(black_box(vec![64, 64]));
                }
            }
        });

        let stats = pool.stats();
        eprintln!("MatMul Pool hit rate: {:.2}%", stats.hit_rate() * 100.0);
    });
}

/// Benchmark: small tensor allocations (single per iteration)
fn bench_small_tensor_allocation_with_pool(c: &mut Criterion) {
    c.bench_function("small_tensor_alloc_with_pool", |b| {
        let config = PoolConfig::new(32, 256).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            let tensor = pool.acquire(black_box(vec![16, 16]));
            drop(tensor);
        });

        let stats = pool.stats();
        eprintln!(
            "Small tensor Pool hit rate: {:.2}%",
            stats.hit_rate() * 100.0
        );
    });
}

/// Benchmark: large tensor allocations (single per iteration)
fn bench_large_tensor_allocation_with_pool(c: &mut Criterion) {
    c.bench_function("large_tensor_alloc_with_pool", |b| {
        let config = PoolConfig::new(4, 16).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            let tensor = pool.acquire(black_box(vec![512, 512]));
            drop(tensor);
        });

        let stats = pool.stats();
        eprintln!(
            "Large tensor Pool hit rate: {:.2}%",
            stats.hit_rate() * 100.0
        );
    });
}

/// Benchmark: sequential allocation and deallocation
fn bench_sequential_alloc_dealloc_with_pool(c: &mut Criterion) {
    c.bench_function("sequential_alloc_dealloc_with_pool", |b| {
        let config = PoolConfig::new(8, 64).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            for _ in 0..50 {
                let tensor = pool.acquire(black_box(vec![64, 64]));
                drop(tensor);
            }
        });

        let stats = pool.stats();
        eprintln!("Sequential Pool hit rate: {:.2}%", stats.hit_rate() * 100.0);
    });
}

/// Benchmark: pool performance with varying batch sizes
fn bench_pool_varying_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_batch_sizes");

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
                    "Batch {} - Pool hit rate: {:.2}%",
                    batch_size,
                    stats.hit_rate() * 100.0
                );
            },
        );
    }
    group.finish();
}

/// Benchmark: pool warm vs cold performance
fn bench_pool_warm_vs_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_warm_cold");

    // Cold pool (no preallocation)
    group.bench_function("cold_pool_no_prealloc", |b| {
        let config = PoolConfig::new(64, 256).with_preallocate(false);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            let tensor = pool.acquire(black_box(vec![64, 64]));
            drop(tensor);
        });

        let stats = pool.stats();
        eprintln!(
            "Cold pool - Pool hit rate: {:.2}%",
            stats.hit_rate() * 100.0
        );
    });

    // Warm pool (with preallocation)
    group.bench_function("warm_pool_with_prealloc", |b| {
        let config = PoolConfig::new(64, 256).with_preallocate(true);
        let mut pool = TensorPool::new(config);

        b.iter(|| {
            let tensor = pool.acquire(black_box(vec![64, 64]));
            drop(tensor);
        });

        let stats = pool.stats();
        eprintln!(
            "Warm pool - Pool hit rate: {:.2}%",
            stats.hit_rate() * 100.0
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_iterative_allocation_without_pool,
    bench_iterative_allocation_with_pool,
    bench_gnn_iteration_with_pool,
    bench_matmul_temporaries_with_pool,
    bench_small_tensor_allocation_with_pool,
    bench_large_tensor_allocation_with_pool,
    bench_sequential_alloc_dealloc_with_pool,
    bench_pool_varying_batch_sizes,
    bench_pool_warm_vs_cold,
);

criterion_main!(benches);
