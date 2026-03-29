//! TensorArena 内存池性能基准测试
//!
//! 测试 TensorArena (bump allocator) 相比 TensorPool 的性能优势
//! 包括：
//! - 分配/回收延迟对比
//! - 内存复用效率
//! - 迭代算法场景性能
//! - 形状感知复用效果

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use god_gragh::tensor::dense::DenseTensor;
use god_gragh::tensor::pool::{PoolConfig, TensorArena, TensorPool};

/// TensorArena vs TensorPool: 基础分配性能对比
fn bench_arena_vs_pool_allocation(c: &mut Criterion) {
    let sizes = vec![32, 64, 128, 256];

    let mut group = c.benchmark_group("arena_vs_pool_allocation");
    for size in sizes {
        let shape = vec![size, size];

        // 传统分配 (baseline)
        group.bench_with_input(
            BenchmarkId::new("traditional_alloc", size),
            &shape,
            |b, s| b.iter(|| black_box(DenseTensor::zeros(s.clone()))),
        );

        // TensorPool 分配
        group.bench_with_input(BenchmarkId::new("pool_alloc", size), &shape, |b, s| {
            let config = PoolConfig::new(16, 64);
            let mut pool = TensorPool::new(config);
            b.iter(|| {
                let _tensor = black_box(pool.acquire(s.clone()));
            })
        });

        // TensorArena 分配
        group.bench_with_input(BenchmarkId::new("arena_alloc", size), &shape, |b, s| {
            let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
            b.iter(|| {
                let _tensor = black_box(arena.allocate(s).unwrap());
            })
        });
    }
    group.finish();
}

/// TensorArena vs TensorPool: 复用场景性能对比
fn bench_arena_vs_pool_reuse(c: &mut Criterion) {
    let sizes = vec![32, 64, 128, 256];
    let iterations = 100;

    let mut group = c.benchmark_group("arena_vs_pool_reuse");
    for size in sizes {
        let shape = vec![size, size];

        // TensorPool 复用
        group.bench_with_input(BenchmarkId::new("pool_reuse", size), &shape, |b, s| {
            let config = PoolConfig::new(16, 64);
            let mut pool = TensorPool::new(config);
            b.iter(|| {
                for _ in 0..iterations {
                    let tensor = black_box(pool.acquire(s.clone()));
                    drop(tensor); // 立即回收
                }
            })
        });

        // TensorArena 复用
        group.bench_with_input(BenchmarkId::new("arena_reuse", size), &shape, |b, s| {
            let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
            b.iter(|| {
                for _ in 0..iterations {
                    let tensor = black_box(arena.allocate(s).unwrap());
                    arena.deallocate(tensor); // 显式回收
                }
            })
        });
    }
    group.finish();
}

/// TensorArena: 形状感知复用效果测试
///
/// 验证形状匹配时的复用效率
fn bench_arena_shape_aware_reuse(c: &mut Criterion) {
    let shapes = vec![vec![16, 16], vec![32, 32], vec![64, 64], vec![128, 128]];
    let iterations = 50;

    let mut group = c.benchmark_group("arena_shape_aware_reuse");

    for shape in shapes {
        let size = shape[0];

        // 场景 1: 相同形状重复分配 (应触发复用)
        group.bench_with_input(
            BenchmarkId::new("same_shape_reuse", size),
            &shape,
            |b, s| {
                let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
                b.iter(|| {
                    let mut tensors = Vec::with_capacity(iterations);
                    for _ in 0..iterations {
                        let tensor = arena.allocate(s).unwrap();
                        tensors.push(tensor);
                    }
                    // 统一回收
                    for tensor in tensors {
                        arena.deallocate(tensor);
                    }
                })
            },
        );

        // 场景 2: 不同形状交替分配 (无复用)
        let shape2 = if size < 64 {
            vec![size * 2, size * 2]
        } else {
            vec![size / 2, size / 2]
        };
        group.bench_with_input(
            BenchmarkId::new("alternating_shapes", size),
            &(shape.clone(), shape2),
            |b, (s1, s2)| {
                let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
                b.iter(|| {
                    for i in 0..iterations {
                        let shape = if i % 2 == 0 { s1 } else { s2 };
                        let tensor = arena.allocate(shape).unwrap();
                        arena.deallocate(tensor);
                    }
                })
            },
        );
    }
    group.finish();
}

/// TensorArena: 迭代算法场景模拟 (如 PageRank, GNN 训练)
fn bench_arena_iterative_algorithms(c: &mut Criterion) {
    let iterations = vec![10, 50, 100, 200];
    let tensor_size = 128;
    let shape = vec![tensor_size, tensor_size];

    let mut group = c.benchmark_group("arena_iterative_algorithms");

    for &iters in &iterations {
        // 传统分配
        group.bench_with_input(
            BenchmarkId::new("traditional_iterative", iters),
            &shape,
            |b, s| {
                b.iter(|| {
                    for _ in 0..iters {
                        let _tensor = black_box(DenseTensor::zeros(s.clone()));
                    }
                })
            },
        );

        // TensorPool
        group.bench_with_input(BenchmarkId::new("pool_iterative", iters), &shape, |b, s| {
            let config = PoolConfig::new(iters, iters * 2);
            let mut pool = TensorPool::new(config);
            b.iter(|| {
                for _ in 0..iters {
                    let tensor = black_box(pool.acquire(s.clone()));
                    drop(tensor);
                }
            })
        });

        // TensorArena
        group.bench_with_input(
            BenchmarkId::new("arena_iterative", iters),
            &shape,
            |b, s| {
                let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
                b.iter(|| {
                    for _ in 0..iters {
                        let tensor = black_box(arena.allocate(s).unwrap());
                        arena.deallocate(tensor);
                    }
                })
            },
        );
    }
    group.finish();
}

/// TensorArena: 统计信息准确性验证
fn bench_arena_statistics(c: &mut Criterion) {
    let shapes = vec![vec![32, 32], vec![64, 64], vec![128, 128]];

    let mut group = c.benchmark_group("arena_statistics");

    for shape in shapes {
        let size = shape[0];

        group.bench_with_input(BenchmarkId::new("stats_overhead", size), &shape, |b, s| {
            let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
            b.iter(|| {
                let tensor = arena.allocate(s).unwrap();
                // 获取统计信息
                let _stats = black_box(arena.stats());
                arena.deallocate(tensor);
            })
        });
    }
    group.finish();
}

/// TensorArena: 内存效率测试
///
/// 测量峰值内存使用和复用率
fn bench_arena_memory_efficiency(c: &mut Criterion) {
    let scenarios = vec![
        ("single_large", vec![256, 256], 10),
        ("multiple_small", vec![16, 16], 100),
        ("mixed_shapes", vec![32, 32], 50),
    ];

    let mut group = c.benchmark_group("arena_memory_efficiency");

    for (name, base_shape, iterations) in scenarios {
        group.bench_with_input(
            BenchmarkId::new("memory_efficiency", name),
            &(base_shape.clone(), iterations),
            |b, &(ref shape, iters)| {
                b.iter(|| {
                    let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);

                    // 分配 - 回收循环
                    for _ in 0..iters {
                        let tensor = arena.allocate(shape).unwrap();
                        arena.deallocate(tensor);
                    }

                    // 返回统计信息
                    let stats = arena.stats();
                    (
                        stats.bytes_in_use,
                        stats.peak_bytes_in_use,
                        stats.reuse_ratio(),
                    )
                })
            },
        );
    }
    group.finish();
}

/// TensorArena: 多形状混合场景
///
/// 模拟真实场景中的多种形状混合分配
fn bench_arena_mixed_shapes(c: &mut Criterion) {
    let shape_sets = [
        vec![vec![16, 16], vec![32, 32], vec![64, 64]],
        vec![vec![8, 8, 8], vec![16, 16, 16]],
        vec![vec![4, 4, 4, 4], vec![8, 8, 8, 8]],
    ];

    let mut group = c.benchmark_group("arena_mixed_shapes");

    for (i, shapes) in shape_sets.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("mixed_ndim", i), shapes, |b, shapes| {
            let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
            b.iter(|| {
                for shape in shapes {
                    let tensor = arena.allocate(shape).unwrap();
                    arena.deallocate(tensor);
                }
            })
        });
    }
    group.finish();
}

/// TensorArena: Reset 性能测试
///
/// 测试 arena.reset() 相比逐个回收的性能
fn bench_arena_reset(c: &mut Criterion) {
    let sizes = vec![10, 50, 100, 200];
    let shape = vec![64, 64];

    let mut group = c.benchmark_group("arena_reset");

    for &num_tensors in &sizes {
        // 逐个回收
        group.bench_with_input(
            BenchmarkId::new("individual_dealloc", num_tensors),
            &shape,
            |b, s| {
                b.iter(|| {
                    let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
                    let mut tensors = Vec::with_capacity(num_tensors);
                    for _ in 0..num_tensors {
                        tensors.push(arena.allocate(s).unwrap());
                    }
                    // 逐个回收
                    for tensor in tensors {
                        arena.deallocate(tensor);
                    }
                })
            },
        );

        // 批量 reset
        group.bench_with_input(
            BenchmarkId::new("bulk_reset", num_tensors),
            &shape,
            |b, s| {
                b.iter(|| {
                    let mut arena = TensorArena::with_capacity(16 * 1024 * 1024);
                    for _ in 0..num_tensors {
                        let _tensor = arena.allocate(s).unwrap();
                    }
                    // 批量 reset
                    arena.reset();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_arena_vs_pool_allocation,
    bench_arena_vs_pool_reuse,
    bench_arena_shape_aware_reuse,
    bench_arena_iterative_algorithms,
    bench_arena_statistics,
    bench_arena_memory_efficiency,
    bench_arena_mixed_shapes,
    bench_arena_reset,
);

criterion_main!(benches);
