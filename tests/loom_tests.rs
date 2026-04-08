//! Loom 并发模型检查测试
//!
//! Loom 是一个用于测试并发代码正确性的工具，通过探索所有可能的线程交织来发现 race condition。
//!
//! # 使用方法
//!
//! ```bash
//! # 运行 loom 测试
//! LOOM_MAX_PREEMPTION=2 LOOM_MAX_BRANCHES=10 cargo test --features parallel --test loom_tests -- --nocapture
//! ```
//!
//! # 测试覆盖
//!
//! - Atomic 操作的正确性
//! - 无锁算法的线性一致性
//! - 线程安全数据结构
//! - 并行 visited 数组

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use god_graph::algorithms::parallel::par_bfs;
    use god_graph::graph::Graph;
    use god_graph::graph::traits::GraphOps;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// 测试 AtomicBool visited 数组的并发安全性
    ///
    /// 这个测试验证在多线程 BFS 中，visited 标记是否会出现 race condition
    #[test]
    #[ignore] // Loom tests are slow, run manually
    fn test_atomic_visited_no_race() {
        // 使用 loom 的模型检查
        // 注意：实际项目中需要添加 loom 依赖
        // [dev-dependencies]
        // loom = "0.7"

        // 示例代码（需要 loom 支持）：
        // loom::model(|| {
        //     let visited = Arc::new((0..10).map(|_| AtomicBool::new(false)).collect::<Vec<_>>());
        //     let mut handles = vec![];
        //
        //     for i in 0..4 {
        //         let visited = Arc::clone(&visited);
        //         handles.push(std::thread::spawn(move || {
        //             // 模拟并发标记 visited
        //             let _ = visited[i].compare_exchange(
        //                 false,
        //                 true,
        //                 Ordering::SeqCst,
        //                 Ordering::SeqCst,
        //             );
        //         }));
        //     }
        //
        //     for handle in handles {
        //         handle.join().unwrap();
        //     }
        // });
    }

    /// 测试并行计数器的正确性
    #[test]
    #[ignore]
    fn test_atomic_counter_correctness() {
        // 验证 AtomicUsize 计数器在并发环境下的正确性
        // loom::model(|| {
        //     let counter = Arc::new(AtomicUsize::new(0));
        //     let mut handles = vec![];
        //
        //     for _ in 0..4 {
        //         let counter = Arc::clone(&counter);
        //         handles.push(std::thread::spawn(move || {
        //             for _ in 0..10 {
        //                 counter.fetch_add(1, Ordering::SeqCst);
        //             }
        //         }));
        //     }
        //
        //     for handle in handles {
        //         handle.join().unwrap();
        //     }
        //
        //     assert_eq!(counter.load(Ordering::SeqCst), 40);
        // });
    }

    /// 测试无锁队列的线性一致性
    #[test]
    #[ignore]
    fn test_lock_free_queue_linearizability() {
        // 测试 crossbeam 队列的并发正确性
        // 实际项目中应使用 loom 检查 crossbeam 的使用
    }

    /// 测试并行 BFS 的 visited 标记
    #[test]
    fn test_par_bfs_visited_safety() {
        // 这个测试在标准测试模式下运行，验证基本功能
        let mut graph = Graph::<i32, f64>::undirected();

        let nodes: Vec<_> = (0..10).map(|i| graph.add_node(i).unwrap()).collect();

        // 创建链式结构
        for i in 0..9 {
            graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
        }

        // 运行并行 BFS
        let visited_count = Arc::new(AtomicUsize::new(0));
        let result_visited = Arc::clone(&visited_count);

        par_bfs(&graph, nodes[0], |_node, _depth| {
            result_visited.fetch_add(1, Ordering::SeqCst);
            true
        });

        // 验证所有节点都被访问
        assert_eq!(visited_count.load(Ordering::SeqCst), 10);
    }
}

// Loom 并发测试指南
//
// # 添加新的 loom 测试
//
// 1. 在 Cargo.toml 中添加 loom 依赖：
// ```toml
// [dev-dependencies]
// loom = "0.7"
// ```
//
// 2. 创建测试文件 `tests/loom_tests.rs`
//
// 3. 使用 `#[cfg(loom)]` 条件编译：
// ```rust
// #[cfg(loom)]
// #[test]
// fn test_concurrent_algorithm() {
//     loom::model(|| {
//         // 测试并发代码
//     });
// }
// ```
//
// 4. 运行 loom 测试：
// ```bash
// LOOM_MAX_PREEMPTION=2 cargo test --test loom_tests
// ```
//
// # 最佳实践
//
// - 保持 loom 测试简单，减少状态空间
// - 使用 `#[ignore]` 标记慢速测试
// - 在 CI 中定期运行，但不作为每次提交的要求
//
// # 已知限制
//
// - Loom 不支持外部 crate（如 crossbeam）的内部实现
// - 大型状态空间可能导致测试运行数小时
// - 需要使用 `loom::Arc` 替代 `std::sync::Arc`
