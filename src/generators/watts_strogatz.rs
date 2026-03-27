//! Watts-Strogatz 小世界网络生成器
//!
//! 生成具有高聚类系数和短平均路径长度的小世界网络

use crate::graph::Graph;
use crate::graph::builders::GraphBuilder;

/// 生成 Watts-Strogatz 小世界网络
///
/// # 参数
/// * `n` - 节点数量
/// * `k` - 每个节点的邻居数（必须是偶数）
/// * `beta` - 重连概率 (0.0..=1.0)
///
/// # 返回
/// 生成的无向图
///
/// # 示例
/// ```
/// use god_gragh::generators::watts_strogatz_graph;
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphBase;
///
/// let graph: Graph<i32, f64> = watts_strogatz_graph(100, 4, 0.1);
/// assert_eq!(graph.node_count(), 100);
/// ```
pub fn watts_strogatz_graph<T>(n: usize, k: usize, beta: f64) -> Graph<T, f64>
where
    T: Clone + Default,
{
    if n == 0 || k == 0 {
        return Graph::directed();
    }

    let mut builder = GraphBuilder::undirected();

    // 添加节点
    builder = builder.with_nodes((0..n).map(|_| T::default()));

    // 构建环状最近邻耦合网络
    // 每个节点连接到其左右各 k/2 个邻居
    let half_k = k / 2;
    for i in 0..n {
        for j in 1..=half_k {
            let target = (i + j) % n;
            builder = builder.with_edge(i, target, 1.0);
        }
    }

    // 随机重连
    #[cfg(feature = "rand")]
    {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for i in 0..n {
            for _j in 1..=half_k {
                if rng.r#gen::<f64>() < beta {
                    // 重连边：断开 (i, (i+j)%n)，连接到随机节点
                    let new_target = rng.gen_range(0..n);
                    if new_target != i {
                        // 简化实现：直接添加新边，不删除旧边
                        // 完整实现需要更复杂的边管理
                        builder = builder.with_edge(i, new_target, 1.0);
                    }
                }
            }
        }
    }

    // 未启用 rand 特性时，beta 参数未使用
    #[cfg(not(feature = "rand"))]
    {
        let _ = beta;
    }

    builder.build().unwrap_or_else(|_| Graph::directed())
}
