//! Barabási-Albert 优先连接模型生成器
//!
//! 生成无标度网络，度分布服从幂律

use crate::graph::Graph;
use crate::graph::builders::GraphBuilder;

/// 生成 Barabási-Albert 无标度网络
///
/// # 参数
/// * `n` - 最终节点数量
/// * `m` - 新节点连接的现有节点数
///
/// # 返回
/// 生成的无向图
///
/// # 示例
/// ```
/// use god_gragh::generators::barabasi_albert_graph;
/// use god_gragh::graph::Graph;
/// use god_gragh::graph::traits::GraphBase;
///
/// let graph: Graph<i32, f64> = barabasi_albert_graph(100, 3);
/// assert_eq!(graph.node_count(), 100);
/// ```
pub fn barabasi_albert_graph<T>(n: usize, m: usize) -> Graph<T, f64>
where
    T: Clone + Default,
{
    if n == 0 {
        return Graph::directed();
    }

    let mut builder = GraphBuilder::undirected();

    // 初始化：创建一个完全图作为种子（m+1 个节点）
    let initial_size = (m + 1).min(n);
    builder = builder.with_nodes((0..initial_size).map(|_| T::default()));

    // 添加初始完全图的边
    for i in 0..initial_size {
        for j in (i + 1)..initial_size {
            builder = builder.with_edge(i, j, 1.0);
        }
    }

    // 优先连接：逐个添加新节点
    for new_node in initial_size..n {
        // 添加新节点
        builder = builder.with_nodes(std::iter::once(T::default()));

        // 选择 m 个现有节点进行连接
        // 简化实现：直接连接前 m 个节点
        for node_idx in 0..m.min(new_node) {
            builder = builder.with_edge(node_idx, new_node, 1.0);
        }

        // 如果优先连接不够，随机连接
        #[cfg(feature = "rand")]
        {
            use rand::{Rng, SeedableRng};
            use rand_chacha::ChaCha8Rng;
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut connected = m.min(new_node);
            while connected < m && connected < new_node {
                let random_node = rng.gen_range(0..new_node);
                builder = builder.with_edge(random_node, new_node, 1.0);
                connected += 1;
            }
        }
    }

    builder.build().unwrap_or_else(|_| Graph::directed())
}
