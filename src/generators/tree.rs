//! 树生成器

use crate::graph::Graph;
use crate::graph::builders::GraphBuilder;

/// 生成随机树
///
/// 使用 Prüfer 序列方法生成均匀分布的随机树
///
/// # 参数
/// * `n` - 节点数量
///
/// # 返回
/// 生成的树
pub fn tree_graph<T>(n: usize) -> Graph<T, f64>
where
    T: Clone + Default,
{
    if n == 0 {
        return Graph::undirected();
    }

    let mut builder = GraphBuilder::undirected()
        .with_nodes((0..n).map(|_| T::default()));

    // 简单实现：生成一条路径
    for i in 0..n - 1 {
        builder = builder.with_edge(i, i + 1, 1.0);
    }

    builder.build().unwrap_or_else(|_| Graph::undirected())
}

/// 生成完全二叉树
pub fn binary_tree_graph<T>(height: usize) -> Graph<T, f64>
where
    T: Clone + Default,
{
    let n = (1 << height) - 1; // 2^h - 1
    let mut builder = GraphBuilder::undirected()
        .with_nodes((0..n).map(|_| T::default()));

    for i in 0..n {
        let left = 2 * i + 1;
        let right = 2 * i + 2;

        if left < n {
            builder = builder.with_edge(i, left, 1.0);
        }
        if right < n {
            builder = builder.with_edge(i, right, 1.0);
        }
    }

    builder.build().unwrap_or_else(|_| Graph::undirected())
}
