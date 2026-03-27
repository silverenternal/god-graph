//! 完全图生成器

use crate::graph::builders::GraphBuilder;
use crate::graph::Graph;

/// 生成完全图 K_n
///
/// 完全图是每个节点都与其他所有节点相连的图
///
/// # 参数
/// * `n` - 节点数量
///
/// # 返回
/// 生成的完全图
pub fn complete_graph<T>(n: usize) -> Graph<T, f64>
where
    T: Clone + Default,
{
    let mut builder = GraphBuilder::undirected().with_nodes((0..n).map(|_| T::default()));

    for i in 0..n {
        for j in (i + 1)..n {
            builder = builder.with_edge(i, j, 1.0);
        }
    }

    builder.build().unwrap_or_else(|_| Graph::undirected())
}
