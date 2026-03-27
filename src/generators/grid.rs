//! 网格图生成器

use crate::graph::Graph;
use crate::graph::builders::GraphBuilder;

/// 生成 2D 网格图
///
/// # 参数
/// * `rows` - 行数
/// * `cols` - 列数
/// * `diagonal` - 是否包含对角线连接
///
/// # 返回
/// 生成的网格图
pub fn grid_graph<T>(rows: usize, cols: usize, diagonal: bool) -> Graph<T, f64>
where
    T: Clone + Default,
{
    let n = rows * cols;
    let mut builder = GraphBuilder::undirected()
        .with_nodes((0..n).map(|_| T::default()));

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;

            // 向右连接
            if c + 1 < cols {
                builder = builder.with_edge(idx, idx + 1, 1.0);
            }

            // 向下连接
            if r + 1 < rows {
                builder = builder.with_edge(idx, idx + cols, 1.0);
            }

            // 对角线连接
            if diagonal {
                if r + 1 < rows && c + 1 < cols {
                    builder = builder.with_edge(idx, idx + cols + 1, 1.0);
                }
                if r + 1 < rows && c > 0 {
                    builder = builder.with_edge(idx, idx + cols - 1, 1.0);
                }
            }
        }
    }

    builder.build().unwrap_or_else(|_| Graph::undirected())
}
