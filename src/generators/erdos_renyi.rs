//! Erdős-Rényi 随机图生成器
//!
//! G(n, p) 模型：每个边以概率 p 独立存在

use crate::graph::Graph;
use crate::graph::builders::GraphBuilder;

/// 生成 Erdős-Rényi 随机图 G(n, p)
///
/// # 参数
/// * `n` - 节点数量
/// * `p` - 边存在的概率 (0.0..=1.0)
/// * `directed` - 是否为有向图
/// * `seed` - 随机种子
///
/// # 返回
/// 生成的图
pub fn erdos_renyi_graph<T>(
    n: usize,
    p: f64,
    directed: bool,
    seed: u64,
) -> Graph<T, f64>
where
    T: Clone + Default,
{
    let mut builder = if directed {
        GraphBuilder::directed()
    } else {
        GraphBuilder::undirected()
    };

    // 添加节点
    builder = builder.with_nodes((0..n).map(|_| T::default()));

    // 添加边（简化实现，实际需要随机数生成器）
    #[cfg(feature = "rand")]
    {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for i in 0..n {
            for j in (if directed { 0 } else { i + 1 })..n {
                if i != j && rng.r#gen::<f64>() < p {
                    builder = builder.with_edge(i, j, 1.0);
                }
            }
        }
    }

    // 未启用 rand 特性时，返回空图
    #[cfg(not(feature = "rand"))]
    {
        let _ = (n, p, seed); // 避免未使用变量警告
    }

    builder.build().unwrap_or_else(|_| Graph::directed())
}
