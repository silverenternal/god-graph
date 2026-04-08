//! 内置图算法插件
//!
//! 提供常用的图算法插件实现
//!
//! # 可用算法
//!
//! ## 基础遍历
//! - **BFS**: 广度优先搜索
//! - **DFS**: 深度优先搜索
//!
//! ## 最短路径
//! - **Dijkstra**: 单源最短路径（非负权重）
//! - **Bellman-Ford**: 单源最短路径（支持负权重）
//!
//! ## 中心性算法
//! - **PageRank**: 节点中心性排名算法
//! - **BetweennessCentrality**: 介数中心性（节点作为桥梁的重要性）
//! - **ClosenessCentrality**: 接近中心性（节点到其他节点的平均距离）
//!
//! ## 社区检测
//! - **ConnectedComponents**: 连通分量检测算法
//! - **Louvain**: Louvain 社区检测算法
//!
//! ## 图遍历应用
//! - **TopologicalSort**: 有向无环图 (DAG) 的拓扑排序
//!
//! # 使用示例
//!
//! ```
//! use god_graph::plugins::algorithms::{
//!     PageRankPlugin, BfsPlugin, DfsPlugin, ConnectedComponentsPlugin,
//!     DijkstraPlugin, LouvainPlugin, TopologicalSortPlugin,
//! };
//! use god_graph::plugins::GraphAlgorithm;
//!
//! // PageRank
//! let pagerank = PageRankPlugin::default();
//! assert_eq!(pagerank.info().name, "pagerank");
//!
//! // Dijkstra
//! let dijkstra = DijkstraPlugin::from_source(0);
//! assert_eq!(dijkstra.info().name, "dijkstra");
//!
//! // Louvain
//! let louvain = LouvainPlugin::default_params();
//! assert_eq!(louvain.info().name, "louvain");
//!
//! // Topological Sort
//! let topo_sort = TopologicalSortPlugin::new();
//! assert_eq!(topo_sort.info().name, "topological-sort");
//! ```

pub mod bellman_ford;
pub mod betweenness_centrality;
pub mod bfs;
pub mod closeness_centrality;
pub mod connected_components;
pub mod dfs;
pub mod dijkstra;
pub mod louvain;
pub mod pagerank;
pub mod topological_sort;

pub use bellman_ford::BellmanFordPlugin;
pub use betweenness_centrality::BetweennessCentralityPlugin;
pub use bfs::BfsPlugin;
pub use closeness_centrality::ClosenessCentralityPlugin;
pub use connected_components::ConnectedComponentsPlugin;
pub use dfs::DfsPlugin;
pub use dijkstra::DijkstraPlugin;
pub use louvain::LouvainPlugin;
pub use pagerank::PageRankPlugin;
pub use topological_sort::TopologicalSortPlugin;

/// 获取所有内置算法的名称列表
pub fn list_builtin_algorithms() -> Vec<&'static str> {
    vec![
        "pagerank",
        "bfs",
        "dfs",
        "connected_components",
        "dijkstra",
        "bellman-ford",
        "betweenness-centrality",
        "closeness-centrality",
        "louvain",
        "topological-sort",
    ]
}

/// 创建默认的 PageRank 插件
pub fn create_pagerank() -> PageRankPlugin {
    PageRankPlugin::default()
}

/// 创建 BFS 插件
pub fn create_bfs(start_node: usize) -> BfsPlugin {
    BfsPlugin::new(start_node)
}

/// 创建 DFS 插件
pub fn create_dfs(start_node: usize) -> DfsPlugin {
    DfsPlugin::new(start_node)
}

/// 创建连通分量插件
pub fn create_connected_components() -> ConnectedComponentsPlugin {
    ConnectedComponentsPlugin::new()
}

/// 创建 Dijkstra 插件
pub fn create_dijkstra_from_source(source: usize) -> DijkstraPlugin {
    DijkstraPlugin::from_source(source)
}

/// 创建 Bellman-Ford 插件
pub fn create_bellman_ford_from_source(source: usize) -> BellmanFordPlugin {
    BellmanFordPlugin::from_source(source)
}

/// 创建介数中心性插件（归一化）
pub fn create_betweenness_centrality_normalized() -> BetweennessCentralityPlugin {
    BetweennessCentralityPlugin::normalized()
}

/// 创建接近中心性插件（改进方法）
pub fn create_closeness_centrality_improved() -> ClosenessCentralityPlugin {
    ClosenessCentralityPlugin::improved()
}

/// 创建 Louvain 社区检测插件（默认参数）
pub fn create_louvain_default() -> LouvainPlugin {
    LouvainPlugin::default_params()
}

/// 创建拓扑排序插件
pub fn create_topological_sort() -> TopologicalSortPlugin {
    TopologicalSortPlugin::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::algorithm::GraphAlgorithm;

    #[test]
    fn test_list_builtin_algorithms() {
        let algorithms = list_builtin_algorithms();
        assert!(algorithms.contains(&"pagerank"));
        assert!(algorithms.contains(&"bfs"));
        assert!(algorithms.contains(&"dfs"));
        assert!(algorithms.contains(&"connected_components"));
        assert!(algorithms.contains(&"dijkstra"));
        assert!(algorithms.contains(&"bellman-ford"));
        assert!(algorithms.contains(&"betweenness-centrality"));
        assert!(algorithms.contains(&"closeness-centrality"));
        assert!(algorithms.contains(&"louvain"));
        assert!(algorithms.contains(&"topological-sort"));
        assert_eq!(algorithms.len(), 10);
    }

    #[test]
    fn test_create_helpers() {
        let pagerank = create_pagerank();
        assert_eq!(pagerank.info().name, "pagerank");

        let bfs = create_bfs(0);
        assert_eq!(bfs.info().name, "bfs");

        let dfs = create_dfs(0);
        assert_eq!(dfs.info().name, "dfs");

        let cc = create_connected_components();
        assert_eq!(cc.info().name, "connected_components");
    }
}
