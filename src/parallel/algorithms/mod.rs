//! 分布式图算法模块
//!
//! 提供分布式环境下的图算法实现
//!
//! # 可用算法
//!
//! - **PageRank**: 分布式 PageRank 算法
//! - **BFS**: 分布式广度优先搜索
//! - **DFS**: 分布式深度优先搜索
//! - **ConnectedComponents**: 分布式连通分量算法
//! - **Dijkstra**: 分布式最短路径算法
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::algorithms::{DistributedPageRank, DistributedBFS, DistributedDFS, DistributedConnectedComponents, DistributedDijkstra};
//! use god_graph::parallel::partitioner::HashPartitioner;
//! use god_graph::graph::Graph;
//!
//! // 创建图和分区器
//! let mut graph = Graph::<(), ()>::undirected();
//! let partitioner = HashPartitioner::new(4);
//! let partitions = partitioner.partition_graph(&graph);
//!
//! // 运行分布式 PageRank
//! let pr = DistributedPageRank::new(0.85, 20, 1e-6);
//! let ranks = pr.compute(&graph, &partitions);
//! ```

pub mod bfs;
pub mod connected_components;
pub mod dfs;
pub mod dijkstra;
pub mod pagerank;

pub use bfs::{BFSConfig, BFSConfigError, BFSResult, DistributedBFS};
pub use connected_components::{
    ConnectedComponentsConfig, ConnectedComponentsConfigError, ConnectedComponentsResult,
    DistributedConnectedComponents,
};
pub use dfs::{DFSConfig, DFSConfigError, DFSResult, DistributedDFS};
pub use dijkstra::{DijkstraConfig, DijkstraConfigError, DijkstraResult, DistributedDijkstra};
pub use pagerank::{DistributedPageRank, PageRankConfig, PageRankConfigError, PageRankResult};
