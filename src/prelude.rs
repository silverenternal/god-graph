//! Prelude 模块
//!
//! 提供常用的导入，方便使用

pub use crate::edge::{EdgeIndex, EdgeRef};
pub use crate::errors::{GraphError, GraphResult};
pub use crate::graph::builders::GraphBuilder;
pub use crate::graph::traits::{Direction, GraphBase, GraphOps, GraphQuery};
pub use crate::graph::Graph;
pub use crate::node::{NodeIndex, NodeRef};

// 常用算法
pub use crate::algorithms::centrality::{degree_centrality, pagerank};
pub use crate::algorithms::community::connected_components;
pub use crate::algorithms::properties::{
    density, diameter, has_cycle, is_bipartite, is_connected, is_dag, is_tree,
};
pub use crate::algorithms::shortest_path::{astar, bellman_ford, dijkstra};
pub use crate::algorithms::traversal::{all_paths, bfs, dfs, topological_sort};

// 常用图生成器
pub use crate::generators::{complete_graph, erdos_renyi_graph, grid_graph, tree_graph};

// 导出工具
pub use crate::export::{to_adjacency_list, to_dot, to_edge_list};

// 工具
pub use crate::utils::{Arena, Padded};
