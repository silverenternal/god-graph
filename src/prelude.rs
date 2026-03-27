//! Prelude 模块
//!
//! 提供常用的导入，方便使用

pub use crate::graph::Graph;
pub use crate::graph::traits::{Direction, GraphBase, GraphOps, GraphQuery};
pub use crate::graph::builders::GraphBuilder;
pub use crate::node::{NodeIndex, NodeRef};
pub use crate::edge::{EdgeIndex, EdgeRef};
pub use crate::errors::{GraphError, GraphResult};

// 常用算法
pub use crate::algorithms::traversal::{bfs, dfs, topological_sort, all_paths};
pub use crate::algorithms::shortest_path::{dijkstra, bellman_ford, astar};
pub use crate::algorithms::centrality::{pagerank, degree_centrality};
pub use crate::algorithms::community::connected_components;
pub use crate::algorithms::properties::{
    is_connected, is_bipartite, is_dag, is_tree, has_cycle, diameter, density,
};

// 常用图生成器
pub use crate::generators::{
    grid_graph, complete_graph, tree_graph, erdos_renyi_graph,
};

// 导出工具
pub use crate::export::{to_dot, to_adjacency_list, to_edge_list};

// 工具
pub use crate::utils::{Arena, Padded};
