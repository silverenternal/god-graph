//! 图生成器模块
//!
//! 提供各种随机图和标准图的生成算法

pub mod erdos_renyi;
pub mod barabasi_albert;
pub mod watts_strogatz;
pub mod grid;
pub mod complete;
pub mod tree;

pub use erdos_renyi::erdos_renyi_graph;
pub use barabasi_albert::barabasi_albert_graph;
pub use watts_strogatz::watts_strogatz_graph;
pub use grid::grid_graph;
pub use complete::complete_graph;
pub use tree::tree_graph;
