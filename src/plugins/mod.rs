//! 插件系统
//!
//! 提供图算法插件的注册、管理和执行机制

pub mod algorithm;
pub mod algorithms;
pub mod registry;

pub use algorithm::{AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo};
pub use algorithms::{
    BellmanFordPlugin, BetweennessCentralityPlugin, BfsPlugin, ClosenessCentralityPlugin,
    ConnectedComponentsPlugin, DfsPlugin, DijkstraPlugin, LouvainPlugin, PageRankPlugin,
    TopologicalSortPlugin,
};
pub use registry::PluginRegistry;
