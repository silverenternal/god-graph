//! 算法模块
//!
//! 包含各种图算法实现

pub mod traversal;
pub mod shortest_path;
pub mod mst;
pub mod centrality;
pub mod flow;
pub mod matching;
pub mod community;
pub mod properties;

/// 并行算法模块（需要 `parallel` 特性）
#[cfg(feature = "parallel")]
pub mod parallel;
