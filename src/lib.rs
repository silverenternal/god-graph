//! # God-Graph: 高性能 Rust 图操作库
//!
//! God-Graph 是一个设计用于高性能计算的图数据结构和算法库，
//! 采用 CSR 格式、Arena 分配器和并行计算优化。
//!
//! ## 特性
//!
//! - **高性能内存布局**: CSR (Compressed Sparse Row) 格式，cache-friendly
//! - **稳定索引**: Arena 分配器 + generation 计数，防止 ABA 问题
//! - **并行算法**: 基于 rayon 的并行 BFS、PageRank 等算法
//! - **泛型支持**: 节点和边支持任意数据类型
//!
//! ## 快速开始
//!
//! ```
//! use god_gragh::prelude::*;
//!
//! // 创建有向图
//! let mut graph = Graph::<String, f64>::directed();
//!
//! // 添加节点
//! let a = graph.add_node("A".to_string()).unwrap();
//! let b = graph.add_node("B".to_string()).unwrap();
//! let c = graph.add_node("C".to_string()).unwrap();
//!
//! // 添加边
//! graph.add_edge(a, b, 1.0).unwrap();
//! graph.add_edge(b, c, 2.0).unwrap();
//! graph.add_edge(a, c, 4.0).unwrap();
//!
//! // 遍历邻居
//! for neighbor in graph.neighbors(a) {
//!     println!("Neighbor: {}", graph[neighbor]);
//! }
//! ```
//!
//! ## 并行算法（需要 `parallel` 特性）
//!
//! ```no_run
//! # #[cfg(feature = "parallel")]
//! # {
//! use god_gragh::prelude::*;
//! use god_gragh::algorithms::parallel::par_pagerank;
//!
//! let mut graph = Graph::<i32, f64>::undirected();
//! // 使用并行 PageRank
//! let ranks = par_pagerank(&graph, 0.85, 20);
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod errors;
pub mod graph;
pub mod node;
pub mod edge;
pub mod algorithms;
pub mod generators;
pub mod export;
pub mod utils;
pub mod prelude;

// 重新导出核心类型
pub use errors::{GraphError, GraphResult};
pub use graph::Graph;
pub use node::{NodeIndex, NodeRef};
pub use edge::{EdgeIndex, EdgeRef};

/// 库版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// 最大支持的节点数量
pub const MAX_NODES: usize = usize::MAX >> 1;

/// 最大支持的边数量
pub const MAX_EDGES: usize = usize::MAX >> 1;
