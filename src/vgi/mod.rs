//! VGI (Virtual Graph Interface) - 虚拟图接口层
//!
//! 提供统一的图抽象接口，支持多种后端实现和插件系统
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Application Layer                        │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Plugin System Layer                       │
//! │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │   │  Algorithm  │  │  Analyzer   │  │  Visualizer │  ...   │
//! │   │   Plugin    │  │   Plugin    │  │   Plugin    │        │
//! │   └─────────────┘  └─────────────┘  └─────────────┘        │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Virtual Graph Interface (VGI)                  │
//! │   ┌─────────────────────────────────────────────────────┐   │
//! │   │              VirtualGraph Trait                      │   │
//! │   │  - nodes() / edges()                                │   │
//! │   │  - add_node() / add_edge()                          │   │
//! │   │  - neighbors() / incident_edges()                   │   │
//! │   │  - metadata() / capabilities()                      │   │
//! │   └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Backend Layer                            │
//! │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │   │   Single    │  │  Distributed│  │   External  │        │
//! │   │   Machine   │  │   Cluster   │  │   Database  │        │
//! │   └─────────────┘  └─────────────┘  └─────────────┘        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Trait Hierarchy
//!
//! VGI 采用分层设计，降低学习成本：
//!
//! ```text
//! GraphRead (只读查询)
//!     ↓
//! GraphUpdate (增量更新)
//!     ↓
//! GraphAdvanced (高级操作)
//!     ↓
//! VirtualGraph (完整功能 = GraphRead + GraphUpdate + GraphAdvanced)
//! ```
//!
//! # 使用指南
//!
//! ## 场景 1: 只读算法
//!
//! ```rust
//! use god_graph::vgi::traits::GraphRead;
//!
//! fn analyze<G: GraphRead>(graph: &G) {
//!     println!("Nodes: {}", graph.node_count());
//! }
//! ```
//!
//! ## 场景 2: 图构建
//!
//! ```rust
//! use god_graph::vgi::traits::GraphUpdate;
//!
//! fn build<G: GraphUpdate>(graph: &mut G) {
//!     let n = graph.add_node("data".to_string()).unwrap();
//! }
//! ```
//!
//! ## 场景 3: 完整功能
//!
//! ```rust
//! use god_graph::vgi::VirtualGraph;
//!
//! fn process<G: VirtualGraph>(graph: &mut G) {
//!     // 使用所有功能
//! }
//! ```

pub mod builder;
pub mod error;
pub mod impl_graph;
pub mod metadata;
pub mod simple;
pub mod traits;

pub use builder::GraphBuilder;
pub use error::{ErrorContext, VgiError, VgiResult};
pub use metadata::{Capability, GraphMetadata, GraphType};
pub use simple::SimpleGraph;
pub use traits::{Backend, GraphAdvanced, GraphRead, GraphUpdate, VirtualGraph};
