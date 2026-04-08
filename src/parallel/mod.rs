//! 并行图处理模块
//!
//! **重要说明**: 当前实现是**单机多线程并行**，不是跨机器分布式集群。
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Parallel Executor                          │
//! │              (Rayon 多线程并行执行)                           │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!              ┌───────────────┼───────────────┐
//!              ▼               ▼               ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │   Partition 1   │ │   Partition 2   │ │   Partition N   │
//! │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
//! │  │ Subgraph  │  │ │  │ Subgraph  │  │ │  │ Subgraph  │  │
//! │  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```
//!
//! # 当前能力
//!
//! - ✅ **图分区**: Hash 分区器、Range 分区器
//! - ✅ **多线程并行**: 使用 Rayon 线程池
//! - ✅ **并行算法**: PageRank、BFS、DFS、连通分量、Dijkstra
//! - ✅ **结果聚合**: 自动合并各分区结果
//!
//! # 未来计划（真正的分布式）
//!
//! - 🔲 **跨机器通信**: gRPC/消息队列
//! - 🔲 **故障恢复**: 检查点、重试机制
//! - 🔲 **数据一致性**: 分布式一致性协议
//!
//! # 使用示例
//!
//! ```rust
//! use god_graph::parallel::partitioner::HashPartitioner;
//! use god_graph::graph::Graph;
//!
//! // 创建图
//! let mut graph = Graph::<String, f64>::directed();
//! for i in 0..100 {
//!     graph.add_node(format!("node_{}", i)).unwrap();
//! }
//!
//! // 创建分区器（4 个分区）
//! let partitioner = HashPartitioner::new(4);
//! let partitions = partitioner.partition_graph(&graph);
//!
//! // 并行处理各分区（使用 Rayon）
//! use rayon::prelude::*;
//! let results: Vec<_> = partitions.par_iter()
//!     .map(|partition| partition.nodes.len())
//!     .collect();
//! ```

pub mod algorithms;
pub mod communication;
pub mod executor;
pub mod fault_tolerance;
pub mod partitioner;

// 重新导出主要类型
pub use algorithms::{
    BFSConfig, ConnectedComponentsConfig, DFSConfig, DijkstraConfig, DistributedBFS,
    DistributedConnectedComponents, DistributedDFS, DistributedDijkstra, DistributedPageRank,
    PageRankConfig,
};
pub use communication::{Channel, CommunicationConfig, Message};
pub use executor::{DistributedExecutor, ExecutorConfig, WorkerInfo};
pub use fault_tolerance::{
    execute_with_retry, CheckpointRecovery, CircuitBreaker, CircuitBreakerBuilder, CircuitState,
    DistributedLogger, FailureDetector, FaultTolerance, FaultToleranceStats, HealthChecker,
    LogEntry, LogLevel, NodeHealth, RecoveryStrategy, RetryPolicy, RetryPolicyBuilder,
};
pub use partitioner::{
    HashPartitioner, Partition, Partitioner, PartitionerConfig, RangePartitioner,
};
