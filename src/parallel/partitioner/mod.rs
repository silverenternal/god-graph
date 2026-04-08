//! 图分区器模块
//!
//! 提供图分区功能，将大图分割成多个小分区，用于分布式处理
//!
//! # 可用分区器
//!
//! - **HashPartitioner**: 基于哈希的简单分区
//! - **RangePartitioner**: 基于范围的分区
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::partitioner::{HashPartitioner, Partitioner};
//! use god_graph::node::NodeIndex;
//!
//! let partitioner = HashPartitioner::new(4);
//! let partition = partitioner.partition_node(NodeIndex::new_public(5));
//! assert!(partition < 4);
//! ```

pub mod hash;
pub mod range;
pub mod traits;

pub use hash::HashPartitioner;
pub use range::RangePartitioner;
pub use traits::{Partition, Partitioner, PartitionerConfig};
