//! 图操作错误类型定义
//!
//! 提供详细的错误类型，支持错误链和上下文信息

use core::fmt;
use core::fmt::{Display, Formatter};

/// 图操作结果类型别名
pub type GraphResult<T> = Result<T, GraphError>;

/// 图操作错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// 节点不存在：索引无效或超出范围
    NodeNotFound {
        /// 导致错误的节点索引
        index: usize,
    },

    /// 节点已删除：generation 不匹配
    NodeDeleted {
        /// 导致错误的节点索引
        index: usize,
        /// 提供的 generation
        provided: u32,
        /// 当前的 generation
        current: u32,
    },

    /// 边不存在：索引无效或超出范围
    EdgeNotFound {
        /// 导致错误的边索引
        index: usize,
    },

    /// 边已删除：generation 不匹配
    EdgeDeleted {
        /// 导致错误的边索引
        index: usize,
        /// 提供的 generation
        provided: u32,
        /// 当前的 generation
        current: u32,
    },

    /// 边已存在：简单图不允许重边
    EdgeAlreadyExists {
        /// 源节点索引
        from: usize,
        /// 目标节点索引
        to: usize,
    },

    /// 不允许自环
    SelfLoopNotAllowed {
        /// 节点索引
        node: usize,
    },

    /// 超过最大节点容量
    GraphCapacityExceeded {
        /// 当前数量
        current: usize,
        /// 最大容量
        max: usize,
    },

    /// 内存分配失败
    MemoryAllocationFailed {
        /// 请求的大小（字节）
        requested: usize,
    },

    /// 无效的图方向操作
    InvalidDirection,

    /// 图不是有向图
    GraphNotDirected,

    /// 图不是无向图
    GraphNotUndirected,

    /// 图中存在环（用于 DAG 操作）
    GraphHasCycle,

    /// 算法不收敛
    AlgorithmNotConverged {
        /// 执行的迭代次数
        iterations: usize,
        /// 最终误差
        error: f64,
    },

    /// 负权环检测到（Bellman-Ford）
    NegativeCycle,

    /// 负权重边检测到（Dijkstra 不支持）
    NegativeWeight {
        /// 源节点索引
        from: usize,
        /// 目标节点索引
        to: usize,
        /// 权重值
        weight: f64,
    },

    /// 索引越界
    IndexOutOfBounds {
        /// 提供的索引
        index: usize,
        /// 边界
        bound: usize,
    },
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NodeNotFound { index } => {
                write!(f, "节点不存在：索引 {} 无效或超出范围", index)
            }
            GraphError::NodeDeleted {
                index,
                provided,
                current,
            } => {
                write!(
                    f,
                    "节点已删除：索引 {} 的 generation 不匹配 (provided: {}, current: {})",
                    index, provided, current
                )
            }
            GraphError::EdgeNotFound { index } => {
                write!(f, "边不存在：索引 {} 无效或超出范围", index)
            }
            GraphError::EdgeDeleted {
                index,
                provided,
                current,
            } => {
                write!(
                    f,
                    "边已删除：索引 {} 的 generation 不匹配 (provided: {}, current: {})",
                    index, provided, current
                )
            }
            GraphError::EdgeAlreadyExists { from, to } => {
                write!(f, "边已存在：从节点 {} 到节点 {} 的边已存在", from, to)
            }
            GraphError::SelfLoopNotAllowed { node } => {
                write!(f, "不允许自环：节点 {} 不能作为自身的边端点", node)
            }
            GraphError::GraphCapacityExceeded { current, max } => {
                write!(f, "超过最大容量：当前 {}，最大 {}", current, max)
            }
            GraphError::MemoryAllocationFailed { requested } => {
                write!(f, "内存分配失败：请求 {} 字节", requested)
            }
            GraphError::InvalidDirection => {
                write!(f, "无效的图方向操作")
            }
            GraphError::GraphNotDirected => {
                write!(f, "图不是有向图")
            }
            GraphError::GraphNotUndirected => {
                write!(f, "图不是无向图")
            }
            GraphError::GraphHasCycle => {
                write!(f, "图中存在环")
            }
            GraphError::AlgorithmNotConverged { iterations, error } => {
                write!(
                    f,
                    "算法不收敛：执行 {} 次迭代后误差为 {}",
                    iterations, error
                )
            }
            GraphError::NegativeCycle => {
                write!(f, "检测到负权环")
            }
            GraphError::NegativeWeight { from, to, weight } => {
                write!(
                    f,
                    "检测到负权重边：节点 {} 到 {} 的权重为 {}（Dijkstra 不支持负权重，建议使用 Bellman-Ford 算法）",
                    from, to, weight
                )
            }
            GraphError::IndexOutOfBounds { index, bound } => {
                write!(f, "索引越界：索引 {} 超出边界 {}", index, bound)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GraphError {}

impl From<GraphError> for String {
    fn from(err: GraphError) -> Self {
        err.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GraphError::NodeNotFound { index: 42 };
        assert!(err.to_string().contains("42"));

        let err = GraphError::EdgeAlreadyExists { from: 1, to: 2 };
        assert!(err.to_string().contains("1"));
        assert!(err.to_string().contains("2"));
    }
}
