//! 图核心 trait 定义
//!
//! 定义图操作的基本接口，支持泛型实现

use crate::edge::{EdgeIndex, EdgeRef};
use crate::errors::{GraphError, GraphResult};
use crate::node::{NodeIndex, NodeRef};

/// 图基础 trait
///
/// 定义所有图类型必须实现的基本操作
pub trait GraphBase {
    /// 节点数据类型
    type NodeData;
    /// 边数据类型
    type EdgeData;

    /// 获取节点数量
    fn node_count(&self) -> usize;

    /// 获取边数量
    fn edge_count(&self) -> usize;

    /// 检查是否为空图
    fn is_empty(&self) -> bool {
        self.node_count() == 0
    }

    /// 检查是否包含节点
    fn contains_node(&self, index: NodeIndex) -> bool;

    /// 检查是否包含边
    fn contains_edge(&self, index: EdgeIndex) -> bool;
}

/// 图查询 trait
///
/// 定义图的只读查询操作
pub trait GraphQuery: GraphBase {
    /// 获取节点引用
    fn get_node(&self, index: NodeIndex) -> GraphResult<&Self::NodeData>;

    /// 获取边引用
    fn get_edge(&self, index: EdgeIndex) -> GraphResult<&Self::EdgeData>;

    /// 获取边的端点
    fn edge_endpoints(&self, index: EdgeIndex) -> GraphResult<(NodeIndex, NodeIndex)>;

    /// 获取节点的出边邻居迭代器
    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex>;

    /// 获取节点的关联边迭代器
    fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex>;

    /// 检查是否存在从 from 到 to 的边
    fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool;

    /// 获取节点的出度
    fn out_degree(&self, node: NodeIndex) -> GraphResult<usize>;

    /// 获取节点的入度（有向图）
    fn in_degree(&self, node: NodeIndex) -> GraphResult<usize> {
        self.out_degree(node)
    }

    /// 获取节点的总度数
    fn degree(&self, node: NodeIndex) -> GraphResult<usize> {
        self.out_degree(node)
    }

    /// 获取所有节点迭代器
    fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, Self::NodeData>>;

    /// 获取所有边迭代器
    fn edges(&self) -> impl Iterator<Item = EdgeRef<'_, Self::EdgeData>>;
}

/// 图可变操作 trait
///
/// 定义图的可变操作
pub trait GraphOps: GraphBase {
    /// 添加节点
    fn add_node(&mut self, data: Self::NodeData) -> GraphResult<NodeIndex>;

    /// 删除节点，返回节点数据
    fn remove_node(&mut self, index: NodeIndex) -> GraphResult<Self::NodeData>;

    /// 添加边
    fn add_edge(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        data: Self::EdgeData,
    ) -> GraphResult<EdgeIndex>;

    /// 删除边，返回边数据
    fn remove_edge(&mut self, index: EdgeIndex) -> GraphResult<Self::EdgeData>;

    /// 更新节点数据，返回旧数据
    fn update_node(
        &mut self,
        index: NodeIndex,
        data: Self::NodeData,
    ) -> GraphResult<Self::NodeData>;

    /// 更新边数据，返回旧数据
    fn update_edge(
        &mut self,
        index: EdgeIndex,
        data: Self::EdgeData,
    ) -> GraphResult<Self::EdgeData>;

    /// 清空图
    fn clear(&mut self);

    /// 预分配容量
    fn reserve(&mut self, additional_nodes: usize, additional_edges: usize);
}

/// 边方向
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Direction {
    /// 出边方向
    Outgoing,
    /// 入边方向
    Incoming,
}

impl Direction {
    /// 获取相反方向
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
        }
    }

    /// 检查是否为出边方向
    #[inline]
    pub fn is_outgoing(self) -> bool {
        matches!(self, Direction::Outgoing)
    }

    /// 检查是否为入边方向
    #[inline]
    pub fn is_incoming(self) -> bool {
        matches!(self, Direction::Incoming)
    }
}

/// 索引验证辅助函数
///
/// 验证 NodeIndex 的 generation 是否匹配
#[inline]
pub fn validate_node_index(provided: NodeIndex, current_generation: u32) -> GraphResult<()> {
    if provided.generation() == current_generation {
        Ok(())
    } else {
        Err(GraphError::NodeDeleted {
            index: provided.index(),
            provided: provided.generation(),
            current: current_generation,
        })
    }
}

/// 索引验证辅助函数
///
/// 验证 EdgeIndex 的 generation 是否匹配
#[inline]
pub fn validate_edge_index(provided: EdgeIndex, current_generation: u32) -> GraphResult<()> {
    if provided.generation() == current_generation {
        Ok(())
    } else {
        Err(GraphError::EdgeDeleted {
            index: provided.index(),
            provided: provided.generation(),
            current: current_generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction() {
        assert_eq!(Direction::Outgoing.opposite(), Direction::Incoming);
        assert_eq!(Direction::Incoming.opposite(), Direction::Outgoing);
        assert!(Direction::Outgoing.is_outgoing());
        assert!(Direction::Incoming.is_incoming());
    }
}
