//! 边索引和边引用模块
//!
//! 提供稳定的边索引机制，使用 generation 计数防止 ABA 问题

use core::fmt;
use core::hash::{Hash, Hasher};

use crate::node::NodeIndex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 边索引：稳定引用边的句柄
///
/// 包含槽位索引和 generation 计数，确保删除后不会错误访问新边
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeIndex {
    /// 在边存储中的索引
    pub(crate) index: usize,
    /// Generation 计数器，防止 ABA 问题
    pub(crate) generation: u32,
}

impl EdgeIndex {
    /// 创建新的 EdgeIndex（内部使用）
    #[inline]
    pub(crate) fn new(index: usize, generation: u32) -> Self {
        Self { index, generation }
    }

    /// 获取索引
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// 获取 generation 计数
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// 检查索引是否有效
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn is_valid(&self) -> bool {
        self.index != usize::MAX
    }

    /// 创建无效索引
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn invalid() -> Self {
        Self {
            index: usize::MAX,
            generation: 0,
        }
    }
}

impl fmt::Debug for EdgeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeIndex({}, {})", self.index, self.generation)
    }
}

impl Hash for EdgeIndex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

/// 边引用：对图中边的借用引用
///
/// 包含边索引、端点和数据引用，生命周期与图绑定
pub struct EdgeRef<'a, E> {
    /// 边索引
    pub index: EdgeIndex,
    /// 源节点索引
    pub source: NodeIndex,
    /// 目标节点索引
    pub target: NodeIndex,
    /// 边数据引用
    pub data: &'a E,
}

impl<'a, E> EdgeRef<'a, E> {
    /// 创建新的 EdgeRef
    #[inline]
    pub fn new(index: EdgeIndex, source: NodeIndex, target: NodeIndex, data: &'a E) -> Self {
        Self {
            index,
            source,
            target,
            data,
        }
    }

    /// 获取边索引
    #[inline]
    pub fn index(&self) -> EdgeIndex {
        self.index
    }

    /// 获取源节点索引
    #[inline]
    pub fn source(&self) -> NodeIndex {
        self.source
    }

    /// 获取目标节点索引
    #[inline]
    pub fn target(&self) -> NodeIndex {
        self.target
    }

    /// 获取边数据引用
    #[inline]
    pub fn data(&self) -> &'a E {
        self.data
    }

    /// 获取端点对
    #[inline]
    pub fn endpoints(&self) -> (NodeIndex, NodeIndex) {
        (self.source, self.target)
    }
}

impl<'a, E> Clone for EdgeRef<'a, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, E> Copy for EdgeRef<'a, E> {}

impl<'a, E> fmt::Debug for EdgeRef<'a, E>
where
    E: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EdgeRef")
            .field("index", &self.index)
            .field("source", &self.source)
            .field("target", &self.target)
            .field("data", self.data)
            .finish()
    }
}

/// 边存储结构
///
/// 包含源节点、目标节点和边数据
/// 使用 64 字节对齐，避免 false sharing
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(serialize = "E: Serialize", deserialize = "E: Deserialize<'de>"))
)]
#[repr(align(64))]
pub(crate) struct EdgeStorage<E> {
    /// 源节点索引
    pub source: NodeIndex,
    /// 目标节点索引
    pub target: NodeIndex,
    /// 边数据
    pub data: Option<E>,
    /// Generation 计数器
    pub generation: u32,
}

impl<E> Clone for EdgeStorage<E> {
    fn clone(&self) -> Self {
        Self {
            source: self.source,
            target: self.target,
            data: None, // 不克隆数据
            generation: self.generation,
        }
    }
}

impl<E> EdgeStorage<E> {
    /// 创建新的边存储
    #[inline]
    pub(crate) fn new(source: NodeIndex, target: NodeIndex, data: E, generation: u32) -> Self {
        Self {
            source,
            target,
            data: Some(data),
            generation,
        }
    }

    /// 创建已删除的边存储
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn deleted(source: NodeIndex, target: NodeIndex, generation: u32) -> Self {
        Self {
            source,
            target,
            data: None,
            generation,
        }
    }

    /// 检查边是否被占用
    #[inline]
    pub(crate) fn is_occupied(&self) -> bool {
        self.data.is_some()
    }

    /// 获取数据引用
    #[inline]
    pub(crate) fn data(&self) -> Option<&E> {
        self.data.as_ref()
    }

    /// 获取数据可变引用
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn data_mut(&mut self) -> Option<&mut E> {
        self.data.as_mut()
    }

    /// 获取端点对
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn endpoints(&self) -> (NodeIndex, NodeIndex) {
        (self.source, self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_index_creation() {
        let idx = EdgeIndex::new(42, 1);
        assert_eq!(idx.index(), 42);
        assert_eq!(idx.generation(), 1);
        assert!(idx.is_valid());
    }

    #[test]
    fn test_edge_index_invalid() {
        let idx = EdgeIndex::invalid();
        assert!(!idx.is_valid());
        assert_eq!(idx.index(), usize::MAX);
    }

    #[test]
    fn test_edge_ref() {
        let source = NodeIndex::new(0, 1);
        let target = NodeIndex::new(1, 1);
        let edge_idx = EdgeIndex::new(0, 1);

        let edge_ref = EdgeRef::new(edge_idx, source, target, &42.0);

        assert_eq!(edge_ref.index(), edge_idx);
        assert_eq!(edge_ref.source(), source);
        assert_eq!(edge_ref.target(), target);
        assert_eq!(*edge_ref.data(), 42.0);
        assert_eq!(edge_ref.endpoints(), (source, target));
    }

    #[test]
    fn test_edge_storage() {
        let source = NodeIndex::new(0, 1);
        let target = NodeIndex::new(1, 1);

        let edge = EdgeStorage::new(source, target, 42.0, 1);
        assert!(edge.is_occupied());
        assert_eq!(edge.data(), Some(&42.0));

        let deleted: EdgeStorage<f64> = EdgeStorage::deleted(source, target, 1);
        assert!(!deleted.is_occupied());
        assert_eq!(deleted.data(), None);
    }
}
