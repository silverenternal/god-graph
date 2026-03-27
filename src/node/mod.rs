//! 节点索引和节点引用模块
//!
//! 提供稳定的节点索引机制，使用 generation 计数防止 ABA 问题

use core::fmt;
use core::hash::{Hash, Hasher};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 节点索引：稳定引用节点的句柄
///
/// 包含槽位索引和 generation 计数，确保删除后不会错误访问新节点
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeIndex {
    /// 在 Arena 中的槽位索引
    pub(crate) index: usize,
    /// Generation 计数器，防止 ABA 问题
    pub(crate) generation: u32,
}

impl NodeIndex {
    /// 创建新的 NodeIndex（内部使用）
    #[inline]
    pub(crate) fn new(index: usize, generation: u32) -> Self {
        Self { index, generation }
    }

    /// 获取槽位索引
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// 获取 generation 计数
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// 检查索引是否有效（用于内部验证）
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

impl fmt::Debug for NodeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeIndex({}, {})", self.index, self.generation)
    }
}

impl fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.index)
    }
}

impl Hash for NodeIndex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

/// 节点引用：对图中节点的借用引用
///
/// 包含节点索引和对节点数据的引用，生命周期与图绑定
pub struct NodeRef<'a, T> {
    /// 节点索引
    pub index: NodeIndex,
    /// 节点数据引用
    pub data: &'a T,
}

impl<'a, T> NodeRef<'a, T> {
    /// 创建新的 NodeRef
    #[inline]
    pub fn new(index: NodeIndex, data: &'a T) -> Self {
        Self { index, data }
    }

    /// 获取节点索引
    #[inline]
    pub fn index(&self) -> NodeIndex {
        self.index
    }

    /// 获取节点数据引用
    #[inline]
    pub fn data(&self) -> &'a T {
        self.data
    }

    /// 解引用到节点数据
    #[inline]
    pub fn get(&self) -> &'a T {
        self.data
    }
}

impl<'a, T> Clone for NodeRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for NodeRef<'a, T> {}

impl<'a, T> fmt::Debug for NodeRef<'a, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeRef")
            .field("index", &self.index)
            .field("data", self.data)
            .finish()
    }
}

/// 节点存储槽位
///
/// 包含 generation 计数和可选数据，支持安全的删除和复用
/// 使用 64 字节对齐，避免 false sharing
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))
)]
#[repr(align(64))]
pub(crate) struct NodeSlot<T> {
    /// Generation 计数器，每次分配递增
    pub generation: u32,
    /// 节点数据，None 表示已删除
    pub data: Option<T>,
}

impl<T> Clone for NodeSlot<T> {
    fn clone(&self) -> Self {
        Self {
            generation: self.generation,
            data: None, // 不克隆数据，因为 Clone 通常用于图结构克隆
        }
    }
}

impl<T> NodeSlot<T> {
    /// 创建新的节点槽位
    #[inline]
    pub(crate) fn new(generation: u32, data: T) -> Self {
        Self {
            generation,
            data: Some(data),
        }
    }

    /// 创建已删除的槽位
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn deleted(generation: u32) -> Self {
        Self {
            generation,
            data: None,
        }
    }

    /// 检查槽位是否被占用
    #[inline]
    pub(crate) fn is_occupied(&self) -> bool {
        self.data.is_some()
    }

    /// 获取数据引用
    #[inline]
    pub(crate) fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    /// 获取数据可变引用
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn data_mut(&mut self) -> Option<&mut T> {
        self.data.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_index_creation() {
        let idx = NodeIndex::new(42, 1);
        assert_eq!(idx.index(), 42);
        assert_eq!(idx.generation(), 1);
        assert!(idx.is_valid());
    }

    #[test]
    fn test_node_index_invalid() {
        let idx = NodeIndex::invalid();
        assert!(!idx.is_valid());
        assert_eq!(idx.index(), usize::MAX);
    }

    #[test]
    fn test_node_index_hash() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let idx1 = NodeIndex::new(42, 1);
        let idx2 = NodeIndex::new(42, 1);
        let idx3 = NodeIndex::new(42, 2);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        let mut h3 = DefaultHasher::new();

        idx1.hash(&mut h1);
        idx2.hash(&mut h2);
        idx3.hash(&mut h3);

        assert_eq!(h1.finish(), h2.finish());
        assert_ne!(h1.finish(), h3.finish());
    }

    #[test]
    fn test_node_slot() {
        let mut slot = NodeSlot::new(1, "test");
        assert!(slot.is_occupied());
        assert_eq!(slot.data(), Some(&"test"));
        assert_eq!(slot.data_mut(), Some(&mut "test"));

        let deleted: NodeSlot<&str> = NodeSlot::deleted(1);
        assert!(!deleted.is_occupied());
        assert_eq!(deleted.data(), None);
    }
}
