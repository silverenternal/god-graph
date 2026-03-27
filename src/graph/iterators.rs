//! 图迭代器模块
//!
//! 提供各种图遍历迭代器

use crate::node::NodeIndex;
use crate::edge::EdgeIndex;
pub use crate::graph::impl_::NeighborsIter;
pub use crate::graph::impl_::IncidentEdgesIter;

/// 节点迭代器
#[allow(dead_code)]
pub struct NodeIter<'a, T> {
    inner: core::slice::Iter<'a, Option<T>>,
    #[allow(dead_code)]
    current_index: usize,
}

impl<'a, T> NodeIter<'a, T> {
    #[allow(dead_code)]
    pub(crate) fn new(data: &'a [Option<T>]) -> Self {
        Self {
            inner: data.iter(),
            current_index: 0,
        }
    }
}

/// 边迭代器
#[allow(dead_code)]
pub struct EdgeIter<'a, E> {
    inner: core::slice::Iter<'a, Option<E>>,
    #[allow(dead_code)]
    current_index: usize,
}

impl<'a, E> EdgeIter<'a, E> {
    #[allow(dead_code)]
    pub(crate) fn new(data: &'a [Option<E>]) -> Self {
        Self {
            inner: data.iter(),
            current_index: 0,
        }
    }
}

/// 邻居迭代器（包装 CSR 迭代器）
pub struct Neighbors<'a> {
    indices: core::ops::Range<usize>,
    col_indices: &'a [usize],
}

impl<'a> Neighbors<'a> {
    #[allow(dead_code)]
    pub(crate) fn new(range: core::ops::Range<usize>, col_indices: &'a [usize]) -> Self {
        Self {
            indices: range,
            col_indices,
        }
    }
}

impl<'a> Iterator for Neighbors<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|idx| NodeIndex::new(self.col_indices[idx], 0))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<'a> ExactSizeIterator for Neighbors<'a> {}
impl<'a> core::iter::FusedIterator for Neighbors<'a> {}

/// 关联边迭代器
pub struct IncidentEdges<'a> {
    indices: core::ops::Range<usize>,
    edge_indices: &'a [usize],
}

impl<'a> IncidentEdges<'a> {
    #[allow(dead_code)]
    pub(crate) fn new(range: core::ops::Range<usize>, edge_indices: &'a [usize]) -> Self {
        Self {
            indices: range,
            edge_indices,
        }
    }
}

impl<'a> Iterator for IncidentEdges<'a> {
    type Item = EdgeIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|idx| EdgeIndex::new(self.edge_indices[idx], 0))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<'a> ExactSizeIterator for IncidentEdges<'a> {}
impl<'a> core::iter::FusedIterator for IncidentEdges<'a> {}
