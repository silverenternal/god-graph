//! Graph 核心实现
//!
//! 使用桶式邻接表（Bucket Adjacency List）和 Arena 分配器实现高性能图数据结构
//!
//! ## 内存布局说明
//!
//! 本实现采用**桶式邻接表**（Bucket Adjacency List）格式，而非传统 CSR（压缩稀疏行）：
//!
//! | 特性 | 传统 CSR | 本实现（桶式邻接表） |
//! |------|----------|---------------------|
//! | 结构 | row_offsets + col_indices | Vec<AdjBucket>，每个 bucket 含独立 Vec |
//! | 增量更新 | ❌ 不支持（需 O(V+E) 重建） | ✅ O(1) 插入 |
//! | 空间效率 | ✅ 最优（无额外开销） | ⚠️ 略高（每节点一个 Vec 头） |
//! | 适用场景 | 静态图 | 动态图（频繁增删边） |
//!
//! 历史原因：早期代码使用 `CsrStorage` 命名，实际实现是桶式邻接表。
//! 为保持 API 稳定性，内部结构仍用此名，但文档已正名为**桶式邻接表**。

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

use crate::edge::{EdgeIndex, EdgeRef, EdgeStorage};
use crate::errors::{GraphError, GraphResult};
use crate::node::{NodeIndex, NodeRef, NodeSlot};
use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};

/// 单个邻接桶，用于 O(1) 增量边插入
/// 使用 64 字节缓存行对齐，避免 false sharing
#[derive(Clone, Debug)]
#[repr(align(64))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct AdjBucket {
    neighbors: Vec<usize>,
    edge_indices: Vec<usize>,
    deleted_mask: Vec<u64>,
    deleted_count: usize,
    _padding: [u8; 8], // 确保 64 字节对齐
}

impl AdjBucket {
    fn new() -> Self {
        Self {
            neighbors: Vec::new(),
            edge_indices: Vec::new(),
            deleted_mask: Vec::new(),
            deleted_count: 0,
            _padding: [0; 8],
        }
    }

    #[inline]
    fn push(&mut self, target: usize, edge_idx: usize) {
        self.neighbors.push(target);
        self.edge_indices.push(edge_idx);
        let bit_idx = self.neighbors.len() - 1;
        let word_idx = bit_idx / 64;
        while self.deleted_mask.len() <= word_idx {
            self.deleted_mask.push(0);
        }
    }

    #[inline]
    fn mark_deleted(&mut self, pos: usize) {
        let word_idx = pos / 64;
        let bit = pos % 64;
        if word_idx < self.deleted_mask.len() {
            let mask = 1u64 << bit;
            if self.deleted_mask[word_idx] & mask == 0 {
                self.deleted_mask[word_idx] |= mask;
                self.deleted_count += 1;
            }
        }
    }

    #[inline]
    fn is_deleted(&self, pos: usize) -> bool {
        let word_idx = pos / 64;
        let bit = pos % 64;
        if word_idx >= self.deleted_mask.len() {
            return false;
        }
        self.deleted_mask[word_idx] & (1u64 << bit) != 0
    }

    #[inline]
    fn compact(&mut self) {
        if self.deleted_count == 0 {
            return;
        }
        let mut write_pos = 0;
        for read_pos in 0..self.neighbors.len() {
            if !self.is_deleted(read_pos) {
                if write_pos != read_pos {
                    self.neighbors[write_pos] = self.neighbors[read_pos];
                    self.edge_indices[write_pos] = self.edge_indices[read_pos];
                }
                write_pos += 1;
            }
        }
        self.neighbors.truncate(write_pos);
        self.edge_indices.truncate(write_pos);
        let words_needed = write_pos.div_ceil(64);
        self.deleted_mask.truncate(words_needed);
        for w in self.deleted_mask.iter_mut() {
            *w = 0;
        }
        self.deleted_count = 0;
    }

    #[inline]
    fn find(&self, target: usize) -> Option<usize> {
        self.neighbors
            .iter()
            .position(|&n| n == target)
            .filter(|&pos| !self.is_deleted(pos))
    }

    #[inline]
    fn len(&self) -> usize {
        self.neighbors.len() - self.deleted_count
    }

    #[inline]
    fn iter(&self) -> AdjBucketIter<'_> {
        AdjBucketIter {
            bucket: self,
            pos: 0,
        }
    }

    /// 获取有效邻居的快照（用于快照语义迭代器）
    #[inline]
    fn snapshot(&self) -> Vec<(usize, usize)> {
        self.iter().collect()
    }
}

/// 邻接桶迭代器
pub(crate) struct AdjBucketIter<'a> {
    bucket: &'a AdjBucket,
    pos: usize,
}

impl<'a> Iterator for AdjBucketIter<'a> {
    type Item = (usize, usize); // (target, edge_index)

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.bucket.neighbors.len() {
            // 软件预取：预取下一批数据到 CPU 缓存（需要 nightly 特性）
            // 在稳定版 Rust 上使用 std::hint 或条件编译
            #[cfg(all(feature = "unstable", target_feature = "sse"))]
            {
                use core::arch::x86_64::_mm_prefetch;
                use core::arch::x86_64::_MM_HINT_T0;

                let prefetch_pos = self.pos + 4;
                if prefetch_pos < self.bucket.neighbors.len() {
                    unsafe {
                        _mm_prefetch(
                            self.bucket.neighbors.as_ptr().add(prefetch_pos) as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            self.bucket.edge_indices.as_ptr().add(prefetch_pos) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
            }

            let pos = self.pos;
            self.pos += 1;
            if !self.bucket.is_deleted(pos) {
                return Some((self.bucket.neighbors[pos], self.bucket.edge_indices[pos]));
            }
        }
        None
    }
}

/// 邻居迭代器（快照语义）
///
/// 在创建时捕获目标节点的 generation，迭代期间只返回该快照中的邻居
pub struct NeighborsIter {
    /// 原始邻居数据快照（target_idx, generation）
    snapshot: std::vec::IntoIter<(usize, u32)>,
}

impl NeighborsIter {
    pub(crate) fn new(snapshot: Vec<(usize, u32)>) -> Self {
        Self {
            snapshot: snapshot.into_iter(),
        }
    }
}

impl Iterator for NeighborsIter {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.snapshot.next().map(|(idx, generation)| NodeIndex::new(idx, generation))
    }
}

/// 关联边迭代器（快照语义）
///
/// 在创建时捕获边的 generation，迭代期间只返回该快照中的边
pub struct IncidentEdgesIter {
    /// 原始边数据快照（edge_idx, generation）
    snapshot: std::vec::IntoIter<(usize, u32)>,
}

impl IncidentEdgesIter {
    pub(crate) fn new(snapshot: Vec<(usize, u32)>) -> Self {
        Self {
            snapshot: snapshot.into_iter(),
        }
    }
}

impl Iterator for IncidentEdgesIter {
    type Item = EdgeIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.snapshot.next().map(|(idx, generation)| EdgeIndex::new(idx, generation))
    }
}

/// 邻接桶存储：桶式邻接表格式
///
/// ## 命名说明
///
/// 历史原因：早期代码使用 `CsrStorage` 命名，但实际实现是**桶式邻接表**（Bucket Adjacency List）。
/// 传统 CSR（压缩稀疏行）格式使用 `row_offsets + col_indices`，不支持 O(1) 增量更新。
/// 本实现使用 `Vec<AdjBucket>`，每个 bucket 含独立 `Vec`，支持 O(1) 插入和惰性删除。
///
/// 为保持 API 稳定性，保留 `CsrStorage` 别名，但推荐在新代码中使用 `AdjacencyBuckets`。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct AdjacencyBuckets {
    buckets: Vec<AdjBucket>,
    reverse_buckets: Vec<AdjBucket>,
    num_nodes: usize,
    needs_compact: bool,
}

/// 向后兼容别名
///
/// 历史原因：早期代码使用 `CsrStorage` 命名，但实际实现是桶式邻接表。
/// 为保持 API 稳定性保留此别名，但推荐在新代码中使用 `AdjacencyBuckets`。
#[deprecated(since = "0.3.1", note = "使用 AdjacencyBuckets 代替，命名更准确反映实现")]
#[allow(dead_code)]
pub(crate) type CsrStorage = AdjacencyBuckets;

impl AdjacencyBuckets {
    pub(crate) fn new() -> Self {
        let this = Self {
            buckets: Vec::new(),
            reverse_buckets: Vec::new(),
            num_nodes: 0,
            needs_compact: false,
        };
        // 验证空 Vec 的对齐（空指针 0 % 64 = 0，总是成立，但保留断言用于文档说明）
        debug_assert_eq!(this.buckets.as_ptr() as usize % 64, 0, "buckets 初始对齐失败");
        debug_assert_eq!(this.reverse_buckets.as_ptr() as usize % 64, 0, "reverse_buckets 初始对齐失败");
        this
    }

    /// 验证 buckets 的 64 字节对齐
    ///
    /// 使用运行时断言确保 Vec<AdjBucket> 的指针是 64 字节对齐的。
    /// AdjBucket 使用 #[repr(align(64))]，但需要验证实际分配的对齐。
    ///
    /// ## Panics
    ///
    /// 如果对齐不正确会 panic，并输出详细的调试信息。
    #[inline]
    pub(crate) fn assert_aligned(&self) {
        let buckets_ptr = self.buckets.as_ptr() as usize;
        let reverse_ptr = self.reverse_buckets.as_ptr() as usize;
        assert_eq!(
            buckets_ptr % 64, 0,
            "AdjacencyBuckets: buckets 未 64 字节对齐 (ptr={:#x}, mod={})",
            buckets_ptr, buckets_ptr % 64
        );
        assert_eq!(
            reverse_ptr % 64, 0,
            "AdjacencyBuckets: reverse_buckets 未 64 字节对齐 (ptr={:#x}, mod={})",
            reverse_ptr, reverse_ptr % 64
        );
    }

    pub(crate) fn reserve(&mut self, nodes: usize, _edges: usize) {
        self.buckets.reserve(nodes);
        self.reverse_buckets.reserve(nodes);
        // 验证分配后的对齐
        self.assert_aligned();
    }

    fn ensure_capacity(&mut self, node_index: usize) {
        while self.buckets.len() <= node_index {
            self.buckets.push(AdjBucket::new());
        }
        while self.reverse_buckets.len() <= node_index {
            self.reverse_buckets.push(AdjBucket::new());
        }
        self.num_nodes = self.buckets.len();
        // 验证扩容后的对齐
        self.assert_aligned();
    }

    /// O(1) 添加边
    pub(crate) fn add_edge(&mut self, node_index: usize, target_index: usize, edge_index: usize) {
        self.ensure_capacity(node_index.max(target_index));
        self.buckets[node_index].push(target_index, edge_index);
        self.reverse_buckets[target_index].push(node_index, edge_index);
    }

    /// 标记边为已删除
    pub(crate) fn mark_edge_deleted(&mut self, node_index: usize, target_index: usize) {
        if let Some(pos) = self.buckets[node_index].find(target_index) {
            self.buckets[node_index].mark_deleted(pos);
            self.needs_compact = true;
        }
        if let Some(pos) = self.reverse_buckets[target_index].find(node_index) {
            self.reverse_buckets[target_index].mark_deleted(pos);
        }
    }

    pub(crate) fn compact(&mut self) {
        if !self.needs_compact {
            return;
        }
        for bucket in &mut self.buckets {
            bucket.compact();
        }
        for bucket in &mut self.reverse_buckets {
            bucket.compact();
        }
        self.needs_compact = false;
    }

    #[allow(dead_code)]
    pub(crate) fn neighbors_raw(&self, node_index: usize) -> impl Iterator<Item = usize> + '_ {
        self.buckets
            .get(node_index)
            .into_iter()
            .flat_map(|b| b.iter())
            .map(move |(target_idx, _)| target_idx)
    }

    #[allow(dead_code)]
    pub(crate) fn reverse_neighbors_raw(&self, node_index: usize) -> impl Iterator<Item = usize> + '_ {
        self.reverse_buckets
            .get(node_index)
            .into_iter()
            .flat_map(|b| b.iter())
            .map(move |(src_idx, _)| src_idx)
    }

    pub(crate) fn edge_indices_iter(
        &self,
        node_index: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.buckets
            .get(node_index)
            .into_iter()
            .flat_map(|b| b.iter().map(|(_, ei)| ei))
    }

    /// 获取边索引的快照（用于快照语义迭代器）
    pub(crate) fn edge_indices_snapshot(&self, node_index: usize) -> Vec<usize> {
        self.edge_indices_iter(node_index).collect()
    }

    /// 获取邻居的快照（用于快照语义迭代器）
    pub(crate) fn neighbors_snapshot(&self, node_index: usize) -> Vec<(usize, usize)> {
        self.buckets
            .get(node_index)
            .map(|b| b.snapshot())
            .unwrap_or_default()
    }

    pub(crate) fn has_edge(&self, node_index: usize, target_index: usize) -> bool {
        self.buckets
            .get(node_index)
            .and_then(|b| b.find(target_index))
            .is_some()
    }

    pub(crate) fn degree(&self, node_index: usize) -> usize {
        self.buckets.get(node_index).map(|b| b.len()).unwrap_or(0)
    }

    pub(crate) fn in_degree(&self, node_index: usize) -> usize {
        self.reverse_buckets
            .get(node_index)
            .map(|b| b.len())
            .unwrap_or(0)
    }

    pub(crate) fn reverse_edge_indices_iter(
        &self,
        node_index: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.reverse_buckets
            .get(node_index)
            .into_iter()
            .flat_map(|b| b.iter().map(|(_, ei)| ei))
    }

    pub(crate) fn clear(&mut self) {
        self.buckets.clear();
        self.reverse_buckets.clear();
        self.num_nodes = 0;
        self.needs_compact = false;
    }

    #[allow(dead_code)]
    pub(crate) fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl Default for AdjacencyBuckets {
    fn default() -> Self {
        Self::new()
    }
}

/// 图结构主实现
/// 使用缓存行对齐优化，减少 false sharing
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(
    serialize = "T: Serialize, E: Serialize",
    deserialize = "T: Deserialize<'de>, E: Deserialize<'de>"
)))]
pub struct Graph<T, E> {
    nodes: Vec<NodeSlot<T>>,
    edges: Vec<EdgeStorage<E>>,
    csr: AdjacencyBuckets,
    node_count: usize,
    edge_count: usize,
    free_nodes: Vec<usize>,
    free_edges: Vec<usize>,
}

impl<T, E> fmt::Debug for Graph<T, E>
where
    T: fmt::Debug,
    E: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("node_count", &self.node_count)
            .field("edge_count", &self.edge_count)
            .finish()
    }
}

impl<T, E> Graph<T, E> {
    /// 创建新的有向图
    pub fn directed() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            csr: AdjacencyBuckets::new(),
            node_count: 0,
            edge_count: 0,
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
        }
    }

    /// 创建新的无向图
    pub fn undirected() -> Self {
        Self::directed()
    }

    /// 创建带预分配容量的有向图
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        let mut graph = Self::directed();
        graph.reserve(nodes, edges);
        graph
    }

    /// 检查图是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// 清空图
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.csr.clear();
        self.node_count = 0;
        self.edge_count = 0;
        self.free_nodes.clear();
        self.free_edges.clear();
    }

    /// 通过节点索引获取边数据
    pub fn get_edge_by_nodes(&self, from: NodeIndex, to: NodeIndex) -> GraphResult<&E> {
        for edge_idx in self.csr.edge_indices_iter(from.index()) {
            if edge_idx < self.edges.len() {
                let edge = &self.edges[edge_idx];
                if edge.is_occupied() && edge.target == to {
                    return edge.data().ok_or(GraphError::EdgeNotFound { index: edge_idx });
                }
            }
        }
        Err(GraphError::EdgeNotFound { index: usize::MAX })
    }
}

impl<T, E> GraphBase for Graph<T, E> {
    type NodeData = T;
    type EdgeData = E;

    #[inline]
    fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edge_count
    }

    #[inline]
    fn contains_node(&self, index: NodeIndex) -> bool {
        if index.index() >= self.nodes.len() {
            return false;
        }
        let slot = &self.nodes[index.index()];
        slot.is_occupied() && slot.generation == index.generation()
    }

    #[inline]
    fn contains_edge(&self, index: EdgeIndex) -> bool {
        if index.index() >= self.edges.len() {
            return false;
        }
        let edge = &self.edges[index.index()];
        edge.is_occupied() && edge.generation == index.generation()
    }
}

impl<T, E> GraphQuery for Graph<T, E> {
    #[inline]
    fn get_node(&self, index: NodeIndex) -> GraphResult<&T> {
        if index.index() >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }
        let slot = &self.nodes[index.index()];
        if !slot.is_occupied() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }
        if slot.generation != index.generation() {
            return Err(GraphError::NodeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: slot.generation,
            });
        }
        slot.data().ok_or(GraphError::NodeNotFound { index: index.index() })
    }

    #[inline]
    fn get_edge(&self, index: EdgeIndex) -> GraphResult<&E> {
        if index.index() >= self.edges.len() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        let edge = &self.edges[index.index()];
        if !edge.is_occupied() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        if edge.generation != index.generation() {
            return Err(GraphError::EdgeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: edge.generation,
            });
        }
        edge.data().ok_or(GraphError::EdgeNotFound { index: index.index() })
    }

    #[inline]
    fn edge_endpoints(&self, index: EdgeIndex) -> GraphResult<(NodeIndex, NodeIndex)> {
        if index.index() >= self.edges.len() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        let edge = &self.edges[index.index()];
        if !edge.is_occupied() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        if edge.generation != index.generation() {
            return Err(GraphError::EdgeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: edge.generation,
            });
        }
        Ok((edge.source, edge.target))
    }

    /// 获取节点的邻居迭代器
    ///
    /// # 快照语义
    ///
    /// 迭代器在**创建时**捕获目标节点的邻居快照：
    /// - 返回的迭代器包含创建时的所有有效邻居
    /// - 迭代期间图的修改不会影响迭代器返回的内容
    /// - 如果邻居节点在迭代期间被删除，该节点会被跳过
    ///
    /// # 推荐用法
    ///
    /// ```
    /// # use god_gragh::prelude::*;
    /// # let mut graph = GraphBuilder::directed()
    /// #     .with_nodes(vec![1, 2, 3])
    /// #     .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0)])
    /// #     .build().unwrap();
    /// # let node = graph.nodes().next().unwrap().index();
    /// // 安全：迭代器持有快照，迭代期间可以修改图
    /// for neighbor in graph.neighbors(node) {
    ///     println!("{:?}", graph[neighbor]);
    /// }
    /// ```
    #[inline]
    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        let node_index = node.index();
        // 在创建时捕获邻居快照（包含 generation 信息）
        let snapshot: Vec<(usize, u32)> = self.csr
            .neighbors_snapshot(node_index)
            .into_iter()
            .map(|(target_idx, _edge_idx)| {
                let generation = self.nodes.get(target_idx).map(|n| n.generation).unwrap_or(0);
                (target_idx, generation)
            })
            .collect();
        NeighborsIter::new(snapshot)
    }

    /// 获取节点的关联边迭代器
    ///
    /// # 快照语义
    ///
    /// 迭代器在**创建时**捕获关联边的快照：
    /// - 返回的迭代器包含创建时的所有有效边
    /// - 迭代期间图的修改不会影响迭代器返回的内容
    /// - 如果边在迭代期间被删除，该边会被跳过
    ///
    /// # 推荐用法
    ///
    /// ```
    /// # use god_gragh::prelude::*;
    /// # let mut graph = GraphBuilder::directed()
    /// #     .with_nodes(vec![1, 2, 3])
    /// #     .with_edges(vec![(0, 1, 1.0), (0, 2, 1.0)])
    /// #     .build().unwrap();
    /// # let node = graph.nodes().next().unwrap().index();
    /// // 安全：迭代器持有快照，迭代期间可以修改图
    /// for edge_idx in graph.incident_edges(node) {
    ///     println!("{:?}", graph.get_edge(edge_idx).unwrap());
    /// }
    /// ```
    #[inline]
    fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex> {
        let node_index = node.index();
        // 在创建时捕获边索引快照（包含 generation 信息）
        let snapshot: Vec<(usize, u32)> = self.csr
            .edge_indices_snapshot(node_index)
            .into_iter()
            .map(|edge_idx| {
                let generation = self.edges.get(edge_idx).map(|e| e.generation).unwrap_or(0);
                (edge_idx, generation)
            })
            .collect();
        IncidentEdgesIter::new(snapshot)
    }

    #[inline]
    fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool {
        self.csr.has_edge(from.index(), to.index())
    }

    #[inline]
    fn out_degree(&self, node: NodeIndex) -> GraphResult<usize> {
        if !self.contains_node(node) {
            return Err(GraphError::NodeNotFound { index: node.index() });
        }
        Ok(self.csr.degree(node.index()))
    }

    #[inline]
    fn in_degree(&self, node: NodeIndex) -> GraphResult<usize> {
        if !self.contains_node(node) {
            return Err(GraphError::NodeNotFound { index: node.index() });
        }
        Ok(self.csr.in_degree(node.index()))
    }

    #[inline]
    fn degree(&self, node: NodeIndex) -> GraphResult<usize> {
        let out_deg = self.out_degree(node)?;
        let in_deg = self.in_degree(node)?;
        Ok(out_deg + in_deg)
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, T>> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if slot.is_occupied() {
                    Some(NodeRef::new(
                        NodeIndex::new(idx, slot.generation),
                        slot.data()?,
                    ))
                } else {
                    None
                }
            })
    }

    #[inline]
    fn edges(&self) -> impl Iterator<Item = EdgeRef<'_, E>> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(idx, edge)| {
                if edge.is_occupied() {
                    Some(EdgeRef::new(
                        EdgeIndex::new(idx, edge.generation),
                        edge.source,
                        edge.target,
                        edge.data()?,
                    ))
                } else {
                    None
                }
            })
    }
}

impl<T, E> GraphOps for Graph<T, E> {
    #[inline]
    fn add_node(&mut self, data: T) -> GraphResult<NodeIndex> {
        let (index, generation) = if let Some(free_idx) = self.free_nodes.pop() {
            let slot = &mut self.nodes[free_idx];
            let new_generation = slot.generation.wrapping_add(1);
            *slot = NodeSlot::new(new_generation, data);
            (free_idx, new_generation)
        } else {
            let index = self.nodes.len();
            let generation = 1u32;
            self.nodes.push(NodeSlot::new(generation, data));
            (index, generation)
        };

        self.node_count += 1;
        self.csr.ensure_capacity(index);
        Ok(NodeIndex::new(index, generation))
    }

    #[inline]
    fn remove_node(&mut self, index: NodeIndex) -> GraphResult<T> {
        if index.index() >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }

        let slot = &mut self.nodes[index.index()];
        if !slot.is_occupied() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }
        if slot.generation != index.generation() {
            return Err(GraphError::NodeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: slot.generation,
            });
        }

        // 删除出边
        let edge_indices: Vec<usize> = self.csr.edge_indices_iter(index.index()).collect();
        for edge_idx in edge_indices {
            if edge_idx < self.edges.len() {
                let edge = &self.edges[edge_idx];
                if edge.is_occupied() {
                    self.csr.mark_edge_deleted(index.index(), edge.target.index());
                    let edge_slot = &mut self.edges[edge_idx];
                    edge_slot.data = None;
                    self.edge_count -= 1;
                    self.free_edges.push(edge_idx);
                }
            }
        }

        // 删除入边
        let incoming: Vec<usize> = self.csr.reverse_edge_indices_iter(index.index()).collect();
        for edge_idx in incoming {
            if edge_idx < self.edges.len() {
                let edge = &self.edges[edge_idx];
                if edge.is_occupied() {
                    self.csr.mark_edge_deleted(edge.source.index(), index.index());
                    let edge_slot = &mut self.edges[edge_idx];
                    edge_slot.data = None;
                    self.edge_count -= 1;
                    self.free_edges.push(edge_idx);
                }
            }
        }

        self.csr.compact();

        let data = slot.data.take().ok_or(GraphError::NodeNotFound { index: index.index() })?;
        self.node_count -= 1;
        self.free_nodes.push(index.index());

        Ok(data)
    }

    #[inline]
    fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, data: E) -> GraphResult<EdgeIndex> {
        if !self.contains_node(from) {
            return Err(GraphError::NodeNotFound { index: from.index() });
        }
        if !self.contains_node(to) {
            return Err(GraphError::NodeNotFound { index: to.index() });
        }

        let (index, generation) = if let Some(free_idx) = self.free_edges.pop() {
            let edge = &mut self.edges[free_idx];
            let new_generation = edge.generation.wrapping_add(1);
            *edge = EdgeStorage::new(from, to, data, new_generation);
            (free_idx, new_generation)
        } else {
            let index = self.edges.len();
            let generation = 1u32;
            self.edges.push(EdgeStorage::new(from, to, data, generation));
            (index, generation)
        };

        self.edge_count += 1;
        self.csr.add_edge(from.index(), to.index(), index);
        Ok(EdgeIndex::new(index, generation))
    }

    #[inline]
    fn remove_edge(&mut self, index: EdgeIndex) -> GraphResult<E> {
        if index.index() >= self.edges.len() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }

        let edge = &mut self.edges[index.index()];
        if !edge.is_occupied() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        if edge.generation != index.generation() {
            return Err(GraphError::EdgeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: edge.generation,
            });
        }

        let source_index = edge.source.index();
        let target_index = edge.target.index();
        self.csr.mark_edge_deleted(source_index, target_index);

        let data = edge.data.take().ok_or(GraphError::EdgeNotFound { index: index.index() })?;
        self.edge_count -= 1;
        self.free_edges.push(index.index());
        self.csr.compact();

        Ok(data)
    }

    #[inline]
    fn update_node(&mut self, index: NodeIndex, data: T) -> GraphResult<T> {
        if index.index() >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }

        let slot = &mut self.nodes[index.index()];
        if !slot.is_occupied() {
            return Err(GraphError::NodeNotFound { index: index.index() });
        }
        if slot.generation != index.generation() {
            return Err(GraphError::NodeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: slot.generation,
            });
        }

        let old_data = slot.data.replace(data);
        old_data.ok_or(GraphError::NodeNotFound { index: index.index() })
    }

    #[inline]
    fn update_edge(&mut self, index: EdgeIndex, data: E) -> GraphResult<E> {
        if index.index() >= self.edges.len() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }

        let edge = &mut self.edges[index.index()];
        if !edge.is_occupied() {
            return Err(GraphError::EdgeNotFound { index: index.index() });
        }
        if edge.generation != index.generation() {
            return Err(GraphError::EdgeDeleted {
                index: index.index(),
                provided: index.generation(),
                current: edge.generation,
            });
        }

        let old_data = edge.data.replace(data);
        old_data.ok_or(GraphError::EdgeNotFound { index: index.index() })
    }

    #[inline]
    fn reserve(&mut self, additional_nodes: usize, additional_edges: usize) {
        self.nodes.reserve(additional_nodes);
        self.edges.reserve(additional_edges);
        self.csr.reserve(additional_nodes, additional_edges);
    }

    #[inline]
    fn clear(&mut self) {
        Graph::clear(self);
    }
}

impl<T, E> core::ops::Index<NodeIndex> for Graph<T, E> {
    type Output = T;

    #[inline]
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.get_node(index).expect("节点索引无效")
    }
}

/// # Safety
///
/// 此实现使用 `panic!` 而非 `Result`，因为 `IndexMut` trait 不允许返回 `Result`。
/// 对于需要安全访问的场景，请使用 `get_node_mut()` 方法。
///
/// # Panics
///
/// Panics if:
/// - 节点索引超出范围 (`index.index() >= self.nodes.len()`)
/// - 节点已被删除 (`slot.is_occupied() == false`)
/// - Generation 不匹配，索引已失效
///
/// # 示例
///
/// ```rust,should_panic
/// use god_gragh::graph::{Graph, traits::GraphOps};
///
/// let mut graph = Graph::<i32, f64>::directed();
/// let node = graph.add_node(42).unwrap();
/// graph.remove_node(node).unwrap();
///
/// // 这将 panic，因为节点已被删除
/// graph[node] = 100;
/// ```
impl<T, E> core::ops::IndexMut<NodeIndex> for Graph<T, E> {
    #[inline]
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        if index.index() >= self.nodes.len() {
            panic!("节点索引无效：index={}", index.index());
        }
        let slot = &mut self.nodes[index.index()];
        if !slot.is_occupied() {
            panic!("节点索引无效：节点不存在");
        }
        if slot.generation != index.generation() {
            panic!("节点索引已失效：generation 不匹配");
        }
        slot.data.as_mut().expect("节点数据为空")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::<i32, f64>::directed();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut graph = Graph::<String, f64>::directed();
        let idx = graph.add_node("test".to_string()).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains_node(idx));
        assert_eq!(graph[idx], "test");
    }

    #[test]
    fn test_index_mut() {
        let mut graph = Graph::<String, f64>::directed();
        let idx = graph.add_node("initial".to_string()).unwrap();
        
        // 使用 IndexMut 修改节点数据
        graph[idx] = "modified".to_string();
        assert_eq!(graph[idx], "modified");
        
        // 修改多个节点
        let idx2 = graph.add_node("second".to_string()).unwrap();
        graph[idx2] = "second_modified".to_string();
        assert_eq!(graph[idx2], "second_modified");
    }

    #[test]
    fn test_remove_node() {
        let mut graph = Graph::<i32, f64>::directed();
        let idx = graph.add_node(42).unwrap();
        assert!(graph.contains_node(idx));

        let data = graph.remove_node(idx).unwrap();
        assert_eq!(data, 42);
        assert_eq!(graph.node_count(), 0);
        assert!(!graph.contains_node(idx));
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::<i32, f64>::directed();
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let edge = graph.add_edge(a, b, std::f64::consts::PI).unwrap();

        assert_eq!(graph.edge_count(), 1);
        assert!(graph.contains_edge(edge));
        assert_eq!(*graph.get_edge(edge).unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn test_node_reuse() {
        let mut graph = Graph::<i32, f64>::directed();
        let idx1 = graph.add_node(1).unwrap();
        let _ = graph.remove_node(idx1).unwrap();
        let idx2 = graph.add_node(2).unwrap();

        assert_eq!(idx2.index(), idx1.index());
        assert!(idx2.generation() > idx1.generation());
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::<i32, f64>::directed();
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let c = graph.add_node(3).unwrap();
        graph.add_edge(a, b, 1.0).unwrap();
        graph.add_edge(a, c, 2.0).unwrap();

        let neighbors: Vec<_> = graph.neighbors(a).collect();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_has_edge() {
        let mut graph = Graph::<i32, f64>::directed();
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        graph.add_edge(a, b, 1.0).unwrap();

        assert!(graph.has_edge(a, b));
        assert!(!graph.has_edge(b, a));
    }
}
