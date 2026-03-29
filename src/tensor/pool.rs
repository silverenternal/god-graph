//! Tensor 内存池：优化张量分配性能
//!
//! 通过复用已分配的内存减少分配开销，特别适用于
//! 迭代算法（如 PageRank、GNN 训练）中的临时张量

use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "tensor-pool")]
use crate::tensor::traits::{TensorBase, TensorOps};

#[cfg(feature = "tensor-pool")]
use crate::tensor::dense::DenseTensor;

#[cfg(feature = "tensor-pool")]
use smallvec::SmallVec;

/// Tensor 内存池配置
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// 初始容量
    pub initial_capacity: usize,
    /// 最大容量
    pub max_capacity: usize,
    /// 是否预分配
    pub preallocate: bool,
    /// 对齐字节数
    pub alignment: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 16,
            max_capacity: 1024,
            preallocate: false,
            alignment: 64,
        }
    }
}

impl PoolConfig {
    /// 创建新的池配置
    pub fn new(initial_capacity: usize, max_capacity: usize) -> Self {
        Self {
            initial_capacity,
            max_capacity,
            ..Default::default()
        }
    }

    /// 设置预分配
    pub fn with_preallocate(mut self, preallocate: bool) -> Self {
        self.preallocate = preallocate;
        self
    }

    /// 设置对齐
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }
}

/// Tensor 内存池
///
/// 提供高效的张量分配和回收机制
#[cfg(feature = "tensor-pool")]
pub struct TensorPool {
    /// 空闲张量列表
    free_list: Vec<DenseTensor>,
    /// 已分配的位图
    allocated: bitvec::vec::BitVec,
    /// 池配置
    config: PoolConfig,
    /// 统计信息
    stats: PoolStats,
}

/// 池统计信息
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// 总分配次数
    pub total_allocations: usize,
    /// 池命中次数
    pub pool_hits: usize,
    /// 池未命中次数
    pub pool_misses: usize,
    /// 当前已使用数量
    pub current_used: usize,
    /// 峰值使用数量
    pub peak_used: usize,
}

impl PoolStats {
    /// Compute the pool hit rate (ratio of reused allocations)
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64
        }
    }

    /// Compute the pool miss rate (ratio of new allocations)
    pub fn miss_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_misses as f64 / self.total_allocations as f64
        }
    }

    /// Compute the allocation reduction percentage
    pub fn allocation_reduction(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64 * 100.0
        }
    }
}

#[cfg(feature = "tensor-pool")]
impl TensorPool {
    /// 创建新的 tensor 池
    pub fn new(config: PoolConfig) -> Self {
        let preallocate = config.preallocate;
        let mut pool = Self {
            free_list: Vec::with_capacity(config.initial_capacity),
            allocated: bitvec::vec::BitVec::new(),
            config,
            stats: PoolStats::default(),
        };

        if preallocate {
            pool.preallocate();
        }

        pool
    }

    /// 预分配池容量
    pub fn preallocate(&mut self) {
        for _ in 0..self.config.initial_capacity {
            self.free_list.push(DenseTensor::zeros(vec![1]));
        }
    }

    /// 从池中获取张量
    pub fn acquire(&mut self, shape: Vec<usize>) -> PooledTensor<'_> {
        self.stats.total_allocations += 1;

        // 尝试从空闲列表复用
        if let Some(mut tensor) = self.free_list.pop() {
            // 重塑为所需形状
            if tensor.numel() >= shape.iter().product::<usize>() {
                tensor = tensor.reshape(&shape);
                self.stats.pool_hits += 1;
            } else {
                // 容量不足，重新分配
                self.stats.pool_misses += 1;
                tensor = DenseTensor::zeros(shape);
            }

            self.stats.current_used += 1;
            if self.stats.current_used > self.stats.peak_used {
                self.stats.peak_used = self.stats.current_used;
            }

            PooledTensor::new(tensor, self)
        } else {
            // 池为空，直接分配
            self.stats.pool_misses += 1;
            self.stats.current_used += 1;
            if self.stats.current_used > self.stats.peak_used {
                self.stats.peak_used = self.stats.current_used;
            }

            PooledTensor::new(DenseTensor::zeros(shape), self)
        }
    }

    /// 回收张量到池中
    fn recycle(&mut self, mut tensor: DenseTensor) {
        if self.free_list.len() < self.config.max_capacity {
            // 清零数据
            for val in tensor.data_mut() {
                *val = 0.0;
            }
            self.free_list.push(tensor);
        }
        // 否则让 tensor 自然 Drop

        self.stats.current_used = self.stats.current_used.saturating_sub(1);
    }

    /// 获取统计信息
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// 清空池
    pub fn clear(&mut self) {
        self.free_list.clear();
        self.allocated.clear();
        self.stats = PoolStats::default();
    }

    /// 获取池使用率
    pub fn utilization(&self) -> f64 {
        if self.config.max_capacity == 0 {
            0.0
        } else {
            self.free_list.len() as f64 / self.config.max_capacity as f64
        }
    }
}

#[cfg(feature = "tensor-pool")]
impl fmt::Debug for TensorPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorPool")
            .field("free_count", &self.free_list.len())
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

/// 池化张量：带有自动回收功能的张量包装器
#[cfg(feature = "tensor-pool")]
pub struct PooledTensor<'pool> {
    /// 内部张量
    tensor: DenseTensor,
    /// 指向父池的引用
    ///
    /// # Safety
    ///
    /// 此原始指针对 `PooledTensor` 拥有可变借用，但不拥有所有权。
    /// 池的生命周期 `'pool` 必须长于 `PooledTensor`，确保指针不会悬垂。
    /// 池本身必须是线程安全的（`TensorPool: Send + Sync`）。
    pool: *mut TensorPool,
    /// 生命周期标记
    _marker: PhantomData<&'pool mut TensorPool>,
}

/// # Safety
///
/// `PooledTensor` 可以安全地发送到其他线程，因为：
/// 1. 内部 `tensor: DenseTensor` 已实现 `Send`
/// 2. `pool` 指针仅用于在 `Drop` 时回收张量，不在线程间共享状态
/// 3. 生命周期 `'pool` 确保指针有效性
#[cfg(feature = "tensor-pool")]
unsafe impl<'pool> Send for PooledTensor<'pool> {}

/// # Safety
///
/// `PooledTensor` 可以安全地在线程间共享，因为：
/// 1. 内部 `tensor: DenseTensor` 已实现 `Sync`
/// 2. `pool` 指针仅在 `Drop` 时访问，且 `TensorPool` 本身是 `Sync` 的
/// 3. 所有可变操作都通过 `&mut self` 方法（如 `tensor_mut()`）进行，由 Rust 借用规则保证安全
#[cfg(feature = "tensor-pool")]
unsafe impl<'pool> Sync for PooledTensor<'pool> {}

#[cfg(feature = "tensor-pool")]
impl<'pool> PooledTensor<'pool> {
    /// 创建新的池化张量
    fn new(tensor: DenseTensor, pool: &'pool mut TensorPool) -> Self {
        Self {
            tensor,
            pool: pool as *mut TensorPool,
            _marker: PhantomData,
        }
    }

    /// 获取内部张量引用
    pub fn tensor(&self) -> &DenseTensor {
        &self.tensor
    }

    /// 获取内部张量可变引用
    pub fn tensor_mut(&mut self) -> &mut DenseTensor {
        &mut self.tensor
    }

    /// 消耗包装器并返回内部张量（不回收）
    pub fn into_inner(mut self) -> DenseTensor {
        let tensor = core::mem::take(&mut self.tensor);
        core::mem::forget(self); // 防止 drop
        tensor
    }
}

#[cfg(feature = "tensor-pool")]
impl<'pool> core::ops::Deref for PooledTensor<'pool> {
    type Target = DenseTensor;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

#[cfg(feature = "tensor-pool")]
impl<'pool> core::ops::DerefMut for PooledTensor<'pool> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

#[cfg(feature = "tensor-pool")]
impl<'pool> Drop for PooledTensor<'pool> {
    fn drop(&mut self) {
        // SAFETY: pool 指针在 PooledTensor 创建时保证有效，
        // 且生命周期 'pool 保证 pool 比 PooledTensor 活得长
        unsafe {
            if let Some(pool) = self.pool.as_mut() {
                pool.recycle(core::mem::take(&mut self.tensor));
            }
        }
    }
}

#[cfg(feature = "tensor-pool")]
impl<'pool> Clone for PooledTensor<'pool> {
    fn clone(&self) -> Self {
        // Clone 不涉及池，直接克隆内部 tensor
        PooledTensor::new(self.tensor.clone(), unsafe { &mut *self.pool })
    }
}

/// 梯度检查点：用于减少反向传播的内存占用
#[cfg(feature = "tensor-autograd")]
pub struct GradientCheckpoint {
    /// 保存的张量
    saved_tensors: std::collections::HashMap<usize, DenseTensor>,
    /// 最大保存数量
    max_saved: usize,
    /// 当前内存使用
    memory_used: usize,
    /// 内存预算（字节）
    memory_budget: usize,
}

#[cfg(feature = "tensor-autograd")]
impl GradientCheckpoint {
    /// 创建新的梯度检查点
    pub fn new(memory_budget: usize) -> Self {
        Self {
            saved_tensors: std::collections::HashMap::new(),
            max_saved: 100,
            memory_used: 0,
            memory_budget,
        }
    }

    /// 保存张量
    pub fn save(&mut self, id: usize, tensor: DenseTensor) {
        let size = tensor.nbytes();

        // 检查内存预算
        if self.memory_used + size > self.memory_budget {
            // 触发重新计算策略（简化：移除最旧的）
            self.evict_oldest();
        }

        if self.saved_tensors.len() < self.max_saved {
            self.memory_used += size;
            self.saved_tensors.insert(id, tensor);
        }
    }

    /// 获取保存的张量
    pub fn get(&self, id: usize) -> Option<&DenseTensor> {
        self.saved_tensors.get(&id)
    }

    /// 移除并返回张量
    pub fn take(&mut self, id: usize) -> Option<DenseTensor> {
        if let Some(tensor) = self.saved_tensors.remove(&id) {
            self.memory_used -= tensor.nbytes();
            Some(tensor)
        } else {
            None
        }
    }

    /// 清除所有保存的张量
    pub fn clear(&mut self) {
        self.saved_tensors.clear();
        self.memory_used = 0;
    }

    /// 获取内存使用量
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// 获取保存的张量数量
    pub fn len(&self) -> usize {
        self.saved_tensors.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.saved_tensors.is_empty()
    }

    /// 移除最旧的张量（简化实现：随机移除）
    fn evict_oldest(&mut self) {
        if let Some((&id, _)) = self.saved_tensors.iter().next() {
            self.take(id);
        }
    }
}

#[cfg(all(feature = "tensor-pool", test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig::new(8, 64);
        let pool = TensorPool::new(config);

        assert_eq!(pool.free_list.len(), 0);
        assert_eq!(pool.stats.total_allocations, 0);
    }

    #[test]
    fn test_pool_acquire() {
        let config = PoolConfig::new(4, 16);
        let mut pool = TensorPool::new(config);

        // 获取张量
        {
            let tensor = pool.acquire(vec![10]);
            assert_eq!(tensor.shape(), &[10]);
        } // tensor 在这里被 drop 并回收

        // 池中应该有 1 个回收的张量
        assert_eq!(pool.free_list.len(), 1);
        assert_eq!(pool.stats.total_allocations, 1);
    }
}

// ============================================================================
// TensorArena: Bump Allocator for High-Performance Tensor Allocation
// ============================================================================

/// Shape key for memory reuse matching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShapeKey {
    shape: SmallVec<[usize; 4]>,
    ndim: usize,
}

impl ShapeKey {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.into(),
            ndim: shape.len(),
        }
    }
}

/// Memory slice from the arena
#[derive(Clone)]
struct ArenaSlice {
    /// Start pointer (raw pointer into arena)
    ptr: *mut f64,
    /// Number of elements
    len: usize,
    /// Shape
    shape: SmallVec<[usize; 4]>,
    /// Whether borrowed (prevents double-free)
    borrowed: bool,
}

/// Arena-allocated tensor wrapper
pub struct ArenaTensor {
    /// Data pointer
    ptr: *mut f64,
    /// Number of elements
    len: usize,
    /// Shape
    shape: SmallVec<[usize; 4]>,
    /// Whether borrowed (prevents double-free on drop)
    borrowed: bool,
}

/// Tensor Arena allocator using bumpalo
///
/// Provides shape-aware memory reuse with bump allocation strategy.
/// Memory is allocated from the arena and can be reused for tensors
/// with the same shape, avoiding repeated allocations.
#[cfg(feature = "tensor-pool")]
pub struct TensorArena {
    /// Underlying bump arena
    arena: bumpalo::Bump,
    /// Shape-aware free lists for reuse
    free_lists: std::collections::HashMap<ShapeKey, Vec<ArenaSlice>>,
    /// Allocation statistics
    stats: ArenaStats,
    /// Total capacity in bytes
    capacity: usize,
}

/// Arena statistics
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total allocations
    pub allocation_count: usize,
    /// Total deallocations
    pub deallocation_count: usize,
    /// Reuse count (from free list)
    pub reuse_count: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Current bytes in use
    pub bytes_in_use: usize,
    /// Peak bytes in use
    pub peak_bytes_in_use: usize,
}

impl ArenaStats {
    /// Reuse ratio (reuse / total allocations)
    pub fn reuse_ratio(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.reuse_count as f64 / self.allocation_count as f64
        }
    }

    /// Memory efficiency (peak use / total allocated)
    pub fn memory_efficiency(&self) -> f64 {
        if self.total_bytes_allocated == 0 {
            0.0
        } else {
            self.peak_bytes_in_use as f64 / self.total_bytes_allocated as f64
        }
    }
}

#[cfg(feature = "tensor-pool")]
impl TensorArena {
    /// Create a new tensor arena with default capacity (16 MB)
    pub fn new() -> Self {
        Self::with_capacity(16 * 1024 * 1024)
    }

    /// Create a new tensor arena with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: bumpalo::Bump::with_capacity(capacity),
            free_lists: std::collections::HashMap::new(),
            stats: ArenaStats::default(),
            capacity,
        }
    }

    /// Allocate a tensor with the given shape
    ///
    /// Tries to reuse memory from the free list if a matching shape exists.
    /// Otherwise allocates new memory from the bump arena.
    pub fn allocate(&mut self, shape: &[usize]) -> Result<ArenaTensor, crate::tensor::error::TensorError>
    {
        let key = ShapeKey::new(shape);
        let size = shape.iter().product::<usize>();

        // Try to reuse from free list
        if let Some(slices) = self.free_lists.get_mut(&key) {
            if let Some(mut slice) = slices.pop() {
                self.stats.reuse_count += 1;
                self.stats.bytes_in_use += size * core::mem::size_of::<f64>();
                self.update_peak();
                
                slice.borrowed = true;
                return Ok(ArenaTensor {
                    ptr: slice.ptr,
                    len: size,
                    shape: slice.shape.clone(),
                    borrowed: true,
                });
            }
        }

        // Allocate new memory from bump arena
        let layout = std::alloc::Layout::from_size_align(
            size * core::mem::size_of::<f64>(),
            64, // 64-byte alignment for SIMD
        ).map_err(|e| crate::tensor::error::TensorError::AllocationError {
            message: format!("Failed to create layout: {}", e),
        })?;

        let ptr = self.arena.alloc_layout(layout).as_ptr() as *mut f64;
        
        self.stats.allocation_count += 1;
        self.stats.total_bytes_allocated += size * core::mem::size_of::<f64>();
        self.stats.bytes_in_use += size * core::mem::size_of::<f64>();
        self.update_peak();

        Ok(ArenaTensor {
            ptr,
            len: size,
            shape: key.shape,
            borrowed: true,
        })
    }

    /// Deallocate a tensor and return its memory to the free list
    pub fn deallocate(&mut self, mut tensor: ArenaTensor) {
        if tensor.borrowed {
            tensor.borrowed = false;
        }

        let key = ShapeKey::new(&tensor.shape);
        let slice = ArenaSlice {
            ptr: tensor.ptr,
            len: tensor.len,
            shape: tensor.shape.clone(),
            borrowed: false,
        };

        self.free_lists
            .entry(key)
            .or_insert_with(Vec::new)
            .push(slice);

        self.stats.deallocation_count += 1;
        self.stats.bytes_in_use -= tensor.len * core::mem::size_of::<f64>();
    }

    /// Reset the arena, clearing all free lists
    ///
    /// This releases all memory back to the system.
    pub fn reset(&mut self) {
        self.arena.reset();
        self.free_lists.clear();
        self.stats = ArenaStats::default();
        self.stats.bytes_in_use = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> &ArenaStats {
        &self.stats
    }

    /// Get the current capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current bytes in use
    pub fn bytes_in_use(&self) -> usize {
        self.stats.bytes_in_use
    }

    /// Update peak memory usage
    fn update_peak(&mut self) {
        if self.stats.bytes_in_use > self.stats.peak_bytes_in_use {
            self.stats.peak_bytes_in_use = self.stats.bytes_in_use;
        }
    }

    /// Force allocate without reuse (for benchmarking)
    pub fn allocate_fresh(&mut self, shape: &[usize]) -> Result<ArenaTensor, crate::tensor::error::TensorError> {
        let size = shape.iter().product::<usize>();
        
        let layout = std::alloc::Layout::from_size_align(
            size * core::mem::size_of::<f64>(),
            64,
        ).map_err(|e| crate::tensor::error::TensorError::AllocationError {
            message: format!("Failed to create layout: {}", e),
        })?;

        let ptr = self.arena.alloc_layout(layout).as_ptr() as *mut f64;
        
        self.stats.allocation_count += 1;
        self.stats.total_bytes_allocated += size * core::mem::size_of::<f64>();
        self.stats.bytes_in_use += size * core::mem::size_of::<f64>();
        self.update_peak();

        Ok(ArenaTensor {
            ptr,
            len: size,
            shape: shape.into(),
            borrowed: true,
        })
    }
}

#[cfg(feature = "tensor-pool")]
impl Default for TensorArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "tensor-pool")]
impl fmt::Debug for TensorArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorArena")
            .field("capacity", &self.capacity)
            .field("free_lists_count", &self.free_lists.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ArenaTensor implementation
impl ArenaTensor {
    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer (unsafe)
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr
    }

    /// Get mutable raw pointer (unsafe)
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr
    }

    /// Get slice (unsafe, for reading)
    ///
    /// # Safety
    /// Caller must ensure no other mutable references exist
    pub unsafe fn as_slice(&self) -> &[f64] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }

    /// Get mutable slice (unsafe, for writing)
    ///
    /// # Safety
    /// Caller must ensure no other references exist
    pub unsafe fn as_mut_slice(&mut self) -> &mut [f64] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }

    /// Zero out the data
    ///
    /// # Safety
    /// Caller must ensure exclusive access
    pub unsafe fn zero(&mut self) {
        std::ptr::write_bytes(self.ptr, 0, self.len);
    }
}

impl Clone for ArenaTensor {
    fn clone(&self) -> Self {
        // Clone creates a new allocation, not a reference
        unsafe {
            let layout = std::alloc::Layout::from_size_align(
                self.len * core::mem::size_of::<f64>(),
                64,
            ).unwrap();
            let new_ptr = std::alloc::alloc(layout) as *mut f64;
            std::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
            
            ArenaTensor {
                ptr: new_ptr,
                len: self.len,
                shape: self.shape.clone(),
                borrowed: false, // Not managed by arena
            }
        }
    }
}

impl Drop for ArenaTensor {
    fn drop(&mut self) {
        // Memory is managed by the arena, don't free here
        // The borrowed flag prevents issues if manually dropped
    }
}

#[cfg(all(feature = "tensor-pool", test, feature = "std"))]
mod arena_tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = TensorArena::with_capacity(1024 * 1024);
        assert_eq!(arena.capacity(), 1024 * 1024);
        assert_eq!(arena.bytes_in_use(), 0);
    }

    #[test]
    fn test_arena_allocate() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        let shape = vec![10, 10];
        
        let tensor = arena.allocate(&shape).unwrap();
        assert_eq!(tensor.shape(), &[10, 10]);
        assert_eq!(tensor.len(), 100);
        
        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.reuse_count, 0);
    }

    #[test]
    fn test_arena_reuse() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        let shape = vec![5, 5];
        
        // Allocate
        let tensor1 = arena.allocate(&shape).unwrap();
        let stats_after_alloc = arena.stats().allocation_count;
        
        // Explicitly deallocate to return to free list
        arena.deallocate(tensor1);
        
        // Allocate again with same shape - should reuse from free list
        let _tensor2 = arena.allocate(&shape).unwrap();
        
        let stats = arena.stats();
        // Should have 1 allocation (first one) and 1 reuse (second one)
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.reuse_count, 1);
    }

    #[test]
    fn test_arena_different_shapes() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        
        let t1 = arena.allocate(&[10]).unwrap();
        let t2 = arena.allocate(&[20]).unwrap();
        let shape1 = t1.shape().to_vec();
        let shape2 = t2.shape().to_vec();
        arena.deallocate(t1);
        arena.deallocate(t2);
        let t3 = arena.allocate(&[10]).unwrap();
        
        // t3 should reuse t1's memory from free list
        assert_eq!(shape1, vec![10]);
        assert_eq!(shape2, vec![20]);
        assert_eq!(t3.shape(), &[10]);
        
        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2); // t1 and t2
        assert_eq!(stats.reuse_count, 1); // t3 reused t1
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        
        let _t1 = arena.allocate(&[100]).unwrap();
        let _t2 = arena.allocate(&[200]).unwrap();
        
        arena.reset();
        
        assert_eq!(arena.bytes_in_use(), 0);
        assert_eq!(arena.stats().allocation_count, 0);
        assert_eq!(arena.stats().reuse_count, 0);
    }

    #[test]
    fn test_arena_stats() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        
        let shape = vec![10, 10];
        let size_bytes = 100 * core::mem::size_of::<f64>();
        
        let t1 = arena.allocate(&shape).unwrap();
        arena.deallocate(t1);
        let _t2 = arena.allocate(&shape).unwrap();
        
        let stats = arena.stats();
        // First allocation + first reuse
        assert_eq!(stats.total_bytes_allocated, size_bytes);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.reuse_count, 1);
        // reuse_ratio = reuse_count / allocation_count = 1/1 = 1.0 (100% reuse)
        assert_eq!(stats.reuse_ratio(), 1.0);
    }

    #[test]
    fn test_arena_tensor_zero() {
        let mut arena = TensorArena::with_capacity(1024 * 1024);
        let mut tensor = arena.allocate(&[10]).unwrap();
        
        unsafe {
            tensor.zero();
            let slice = tensor.as_slice();
            for &val in slice {
                assert_eq!(val, 0.0);
            }
        }
    }
}
