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
        PooledTensor::new(
            self.tensor.clone(),
            unsafe { &mut *self.pool }
        )
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
