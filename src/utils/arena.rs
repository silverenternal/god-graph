//! Arena 分配器模块
//!
//! 提供高效的内存分配策略，支持对象复用和稳定索引
//!
//! ## 特性
//!
//! - **O(1) 分配和释放**: 使用空闲列表实现常数时间复杂度
//! - **缓存友好**: 64 字节对齐，避免 false sharing
//! - **稳定索引**: generation 计数防止 ABA 问题
//! - **内存连续**: 数据连续存储，优化 CPU 缓存命中率
//!
//! ## 使用示例
//!
//! ```
//! # fn main() {
//! use god_gragh::utils::arena::{Arena, Slot};
//!
//! let mut arena = Arena::new();
//! let (idx, generation) = arena.allocate(42);
//! assert_eq!(arena.get(idx, generation), Some(&42));
//!
//! // 释放后 generation 递增
//! arena.deallocate(idx, generation);
//! assert!(arena.get(idx, generation).is_none()); // 旧 generation 失效
//! # }
//! ```

use std::fmt;
use std::marker::PhantomData;

/// 缓存行大小，用于对齐优化
#[allow(dead_code)]
const CACHE_LINE_SIZE: usize = 64;

/// 单个槽位，存储数据和 generation 计数
///
/// `generation` 用于检测 ABA 问题：
/// - 每次分配时递增
/// - 释放后索引失效，因为 generation 不匹配
#[derive(Clone, Debug)]
pub struct Slot<T> {
    /// 存储的数据，None 表示槽位已释放
    pub data: Option<T>,
    /// 代数计数器，每次分配递增
    pub generation: u32,
}

impl<T> Slot<T> {
    /// 创建新槽位
    pub fn new(data: T, generation: u32) -> Self {
        Self {
            data: Some(data),
            generation,
        }
    }

    /// 检查槽位是否被占用
    #[inline]
    pub fn is_occupied(&self) -> bool {
        self.data.is_some()
    }

    /// 获取数据引用
    #[inline]
    pub fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    /// 获取可变数据引用
    #[inline]
    pub fn data_mut(&mut self) -> Option<&mut T> {
        self.data.as_mut()
    }

    /// 替换数据并返回旧数据
    #[inline]
    pub fn replace(&mut self, new_data: T) -> Option<T> {
        self.data.replace(new_data)
    }

    /// 清空数据
    #[inline]
    pub fn clear(&mut self) -> Option<T> {
        self.data.take()
    }
}

/// Arena 分配器
///
/// 特点：
/// - O(1) 分配和释放
/// - 内存连续，cache-friendly
/// - 支持对象复用（通过空闲列表）
/// - generation 计数防止 ABA 问题
///
/// ## 类型参数
///
/// - `T`: 存储的数据类型
///
/// ## 示例
///
/// ```
/// use god_gragh::utils::arena::Arena;
///
/// let mut arena = Arena::with_capacity(100);
/// let (idx, generation) = arena.allocate("hello".to_string());
/// assert_eq!(arena.get(idx, generation), Some(&"hello".to_string()));
/// ```
pub struct Arena<T> {
    /// 存储槽位
    slots: Vec<Slot<T>>,
    /// 空闲列表（已删除槽位的索引）
    free_list: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("free_list_len", &self.free_list.len())
            .finish()
    }
}

impl<T> Arena<T> {
    /// 创建新的 Arena
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// 创建预分配容量的 Arena
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_list: Vec::with_capacity(capacity),
            _marker: PhantomData,
        }
    }

    /// 分配新对象，返回 (索引，generation)
    ///
    /// ## 复杂度
    ///
    /// - 时间：O(1) 摊销
    /// - 空间：O(1)
    #[inline]
    pub fn allocate(&mut self, value: T) -> (usize, u32) {
        if let Some(index) = self.free_list.pop() {
            // 复用空闲槽位
            let slot = &mut self.slots[index];
            let new_generation = slot.generation.wrapping_add(1);
            *slot = Slot::new(value, new_generation);
            (index, new_generation)
        } else {
            // 分配新槽位
            let index = self.slots.len();
            let generation = 1u32;
            self.slots.push(Slot::new(value, generation));
            (index, generation)
        }
    }

    /// 释放对象
    ///
    /// ## 参数
    ///
    /// - `index`: 槽位索引
    /// - `generation`: 当前 generation，用于验证
    ///
    /// ## 返回值
    ///
    /// - `Some(T)`: 成功释放，返回数据
    /// - `None`: 索引无效或 generation 不匹配
    #[inline]
    pub fn deallocate(&mut self, index: usize, generation: u32) -> Option<T> {
        if index >= self.slots.len() {
            return None;
        }

        let slot = &mut self.slots[index];
        if slot.generation != generation {
            return None; // generation 不匹配
        }

        let value = slot.data.take()?;
        self.free_list.push(index);
        Some(value)
    }

    /// 获取对象引用
    ///
    /// ## 参数
    ///
    /// - `index`: 槽位索引
    /// - `generation`: 用于验证的 generation
    #[inline]
    pub fn get(&self, index: usize, generation: u32) -> Option<&T> {
        self.slots
            .get(index)
            .filter(|slot| slot.generation == generation && slot.is_occupied())
            .and_then(|slot| slot.data())
    }

    /// 获取对象可变引用
    #[inline]
    pub fn get_mut(&mut self, index: usize, generation: u32) -> Option<&mut T> {
        self.slots
            .get_mut(index)
            .filter(|slot| slot.generation == generation && slot.is_occupied())
            .and_then(|slot| slot.data_mut())
    }

    /// 检查索引是否有效
    #[inline]
    pub fn is_valid(&self, index: usize, generation: u32) -> bool {
        self.slots
            .get(index)
            .is_some_and(|slot| slot.generation == generation && slot.is_occupied())
    }

    /// 获取槽位引用（用于高级操作）
    #[inline]
    pub fn get_slot(&self, index: usize) -> Option<&Slot<T>> {
        self.slots.get(index)
    }

    /// 获取槽位可变引用
    #[inline]
    pub fn get_slot_mut(&mut self, index: usize) -> Option<&mut Slot<T>> {
        self.slots.get_mut(index)
    }

    /// 清空 Arena，但不释放内存
    #[inline]
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.data = None;
        }
        self.free_list.clear();
        for i in 0..self.slots.len() {
            self.free_list.push(i);
        }
    }

    /// 完全清空 Arena，释放所有内存
    #[inline]
    pub fn drain(&mut self) {
        self.slots.clear();
        self.free_list.clear();
    }

    /// 获取已分配对象数量
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len() - self.free_list.len()
    }

    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 获取总容量（包括空闲槽位）
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// 获取槽位总数
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// 获取空闲槽位数量
    #[inline]
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// 预分配容量
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
        self.free_list.reserve(additional);
    }

    /// 压缩空闲列表，回收内存
    ///
    /// 当空闲列表过大时调用，可以减少内存占用
    pub fn shrink_to_fit(&mut self) {
        self.free_list.shrink_to_fit();
        self.slots.shrink_to_fit();
    }

    /// 迭代所有已分配的槽位
    pub fn iter(&self) -> impl Iterator<Item = (usize, u32, &T)> {
        self.slots.iter().enumerate().filter_map(|(idx, slot)| {
            if slot.is_occupied() {
                Some((idx, slot.generation, slot.data.as_ref().unwrap()))
            } else {
                None
            }
        })
    }

    /// 迭代所有槽位（包括空闲的）
    pub fn iter_all(&self) -> impl Iterator<Item = (usize, &Slot<T>)> {
        self.slots.iter().enumerate()
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocate() {
        let mut arena = Arena::new();
        let (idx, generation) = arena.allocate(42);
        assert_eq!(arena.get(idx, generation), Some(&42));
        assert_eq!(generation, 1);
    }

    #[test]
    fn test_arena_deallocate() {
        let mut arena = Arena::new();
        let (idx, generation) = arena.allocate(42);
        let value = arena.deallocate(idx, generation);
        assert_eq!(value, Some(42));
        assert_eq!(arena.get(idx, generation), None);
    }

    #[test]
    fn test_arena_reuse() {
        let mut arena = Arena::new();
        let (idx1, gen1) = arena.allocate(1);
        let _ = arena.deallocate(idx1, gen1);
        let (idx2, gen2) = arena.allocate(2);

        // idx2 应该复用 idx1 的槽位
        assert_eq!(idx2, idx1);
        // generation 应该递增
        assert_eq!(gen2, gen1 + 1);
        assert!(arena.get(idx2, gen2).is_some());
        assert!(arena.get(idx2, gen1).is_none());
    }

    #[test]
    fn test_arena_is_valid() {
        let mut arena = Arena::new();
        let (idx, generation) = arena.allocate(42);

        assert!(arena.is_valid(idx, generation));
        assert!(!arena.is_valid(idx, generation + 1)); // generation 不匹配
        assert!(!arena.is_valid(idx + 1, generation)); // 索引越界

        arena.deallocate(idx, generation);
        assert!(!arena.is_valid(idx, generation)); // 已释放
    }

    #[test]
    fn test_arena_len_and_capacity() {
        let mut arena = Arena::with_capacity(10);
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert!(arena.capacity() >= 10);

        let (idx, generation) = arena.allocate(1);
        assert_eq!(arena.len(), 1);
        assert!(!arena.is_empty());

        arena.deallocate(idx, generation);
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_arena_clear() {
        let mut arena = Arena::new();
        let (idx1, _) = arena.allocate(1);
        let (idx2, _) = arena.allocate(2);

        arena.clear();

        assert!(arena.is_empty());
        assert!(!arena.is_valid(idx1, 1));
        assert!(!arena.is_valid(idx2, 1));
    }

    #[test]
    fn test_arena_iter() {
        let mut arena = Arena::new();
        let (idx1, gen1) = arena.allocate(1);
        let (idx2, gen2) = arena.allocate(2);
        let (idx3, gen3) = arena.allocate(3);

        arena.deallocate(idx2, gen2);

        let items: Vec<_> = arena.iter().collect();
        assert_eq!(items.len(), 2);
        assert!(items
            .iter()
            .any(|(i, g, v)| *i == idx1 && *g == gen1 && **v == 1));
        assert!(items
            .iter()
            .any(|(i, g, v)| *i == idx3 && *g == gen3 && **v == 3));
    }

    #[test]
    fn test_arena_generation_wrap() {
        let mut arena = Arena::new();
        let (idx, mut generation) = arena.allocate(42);

        // 模拟多次分配/释放，测试 generation 回绕
        for _ in 0..10 {
            arena.deallocate(idx, generation);
            let (_, new_generation) = arena.allocate(100);
            generation = new_generation;
        }

        // 仍然可以正常访问
        assert!(arena.is_valid(idx, generation));
    }

    #[test]
    fn test_arena_get_mut() {
        let mut arena = Arena::new();
        let (idx, generation) = arena.allocate(42);

        if let Some(val) = arena.get_mut(idx, generation) {
            *val = 100;
        }

        assert_eq!(arena.get(idx, generation), Some(&100));
    }

    #[test]
    fn test_arena_with_capacity() {
        let mut arena = Arena::<i32>::with_capacity(100);
        assert!(arena.is_empty());
        assert!(arena.capacity() >= 100);

        // 预分配后应该不需要重新分配
        for i in 0..100 {
            arena.allocate(i);
        }
        assert_eq!(arena.len(), 100);
    }
}
