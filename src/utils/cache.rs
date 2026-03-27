//! Cache 优化和预取工具
//!
//! 提供 CPU 缓存优化功能，包括：
//! - 64 字节对齐的填充类型，避免多线程访问时的 false sharing
//! - 软件预取指令，提前将数据加载到 CPU 缓存
//!
//! ## 使用示例
//!
//! ```rust
//! use god_gragh::utils::cache::{Padded, prefetch_read};
//!
//! // 创建对齐的填充值
//! let padded = Padded::new(42);
//! assert_eq!(*padded, 42);
//!
//! // 预取数据到缓存
//! let data = vec![1, 2, 3, 4, 5];
//! prefetch_read(&data[0]);
//! ```

/// 64 字节对齐的填充类型，避免 false sharing
///
/// 在多线程环境中，当多个线程访问相邻内存时，
/// 即使访问不同的变量，也可能因为缓存行共享而导致性能下降。
/// `Padded<T>` 通过将数据填充到 64 字节（缓存行大小）来避免这个问题。
///
/// ## 类型参数
///
/// - `T`: 要包装的数据类型
///
/// ## 示例
///
/// ```rust
/// use god_gragh::utils::cache::Padded;
///
/// let padded = Padded::new(42);
/// assert_eq!(*padded, 42);
/// assert!(Padded::<u8>::size() >= 64); // 至少 64 字节
/// ```
#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Padded<T> {
    /// 包装的数据
    pub data: T,
    /// 填充字节，确保 64 字节对齐
    _padding: [u8; 64],
}

impl<T> Padded<T> {
    /// 创建新的 Padded 包装
    ///
    /// ## 参数
    ///
    /// - `data`: 要包装的数据
    ///
    /// ## 示例
    ///
    /// ```rust
    /// use god_gragh::utils::cache::Padded;
    ///
    /// let padded = Padded::new(42);
    /// assert_eq!(padded.data, 42);
    /// ```
    pub fn new(data: T) -> Self {
        Self {
            data,
            _padding: [0; 64],
        }
    }

    /// 获取 Padded 结构的大小（字节）
    ///
    /// 返回值总是 >= 64，因为缓存行对齐
    pub fn size() -> usize {
        core::mem::size_of::<Self>()
    }

    /// 验证指针是否 64 字节对齐
    ///
    /// ## 参数
    ///
    /// - `ptr`: 要验证的指针
    ///
    /// ## 返回
    ///
    /// - `true`: 对齐正确
    /// - `false`: 对齐错误
    ///
    /// ## 示例
    ///
    /// ```rust
    /// use god_gragh::utils::cache::Padded;
    ///
    /// let padded = Padded::new(42);
    /// assert!(Padded::is_aligned(&padded));
    /// ```
    pub fn is_aligned(ptr: &T) -> bool {
        (ptr as *const T as usize) % 64 == 0
    }

    /// 验证 Vec 中 T 的指针是否 64 字节对齐
    ///
    /// ## 参数
    ///
    /// - `vec`: 要验证的 Vec
    ///
    /// ## 返回
    ///
    /// - `true`: 对齐正确
    /// - `false`: 对齐错误
    ///
    /// ## 示例
    ///
    /// ```rust
    /// use god_gragh::utils::cache::Padded;
    ///
    /// let vec: Vec<Padded<u8>> = vec![Padded::new(1), Padded::new(2)];
    /// assert!(Padded::is_vec_aligned(&vec));
    /// ```
    pub fn is_vec_aligned(vec: &[T]) -> bool {
        if vec.is_empty() {
            return true;
        }
        (vec.as_ptr() as usize) % 64 == 0
    }
}

impl<T> core::ops::Deref for Padded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> core::ops::DerefMut for Padded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: Default> Default for Padded<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// 预取数据到 CPU 缓存用于读取
///
/// 使用 CPU 的 prefetch 指令提前将数据加载到缓存中，
/// 可以减少后续访问的延迟。这是一个性能优化提示，
/// 在某些架构上可能被忽略。
///
/// ## 参数
///
/// - `data`: 要预取的数据
///
/// ## 平台支持
///
/// - x86_64: 使用 `_mm_prefetch` 指令
/// - 其他架构：使用 `std::hint::prefetch_read_data`（需要 std）
///
/// ## 示例
///
/// ```rust
/// use god_gragh::utils::cache::prefetch_read;
///
/// let data = vec![1, 2, 3, 4, 5];
/// prefetch_read(&data[0]);
/// // 现在访问 data[0] 可能更快
/// ```
#[inline]
pub fn prefetch_read<T>(data: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` 是 CPU 内置指令，仅读取缓存不修改内存，
    // 指针由 Rust 引用转换而来，保证有效且对齐
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            data as *const T as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        // SAFETY: `prefetch_read_data` 仅预取数据到缓存，不修改内存，
        // 指针有效且大小正确
        unsafe {
            std::hint::prefetch_read_data(
                data as *const T as *const _,
                core::mem::size_of::<T>(),
            );
        }
    }
}

/// 预取数据到 CPU 缓存用于写入
///
/// 使用 CPU 的 prefetch 指令提前将数据加载到缓存中，
/// 为后续的写入操作做准备。
///
/// ## 参数
///
/// - `data`: 要预取的数据
///
/// ## 平台支持
///
/// - x86_64: 使用 `_mm_prefetch` 指令
/// - 其他架构：无操作（空实现）
///
/// ## 示例
///
/// ```rust
/// use god_gragh::utils::cache::prefetch_write;
///
/// let mut data = vec![1, 2, 3, 4, 5];
/// prefetch_write(&mut data[0]);
/// // 现在写入 data[0] 可能更快
/// ```
#[inline]
pub fn prefetch_write<T>(data: &mut T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` 是 CPU 内置指令，预取到缓存用于后续写入，
    // 可变引用保证指针有效且独占访问
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            data as *mut T as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_size() {
        assert!(Padded::<u8>::size() >= 64);
    }

    #[test]
    fn test_padded_deref() {
        let padded = Padded::new(42);
        assert_eq!(*padded, 42);

        let mut padded = Padded::new(42);
        *padded = 43;
        assert_eq!(padded.data, 43);
    }
}
