//! 分支预测优化工具
//!
//! 提供分支预测提示（branch prediction hints）帮助编译器优化热点代码路径
//!
//! # 性能优化原理
//!
//! 现代 CPU 使用流水线执行指令，当遇到条件分支时，CPU 需要预测执行哪个分支。
//! 错误的预测会导致流水线清空，造成 10-20 个时钟周期的损失。
//!
//! 这些工具函数使用 `#[cold]` 和 `#[inline(always)]` 属性来提示编译器：
//! - 热点路径（hot path）：内联并优先优化
//! - 冷路径（cold path）：不频繁执行，可以放在单独的代码段
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use god_graph::utils::branch::{likely, unlikely};
//!
//! fn process_data(data: &[i32]) -> i32 {
//!     let mut sum = 0;
//!     for &x in data {
//!         // 常见情况：数据有效（热点路径）
//!         if likely(x > 0) {
//!             sum += x;
//!         } else {
//!             // 罕见情况：数据无效（冷路径）
//!             continue;
//!         }
//!     }
//!     sum
//! }
//! ```

/// 提示编译器该分支很可能为真（热点路径）
///
/// # 性能影响
/// - 帮助编译器正确安排代码布局
/// - 提高分支预测准确率
/// - 减少流水线清空损失
///
/// # 使用场景
/// - 错误检查（通常不会出错）
/// - 缓存命中（通常命中）
/// - 常见数据模式
#[inline(always)]
pub fn likely(b: bool) -> bool {
    b
}

/// 提示编译器该分支很可能为假（冷路径）
///
/// # 性能影响
/// - 帮助编译器将冷路径代码移到单独段
/// - 提高指令缓存利用率
/// - 减少热点路径的指令缓存压力
///
/// # 使用场景
/// - 错误处理（通常不触发）
/// - 边界条件（罕见情况）
/// - 异常数据模式
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    b
}

/// 标记冷函数（不频繁执行的函数）
///
/// # 用途
/// - 错误处理函数
/// - 初始化/清理代码
/// - 罕见情况处理
///
/// # 示例
///
/// ```rust,no_run
/// use god_graph::utils::branch::cold;
///
/// #[cold]
/// fn handle_error(msg: &str) {
///     eprintln!("Error: {}", msg);
/// }
/// ```
#[inline(always)]
pub fn cold() {}

/// 内联提示：总是内联
///
/// # 用途
/// - 小型热点函数
/// - 性能关键的 accessor
/// - 闭包和回调
///
/// # 注意
/// 过度使用会增加代码大小，可能降低性能
#[inline(always)]
pub fn inline_always() {}

/// 不内联提示
///
/// # 用途
/// - 大型函数
/// - 冷路径函数
/// - 减少编译时间和二进制大小
///
/// # 注意
/// 可能增加函数调用开销，但减少代码膨胀
#[inline(never)]
pub fn noinline() {}

/// 分支预测优化宏：likely
///
/// 使用示例：
/// ```rust,no_run
/// use god_graph::branch_likely;
///
/// if branch_likely!(x > 0) {
///     // 热点路径
/// } else {
///     // 冷路径
/// }
/// ```
#[macro_export]
macro_rules! branch_likely {
    ($cond:expr) => {
        $crate::utils::branch::likely($cond)
    };
}

/// 分支预测优化宏：unlikely
///
/// 使用示例：
/// ```rust,no_run
/// use god_graph::branch_unlikely;
///
/// if branch_unlikely!(error_occurred) {
///     // 错误处理（冷路径）
/// }
/// ```
#[macro_export]
macro_rules! branch_unlikely {
    ($cond:expr) => {
        $crate::utils::branch::unlikely($cond)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likely_true() {
        assert!(likely(true));
    }

    #[test]
    fn test_likely_false() {
        assert!(!likely(false));
    }

    #[test]
    fn test_unlikely_true() {
        assert!(unlikely(true));
    }

    #[test]
    fn test_unlikely_false() {
        assert!(!unlikely(false));
    }

    #[test]
    fn test_branch_macros() {
        let x = 5;
        assert!(branch_likely!(x > 0));
        assert!(!branch_unlikely!(x < 0));
    }

    #[test]
    fn test_performance_pattern() {
        // 典型使用场景：错误检查
        fn process_values(values: &[i32]) -> Result<i32, &'static str> {
            for &val in values {
                // 常见情况：值有效
                if likely(val >= 0) {
                    continue;
                }
                // 罕见情况：负值错误
                if unlikely(val < 0) {
                    return Err("Negative value detected");
                }
            }
            Ok(values.iter().sum())
        }

        assert_eq!(process_values(&[1, 2, 3]), Ok(6));
        assert!(process_values(&[1, -2, 3]).is_err());
    }
}
