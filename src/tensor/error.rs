//! Tensor 错误类型
//!
//! 定义 tensor 操作中可能出现的各种错误

use core::fmt;

/// Tensor 操作错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// 形状不匹配错误
    ShapeMismatch {
        /// 期望的形状
        expected: Vec<usize>,
        /// 实际得到的形状
        got: Vec<usize>,
    },

    /// 索引越界错误
    IndexOutOfBounds {
        /// 越界的索引
        index: usize,
        /// 维度
        dim: usize,
        /// 该维度的大小
        size: usize,
    },

    /// 维度不匹配错误
    DimensionMismatch {
        /// 期望的维度数
        expected: usize,
        /// 实际得到的维度数
        got: usize,
    },

    /// 数据类型不支持错误
    UnsupportedDType {
        /// 不支持的数据类型
        dtype: String,
        /// 操作名称
        operation: String,
    },

    /// 设备不支持错误
    UnsupportedDevice {
        /// 不支持的设备
        device: String,
    },

    /// 内存不足错误
    OutOfMemory {
        /// 请求的字节数
        requested_bytes: usize,
    },

    /// BLAS 错误
    BlasError {
        /// BLAS 错误码
        code: i32,
        /// 错误描述
        description: String,
    },

    /// 稀疏格式转换错误
    SparseFormatError {
        /// 源格式
        from: String,
        /// 目标格式
        to: String,
        /// 错误描述
        description: String,
    },

    /// 广播错误
    BroadcastError {
        /// 第一个形状
        shape1: Vec<usize>,
        /// 第二个形状
        shape2: Vec<usize>,
    },

    /// 切片错误
    SliceError {
        /// 错误描述
        description: String,
    },

    /// 设备间传输错误
    DeviceTransferError {
        /// 源设备
        from: String,
        /// 目标设备
        to: String,
        /// 错误描述
        description: String,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::IndexOutOfBounds { index, dim, size } => {
                write!(f, "Index {} out of bounds for dimension {} (size {})", index, dim, size)
            }
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}D, got {}D", expected, got)
            }
            TensorError::UnsupportedDType { dtype, operation } => {
                write!(f, "Unsupported dtype {} for operation {}", dtype, operation)
            }
            TensorError::UnsupportedDevice { device } => {
                write!(f, "Unsupported device: {}", device)
            }
            TensorError::OutOfMemory { requested_bytes } => {
                write!(f, "Out of memory: requested {} bytes", requested_bytes)
            }
            TensorError::BlasError { code, description } => {
                write!(f, "BLAS error (code {}): {}", code, description)
            }
            TensorError::SparseFormatError { from, to, description } => {
                write!(f, "Sparse format error converting {} to {}: {}", from, to, description)
            }
            TensorError::BroadcastError { shape1, shape2 } => {
                write!(f, "Cannot broadcast shapes {:?} and {:?}", shape1, shape2)
            }
            TensorError::SliceError { description } => {
                write!(f, "Slice error: {}", description)
            }
            TensorError::DeviceTransferError { from, to, description } => {
                write!(f, "Device transfer error from {} to {}: {}", from, to, description)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

/// Tensor 操作结果类型别名
pub type TensorResult<T> = Result<T, TensorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        };
        assert_eq!(
            format!("{}", err),
            "Shape mismatch: expected [2, 3], got [3, 2]"
        );

        let err = TensorError::IndexOutOfBounds {
            index: 5,
            dim: 0,
            size: 3,
        };
        assert_eq!(
            format!("{}", err),
            "Index 5 out of bounds for dimension 0 (size 3)"
        );
    }

    #[test]
    fn test_error_equality() {
        let err1 = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        };
        let err2 = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        };
        assert_eq!(err1, err2);

        let err3 = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![2, 3],
        };
        assert_ne!(err1, err3);
    }
}
