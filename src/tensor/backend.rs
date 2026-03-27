//! Tensor 后端抽象：支持多种 backend 实现
//!
//! 本模块提供了 tensor 后端的抽象层次，支持：
//! - NdArrayBackend: 基于 ndarray 的 CPU backend
//! - DfdxBackend: 基于 dfdx 的 GPU backend（支持自动微分）
//! - CandleBackend: 基于 candle 的轻量级 backend

use core::fmt::Debug;

use crate::tensor::traits::{DType, Device};

/// Tensor 存储后端 trait
pub trait TensorStorage: Clone + Send + Sync + Debug {
    /// 获取数据类型
    fn dtype(&self) -> DType;

    /// 获取设备类型
    fn device(&self) -> Device;

    /// 获取字节大小
    fn nbytes(&self) -> usize;

    /// 检查是否连续存储
    fn is_contiguous(&self) -> bool;

    /// 获取对齐字节数
    fn alignment(&self) -> usize;
}

/// NdArray 存储后端
#[derive(Clone, Debug)]
#[cfg(feature = "tensor")]
pub struct NdArrayStorage {
    /// 数据（64 字节对齐）
    data: Vec<f64>,
    /// 数据类型
    dtype: DType,
}

#[cfg(feature = "tensor")]
impl NdArrayStorage {
    /// 创建新的 NdArray 存储
    pub fn new(data: Vec<f64>, dtype: DType) -> Self {
        Self { data, dtype }
    }

    /// 获取数据切片
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// 获取可变数据切片
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

#[cfg(feature = "tensor")]
impl TensorStorage for NdArrayStorage {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn nbytes(&self) -> usize {
        self.data.len() * self.dtype.size_bytes()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn alignment(&self) -> usize {
        64 // Vec<f64> 默认对齐
    }
}

/// Dfdx 存储后端（GPU 加速）
#[cfg(feature = "tensor-gpu")]
#[derive(Clone, Debug)]
pub struct DfdxStorage {
    /// 内部存储（使用 dfdx 的 Tensor 类型）
    inner: dfdx::tensor::Tensor1D<f64>,
}

#[cfg(feature = "tensor-gpu")]
impl DfdxStorage {
    /// 从 dfdx tensor 创建
    pub fn from_dfdx(tensor: dfdx::tensor::Tensor1D<f64>) -> Self {
        Self { inner: tensor }
    }

    /// 获取内部 dfdx tensor
    pub fn inner(&self) -> &dfdx::tensor::Tensor1D<f64> {
        &self.inner
    }
}

#[cfg(feature = "tensor-gpu")]
impl TensorStorage for DfdxStorage {
    fn dtype(&self) -> DType {
        DType::F64
    }

    fn device(&self) -> Device {
        Device::Cuda(0) // 默认使用 GPU 0
    }

    fn nbytes(&self) -> usize {
        self.inner.shape().0 * 8
    }

    fn is_contiguous(&self) -> bool {
        true // dfdx tensors are contiguous
    }

    fn alignment(&self) -> usize {
        128 // CUDA memory alignment
    }
}

/// Candle 存储后端（Hugging Face 轻量级 backend）
#[cfg(feature = "tensor-candle")]
#[derive(Clone, Debug)]
pub struct CandleStorage {
    /// 内部 candle tensor
    inner: candle_core::Tensor,
}

#[cfg(feature = "tensor-candle")]
impl CandleStorage {
    /// 从 candle tensor 创建
    pub fn from_candle(tensor: candle_core::Tensor) -> Self {
        Self { inner: tensor }
    }

    /// 获取内部 candle tensor
    pub fn inner(&self) -> &candle_core::Tensor {
        &self.inner
    }
}

#[cfg(feature = "tensor-candle")]
impl TensorStorage for CandleStorage {
    fn dtype(&self) -> DType {
        match self.inner.dtype() {
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::F64 => DType::F64,
            candle_core::DType::I32 => DType::I32,
            candle_core::DType::I64 => DType::I64,
            _ => DType::F64,
        }
    }

    fn device(&self) -> Device {
        match self.inner.device() {
            candle_core::Device::Cpu => Device::Cpu,
            candle_core::Device::Cuda(_) => Device::Cuda(0),
            candle_core::Device::Metal(_) => Device::Cpu, // Treat as CPU for now
        }
    }

    fn nbytes(&self) -> usize {
        self.inner.elem_count() * self.dtype().size_bytes()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn alignment(&self) -> usize {
        64
    }
}

/// 统一的存储后端枚举
#[derive(Clone)]
pub enum UnifiedStorage {
    /// NdArray 后端
    NdArray(NdArrayStorage),
    /// Dfdx 后端（GPU）
    #[cfg(feature = "tensor-gpu")]
    Dfdx(DfdxStorage),
    /// Candle 后端
    #[cfg(feature = "tensor-candle")]
    Candle(CandleStorage),
}

#[cfg(feature = "tensor")]
impl UnifiedStorage {
    /// 从 NdArray 存储创建
    pub fn ndarray(data: Vec<f64>, dtype: DType) -> Self {
        UnifiedStorage::NdArray(NdArrayStorage::new(data, dtype))
    }
}

impl TensorStorage for UnifiedStorage {
    fn dtype(&self) -> DType {
        match self {
            UnifiedStorage::NdArray(s) => s.dtype(),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(s) => s.dtype(),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(s) => s.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            UnifiedStorage::NdArray(s) => s.device(),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(s) => s.device(),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(s) => s.device(),
        }
    }

    fn nbytes(&self) -> usize {
        match self {
            UnifiedStorage::NdArray(s) => s.nbytes(),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(s) => s.nbytes(),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(s) => s.nbytes(),
        }
    }

    fn is_contiguous(&self) -> bool {
        match self {
            UnifiedStorage::NdArray(s) => s.is_contiguous(),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(s) => s.is_contiguous(),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(s) => s.is_contiguous(),
        }
    }

    fn alignment(&self) -> usize {
        match self {
            UnifiedStorage::NdArray(s) => s.alignment(),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(s) => s.alignment(),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(s) => s.alignment(),
        }
    }
}

impl Debug for UnifiedStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UnifiedStorage::NdArray(_) => write!(f, "UnifiedStorage::NdArray"),
            #[cfg(feature = "tensor-gpu")]
            UnifiedStorage::Dfdx(_) => write!(f, "UnifiedStorage::Dfdx"),
            #[cfg(feature = "tensor-candle")]
            UnifiedStorage::Candle(_) => write!(f, "UnifiedStorage::Candle"),
        }
    }
}
