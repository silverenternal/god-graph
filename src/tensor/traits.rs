//! Tensor Trait 系统：定义张量的抽象接口
//!
//! 本模块提供了 tensor 的 trait 层次结构，用于后端抽象和操作定义

use core::fmt::Debug;

/// 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32 位浮点数
    F32,
    /// 64 位浮点数
    F64,
    /// 32 位整数
    I32,
    /// 64 位整数
    I64,
    /// 布尔类型
    Bool,
}

impl DType {
    /// 获取数据类型的字节大小
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }
}

/// 设备类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU 设备
    Cpu,
    /// CUDA GPU 设备（未来支持）
    Cuda(usize),
    /// WebGPU 设备（未来支持）
    Wgpu,
}

impl Default for Device {
    #[inline]
    fn default() -> Self {
        Device::Cpu
    }
}

/// Tensor 基础 trait：所有张量必须实现的核心操作
///
/// 提供形状、数据类型、设备等基本信息查询
pub trait TensorBase: Clone + Send + Sync + Debug {
    /// 获取张量的形状（各维度大小）
    fn shape(&self) -> &[usize];

    /// 获取数据类型
    fn dtype(&self) -> DType;

    /// 获取设备类型
    fn device(&self) -> Device;

    /// 获取张量的维度数（rank）
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// 获取总元素数量
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// 检查是否为标量（0 维张量）
    fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// 检查是否为向量（1 维张量）
    fn is_vector(&self) -> bool {
        self.ndim() == 1
    }

    /// 检查是否为矩阵（2 维张量）
    fn is_matrix(&self) -> bool {
        self.ndim() == 2
    }

    /// 转换为密集张量
    fn to_dense(&self) -> crate::tensor::dense::DenseTensor;

    /// 转换为稀疏张量（如果适用）
    #[cfg(feature = "tensor-sparse")]
    fn to_sparse(&self) -> Option<crate::tensor::sparse::SparseTensor>;

    /// 转换为稀疏张量（当 tensor-sparse 特性未启用时返回 None）
    #[cfg(not(feature = "tensor-sparse"))]
    fn to_sparse(&self) -> Option<()> {
        None
    }
}

/// Tensor 操作 trait：定义数学运算
///
/// 提供加法、乘法、矩阵乘法、转置等操作
pub trait TensorOps: TensorBase {
    /// 张量加法
    fn add(&self, other: &Self) -> Self;

    /// 张量减法
    fn sub(&self, other: &Self) -> Self;

    /// 逐元素乘法（Hadamard 积）
    fn mul(&self, other: &Self) -> Self;

    /// 逐元素除法
    fn div(&self, other: &Self) -> Self;

    /// 矩阵乘法（仅适用于 2D 张量）
    fn matmul(&self, other: &Self) -> Self;

    /// 转置（交换维度）
    fn transpose(&self, axes: Option<&[usize]>) -> Self;

    /// 沿指定轴求和
    fn sum(&self, axes: Option<&[usize]>) -> Self;

    /// 沿指定轴求均值
    fn mean(&self, axes: Option<&[usize]>) -> Self;

    /// 逐元素乘以标量
    fn mul_scalar(&self, scalar: f64) -> Self;

    /// 逐元素加上标量
    fn add_scalar(&self, scalar: f64) -> Self;

    /// 逐元素应用函数
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync;

    /// 重塑张量形状
    fn reshape(&self, new_shape: &[usize]) -> Self;

    /// 切片操作
    fn slice(&self, axes: &[usize], ranges: &[core::ops::Range<usize>]) -> Self;

    /// 拼接两个张量
    fn concat(&self, other: &Self, axis: usize) -> Self;

    /// 获取最大值
    fn max(&self) -> f64;

    /// 获取最小值
    fn min(&self) -> f64;

    /// 获取 L2 范数
    fn norm(&self) -> f64;

    /// 归一化到单位范数
    fn normalize(&self) -> Self;
}

/// 稀疏张量操作 trait
pub trait SparseTensorOps: Clone + Send + Sync + TensorBase {
    /// 获取非零元素数量
    fn nnz(&self) -> usize;

    /// 获取稀疏度（非零元素比例）
    fn sparsity(&self) -> f64 {
        let total = self.numel();
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total as f64)
        }
    }

    /// 获取 COO 格式视图
    fn coo(&self) -> COOView<'_>;

    /// 获取行索引
    fn row_indices(&self) -> &[usize];

    /// 获取列索引
    fn col_indices(&self) -> &[usize];

    /// 获取非零值（作为 DenseTensor）
    fn values(&self) -> &crate::tensor::dense::DenseTensor;
}

/// COO 格式视图：稀疏张量的只读视图
#[derive(Debug, Clone)]
pub struct COOView<'a> {
    /// 行索引
    pub row_indices: &'a [usize],
    /// 列索引
    pub col_indices: &'a [usize],
    /// 非零值
    pub values: &'a [f64],
    /// 张量形状
    pub shape: [usize; 2],
}

impl<'a> COOView<'a> {
    /// 创建新的 COO 视图
    pub fn new(
        row_indices: &'a [usize],
        col_indices: &'a [usize],
        values: &'a [f64],
        shape: [usize; 2],
    ) -> Self {
        Self {
            row_indices,
            col_indices,
            values,
            shape,
        }
    }

    /// 获取非零元素数量
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// 迭代所有非零元素
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
    }
}
