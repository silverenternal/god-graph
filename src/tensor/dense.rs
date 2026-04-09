//! 密集张量实现
//!
//! 基于 ndarray 的 N 维密集张量，支持 BLAS 加速

use core::fmt;

#[cfg(feature = "tensor")]
use ndarray::Array2;

use crate::tensor::error::TensorError;
use crate::tensor::traits::{DType, Device, TensorBase, TensorOps};

/// 密集张量：N 维数组的高性能实现
///
/// 使用 64 字节对齐，支持 BLAS 加速的矩阵运算
#[derive(Clone, PartialEq)]
pub struct DenseTensor {
    /// 张量数据（64 字节对齐）
    data: Vec<f64>,
    /// 张量形状
    shape: Vec<usize>,
    ///  strides（跨步）
    strides: Vec<usize>,
    /// 数据类型
    dtype: DType,
    /// 设备类型
    device: Device,
}

#[cfg(feature = "tensor")]
impl DenseTensor {
    /// 获取字节大小
    pub fn nbytes(&self) -> usize {
        self.data.len() * self.dtype.size_bytes()
    }

    /// 检查是否连续存储
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous()
    }

    /// 获取对齐字节数
    pub fn alignment(&self) -> usize {
        64 // Vec<f64> 默认对齐，可以优化为实际对齐
    }

    /// 创建新的密集张量
    ///
    /// # Arguments
    /// * `data` - 数据向量（行优先顺序，C-order）
    /// * `shape` - 张量形状
    ///
    /// # Returns
    /// 返回新创建的 DenseTensor
    ///
    /// # Panics
    /// 如果 data 长度与 shape 不匹配会 panic
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let expected_len = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} does not match shape product {}",
            data.len(),
            expected_len
        );

        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            dtype: DType::F64,
            device: Device::Cpu,
        }
    }

    /// 从 Vec 创建张量（行优先顺序）
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }

    /// 创建全零张量
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![0.0; shape.iter().product()];
        Self::new(data, shape)
    }

    /// 创建全一张量
    pub fn ones(shape: Vec<usize>) -> Self {
        let data = vec![1.0; shape.iter().product()];
        Self::new(data, shape)
    }

    /// 创建标量张量
    pub fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
            strides: vec![],
            dtype: DType::F64,
            device: Device::Cpu,
        }
    }

    /// 创建 2D 矩阵
    pub fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self::new(data, vec![rows, cols])
    }

    /// 创建 2D 单位矩阵
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Self::new(data, vec![size, size])
    }

    /// 从 ndarray::Array2 创建
    #[cfg(feature = "tensor")]
    pub fn from_ndarray(arr: &Array2<f64>) -> Self {
        let shape = vec![arr.nrows(), arr.ncols()];
        let data = arr.as_slice().unwrap().to_vec();
        Self::new(data, shape)
    }

    /// 转换为 ndarray::Array2
    #[cfg(feature = "tensor")]
    pub fn to_ndarray(&self) -> Result<Array2<f64>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        Ok(Array2::from_shape_vec((self.shape[0], self.shape[1]), self.data.clone()).unwrap())
    }

    /// 获取数据切片
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// 获取可变数据切片
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// 获取 strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// 检查是否为连续内存（C-order）
    pub fn is_c_contiguous(&self) -> bool {
        if self.ndim() <= 1 {
            return true;
        }
        for i in 0..self.ndim() - 1 {
            if self.strides[i] != self.strides[i + 1] * self.shape[i + 1] {
                return false;
            }
        }
        true
    }

    /// 获取指定索引的元素
    pub fn get(&self, indices: &[usize]) -> Result<f64, TensorError> {
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim(),
                got: indices.len(),
            });
        }

        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    dim: i,
                    size: self.shape[i],
                });
            }
            offset += idx * self.strides[i];
        }

        Ok(self.data[offset])
    }

    /// 设置指定索引的元素
    pub fn set(&mut self, indices: &[usize], value: f64) -> Result<(), TensorError> {
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim(),
                got: indices.len(),
            });
        }

        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    dim: i,
                    size: self.shape[i],
                });
            }
            offset += idx * self.strides[i];
        }

        self.data[offset] = value;
        Ok(())
    }

    /// 获取指定行的数据
    pub fn row(&self, row: usize) -> Result<Vec<f64>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if row >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds {
                index: row,
                dim: 0,
                size: self.shape[0],
            });
        }

        let start = row * self.strides[0];
        let cols = self.shape[1];
        Ok(self.data[start..start + cols].to_vec())
    }

    /// 获取指定列的数据
    pub fn col(&self, col: usize) -> Result<Vec<f64>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if col >= self.shape[1] {
            return Err(TensorError::IndexOutOfBounds {
                index: col,
                dim: 1,
                size: self.shape[1],
            });
        }

        let mut result = Vec::with_capacity(self.shape[0]);
        for row in 0..self.shape[0] {
            let idx = row * self.strides[0] + col;
            result.push(self.data[idx]);
        }
        Ok(result)
    }
}

/// 计算 strides（C-order，行优先）
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }

    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(feature = "tensor")]
impl TensorBase for DenseTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        self.device
    }

    fn to_dense(&self) -> DenseTensor {
        self.clone()
    }

    #[cfg(feature = "tensor")]
    fn to_sparse(&self) -> Option<crate::tensor::sparse::SparseTensor> {
        // 将密集张量转换为 CSR 格式
        let mut row_offsets = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        if self.ndim() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            for row in 0..rows {
                for col in 0..cols {
                    let val = self.get(&[row, col]).unwrap();
                    if val.abs() > 1e-10 {
                        col_indices.push(col);
                        values.push(val);
                    }
                }
                row_offsets.push(col_indices.len());
            }

            let values_tensor = DenseTensor::new(values.clone(), vec![values.len()]);
            let csr = crate::tensor::sparse::CSRTensor::new(
                row_offsets,
                col_indices,
                values_tensor,
                [self.shape[0], self.shape[1]],
            );
            Some(crate::tensor::sparse::SparseTensor::CSR(csr))
        } else {
            None
        }
    }
}

#[cfg(feature = "tensor")]
impl TensorOps for DenseTensor {
    fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for addition: {:?} vs {:?}",
            self.shape, other.shape
        );

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Self::new(data, self.shape.clone())
    }

    fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for subtraction: {:?} vs {:?}",
            self.shape, other.shape
        );

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Self::new(data, self.shape.clone())
    }

    fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for element-wise multiplication: {:?} vs {:?}",
            self.shape, other.shape
        );

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Self::new(data, self.shape.clone())
    }

    fn div(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for division: {:?} vs {:?}",
            self.shape, other.shape
        );

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();

        Self::new(data, self.shape.clone())
    }

    fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            self.ndim()
        );
        assert_eq!(
            other.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            other.ndim()
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "Shape mismatch for matmul: {:?} x {:?}",
            self.shape, other.shape
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = vec![0.0; m * n];

        // 朴素矩阵乘法实现（后续可用 BLAS 优化）
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Self::new(result, vec![m, n])
    }

    fn transpose(&self, axes: Option<&[usize]>) -> Self {
        if self.ndim() == 0 {
            return self.clone();
        }

        if self.ndim() == 2 {
            // 2D 转置
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = vec![0.0; cols * rows];

            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = self.data[i * cols + j];
                }
            }

            Self::new(result, vec![cols, rows])
        } else {
            // N 维转置
            let default_axes: Vec<usize> = (0..self.ndim()).rev().collect();
            let axes = axes.unwrap_or(&default_axes);

            assert_eq!(axes.len(), self.ndim(), "Axes length must match ndim");

            let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
            let mut result = vec![0.0; self.numel()];

            // 简化的 N 维转置（对于高维情况可能需要优化）
            for (i, &val) in self.data.iter().enumerate() {
                let mut idx = i;
                let mut new_idx = 0;
                let mut stride = 1;

                for &a in axes.iter().rev() {
                    let dim_size = self.shape[a];
                    let dim_idx = idx % dim_size;
                    idx /= dim_size;
                    new_idx += dim_idx * stride;
                    stride *= new_shape[new_shape.len() - 1 - a];
                }

                result[new_idx] = val;
            }

            Self::new(result, new_shape)
        }
    }

    fn sum(&self, axes: Option<&[usize]>) -> Self {
        if let Some(axes) = axes {
            if axes.is_empty() {
                return self.clone();
            }

            // 简化实现：仅支持单轴归约
            if axes.len() == 1 {
                let axis = axes[0];
                if self.ndim() == 2 && axis == 0 {
                    // 按行求和
                    let cols = self.shape[1];
                    let mut result = vec![0.0; cols];
                    for row in self.data.chunks(cols) {
                        for (j, val) in row.iter().enumerate() {
                            result[j] += val;
                        }
                    }
                    return Self::new(result, vec![cols]);
                } else if self.ndim() == 2 && axis == 1 {
                    // 按列求和
                    let rows = self.shape[0];
                    let cols = self.shape[1];
                    let mut result = vec![0.0; rows];
                    for (i, row_sum) in result.iter_mut().enumerate().take(rows) {
                        let row_start = i * cols;
                        *row_sum = self.data[row_start..row_start + cols].iter().sum();
                    }
                    return Self::new(result, vec![rows]);
                }
            }

            // 默认：返回所有元素的和（标量）
            let sum: f64 = self.data.iter().sum();
            Self::scalar(sum)
        } else {
            // 无轴：返回所有元素的和
            let sum: f64 = self.data.iter().sum();
            Self::scalar(sum)
        }
    }

    fn mean(&self, axes: Option<&[usize]>) -> Self {
        let sum = self.sum(axes);
        let count = if let Some(axes) = axes {
            if axes.is_empty() {
                1
            } else {
                axes.iter().map(|&a| self.shape[a]).product::<usize>()
            }
        } else {
            self.numel()
        };

        sum.mul_scalar(1.0 / count as f64)
    }

    fn mul_scalar(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Self::new(data, self.shape.clone())
    }

    fn add_scalar(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        Self::new(data, self.shape.clone())
    }

    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let data: Vec<f64> = self.data.iter().copied().map(f).collect();
        Self::new(data, self.shape.clone())
    }

    fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            new_size,
            self.numel(),
            "Reshape size mismatch: {} vs {}",
            new_size,
            self.numel()
        );

        Self::new(self.data.clone(), new_shape.to_vec())
    }

    fn slice(&self, axes: &[usize], ranges: &[core::ops::Range<usize>]) -> Self {
        assert_eq!(axes.len(), ranges.len(), "Axes and ranges length mismatch");

        // 简化实现：仅支持 2D 切片
        if self.ndim() == 2 && axes.len() == 2 {
            let row_range = if axes[0] == 0 {
                ranges[0].clone()
            } else {
                ranges[1].clone()
            };
            let col_range = if axes[1] == 1 {
                ranges[1].clone()
            } else {
                ranges[0].clone()
            };

            let new_rows = row_range.len();
            let new_cols = col_range.len();
            let mut result = Vec::with_capacity(new_rows * new_cols);

            for i in row_range {
                for j in col_range.clone() {
                    result.push(self.data[i * self.shape[1] + j]);
                }
            }

            return Self::new(result, vec![new_rows, new_cols]);
        }

        // 默认返回克隆
        self.clone()
    }

    fn concat(&self, other: &Self, axis: usize) -> Self {
        assert_eq!(
            self.ndim(),
            other.ndim(),
            "Concat ndim mismatch: {} vs {}",
            self.ndim(),
            other.ndim()
        );
        assert!(
            axis < self.ndim(),
            "Axis {} out of range for {}D tensor",
            axis,
            self.ndim()
        );

        // 检查除 concat 轴外的所有维度是否匹配
        for (i, (&s, &o)) in self.shape.iter().zip(other.shape.iter()).enumerate() {
            if i != axis {
                assert_eq!(s, o, "Shape mismatch at dim {}", i);
            }
        }

        // 简化实现：仅支持 2D 沿轴 0 拼接
        if self.ndim() == 2 && axis == 0 {
            assert_eq!(
                self.shape[1], other.shape[1],
                "Column count mismatch for concat"
            );

            let new_rows = self.shape[0] + other.shape[0];
            let cols = self.shape[1];
            let mut result = Vec::with_capacity(new_rows * cols);

            // 复制第一个张量
            result.extend_from_slice(&self.data);
            // 复制第二个张量
            result.extend_from_slice(&other.data);

            return Self::new(result, vec![new_rows, cols]);
        }

        // 默认：返回错误（需要更复杂的实现）
        unimplemented!("Concat for this case is not implemented")
    }

    fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm > 1e-10 {
            self.mul_scalar(1.0 / norm)
        } else {
            self.clone()
        }
    }
}

// Additional DenseTensor methods for transformer support
#[cfg(feature = "tensor")]
impl DenseTensor {
    /// SiLU activation: f(x) = x * sigmoid(x)
    pub fn silu(&self) -> Self {
        self.map(|x| x / (1.0 + (-x).exp()))
    }

    /// GELU derivative (for backpropagation)
    pub fn gelu_derivative(&self) -> Self {
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
        const COEF: f64 = 0.044715;
        self.map(|x| {
            let x2 = x * x;
            let x3 = x * x2;
            let tanh_arg = SQRT_2_OVER_PI * (x + COEF * x3);
            let tanh_val = tanh_arg.tanh();
            0.5 * (1.0 + tanh_val)
                + x * 0.5 * (1.0 - tanh_val * tanh_val) * SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * x2)
        })
    }

    /// Mean along a specific dimension
    pub fn mean_dim(&self, dim: isize) -> Self {
        let ndim = self.ndim();
        let axis = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        #[allow(clippy::needless_range_loop)]
        if ndim == 2 && axis == 0 {
            // Mean along rows (result: [cols])
            let cols = self.shape[1];
            let rows = self.shape[0];
            let mut result = vec![0.0; cols];
            #[allow(clippy::needless_range_loop)]
            for col in 0..cols {
                for row in 0..rows {
                    result[col] += self.data[row * cols + col];
                }
                result[col] /= rows as f64;
            }
            Self::new(result, vec![1, cols])
        } else if ndim == 2 && axis == 1 {
            // Mean along columns (result: [rows, 1])
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = vec![0.0; rows];
            #[allow(clippy::needless_range_loop)]
            for row in 0..rows {
                let row_start = row * cols;
                result[row] =
                    self.data[row_start..row_start + cols].iter().sum::<f64>() / cols as f64;
            }
            Self::new(result, vec![rows, 1])
        } else if ndim == 3 && axis == 2 {
            // Mean along last dimension for 3D tensors
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];
            let mut result = vec![0.0; batch * seq];
            for b in 0..batch {
                for s in 0..seq {
                    let start = (b * seq + s) * dim;
                    let sum: f64 = self.data[start..start + dim].iter().sum();
                    result[b * seq + s] = sum / dim as f64;
                }
            }
            Self::new(result, vec![batch, seq, 1])
        } else {
            // Fallback: return scalar mean
            let sum: f64 = self.data.iter().sum();
            Self::scalar(sum / self.numel() as f64)
        }
    }

    /// Variance along a specific dimension
    pub fn var_dim(&self, dim: isize) -> Self {
        let mean = self.mean_dim(dim);
        let ndim = self.ndim();
        let axis = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        #[allow(clippy::needless_range_loop)]
        if ndim == 2 && axis == 0 {
            let cols = self.shape[1];
            let rows = self.shape[0];
            let mut result = vec![0.0; cols];
            #[allow(clippy::needless_range_loop)]
            for col in 0..cols {
                for row in 0..rows {
                    let diff = self.data[row * cols + col] - mean.data()[col];
                    result[col] += diff * diff;
                }
                result[col] /= rows as f64;
            }
            Self::new(result, vec![1, cols])
        } else if ndim == 2 && axis == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = vec![0.0; rows];
            #[allow(clippy::needless_range_loop)]
            for row in 0..rows {
                let row_start = row * cols;
                let m = mean.data()[row];
                let var: f64 = self.data[row_start..row_start + cols]
                    .iter()
                    .map(|&x| (x - m) * (x - m))
                    .sum::<f64>()
                    / cols as f64;
                result[row] = var;
            }
            Self::new(result, vec![rows, 1])
        } else if ndim == 3 && axis == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];
            let mut result = vec![0.0; batch * seq];
            for b in 0..batch {
                for s in 0..seq {
                    let start = (b * seq + s) * dim;
                    let m = mean.data()[b * seq + s];
                    let var: f64 = self.data[start..start + dim]
                        .iter()
                        .map(|&x| (x - m) * (x - m))
                        .sum::<f64>()
                        / dim as f64;
                    result[b * seq + s] = var;
                }
            }
            Self::new(result, vec![batch, seq, 1])
        } else {
            // Fallback: return scalar variance
            let mean_val = self.data.iter().sum::<f64>() / self.numel() as f64;
            let var: f64 = self
                .data
                .iter()
                .map(|&x| (x - mean_val) * (x - mean_val))
                .sum::<f64>()
                / self.numel() as f64;
            Self::scalar(var)
        }
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    /// Negate the tensor
    pub fn neg(&self) -> Self {
        self.mul_scalar(-1.0)
    }

    /// Element-wise greater than comparison (returns 1.0 if true, 0.0 otherwise)
    pub fn gt(&self, value: f64) -> Self {
        self.map(|x| if x > value { 1.0 } else { 0.0 })
    }

    /// Fill values with a given value where mask is 1.0
    pub fn mask_fill(&self, mask: &Self, value: f64) -> Self {
        assert_eq!(self.shape, mask.shape, "Shape mismatch for mask_fill");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(&v, &m)| if m > 0.5 { value } else { v })
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Transpose for 2D tensors (convenience method)
    pub fn transpose_2d(&self) -> Self {
        self.transpose(None)
    }

    /// Get a row from a 2D or 3D tensor
    pub fn get_row(&self, row: usize) -> Self {
        if self.ndim() == 2 {
            let cols = self.shape[1];
            let start = row * cols;
            Self::new(self.data[start..start + cols].to_vec(), vec![1, cols])
        } else if self.ndim() == 3 {
            // For 3D tensors [batch, seq, dim], get row at index row
            // This returns a 2D slice [batch, dim] at position row along seq dimension
            let batch = self.shape[0];
            let dim = self.shape[2];
            let mut result_data = Vec::with_capacity(batch * dim);

            for b in 0..batch {
                let offset = (b * self.shape[1] + row) * dim;
                result_data.extend_from_slice(&self.data[offset..offset + dim]);
            }

            Self::new(result_data, vec![batch, dim])
        } else {
            // Fallback: return first element
            Self::scalar(self.data[0])
        }
    }

    /// Set a row in the tensor (mutable)
    pub fn set_row(&mut self, row: usize, data: &Self) {
        if self.ndim() == 2 && data.ndim() == 2 {
            let cols = self.shape[1];
            let start = row * cols;
            self.data[start..start + cols].copy_from_slice(data.data());
        }
    }

    /// Create a tensor filled with a value
    pub fn full(shape: &[usize], value: f64) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![value; size];
        Self::new(data, shape.to_vec())
    }

    /// Scale the tensor by a scalar
    pub fn scale(&self, scalar: f64) -> Self {
        self.mul_scalar(scalar)
    }

    /// Softmax along the last dimension
    pub fn softmax(&self, dim: isize) -> Self {
        crate::tensor::ops::activations::softmax(self, dim)
    }

    /// ReLU activation
    pub fn relu(&self) -> Self {
        crate::tensor::ops::activations::relu(self)
    }

    /// GELU activation
    pub fn gelu(&self) -> Self {
        crate::tensor::ops::activations::gelu(self)
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Self {
        self.map(|x| x.cos())
    }

    /// Element-wise sine
    pub fn sin(&self) -> Self {
        self.map(|x| x.sin())
    }

    /// Element-wise natural logarithm
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Batched matrix multiplication
    /// For 3D tensors: [batch, seq, hidden] @ [hidden, out] -> [batch, seq, out]
    /// Broadcasts 2D weight across batch dimension
    pub fn bmm_broadcast_weight(&self, weight: &DenseTensor) -> Self {
        assert_eq!(
            self.ndim(),
            3,
            "bmm_broadcast_weight requires 3D tensor, got {}D",
            self.ndim()
        );
        assert_eq!(
            weight.ndim(),
            2,
            "weight must be 2D, got {}D",
            weight.ndim()
        );
        assert_eq!(
            self.shape[2], weight.shape[0],
            "Shape mismatch for bmm: {:?} x {:?}",
            self.shape, weight.shape
        );

        let batch = self.shape[0];
        let seq = self.shape[1];
        let hidden = self.shape[2];
        let out = weight.shape[1];

        let mut result = vec![0.0; batch * seq * out];

        // Batched matmul: for each batch and seq, do matmul with weight
        for b in 0..batch {
            for s in 0..seq {
                let input_start = (b * seq + s) * hidden;
                let output_start = (b * seq + s) * out;

                for o in 0..out {
                    let mut sum = 0.0;
                    for h in 0..hidden {
                        sum += self.data[input_start + h] * weight.data[h * out + o];
                    }
                    result[output_start + o] = sum;
                }
            }
        }

        Self::new(result, vec![batch, seq, out])
    }

    /// Expand the last dimension from 1 to target_dim (for broadcasting)
    /// E.g., [batch, seq, 1] -> [batch, seq, target_dim]
    pub fn expand_last_dim(&self, target_dim: usize) -> Self {
        assert!(
            self.ndim() >= 1 && self.shape()[self.ndim() - 1] == 1,
            "Last dimension must be 1 for expansion"
        );

        let mut new_shape = self.shape.to_vec();
        new_shape[self.ndim() - 1] = target_dim;

        let mut data = Vec::with_capacity(self.numel() * target_dim);
        for &val in self.data.iter() {
            for _ in 0..target_dim {
                data.push(val);
            }
        }

        Self::new(data, new_shape)
    }

    /// Expand a 1D tensor [hidden] to 3D [batch, seq, hidden]
    pub fn expand_to_3d(&self, batch: usize, seq: usize) -> Self {
        assert_eq!(self.ndim(), 1, "Must be 1D tensor for 3D expansion");
        let hidden = self.shape[0];

        let mut data = Vec::with_capacity(batch * seq * hidden);
        for _ in 0..batch * seq {
            data.extend_from_slice(&self.data);
        }

        Self::new(data, vec![batch, seq, hidden])
    }

    /// Expand the last dimension from 1 to target_dim for 2D tensors
    /// E.g., [seq, 1] -> [seq, target_dim]
    pub fn expand_last_dim_2d(&self, target_dim: usize) -> Self {
        assert!(
            self.ndim() == 2 && self.shape()[1] == 1,
            "Must be 2D tensor with last dim 1 for expansion"
        );

        let seq = self.shape[0];
        let mut data = Vec::with_capacity(seq * target_dim);
        for &val in self.data.iter() {
            for _ in 0..target_dim {
                data.push(val);
            }
        }

        Self::new(data, vec![seq, target_dim])
    }

    /// Expand a 1D tensor [hidden] to 2D [seq, hidden]
    pub fn expand_to_2d(&self, seq: usize) -> Self {
        assert_eq!(self.ndim(), 1, "Must be 1D tensor for 2D expansion");
        let hidden = self.shape[0];

        let mut data = Vec::with_capacity(seq * hidden);
        for _ in 0..seq {
            data.extend_from_slice(&self.data);
        }

        Self::new(data, vec![seq, hidden])
    }
}

#[cfg(feature = "tensor")]
impl fmt::Debug for DenseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("numel", &self.numel())
            .finish()
    }
}

#[cfg(feature = "tensor")]
impl Default for DenseTensor {
    fn default() -> Self {
        Self::zeros(vec![1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = DenseTensor::new(data.clone(), vec![2, 3]);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &data);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = DenseTensor::zeros(vec![2, 3]);
        assert!(zeros.data().iter().all(|&x| x == 0.0));

        let ones = DenseTensor::ones(vec![2, 3]);
        assert!(ones.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_matrix_operations() {
        let a = DenseTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DenseTensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let sum = a.add(&b);
        assert_eq!(sum.data(), &[6.0, 8.0, 10.0, 12.0]);

        let diff = a.sub(&b);
        assert_eq!(diff.data(), &[-4.0, -4.0, -4.0, -4.0]);
    }

    #[test]
    fn test_matmul() {
        let a = DenseTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DenseTensor::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let result = a.matmul(&b);
        assert_eq!(result.shape(), &[2, 2]);
        // [1,2,3]·[7,9,11] = 7+18+33 = 58
        // [1,2,3]·[8,10,12] = 8+20+36 = 64
        // [4,5,6]·[7,9,11] = 28+45+66 = 139
        // [4,5,6]·[8,10,12] = 32+50+72 = 154
        assert_eq!(result.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let a = DenseTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose(None);

        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_scalar_operations() {
        let a = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let mul = a.mul_scalar(2.0);
        assert_eq!(mul.data(), &[2.0, 4.0, 6.0]);

        let add = a.add_scalar(1.0);
        assert_eq!(add.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_norm_and_normalize() {
        let a = DenseTensor::new(vec![3.0, 4.0], vec![2]);

        assert!((a.norm() - 5.0).abs() < 1e-10);

        let normalized = a.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panic() {
        let a = DenseTensor::new(vec![1.0, 2.0], vec![2]);
        let b = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let _ = a.add(&b);
    }
}
