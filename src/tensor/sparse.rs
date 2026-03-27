//! 稀疏张量实现
//!
//! 提供 COO（Coordinate）和 CSR（Compressed Sparse Row）格式
//! 用于高效的图神经网络计算

use core::fmt;

use crate::tensor::traits::{TensorBase, DType, Device, COOView, SparseTensorOps};
use crate::tensor::dense::DenseTensor;
use crate::tensor::error::TensorError;

/// COO（Coordinate）格式稀疏张量
#[derive(Debug, Clone)]
pub struct COOTensor {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: DenseTensor,
    shape: [usize; 2],
}

#[cfg(feature = "tensor-sparse")]
impl COOTensor {
    /// 创建新的 COO 张量
    pub fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, values: DenseTensor, shape: [usize; 2]) -> Self {
        assert_eq!(
            row_indices.len(),
            col_indices.len(),
            "Row and column indices must have the same length"
        );
        assert_eq!(
            row_indices.len(),
            values.numel(),
            "Indices length must match values length"
        );
        Self {
            row_indices,
            col_indices,
            values,
            shape,
        }
    }

    /// 获取非零元素数量
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }

    /// 从边列表创建 COO 张量
    pub fn from_edges(edges: &[(usize, usize, f64)], shape: [usize; 2]) -> Self {
        let row_indices: Vec<usize> = edges.iter().map(|&(r, _, _)| r).collect();
        let col_indices: Vec<usize> = edges.iter().map(|&(_, c, _)| c).collect();
        let values_data: Vec<f64> = edges.iter().map(|&(_, _, v)| v).collect();
        let values = DenseTensor::new(values_data, vec![edges.len()]);
        Self::new(row_indices, col_indices, values, shape)
    }

    /// 获取行索引
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// 获取列索引
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// 获取值
    pub fn values(&self) -> &DenseTensor {
        &self.values
    }
}

/// CSR（Compressed Sparse Row）格式稀疏张量
#[derive(Debug, Clone)]
pub struct CSRTensor {
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: DenseTensor,
    shape: [usize; 2],
}

#[cfg(feature = "tensor-sparse")]
impl CSRTensor {
    /// 创建新的 CSR 张量
    pub fn new(row_offsets: Vec<usize>, col_indices: Vec<usize>, values: DenseTensor, shape: [usize; 2]) -> Self {
        assert_eq!(
            col_indices.len(),
            values.numel(),
            "Column indices length must match values length"
        );
        Self {
            row_offsets,
            col_indices,
            values,
            shape,
        }
    }

    /// 获取非零元素数量
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }

    /// 从 COO 张量转换为 CSR
    pub fn from_coo(coo: &COOTensor) -> Self {
        let mut row_offsets = vec![0; coo.shape[0] + 1];
        let mut col_indices = vec![0; coo.nnz()];
        let mut values_data = vec![0.0; coo.nnz()];

        // 计算每行的非零元素数量
        for &row in &coo.row_indices {
            row_offsets[row + 1] += 1;
        }

        // 转换为偏移量
        for i in 1..row_offsets.len() {
            row_offsets[i] += row_offsets[i - 1];
        }

        // 填充列索引和值
        let mut row_pos = row_offsets.clone();
        for (i, (&row, &col)) in coo.row_indices.iter().zip(coo.col_indices.iter()).enumerate() {
            let pos = row_pos[row];
            col_indices[pos] = col;
            values_data[pos] = coo.values.data()[i];
            row_pos[row] += 1;
        }

        let values = DenseTensor::new(values_data, vec![coo.nnz()]);
        Self::new(row_offsets, col_indices, values, coo.shape)
    }

    /// 获取行偏移量
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// 获取列索引
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// 获取值
    pub fn values(&self) -> &DenseTensor {
        &self.values
    }
}

/// 稀疏张量枚举：支持多种稀疏格式
#[derive(Clone)]
pub enum SparseTensor {
    /// COO（Coordinate）格式
    COO(COOTensor),
    /// CSR（Compressed Sparse Row）格式
    CSR(CSRTensor),
}

#[cfg(feature = "tensor-sparse")]
impl SparseTensor {
    /// 创建 COO 格式稀疏张量
    pub fn coo(row_indices: Vec<usize>, col_indices: Vec<usize>, values: DenseTensor, shape: [usize; 2]) -> Self {
        SparseTensor::COO(COOTensor::new(row_indices, col_indices, values, shape))
    }

    /// 创建 CSR 格式稀疏张量
    pub fn csr(row_offsets: Vec<usize>, col_indices: Vec<usize>, values: DenseTensor, shape: [usize; 2]) -> Self {
        SparseTensor::CSR(CSRTensor::new(row_offsets, col_indices, values, shape))
    }

    /// 获取非零元素数量
    pub fn nnz(&self) -> usize {
        match self {
            SparseTensor::COO(coo) => coo.nnz(),
            SparseTensor::CSR(csr) => csr.nnz(),
        }
    }

    /// 转换为 CSR 格式
    pub fn to_csr(&self) -> CSRTensor {
        match self {
            SparseTensor::COO(coo) => CSRTensor::from_coo(coo),
            SparseTensor::CSR(csr) => csr.clone(),
        }
    }

    /// 转换为 COO 格式
    pub fn to_coo(&self) -> COOTensor {
        match self {
            SparseTensor::COO(coo) => coo.clone(),
            SparseTensor::CSR(csr) => {
                // CSR 转 COO
                let mut row_indices = Vec::with_capacity(csr.nnz());
                let col_indices = csr.col_indices.clone();
                let mut values_data = Vec::with_capacity(csr.nnz());

                for row in 0..csr.shape[0] {
                    let start = csr.row_offsets[row];
                    let end = csr.row_offsets[row + 1];
                    for _ in start..end {
                        row_indices.push(row);
                    }
                    for i in start..end {
                        values_data.push(csr.values.data()[i]);
                    }
                }

                let values = DenseTensor::new(values_data, vec![csr.nnz()]);
                COOTensor::new(row_indices, col_indices, values, csr.shape)
            }
        }
    }

    /// 获取 COO 视图
    pub fn coo_view(&self) -> COOView<'_> {
        match self {
            SparseTensor::COO(coo) => {
                COOView::new(&coo.row_indices, &coo.col_indices, coo.values.data(), coo.shape)
            }
            SparseTensor::CSR(_) => {
                // For CSR, we need to convert to COO first, but we can't return a view
                // So we return an empty view as a workaround (this is a limitation)
                COOView::new(&[], &[], &[], [0, 0])
            }
        }
    }

    /// 从边列表创建稀疏张量（COO 格式）
    pub fn from_edges(edges: &[(usize, usize, f64)], shape: [usize; 2]) -> Self {
        SparseTensor::COO(COOTensor::from_edges(edges, shape))
    }

    /// 稀疏矩阵 - 稠密向量乘法
    pub fn spmv(&self, x: &DenseTensor) -> Result<DenseTensor, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }

        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        if x.shape() != [cols] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![cols],
                got: x.shape().to_vec(),
            });
        }

        let mut result = vec![0.0; rows];
        let coo = self.to_coo();

        for (i, (&row, &col)) in coo.row_indices.iter().zip(coo.col_indices.iter()).enumerate() {
            let val = coo.values.data()[i];
            let x_val = x.data()[col];
            result[row] += val * x_val;
        }

        Ok(DenseTensor::new(result, vec![rows]))
    }

    /// 稀疏矩阵 - 稀疏矩阵乘法
    pub fn spmm(&self, other: &Self) -> Result<Self, TensorError> {
        let shape_a = self.shape();
        let shape_b = other.shape();
        let (rows_a, cols_a) = (shape_a[0], shape_a[1]);
        let (rows_b, cols_b) = (shape_b[0], shape_b[1]);

        if cols_a != rows_b {
            return Err(TensorError::ShapeMismatch {
                expected: vec![cols_a],
                got: vec![rows_b],
            });
        }

        // 转换为 COO 进行乘法
        let coo_a = self.to_coo();
        let coo_b = other.to_coo();

        // 使用哈希表累加结果
        use std::collections::HashMap;
        let mut result_map: HashMap<(usize, usize), f64> = HashMap::new();

        for (i, (&row_a, &col_a)) in coo_a.row_indices.iter().zip(coo_a.col_indices.iter()).enumerate() {
            let val_a = coo_a.values.data()[i];
            for (j, (&row_b, &col_b)) in coo_b.row_indices.iter().zip(coo_b.col_indices.iter()).enumerate() {
                if col_a == row_b {
                    let val_b = coo_b.values.data()[j];
                    *result_map.entry((row_a, col_b)).or_insert(0.0) += val_a * val_b;
                }
            }
        }

        // 转换回 COO 格式
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values_data = Vec::new();

        let mut entries: Vec<_> = result_map.into_iter().collect();
        entries.sort_by_key(|&(pos, _)| pos);

        for ((row, col), val) in entries {
            row_indices.push(row);
            col_indices.push(col);
            values_data.push(val);
        }

        let values = DenseTensor::new(values_data.clone(), vec![values_data.len()]);
        Ok(SparseTensor::COO(COOTensor::new(row_indices, col_indices, values, [rows_a, cols_b])))
    }
}

#[cfg(feature = "tensor-sparse")]
impl SparseTensorOps for SparseTensor {
    fn nnz(&self) -> usize {
        match self {
            SparseTensor::COO(coo) => coo.nnz(),
            SparseTensor::CSR(csr) => csr.nnz(),
        }
    }

    fn coo(&self) -> COOView<'_> {
        self.coo_view()
    }

    fn row_indices(&self) -> &[usize] {
        match self {
            SparseTensor::COO(coo) => coo.row_indices(),
            SparseTensor::CSR(_) => &[],
        }
    }

    fn col_indices(&self) -> &[usize] {
        match self {
            SparseTensor::COO(coo) => coo.col_indices(),
            SparseTensor::CSR(csr) => csr.col_indices(),
        }
    }

    fn values(&self) -> &DenseTensor {
        match self {
            SparseTensor::COO(coo) => coo.values(),
            SparseTensor::CSR(csr) => csr.values(),
        }
    }
}

#[cfg(feature = "tensor-sparse")]
impl TensorBase for SparseTensor {
    fn shape(&self) -> &[usize] {
        match self {
            SparseTensor::COO(coo) => &coo.shape[..],
            SparseTensor::CSR(csr) => &csr.shape[..],
        }
    }

    fn dtype(&self) -> DType {
        DType::F64
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn to_dense(&self) -> DenseTensor {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        let mut data = vec![0.0; rows * cols];
        let coo = self.to_coo();

        for (i, (&row, &col)) in coo.row_indices.iter().zip(coo.col_indices.iter()).enumerate() {
            let val = coo.values.data()[i];
            data[row * cols + col] = val;
        }

        DenseTensor::new(data, vec![rows, cols])
    }

    #[cfg(feature = "tensor-sparse")]
    fn to_sparse(&self) -> Option<SparseTensor> {
        Some(self.clone())
    }
}

#[cfg(feature = "tensor-sparse")]
impl fmt::Debug for SparseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        f.debug_struct("SparseTensor")
            .field("shape", &[rows, cols])
            .field("nnz", &self.nnz())
            .field("sparsity", &self.sparsity())
            .finish()
    }
}

/// COO 张量实现
impl COOTensor {
    /// 获取形状
    pub fn shape_array(&self) -> [usize; 2] {
        self.shape
    }
}

/// CSR 张量实现
impl CSRTensor {
    /// 获取形状
    pub fn shape_array(&self) -> [usize; 2] {
        self.shape
    }

    /// 获取指定行的非零元素
    pub fn row(&self, row: usize) -> Option<Vec<(usize, f64)>> {
        if row >= self.shape[0] {
            return None;
        }

        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];

        if start == end {
            return Some(Vec::new());
        }

        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            result.push((self.col_indices[i], self.values.data()[i]));
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_creation() {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, 3.0),
            (2, 0, 4.0),
        ];
        let coo = SparseTensor::from_edges(&edges, [3, 3]);

        assert_eq!(coo.nnz(), 4);
        assert_eq!(coo.shape(), &[3, 3]);
    }

    #[test]
    fn test_coo_to_csr() {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, 3.0),
            (2, 0, 4.0),
        ];
        let coo = SparseTensor::from_edges(&edges, [3, 3]);
        let csr = coo.to_csr();

        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.row_offsets(), &[0, 2, 3, 4]);
    }

    #[test]
    fn test_sparse_dense_conversion() {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, 3.0),
            (2, 0, 4.0),
        ];
        let sparse = SparseTensor::from_edges(&edges, [3, 3]);
        let dense = sparse.to_dense();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(dense.get(&[0, 2]).unwrap(), 2.0);
        assert_eq!(dense.get(&[2, 0]).unwrap(), 4.0);
    }

    #[test]
    fn test_spmv() {
        let edges = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 3.0),
            (1, 1, 4.0),
        ];
        let sparse = SparseTensor::from_edges(&edges, [2, 2]);
        let x = DenseTensor::new(vec![1.0, 2.0], vec![2]);

        let result = sparse.spmv(&x).unwrap();
        // [1,2; 3,4] * [1; 2] = [1*1+2*2; 3*1+4*2] = [5; 11]
        assert_eq!(result.data(), &[5.0, 11.0]);
    }

    #[test]
    fn test_spmm() {
        let edges_a = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 3.0),
            (1, 1, 4.0),
        ];
        let a = SparseTensor::from_edges(&edges_a, [2, 2]);

        let edges_b = vec![
            (0, 0, 5.0),
            (0, 1, 6.0),
            (1, 0, 7.0),
            (1, 1, 8.0),
        ];
        let b = SparseTensor::from_edges(&edges_b, [2, 2]);

        let result = a.spmm(&b).unwrap();
        let result_dense = result.to_dense();

        // [1,2; 3,4] * [5,6; 7,8] = [19,22; 43,50]
        assert_eq!(result_dense.get(&[0, 0]).unwrap(), 19.0);
        assert_eq!(result_dense.get(&[0, 1]).unwrap(), 22.0);
        assert_eq!(result_dense.get(&[1, 0]).unwrap(), 43.0);
        assert_eq!(result_dense.get(&[1, 1]).unwrap(), 50.0);
    }
}
