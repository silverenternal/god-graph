//! Tensor 高级操作
//!
//! 提供额外的 tensor 操作，包括激活函数、归一化等

use crate::tensor::dense::DenseTensor;
use crate::tensor::error::TensorError;
use crate::tensor::traits::{TensorBase, TensorOps};

/// 激活函数实现
pub mod activations {
    use super::*;

    /// ReLU 激活函数：f(x) = max(0, x)
    pub fn relu(tensor: &DenseTensor) -> DenseTensor {
        tensor.map(|x| x.max(0.0))
    }

    /// Sigmoid 激活函数：f(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(tensor: &DenseTensor) -> DenseTensor {
        tensor.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Tanh 激活函数：f(x) = tanh(x)
    pub fn tanh(tensor: &DenseTensor) -> DenseTensor {
        tensor.map(|x| x.tanh())
    }

    /// Leaky ReLU 激活函数：f(x) = x if x > 0 else alpha * x
    pub fn leaky_relu(tensor: &DenseTensor, alpha: f64) -> DenseTensor {
        tensor.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Softmax 函数（沿指定轴）
    pub fn softmax(tensor: &DenseTensor, axis: isize) -> DenseTensor {
        let ndim = tensor.ndim();
        // Handle negative axis
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        if ndim == 1 {
            // 1D 情况：直接计算 softmax
            let max_val = tensor.max();
            let exp_data: Vec<f64> = tensor.data().iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_data.iter().sum();
            let data: Vec<f64> = exp_data.iter().map(|&x| x / sum).collect();
            DenseTensor::new(data, tensor.shape().to_vec())
        } else if ndim == 2 {
            // 2D 情况：按行或按列计算 softmax
            let rows = tensor.shape()[0];
            let cols = tensor.shape()[1];

            if axis == 0 {
                // 按列 softmax
                let mut result = vec![0.0; rows * cols];
                for col in 0..cols {
                    let mut col_data = Vec::with_capacity(rows);
                    for row in 0..rows {
                        col_data.push(tensor.data()[row * cols + col]);
                    }
                    let max_val = col_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_data: Vec<f64> =
                        col_data.iter().map(|&x| (x - max_val).exp()).collect();
                    let sum: f64 = exp_data.iter().sum();
                    for row in 0..rows {
                        result[row * cols + col] = exp_data[row] / sum;
                    }
                }
                DenseTensor::new(result, vec![rows, cols])
            } else {
                // 按行 softmax
                let mut result = vec![0.0; rows * cols];
                for row in 0..rows {
                    let row_start = row * cols;
                    let row_data = &tensor.data()[row_start..row_start + cols];
                    let max_val = row_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_data: Vec<f64> =
                        row_data.iter().map(|&x| (x - max_val).exp()).collect();
                    let sum: f64 = exp_data.iter().sum();
                    for col in 0..cols {
                        result[row_start + col] = exp_data[col] / sum;
                    }
                }
                DenseTensor::new(result, vec![rows, cols])
            }
        } else if ndim == 3 {
            // 3D 情况：[batch, seq, dim]
            let batch = tensor.shape()[0];
            let seq = tensor.shape()[1];
            let dim = tensor.shape()[2];

            if axis == 2 {
                // Softmax along last dimension (most common for transformers)
                let mut result = Vec::with_capacity(batch * seq * dim);
                for b in 0..batch {
                    for s in 0..seq {
                        let start = (b * seq + s) * dim;
                        let row_data = &tensor.data()[start..start + dim];
                        let max_val = row_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let exp_data: Vec<f64> =
                            row_data.iter().map(|&x| (x - max_val).exp()).collect();
                        let sum: f64 = exp_data.iter().sum();
                        for &e in &exp_data {
                            result.push(e / sum);
                        }
                    }
                }
                DenseTensor::new(result, vec![batch, seq, dim])
            } else {
                panic!("Softmax for 3D tensors only supports axis=2 or axis=-1");
            }
        } else {
            // N 维情况：简化处理，沿最后一个轴计算 softmax
            if axis == ndim - 1 {
                let outer_size: usize = tensor.shape()[..ndim - 1].iter().product();
                let inner_size = tensor.shape()[ndim - 1];
                let mut result = Vec::with_capacity(tensor.numel());

                for i in 0..outer_size {
                    let start = i * inner_size;
                    let row_data = &tensor.data()[start..start + inner_size];
                    let max_val = row_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_data: Vec<f64> =
                        row_data.iter().map(|&x| (x - max_val).exp()).collect();
                    let sum: f64 = exp_data.iter().sum();
                    for &e in &exp_data {
                        result.push(e / sum);
                    }
                }
                DenseTensor::new(result, tensor.shape().to_vec())
            } else {
                panic!(
                    "Softmax for {}D tensors with axis={} is not yet implemented",
                    ndim, axis
                );
            }
        }
    }

    /// GELU 激活函数：f(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn gelu(tensor: &DenseTensor) -> DenseTensor {
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
        const COEF: f64 = 0.044715;
        tensor.map(|x| {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEF * x3);
            x * 0.5 * (1.0 + inner.tanh())
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_relu() {
            let t = DenseTensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
            let result = relu(&t);
            assert_eq!(result.data(), &[0.0, 2.0, 0.0, 4.0]);
        }

        #[test]
        fn test_sigmoid() {
            let t = DenseTensor::new(vec![0.0], vec![1]);
            let result = sigmoid(&t);
            assert!((result.data()[0] - 0.5).abs() < 1e-6);
        }

        #[test]
        fn test_softmax_1d() {
            let t = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let result = softmax(&t, 0);
            let sum: f64 = result.data().iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_softmax_2d() {
            let t = DenseTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
            let result = softmax(&t, 1);
            // 每行的和应该为 1
            let row0_sum = result.data()[0] + result.data()[1];
            let row1_sum = result.data()[2] + result.data()[3];
            assert!((row0_sum - 1.0).abs() < 1e-6);
            assert!((row1_sum - 1.0).abs() < 1e-6);
        }
    }
}

/// 归一化操作
pub mod normalization {
    use super::*;

    /// Layer Normalization
    pub fn layer_norm(tensor: &DenseTensor, epsilon: f64) -> DenseTensor {
        if tensor.ndim() == 1 {
            let mean = tensor.mean(None).data()[0];
            let centered = tensor.add_scalar(-mean);
            let std =
                centered.data().iter().map(|&x| x * x).sum::<f64>().sqrt() / tensor.numel() as f64;
            centered.mul_scalar(1.0 / (std + epsilon))
        } else {
            // 对于高维张量，沿最后一个轴归一化
            // TODO: 实现完整的 N 维 LayerNorm
            panic!(
                "LayerNorm for {}D tensors is not yet implemented",
                tensor.ndim()
            );
        }
    }

    /// Batch Normalization（简化版）
    pub fn batch_norm(
        tensor: &DenseTensor,
        mean: &DenseTensor,
        var: &DenseTensor,
        epsilon: f64,
    ) -> DenseTensor {
        let centered = tensor.sub(mean);
        let std = var.map(|v| (v + epsilon).sqrt());
        centered.div(&std)
    }

    /// Graph Normalization（图归一化）
    pub fn graph_norm(tensor: &DenseTensor, epsilon: f64) -> DenseTensor {
        // 对整个图的特征进行归一化
        let mean = tensor.mean(None).data()[0];
        let centered = tensor.add_scalar(-mean);
        let std =
            centered.data().iter().map(|&x| x * x).sum::<f64>().sqrt() / tensor.numel() as f64;
        centered.mul_scalar(1.0 / (std + epsilon))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_layer_norm_1d() {
            let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
            let result = layer_norm(&t, 1e-5);
            let mean = result.mean(None).data()[0];
            assert!(mean.abs() < 1e-5);
        }
    }
}

/// 矩阵操作
pub mod matrix {
    use super::*;

    /// 矩阵转置（2D 专用）
    pub fn transpose(tensor: &DenseTensor) -> DenseTensor {
        tensor.transpose(None)
    }

    /// 矩阵求逆（使用高斯 - 约旦消元法，仅适用于小矩阵）
    pub fn inverse(tensor: &DenseTensor) -> Result<DenseTensor, TensorError> {
        if tensor.ndim() != 2 || tensor.shape()[0] != tensor.shape()[1] {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: tensor.ndim(),
            });
        }

        let n = tensor.shape()[0];
        let mut augmented = vec![0.0; n * n * 2];

        // 构建增广矩阵 [A|I]
        for i in 0..n {
            for j in 0..n {
                augmented[i * (n * 2) + j] = tensor.data()[i * n + j];
                if i == j {
                    augmented[i * (n * 2) + n + j] = 1.0;
                }
            }
        }

        // 高斯 - 约旦消元
        for col in 0..n {
            // 寻找主元
            let mut max_row = col;
            for row in col + 1..n {
                if augmented[row * (n * 2) + col].abs() > augmented[max_row * (n * 2) + col].abs() {
                    max_row = row;
                }
            }

            // 交换行
            if max_row != col {
                for j in 0..n * 2 {
                    augmented.swap(col * (n * 2) + j, max_row * (n * 2) + j);
                }
            }

            let pivot = augmented[col * (n * 2) + col];
            if pivot.abs() < 1e-10 {
                return Err(TensorError::MatrixError {
                    message: "Matrix is singular".to_string(),
                });
            }

            // 归一化当前行
            for j in 0..n * 2 {
                augmented[col * (n * 2) + j] /= pivot;
            }

            // 消去其他行
            for row in 0..n {
                if row != col {
                    let factor = augmented[row * (n * 2) + col];
                    for j in 0..n * 2 {
                        augmented[row * (n * 2) + j] -= factor * augmented[col * (n * 2) + j];
                    }
                }
            }
        }

        // 提取逆矩阵
        let mut inv_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                inv_data[i * n + j] = augmented[i * (n * 2) + n + j];
            }
        }

        Ok(DenseTensor::new(inv_data, vec![n, n]))
    }

    /// 矩阵行列式（使用余子式展开递归计算）
    pub fn determinant(tensor: &DenseTensor) -> Result<f64, TensorError> {
        if tensor.ndim() != 2 || tensor.shape()[0] != tensor.shape()[1] {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: tensor.ndim(),
            });
        }

        let n = tensor.shape()[0];
        if n == 1 {
            return Ok(tensor.data()[0]);
        }
        if n == 2 {
            return Ok(tensor.data()[0] * tensor.data()[3] - tensor.data()[1] * tensor.data()[2]);
        }

        // 使用余子式展开（对于小矩阵）
        let mut det = 0.0;
        for j in 0..n {
            let minor = get_minor(tensor, 0, j);
            let cofactor = if j % 2 == 0 { 1.0 } else { -1.0 };
            det += cofactor * tensor.data()[j] * determinant(&minor)?;
        }

        Ok(det)
    }

    /// 获取余子式矩阵
    fn get_minor(tensor: &DenseTensor, row: usize, col: usize) -> DenseTensor {
        let n = tensor.shape()[0];
        let mut minor_data = Vec::with_capacity((n - 1) * (n - 1));

        for i in 0..n {
            if i == row {
                continue;
            }
            for j in 0..n {
                if j == col {
                    continue;
                }
                let src_idx = i * n + j;
                minor_data.push(tensor.data()[src_idx]);
            }
        }

        DenseTensor::new(minor_data, vec![n - 1, n - 1])
    }

    /// 特征值和特征向量（使用幂迭代法，仅适用于对称矩阵）
    pub fn power_iteration(tensor: &DenseTensor, max_iter: usize, tol: f64) -> (f64, DenseTensor) {
        let n = tensor.shape()[0];
        let mut v = DenseTensor::ones(vec![n]).normalize();

        let mut eigenvalue = 0.0;
        for _ in 0..max_iter {
            // Av
            let av = tensor.matmul(&v);
            let new_eigenvalue = v
                .data()
                .iter()
                .zip(av.data().iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();

            // 归一化
            v = av.normalize();

            if (new_eigenvalue - eigenvalue).abs() < tol {
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        (eigenvalue, v)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_transpose() {
            let t = DenseTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let result = transpose(&t);
            assert_eq!(result.shape(), &[3, 2]);
            assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }

        #[test]
        fn test_inverse_2x2() {
            // [1, 2; 3, 4] 的逆矩阵是 [-2, 1; 1.5, -0.5]
            let t = DenseTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
            let inv = inverse(&t).unwrap();
            assert!((inv.data()[0] - (-2.0)).abs() < 1e-6);
            assert!((inv.data()[1] - 1.0).abs() < 1e-6);
            assert!((inv.data()[2] - 1.5).abs() < 1e-6);
            assert!((inv.data()[3] - (-0.5)).abs() < 1e-6);
        }

        #[test]
        fn test_determinant() {
            let t = DenseTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
            let det = determinant(&t).unwrap();
            assert!((det - (-2.0)).abs() < 1e-6);
        }
    }
}

/// 随机操作
#[cfg(feature = "rand")]
pub mod random {
    use super::*;
    use rand::Rng;

    /// 创建随机初始化的张量（Xavier 初始化）
    pub fn xavier_init(rows: usize, cols: usize) -> DenseTensor {
        let limit = (6.0 / (rows + cols) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        DenseTensor::new(data, vec![rows, cols])
    }

    /// 创建随机初始化的张量（He 初始化）
    pub fn he_init(rows: usize, cols: usize) -> DenseTensor {
        let std = (2.0 / rows as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| {
                // Box-Muller 变换生成正态分布
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        DenseTensor::new(data, vec![rows, cols])
    }

    /// Dropout（训练时使用）
    pub fn dropout(tensor: &DenseTensor, p: f64) -> DenseTensor {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - p);
        let data: Vec<f64> = tensor
            .data()
            .iter()
            .map(|&x| if rng.gen::<f64>() < p { 0.0 } else { x * scale })
            .collect();
        DenseTensor::new(data, tensor.shape().to_vec())
    }
}
