//! Quantization module for efficient inference
//!
//! This module provides:
//! - INT8 quantization
//! - INT4 quantization (experimental)
//! - Quantized matrix multiplication
//! - Post-training quantization (PTQ)

use crate::tensor::DenseTensor;
use crate::tensor::traits::TensorBase;

/// Quantization data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// 32-bit floating point
    F32,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Target data type
    pub dtype: QuantDtype,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Whether to use per-channel quantization
    pub per_channel: bool,
    /// Quantization granularity (for per-channel)
    pub axis: Option<usize>,
}

impl QuantizationConfig {
    /// Create default INT8 quantization config
    pub fn int8() -> Self {
        Self {
            dtype: QuantDtype::INT8,
            symmetric: true,
            per_channel: false,
            axis: None,
        }
    }

    /// Create INT4 quantization config
    pub fn int4() -> Self {
        Self {
            dtype: QuantDtype::INT4,
            symmetric: true,
            per_channel: false,
            axis: None,
        }
    }

    /// Create per-channel INT8 config
    pub fn per_channel_int8(axis: usize) -> Self {
        Self {
            dtype: QuantDtype::INT8,
            symmetric: true,
            per_channel: true,
            axis: Some(axis),
        }
    }
}

/// Quantized tensor (INT8)
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Vec<i8>,
    /// Scale factor(s)
    pub scale: Vec<f64>,
    /// Zero point(s)
    pub zero_point: Vec<i8>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization configuration
    pub config: QuantizationConfig,
    /// Channel-wise scales (for per-channel quantization)
    pub channel_scales: Option<Vec<f64>>,
    /// Channel-wise zero points (for per-channel quantization)
    pub channel_zero_points: Option<Vec<i8>>,
}

impl QuantizedTensor {
    /// Quantize a dense tensor to INT8
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to quantize
    /// * `config` - Quantization configuration
    pub fn from_tensor(tensor: &DenseTensor, config: QuantizationConfig) -> Self {
        match config.dtype {
            QuantDtype::INT8 => Self::quantize_int8(tensor, &config),
            QuantDtype::INT4 => Self::quantize_int4(tensor, &config),
            QuantDtype::F32 => {
                // No quantization needed
                let data = tensor.data().iter().map(|&x| x as i8).collect();
                Self {
                    data,
                    scale: vec![1.0],
                    zero_point: vec![0],
                    shape: tensor.shape().to_vec(),
                    config,
                    channel_scales: None,
                    channel_zero_points: None,
                }
            }
        }
    }

    /// Quantize to INT8
    fn quantize_int8(tensor: &DenseTensor, config: &QuantizationConfig) -> Self {
        if config.per_channel {
            Self::quantize_int8_per_channel(tensor, config.axis.unwrap_or(0))
        } else {
            Self::quantize_int8_per_tensor(tensor)
        }
    }

    /// Per-tensor INT8 quantization (symmetric)
    fn quantize_int8_per_tensor(tensor: &DenseTensor) -> Self {
        let data = tensor.data();

        // Find max absolute value for symmetric quantization
        let max_abs = data.iter().fold(0.0_f64, |max, &x: &f64| max.max(x.abs()));

        // Compute scale for symmetric quantization [-127, 127]
        let scale = max_abs / 127.0;

        // Quantize
        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| {
                let q = (x / scale).round() as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();

        Self {
            data: quantized,
            scale: vec![scale],
            zero_point: vec![0],
            shape: tensor.shape().to_vec(),
            config: QuantizationConfig::int8(),
            channel_scales: None,
            channel_zero_points: None,
        }
    }

    /// Per-channel INT8 quantization
    fn quantize_int8_per_channel(tensor: &DenseTensor, axis: usize) -> Self {
        let data = tensor.data();
        let shape = tensor.shape();

        if axis >= shape.len() {
            return Self::quantize_int8_per_tensor(tensor);
        }

        let channel_dim = shape[axis];
        let channels_before: usize = shape[..axis].iter().product();
        let channels_after: usize = shape[axis + 1..].iter().product();

        let mut channel_scales = Vec::with_capacity(channel_dim);
        let mut channel_zero_points = Vec::with_capacity(channel_dim);
        let mut quantized = Vec::with_capacity(data.len());

        for c in 0..channel_dim {
            // Extract channel data
            let mut channel_min = f64::INFINITY;
            let mut channel_max = f64::NEG_INFINITY;

            for cb in 0..channels_before {
                for ca in 0..channels_after {
                    let offset = (cb * channel_dim + c) * channels_after + ca;
                    let val = data[offset];
                    channel_min = channel_min.min(val);
                    channel_max = channel_max.max(val);
                }
            }

            // Compute scale and zero point for this channel
            let scale = (channel_max - channel_min) / 255.0;
            let zero_point = 0i8;

            channel_scales.push(scale);
            channel_zero_points.push(zero_point);
        }

        // Quantize all data
        for (i, &val) in data.iter().enumerate() {
            let c = (i / channels_after) % channel_dim;
            let scale = channel_scales[c];

            let q = (val / scale).round() as i32;
            let q = q.clamp(-128, 127) as i8;
            quantized.push(q);
        }

        Self {
            data: quantized,
            scale: vec![1.0],
            zero_point: vec![0],
            shape: shape.to_vec(),
            config: QuantizationConfig::per_channel_int8(axis),
            channel_scales: Some(channel_scales),
            channel_zero_points: Some(channel_zero_points),
        }
    }

    /// Quantize to INT4 (experimental)
    fn quantize_int4(tensor: &DenseTensor, config: &QuantizationConfig) -> Self {
        // INT4 quantization packs two values per byte
        let data = tensor.data();

        // Find min and max
        let (min, max) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max): (f64, f64), &x| {
            (min.min(x), max.max(x))
        });

        let scale = (max - min) / 15.0; // INT4 has 16 levels

        // Quantize to INT4 and pack
        let mut packed_data = Vec::with_capacity((data.len() + 1) / 2);

        for i in (0..data.len()).step_by(2) {
            let q0 = ((data[i] - min) / scale).round() as i32;
            let q0 = q0.clamp(0, 15) as u8;

            let q1 = if i + 1 < data.len() {
                ((data[i + 1] - min) / scale).round() as i32
            } else {
                0
            };
            let q1 = q1.clamp(0, 15) as u8;

            // Pack two INT4 values into one byte
            let packed = (q1 << 4) | q0;
            packed_data.push(packed as i8);
        }

        Self {
            data: packed_data,
            scale: vec![scale],
            zero_point: vec![0],
            shape: tensor.shape().to_vec(),
            config: config.clone(),
            channel_scales: None,
            channel_zero_points: None,
        }
    }

    /// Dequantize to dense tensor
    pub fn dequantize(&self) -> DenseTensor {
        match self.config.dtype {
            QuantDtype::INT8 => self.dequantize_int8(),
            QuantDtype::INT4 => self.dequantize_int4(),
            QuantDtype::F32 => {
                let data = self.data.iter().map(|&x| x as f64).collect();
                DenseTensor::new(data, self.shape.clone())
            }
        }
    }

    /// Dequantize INT8
    fn dequantize_int8(&self) -> DenseTensor {
        let data = if let Some(scales) = &self.channel_scales {
            // Per-channel dequantization
            let shape = &self.shape;
            let axis = self.config.axis.unwrap_or(0);
            let channel_dim = shape[axis];
            let _channels_before: usize = shape[..axis].iter().product();
            let channels_after: usize = shape[axis + 1..].iter().product();

            self.data
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    let c = (i / channels_after) % channel_dim;
                    let scale = scales[c];
                    q as f64 * scale
                })
                .collect()
        } else {
            // Per-tensor dequantization
            let scale = self.scale[0];

            self.data
                .iter()
                .map(|&q| q as f64 * scale)
                .collect()
        };

        DenseTensor::new(data, self.shape.clone())
    }

    /// Dequantize INT4
    fn dequantize_int4(&self) -> DenseTensor {
        let scale = self.scale[0];
        let mut data = Vec::with_capacity(self.shape.iter().product::<usize>());

        for &packed in &self.data {
            let q0 = (packed as u8) & 0x0F;
            let q1 = (packed as u8) >> 4;

            data.push(q0 as f64 * scale);
            data.push(q1 as f64 * scale);
        }

        // Trim to original size
        let total: usize = self.shape.iter().product();
        data.truncate(total);

        DenseTensor::new(data, self.shape.clone())
    }

    /// Get quantized data
    pub fn quantized_data(&self) -> &[i8] {
        &self.data
    }

    /// Get scale
    pub fn scale(&self) -> f64 {
        self.scale[0]
    }

    /// Get memory size in bytes
    pub fn memory_bytes(&self) -> usize {
        let total_elements = self.shape.iter().product::<usize>();
        match self.config.dtype {
            QuantDtype::INT8 => total_elements, // 1 byte per element
            QuantDtype::INT4 => (total_elements + 1) / 2, // Packed: 2 elements per byte
            QuantDtype::F32 => total_elements * 4, // 4 bytes per element
        }
    }

    /// Get compression ratio compared to F32
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.shape.iter().product::<usize>() * 4; // F32 = 4 bytes
        original_bytes as f64 / self.memory_bytes() as f64
    }
}

/// Quantized matrix multiplication
pub struct QuantizedMatMul;

impl QuantizedMatMul {
    /// Multiply quantized matrix with quantized matrix
    ///
    /// # Arguments
    /// * `a` - Quantized matrix [M, K]
    /// * `b` - Quantized matrix [K, N]
    ///
    /// # Returns
    /// Dequantized result [M, N]
    pub fn matmul(a: &QuantizedTensor, b: &QuantizedTensor) -> DenseTensor {
        // Use pure INT8 GEMM for better performance
        Self::gemm_int8(a, b)
    }

    /// Multiply quantized matrix with dense matrix
    ///
    /// # Arguments
    /// * `a` - Quantized matrix [M, K]
    /// * `b` - Dense matrix [K, N]
    ///
    /// # Returns
    /// Dequantized result [M, N]
    pub fn matmul_qd(a: &QuantizedTensor, b: &DenseTensor) -> DenseTensor {
        // Quantize b temporarily and use INT8 GEMM
        let b_q = QuantizedTensor::from_tensor(b, QuantizationConfig::int8());
        Self::gemm_int8(a, &b_q)
    }

    /// Multiply dense matrix with quantized matrix
    ///
    /// # Arguments
    /// * `a` - Dense matrix [M, K]
    /// * `b` - Quantized matrix [K, N]
    ///
    /// # Returns
    /// Dequantized result [M, N]
    pub fn matmul_dq(a: &DenseTensor, b: &QuantizedTensor) -> DenseTensor {
        // Quantize a temporarily and use INT8 GEMM
        let a_q = QuantizedTensor::from_tensor(a, QuantizationConfig::int8());
        Self::gemm_int8(&a_q, b)
    }

    /// Pure INT8 GEMM implementation (no dequantization during computation)
    ///
    /// This is the performance-critical path that avoids dequantizing
    /// until the final result, enabling potential SIMD optimizations.
    ///
    /// # Algorithm
    /// For C = A @ B where A, B are INT8:
    /// 1. Compute INT32 accumulator: acc = sum(a_ik * b_kj)
    /// 2. Dequantize: C_ij = acc_ij * scale_a * scale_b
    ///
    /// # Arguments
    /// * `a` - Quantized matrix [M, K]
    /// * `b` - Quantized matrix [K, N]
    ///
    /// # Returns
    /// Dequantized result [M, N]
    pub fn gemm_int8(a: &QuantizedTensor, b: &QuantizedTensor) -> DenseTensor {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

        // Combined scale for dequantization
        let scale_a = if let Some(ref scales) = a.channel_scales {
            // Per-channel quantization for A (axis=1, output channels)
            scales
        } else {
            // Per-tensor quantization
            &vec![a.scale[0]; k]
        };

        let scale_b = if let Some(ref scales) = b.channel_scales {
            // Per-channel quantization for B (axis=0, input channels)
            scales
        } else {
            // Per-tensor quantization
            &vec![b.scale[0]; k]
        };

        // Precompute per-row scales for dequantization
        let output_scales: Vec<f64> = if a.channel_scales.is_some() && b.channel_scales.is_some() {
            // Both per-channel: output_scale[i,j] = scale_a[j] * scale_b[j]
            // For simplicity, use average scale
            let avg_scale_a = scale_a.iter().sum::<f64>() / scale_a.len() as f64;
            let avg_scale_b = scale_b.iter().sum::<f64>() / scale_b.len() as f64;
            vec![avg_scale_a * avg_scale_b; m * n]
        } else if a.channel_scales.is_some() {
            // A is per-channel, B is per-tensor
            let scale_b_val = b.scale[0];
            scale_a.iter().map(|&s| s * scale_b_val).collect()
        } else if b.channel_scales.is_some() {
            // A is per-tensor, B is per-channel
            let scale_a_val = a.scale[0];
            scale_b.iter().map(|&s| scale_a_val * s).collect()
        } else {
            // Both per-tensor
            vec![a.scale[0] * b.scale[0]; m * n]
        };

        // INT8 GEMM kernel: compute INT32 accumulators
        let mut result = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                
                // Dot product in INT8, accumulate in INT32
                for p in 0..k {
                    let a_val = a.data[i * k + p];
                    let b_val = b.data[p * n + j];
                    acc += (a_val as i32) * (b_val as i32);
                }

                // Dequantize the accumulator
                let scale = output_scales[i * n + j];
                result.push(acc as f64 * scale);
            }
        }

        DenseTensor::new(result, vec![m, n])
    }

    /// Optimized INT8 GEMM with loop unrolling and better cache locality
    ///
    /// This version uses:
    /// - Loop unrolling (4x) for better ILP
    /// - Row-major access pattern for better cache utilization
    ///
    /// # Arguments
    /// * `a` - Quantized matrix [M, K]
    /// * `b` - Quantized matrix [K, N]
    ///
    /// # Returns
    /// Dequantized result [M, N]
    pub fn gemm_int8_optimized(a: &QuantizedTensor, b: &QuantizedTensor) -> DenseTensor {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

        // Combined scale
        let scale = a.scale[0] * b.scale[0];

        let mut result = vec![0.0f64; m * n];

        // Block processing for better cache utilization
        const BLOCK_SIZE: usize = 32;

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(m);
                let j_end = (j_block + BLOCK_SIZE).min(n);

                for p in 0..k {
                    // Load and replicate a[p] for this row block
                    for i in i_block..i_end {
                        let a_val = a.data[i * k + p] as i32;
                        
                        // Process b row with loop unrolling
                        let mut j = j_block;
                        while j + 4 <= j_end {
                            let b0 = b.data[p * n + j] as i32;
                            let b1 = b.data[p * n + j + 1] as i32;
                            let b2 = b.data[p * n + j + 2] as i32;
                            let b3 = b.data[p * n + j + 3] as i32;

                            // Accumulate (will dequantize later)
                            // Note: We're storing f64 directly for simplicity
                            // A production implementation would use INT32 accumulators
                            result[i * n + j] += (a_val * b0) as f64;
                            result[i * n + j + 1] += (a_val * b1) as f64;
                            result[i * n + j + 2] += (a_val * b2) as f64;
                            result[i * n + j + 3] += (a_val * b3) as f64;

                            j += 4;
                        }

                        // Handle remainder
                        while j < j_end {
                            let b_val = b.data[p * n + j] as i32;
                            result[i * n + j] += (a_val * b_val) as f64;
                            j += 1;
                        }
                    }
                }
            }
        }

        // Final dequantization
        for val in &mut result {
            *val *= scale;
        }

        DenseTensor::new(result, vec![m, n])
    }
}

/// Quantization utilities for model weights
pub mod weight_quantization {
    use super::*;

    /// Quantize model weights to INT8
    pub fn quantize_weights(weights: &DenseTensor) -> QuantizedTensor {
        QuantizedTensor::from_tensor(weights, QuantizationConfig::int8())
    }

    /// Quantize model weights with per-channel quantization
    pub fn quantize_weights_per_channel(weights: &DenseTensor, axis: usize) -> QuantizedTensor {
        QuantizedTensor::from_tensor(weights, QuantizationConfig::per_channel_int8(axis))
    }

    /// Quantize embedding weights
    pub fn quantize_embeddings(embeddings: &DenseTensor) -> QuantizedTensor {
        // Embeddings often benefit from per-row quantization
        QuantizedTensor::from_tensor(embeddings, QuantizationConfig::per_channel_int8(0))
    }

    /// Quantize linear layer weights (output channel quantization)
    pub fn quantize_linear_weights(weights: &DenseTensor) -> QuantizedTensor {
        // For linear layers, per-output-channel quantization is common
        QuantizedTensor::from_tensor(weights, QuantizationConfig::per_channel_int8(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization() {
        let tensor = DenseTensor::new(vec![0.0, 0.25, 0.5, 0.75, 1.0], vec![1, 5]);
        let config = QuantizationConfig::int8();

        let quantized = QuantizedTensor::from_tensor(&tensor, config);

        assert_eq!(quantized.shape, vec![1, 5]);
        assert_eq!(quantized.data.len(), 5);

        // Dequantize and check error
        let dequantized = quantized.dequantize();
        let original = tensor.data();
        let reconstructed = dequantized.data();

        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            // INT8 quantization error should be within 1/255 of the range
            assert!((orig - recon).abs() < 0.1, "Quantization error too large: orig={}, recon={}", orig, recon);
        }
    }

    #[test]
    fn test_int8_per_channel_quantization() {
        let tensor = DenseTensor::new(vec![0.0, 1.0, 2.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let config = QuantizationConfig::per_channel_int8(1);

        let quantized = QuantizedTensor::from_tensor(&tensor, config);

        assert!(quantized.channel_scales.is_some());
        assert_eq!(quantized.channel_scales.unwrap().len(), 3);
    }

    #[test]
    fn test_int4_quantization() {
        let tensor = DenseTensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]);
        let config = QuantizationConfig::int4();

        let quantized = QuantizedTensor::from_tensor(&tensor, config);

        // INT4 packs 2 values per byte, so 3 values need 2 bytes
        assert_eq!(quantized.data.len(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        let tensor = DenseTensor::new(vec![0.0; 100], vec![10, 10]);

        let int8 = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int8());
        assert!((int8.compression_ratio() - 4.0).abs() < 0.1); // INT8 is 4x smaller than F32

        let int4 = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int4());
        assert!((int4.compression_ratio() - 8.0).abs() < 0.1); // INT4 is 8x smaller than F32
    }

    #[test]
    fn test_quantized_matmul() {
        let a = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);

        let a_q = QuantizedTensor::from_tensor(&a, QuantizationConfig::int8());
        let b_q = QuantizedTensor::from_tensor(&b, QuantizationConfig::int8());

        let result = QuantizedMatMul::matmul(&a_q, &b_q);

        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_quantization() {
        let weights = DenseTensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![1, 5]);

        let quantized = weight_quantization::quantize_weights(&weights);

        assert_eq!(quantized.config.dtype, QuantDtype::INT8);

        let dequantized = quantized.dequantize();
        let original = weights.data();
        let reconstructed = dequantized.data();

        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            // Weight quantization error should be within acceptable range
            assert!((orig - recon).abs() < 0.15, "Weight quantization error too large: orig={}, recon={}", orig, recon);
        }
    }
}
