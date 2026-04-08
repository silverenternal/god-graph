//! Performance optimization utilities for Transformer inference
//!
//! This module provides:
//! - Memory pool integration for reduced allocation overhead
//! - Optimized kernels for attention and FFN
//! - Benchmark utilities for performance measurement

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::tensor::DenseTensor;
use crate::tensor::traits::TensorBase;

/// Memory pool for Transformer inference
///
/// Reuses intermediate buffers to reduce allocation overhead during inference.
/// Typical use case: allocate once, reuse across multiple forward passes.
#[derive(Debug)]
pub struct TransformerMemoryPool {
    /// Buffer for attention scores [batch, num_heads, seq_len, seq_len]
    attn_score_buffer: Option<Vec<f64>>,
    /// Buffer for attention weights [batch, num_heads, seq_len, seq_len]
    attn_weight_buffer: Option<Vec<f64>>,
    /// Buffer for QKV projections [batch, seq_len, hidden_dim]
    qkv_buffer: Option<Vec<f64>>,
    /// Buffer for output [batch, seq_len, hidden_dim]
    output_buffer: Option<Vec<f64>>,
    /// Current batch size
    batch_size: usize,
    /// Current sequence length
    seq_len: usize,
    /// Current hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
}

impl TransformerMemoryPool {
    /// Create a new memory pool with specified dimensions
    pub fn new(batch_size: usize, seq_len: usize, hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            attn_score_buffer: None,
            attn_weight_buffer: None,
            qkv_buffer: None,
            output_buffer: None,
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
        }
    }

    /// Update pool dimensions if needed
    pub fn resize(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
    ) {
        let needs_resize = self.batch_size != batch_size
            || self.seq_len != seq_len
            || self.hidden_dim != hidden_dim
            || self.num_heads != num_heads;

        if needs_resize {
            self.batch_size = batch_size;
            self.seq_len = seq_len;
            self.hidden_dim = hidden_dim;
            self.num_heads = num_heads;

            // Clear buffers to force reallocation
            self.attn_score_buffer = None;
            self.attn_weight_buffer = None;
            self.qkv_buffer = None;
            self.output_buffer = None;
        }
    }

    /// Get or allocate attention score buffer
    ///
    /// # Panics
    ///
    /// This method should never panic as it allocates the buffer if needed.
    /// Panic would only occur if memory allocation fails.
    #[must_use]
    pub fn get_attn_score_buffer(&mut self) -> &mut Vec<f64> {
        if self.attn_score_buffer.is_none() {
            let size = self.batch_size * self.num_heads * self.seq_len * self.seq_len;
            self.attn_score_buffer = Some(vec![0.0f64; size]);
        }
        self.attn_score_buffer.as_mut().unwrap()
    }

    /// Get or allocate attention weight buffer
    ///
    /// # Panics
    ///
    /// This method should never panic as it allocates the buffer if needed.
    /// Panic would only occur if memory allocation fails.
    #[must_use]
    pub fn get_attn_weight_buffer(&mut self) -> &mut Vec<f64> {
        if self.attn_weight_buffer.is_none() {
            let size = self.batch_size * self.num_heads * self.seq_len * self.seq_len;
            self.attn_weight_buffer = Some(vec![0.0f64; size]);
        }
        self.attn_weight_buffer.as_mut().unwrap()
    }

    /// Get or allocate QKV projection buffer
    ///
    /// # Panics
    ///
    /// This method should never panic as it allocates the buffer if needed.
    /// Panic would only occur if memory allocation fails.
    #[must_use]
    pub fn get_qkv_buffer(&mut self) -> &mut Vec<f64> {
        if self.qkv_buffer.is_none() {
            let size = self.batch_size * self.seq_len * self.hidden_dim;
            self.qkv_buffer = Some(vec![0.0f64; size]);
        }
        self.qkv_buffer.as_mut().unwrap()
    }

    /// Get or allocate output buffer
    ///
    /// # Panics
    ///
    /// This method should never panic as it allocates the buffer if needed.
    /// Panic would only occur if memory allocation fails.
    #[must_use]
    pub fn get_output_buffer(&mut self) -> &mut Vec<f64> {
        if self.output_buffer.is_none() {
            let size = self.batch_size * self.seq_len * self.hidden_dim;
            self.output_buffer = Some(vec![0.0f64; size]);
        }
        self.output_buffer.as_mut().unwrap()
    }

    /// Get estimated memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let mut bytes = 0;

        if let Some(ref buf) = self.attn_score_buffer {
            bytes += buf.len() * 8; // f64 = 8 bytes
        }
        if let Some(ref buf) = self.attn_weight_buffer {
            bytes += buf.len() * 8;
        }
        if let Some(ref buf) = self.qkv_buffer {
            bytes += buf.len() * 8;
        }
        if let Some(ref buf) = self.output_buffer {
            bytes += buf.len() * 8;
        }

        bytes
    }
}

impl Default for TransformerMemoryPool {
    fn default() -> Self {
        Self::new(1, 512, 4096, 32) // Default: LLaMA-7B inference
    }
}

/// Optimized softmax implementation using SIMD
///
/// # Arguments
/// * `data` - Input data (will be overwritten with softmax result)
/// * `dim` - Dimension along which to compute softmax
/// * `shape` - Tensor shape
pub fn softmax_inplace_simd(data: &mut [f64], shape: &[usize], dim: usize) {
    assert!(dim < shape.len(), "Invalid dimension");

    let ndim = shape.len();
    let dim_size = shape[dim];

    // Calculate stride for the softmax dimension
    let mut stride = 1;
    for &size in shape.iter().take(ndim).skip(dim + 1) {
        stride *= size;
    }

    // Calculate outer iterations
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();

    #[cfg(feature = "simd")]
    {
        // SIMD-optimized softmax
        for o in 0..outer {
            for i in 0..inner {
                let base = o * dim_size * stride + i;

                // Find max for numerical stability (SIMD)
                let mut max_val = f64::NEG_INFINITY;
                for d in (0..dim_size).step_by(4) {
                    if d + 4 <= dim_size {
                        let vals = [
                            data[base + d * stride],
                            data[base + (d + 1) * stride],
                            data[base + (d + 2) * stride],
                            data[base + (d + 3) * stride],
                        ];
                        let simd_vals = f64x4::new(vals);
                        let max_simd = simd_vals.max(f64x4::new([max_val; 4]));
                        let max_arr = max_simd.to_array();
                        max_val = max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3]);
                    } else {
                        for rem_d in d..dim_size {
                            max_val = max_val.max(data[base + rem_d * stride]);
                        }
                    }
                }

                // Compute exp(x - max) and sum (SIMD)
                let mut sum_exp = 0.0;
                for d in (0..dim_size).step_by(4) {
                    if d + 4 <= dim_size {
                        let vals = [
                            (data[base + d * stride] - max_val).exp(),
                            (data[base + (d + 1) * stride] - max_val).exp(),
                            (data[base + (d + 2) * stride] - max_val).exp(),
                            (data[base + (d + 3) * stride] - max_val).exp(),
                        ];
                        let simd_vals = f64x4::new(vals);
                        let sum_simd = simd_vals.reduce_add();
                        sum_exp += sum_simd;

                        // Store back
                        let exp_vals = simd_vals.to_array();
                        data[base + d * stride] = exp_vals[0];
                        data[base + (d + 1) * stride] = exp_vals[1];
                        data[base + (d + 2) * stride] = exp_vals[2];
                        data[base + (d + 3) * stride] = exp_vals[3];
                    } else {
                        for rem_d in d..dim_size {
                            let exp_val = (data[base + rem_d * stride] - max_val).exp();
                            sum_exp += exp_val;
                            data[base + rem_d * stride] = exp_val;
                        }
                    }
                }

                // Normalize (SIMD)
                let inv_sum = 1.0 / sum_exp;
                let inv_sum_simd = f64x4::new([inv_sum; 4]);
                for d in (0..dim_size).step_by(4) {
                    if d + 4 <= dim_size {
                        let vals = [
                            data[base + d * stride],
                            data[base + (d + 1) * stride],
                            data[base + (d + 2) * stride],
                            data[base + (d + 3) * stride],
                        ];
                        let simd_vals = f64x4::new(vals) * inv_sum_simd;
                        let norm_vals = simd_vals.to_array();
                        data[base + d * stride] = norm_vals[0];
                        data[base + (d + 1) * stride] = norm_vals[1];
                        data[base + (d + 2) * stride] = norm_vals[2];
                        data[base + (d + 3) * stride] = norm_vals[3];
                    } else {
                        for rem_d in d..dim_size {
                            data[base + rem_d * stride] *= inv_sum;
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback: naive implementation
        for o in 0..outer {
            for i in 0..inner {
                let base = o * dim_size * stride + i;

                // Find max for numerical stability
                let max_val = (0..dim_size)
                    .map(|d| data[base + d * stride])
                    .fold(f64::NEG_INFINITY, f64::max);

                // Compute exp(x - max) and sum
                let sum_exp: f64 = (0..dim_size)
                    .map(|d| {
                        let exp_val = (data[base + d * stride] - max_val).exp();
                        data[base + d * stride] = exp_val;
                        exp_val
                    })
                    .sum();

                // Normalize
                let inv_sum = 1.0 / sum_exp;
                for d in 0..dim_size {
                    data[base + d * stride] *= inv_sum;
                }
            }
        }
    }
}

/// Optimized matrix multiplication with pre-allocated buffer
///
/// # Arguments
/// * `a` - Matrix A [M, K]
/// * `b` - Matrix B [K, N]
/// * `buffer` - Pre-allocated output buffer [M * N]
///
/// # Returns
/// Result matrix [M, N]
pub fn matmul_with_buffer(a: &DenseTensor, b: &DenseTensor, buffer: &mut Vec<f64>) -> DenseTensor {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    assert_eq!(a.shape()[1], b.shape()[0], "Inner dimensions must match");

    // Ensure buffer is large enough
    if buffer.len() < m * n {
        *buffer = vec![0.0; m * n];
    }

    #[cfg(feature = "simd")]
    {
        // SIMD-optimized matmul
        for i in 0..m {
            for j in (0..n).step_by(4) {
                if j + 4 <= n {
                    let mut sum_simd = f64x4::new([0.0; 4]);

                    for p in 0..k {
                        let a_val = a.data()[i * k + p];
                        let a_simd = f64x4::new([a_val; 4]);

                        let b_vals = [
                            b.data()[p * n + j],
                            b.data()[p * n + j + 1],
                            b.data()[p * n + j + 2],
                            b.data()[p * n + j + 3],
                        ];
                        let b_simd = f64x4::new(b_vals);

                        sum_simd += a_simd * b_simd;
                    }

                    let sums = sum_simd.to_array();
                    buffer[i * n + j] = sums[0];
                    buffer[i * n + j + 1] = sums[1];
                    buffer[i * n + j + 2] = sums[2];
                    buffer[i * n + j + 3] = sums[3];
                } else {
                    // Handle remainder
                    for rem_j in j..n {
                        let mut sum = 0.0;
                        for p in 0..k {
                            sum += a.data()[i * k + p] * b.data()[p * n + rem_j];
                        }
                        buffer[i * n + rem_j] = sum;
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback: naive implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.data()[i * k + p] * b.data()[p * n + j];
                }
                buffer[i * n + j] = sum;
            }
        }
    }

    DenseTensor::new(buffer[..m * n].to_vec(), vec![m, n])
}

/// Benchmark utilities for measuring inference performance
pub mod benchmark {
    use std::time::Instant;

    /// Measure execution time of a function
    ///
    /// # Arguments
    /// * `name` - Benchmark name
    /// * `f` - Function to benchmark
    ///
    /// # Returns
    /// Elapsed time in milliseconds
    pub fn measure_time<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();

        println!("{}: {:.3} ms", name, elapsed.as_secs_f64() * 1000.0);
        result
    }

    /// Benchmark throughput (operations per second)
    ///
    /// # Arguments
    /// * `name` - Benchmark name
    /// * `iterations` - Number of iterations
    /// * `f` - Function to benchmark
    pub fn benchmark_throughput<F>(name: &str, iterations: usize, f: F)
    where
        F: Fn(),
    {
        let start = Instant::now();

        for _ in 0..iterations {
            f();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!(
            "{}: {:.2} ops/sec ({:.3} ms/op)",
            name,
            ops_per_sec,
            elapsed.as_secs_f64() * 1000.0 / iterations as f64
        );
    }

    /// Measure tokens per second for inference
    ///
    /// # Arguments
    /// * `num_tokens` - Number of tokens generated
    /// * `elapsed_ms` - Elapsed time in milliseconds
    pub fn tokens_per_second(num_tokens: usize, elapsed_ms: f64) -> f64 {
        num_tokens as f64 / (elapsed_ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::perf::benchmark;

    #[test]
    fn test_memory_pool() {
        let mut pool = TransformerMemoryPool::new(2, 128, 768, 8);

        // Test buffer allocation
        let attn_score_buf = pool.get_attn_score_buffer();
        assert_eq!(attn_score_buf.len(), 2 * 8 * 128 * 128);

        let attn_weight_buf = pool.get_attn_weight_buffer();
        assert_eq!(attn_weight_buf.len(), 2 * 8 * 128 * 128);

        let qkv_buf = pool.get_qkv_buffer();
        assert_eq!(qkv_buf.len(), 2 * 128 * 768);

        let output_buf = pool.get_output_buffer();
        assert_eq!(output_buf.len(), 2 * 128 * 768);

        // Test resize
        pool.resize(4, 256, 1024, 16);
        assert_eq!(pool.batch_size, 4);
        assert_eq!(pool.seq_len, 256);
    }

    #[test]
    fn test_softmax_simd() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        softmax_inplace_simd(&mut data, &shape, 1);

        // Check that each row sums to 1.0
        for i in 0..2 {
            let row_sum: f64 = data[i * 3..(i + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_matmul_with_buffer() {
        let a = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);

        let mut buffer = vec![0.0; 4];
        let result = matmul_with_buffer(&a, &b, &mut buffer);

        assert_eq!(result.shape(), &[2, 2]);
        assert!((result.data()[0] - 1.5).abs() < 1e-5);
        assert!((result.data()[1] - 1.5).abs() < 1e-5);
        assert!((result.data()[2] - 3.5).abs() < 1e-5);
        assert!((result.data()[3] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_benchmark_utils() {
        // Test measure_time
        let elapsed = std::time::Instant::now();
        benchmark::measure_time("test", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
        let actual_elapsed = elapsed.elapsed().as_secs_f64() * 1000.0;

        assert!(actual_elapsed >= 10.0, "Should have slept for at least 10ms");

        // Test tokens_per_second
        let tps = benchmark::tokens_per_second(100, 1000.0); // 100 tokens in 1000ms
        assert!((tps - 100.0).abs() < 1e-5);
    }
}
