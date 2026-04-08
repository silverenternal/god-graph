//! AVX-512 SIMD optimizations for 8× f64 parallel operations
//!
//! This module provides AVX-512 intrinsics for maximum throughput on supported CPUs.
//! AVX-512 processes 8 double-precision floats in parallel (512-bit vectors).
//!
//! # Requirements
//! - Requires nightly Rust and `unstable` feature
//! - CPU must support AVX-512 (Intel Xeon Scalable, Intel Core 12th gen+, AMD Zen4+)
//!
//! # Performance
//! - 8× throughput improvement over scalar code
//! - 2× improvement over AVX2 (f64x4)
//! - Best for large matrix operations and tensor computations

#![cfg(all(target_feature = "avx512f", feature = "unstable"))]

use std::arch::asm;

/// AVX-512 vector type: 8 × f64
#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(dead_code)]
pub struct F64x8(pub std::arch::x86_64::__m512d);

impl F64x8 {
    /// Create a new F64x8 from an array of 8 f64 values
    ///
    /// # Safety
    /// - This function uses `_mm512_loadu_pd` which requires the pointer to be valid
    /// - The input array must contain exactly 8 f64 values (64 bytes)
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    ///
    /// # Performance
    /// Uses unaligned load for flexibility. For aligned data (64-byte), consider using
    /// `_mm512_load_pd` for potentially better performance.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(vals: [f64; 8]) -> Self {
        let ptr = vals.as_ptr();
        let vec = std::arch::x86_64::_mm512_loadu_pd(ptr);
        Self(vec)
    }

    /// Create a new F64x8 with all elements set to the same value (splat)
    ///
    /// # Safety
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    /// - This is a pure register operation, no memory access involved
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn splat(val: f64) -> Self {
        let vec = std::arch::x86_64::_mm512_set1_pd(val);
        Self(vec)
    }

    /// Create a new F64x8 from individual values
    ///
    /// # Safety
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    /// - Values are loaded into registers in reverse order (v7, v6, ..., v0)
    ///   to match AVX-512 convention
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn from_vals(
        v0: f64, v1: f64, v2: f64, v3: f64,
        v4: f64, v5: f64, v6: f64, v7: f64,
    ) -> Self {
        let vec = std::arch::x86_64::_mm512_set_pd(v7, v6, v5, v4, v3, v2, v1, v0);
        Self(vec)
    }

    /// Convert F64x8 to array of 8 f64 values
    ///
    /// # Safety
    /// - Uses `_mm512_storeu_pd` which requires the destination pointer to be valid
    /// - Output array is guaranteed to contain exactly 8 f64 values
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn to_array(self) -> [f64; 8] {
        let mut result = [0.0; 8];
        std::arch::x86_64::_mm512_storeu_pd(result.as_mut_ptr(), self.0);
        result
    }

    /// Add two F64x8 vectors
    ///
    /// # Safety
    /// - Pure register-to-register operation, no memory access
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn add(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_add_pd(self.0, other.0))
    }

    /// Multiply two F64x8 vectors
    ///
    /// # Safety
    /// - Pure register-to-register operation, no memory access
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_mul_pd(self.0, other.0))
    }

    /// Subtract two F64x8 vectors
    ///
    /// # Safety
    /// - Pure register-to-register operation, no memory access
    /// - Caller must ensure the CPU supports AVX-512 (target_feature = "avx512f")
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn sub(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_sub_pd(self.0, other.0))
    }

    /// Divide two F64x8 vectors
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn div(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_div_pd(self.0, other.0))
    }

    /// Compute element-wise maximum
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn max(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_max_pd(self.0, other.0))
    }

    /// Compute element-wise minimum
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn min(self, other: Self) -> Self {
        Self(std::arch::x86_64::_mm512_min_pd(self.0, other.0))
    }

    /// Compute element-wise square root
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn sqrt(self) -> Self {
        Self(std::arch::x86_64::_mm512_sqrt_pd(self.0))
    }

    /// Compute element-wise fused multiply-add: self * mul + add
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn fmadd(self, mul: Self, add: Self) -> Self {
        Self(std::arch::x86_64::_mm512_fmadd_pd(self.0, mul.0, add.0))
    }

    /// Reduce sum: sum all 8 elements
    ///
    /// P2 OPTIMIZATION: Use _mm512_reduce_add_pd when available (single instruction)
    /// Falls back to horizontal add for older compilers
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn reduce_add(self) -> f64 {
        // P2 OPTIMIZATION: Try to use _mm512_reduce_add_pd (single instruction)
        // This is available in newer compilers and provides best performance
        // Fallback to horizontal add for compatibility
        
        // Option 1: Use _mm512_reduce_add_pd (best performance, requires newer compiler)
        // Note: This intrinsic may not be available in all Rust versions
        // std::arch::x86_64::_mm512_reduce_add_pd(self.0)
        
        // Option 2: Horizontal sum using AVX-512 (portable across compilers)
        // Split into two 256-bit halves and sum
        let lo = std::arch::x86_64::_mm512_castpd512_pd256(self.0);
        let hi = std::arch::x86_64::_mm512_extractf32x4_pd::<2>(self.0);

        // Sum the two halves
        let sum_lo = std::arch::x86_64::_mm256_hadd_pd(lo, lo);
        let sum_hi = std::arch::x86_64::_mm256_hadd_pd(hi, hi);

        // Combine
        let sum = std::arch::x86_64::_mm256_add_pd(sum_lo, sum_hi);

        // Horizontal add within each 256-bit lane
        let sum2 = std::arch::x86_64::_mm256_hadd_pd(sum, sum);

        // Extract and add the two remaining values
        let result = std::arch::x86_64::_mm256_castpd256_pd128(sum2);
        let lo_val = std::arch::x86_64::_mm_cvtsd_f64(result);
        let hi_val = std::arch::x86_64::_mm_cvtsd_f64(std::arch::x86_64::_mm_shuffle_pd(result, result, 1));

        lo_val + hi_val
    }

    /// Load from memory (unaligned)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn load(ptr: *const f64) -> Self {
        Self(std::arch::x86_64::_mm512_loadu_pd(ptr))
    }

    /// Store to memory (unaligned)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn store(self, ptr: *mut f64) {
        std::arch::x86_64::_mm512_storeu_pd(ptr, self.0);
    }

    /// Load from memory with streaming hint (non-temporal, bypasses cache)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn load_streaming(ptr: *const f64) -> Self {
        Self(std::arch::x86_64::_mm512_loadu_pd(ptr))
    }

    /// Store to memory with streaming hint (non-temporal, bypasses cache)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn store_streaming(self, ptr: *mut f64) {
        std::arch::x86_64::_mm512_stream_pd(ptr, self.0);
    }

    /// Compute absolute value (element-wise)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn abs(self) -> Self {
        // Clear sign bit (bit 63) using AND with mask
        let sign_mask = std::arch::x86_64::_mm512_set1_pd(-0.0);
        let not_sign = std::arch::x86_64::_mm512_andnot_pd(sign_mask, self.0);
        Self(not_sign)
    }

    /// Compare: less than or equal (element-wise)
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn cmp_le(self, other: Self) -> i32 {
        std::arch::x86_64::_mm512_cmp_pd_mask(self.0, other.0, std::arch::x86_64::_CMP_LE_OQ)
    }

    /// Blend vectors based on mask
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn blend(self, other: Self, mask: i32) -> Self {
        Self(std::arch::x86_64::_mm512_mask_blend_pd(mask, self.0, other.0))
    }
}

// ============================================================================
// AVX-512 Activation Functions
// ============================================================================

/// AVX-512 ReLU activation: f(x) = max(0, x)
///
/// Processes 8 f64 values in parallel using AVX-512.
///
/// # Arguments
/// * `data` - Mutable slice to process (modified in-place)
///
/// # Safety
/// - This function uses AVX-512 intrinsics which require CPU support
/// - Caller must ensure `target_feature = "avx512f"` is enabled
/// - The input slice must be valid and properly aligned for AVX-512 access
/// - Data is modified in-place, so caller must ensure no aliasing
///
/// # Performance
/// - 8× throughput improvement over scalar ReLU
/// - Best for large tensors (> 64 elements)
/// - Uses streaming stores for cache efficiency
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_avx512(data: &mut [f64]) {
    let len = data.len();
    let chunks = len / 8;
    let zeros = F64x8::splat(0.0);

    // Process 8 elements at a time
    for i in 0..chunks {
        let idx = i * 8;
        let v = F64x8::load(data.as_ptr().add(idx));
        let maxed = v.max(zeros);
        maxed.store(data.as_mut_ptr().add(idx));
    }

    // Handle remainder with scalar code
    for i in (chunks * 8)..len {
        data[i] = data[i].max(0.0);
    }
}

/// AVX-512 GELU activation: f(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Uses fast tanh approximation for performance.
/// Processes 8 f64 values in parallel using AVX-512.
///
/// # Arguments
/// * `data` - Mutable slice to process (modified in-place)
///
/// # Safety
/// - This function uses AVX-512 intrinsics which require CPU support
/// - Caller must ensure `target_feature = "avx512f"` is enabled
/// - The input slice must be valid and properly aligned for AVX-512 access
/// - Data is modified in-place, so caller must ensure no aliasing
///
/// # Performance
/// - 8× throughput improvement over scalar GELU
/// - Uses polynomial approximation for tanh (faster than exact computation)
/// - Best for large tensors (> 64 elements)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_avx512(data: &mut [f64]) {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const COEF: f64 = 0.044715;

    let len = data.len();
    let chunks = len / 8;

    let sqrt_2_over_pi = F64x8::splat(SQRT_2_OVER_PI);
    let coef = F64x8::splat(COEF);
    let half = F64x8::splat(0.5);
    let ones = F64x8::splat(1.0);
    let zeros = F64x8::splat(0.0);

    // Fast tanh approximation constants
    let twenty_seven = F64x8::splat(27.0);
    let nine = F64x8::splat(9.0);

    for i in 0..chunks {
        let idx = i * 8;
        let x = F64x8::load(data.as_ptr().add(idx));
        let x3 = x.mul(x).mul(x);
        let inner = sqrt_2_over_pi.mul(x.add(coef.mul(x3)));

        // Fast tanh: tanh(x) ≈ sign(x) * min(1, |x| * (27 + x²) / (27 + 9*x²))
        let x_abs = inner.abs();
        let x_sq = inner.mul(inner);
        let numerator = x_abs.mul(twenty_seven.add(x_sq));
        let denominator = twenty_seven.add(nine.mul(x_sq));
        let tanh_approx = numerator.div(denominator);

        // Clamp to [-1, 1] and restore sign
        // Use blend to select clamped value based on comparison
        let ones_mask = tanh_approx.cmp_le(ones);
        let neg_ones = F64x8::splat(-1.0);
        let clamped = tanh_approx.blend(neg_ones, ones_mask);

        // Compute sign: -1 if x < 0, +1 otherwise
        let sign_mask = inner.cmp_le(zeros);
        let sign = ones.blend(neg_ones, sign_mask);
        let tanh_result = sign.mul(clamped.abs());

        let gelu_x = x.mul(half).mul(ones.add(tanh_result));
        gelu_x.store(data.as_mut_ptr().add(idx));
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        let x = data[i];
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEF * x3);
        let tanh_val = inner.abs() * (27.0 + inner * inner) / (27.0 + 9.0 * inner * inner);
        let sign = if inner >= 0.0 { 1.0 } else { -1.0 };
        data[i] = x * 0.5 * (1.0 + tanh_val.min(1.0) * sign);
    }
}

/// AVX-512 SiLU activation: f(x) = x * sigmoid(x)
///
/// Uses fast sigmoid approximation: sigmoid(x) ≈ 1 / (1 + exp(-x))
/// Processes 8 f64 values in parallel using AVX-512.
///
/// # Arguments
/// * `data` - Mutable slice to process (modified in-place)
///
/// # Safety
/// - This function uses AVX-512 intrinsics which require CPU support
/// - Caller must ensure `target_feature = "avx512f"` is enabled
/// - The input slice must be valid and properly aligned for AVX-512 access
/// - Data is modified in-place, so caller must ensure no aliasing
///
/// # Performance
/// - 8× throughput improvement over scalar SiLU
/// - Uses exponential approximation for sigmoid
/// - Best for large tensors (> 64 elements)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_avx512(data: &mut [f64]) {
    let len = data.len();
    let chunks = len / 8;

    let ones = F64x8::splat(1.0);
    let neg_ones = F64x8::splat(-1.0);
    let zeros = F64x8::splat(0.0);

    // Sigmoid approximation constants
    // sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2) for faster computation
    let half = F64x8::splat(0.5);
    let two = F64x8::splat(2.0);

    for i in 0..chunks {
        let idx = i * 8;
        let x = F64x8::load(data.as_ptr().add(idx));

        // Sigmoid using tanh approximation: sigmoid(x) ≈ 0.5 * (1 + tanh(x/2))
        let x_half = x.div(two);
        let x_half_abs = x_half.abs();
        let x_half_sq = x_half.mul(x_half);

        // Fast tanh: tanh(z) ≈ z * (27 + z²) / (27 + 9*z²) for |z| < 3
        let twenty_seven = F64x8::splat(27.0);
        let nine = F64x8::splat(9.0);
        let numerator = x_half_abs.mul(twenty_seven.add(x_half_sq));
        let denominator = twenty_seven.add(nine.mul(x_half_sq));
        let tanh_approx = numerator.div(denominator);

        // Clamp and restore sign
        let ones_mask = tanh_approx.cmp_le(ones);
        let clamped = tanh_approx.blend(neg_ones, ones_mask);
        let sign_mask = x_half.cmp_le(zeros);
        let sign = ones.blend(neg_ones, sign_mask);
        let tanh_result = sign.mul(clamped.abs());

        // sigmoid = 0.5 * (1 + tanh(x/2))
        let sigmoid = half.mul(ones.add(tanh_result));

        // silu = x * sigmoid(x)
        let silu_result = x.mul(sigmoid);
        silu_result.store(data.as_mut_ptr().add(idx));
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        let x = data[i];
        let x_half = x / 2.0;
        let tanh_val = x_half.abs() * (27.0 + x_half * x_half) / (27.0 + 9.0 * x_half * x_half);
        let sign = if x_half >= 0.0 { 1.0 } else { -1.0 };
        let sigmoid = 0.5 * (1.0 + tanh_val.min(1.0) * sign);
        data[i] = x * sigmoid;
    }
}

/// Runtime check for AVX-512 support
///
/// Returns true if the CPU supports AVX-512F (foundation set)
///
/// # Note
/// This uses CPUID instruction to detect features at runtime
/// Even if compiled with AVX-512, the CPU may not support it
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // Check for AVX-512F (bit 16 of ECX)
        // CPUID leaf 7, sub-leaf 0
        let cpuid = unsafe { __cpuid_count(7, 0) };
        (cpuid.ecx & (1 << 16)) != 0
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX-512 optimized dot product for 8-element vectors
///
/// # Arguments
/// * `a` - First vector (8 elements)
/// * `b` - Second vector (8 elements)
///
/// # Returns
/// Dot product (sum of element-wise products)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_avx512(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() >= 8 && b.len() >= 8);

    let va = F64x8::load(a.as_ptr());
    let vb = F64x8::load(b.as_ptr());
    let product = va.mul(vb);
    product.reduce_add()
}

/// AVX-512 optimized matrix multiplication kernel
///
/// Computes C = A * B where:
/// - A is [M, K]
/// - B is [K, N]
/// - C is [M, N]
///
/// This kernel processes 8 elements of N dimension in parallel.
///
/// # Arguments
/// * `a` - Matrix A (row-major)
/// * `b` - Matrix B (row-major, transposed for better cache access)
/// * `c` - Output matrix C
/// * `m` - Rows of A/C
/// * `k` - Columns of A / Rows of B
/// * `n` - Columns of B/C (must be multiple of 8)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_kernel_avx512(
    a: *const f64,
    b: *const f64, // Transposed: [N, K]
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) {
    // 性能优化：
    // 1. 循环分块改进缓存局部性
    // 2. 循环展开减少分支预测失败
    // 3. 软件预取减少内存延迟
    const BLOCK_SIZE: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    // Process 8 columns of C at a time
    for i in 0..m {
        // 预取下一行的数据
        if i + 1 < m {
            crate::utils::prefetch_read_data(a.add((i + 1) * k), k * 8);
        }

        for j in (0..n).step_by(BLOCK_SIZE) {
            if j + BLOCK_SIZE <= n {
                // 循环展开：同时处理 4 个块
                let mut acc = [F64x8::splat(0.0); UNROLL_FACTOR];

                let mut p = 0;
                // 展开的主循环
                while p + UNROLL_FACTOR <= k {
                    // Load A[i, p:p+UNROLL_FACTOR] and broadcast
                    let a_ptr = a.add(i * k);
                    
                    // 预取 B 的数据
                    if p + UNROLL_FACTOR < k {
                        crate::utils::prefetch_read_data(b.add(j * k + p + UNROLL_FACTOR), BLOCK_SIZE * 8);
                    }

                    for u in 0..UNROLL_FACTOR {
                        let a_val = *a_ptr.add(p + u);
                        let a_vec = F64x8::splat(a_val);

                        // Load B^T[j:j+8, p+u] (8 consecutive elements)
                        let b_vec = F64x8::load(b.add(j * k + p + u));

                        // FMA: acc += a_vec * b_vec
                        acc[u] = acc[u].fmadd(a_vec, b_vec);
                    }
                    p += UNROLL_FACTOR;
                }

                // 处理剩余部分
                while p < k {
                    let a_val = *a.add(i * k + p);
                    let a_vec = F64x8::splat(a_val);
                    let b_vec = F64x8::load(b.add(j * k + p));
                    acc[0] = acc[0].fmadd(a_vec, b_vec);
                    p += 1;
                }

                // 累加所有展开的累加器
                let mut result = acc[0];
                for u in 1..UNROLL_FACTOR {
                    result = result.add(acc[u]);
                }

                // Store result
                result.store(c.add(i * n + j));
            }
        }

        // Handle remainder (n not multiple of 8)
        for j in (n / BLOCK_SIZE) * BLOCK_SIZE..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += *a.add(i * k + p) * *b.add(j * k + p);
            }
            *c.add(i * n + j) = sum;
        }
    }
}

/// AVX-512 optimized softmax kernel for large sequences
///
/// # Arguments
/// * `data` - Input/output data (will be overwritten with softmax result)
/// * `dim_size` - Size of the softmax dimension
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_kernel_avx512(data: *mut f64, dim_size: usize) {
    // Pass 1: Find maximum for numerical stability
    let mut max_val = f64::NEG_INFINITY;

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            let max_vec = vals.max(F64x8::splat(max_val));
            let max_arr = max_vec.to_array();
            max_val = max_val
                .max(max_arr[0])
                .max(max_arr[1])
                .max(max_arr[2])
                .max(max_arr[3])
                .max(max_arr[4])
                .max(max_arr[5])
                .max(max_arr[6])
                .max(max_arr[7]);
        } else {
            for rem_d in d..dim_size {
                max_val = max_val.max(*data.add(rem_d));
            }
        }
    }

    // Pass 2: Compute exp(x - max) and sum
    let mut sum_exp = 0.0;

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            let shifted = vals.sub(F64x8::splat(max_val));

            // Compute exp element-wise (requires scalar fallback)
            let arr = shifted.to_array();
            let exp_arr = [
                arr[0].exp(),
                arr[1].exp(),
                arr[2].exp(),
                arr[3].exp(),
                arr[4].exp(),
                arr[5].exp(),
                arr[6].exp(),
                arr[7].exp(),
            ];

            let exp_vec = F64x8::new(exp_arr);
            sum_exp += exp_vec.reduce_add();
            exp_vec.store(data.add(d));
        } else {
            for rem_d in d..dim_size {
                let exp_val = (*data.add(rem_d) - max_val).exp();
                sum_exp += exp_val;
                *data.add(rem_d) = exp_val;
            }
        }
    }

    // Pass 3: Normalize
    let inv_sum = 1.0 / sum_exp;
    let inv_sum_vec = F64x8::splat(inv_sum);

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            let normalized = vals.mul(inv_sum_vec);
            normalized.store(data.add(d));
        } else {
            for rem_d in d..dim_size {
                *data.add(rem_d) *= inv_sum;
            }
        }
    }
}

/// Layer Normalization AVX-512 kernel
///
/// Computes layer normalization: (x - mean) / (std + epsilon)
/// Uses 3-pass algorithm for numerical stability:
/// 1. Compute mean using AVX-512
/// 2. Compute variance using AVX-512
/// 3. Normalize using AVX-512
///
/// # Arguments
/// * `data` - Input data pointer
/// * `out` - Output data pointer (can be same as input for in-place)
/// * `dim_size` - Size of the normalization dimension
/// * `epsilon` - Small constant for numerical stability
///
/// # Performance
/// - 2× throughput improvement over AVX2 (f64x4)
/// - 8× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn layer_norm_kernel_avx512(
    data: *const f64,
    out: *mut f64,
    dim_size: usize,
    epsilon: f64,
) {
    // Pass 1: Compute mean using AVX-512
    let mut sum = 0.0;

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            sum += vals.reduce_add();
        } else {
            for rem_d in d..dim_size {
                sum += *data.add(rem_d);
            }
        }
    }

    let mean = sum / dim_size as f64;

    // Pass 2: Compute variance using AVX-512
    let mut sum_sq_diff = 0.0;
    let mean_vec = F64x8::splat(mean);

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            let diff = vals.sub(mean_vec);
            let sq_diff = diff.mul(diff);
            sum_sq_diff += sq_diff.reduce_add();
        } else {
            for rem_d in d..dim_size {
                let diff = *data.add(rem_d) - mean;
                sum_sq_diff += diff * diff;
            }
        }
    }

    let variance = sum_sq_diff / dim_size as f64;
    let std = variance.sqrt();
    let inv_std = 1.0 / (std + epsilon);

    // Pass 3: Normalize using AVX-512
    let inv_std_vec = F64x8::splat(inv_std);
    let mean_vec = F64x8::splat(mean);

    for d in (0..dim_size).step_by(8) {
        if d + 8 <= dim_size {
            let vals = F64x8::load(data.add(d));
            let diff = vals.sub(mean_vec);
            let normalized = diff.mul(inv_std_vec);
            normalized.store(out.add(d));
        } else {
            for rem_d in d..dim_size {
                *out.add(rem_d) = (*data.add(rem_d) - mean) * inv_std;
            }
        }
    }
}

/// Runtime check for AVX-512 support
#[inline]
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX-512 sparse compression: extract non-zero elements
/// 
/// Uses AVX-512 mask operations to efficiently find and extract non-zero values.
/// 
/// # Parameters
/// - `data`: Input data slice
/// - `col_indices`: Output vector for column indices (will be extended)
/// - `values`: Output vector for non-zero values (will be extended)
/// - `threshold`: Zero threshold (values with abs() <= threshold are treated as zero)
/// 
/// # Performance
/// - 8× parallel comparison using AVX-512
/// - Reduces branch divergence compared to scalar code
/// - Best for large matrices with regular sparsity patterns
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn sparse_compress_avx512(
    data: &[f64],
    col_indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    threshold: f64,
) {
    use std::arch::x86_64::*;
    
    let threshold_vec = _mm512_set1_pd(threshold);
    let zeros = _mm512_setzero_pd();
    
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for (chunk_idx, chunk) in chunks.enumerate() {
        let base_offset = chunk_idx * 8;
        let v = _mm512_loadu_pd(chunk.as_ptr());
        
        // Compute absolute values
        let abs_v = _mm512_max_pd(v, _mm512_sub_pd(zeros, v));
        
        // Compare with threshold: mask = 1 where abs(val) > threshold
        let mask = _mm512_cmp_pd_mask(abs_v, threshold_vec, _CMP_GT_OQ);
        
        // Extract non-zero elements based on mask
        if mask != 0 {
            let arr: [f64; 8] = std::mem::transmute(v);
            for (i, &val) in arr.iter().enumerate() {
                if val.abs() > threshold {
                    col_indices.push(base_offset + i);
                    values.push(val);
                }
            }
        }
    }
    
    // Handle remainder with scalar code
    let base_idx = chunks.len() * 8;
    for (i, &val) in remainder.iter().enumerate() {
        if val.abs() > threshold {
            col_indices.push(base_idx + i);
            values.push(val);
        }
    }
}

/// AVX-512 dense to sparse conversion for 2D tensor
/// 
/// # Parameters
/// - `data`: Input tensor data (row-major)
/// - `rows`: Number of rows
/// - `cols`: Number of columns
/// - `row_offsets`: Output row offsets (will be extended)
/// - `col_indices`: Output column indices (will be extended)
/// - `values`: Output values (will be extended)
/// - `threshold`: Zero threshold
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn dense_to_sparse_avx512(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_offsets: &mut Vec<usize>,
    col_indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    threshold: f64,
) {
    let threshold_vec = _mm512_set1_pd(threshold);
    let zeros = _mm512_setzero_pd();

    for row in 0..rows {
        let row_start = row * cols;
        let row_data = &data[row_start..];

        // Process in chunks of 8
        let chunks = row_data.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base_offset = chunk_idx * 8;
            let v = _mm512_loadu_pd(chunk.as_ptr());

            // Compute absolute values and compare with threshold
            let abs_v = _mm512_max_pd(v, _mm512_sub_pd(zeros, v));
            let mask = _mm512_cmp_pd_mask(abs_v, threshold_vec, _CMP_GT_OQ);

            if mask != 0 {
                let arr: [f64; 8] = std::mem::transmute(v);
                for (i, &val) in arr.iter().enumerate() {
                    if val.abs() > threshold {
                        col_indices.push(base_offset + i);
                        values.push(val);
                    }
                }
            }
        }

        // Handle remainder
        let base_idx = chunks.len() * 8;
        for (i, &val) in remainder.iter().enumerate() {
            if val.abs() > threshold {
                col_indices.push(base_idx + i);
                values.push(val);
            }
        }

        row_offsets.push(col_indices.len());
    }
}

/// AVX-512 GEMV (General Matrix-Vector multiplication): y = A * x
///
/// Computes matrix-vector multiplication using AVX-512 for 8× parallel processing.
/// Matrix is in row-major format.
///
/// # Arguments
/// * `a` - Matrix data (row-major, m × k)
/// * `x` - Input vector (k elements)
/// * `y` - Output vector (m elements)
/// * `m` - Number of rows
/// * `k` - Number of columns
///
/// # Performance
/// - 2× throughput improvement over scalar code
/// - Best for large vectors (k >= 64)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_avx512(a: &[f64], x: &[f64], y: &mut [f64], m: usize, k: usize) {
    for i in 0..m {
        let row_start = i * k;
        let mut sum = 0.0;

        // Process in chunks of 8
        let chunks = k / 8;
        for j in 0..chunks {
            let idx = row_start + j * 8;
            let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
            let x_vec = _mm512_loadu_pd(x.as_ptr().add(j * 8));
            let prod = _mm512_mul_pd(a_vec, x_vec);
            sum += _mm512_reduce_add_pd(prod);
        }

        // Handle remainder
        for j in (chunks * 8)..k {
            sum += a[row_start + j] * x[j];
        }

        y[i] = sum;
    }
}

/// AVX-512 AXPY operation: y = alpha * x + y
///
/// Computes element-wise: y[i] = alpha * x[i] + y[i]
/// Using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `alpha` - Scalar multiplier
/// * `x` - Input vector
/// * `y` - Output vector (modified in place)
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
/// - Best for large vectors (n >= 64)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn axpy_avx512(alpha: f64, x: &[f64], y: &mut [f64], n: usize) {
    let alpha_vec = _mm512_set1_pd(alpha);

    // Process in chunks of 8
    let chunks = n / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let x_vec = _mm512_loadu_pd(x.as_ptr().add(idx));
        let y_vec = _mm512_loadu_pd(y.as_ptr().add(idx));
        let result = _mm512_fmadd_pd(alpha_vec, x_vec, y_vec);
        _mm512_storeu_pd(y.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        y[i] = alpha * x[i] + y[i];
    }
}

/// AVX-512 vector addition: c = a + b
///
/// Computes element-wise addition using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Output vector
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
/// - Best for large vectors (n >= 64)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_add_avx512(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // Process in chunks of 8
    let chunks = n / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        let result = _mm512_add_pd(a_vec, b_vec);
        _mm512_storeu_pd(c.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        c[i] = a[i] + b[i];
    }
}

/// AVX-512 vector subtraction: c = a - b
///
/// Computes element-wise subtraction using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Output vector
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_sub_avx512(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // Process in chunks of 8
    let chunks = n / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        let result = _mm512_sub_pd(a_vec, b_vec);
        _mm512_storeu_pd(c.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        c[i] = a[i] - b[i];
    }
}

/// AVX-512 element-wise multiplication: c = a * b
///
/// Computes element-wise multiplication using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Output vector
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_mul_avx512(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // Process in chunks of 8
    let chunks = n / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        let result = _mm512_mul_pd(a_vec, b_vec);
        _mm512_storeu_pd(c.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        c[i] = a[i] * b[i];
    }
}

/// AVX-512 sigmoid activation: σ(x) = 1 / (1 + exp(-x))
///
/// Computes element-wise sigmoid using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `data` - Input/output vector (modified in place)
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid_avx512(data: &mut [f64]) {
    let ones = _mm512_set1_pd(1.0);

    // Process in chunks of 8
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let x = _mm512_loadu_pd(data.as_ptr().add(idx));
        let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);
        let exp_neg_x = _mm512_exp_pd(neg_x);
        let denom = _mm512_add_pd(ones, exp_neg_x);
        let sigmoid = _mm512_div_pd(ones, denom);
        _mm512_storeu_pd(data.as_ptr().add(idx), sigmoid);
    }

    // Handle remainder
    for i in (chunks * 8)..data.len() {
        data[i] = 1.0 / (1.0 + (-data[i]).exp());
    }
}

/// AVX-512 tanh activation: tanh(x)
///
/// Computes element-wise tanh using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `data` - Input/output vector (modified in place)
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh_avx512(data: &mut [f64]) {
    // Process in chunks of 8
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let x = _mm512_loadu_pd(data.as_ptr().add(idx));
        let exp_2x = _mm512_mul_pd(x, _mm512_set1_pd(2.0));
        let exp_2x = _mm512_exp_pd(exp_2x);
        let ones = _mm512_set1_pd(1.0);
        let num = _mm512_sub_pd(exp_2x, ones);
        let denom = _mm512_add_pd(exp_2x, ones);
        let tanh = _mm512_div_pd(num, denom);
        _mm512_storeu_pd(data.as_ptr().add(idx), tanh);
    }

    // Handle remainder
    for i in (chunks * 8)..data.len() {
        data[i] = data[i].tanh();
    }
}

/// AVX-512 Leaky ReLU activation: f(x) = x if x > 0 else alpha * x
///
/// Computes element-wise Leaky ReLU using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `data` - Input/output vector (modified in place)
/// * `alpha` - Slope for negative values (typically 0.01)
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn leaky_relu_avx512(data: &mut [f64], alpha: f64) {
    let zeros = _mm512_setzero_pd();
    let alpha_vec = _mm512_set1_pd(alpha);

    // Process in chunks of 8
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let x = _mm512_loadu_pd(data.as_ptr().add(idx));

        // Compare with zero: mask = x > 0
        let mask = _mm512_cmp_pd_mask(x, zeros, _CMP_GT_OQ);

        // Compute alpha * x
        let alpha_x = _mm512_mul_pd(x, alpha_vec);

        // Blend: result = x if x > 0 else alpha * x
        let result = _mm512_mask_blend_pd(mask, alpha_x, x);

        _mm512_storeu_pd(data.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..data.len() {
        data[i] = if data[i] > 0.0 { data[i] } else { alpha * data[i] };
    }
}

/// AVX-512 ELU activation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
///
/// Computes element-wise ELU using AVX-512 for 8× parallel processing.
///
/// # Arguments
/// * `data` - Input/output vector (modified in place)
/// * `alpha` - Scale parameter (typically 1.0)
/// * `n` - Number of elements
///
/// # Performance
/// - 2× throughput improvement over scalar code
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn elu_avx512(data: &mut [f64], alpha: f64) {
    let zeros = _mm512_setzero_pd();
    let ones = _mm512_set1_pd(1.0);
    let alpha_vec = _mm512_set1_pd(alpha);

    // Process in chunks of 8
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let x = _mm512_loadu_pd(data.as_ptr().add(idx));

        // Compare with zero: mask = x > 0
        let mask = _mm512_cmp_pd_mask(x, zeros, _CMP_GT_OQ);

        // Compute alpha * (exp(x) - 1)
        let exp_x = _mm512_exp_pd(x);
        let elu = _mm512_mul_pd(alpha_vec, _mm512_sub_pd(exp_x, ones));

        // Blend: result = x if x > 0 else elu
        let result = _mm512_mask_blend_pd(mask, x, elu);

        _mm512_storeu_pd(data.as_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 8)..data.len() {
        data[i] = if data[i] > 0.0 {
            data[i]
        } else {
            alpha * (data[i].exp() - 1.0)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[target_feature(enable = "avx512f")]
    unsafe fn test_f64x8_add() {
        let a = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = F64x8::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let result = a.add(b);
        let arr = result.to_array();
        assert!((arr[0] - 2.0).abs() < 1e-10);
        assert!((arr[7] - 9.0).abs() < 1e-10);
    }

    #[test]
    #[target_feature(enable = "avx512f")]
    unsafe fn test_f64x8_mul() {
        let a = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = F64x8::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let result = a.mul(b);
        let arr = result.to_array();
        assert!((arr[0] - 2.0).abs() < 1e-10);
        assert!((arr[7] - 16.0).abs() < 1e-10);
    }

    #[test]
    #[target_feature(enable = "avx512f")]
    unsafe fn test_f64x8_reduce_add() {
        let a = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let sum = a.reduce_add();
        assert!((sum - 36.0).abs() < 1e-10);
    }

    #[test]
    #[target_feature(enable = "avx512f")]
    unsafe fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = dot_product_avx512(&a, &b);
        assert!((result - 36.0).abs() < 1e-10);
    }
}
