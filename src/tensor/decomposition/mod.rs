//! Tensor Decomposition Tools
//!
//! This module provides various tensor decomposition algorithms as the foundation
//! for Lie group optimization and tensor ring compression.
//!
//! ## Algorithms
//!
//! - QR Decomposition (orthogonalization)
//! - SVD Decomposition (baseline comparison)
//! - Tensor Ring Decomposition (core algorithm)
//! - CP Decomposition (optional)
//! - Tucker Decomposition (optional)
//!
//! ## Example
//!
//! ```no_run
//! use god_gragh::tensor::decomposition::{qr_decompose, svd_decompose, tensor_ring_decompose};
//! use god_gragh::tensor::DenseTensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a test tensor
//! let tensor = DenseTensor::from_vec(
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
//!     vec![4, 2],
//! );
//!
//! // QR decomposition
//! let (q, r) = qr_decompose(&tensor)?;
//!
//! // SVD decomposition
//! let (u, s, v) = svd_decompose(&tensor, None)?;
//!
//! // Tensor ring decomposition
//! let cores = tensor_ring_decompose(&tensor, &[2, 2])?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "tensor")]
pub mod qr;

#[cfg(feature = "tensor")]
pub mod svd;

#[cfg(feature = "tensor")]
pub mod tensor_ring;

#[cfg(feature = "tensor")]
pub mod lie_algebra;

// Re-export decomposition functions
#[cfg(feature = "tensor")]
pub use qr::qr_decompose;

#[cfg(feature = "tensor")]
pub use svd::svd_decompose;

#[cfg(feature = "tensor")]
pub use tensor_ring::tensor_ring_decompose;

#[cfg(feature = "tensor")]
pub use lie_algebra::{lie_exponential, lie_logarithm, skew_symmetric_projection};
