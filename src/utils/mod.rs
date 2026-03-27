//! 工具模块

pub mod arena;
pub mod cache;

#[cfg(feature = "matrix")]
pub mod matrix;

pub use arena::Arena;
pub use cache::{Padded, prefetch_read, prefetch_write};

#[cfg(feature = "matrix")]
pub use matrix::{AdjacencyMatrix, LaplacianMatrix, fiedler_vector, spectral_radius};
