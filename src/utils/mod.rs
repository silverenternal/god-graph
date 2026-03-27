//! 工具模块

pub mod arena;
pub mod cache;

#[cfg(feature = "matrix")]
pub mod matrix;

pub use arena::Arena;
pub use cache::{prefetch_read, prefetch_write, Padded};

#[cfg(feature = "matrix")]
pub use matrix::{fiedler_vector, spectral_radius, AdjacencyMatrix, LaplacianMatrix};
